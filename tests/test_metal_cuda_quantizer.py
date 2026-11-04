"""Training-level checks for intentional CUDA quantizer behavior.

These scalar formulas come from quantiser.cu/quantiser.cuh; no NVIDIA execution
is implied. In particular, CPU training does not quantize positive curvature.
"""
import json
import math

import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.parametrize("small_hessian", [np.float32(2.**-149), np.float32(2.**-50)])
def test_cuda_positive_curvature_minimum_in_training(metal, tree_method, small_hessian):
    x = np.array([[0.], [0.], [1.], [1.]], dtype=np.float32)
    grad = np.array([-1., -1., 1., 1.], dtype=np.float32)
    hess = np.array([small_hessian, small_hessian, 2.**65, 2.**65], dtype=np.float32)
    data = xgb.DMatrix(x, label=np.zeros(len(x)), nthread=1)
    params = dict(tree_method=tree_method, max_bin=2, max_depth=1, base_score=0.,
                  eta=1., min_child_weight=0., reg_lambda=1., nthread=1)

    def objective(pred, dtrain):
        return grad.copy(), hess.copy()

    # CUDA: the positive bound, rounding factor and minimum-one Hessian.
    bound = math.fsum(float(h) for h in hess)
    delta = bound / (1. - 2. * len(hess) * np.finfo(np.float64).eps)
    _, exponent = math.frexp(delta)
    unit = math.ldexp(1., exponent - 62)
    left_hessian = 2. * unit
    assert float(small_hessian) / unit < 1.
    expected = np.array([2. / (left_hessian + 1.)] * 2 + [-2. / (2. * float(hess[2]) + 1.)] * 2)

    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data, 1, obj=objective)
    after = xgb.metal_info()
    assert after["gradient_bound_dispatches"] > before["gradient_bound_dispatches"]
    assert after["quantization_dispatches"] > before["quantization_dispatches"]
    np.testing.assert_allclose(gpu.predict(data), expected, rtol=1e-6, atol=0.)
    tree = json.loads(gpu.get_dump(dump_format="json", with_stats=True)[0])
    assert len(tree["children"]) == 2
    assert min(child["cover"] for child in tree["children"]) == left_hessian

    # CPU follows the original float Hessians. The difference is intentional;
    # matching this CPU prediction would violate the CUDA minimum-one rule.
    cpu = xgb.train(params | {"device": "cpu"}, data, 1, obj=objective)
    np.testing.assert_allclose(cpu.predict(data)[:2], 2. / (2. * float(small_hessian) + 1.),
                               rtol=1e-6, atol=0.)
    assert np.max(np.abs(cpu.predict(data)[:2] - gpu.predict(data)[:2])) > 1.
