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


@pytest.mark.parametrize("tree_method,targets", [("hist", 1), ("hist", 3), ("approx", 1)])
@pytest.mark.parametrize("layout", ["dense", "missing", "empty", "external"])
@pytest.mark.parametrize("exponent", [-80, 0, 54, 100])
def test_all_roots_use_shared_integer_gradients(metal, tree_method, targets, layout, exponent, tmp_path):
    """Cancellation must survive before conversion, including roots with no bins."""
    big, small = np.float32(2.**exponent), np.float32(2.**(exponent - 54))
    grad = np.array([big, small, small, -big], dtype=np.float32)[:, None]
    grad = grad * np.array([2.**t for t in range(targets)], dtype=np.float32)[None, :]
    hess = np.tile(np.array([0., 1., 1., 0.], dtype=np.float32)[:, None], (1, targets))
    x = np.zeros((4, 2), dtype=np.float32)
    if layout == "missing":
        x[:, 0], x[1, 1] = np.nan, np.nan
    if layout == "empty":
        x[:] = np.nan
    label = np.zeros((4, targets), dtype=np.float32)

    class Pages(xgb.DataIter):
        def __init__(self):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / "quantized-roots"))
        def reset(self):
            self.page = 0
        def next(self, input_data):
            if self.page == 2:
                return False
            begin, end = (0, 3) if self.page == 0 else (3, 4)
            input_data(data=x[begin:end], label=label[begin:end])
            self.page += 1
            return True

    data = (xgb.DMatrix(Pages(), nthread=1) if layout == "external"
            else xgb.DMatrix(x, label=label, nthread=1))
    before = xgb.metal_info()
    model = xgb.train(dict(device="mps", tree_method=tree_method, nthread=1,
                           multi_strategy="multi_output_tree" if targets > 1 else "one_output_per_tree",
                           base_score=0., eta=1.,
                           reg_lambda=0., min_child_weight=0., max_depth=1, max_bin=2),
                      data, 1, obj=lambda p, d: (grad.copy(), hess.copy()))
    after = xgb.metal_info()
    assert after["quantized_gradient_sum_dispatches"] - before["quantized_gradient_sum_dispatches"] == 1
    assert after["quantization_dispatches"] - before["quantization_dispatches"] == targets
    expected = np.tile(-small * np.array([2.**t for t in range(targets)], dtype=np.float32), (4, 1))
    if targets == 1:
        expected = expected[:, 0]
    np.testing.assert_array_equal(model.predict(data), expected)
    np.testing.assert_array_equal(model.inplace_predict(x), expected)


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.parametrize("missing", [False, True])
def test_uniform_quantizes_before_rejection(metal, tree_method, missing):
    from cuda_uniform_reference import weights

    # Choose a host seed with rejected large curvature and selected tiny positive
    # curvature. The original scale MUST survive rejection of the large row.
    seed = next(s for s in range(1000) if np.array_equal(
        weights(int(np.random.RandomState(s).randint(0, 1 << 32, dtype=np.uint32)), 4) <= .5,
        [False, True, True, False]))
    grad = np.array([0., -1., -1., 0.], dtype=np.float32)
    hess = np.array([2.**65, 2.**-100, 2.**-100, 0.], dtype=np.float32)
    x = np.zeros((4, 2), dtype=np.float32)
    if missing:
        x[:, 0] = np.nan
    data = xgb.DMatrix(x, label=np.zeros(4), nthread=1)
    before = xgb.metal_info()
    model = xgb.train(dict(device="mps", tree_method=tree_method, nthread=1, seed=seed,
                           subsample=.5, base_score=0., eta=1., reg_lambda=1.,
                           max_depth=1, min_child_weight=0., max_bin=2), data, 1,
                      obj=lambda p, d: (grad.copy(), hess.copy()))
    after = xgb.metal_info()
    assert after["cuda_uniform_sampling_dispatches"] - before["cuda_uniform_sampling_dispatches"] == 1
    assert after["quantization_dispatches"] - before["quantization_dispatches"] == 1
    # frexp(2^65/(1-8*eps)) exponent=66, dequantization unit=2^(66-62)=16.
    # Two selected minimum-one integer Hessians => cover32, leaf2/(32+1).
    np.testing.assert_array_equal(model.predict(data), np.full(4, np.float32(2./33.)))
    root = json.loads(model.get_dump(dump_format="json", with_stats=True)[0])
    assert root["cover"] == 32.
