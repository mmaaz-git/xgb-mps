"""Quantile initialization must use a single stable GPU sort for all levels."""
import json
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_weighted_quantile_cuda_threshold_precision(metal, multiple, reverse):
    # CUDA promotes float alpha/CDF to double; CPU WeightedQuantile narrows the
    # threshold to float and is intentionally not the oracle at this boundary.
    alpha = np.float32(.1)
    levels = ([np.nextafter(alpha, np.float32(0)), alpha,
               np.nextafter(alpha, np.float32(1))] if multiple else [alpha])
    expected = np.array([0, 1, 1] if multiple else [1], dtype=np.float32)
    assert float(alpha) * 10 > 1
    assert np.float32(float(alpha) * 10) == 1
    order = [1, 0] if reverse else [0, 1]
    matrix = xgb.DMatrix(np.array([[0.], [1.]], dtype=np.float32)[order],
                        label=np.array([0, 1], dtype=np.float32)[order],
                        weight=np.array([1, 9], dtype=np.float32)[order], nthread=1)
    before = xgb.metal_info()
    model = xgb.train(dict(objective="reg:quantileerror", quantile_alpha=[float(a) for a in levels],
                          device="mps", eta=0, max_depth=1, nthread=1), matrix, 1)
    after = xgb.metal_info()
    assert after["quantile_dispatches"] == before["quantile_dispatches"] + 1
    assert after["vector_arithmetic_dispatches"] == before["vector_arithmetic_dispatches"] + 1
    base = json.loads(json.loads(model.save_config())["learner"]["learner_model_param"]["base_score"])
    np.testing.assert_array_equal(base, expected)
    np.testing.assert_array_equal(model.predict(matrix).reshape(2, -1),
                                  np.broadcast_to(expected, (2, len(expected))))


@pytest.mark.parametrize("rows", [1, 257, 4097])
@pytest.mark.parametrize("mode", ["unweighted", "weighted", "zero_weights", "subnormal_weights", "ties"])
def test_quantile_initialization(metal, rows, mode):
    rng = np.random.default_rng(482)
    x = rng.normal(size=(rows, 5)).astype(np.float32)
    y = rng.normal(size=rows).astype(np.float32)
    weights = None if mode == "unweighted" else rng.uniform(.1, 3, rows).astype(np.float32)
    if mode == "zero_weights":
        weights[:] = 0
    elif mode == "subnormal_weights":
        weights *= 1e-40
    elif mode == "ties":
        y = rng.integers(-3, 4, rows).astype(np.float32)
    d = xgb.DMatrix(x, label=y, weight=weights, nthread=2)
    p = dict(objective="reg:quantileerror", quantile_alpha=[0.001, .1, .5, .9, .999],
             eta=0, max_depth=1, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, d, 1)
    before = xgb.metal_info()
    gpu = xgb.train(p | {"device": "mps"}, d, 1)
    after = xgb.metal_info()
    assert after["quantile_dispatches"] - before["quantile_dispatches"] == 1
    assert after["radix_sort_dispatches"] - before["radix_sort_dispatches"] == 1
    assert after["quantile_gpu_time_ns"] > before["quantile_gpu_time_ns"]
    assert after["vector_arithmetic_dispatches"] == before["vector_arithmetic_dispatches"] + 1
    cpu_base = json.loads(json.loads(cpu.save_config())["learner"]["learner_model_param"]["base_score"])
    gpu_base = json.loads(json.loads(gpu.save_config())["learner"]["learner_model_param"]["base_score"])
    np.testing.assert_array_equal(gpu_base, cpu_base)
    np.testing.assert_array_equal(gpu.predict(d), cpu.predict(d))
