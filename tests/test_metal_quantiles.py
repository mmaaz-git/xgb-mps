"""Quantile initialization must use a single stable GPU sort for all levels."""
import json
import numpy as np
import pytest
import xgboost as xgb


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
