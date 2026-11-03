"""Base-score means must execute on GPU and agree with CPU initialization."""
import json
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("targets", [1, 3])
@pytest.mark.parametrize("mode", ["normal", "tiny_weights", "large_weights", "constant"])
def test_base_score_means(metal, weighted, targets, mode):
    rng = np.random.default_rng(620)
    x = rng.normal(size=(4097, 5)).astype(np.float32)
    y = rng.normal(size=(len(x), targets)).astype(np.float32)
    if mode == "constant":
        y.fill(127 * 2.0**-30)
    weights = rng.uniform(.1, 3, len(x)).astype(np.float32) if weighted else None
    if weighted and mode == "tiny_weights":
        weights *= 1e-30
    if weighted and mode == "large_weights":
        weights *= 1e30
    d = xgb.DMatrix(x, label=y, weight=weights, nthread=2)
    p = dict(objective="reg:squarederror", nthread=2, eta=0, max_depth=1,
             multi_strategy="multi_output_tree")
    cpu = xgb.train(p | {"device": "cpu"}, d, 1)
    before = xgb.metal_info()
    gpu = xgb.train(p | {"device": "mps"}, d, 1)
    after = xgb.metal_info()
    assert after["mean_dispatches"] - before["mean_dispatches"] == 1
    assert after["mean_gpu_time_ns"] > before["mean_gpu_time_ns"]
    cpu_base = json.loads(json.loads(cpu.save_config())["learner"]["learner_model_param"]["base_score"])
    gpu_base = json.loads(json.loads(gpu.save_config())["learner"]["learner_model_param"]["base_score"])
    np.testing.assert_allclose(gpu_base, cpu_base, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_zero_weight_mean_rejected(metal, device):
    d = xgb.DMatrix(np.ones((257, 2)), label=np.ones(257), weight=np.zeros(257))
    with pytest.raises(xgb.core.XGBoostError, match="at least one non-zero value"):
        xgb.train(dict(device=device, nthread=2), d, 1)


@pytest.mark.parametrize("targets", [1, 3])
@pytest.mark.parametrize("weight_mode", ["normal", "tiny", "zero"])
def test_absolute_initial_weight_sum(metal, targets, weight_mode):
    rng = np.random.default_rng(140)
    x = rng.normal(size=(1025, 5)).astype(np.float32)
    y = rng.normal(size=(len(x), targets)).astype(np.float32)
    weights = rng.uniform(.1, 3, len(x)).astype(np.float32)
    if weight_mode == "tiny":
        weights *= 1e-40
    elif weight_mode == "zero":
        weights.fill(0)
    d = xgb.DMatrix(x, label=y, weight=weights, nthread=2)
    p = dict(objective="reg:absoluteerror", nthread=2, eta=0, max_depth=1)
    cpu = xgb.train(p | {"device": "cpu"}, d, 1)
    before = xgb.metal_info()
    gpu = xgb.train(p | {"device": "mps"}, d, 1)
    assert xgb.metal_info()["float_sum_dispatches"] - before["float_sum_dispatches"] == 1
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-6, atol=2e-7)
