"""Objective-specific initialization must execute GPU work only when needed."""
import json

import numpy as np
import pytest
import xgboost as xgb

# Fixed quantile cuts isolate CPU tree/objective equivalence from the distinct
# CPU and CUDA sketch algorithms. Ordinary Metal cuts have source-oracle tests.


@pytest.mark.parametrize("objective", ["reg:expectileerror", "reg:absoluteerror"])
@pytest.mark.parametrize("targets", [1, 3])
@pytest.mark.parametrize("mode", ["unweighted", "weighted", "constant"])
def test_initialization(metal, objective, targets, mode):
    rng = np.random.default_rng(651)
    x = rng.normal(size=(4097, 7)).astype(np.float32)
    y = (x[:, 0] + .2 * x[:, 1] ** 2).astype(np.float32)
    if mode == "constant":
        y.fill(.375)
    if objective == "reg:absoluteerror" and targets > 1:
        y = np.column_stack([y, 2 * y + .25, -y - .5])
    w = None if mode == "unweighted" else rng.uniform(0, 2, len(x)).astype(np.float32)
    if w is not None:
        w[::17] = 0
    data = xgb.QuantileDMatrix(x, label=y, weight=w, nthread=2, max_bin=32)
    params = dict(objective=objective, nthread=2, max_depth=2, max_bin=32, eta=0)
    if objective == "reg:expectileerror":
        params["expectile_alpha"] = [.5] if targets == 1 else [.1, .5, .9]
    counter = "expectile_init_dispatches" if objective == "reg:expectileerror" else "prediction_init_dispatches"
    before = xgb.metal_info()
    cpu = xgb.train(params | {"device": "cpu"}, data, 1)
    assert xgb.metal_info()[counter] == before[counter]
    gpu = xgb.train(params | {"device": "mps"}, data, 1)
    after = xgb.metal_info()
    # Multi-target absolute error also broadcasts the final vector intercept
    # into its prediction cache. That broadcast now uses this same GPU kernel.
    cache_broadcast = int(objective == "reg:absoluteerror" and targets > 1)
    assert after[counter] == before[counter] + 1 + cache_broadcast
    assert after["initialization_gpu_time_ns"] > before["initialization_gpu_time_ns"]
    def base(model):
        return json.loads(json.loads(model.save_config())["learner"]["learner_model_param"]["base_score"])
    np.testing.assert_allclose(base(gpu), base(cpu), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(gpu.predict(data), cpu.predict(data), rtol=2e-5, atol=2e-5)
    # train() resets its caches before returning, so the first prediction also
    # broadcasts a vector intercept into a fresh cache.
    before_resume = xgb.metal_info()[counter]
    assert before_resume == after[counter] + cache_broadcast
    # Continued training and loaded models retain the estimated intercept.
    saved = gpu.save_raw()
    resumed = xgb.train(params | {"device": "mps", "eta": .3}, data, 2, xgb_model=saved)
    reference = xgb.train(params | {"device": "cpu", "eta": .3}, data, 2, xgb_model=cpu)
    assert xgb.metal_info()[counter] == before_resume + cache_broadcast
    np.testing.assert_allclose(resumed.predict(data), reference.predict(data), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("objective", ["reg:expectileerror", "reg:absoluteerror"])
def test_explicit_intercept_skips_initialization(metal, objective):
    d = xgb.DMatrix(np.arange(513, dtype=np.float32).reshape(-1, 1), label=np.ones(513))
    before = xgb.metal_info()
    params = dict(objective=objective, device="mps", nthread=2, base_score=.5)
    if objective == "reg:expectileerror":
        params["expectile_alpha"] = [.5]
    xgb.train(params, d, 1)
    after = xgb.metal_info()
    for name in ["expectile_init_dispatches", "prediction_init_dispatches"]:
        assert after[name] == before[name]


def test_zero_weight_absolute_skips_broadcast(metal):
    d = xgb.DMatrix(np.ones((257, 2)), label=np.ones(257), weight=np.zeros(257))
    before = xgb.metal_info()["prediction_init_dispatches"]
    xgb.train(dict(objective="reg:absoluteerror", device="mps", nthread=2), d, 1)
    assert xgb.metal_info()["prediction_init_dispatches"] == before
