"""Metric-only comparisons use fixed margins, independent of training differences."""
import numpy as np
import pytest
import xgboost as xgb

METRICS = ["rmse", "rmsle", "mae", "mape", "logloss", "error@0.371234567",
           "poisson-nloglik", "gamma-deviance", "gamma-nloglik",
           "tweedie-nloglik@1.3712345", "mphe"]


def score(model, matrix):
    text = model.eval(matrix)
    return float(text.rsplit(":", 1)[1])

@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("targets", [1, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_scalar_metrics(metal, metric, targets, weighted):
    rng = np.random.default_rng(51)
    x = rng.normal(size=(1031, 4)).astype(np.float32)
    y = rng.uniform(0.05, 0.95, size=(len(x), targets)).astype(np.float32)
    prediction = rng.uniform(0.05, 0.95, size=y.shape).astype(np.float32)
    weights = rng.uniform(0.1, 2, len(x)).astype(np.float32) if weighted else None
    if weighted:
        weights[::17] = 0
    d = xgb.DMatrix(x, label=y, weight=weights, base_margin=prediction)
    params = dict(objective="reg:squarederror", eval_metric=metric,
                  base_score=0.5, nthread=2, num_target=targets)
    if metric == "mphe":
        params["huber_slope"] = 1.3
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    expected = score(cpu, d)
    before = xgb.metal_info()["metric_dispatches"]
    actual = score(gpu, d)
    assert xgb.metal_info()["metric_dispatches"] == before + 1
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=3e-6)

@pytest.mark.parametrize("metric", ["mlogloss", "merror"])
@pytest.mark.parametrize("classes", [3, 37])
def test_multiclass_metrics(metal, metric, classes):
    rng = np.random.default_rng(89)
    x = rng.normal(size=(1031, 3)).astype(np.float32)
    label = rng.integers(0, classes, len(x))
    logits = rng.normal(size=(len(x), classes)).astype(np.float32)
    weights = rng.uniform(0, 2, len(x))
    d = xgb.DMatrix(x, label=label, base_margin=logits, weight=weights)
    p = dict(objective="multi:softprob", num_class=classes, eval_metric=metric,
             base_score=0.5, nthread=2)
    expected = score(xgb.train(p | {"device": "cpu"}, d, 0), d)
    gpu = xgb.train(p | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["metric_dispatches"]
    actual = score(gpu, d)
    assert xgb.metal_info()["metric_dispatches"] == before + 1
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=3e-6)

@pytest.mark.parametrize("metric", METRICS)
def test_zero_weights(metal, metric):
    x = np.ones((513, 2), dtype=np.float32)
    d = xgb.DMatrix(x, label=np.ones(513), weight=np.zeros(513))
    p = dict(objective="reg:squarederror", eval_metric=metric, base_score=0.5, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, d, 0)
    gpu = xgb.train(p | {"device": "mps"}, d, 0)
    np.testing.assert_equal(score(gpu, d), score(cpu, d))

@pytest.mark.parametrize("weight", [1e-30, 1e30, 1e37])
@pytest.mark.parametrize("metric", ["rmse", "logloss", "poisson-nloglik"])
def test_metric_reduction_extreme_weights(metal, weight, metric):
    # Summing 1e37 in a block of 256 overflows FP32, but the normalized metric is finite.
    x = np.ones((1031, 3), dtype=np.float32)
    d = xgb.DMatrix(x, label=np.full(1031, 0.7), weight=np.full(1031, weight))
    p = dict(eval_metric=metric, base_score=0.3, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, d, 0)
    gpu = xgb.train(p | {"device": "mps"}, d, 0)
    np.testing.assert_allclose(score(gpu, d), score(cpu, d), rtol=5e-6, atol=3e-6)

@pytest.mark.parametrize("label", [-2.5, -1, 0, 1, 100, 10000])
def test_poisson_log_gamma_boundaries(metal, label):
    d = xgb.DMatrix(np.ones((513, 2)), label=np.full(513, label))
    p = dict(eval_metric="poisson-nloglik", base_score=0.5, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, d, 0)
    gpu = xgb.train(p | {"device": "mps"}, d, 0)
    np.testing.assert_allclose(score(gpu, d), score(cpu, d), rtol=5e-6, atol=3e-6)
