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
    before = xgb.metal_info()
    actual = score(gpu, d)
    after = xgb.metal_info()
    assert after["metric_dispatches"] == before["metric_dispatches"] + 1
    assert after["metric_reduction_passes"] > before["metric_reduction_passes"]
    assert after["metric_final_readback_bytes"] == before["metric_final_readback_bytes"] + 16
    assert after["metric_host_input_bytes"] == before["metric_host_input_bytes"]
    input_bytes = prediction.nbytes + y.nbytes + (0 if weights is None else weights.nbytes)
    assert after["metric_shared_input_bytes"] - before["metric_shared_input_bytes"] == input_bytes
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
    before = xgb.metal_info()
    actual = score(gpu, d)
    after = xgb.metal_info()
    assert after["metric_dispatches"] == before["metric_dispatches"] + 1
    assert after["metric_reduction_passes"] > before["metric_reduction_passes"]
    assert after["metric_final_readback_bytes"] == before["metric_final_readback_bytes"] + 16
    assert after["metric_host_input_bytes"] == before["metric_host_input_bytes"]
    # DMatrix stores labels and weights as FP32, independently of NumPy input dtype.
    input_bytes = logits.nbytes + (label.size + weights.size) * 4
    assert after["metric_shared_input_bytes"] - before["metric_shared_input_bytes"] == input_bytes
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


@pytest.mark.parametrize("metric", ["mlogloss", "merror"])
def test_multiclass_metric_casts_labels_before_validation(metal, metric):
    # CUDA's metric casts to int first, unlike the objective's label predicate.
    labels = np.array([-.75, .75, 1.75, 2.75], dtype=np.float32)
    margins = np.array([[3, 2, 1], [1, 3, 2], [2, 1, 3], [1, 2, 3]], dtype=np.float32)
    d = xgb.DMatrix(np.ones((4, 1)), label=labels, base_margin=margins)
    params = dict(objective="multi:softprob", num_class=3, eval_metric=metric,
                  base_score=.5, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    expected = score(cpu, d)
    before = xgb.metal_info()["metric_label_dispatches"]
    actual = score(gpu, d)
    assert xgb.metal_info()["metric_label_dispatches"] == before + 1
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=3e-6)


@pytest.mark.parametrize("metric", ["mlogloss", "merror"])
@pytest.mark.parametrize("label", [-1., 3., 2147483648.])
def test_multiclass_metric_label_error_runs_on_gpu(metal, metric, label):
    d = xgb.DMatrix(np.ones((1, 1)), label=[label], base_margin=np.ones((1, 3)))
    model = xgb.train(dict(device="mps", objective="multi:softprob", num_class=3,
                           eval_metric=metric, base_score=.5, nthread=2), d, 0)
    before = xgb.metal_info()["metric_label_dispatches"]
    with pytest.raises(xgb.core.XGBoostError, match="MultiClassEvaluation"):
        score(model, d)
    assert xgb.metal_info()["metric_label_dispatches"] == before + 1
