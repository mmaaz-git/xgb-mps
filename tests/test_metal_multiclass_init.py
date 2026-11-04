"""CUDA-derived packed class priors and GPU base-score arithmetic."""
import json
import numpy as np
import pytest
import xgboost as xgb


def intercept(model):
    return np.asarray(json.loads(json.loads(model.save_config())["learner"]["learner_model_param"]["base_score"]), dtype=np.float32)


@pytest.mark.parametrize("classes", [2, 3, 17])
@pytest.mark.parametrize("layout", ["complete", "missing_first", "gaps"])
@pytest.mark.parametrize("weighted", [False, True])
def test_multiclass_cuda_initialization(metal, classes, layout, weighted):
    rng = np.random.default_rng(842)
    labels = np.tile(np.arange(classes, dtype=np.float32), 129)
    if layout == "missing_first":
        labels[labels == 0] = classes - 1
    elif layout == "gaps":
        labels = (labels // 2) * 2
    rng.shuffle(labels)
    weights = rng.integers(0, 8, len(labels)).astype(np.float32) if weighted else None
    data = xgb.DMatrix(rng.normal(size=(len(labels), 3)), label=labels, weight=weights, nthread=2)
    params = dict(objective="multi:softprob", num_class=classes, eta=0, max_depth=1, nthread=2)
    before = xgb.metal_info()
    model = xgb.train(params | {"device": "mps"}, data, 1)
    after = xgb.metal_info()
    assert after["small_histogram_dispatches"] == before["small_histogram_dispatches"] + 1
    assert after["vector_arithmetic_dispatches"] == before["vector_arithmetic_dispatches"] + 3
    assert after["float_mean_dispatches"] == before["float_mean_dispatches"] + 1
    # CUDA RLE sums are packed into leading bins; unique labels are not scattered.
    sums = np.zeros(classes, dtype=np.float32)
    for run, label in enumerate(np.unique(labels)):
        sums[run] = np.count_nonzero(labels == label) if weights is None else weights[labels == label].sum(dtype=np.float64)
    denom = float(len(labels)) if weights is None else weights.sum(dtype=np.float64)
    logs = np.log((sums.astype(np.float64) * (1. / denom)).astype(np.float32) + np.float32(1e-6)).astype(np.float32)
    expected = logs - (logs / np.float32(classes)).sum(dtype=np.float32)
    np.testing.assert_allclose(intercept(model), expected, rtol=2e-6, atol=2e-6)
    if layout == "complete":
        cpu = xgb.train(params | {"device": "cpu"}, data, 1)
        np.testing.assert_allclose(intercept(model), intercept(cpu), rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(model.predict(data), cpu.predict(data), rtol=2e-6, atol=2e-6)
    reloaded = xgb.Booster(model_file=model.save_raw())
    reloaded.set_param({"device": "mps", "nthread": 2})
    np.testing.assert_array_equal(model.predict(data), reloaded.predict(data))


@pytest.mark.parametrize("label", [-.5, .5, 2.5, 3.])
def test_fractional_objective_labels_rejected_before_histogram(metal, label):
    data = xgb.DMatrix(np.ones((3, 1)), label=[0., 1., label])
    before = xgb.metal_info()["small_histogram_dispatches"]
    with pytest.raises(xgb.core.XGBoostError, match="discrete"):
        xgb.train(dict(device="mps", objective="multi:softprob", num_class=3, nthread=2), data, 1)
    assert xgb.metal_info()["small_histogram_dispatches"] == before


@pytest.mark.parametrize("weight", [0., 1e-40])
def test_zero_or_tiny_initial_weight_error(metal, weight):
    data = xgb.DMatrix(np.ones((3, 1)), label=[0, 1, 2], weight=[weight] * 3)
    before = xgb.metal_info()["small_histogram_dispatches"]
    with pytest.raises(xgb.core.XGBoostError):
        xgb.train(dict(device="mps", objective="multi:softprob", num_class=3, nthread=2), data, 1)
    assert xgb.metal_info()["small_histogram_dispatches"] == before + 1
