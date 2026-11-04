"""Grouped ranking ROC uses GPU label sorting, pair counts and normalization."""
import numpy as np
import pytest
import xgboost as xgb


def score(model, data):
    return float(model.eval(data).rsplit(":", 1)[1])


@pytest.mark.parametrize("weights", ["none", "normal", "zero", "tiny", "large"])
@pytest.mark.parametrize("ties", [False, True])
def test_ranking_roc(metal, weights, ties):
    rng = np.random.default_rng(481)
    groups = np.array([0, 1, 2, 3, 31, 0, 257, 1025], dtype=np.uint32)
    n = int(groups.sum())
    labels = rng.integers(0, 7, size=n).astype(np.float32)
    margin = (rng.integers(-2, 3, size=n) if ties else rng.normal(size=n)).astype(np.float32)
    weight = None
    if weights != "none":
        weight = np.arange(1, len(groups) + 1, dtype=np.float32) / 7
        if weights == "zero":
            weight[:] = 0
        if weights == "tiny":
            weight = np.ldexp(np.ones_like(weight), -74)
        if weights == "large":
            weight *= np.float32(1e12)
    data = xgb.DMatrix(np.ones((n, 2), dtype=np.float32), label=labels,
                       base_margin=margin, group=groups, weight=weight, nthread=2)
    p = dict(objective="reg:squarederror", eval_metric="auc", base_score=.5, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, data, 0)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    before = xgb.metal_info()["ranking_roc_auc_dispatches"]
    expected = score(cpu, data)
    assert xgb.metal_info()["ranking_roc_auc_dispatches"] == before
    np.testing.assert_allclose(score(gpu, data), expected, rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()["ranking_roc_auc_dispatches"] == before + 1
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps", "eval_metric": "auc"})
    np.testing.assert_allclose(score(loaded, data), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("groups", [[0, 0], [1, 2], [0, 2, 0, 1]])
def test_ranking_roc_no_valid_queries(metal, groups):
    n = sum(groups)
    data = xgb.DMatrix(np.ones((n, 2)), label=np.zeros(n), group=groups, nthread=2)
    p = dict(eval_metric="auc", base_score=.5, nthread=2)
    before = xgb.metal_info()["ranking_roc_auc_dispatches"]
    cpu = xgb.train(p | {"device": "cpu"}, data, 0)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    assert np.isnan(score(cpu, data)) and np.isnan(score(gpu, data))
    assert xgb.metal_info()["ranking_roc_auc_dispatches"] == before + int(n != 0)


def test_ranking_roc_weight_overflow(metal):
    data = xgb.DMatrix(np.ones((3, 1)), label=[0, 1, 2], group=[3], weight=[1e30], nthread=2)
    for device in ["cpu", "mps"]:
        model = xgb.train(dict(device=device, eval_metric="auc", base_score=.5, nthread=2), data, 0)
        with pytest.raises(xgb.core.XGBoostError):
            score(model, data)


def test_ranking_roc_training(metal):
    rng = np.random.default_rng(873)
    x = rng.normal(size=(1032, 5)).astype(np.float32)
    data = xgb.DMatrix(x, label=rng.integers(0, 7, size=len(x)), group=[129] * 8, nthread=2)
    p = dict(objective="rank:pairwise", eval_metric="auc", nthread=2, max_depth=3, max_bin=33)
    histories = {}
    predictions = {}
    before = xgb.metal_info()["ranking_roc_auc_dispatches"]
    for device in ["cpu", "mps"]:
        histories[device] = {}
        model = xgb.train(p | {"device": device}, data, 5, evals=[(data, "train")],
                          evals_result=histories[device], verbose_eval=False)
        predictions[device] = model.predict(data)
    assert xgb.metal_info()["ranking_roc_auc_dispatches"] == before + 5
    np.testing.assert_allclose(histories["mps"]["train"]["auc"], histories["cpu"]["train"]["auc"],
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(predictions["mps"], predictions["cpu"], rtol=2e-5, atol=2e-5)
