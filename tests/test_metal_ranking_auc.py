"""Grouped ranking curves use GPU sorting, counts/integrals and normalization."""
import numpy as np
import pytest
import xgboost as xgb


def score(model, data):
    return float(model.eval(data).rsplit(":", 1)[1])


@pytest.mark.parametrize("weights", ["none", "normal", "zero", "tiny", "large"])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("metric", ["auc", "aucpr"])
def test_ranking_auc(metal, weights, ties, metric):
    rng = np.random.default_rng(481)
    groups = np.array([0, 1, 2, 3, 31, 0, 257, 1025], dtype=np.uint32)
    n = int(groups.sum())
    labels = rng.integers(0, 2 if metric == "aucpr" else 7, size=n).astype(np.float32)
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
    p = dict(objective="reg:squarederror", eval_metric=metric, base_score=.5, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, data, 0)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    counter = "ranking_pr_auc_dispatches" if metric == "aucpr" else "ranking_roc_auc_dispatches"
    before = xgb.metal_info()[counter]
    expected = score(cpu, data)
    assert xgb.metal_info()[counter] == before
    np.testing.assert_allclose(score(gpu, data), expected, rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()[counter] == before + 1
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps", "eval_metric": metric})
    np.testing.assert_allclose(score(loaded, data), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("groups", [[0, 0], [1, 2], [0, 2, 0, 1]])
@pytest.mark.parametrize("metric", ["auc", "aucpr"])
def test_ranking_auc_no_valid_queries(metal, groups, metric):
    n = sum(groups)
    data = xgb.DMatrix(np.ones((n, 2)), label=np.zeros(n), group=groups, nthread=2)
    p = dict(eval_metric=metric, base_score=.5, nthread=2)
    counter = "ranking_pr_auc_dispatches" if metric == "aucpr" else "ranking_roc_auc_dispatches"
    before = xgb.metal_info()[counter]
    cpu = xgb.train(p | {"device": "cpu"}, data, 0)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    assert np.isnan(score(cpu, data)) and np.isnan(score(gpu, data))
    assert xgb.metal_info()[counter] == before + int(n != 0)


def test_ranking_roc_weight_overflow(metal):
    data = xgb.DMatrix(np.ones((3, 1)), label=[0, 1, 2], group=[3], weight=[1e30], nthread=2)
    for device in ["cpu", "mps"]:
        model = xgb.train(dict(device=device, eval_metric="auc", base_score=.5, nthread=2), data, 0)
        with pytest.raises(xgb.core.XGBoostError):
            score(model, data)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
def test_ranking_auc_training(metal, metric):
    rng = np.random.default_rng(873)
    x = rng.normal(size=(1032, 5)).astype(np.float32)
    data = xgb.DMatrix(x, label=rng.integers(0, 2 if metric == "aucpr" else 7, size=len(x)), group=[129] * 8, nthread=2)
    p = dict(objective="rank:pairwise", eval_metric=metric, nthread=2, max_depth=3, max_bin=33)
    histories = {}
    predictions = {}
    counter = "ranking_pr_auc_dispatches" if metric == "aucpr" else "ranking_roc_auc_dispatches"
    before = xgb.metal_info()[counter]
    for device in ["cpu", "mps"]:
        histories[device] = {}
        model = xgb.train(p | {"device": device}, data, 5, evals=[(data, "train")],
                          evals_result=histories[device], verbose_eval=False)
        predictions[device] = model.predict(data)
    assert xgb.metal_info()[counter] == before + 5
    np.testing.assert_allclose(histories["mps"]["train"][metric], histories["cpu"]["train"][metric],
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(predictions["mps"], predictions["cpu"], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("invalid", [-np.finfo(np.float32).smallest_subnormal, -1., 1.1])
def test_ranking_pr_label_validation(metal, invalid):
    data = xgb.DMatrix(np.ones((3, 1)), label=np.array([0, 1, invalid], dtype=np.float32), group=[3], nthread=2)
    for device in ["cpu", "mps"]:
        model = xgb.train(dict(device=device, eval_metric="aucpr", base_score=.5, nthread=2), data, 0)
        with pytest.raises(xgb.core.XGBoostError, match="binary relevance"):
            score(model, data)
