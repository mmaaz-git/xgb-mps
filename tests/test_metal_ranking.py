"""Ranking sort integration: exact ordering must execute on the GPU."""
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("objective", ["rank:pairwise", "rank:ndcg", "rank:map"])
@pytest.mark.parametrize("method", ["topk", "mean"])
@pytest.mark.parametrize("large", [False, True])
def test_ranking_training(metal, objective, method, large):
    rng = np.random.default_rng(786)
    sizes = np.array([1, 7, 32, 256, 257 if large else 63, 19], dtype=np.uint32)
    n = int(sizes.sum())
    x = rng.normal(size=(n, 8)).astype(np.float32)
    y = (x[:, 0] + .3 * x[:, 1] > 0).astype(np.float32)
    d = xgb.DMatrix(x, label=y, group=sizes,
                    weight=np.array([1, .5, 2, 3, 1.5, .25], dtype=np.float32), nthread=2)
    params = dict(objective=objective, tree_method="hist", max_depth=3, nthread=2,
                  eval_metric=["ndcg@7", "map@7", "pre@7"], seed=17,
                  lambdarank_pair_method=method, lambdarank_num_pair_per_sample=8)
    histories = {}
    models = {}
    before = None
    for device in ["cpu", "mps"]:
        histories[device] = {}
        if device == "mps":
            before = xgb.metal_info()
        models[device] = xgb.train(params | {"device": device}, d, 5, evals=[(d, "train")],
                                    evals_result=histories[device], verbose_eval=False)
    after = xgb.metal_info()
    # Objective ranking, NDCG ideal order and all three metric orders use GPU.
    assert after["group_sort_dispatches"] - before["group_sort_dispatches"] >= 20
    assert after["float_sum_dispatches"] > before["float_sum_dispatches"]
    assert (after["radix_sort_dispatches"] > before["radix_sort_dispatches"]) == large
    np.testing.assert_allclose(models["mps"].predict(d), models["cpu"].predict(d),
                               rtol=2e-5, atol=2e-5)
    for metric in params["eval_metric"]:
        np.testing.assert_allclose(histories["mps"]["train"][metric],
                                   histories["cpu"]["train"][metric], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("metric", ["ndcg", "ndcg@3-", "map", "map@3-", "pre@3"])
@pytest.mark.parametrize("large", [False, True])
def test_ranking_metric_ties(metal, metric, large):
    sizes = np.array([5, 257 if large else 63, 3, 32], dtype=np.uint32)
    n = int(sizes.sum())
    rng = np.random.default_rng(342)
    labels = rng.integers(0, 2, n).astype(np.float32)
    scores = rng.integers(-2, 3, n).astype(np.float32)
    scores[:5] = np.array([-0., 0., -0., 0., 0.], dtype=np.float32)
    d = xgb.DMatrix(np.ones((n, 1)), label=labels, group=sizes, base_margin=scores, nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, nthread=2, base_score=.5)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["group_sort_dispatches"]
    assert gpu.eval(d) == cpu.eval(d)
    assert xgb.metal_info()["group_sort_dispatches"] > before
