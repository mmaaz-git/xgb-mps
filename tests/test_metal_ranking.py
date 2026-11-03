"""Ranking sort integration: exact ordering must execute on the GPU."""
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("objective", ["rank:pairwise", "rank:ndcg", "rank:map"])
@pytest.mark.parametrize("method", ["topk", "mean"])
@pytest.mark.parametrize("largest", [63, 257, 4097])
def test_ranking_training(metal, objective, method, largest):
    rng = np.random.default_rng(786)
    sizes = np.array([1, 7, 32, 256, largest, 19], dtype=np.uint32)
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
    assert after["ranking_metric_dispatches"] - before["ranking_metric_dispatches"] == 10
    assert (after["radix_sort_dispatches"] > before["radix_sort_dispatches"]) == (largest > 4096)
    np.testing.assert_allclose(models["mps"].predict(d), models["cpu"].predict(d),
                               rtol=2e-5, atol=2e-5)
    for metric in params["eval_metric"]:
        np.testing.assert_allclose(histories["mps"]["train"][metric],
                                   histories["cpu"]["train"][metric], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("metric", ["ndcg", "ndcg@3-", "map", "map@3-", "pre@3"])
@pytest.mark.parametrize("largest", [63, 257, 4097])
def test_ranking_metric_ties(metal, metric, largest):
    sizes = np.array([5, largest, 3, 32], dtype=np.uint32)
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
    gpu_value = float(gpu.eval(d).rsplit(":", 1)[1])
    cpu_value = float(cpu.eval(d).rsplit(":", 1)[1])
    np.testing.assert_allclose(gpu_value, cpu_value, rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()["group_sort_dispatches"] > before


@pytest.mark.parametrize("metric", ["ndcg@10", "ndcg@10-", "pre@10"])
@pytest.mark.parametrize("mode", ["weighted", "zero_weights", "zero_labels", "linear_gain"])
def test_ranking_score_edge_cases(metal, metric, mode):
    rng = np.random.default_rng(7192)
    sizes = np.array([3, 37, 4096, 19], dtype=np.uint32)
    n = int(sizes.sum())
    labels = rng.integers(0, 2, n).astype(np.float32)
    if mode == "zero_labels":
        labels.fill(0)
    if mode == "linear_gain" and metric.startswith("ndcg"):
        labels = rng.uniform(-.1, 3, n).astype(np.float32)
    weights = rng.uniform(1, 3, len(sizes)).astype(np.float32)
    if mode == "zero_weights":
        weights.fill(0)
    d = xgb.DMatrix(np.ones((n, 1)), label=labels, group=sizes, weight=weights,
                    base_margin=rng.normal(size=n).astype(np.float32), nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, nthread=2, base_score=.5,
                  ndcg_exp_gain=mode != "linear_gain")
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["ranking_metric_dispatches"]
    gpu_value = float(gpu.eval(d).rsplit(":", 1)[1])
    cpu_value = float(cpu.eval(d).rsplit(":", 1)[1])
    np.testing.assert_allclose(gpu_value, cpu_value, rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()["ranking_metric_dispatches"] == before + 1
