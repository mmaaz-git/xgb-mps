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
    assert after["ranking_metric_dispatches"] - before["ranking_metric_dispatches"] == 15
    # NDCG also builds a temporary objective cache for intercept estimation.
    assert after["ranking_cache_dispatches"] - before["ranking_cache_dispatches"] == (3 if objective == "rank:ndcg" else 1)
    assert after["ranking_label_dispatches"] > before["ranking_label_dispatches"]
    assert after["ranking_gradient_dispatches"] - before["ranking_gradient_dispatches"] == 6
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


@pytest.mark.parametrize("metric", ["ndcg@10", "ndcg@10-", "pre@10", "map@10", "map@10-"])
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


@pytest.mark.parametrize("metric", ["pre", "map", "map-"])
@pytest.mark.parametrize("reverse", [False, True])
def test_ranking_soft_label_cancellation(metal, metric, reverse):
    # All three are accepted as near-binary labels upstream. Ordered double
    # addition loses the subnormal when it precedes +/- epsilon; reordering
    # those terms would change MAP's zero-hit fallback.
    labels = np.array([np.nextafter(np.float32(0), np.float32(1)), -1e-7, 1e-7], dtype=np.float32)
    if reverse:
        labels = labels[::-1].copy()
    d = xgb.DMatrix(np.ones((3, 1)), label=labels, group=[3], base_margin=[3, 2, 1], nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, nthread=2, base_score=.5)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["ranking_metric_dispatches"]
    if reverse and metric.startswith("map"):
        # The CPU formula exceeds one for this signed near-zero denominator.
        for model in [cpu, gpu]:
            with pytest.raises(xgb.core.XGBoostError, match="score"):
                model.eval(d)
    else:
        np.testing.assert_allclose(float(gpu.eval(d).rsplit(":", 1)[1]),
                                   float(cpu.eval(d).rsplit(":", 1)[1]), rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()["ranking_metric_dispatches"] == before + 1


@pytest.mark.parametrize("metric", ["ndcg", "map", "pre"])
@pytest.mark.parametrize("label", [-.1, .5, 32., -np.nextafter(np.float32(0), np.float32(1))])
def test_ranking_label_validation(metal, metric, label):
    labels = np.ones(257, dtype=np.float32)
    labels[-1] = label
    d = xgb.DMatrix(np.ones((257, 1)), label=labels, group=[31, 226], nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, nthread=2, base_score=.5)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["ranking_label_dispatches"]
    if abs(label) < 1e-6 and metric != "ndcg":
        np.testing.assert_allclose(float(gpu.eval(d).rsplit(":", 1)[1]),
                                   float(cpu.eval(d).rsplit(":", 1)[1]), rtol=2e-12, atol=2e-12)
    else:
        for model in [cpu, gpu]:
            with pytest.raises(xgb.core.XGBoostError, match="integer|Relevance|binary"):
                model.eval(d)
    assert xgb.metal_info()["ranking_label_dispatches"] == before + 1


def test_ndcg_linear_cache_and_reuse(metal):
    rng = np.random.default_rng(813)
    labels = rng.uniform(-.2, 3, 8193).astype(np.float32)
    d = xgb.DMatrix(np.ones((8193, 1)), label=labels, group=[4097, 4096],
                    base_margin=rng.normal(size=8193).astype(np.float32), nthread=2)
    params = dict(objective="reg:squarederror", eval_metric="ndcg", ndcg_exp_gain=False,
                  nthread=2, base_score=.5)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()
    for _ in range(3):
        np.testing.assert_allclose(float(gpu.eval(d).rsplit(":", 1)[1]),
                                   float(cpu.eval(d).rsplit(":", 1)[1]), rtol=2e-12, atol=2e-12)
    after = xgb.metal_info()
    assert after["ranking_cache_dispatches"] == before["ranking_cache_dispatches"] + 1
    assert after["ranking_label_dispatches"] == before["ranking_label_dispatches"]
    assert after["group_sort_dispatches"] == before["group_sort_dispatches"] + 4


@pytest.mark.parametrize("objective", ["rank:pairwise", "rank:ndcg", "rank:map"])
def test_unbiased_gradient_path_is_explicit(metal, objective):
    rng = np.random.default_rng(913)
    x = rng.normal(size=(96, 4)).astype(np.float32)
    labels = np.tile(np.r_[np.ones(8), np.zeros(24)], 3).astype(np.float32)
    d = xgb.DMatrix(x, label=labels, group=[32, 32, 32], nthread=2)
    params = dict(objective=objective, tree_method="hist", max_depth=3, nthread=2,
                  lambdarank_unbiased=True, lambdarank_pair_method="topk",
                  lambdarank_num_pair_per_sample=8, base_score=0)
    cpu = xgb.train(params | {"device": "cpu"}, d, 3)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, d, 3)
    after = xgb.metal_info()
    assert after["group_sort_dispatches"] > before["group_sort_dispatches"]
    # This mode is still an explicit CPU-gradient gap; sorting/cache are Metal.
    assert after["ranking_gradient_dispatches"] == before["ranking_gradient_dispatches"]
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


def test_mean_ranking_rounding_feedback(metal):
    # Native Metal exp/division errors were amplified after ~15 rounds here.
    # Keep the original benchmark's input, seed and prediction tolerance.
    rng = np.random.default_rng(17)
    n = 100_000
    x = rng.standard_normal(size=(n, 32), dtype=np.float32)
    y = (2 * x[:, 0] - x[:, 1] + x[:, 2] ** 2 + rng.normal(scale=.1, size=n)).astype(np.float32)
    y = np.digitize(y, [-2, -1, 0, 1, 2, 4, 6]).astype(np.float32)
    groups = np.diff(np.r_[np.arange(0, n, 128), n]).astype(np.uint32)
    d = xgb.DMatrix(x, label=y, group=groups, nthread=8)
    params = dict(tree_method="hist", objective="rank:pairwise", max_depth=6,
                  max_bin=256, nthread=8, seed=7, subsample=1.,
                  lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8)
    cpu = xgb.train(params | {"device": "cpu"}, d, 20)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, d, 20)
    assert xgb.metal_info()["ranking_gradient_dispatches"] - before["ranking_gradient_dispatches"] == 21
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)
