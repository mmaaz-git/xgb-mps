"""Ranking sort integration: exact ordering must execute on the GPU."""
import json

import numpy as np
import pytest
import xgboost as xgb


def cuda_ranking_reference(metric, labels, scores, sizes, weights, exponential=True):
    """Host evaluation of rank_metric.cu formulas, not a CUDA execution oracle.

    Used for well-conditioned prefixes; CUB's scan/reduction association may
    differ. Catastrophic signed cancellation has a separate Metal-order test.
    """
    name, _, suffix = metric.partition("@")
    minus = metric.endswith("-")
    name = name.rstrip("-")
    # This vendored version defaults unqualified metrics to LambdaRank DefaultK().
    topk = int(suffix.rstrip("-")) if suffix else 32
    total = 0.0
    begin = 0
    for size, weight in zip(sizes, weights):
        size, weight = int(size), float(weight)
        y = np.asarray(labels[begin:begin + size], dtype=np.float64)
        order = np.argsort(-scores[begin:begin + size], kind="stable")
        ranked = y[order]
        n = min(size, topk)
        if name == "pre":
            score = sum(float(v) / n * weight for v in ranked[:n])
        elif name == "map":
            prefix = np.cumsum(ranked, dtype=np.float64)
            hits = float(prefix[-1]) if size else 0.0
            area = sum(float(prefix[i]) / (i + 1) * float(ranked[i]) for i in range(n))
            score = (area * weight) / min(hits, topk) if hits > 0 else float(not minus)
        else:
            def dcg(values):
                return sum(((2**int(v) - 1) if exponential else float(v)) /
                           np.log2(i + 2) for i, v in enumerate(values[:n]))
            ideal = dcg(np.sort(y)[::-1])
            inverse = 1.0 / ideal if ideal else 0.0
            score = float(not minus) if inverse <= 0 else (dcg(ranked) * inverse) * weight
        total += score
        begin += size
    sw = sum(float(w) for w in weights)
    return total / sw if sw > 0 else total


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
        expected = np.asarray(histories["cpu"]["train"][metric])
        if metric.startswith("map"):
            # CUDA's zero-relevance fallback is unweighted. Its difference from
            # CPU depends only on labels/weights, so applies to every round.
            offset, delta = 0, 0.0
            for size, weight in zip(sizes, d.get_weight()):
                if not np.any(y[offset:offset + size]):
                    delta += 1.0 - float(weight)
                offset += size
            expected = expected + delta / sum(float(w) for w in d.get_weight())
        np.testing.assert_allclose(histories["mps"]["train"][metric], expected,
                                   rtol=2e-12, atol=2e-12)


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
    scores = rng.normal(size=n).astype(np.float32)
    d = xgb.DMatrix(np.ones((n, 1)), label=labels, group=sizes, weight=weights,
                    base_margin=scores, nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, nthread=2, base_score=.5,
                  ndcg_exp_gain=mode != "linear_gain")
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["ranking_metric_dispatches"]
    gpu_value = float(gpu.eval(d).rsplit(":", 1)[1])
    cpu_value = float(cpu.eval(d).rsplit(":", 1)[1])
    expected = cuda_ranking_reference(metric, labels, scores, sizes, weights,
                                      exponential=mode != "linear_gain")
    np.testing.assert_allclose(gpu_value, expected, rtol=2e-12, atol=2e-12)
    if not (metric.startswith("map") and not metric.endswith("-") and
            any(not np.any(g) for g in np.split(labels, np.cumsum(sizes)[:-1]))):
        np.testing.assert_allclose(gpu_value, cpu_value, rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()["ranking_metric_dispatches"] == before + 1


@pytest.mark.parametrize("metric", ["pre", "map", "map-"])
@pytest.mark.parametrize("reverse", [False, True])
def test_ranking_soft_label_cancellation(metal, metric, reverse):
    # CUDA uses a parallel scan, whose association need not match CPU. These
    # accepted labels are ill-conditioned: (a+b)+c and a+(b+c) disagree on the
    # sign of the total. Pin the Metal scan's three-term result without claiming
    # that a particular NVIDIA/CUB architecture makes the same branch decision.
    labels = np.array([np.nextafter(np.float32(0), np.float32(1)), -1e-7, 1e-7], dtype=np.float32)
    if reverse:
        labels = labels[::-1].copy()
    d = xgb.DMatrix(np.ones((3, 1)), label=labels, group=[3], base_margin=[3, 2, 1], nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, nthread=2, base_score=.5)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["ranking_metric_dispatches"]
    if metric.startswith("map"):
        invalid, fallback = (cpu, gpu) if reverse else (gpu, cpu)
        with pytest.raises(xgb.core.XGBoostError, match="score"):
            invalid.eval(d)
        assert float(fallback.eval(d).rsplit(":", 1)[1]) == float(not metric.endswith("-"))
    else:
        np.testing.assert_allclose(float(gpu.eval(d).rsplit(":", 1)[1]),
                                   float(cpu.eval(d).rsplit(":", 1)[1]), rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()["ranking_metric_dispatches"] == before + 1


@pytest.mark.parametrize("minus", [False, True])
@pytest.mark.parametrize("weight", [0.0, 0.5, 2.0])
def test_cuda_map_unweighted_fallback(metal, minus, weight):
    metric = "map@3" + ("-" if minus else "")
    d = xgb.DMatrix(np.ones((4, 1)), label=np.zeros(4), group=[3, 1],
                    weight=[weight, 4], nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, base_score=.5, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["ranking_metric_dispatches"]
    actual = float(gpu.eval(d).rsplit(":", 1)[1])
    expected = 0.0 if minus else 2.0 / (weight + 4)
    assert actual == expected
    assert float(cpu.eval(d).rsplit(":", 1)[1]) == float(not minus)
    assert xgb.metal_info()["ranking_metric_dispatches"] == before + 1


@pytest.mark.parametrize("metric", ["pre@3", "map", "map-"])
def test_cuda_ranking_empty_queries(metal, metric):
    d = xgb.DMatrix(np.ones((4, 1)), label=np.zeros(4), group=[0, 3, 0, 1],
                    weight=[3, 0, 2, 4], nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, base_score=.5, nthread=2)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["ranking_metric_dispatches"]
    actual = float(gpu.eval(d).rsplit(":", 1)[1])
    assert actual == (4.0 / 9 if metric == "map" else 0.0)
    assert xgb.metal_info()["ranking_metric_dispatches"] == before + 1


def test_cuda_precision_fp64_terms(metal):
    labels = np.array([.99999994, 0, 0, .99999994], dtype=np.float32)
    scores = np.array([4, 3, 2, 1], dtype=np.float32)
    weights = np.array([.7, 1.1], dtype=np.float32)
    d = xgb.DMatrix(np.ones((4, 1)), label=labels, group=[3, 1], weight=weights,
                    base_margin=scores, nthread=2)
    params = dict(objective="reg:squarederror", eval_metric="pre@3", base_score=.5, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    expected = cuda_ranking_reference("pre@3", labels, scores, [3, 1], weights)
    actual = float(gpu.eval(d).rsplit(":", 1)[1])
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
    assert abs(float(cpu.eval(d).rsplit(":", 1)[1]) - expected) > 1e-9


@pytest.mark.parametrize("metric", ["map", "map@10-", "map@4294967294"])
@pytest.mark.parametrize("largest", [257, 65537])
def test_cuda_map_segmented_soft_labels(metal, metric, largest):
    rng = np.random.default_rng(741)
    sizes = np.array([0, 31, 0, largest, 3, 0], dtype=np.uint32)
    n = int(sizes.sum())
    # Actual near-binary labels must be scanned, not converted to integer hits.
    labels = rng.choice(np.array([1, .99999994, 1e-7, -1e-7], dtype=np.float32), n)
    labels[-3:] = [.99999994, 1e-7, -1e-7]
    scores = rng.integers(-7, 8, n).astype(np.float32)
    weights = np.array([2, 3, 4, 5, 3, 2], dtype=np.float32)
    d = xgb.DMatrix(np.ones((n, 1)), label=labels, group=sizes, weight=weights,
                    base_margin=scores, nthread=2)
    params = dict(objective="reg:squarederror", eval_metric=metric, base_score=.5, nthread=2)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()["ranking_metric_dispatches"]
    expected = cuda_ranking_reference(metric, labels, scores, sizes, weights)
    np.testing.assert_allclose(float(gpu.eval(d).rsplit(":", 1)[1]), expected,
                               rtol=2e-12, atol=2e-12)
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
@pytest.mark.parametrize("method", ["topk", "mean"])
@pytest.mark.parametrize("query_size", [32, 4097])
def test_unbiased_gradient_and_bias_updates(metal, objective, method, query_size):
    rng = np.random.default_rng(913)
    x = rng.normal(size=(3 * query_size, 4)).astype(np.float32)
    labels = np.tile(np.r_[np.ones(4), np.zeros(query_size - 4)], 3).astype(np.float32)
    d = xgb.DMatrix(x, label=labels, group=[query_size] * 3, nthread=2)
    params = dict(objective=objective, tree_method="hist", max_depth=3, nthread=2,
                  lambdarank_unbiased=True, lambdarank_pair_method=method,
                  lambdarank_num_pair_per_sample=8, base_score=0)
    cpu = xgb.train(params | {"device": "cpu"}, d, 3)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, d, 3)
    after = xgb.metal_info()
    assert after["group_sort_dispatches"] > before["group_sort_dispatches"]
    assert after["ranking_gradient_dispatches"] - before["ranking_gradient_dispatches"] == 3
    assert after["ranking_bias_dispatches"] - before["ranking_bias_dispatches"] == 3
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)
    cpu_state = json.loads(cpu.save_config())["learner"]["objective"]
    gpu_state = json.loads(gpu.save_config())["learner"]["objective"]
    for key in ["ti+", "tj-"]:
        np.testing.assert_allclose(gpu_state[key], cpu_state[key], rtol=2e-5, atol=2e-6)
    assert not np.allclose(gpu_state["ti+"], 1)



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
