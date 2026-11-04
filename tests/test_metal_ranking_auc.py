"""Grouped ranking curves use GPU sorting, counts/integrals and normalization."""
import numpy as np
import pytest
import xgboost as xgb

# Fixed quantile cuts isolate CPU tree/objective equivalence from the distinct
# CPU and CUDA sketch algorithms. Ordinary Metal cuts have source-oracle tests.


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
    before_bindings = xgb.metal_info()
    np.testing.assert_allclose(score(gpu, data), expected, rtol=2e-12, atol=2e-12)
    after = xgb.metal_info()
    assert after[counter] == before + 1
    assert after["metric_host_input_bytes"] == before_bindings["metric_host_input_bytes"]
    assert after["metric_shared_input_bytes"] - before_bindings["metric_shared_input_bytes"] == 4 * (2 * n + (0 if weight is None else len(groups)))
    assert after["auc_final_readback_bytes"] - before_bindings["auc_final_readback_bytes"] == 16
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


@pytest.mark.parametrize("weight", [1e20, 1e30, np.finfo(np.float32).max])
@pytest.mark.parametrize("margin", [[0, 1, 2], [2, 1, 0], [1, 1, 1]])
def test_ranking_roc_weight_overflow(metal, weight, margin):
    # Finite, accepted metadata. CUDA's FP32 square overflows; Inf/Inf or NaN/Inf
    # gives NaN. CPU alone has a per-query CHECK that throws before aggregation.
    data = xgb.DMatrix(np.ones((3, 1)), label=[0, 1, 2], base_margin=margin,
                       group=[3], weight=[weight], nthread=2)
    p = dict(eval_metric="auc", base_score=.5, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, data, 0)
    with pytest.raises(xgb.core.XGBoostError):
        score(cpu, data)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    before = xgb.metal_info()["ranking_roc_auc_dispatches"]
    with pytest.warns(UserWarning, match="only positive or negative"):
        assert np.isnan(score(gpu, data))
    assert xgb.metal_info()["ranking_roc_auc_dispatches"] == before + 1
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps", "eval_metric": "auc"})
    with pytest.warns(UserWarning, match="only positive or negative"):
        assert np.isnan(score(loaded, data))


def cuda_grouped_roc_reference(labels, margins, sizes, weights):
    """Host CUDA-source FP32 pair terms and FP64 sums, not NVIDIA execution."""
    total_area, valid, begin = 0.0, 0, 0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for size, weight in zip(sizes, weights):
            size = int(size)
            y = labels[begin:begin + size]
            predictions = margins[begin:begin + size]
            begin += size
            if size < 3:
                continue
            order = np.argsort(-y, kind="stable")
            squared = np.float32(np.float32(weight) * np.float32(weight))
            area, pair_weight = 0.0, 0.0
            for i in range(size):
                for j in range(i + 1, size):
                    difference = np.float32(predictions[order[i]] - predictions[order[j]])
                    value = np.float32(1 if difference > 0 else .5 if difference == 0 else 0)
                    area += float(np.float32(value * squared))
                    pair_weight += float(squared)
            total_area += area / pair_weight if pair_weight > 0 else 0.0
            valid += 1
    return total_area / valid if valid else float("nan")


@pytest.mark.parametrize("weight", [0., .7, 1e19, 2.**-100,
                                     float.fromhex("0x1.8p-75"), float.fromhex("0x1.4p-74")])
@pytest.mark.parametrize("ties", [False, True])
def test_cuda_ranking_roc_pair_terms(metal, weight, ties):
    groups = np.array([2, 3, 31, 257], dtype=np.uint32)
    n = int(groups.sum())
    labels = (np.arange(n) % 7).astype(np.float32) / 7
    margin = (np.ones(n) if ties else (np.arange(n) * 17) % 23).astype(np.float32)
    weights = np.full(len(groups), weight, dtype=np.float32)
    expected = cuda_grouped_roc_reference(labels, margin, groups, weights)
    data = xgb.DMatrix(np.ones((n, 1)), label=labels, base_margin=margin,
                       group=groups, weight=weights, nthread=2)
    p = dict(eval_metric="auc", base_score=.5, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, data, 0)
    np.testing.assert_allclose(score(cpu, data), expected, rtol=2e-12, atol=2e-12)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    before = xgb.metal_info()["ranking_roc_auc_dispatches"]
    np.testing.assert_allclose(score(gpu, data), expected, rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()["ranking_roc_auc_dispatches"] == before + 1
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps", "eval_metric": "auc"})
    np.testing.assert_allclose(score(loaded, data), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
def test_ranking_auc_training(metal, metric):
    rng = np.random.default_rng(873)
    x = rng.normal(size=(1032, 5)).astype(np.float32)
    data = xgb.QuantileDMatrix(x, label=rng.integers(0, 2 if metric == "aucpr" else 7, size=len(x)), group=[129] * 8, nthread=2, max_bin=33)
    p = dict(objective="rank:pairwise", eval_metric=metric, nthread=2, max_depth=3, max_bin=33)
    history = {}
    counter = "ranking_pr_auc_dispatches" if metric == "aucpr" else "ranking_roc_auc_dispatches"
    before = xgb.metal_info()[counter]
    model = xgb.train(p | {"device": "mps"}, data, 5, evals=[(data, "train")],
                      evals_result=history, verbose_eval=False)
    assert xgb.metal_info()[counter] == before + 5
    # Ranking gradients now use CUDA's truncation; CPU training has different
    # gradients. Compare each learned prefix on the CPU predictor/evaluator.
    for r in range(5):
        cpu = model[:r + 1]
        cpu.set_param({"device": "cpu", "eval_metric": metric})
        np.testing.assert_allclose(model.predict(data, iteration_range=(0, r + 1)),
                                   cpu.predict(data), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(history["train"][metric][r], score(cpu, data),
                                   rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("invalid", [-np.finfo(np.float32).smallest_subnormal, -1., 1.1])
def test_ranking_pr_label_validation(metal, invalid):
    data = xgb.DMatrix(np.ones((3, 1)), label=np.array([0, 1, invalid], dtype=np.float32), group=[3], nthread=2)
    for device in ["cpu", "mps"]:
        model = xgb.train(dict(device=device, eval_metric="aucpr", base_score=.5, nthread=2), data, 0)
        with pytest.raises(xgb.core.XGBoostError, match="binary relevance"):
            score(model, data)


def cuda_grouped_pr_reference(labels, margins, sizes, weights):
    """Host CUDA-source formula; parallel association/libdevice are not emulated."""
    total_area, valid, begin = 0.0, 0, 0
    for size, weight in zip(sizes, weights):
        size, weight = int(size), np.float32(weight)
        y = np.asarray(labels[begin:begin + size], dtype=np.float32)
        predictions = margins[begin:begin + size]
        begin += size
        if size < 2:
            continue
        total = sum(float(np.float32(v * weight)) for v in y)
        order = np.argsort(-predictions, kind="stable")
        fp, tp, fp_prev, tp_prev, area = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, row in enumerate(order):
            # CUDA rounds to FP32 after binary64 subtraction and multiplication.
            fp += float(np.float32((1.0 - float(y[row])) * float(weight)))
            tp += float(np.float32(y[row] * weight))
            if i + 1 < size and predictions[row] == predictions[order[i + 1]]:
                continue
            if total > 0:
                previous, recall = tp_prev / total, tp / total
                h = (fp - fp_prev) / (tp - tp_prev) if tp != tp_prev else 0.0
                a = h + 1.0
                b = (fp_prev - h * tp_prev) / total if tp != tp_prev else 0.0
                area += ((recall - previous - b / a *
                          (np.log(a * recall + b) - np.log(a * previous + b))) / a
                         if b != 0 else (recall - previous) / a)
            fp_prev, tp_prev = fp, tp
        if fp * tp > 0:
            total_area += area
            valid += 1
    return total_area / valid if valid else float("nan")


@pytest.mark.parametrize("label", [.125, .5, .875])
@pytest.mark.parametrize("weight", [1.0, .7])
def test_cuda_ranking_pr_single_row_excluded(metal, label, weight):
    data = xgb.DMatrix(np.ones((1, 1)), label=[label], group=[1], weight=[weight], nthread=2)
    p = dict(eval_metric="aucpr", base_score=.5, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, data, 0)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    before = xgb.metal_info()["ranking_pr_auc_dispatches"]
    with pytest.warns(UserWarning, match="only positive or negative"):
        assert np.isnan(score(gpu, data))
    assert np.isfinite(score(cpu, data))
    assert xgb.metal_info()["ranking_pr_auc_dispatches"] == before + 1


def test_cuda_ranking_pr_false_positive_rounding(metal):
    labels = np.array([.25, .25], dtype=np.float32)
    margins = np.zeros(2, dtype=np.float32)
    weights = np.array([.7], dtype=np.float32)
    data = xgb.DMatrix(np.ones((2, 1)), label=labels, base_margin=margins, group=[2],
                       weight=weights, nthread=2)
    p = dict(eval_metric="aucpr", base_score=.5, nthread=2)
    cpu = xgb.train(p | {"device": "cpu"}, data, 0)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    expected = cuda_grouped_pr_reference(labels, margins, [2], weights)
    before = xgb.metal_info()["ranking_pr_auc_dispatches"]
    assert score(gpu, data) == expected
    assert abs(score(cpu, data) - expected) > 1e-10
    assert xgb.metal_info()["ranking_pr_auc_dispatches"] == before + 1


@pytest.mark.parametrize("size", [257, 65537])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("weight_mode", ["normal", "tiny", "large"])
def test_cuda_ranking_pr_fractional_queries(metal, size, ties, weight_mode):
    rng = np.random.default_rng(943)
    groups = np.array([0, 1, 31, 0, size, 1, 2, 0], dtype=np.uint32)
    n = int(groups.sum())
    labels = rng.uniform(0, 1, n).astype(np.float32)
    labels[:3] = [.5, np.nextafter(np.float32(1), np.float32(0)), 1e-30]
    margins = (rng.integers(-4, 5, n) if ties else rng.normal(size=n)).astype(np.float32)
    weights = np.arange(1, len(groups) + 1, dtype=np.float32) / 7
    if weight_mode == "tiny":
        weights = np.arange(1, len(groups) + 1, dtype=np.float32) * np.finfo(np.float32).smallest_subnormal
    elif weight_mode == "large":
        weights *= np.float32(1e30)
    data = xgb.DMatrix(np.ones((n, 1)), label=labels, base_margin=margins, group=groups,
                       weight=weights, nthread=2)
    p = dict(eval_metric="aucpr", base_score=.5, nthread=2)
    gpu = xgb.train(p | {"device": "mps"}, data, 0)
    expected = cuda_grouped_pr_reference(labels, margins, groups, weights)
    before = xgb.metal_info()["ranking_pr_auc_dispatches"]
    np.testing.assert_allclose(score(gpu, data), expected, rtol=2e-12, atol=2e-12)
    assert xgb.metal_info()["ranking_pr_auc_dispatches"] == before + 1
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps", "eval_metric": "aucpr"})
    np.testing.assert_allclose(score(loaded, data), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_cuda_ranking_pr_zero_negative_mass(metal, weight):
    data = xgb.DMatrix(np.ones((3, 1)), label=[1, 1, 1], group=[3], weight=[weight], nthread=2)
    gpu = xgb.train(dict(device="mps", eval_metric="aucpr", base_score=.5, nthread=2), data, 0)
    with pytest.warns(UserWarning, match="only positive or negative"):
        assert np.isnan(score(gpu, data))
