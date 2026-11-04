"""Classification AUC CUDA formulas, GPU dispatch and compatible CPU checks."""
import numpy as np
import pytest
import xgboost as xgb

# Fixed quantile cuts isolate CPU tree/objective equivalence from the distinct
# CPU and CUDA sketch algorithms. Ordinary Metal cuts have source-oracle tests.


def score(model, d):
    return float(model.eval(d).rsplit(":", 1)[1])


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
@pytest.mark.parametrize("problem", ["binary", "multiclass", "multilabel"])
@pytest.mark.parametrize("weight_mode", ["none", "normal", "tiny", "large"])
@pytest.mark.parametrize("ties", [False, True])
def test_roc_auc(metal, metric, problem, weight_mode, ties):
    rng = np.random.default_rng(361)
    n, targets = 4097, 1 if problem == "binary" else 3
    x = rng.normal(size=(n, 2)).astype(np.float32)
    labels = rng.integers(0, 2, size=(n, targets)).astype(np.float32)
    margin = rng.normal(size=(n, targets)).astype(np.float32)
    if ties:
        margin = rng.integers(-2, 3, size=margin.shape).astype(np.float32)
    params = dict(objective="reg:squarederror", num_target=targets)
    if problem == "multiclass":
        labels = rng.integers(0, targets, size=n)
        params = dict(objective="multi:softprob", num_class=targets)
        # Fixed score order avoids near-ties introduced by different CPU/GPU exp rounding.
        margin = np.tile(np.eye(3, dtype=np.float32), (n // 3 + 1, 1))[:n]
        if ties:
            margin[:] = 0
    weights = None
    if weight_mode != "none":
        weights = rng.integers(0, 16, n).astype(np.float32)
        if weight_mode == "tiny":
            weights *= np.nextafter(np.float32(0), np.float32(1))
        if weight_mode == "large":
            weights *= np.float32(1e35)
    d = xgb.DMatrix(x, label=labels, base_margin=margin, weight=weights, nthread=2)
    params |= dict(eval_metric=metric, base_score=.5, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    expected = score(cpu, d)
    before = xgb.metal_info()
    actual = score(gpu, d)
    after = xgb.metal_info()
    assert after[("roc_auc_dispatches" if metric == "auc" else "pr_auc_dispatches")] - before[("roc_auc_dispatches" if metric == "auc" else "pr_auc_dispatches")] == 1
    assert after["radix_sort_dispatches"] - before["radix_sort_dispatches"] == 1
    assert after["metric_dispatches"] - before["metric_dispatches"] == (1 if problem == "binary" else 2)
    input_bytes = 4 * (n * targets + np.asarray(labels).size + (0 if weights is None else n))
    summary_bytes = 0 if problem == "binary" else 24 * targets
    assert after["metric_host_input_bytes"] == before["metric_host_input_bytes"]
    assert after["metric_host_output_bytes"] == before["metric_host_output_bytes"]
    assert after["metric_shared_input_bytes"] - before["metric_shared_input_bytes"] == input_bytes + summary_bytes
    assert after["metric_shared_output_bytes"] - before["metric_shared_output_bytes"] == 24 * targets
    assert after["metric_copy_bytes"] - before["metric_copy_bytes"] == 24 * targets
    assert after["auc_final_readback_bytes"] - before["auc_final_readback_bytes"] == (0 if problem == "binary" else 8)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps", "eval_metric": metric})
    np.testing.assert_allclose(score(loaded, d), expected, rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
@pytest.mark.parametrize("problem", ["binary", "multiclass", "multilabel"])
@pytest.mark.parametrize("mode", ["empty", "one_class", "zero_weights"])
def test_roc_degenerate(metal, metric, problem, mode):
    n = 0 if mode == "empty" else 257
    targets = 1 if problem == "binary" else 3
    y = np.ones((n, targets), dtype=np.float32)
    if mode == "zero_weights":
        y[::2] = 0
    params = dict(objective="reg:squarederror", num_target=targets)
    if problem == "multiclass":
        y = y[:, 0]
        params = dict(objective="multi:softprob", num_class=targets)
    d = xgb.DMatrix(np.ones((n, 2), dtype=np.float32), label=y,
                    weight=np.zeros(n) if mode == "zero_weights" else None, nthread=2)
    params |= dict(eval_metric=metric, base_score=.5, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    before = xgb.metal_info()[("roc_auc_dispatches" if metric == "auc" else "pr_auc_dispatches")]
    np.testing.assert_equal(score(gpu, d), score(cpu, d))
    assert xgb.metal_info()[("roc_auc_dispatches" if metric == "auc" else "pr_auc_dispatches")] - before == int(n != 0)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
@pytest.mark.parametrize("targets", [1, 3])
def test_roc_training(metal, metric, targets):
    rng = np.random.default_rng(819)
    x = rng.normal(size=(8193, 5)).astype(np.float32)
    y = (x[:, :targets] > 0).astype(np.float32)
    d = xgb.QuantileDMatrix(x, label=y, nthread=2, max_bin=256)
    params = dict(objective="binary:logistic", eval_metric=metric, nthread=2, max_depth=4)
    cpu_log, gpu_log = {}, {}
    cpu = xgb.train(params | {"device": "cpu"}, d, 5, evals=[(d, "train")],
                    evals_result=cpu_log, verbose_eval=False)
    before = xgb.metal_info()[("roc_auc_dispatches" if metric == "auc" else "pr_auc_dispatches")]
    gpu = xgb.train(params | {"device": "mps"}, d, 5, evals=[(d, "train")],
                    evals_result=gpu_log, verbose_eval=False)
    assert xgb.metal_info()[("roc_auc_dispatches" if metric == "auc" else "pr_auc_dispatches")] - before == 5
    np.testing.assert_allclose(gpu_log["train"][metric], cpu_log["train"][metric], rtol=2e-12, atol=2e-10)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


def test_pr_extreme_weight_integral(metal):
    # With a huge negative weight ranked above a small positive one, true
    # PR-AUC is about 1/(2*w). The upstream subtraction of two rounded logs
    # returns one instead. CUDA uses that same expression, so source-aligned
    # Metal intentionally reproduces one; this is not an accuracy improvement.
    weights = np.array([1e30, 1], dtype=np.float32)
    d = xgb.DMatrix(np.ones((2, 1)), label=[0, 1], weight=weights, base_margin=[1, 0])
    params = dict(objective="reg:squarederror", eval_metric="aucpr", base_score=.5, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    assert score(cpu, d) == 1.0
    before = xgb.metal_info()["pr_auc_dispatches"]
    assert score(gpu, d) == 1.0
    assert xgb.metal_info()["pr_auc_dispatches"] == before + 1


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
def test_cuda_auc_underflow_validity(metal, metric):
    labels = np.full(2, np.nextafter(np.float32(1), np.float32(0)), dtype=np.float32)
    weights = np.full(2, np.finfo(np.float32).smallest_subnormal, dtype=np.float32)
    d = xgb.DMatrix(np.ones((2, 1)), label=labels, weight=weights, base_margin=[1, 0], nthread=2)
    p = dict(device="mps", eval_metric=metric, base_score=.5, nthread=2)
    gpu = xgb.train(p, d, 0)
    before = xgb.metal_info()["metric_dispatches"]
    with pytest.warns(UserWarning, match="only positive or negative"):
        assert np.isnan(score(gpu, d))
    assert xgb.metal_info()["metric_dispatches"] == before + 1


def cuda_curve_reference(labels, scores, weights, pr):
    """Source-formula oracle for well-conditioned CUDA classification curves."""
    order = np.argsort(-scores, kind="stable")
    fp_terms = np.array([(1.0 - float(labels[i])) * float(weights[i]) for i in order], dtype=np.float32)
    tp_terms = labels[order] * weights[order]
    total_fp = float(np.sum((np.float32(1) - labels[order]) * weights[order], dtype=np.float64))
    total_tp = float(np.sum(tp_terms, dtype=np.float64))
    if pr and (total_fp <= 0 or total_tp <= 0):
        return float("nan")
    fp, tp, fp0, tp0, area = 0.0, 0.0, 0.0, 0.0, 0.0
    for k, i in enumerate(order):
        fp += float(fp_terms[k])
        tp += float(tp_terms[k])
        if k + 1 < len(order) and scores[i] == scores[order[k + 1]]:
            continue
        if pr:
            previous, recall = tp0 / total_tp, tp / total_tp
            h = (fp - fp0) / (tp - tp0) if tp != tp0 else 0.0
            a = h + 1
            b = (fp0 - h * tp0) / total_tp if tp != tp0 else 0.0
            area += ((recall - previous - b / a *
                      (np.log(a * recall + b) - np.log(a * previous + b))) / a
                     if b != 0 else (recall - previous) / a)
        else:
            area += abs(fp - fp0) * (tp + tp0) * .5
        fp0, tp0 = fp, tp
    return area if pr else area / (fp * tp) if fp * tp else float("nan")


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
@pytest.mark.parametrize("targets", [1, 3])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("mode", ["normal", "tiny", "large"])
def test_cuda_auc_fractional_terms(metal, metric, targets, ties, mode):
    rng = np.random.default_rng(427)
    n = 4097
    labels = rng.uniform(.01, .99, (n, targets)).astype(np.float32)
    margins = (rng.integers(-4, 5, (n, targets)) if ties else rng.normal(size=(n, targets))).astype(np.float32)
    weights = rng.uniform(.1, 4, n).astype(np.float32)
    if mode == "tiny":
        weights = (rng.integers(1, 9, n).astype(np.float32) * np.finfo(np.float32).smallest_subnormal)
    elif mode == "large":
        weights *= np.float32(1e30)
    d = xgb.DMatrix(np.ones((n, 1)), label=labels, weight=weights, base_margin=margins, nthread=2)
    p = dict(device="mps", objective="reg:squarederror", num_target=targets,
             eval_metric=metric, base_score=.5, nthread=2)
    gpu = xgb.train(p, d, 0)
    expected = np.mean([cuda_curve_reference(labels[:, t], margins[:, t], weights, metric == "aucpr")
                        for t in range(targets)])
    before = xgb.metal_info()["metric_dispatches"]
    np.testing.assert_allclose(score(gpu, d), expected, rtol=2e-12, atol=2e-10)
    assert xgb.metal_info()["metric_dispatches"] == before + (1 if targets == 1 else 2)
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps", "eval_metric": metric})
    np.testing.assert_allclose(score(loaded, d), expected, rtol=2e-12, atol=2e-10)
