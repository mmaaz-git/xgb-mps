"""Quantile/expectile metrics must execute Metal reductions and match CPU math."""
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("objective,metric,param", [
    ("reg:quantileerror", "quantile", "quantile_alpha"),
    ("reg:expectileerror", "expectile", "expectile_alpha"),
])
@pytest.mark.parametrize("levels", [1, 3, 17])
@pytest.mark.parametrize("weight_scale", [None, 1.0, 1e-30, 1e30])
def test_quantile_metrics_on_metal(metal, objective, metric, param, levels, weight_scale):
    rng = np.random.default_rng(113)
    x = rng.normal(size=(513, 5)).astype(np.float32)
    y = (x[:, 0] * 2 - x[:, 1]).astype(np.float32)
    alpha = np.linspace(0.05, 0.95, levels, dtype=np.float32)
    if levels == 1:
        alpha[:] = 0.5
    train = xgb.DMatrix(x, label=y)
    params = dict(objective=objective, nthread=2, max_depth=2, max_bin=16,
                  eval_metric=metric, **{param: alpha.tolist()})
    cpu = xgb.train(params | {"device": "cpu"}, train, 2)
    gpu = xgb.Booster(params={"device": "mps"}, model_file=cpu.save_raw())
    weights = None
    if weight_scale is not None:
        weights = (rng.uniform(0.1, 2, len(x)) * weight_scale).astype(np.float32)
        weights[::11] = 0
    d = xgb.DMatrix(x, label=y, weight=weights)
    before = xgb.metal_info()
    expected = float(cpu.eval(d).rsplit(":", 1)[1])
    assert xgb.metal_info()["metric_dispatches"] == before["metric_dispatches"]
    actual = float(gpu.eval(d).rsplit(":", 1)[1])
    after = xgb.metal_info()
    assert after["metric_dispatches"] == before["metric_dispatches"] + 1
    assert after["metric_reduction_passes"] > before["metric_reduction_passes"]
    assert after["metric_final_readback_bytes"] == before["metric_final_readback_bytes"] + 16
    assert after["metric_host_input_bytes"] == before["metric_host_input_bytes"]
    input_bytes = (len(x) * levels + y.size + levels + (0 if weights is None else weights.size)) * 4
    assert after["metric_shared_input_bytes"] - before["metric_shared_input_bytes"] == input_bytes
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)
    pred = gpu.predict(d).reshape(len(x), levels)
    diff = pred - y[:, None]
    if metric == "quantile":
        loss = np.where(diff >= 0, (1 - alpha) * diff, -alpha * diff)
    else:
        loss = np.where(diff >= 0, 1 - alpha, alpha) * diff * diff
    w = np.ones(len(x), dtype=np.float32) if weights is None else weights
    reference = np.sum(loss * w[:, None], dtype=np.float64) / (np.sum(w, dtype=np.float64) * levels)
    np.testing.assert_allclose(actual, reference, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("objective,param", [
    ("reg:quantileerror", "quantile_alpha"),
    ("reg:expectileerror", "expectile_alpha"),
])
def test_quantile_metrics_zero_weight_error(metal, objective, param):
    x = np.arange(32, dtype=np.float32).reshape(16, 2)
    train = xgb.DMatrix(x, label=x[:, 0])
    model = xgb.train(dict(objective=objective, nthread=2, **{param: [0.1, 0.9]}), train, 1)
    empty_weight = xgb.DMatrix(x, label=x[:, 0], weight=np.zeros(len(x)))
    for device in ["cpu", "mps"]:
        model.set_param({"device": device})
        with pytest.raises(xgb.core.XGBoostError, match="Check failed"):
            model.eval(empty_weight)


@pytest.mark.parametrize("objective,metric,param", [
    ("reg:quantileerror", "quantile", "quantile_alpha"),
    ("reg:expectileerror", "expectile", "expectile_alpha"),
])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_ordered_metrics_reject_query_weights(metal, objective, metric, param, device):
    # One row per group makes the weight lengths valid under either convention.
    # The row-wise metric must still reject query weights, just as CUDA does.
    matrix = xgb.DMatrix(np.ones((3, 1), dtype=np.float32), label=[0, 1, 2],
                         group=[1, 1, 1], weight=[1, 1, 1])
    model = xgb.train(dict(objective=objective, eval_metric=metric, device=device,
                           nthread=2, base_score=.5, **{param: [.1, .9]}), matrix, 0)
    before = xgb.metal_info()
    with pytest.raises(xgb.core.XGBoostError, match="Row-wise metric does not support query group weights"):
        model.eval(matrix)
    assert xgb.metal_info()["metric_dispatches"] == before["metric_dispatches"]
