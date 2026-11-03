"""AFT gradients, transforms and censored survival metrics must run on Metal."""
import json
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
@pytest.mark.parametrize("scale", [.5, 1., 2.])
@pytest.mark.parametrize("weighted", [False, True])
def test_survival_training(metal, distribution, scale, weighted):
    rng = np.random.default_rng(277)
    x = rng.normal(size=(701, 8)).astype(np.float32)
    y = np.exp(.3*x[:, 0] - .2*x[:, 1] + rng.normal(scale=.3, size=len(x))).astype(np.float32)
    lo, hi = y.copy(), y.copy()
    lo[1::5] = 0
    hi[2::5] = np.inf
    lo[3::5] *= .7
    hi[3::5] *= 1.4
    lo[4::5] = 0
    hi[4::5] = np.inf
    w = rng.uniform(.1, 3, len(x)).astype(np.float32) if weighted else None
    d = xgb.DMatrix(x, label_lower_bound=lo, label_upper_bound=hi, weight=w, nthread=2)
    p = dict(objective="survival:aft", aft_loss_distribution=distribution,
             aft_loss_distribution_scale=scale, max_depth=3, tree_method="hist", nthread=2,
             eval_metric=["aft-nloglik", "interval-regression-accuracy"])
    expected, actual = {}, {}
    cpu = xgb.train(p | {"device": "cpu"}, d, 5, evals=[(d, "data")], evals_result=expected, verbose_eval=False)
    before = xgb.metal_info()
    gpu = xgb.train(p | {"device": "mps"}, d, 5, evals=[(d, "data")], evals_result=actual, verbose_eval=False)
    after = xgb.metal_info()
    assert after["survival_gradient_dispatches"] - before["survival_gradient_dispatches"] == 5
    assert after["survival_metric_dispatches"] - before["survival_metric_dispatches"] == 10
    corrections = after["survival_cpu_corrections"] - before["survival_cpu_corrections"]
    if distribution == "extreme" and scale == .5:
        # This sharp distribution puts a few censored rows in the sensitive tail.
        # Verify the advertised correction path without allowing bulk CPU fallback.
        assert 0 < corrections < 5*len(x)/100
    else:
        assert corrections == 0
    for metric in expected["data"]:
        np.testing.assert_allclose(actual["data"][metric], expected["data"][metric], rtol=2e-6, atol=2e-6)
    before = xgb.metal_info()
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-6)
    assert xgb.metal_info()["transform_dispatches"] > before["transform_dispatches"]
    np.testing.assert_allclose(gpu.predict(d, output_margin=True), cpu.predict(d, output_margin=True), rtol=2e-5, atol=2e-6)
    reloaded = xgb.Booster(model_file=gpu.save_raw())
    reloaded.set_param({"device": "mps", "nthread": 2})
    np.testing.assert_array_equal(reloaded.predict(d), gpu.predict(d))
    assert json.loads(reloaded.save_config())["learner"]["objective"]["name"] == "survival:aft"


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
@pytest.mark.parametrize("weight_scale", [0., 1e-40, 1e30])
def test_survival_metric_weights(metal, distribution, weight_scale):
    rows = 257
    lo = np.linspace(.1, 10, rows, dtype=np.float32)
    hi = lo * 1.4
    hi[::3] = np.inf
    lo[1::3] = 0
    d = xgb.DMatrix(np.zeros((rows, 1)), label_lower_bound=lo, label_upper_bound=hi,
                    weight=np.full(rows, weight_scale), base_margin=np.linspace(-100, 100, rows), nthread=2)
    p = dict(objective="survival:aft", aft_loss_distribution=distribution, nthread=2,
             eval_metric=["aft-nloglik", "interval-regression-accuracy"])
    models = {device: xgb.train(p | {"device": device}, d, 0) for device in ["cpu", "mps"]}
    cpu = models["cpu"].eval(d)
    before = xgb.metal_info()
    gpu = models["mps"].eval(d)
    assert xgb.metal_info()["survival_metric_dispatches"] - before["survival_metric_dispatches"] == 2
    def parse(s): return [float(item.split(":")[-1]) for item in s.split()[1:]]
    np.testing.assert_allclose(parse(gpu), parse(cpu), rtol=2e-7, atol=1e-7)
