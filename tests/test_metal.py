"""Backend dispatch, equivalence, and model compatibility tests.

CPU execution never satisfies a GPU test: every nonempty training job must
increase the counter of completed Metal histogram dispatches.
"""
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xgboost as xgb


def dataset(seed=13, layout="dense", objective="reg:squarederror"):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1024, 12)).astype(np.float32)
    score = 2 * x[:, 0] - x[:, 1] + 0.7 * x[:, 2] ** 2
    if objective == "binary:logistic":
        y = (score > np.median(score)).astype(np.float32)
    elif objective == "multi:softprob":
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3])).astype(np.float32)
    else:
        y = score + rng.normal(scale=0.1, size=x.shape[0])
    if layout == "missing":
        x[rng.random(x.shape) < 0.23] = np.nan
        x[:, -1] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < 0.8] = 0
        x = sp.csr_matrix(x)
    weights = rng.uniform(0.1, 2, len(y)).astype(np.float32)
    weights[::17] = 0
    return x, y, weights


def parameters(objective="reg:squarederror", **kwargs):
    p = dict(tree_method="hist", objective=objective, max_depth=4, max_bin=64,
             nthread=2, seed=7, learning_rate=0.2)
    if objective == "multi:softprob":
        p["num_class"] = 3
    p.update(kwargs)
    return p


def train_metal(d, p, rounds=10, **kwargs):
    before = xgb.metal_info()["histogram_dispatches"]
    model = xgb.train({**p, "device": "mps"}, d, num_boost_round=rounds, **kwargs)
    assert xgb.metal_info()["histogram_dispatches"] > before
    assert json.loads(model.save_config())["learner"]["generic_param"]["device"] == "mps"
    return model


def test_cpu_still_trains_and_serializes(tmp_path):
    x, y, w = dataset()
    d = xgb.DMatrix(x, label=y, weight=w)
    before = xgb.metal_info()["histogram_dispatches"]
    model = xgb.train(parameters(device="cpu"), d, num_boost_round=8)
    prediction = model.predict(d)
    assert np.isfinite(prediction).all()
    assert np.mean((prediction - y) ** 2) < np.var(y)
    assert xgb.metal_info()["histogram_dispatches"] == before
    path = tmp_path / "model.ubj"
    model.save_model(path)
    loaded = xgb.Booster(model_file=path)
    np.testing.assert_array_equal(loaded.predict(d), prediction)


def test_no_gpu_means_error_not_silent_cpu_fallback():
    if xgb.metal_info()["available"]:
        pytest.skip("This check is specific to hosts without accessible Metal devices")
    d = xgb.DMatrix(np.arange(32, dtype=np.float32).reshape(16, 2), label=np.arange(16))
    with pytest.raises(xgb.core.XGBoostError, match="no Metal device|USE_METAL"):
        xgb.train(parameters(device="mps"), d, num_boost_round=1)
    assert xgb.metal_info()["histogram_dispatches"] == 0


@pytest.mark.parametrize("extra", [
    {"tree_method": "exact"}, {"tree_method": "approx"}, {"booster": "gblinear"},
    {"updater": "grow_quantile_histmaker"}, {"device": "mps:1"},
])
def test_unsupported_configuration_is_rejected(extra):
    x, y, _ = dataset()
    d = xgb.DMatrix(x, label=y)
    with pytest.raises(xgb.core.XGBoostError):
        xgb.train(parameters(device="mps", **{k: v for k, v in extra.items() if k != "device"})
                  | ({"device": extra["device"]} if "device" in extra else {}), d, num_boost_round=1)


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [13, 41])
@pytest.mark.parametrize("layout", ["dense", "missing", "sparse"])
@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "multi:softprob"])
def test_training_equivalence(metal, seed, layout, objective):
    x, y, w = dataset(seed, layout, objective)
    d = xgb.DMatrix(x, label=y, weight=w)
    p = parameters(objective)
    cpu = xgb.train({**p, "device": "cpu"}, d, num_boost_round=10)
    gpu = train_metal(d, p)
    np.testing.assert_allclose(gpu.predict(d, output_margin=True),
                               cpu.predict(d, output_margin=True), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("max_bin", [16, 256, 1024])
def test_quantile_matrix_and_bin_widths(metal, max_bin):
    x, y, _ = dataset()
    d = xgb.QuantileDMatrix(x, y, max_bin=max_bin)
    p = parameters(max_bin=max_bin)
    cpu = xgb.train(p, d, num_boost_round=6)
    gpu = train_metal(d, p, rounds=6)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("extra", [
    {"subsample": 0.7, "colsample_bytree": 0.7},
    {"grow_policy": "lossguide", "max_leaves": 12, "max_depth": 0},
    {"monotone_constraints": "(1,-1,0,0,0,0,0,0,0,0,0,0)"},
    {"interaction_constraints": "[[0,1,2],[3,4,5],[6,7,8,9,10,11]]"},
    {"reg_alpha": 0.3, "reg_lambda": 2.0, "max_delta_step": 0.5},
    {"max_cached_hist_node": 2},
])
def test_hist_features(metal, extra):
    x, y, w = dataset()
    d = xgb.DMatrix(x, label=y, weight=w)
    p = parameters(**extra)
    cpu = xgb.train(p, d, num_boost_round=6)
    gpu = train_metal(d, p, rounds=6)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
def test_categorical_and_model_cpu_interchange(metal, tmp_path):
    x, y, _ = dataset()
    frame = pd.DataFrame(x[:, :4], columns=["a", "b", "c", "d"])
    frame["category"] = pd.Categorical(np.arange(len(y)) % 7)
    d = xgb.DMatrix(frame, label=y, enable_categorical=True)
    p = parameters(max_cat_to_onehot=3)
    cpu = xgb.train(p, d, num_boost_round=6)
    gpu = train_metal(d, p, rounds=6)
    expected = gpu.predict(d)
    np.testing.assert_allclose(expected, cpu.predict(d), rtol=2e-5, atol=2e-5)
    path = tmp_path / "metal.ubj"
    gpu.save_model(path)
    loaded = xgb.Booster(params={"device": "cpu"}, model_file=path)
    np.testing.assert_array_equal(loaded.predict(d), expected)


@pytest.mark.gpu
def test_continuation_and_device_switch(metal):
    x, y, _ = dataset()
    d = xgb.DMatrix(x, label=y)
    p = parameters()
    first = train_metal(d, p, rounds=3)
    continued = train_metal(d, p, rounds=3, xgb_model=first)
    full = train_metal(d, p, rounds=6)
    np.testing.assert_array_equal(continued.predict(d), full.predict(d))
    before = xgb.metal_info()["histogram_dispatches"]
    cpu = xgb.train({**p, "device": "cpu"}, d, num_boost_round=1, xgb_model=continued)
    assert cpu.num_boosted_rounds() == 7
    assert xgb.metal_info()["histogram_dispatches"] == before
