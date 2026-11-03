"""Backend dispatch, equivalence, and model compatibility tests.

CPU execution never satisfies a GPU test: every nonempty training job must
increase the counter of completed Metal histogram dispatches.
"""
import json
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.gpu
def test_shared_histogram_training(metal):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32769, 9)).astype(np.float32)
    y = 3 * x[:, 0] + x[:, 1] ** 2 - x[:, 2]
    d = xgb.DMatrix(x, label=y)
    before = xgb.metal_info()["shared_histogram_dispatches"]
    p = parameters(max_bin=256)
    gpu = train_metal(d, p, rounds=4)
    cpu = xgb.train(p, d, num_boost_round=4)
    assert xgb.metal_info()["shared_histogram_dispatches"] > before
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_multiple_targets(metal, strategy):
    x, y, _ = dataset()
    y = np.column_stack([y, y * 2 + x[:, 3]])
    d = xgb.DMatrix(x, label=y)
    p = parameters(multi_strategy=strategy)
    cpu = xgb.train(p, d, num_boost_round=5)
    gpu = train_metal(d, p, rounds=5)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
def test_external_memory_pages(metal, tmp_path):
    x, y, _ = dataset(layout="missing")

    class Batches(xgb.DataIter):
        def __init__(self):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / "batches"))

        def reset(self):
            self.page = 0

        def next(self, input_data):
            if self.page == 4:
                return False
            start = self.page * 256
            input_data(data=x[start:start+256], label=y[start:start+256])
            self.page += 1
            return True

    d = xgb.ExtMemQuantileDMatrix(Batches(), max_bin=64)
    p = parameters()
    cpu = xgb.train(p, d, num_boost_round=4)
    gpu = train_metal(d, p, rounds=4)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1e6, 1e12])
def test_custom_gradient_scales(metal, scale):
    x, y, _ = dataset()
    d = xgb.DMatrix(x, label=y)
    p = parameters(base_score=0, reg_lambda=0, min_child_weight=0)

    def objective(prediction, data):
        g = (prediction - data.get_label()) * scale
        h = np.full(prediction.shape, scale, dtype=np.float32)
        return g, h

    cpu = xgb.train(p, d, num_boost_round=3, obj=objective)
    gpu = train_metal(d, p, rounds=3, obj=objective)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (17, 1), (513, 24)])
def test_small_and_uneven_shapes(metal, shape):
    rng = np.random.default_rng(95)
    x = rng.integers(0, 8, size=shape).astype(np.float32)
    if shape[1] > 1:
        x[:, -1] = np.nan
    y = rng.normal(size=shape[0]).astype(np.float32)
    d = xgb.DMatrix(x, label=y)
    p = parameters(base_score=0)
    cpu = xgb.train(p, d, num_boost_round=3)
    gpu = train_metal(d, p, rounds=3)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
def test_sklearn_and_held_out_predictions(metal):
    x, y, w = dataset()
    before = xgb.metal_info()["histogram_dispatches"]
    params = dict(n_estimators=8, max_depth=3, max_bin=64, n_jobs=2, tree_method="hist")
    cpu = xgb.XGBRegressor(**params, device="cpu").fit(x[:800], y[:800], sample_weight=w[:800])
    gpu = xgb.XGBRegressor(**params, device="mps").fit(x[:800], y[:800], sample_weight=w[:800])
    assert xgb.metal_info()["histogram_dispatches"] > before
    np.testing.assert_allclose(gpu.predict(x[800:]), cpu.predict(x[800:]), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
def test_ranking(metal):
    x, y, _ = dataset()
    d = xgb.DMatrix(x, label=np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75])))
    d.set_group([16] * 64)
    p = parameters(objective="rank:pairwise")
    cpu = xgb.train(p, d, num_boost_round=4)
    gpu = train_metal(d, p, rounds=4)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
def test_concurrent_independent_models(metal):
    def fit(seed):
        x, y, w = dataset(seed)
        d = xgb.DMatrix(x, label=y, weight=w)
        p = parameters()
        cpu = xgb.train(p, d, num_boost_round=4)
        gpu = train_metal(d, p, rounds=4)
        np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fit, [7, 9, 11, 13]))
