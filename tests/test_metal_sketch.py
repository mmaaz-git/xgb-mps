"""Fresh matrices finalize numeric sketch levels on Metal with exact CPU cuts."""
import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb


@pytest.mark.parametrize("layout", ["dense", "missing", "sparse"])
@pytest.mark.parametrize("weights", ["none", "normal", "tiny", "zero"])
@pytest.mark.parametrize("categorical", [False, True])
def test_sketch_finalization(metal, layout, weights, categorical):
    rng = np.random.default_rng(971)
    x = rng.normal(size=(16387, 9)).astype(np.float32)
    y = (2 * x[:, 0] - x[:, 1] + x[:, 2] ** 2).astype(np.float32)
    x[:, 3] = (np.arange(len(x)) % 127).astype(np.float32) / 7
    x[:, 4] = (np.arange(len(x)) % 23 - 11) * np.finfo(np.float32).smallest_subnormal
    x[:, 8] = np.nan
    types = ["q"] * x.shape[1]
    if categorical:
        x[:, 5] = rng.integers(0, 7, len(x)).astype(np.float32) * 2
        types[5] = "c"
    if layout == "missing":
        x[rng.random(x.shape) < .4] = np.nan
    if layout == "sparse":
        x[rng.random(x.shape) < .7] = 0
        x = sp.csr_matrix(x)
    weight = None
    if weights != "none":
        weight = rng.uniform(0, 2, x.shape[0]).astype(np.float32)
        weight[::11] = 0
        if weights == "tiny":
            weight *= np.float32(2. ** -135)
        if weights == "zero":
            weight[:] = 0
    params = dict(max_bin=33, max_depth=3, nthread=2, seed=37, base_score=0)
    data = {device: xgb.DMatrix(x, label=y, weight=weight, feature_types=types, nthread=2)
            for device in ["cpu", "mps"]}
    before = xgb.metal_info()
    cpu = xgb.train(params | {"device": "cpu"}, data["cpu"], 3)
    assert xgb.metal_info()["sketch_finalize_dispatches"] == before["sketch_finalize_dispatches"]
    gpu = xgb.train(params | {"device": "mps"}, data["mps"], 3)
    after = xgb.metal_info()
    assert after["sketch_finalize_dispatches"] == before["sketch_finalize_dispatches"] + 1
    assert after["sketch_finalize_columns"] == before["sketch_finalize_columns"] + types.count("q")
    assert after["sketch_finalize_gpu_time_ns"] > before["sketch_finalize_gpu_time_ns"]
    for actual, expected in zip(data["mps"].get_quantile_cut(), data["cpu"].get_quantile_cut()):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_allclose(gpu.predict(data["mps"]), cpu.predict(data["cpu"]), rtol=2e-5, atol=2e-5)
    # Cached gradient indices must reuse their completed cuts on continuation.
    prior = xgb.metal_info()["sketch_finalize_dispatches"]
    xgb.train(params | {"device": "mps"}, data["mps"], 1, xgb_model=gpu)
    assert xgb.metal_info()["sketch_finalize_dispatches"] == prior


def test_categorical_only_sketch(metal):
    data = xgb.DMatrix(np.arange(30).reshape(10, 3) % 5, label=np.arange(10),
                       feature_types=["c"] * 3, nthread=2)
    before = xgb.metal_info()["sketch_finalize_dispatches"]
    xgb.train(dict(device="mps", max_depth=2, nthread=2), data, 2)
    assert xgb.metal_info()["sketch_finalize_dispatches"] == before


def test_group_weighted_sketch(metal):
    rng = np.random.default_rng(991)
    x = rng.normal(size=(8192, 5)).astype(np.float32)
    labels = rng.integers(0, 4, len(x))
    weights = np.array([0, .01, 10, 1, 1e-4, 2, 3, 9], dtype=np.float32)
    params = dict(objective="rank:pairwise", max_depth=2, max_bin=17, nthread=2, seed=31)
    matrices = {d: xgb.DMatrix(x, label=labels, group=[1024] * 8, weight=weights, nthread=2)
                for d in ["cpu", "mps"]}
    models = {d: xgb.train(params | {"device": d}, matrices[d], 2) for d in matrices}
    for actual, expected in zip(matrices["mps"].get_quantile_cut(), matrices["cpu"].get_quantile_cut()):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_allclose(models["mps"].predict(matrices["mps"]),
                               models["cpu"].predict(matrices["cpu"]), rtol=2e-5, atol=2e-5)


def test_external_page_sketch(metal, tmp_path):
    rng = np.random.default_rng(993)
    x = rng.normal(size=(4124, 5)).astype(np.float32)
    y = (x[:, 0] + x[:, 2] ** 2).astype(np.float32)
    weights = rng.uniform(.01, 10, len(x)).astype(np.float32)
    x[rng.random(x.shape) < .3] = np.nan

    class Pages(xgb.DataIter):
        def __init__(self, device):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / device))

        def reset(self):
            self.page = 0

        def next(self, input_data):
            if self.page == 4:
                return False
            begin = 1031 * self.page
            end = begin + 1031
            input_data(data=x[begin:end], label=y[begin:end], weight=weights[begin:end])
            self.page += 1
            return True

    data = {d: xgb.DMatrix(Pages(d), nthread=2) for d in ["cpu", "mps"]}
    params = dict(max_depth=3, max_bin=33, nthread=2, seed=31)
    before = xgb.metal_info()["sketch_finalize_dispatches"]
    models = {d: xgb.train(params | {"device": d}, data[d], 3) for d in data}
    assert xgb.metal_info()["sketch_finalize_dispatches"] == before + 1
    for actual, expected in zip(data["mps"].get_quantile_cut(), data["cpu"].get_quantile_cut()):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_allclose(models["mps"].predict(data["mps"]), models["cpu"].predict(data["cpu"]),
                               rtol=2e-5, atol=2e-5)
