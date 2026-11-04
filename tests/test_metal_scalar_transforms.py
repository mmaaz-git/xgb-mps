"""Public objective transforms preserve CUDA float comparisons and boundaries."""
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:gamma", "binary:hinge"])
@pytest.mark.parametrize("case", ["ordinary", "tiny", "extreme"])
def test_scalar_transform_boundaries(metal, objective, case):
    if case == "ordinary":
        margin = np.linspace(-12, 12, 513, dtype=np.float32)
    elif case == "tiny":
        margin = np.arange(-256, 257, dtype=np.float32) * np.nextafter(np.float32(0), np.float32(1))
    else:
        margin = np.asarray([-110, -104, -103, -90, -88.7, -20, -0., 0., 20, 80, 88, 88.7, 90], dtype=np.float32)
    data = xgb.DMatrix(np.ones((len(margin), 1)), base_margin=margin, nthread=2)
    params = dict(objective=objective, base_score=.5, boost_from_average=0, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, data, 0)
    gpu = xgb.train(params | {"device": "mps"}, data, 0)
    before = xgb.metal_info()["transform_dispatches"]
    actual = gpu.predict(data)
    assert xgb.metal_info()["transform_dispatches"] == before + 1
    expected = cpu.predict(data)
    if objective == "binary:hinge":
        np.testing.assert_array_equal(actual, margin > 0)
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_array_max_ulp(actual, expected, maxulp=4)
    np.testing.assert_array_equal(gpu.predict(data, output_margin=True), margin)
    reload = xgb.Booster(model_file=gpu.save_raw())
    reload.set_param({"device": "mps", "nthread": 2})
    np.testing.assert_array_equal(reload.predict(data), actual)


@pytest.mark.parametrize("objective", ["reg:quantileerror", "reg:expectileerror"])
@pytest.mark.parametrize("case", ["ordinary", "tiny", "extreme"])
def test_row_transform_boundaries(metal, objective, case):
    rng = np.random.default_rng(867)
    margins = rng.uniform(-6, 6, (19, 3)).astype(np.float32)
    if case == "tiny":
        margins = rng.integers(-1000, 1000, margins.shape).astype(np.float32) * np.nextafter(np.float32(0), np.float32(1))
    elif case == "extreme":
        margins = rng.choice(np.asarray([-110, -90, -20, -0., 0., 20, 90, 110], dtype=np.float32), margins.shape)
    data = xgb.DMatrix(np.ones((19, 1)), label=np.zeros(19), base_margin=margins, nthread=2)
    level_name = "quantile_alpha" if objective == "reg:quantileerror" else "expectile_alpha"
    params = dict(objective=objective, base_score=.5, boost_from_average=0, nthread=2,
                  **{level_name: [.1, .5, .9]})
    cpu = xgb.train(params | {"device": "cpu"}, data, 0)
    gpu = xgb.train(params | {"device": "mps"}, data, 0)
    before = xgb.metal_info()["transform_dispatches"]
    actual = gpu.predict(data)
    assert xgb.metal_info()["transform_dispatches"] == before + 1
    expected = cpu.predict(data)
    if objective == "reg:quantileerror":
        np.testing.assert_array_equal(actual, np.sort(margins, axis=1))
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_array_max_ulp(actual, expected, maxulp=4)
    np.testing.assert_array_equal(gpu.predict(data, output_margin=True), margins)
    reload = xgb.Booster(model_file=gpu.save_raw())
    reload.set_param({"device": "mps", "nthread": 2})
    np.testing.assert_array_equal(reload.predict(data), actual)
