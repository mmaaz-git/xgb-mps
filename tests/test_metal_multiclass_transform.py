"""Public multiclass prediction with margins that stress CUDA softmax rules."""
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("classes", [2, 17, 257])
@pytest.mark.parametrize("objective", ["multi:softprob", "multi:softmax"])
@pytest.mark.parametrize("case", ["ordinary", "negative", "peaked", "ties", "subnormal"])
def test_multiclass_transform_source_behavior(metal, classes, objective, case):
    rng = np.random.default_rng(550)
    margins = rng.uniform(-15, 15, (19, classes)).astype(np.float32)
    if case == "negative":
        margins -= np.float32(150)
    elif case == "peaked":
        margins[:] = -90
        margins[np.arange(19), np.arange(19) % classes] = 90
    elif case == "ties":
        margins[:] = -5
        margins[:, 1::2] = 0
    elif case == "subnormal":
        margins = (rng.integers(-1000, 1000, margins.shape).astype(np.float32)
                   * np.nextafter(np.float32(0), np.float32(1)))
    data = xgb.DMatrix(np.ones((19, 1)), base_margin=margins, nthread=2)
    params = dict(objective=objective, num_class=classes, base_score=0,
                  boost_from_average=0, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, data, 0)
    gpu = xgb.train(params | {"device": "mps"}, data, 0)
    before = xgb.metal_info()["transform_dispatches"]
    actual = gpu.predict(data)
    assert xgb.metal_info()["transform_dispatches"] == before + 1
    expected = cpu.predict(data)
    if objective == "multi:softmax":
        np.testing.assert_array_equal(actual, margins.argmax(axis=1).astype(np.float32))
        np.testing.assert_array_equal(actual, expected)
    else:
        # The CPU and Metal exp primitives can differ; source cast/reduction
        # boundaries have separate exact native tests. No absolute-error floor.
        np.testing.assert_array_max_ulp(actual, expected, maxulp=4)
        np.testing.assert_allclose(actual.sum(axis=1), 1, rtol=2e-7, atol=0)
    np.testing.assert_array_equal(gpu.predict(data, output_margin=True), margins)
    reloaded = xgb.Booster(model_file=gpu.save_raw())
    reloaded.set_param({"device": "mps", "nthread": 2})
    np.testing.assert_array_equal(reloaded.predict(data), actual)
