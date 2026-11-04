"""Public objectives retain CPU label errors while validating on Metal."""
import numpy as np
import pytest
import xgboost as xgb

TINY = np.finfo(np.float32).smallest_subnormal
CASES = [
    ("reg:logistic", -TINY), ("binary:logistic", -TINY),
    ("binary:logitraw", np.nextafter(np.float32(1), np.float32(2))),
    ("reg:squaredlogerror", -1.), ("reg:gamma", 0.), ("reg:gamma", -TINY),
    ("count:poisson", -TINY), ("reg:tweedie", -TINY),
    ("multi:softprob", TINY), ("multi:softprob", 1.5),
    ("multi:softmax", -TINY), ("multi:softmax", 3.),
]


@pytest.mark.parametrize("objective,invalid", CASES)
def test_invalid_labels_on_gpu(metal, objective, invalid):
    x = np.arange(513*2, dtype=np.float32).reshape(513, 2)
    y = np.full(513, 1, dtype=np.float32)
    y[257] = invalid
    data = xgb.DMatrix(x, label=y, nthread=2)
    params = dict(objective=objective, base_score=.5, nthread=2, max_depth=1)
    if objective.startswith("multi:"):
        params["num_class"] = 3
    before = xgb.metal_info()["label_validation_dispatches"]
    for device in ["cpu", "mps"]:
        with pytest.raises(xgb.core.XGBoostError, match="label must"):
            xgb.train(params | {"device": device}, data, 1)
        assert xgb.metal_info()["label_validation_dispatches"] == before + int(device == "mps")


@pytest.mark.parametrize("objective", ["reg:logistic", "binary:logistic", "binary:logitraw",
    "reg:squaredlogerror", "reg:gamma", "count:poisson", "reg:tweedie", "multi:softprob", "multi:softmax"])
@pytest.mark.parametrize("targets", [1, 3])
def test_label_validation_training(metal, objective, targets):
    rng = np.random.default_rng(913)
    x = rng.normal(size=(1025, 5)).astype(np.float32)
    multi = objective.startswith("multi:")
    y = rng.integers(0, 3, len(x)).astype(np.float32) if multi else rng.uniform(.1, .9, (len(x), targets)).astype(np.float32)
    data = xgb.DMatrix(x, label=y, weight=rng.uniform(0, 2, len(x)), nthread=2)
    params = dict(objective=objective, base_score=.5, nthread=2, max_depth=2, max_bin=32, eta=.1)
    if multi:
        params["num_class"] = 3
    cpu = xgb.train(params | {"device": "cpu"}, data, 3)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data, 3)
    after = xgb.metal_info()
    assert after["label_validation_dispatches"] == before["label_validation_dispatches"] + 1
    assert after["label_validation_gpu_time_ns"] > before["label_validation_gpu_time_ns"]
    np.testing.assert_allclose(gpu.predict(data), cpu.predict(data), rtol=2e-5, atol=2e-5)


def test_unconditional_label_check_skips_dispatch(metal):
    d = xgb.DMatrix(np.ones((257, 2)), label=np.linspace(-1, 1, 257))
    before = xgb.metal_info()["label_validation_dispatches"]
    xgb.train(dict(device="mps", nthread=2, objective="reg:squarederror"), d, 1)
    assert xgb.metal_info()["label_validation_dispatches"] == before
