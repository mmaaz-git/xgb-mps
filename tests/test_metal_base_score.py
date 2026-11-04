"""Base-score validation/conversion must use Metal while preserving CPU margins."""
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logitraw", "reg:logistic",
    "binary:logistic", "reg:gamma", "count:poisson", "reg:tweedie", "survival:cox"])
@pytest.mark.parametrize("base", [.1, .5, .9])
def test_base_score_conversion(metal, objective, base):
    d = xgb.DMatrix(np.ones((257, 3)), label=np.ones(257), nthread=2)
    params = dict(objective=objective, base_score=base, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    expected = cpu.predict(d, output_margin=True)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    actual = gpu.predict(d, output_margin=True)
    after = xgb.metal_info()
    identity = objective in {"reg:squarederror", "binary:logitraw"}
    assert (after["base_score_dispatches"] > before["base_score_dispatches"]) == (not identity)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps"})
    np.testing.assert_array_equal(loaded.predict(d, output_margin=True).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("objective,base", [("reg:logistic", -.1), ("binary:logistic", 1.1),
                                           ("reg:gamma", 0.), ("reg:gamma", -.1)])
def test_invalid_intercepts(metal, objective, base):
    d = xgb.DMatrix(np.ones((33, 2)), label=np.ones(33), nthread=2)
    before = xgb.metal_info()["base_score_dispatches"]
    for device in ["cpu", "mps"]:
        with pytest.raises(xgb.core.XGBoostError, match="base_score.*must"):
            xgb.train(dict(device=device, objective=objective, base_score=base, nthread=2), d, 1)
        assert (xgb.metal_info()["base_score_dispatches"] > before) == (device == "mps")


@pytest.mark.parametrize("base", [0., 1., 1e-8, 1.-1e-7])
def test_logistic_clamp(metal, base):
    d = xgb.DMatrix(np.ones((33, 2)), label=np.ones(33), nthread=2)
    params = dict(objective="binary:logistic", base_score=base, nthread=2)
    cpu = xgb.train(params | {"device": "cpu"}, d, 0)
    gpu = xgb.train(params | {"device": "mps"}, d, 0)
    np.testing.assert_array_equal(cpu.predict(d, output_margin=True).view(np.uint32),
                                  gpu.predict(d, output_margin=True).view(np.uint32))
