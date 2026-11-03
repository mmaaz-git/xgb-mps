"""CPU comparisons require both gradient and transform GPU execution."""
import numpy as np
import pytest
import xgboost as xgb

OBJECTIVES = [
    ("reg:squarederror", {}), ("reg:squaredlogerror", {}),
    ("reg:logistic", {"scale_pos_weight": 1.7}),
    ("binary:logistic", {"scale_pos_weight": 2.3}), ("binary:logitraw", {}),
    ("reg:gamma", {}), ("reg:pseudohubererror", {"huber_slope": 1.7}),
    ("count:poisson", {"max_delta_step": 0.4}),
    ("reg:tweedie", {"tweedie_variance_power": 1.3}), ("binary:hinge", {}),
    ("multi:softprob", {"num_class": 5}), ("multi:softmax", {"num_class": 5}),
    ("reg:expectileerror", {"expectile_alpha": [0.1, 0.5, 0.9]}),
    ("reg:absoluteerror", {}),
    ("reg:quantileerror", {"quantile_alpha": [0.1, 0.5, 0.9]}),
]

@pytest.mark.parametrize("objective,extra", OBJECTIVES)
@pytest.mark.parametrize("weighted", [False, True])
def test_objectives_execute_on_metal(metal, objective, extra, weighted):
    rng = np.random.default_rng(451)
    x = rng.normal(size=(513, 7)).astype(np.float32)
    if "logistic" in objective or objective.startswith("binary:"):
        y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)
    elif objective.startswith("multi:"):
        y = rng.integers(0, 5, len(x)).astype(np.float32)
    else:
        y = np.exp(x[:, 0] * 0.3 + x[:, 1] * 0.2).astype(np.float32)
    weights = rng.uniform(0.1, 2, len(x)).astype(np.float32) if weighted else None
    if weighted:
        weights[::19] = 0
    d = xgb.DMatrix(x, label=y, weight=weights)
    params = dict(objective=objective, tree_method="hist", max_depth=3,
                  max_bin=32, nthread=2, seed=9, **extra)
    cpu = xgb.train(params | {"device": "cpu"}, d, num_boost_round=4)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, d, num_boost_round=4)
    after = xgb.metal_info()
    assert after["objective_dispatches"] - before["objective_dispatches"] >= 4
    assert after["quantization_dispatches"] > before["quantization_dispatches"]
    assert after["gradient_bound_dispatches"] > before["gradient_bound_dispatches"]
    assert after["gradient_sum_dispatches"] > before["gradient_sum_dispatches"]
    uses_stump = objective in {"reg:squaredlogerror", "reg:pseudohubererror", "binary:hinge",
                              "reg:expectileerror", "reg:absoluteerror"} or extra.get("scale_pos_weight", 1) != 1
    assert after["stump_dispatches"] - before["stump_dispatches"] == int(uses_stump)
    np.testing.assert_allclose(gpu.predict(d, output_margin=True),
                               cpu.predict(d, output_margin=True), rtol=2e-5, atol=2e-5)
    before = xgb.metal_info()
    actual = gpu.predict(d)
    after = xgb.metal_info()
    np.testing.assert_allclose(actual, cpu.predict(d), rtol=2e-5, atol=2e-5)
    transformed = objective in {"reg:logistic", "binary:logistic", "reg:gamma", "count:poisson",
                               "reg:tweedie", "binary:hinge", "multi:softprob", "multi:softmax",
                               "reg:expectileerror", "reg:quantileerror"}
    assert (after["transform_dispatches"] > before["transform_dispatches"]) == transformed

@pytest.mark.parametrize("objective,extra", OBJECTIVES[:10])
def test_multitarget_gradients(metal, objective, extra):
    rng = np.random.default_rng(321)
    x = rng.normal(size=(257, 4)).astype(np.float32)
    y = (rng.uniform(size=(257, 3)) > 0.5).astype(np.float32) if "logistic" in objective or objective.startswith("binary:") else rng.uniform(0.1, 2, (257, 3)).astype(np.float32)
    d = xgb.DMatrix(x, label=y, weight=rng.uniform(0.1, 2, len(x)))
    p = dict(objective=objective, tree_method="hist", max_depth=2, max_bin=16, nthread=2, **extra)
    cpu = xgb.train(p | {"device": "cpu"}, d, 3)
    before = xgb.metal_info()["objective_dispatches"]
    gpu = xgb.train(p | {"device": "mps"}, d, 3)
    assert xgb.metal_info()["objective_dispatches"] - before >= 3
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)

@pytest.mark.parametrize("objective,label", [("binary:logistic", 2), ("reg:gamma", -1),
                                            ("reg:squaredlogerror", -2), ("count:poisson", -1),
                                            ("multi:softprob", 5)])
def test_invalid_labels_still_rejected(metal, objective, label):
    d = xgb.DMatrix(np.ones((10, 2)), label=np.full(10, label))
    p = dict(device="mps", objective=objective, base_score=0.5, nthread=2)
    if objective.startswith("multi:"):
        p["num_class"] = 3
    with pytest.raises(xgb.core.XGBoostError, match="label"):
        xgb.train(p, d, 1)

@pytest.mark.parametrize("objective", ["reg:absoluteerror", "reg:quantileerror"])
@pytest.mark.parametrize("case", ["zero_residual", "zero_weights", "tiny_weights", "large_scale", "multi_target"])
def test_adaptive_scales_and_boundaries(metal, objective, case):
    rng = np.random.default_rng(957)
    x = rng.normal(size=(1027, 5)).astype(np.float32)
    y = (x[:, 0] * 2 - x[:, 1]).astype(np.float32)
    weights = np.ones(len(x), dtype=np.float32)
    extra = {}
    if objective == "reg:quantileerror":
        extra["quantile_alpha"] = [0.01, 0.5, 0.99]
    elif case == "multi_target":
        y = np.column_stack([y, y * 2 + x[:, 2]])
        extra["multi_strategy"] = "multi_output_tree"
    if case == "zero_residual":
        y[:] = 0
    elif case == "zero_weights":
        weights[:] = 0
    elif case == "tiny_weights":
        weights[:] = 1e-12
    elif case == "large_scale":
        y *= 1e12
    d = xgb.DMatrix(x, label=y, weight=weights)
    p = dict(objective=objective, nthread=2, max_depth=3, max_bin=32,
             base_score=0, reg_lambda=0, min_child_weight=0, **extra)
    cpu = xgb.train(p | {"device": "cpu"}, d, 3)
    before = xgb.metal_info()
    gpu = xgb.train(p | {"device": "mps"}, d, 3)
    assert xgb.metal_info()["objective_dispatches"] - before["objective_dispatches"] == 3
    assert xgb.metal_info()["residual_scale_dispatches"] - before["residual_scale_dispatches"] == 3
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5,
                               atol=2e-5 * max(1, float(np.max(np.abs(y)))))
