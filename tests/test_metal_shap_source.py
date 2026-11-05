"""Production Metal SHAP against a CUDA-expression FP32 replay."""
import json

import numpy as np
import pytest
import xgboost as xgb

from cuda_shap_reference import predict as source_predict
from test_metal_shap import handcrafted_tree, metal_copy


def compare_source(cpu, x, interactions, feature_types=None, base_margin=None):
    raw = json.loads(cpu.save_raw(raw_format="json"))
    expected = source_predict(raw, x, interactions, base_margin)
    matrix = xgb.DMatrix(x, nthread=2, feature_types=feature_types,
                         enable_categorical=True, base_margin=base_margin)
    gpu = metal_copy(cpu)
    args = {"pred_interactions" if interactions else "pred_contribs": True, "strict_shape": True}
    before = xgb.metal_info()
    actual = gpu.predict(matrix, **args)
    after = xgb.metal_info()
    key = "shap_interaction_dispatches" if interactions else "shap_dispatches"
    assert after[key] > before[key]
    # Exact finite FP32 fields; NaN payload bits are not part of the contract.
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint32), expected[valid].view(np.uint32))
    return actual, raw, matrix


@pytest.mark.parametrize("kind", ["regression", "multiclass", "vector", "dart", "weighted_vector"])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("interactions", [False, True])
def test_shap_cuda_expressions(metal, kind, categorical, interactions):
    rng = np.random.default_rng(3431)
    x = rng.normal(size=(257, 3)).astype(np.float32)
    feature_types = ["q"] * 3
    if categorical:
        x[:, 0] = rng.integers(0, 73, len(x))
        feature_types[0] = "c"
    y = np.sin(x[:, 0]) * x[:, 1] - x[:, 2] ** 2
    params = dict(nthread=2, max_depth=4, base_score=.2, seed=37, max_cat_to_onehot=1)
    groups = 1
    if kind == "multiclass":
        groups = 3
        params.update(objective="multi:softprob", num_class=groups)
        y = np.digitize(y, [-1.5, -.1])
    elif kind in {"vector", "weighted_vector"}:
        groups = 3
        params["multi_strategy"] = "multi_output_tree"
        y = np.column_stack([y, -y * .731, y * .173])
    elif kind == "dart":
        params.update(booster="dart", rate_drop=.5)
    if kind == "weighted_vector":
        params["rate_drop"] = .5
    train = xgb.DMatrix(x, label=y, feature_types=feature_types,
                        enable_categorical=True, nthread=2)
    cpu = xgb.train(params, train, 3)
    if kind in {"dart", "weighted_vector"}:
        raw = json.loads(cpu.save_raw(raw_format="json"))
        raw["learner"]["gradient_booster"]["model"]["weight_drop"] = [.73113, -.17341, 0.]
        cpu = xgb.Booster(model_file=bytearray(json.dumps(raw).encode()))
        cpu.set_param({"device": "cpu", "nthread": 2})
        saved = json.loads(cpu.save_raw(raw_format="json"))
        np.testing.assert_array_equal(
            np.asarray(saved["learner"]["gradient_booster"]["model"]["weight_drop"], dtype=np.float32),
            np.asarray([.73113, -.17341, 0.], dtype=np.float32))
    values = x[:7].copy()
    values[::3, :] = np.nan
    if categorical:
        values[1, 0], values[2, 0], values[4, 0] = -1., 1000., 16777216.
    margin = rng.normal(size=(len(values), groups)).astype(np.float32)
    actual, raw, matrix = compare_source(cpu, values, interactions, feature_types, margin)
    # Ordinary CPU comparisons retain the original tolerance.
    args = {"pred_interactions" if interactions else "pred_contribs": True, "strict_shape": True}
    np.testing.assert_allclose(actual, cpu.predict(matrix, **args), rtol=2e-5, atol=2e-5)
    serial = source_predict(raw, values, interactions, margin, serial=True)
    assert np.any(serial.view(np.uint32) != actual.view(np.uint32)), "Missing reduction-order witness"


@pytest.mark.parametrize("depth", [2, 17, 65, 127, 130])
@pytest.mark.parametrize("zero_cover", [False, True])
@pytest.mark.parametrize("interactions", [False, True])
def test_shap_cuda_deep_recurrence(metal, depth, zero_cover, interactions):
    cpu, _ = handcrafted_tree(depth, zero_cover)
    values = np.array([[-1, -1, -1], [1, 1, 1], [np.nan] * 3, [.5, -1, np.nan]], dtype=np.float32)
    actual, _, matrix = compare_source(cpu, values, interactions)
    assert np.isfinite(actual).all()
    if zero_cover and depth >= 127:
        args = {"pred_interactions" if interactions else "pred_contribs": True, "strict_shape": True}
        assert np.isnan(cpu.predict(matrix, **args)).any()
        # CUDA's recurrence avoids the separate path-product underflow and
        # basis overflow of this CPU formulation. Depth130 is a Metal extension
        # beyond CUDA's depth127 limit, checked against the same recurrence.


@pytest.mark.parametrize("interactions", [False, True])
@pytest.mark.parametrize("cover", [0., 1e-35, 1e30])
@pytest.mark.parametrize("leaf_scale", [1e-40, 1e30])
def test_shap_cuda_extreme_payloads(metal, interactions, cover, leaf_scale):
    cpu, _ = handcrafted_tree(3)
    raw = json.loads(cpu.save_raw(raw_format="json"))
    model = raw["learner"]["gradient_booster"]["model"]
    tree = model["trees"][0]
    tree["sum_hessian"][1] = cover
    for i, left in enumerate(tree["left_children"]):
        if left == -1:
            tree["split_conditions"][i] *= leaf_scale
    model["weight_drop"] = [.73113]
    cpu = xgb.Booster(model_file=bytearray(json.dumps(raw).encode()))
    cpu.set_param({"device": "cpu", "nthread": 2})
    x = np.array([[-1, -1, -1], [1, 1, 1], [np.nan] * 3], dtype=np.float32)
    compare_source(cpu, x, interactions, base_margin=np.zeros((3, 1), dtype=np.float32))
