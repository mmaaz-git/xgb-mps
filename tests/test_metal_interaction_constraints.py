"""CUDA interaction behavior: overlapping groups, training integration and reuse."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from cuda_split_reference import RecordedSquaredError, audit_recorded, audit_tree, trees
from cuda_multi_split_reference import audit_recorded as audit_vector
from cuda_multi_split_reference import audit_tree as audit_vector_tree


MODES = [("one_output_per_tree", "hist"), ("one_output_per_tree", "approx"),
         ("multi_output_tree", "hist")]


def fixture(strategy, method, kind="numeric"):
    x = np.tile(list(itertools.product([0., 1.], repeat=3)), (16, 1)).astype(np.float32)
    y = 8*x[:, 0] + 4*x[:, 1] + 2*x[:, 2]
    if strategy == "multi_output_tree":
        y = np.column_stack([y, -y*.5]).astype(np.float32)
    types = ["q"]*3 if kind == "numeric" else ["c"]*3
    d = xgb.DMatrix(x, label=y, feature_types=types)
    p = dict(device="mps", tree_method=method, multi_strategy=strategy, nthread=2,
             base_score=0, learning_rate=.25, reg_lambda=0, min_child_weight=0,
             max_depth=3, max_bin=4, max_cached_hist_node=1,
             max_cat_to_onehot=8 if kind == "onehot" else 1,
             interaction_constraints="[[0,1],[1,2]]")
    return x, d, p


def internal_paths(tree):
    def visit(node, path):
        left, right = tree["left_children"][node], tree["right_children"][node]
        if left < 0:
            yield path
        else:
            path = path + [tree["split_indices"][node]]
            yield from visit(left, path)
            yield from visit(right, path)
    return list(visit(0, []))


def check_counters(before, model, enabled=True):
    after = xgb.metal_info()
    for key in ("feature_host_input_bytes", "feature_readback_bytes"):
        assert after[key] == before[key], key
    assert after["feature_task_dispatches"] > before["feature_task_dispatches"]
    models = trees(model)
    splits = sum(sum(child >= 0 for child in t["left_children"]) for t in models)
    assert after["interaction_reset_dispatches"] - before["interaction_reset_dispatches"] == (
        len(models) if enabled else 0)
    assert after["interaction_split_dispatches"] - before["interaction_split_dispatches"] == (
        splits if enabled else 0)
    if enabled:
        assert after["interaction_query_dispatches"] > before["interaction_query_dispatches"]
    else:
        assert after["interaction_query_dispatches"] == before["interaction_query_dispatches"]


def check_predictions(model, matrix, data):
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(matrix), cpu.predict(matrix))
    np.testing.assert_array_equal(model.inplace_predict(data), cpu.inplace_predict(data))


@pytest.mark.parametrize("strategy,method", MODES)
@pytest.mark.parametrize("growth", ["depthwise", "lossguide"])
@pytest.mark.parametrize("kind", ["numeric", "onehot", "partition"])
def test_overlapping_groups_follow_cuda(metal, strategy, method, growth, kind):
    x, d, p = fixture(strategy, method, kind)
    p["grow_policy"] = growth
    recorded = RecordedSquaredError()
    before = xgb.metal_info()
    model = xgb.train(p, d, 3, obj=recorded)
    check_counters(before, model)
    assert all(path == [0, 1, 2] for path in internal_paths(trees(model)[0]))
    audit = audit_vector if strategy == "multi_output_tree" else audit_recorded
    assert min(audit(model, d, x, recorded, p)) > 0
    check_predictions(model, d, x)
    # CPU's history intersection forbids feature 2 after splits 0 and 1.
    # This deliberate tree difference is why CPU-only comparison is insufficient.
    cpu_model = xgb.train({**p, "device": "cpu"}, d, 1)
    assert all(path == [0, 1] for path in internal_paths(trees(cpu_model)[0]))


@pytest.mark.parametrize("strategy,method", MODES)
@pytest.mark.parametrize("groups", [None, "[]", "[[],[0,0,1],[1,2,2],[]]"])
def test_disabled_empty_and_duplicate_groups(metal, strategy, method, groups):
    x, d, p = fixture(strategy, method)
    if groups is None:
        del p["interaction_constraints"]
    else:
        p["interaction_constraints"] = groups
    recorded = RecordedSquaredError()
    before = xgb.metal_info()
    model = xgb.train(p, d, 2, obj=recorded)
    check_counters(before, model, groups is not None)
    expected = [0] if groups == "[]" else [0, 1, 2]
    assert all(path == expected for path in internal_paths(trees(model)[0]))
    audit = audit_vector if strategy == "multi_output_tree" else audit_recorded
    assert min(audit(model, d, x, recorded, p)) > 0
    check_predictions(model, d, x)


@pytest.mark.parametrize("strategy,method", MODES)
def test_reconfigure_constraints_when_continuing_model(metal, strategy, method):
    x, d, p = fixture(strategy, method)
    model = xgb.train(p, d, 1)
    assert all(path == [0, 1, 2] for path in internal_paths(trees(model)[0]))
    p["interaction_constraints"] = "[[0,2],[1,2]]"
    # Preserve the loaded model's intercept vector; reapplying scalar base_score
    # after loading a multi-target model fails upstream length validation.
    del p["base_score"]
    recorded = RecordedSquaredError()
    before = xgb.metal_info()
    continued = xgb.train(p, d, 1, obj=recorded, xgb_model=model.save_raw())
    tree = trees(continued)[-1]
    assert all(path == [0, 2, 1] for path in internal_paths(tree))
    after = xgb.metal_info()
    assert after["interaction_reset_dispatches"] - before["interaction_reset_dispatches"] == 1
    assert after["interaction_split_dispatches"] - before["interaction_split_dispatches"] == 7
    g, h = recorded.gradients[0]
    if strategy == "multi_output_tree":
        audit_vector_tree(tree, x, d.get_quantile_cut(), d.feature_types, g, h, p)
    else:
        audit_tree(tree, x, d.get_quantile_cut(), d.feature_types, g.reshape(-1), h.reshape(-1), p)
    check_predictions(continued, d, x)
