"""GPU split finalization must be visible in production scalar/vector trees."""
import numpy as np
import pytest
import xgboost as xgb

from cuda_split_reference import RecordedSquaredError, trees, audit_recorded as audit_scalar
from cuda_multi_split_reference import audit_recorded as audit_vector, audit_tree


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("growth", ["depthwise", "lossguide"])
def test_gpu_split_finalization_and_bounds(metal, strategy, constrained, growth):
    rng = np.random.default_rng(361)
    x = rng.normal(size=(193, 5)).astype(np.float32)
    y = np.column_stack([x[:, 0] - x[:, 1], -x[:, 1] + x[:, 2]**2]).astype(np.float32)
    x[rng.random(x.shape) < .15] = np.nan
    d = xgb.DMatrix(x, label=y, weight=rng.integers(0, 5, len(x)).astype(np.float32)*.25)
    p = dict(device="mps", nthread=2, tree_method="hist", multi_strategy=strategy, base_score=0,
             max_depth=4, grow_policy=growth, max_cached_hist_node=1, max_bin=17, reg_alpha=.25)
    if constrained:
        p["monotone_constraints"] = "(1,-1,0,0,0)"
    recorded = RecordedSquaredError()
    before = xgb.metal_info()
    model = xgb.train(p, d, 2, obj=recorded)
    after = xgb.metal_info()
    audit = audit_scalar if strategy == "one_output_per_tree" else audit_vector
    counts = audit(model, d, x, recorded, p)
    assert min(counts) > 0
    assert after["split_finalize_dispatches"] - before["split_finalize_dispatches"] > len(trees(model))
    assert after["split_bounds_dispatches"] - before["split_bounds_dispatches"] == (counts[0] if constrained else 0)
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(d), cpu.predict(d))


@pytest.mark.parametrize("targets", [3, 17, 129])
def test_vector_root_trace_retains_small_targets(metal, targets):
    x = np.full((1, 2), np.nan, dtype=np.float32)
    g = np.zeros((1, targets), dtype=np.float32)
    h = np.ones_like(g)
    h[0, 0] = 2.**24
    d = xgb.DMatrix(x, label=g)
    p = dict(device="mps", tree_method="hist", multi_strategy="multi_output_tree", base_score=0,
             nthread=1, max_depth=1)
    before = xgb.metal_info()
    model = xgb.train(p, d, 1, obj=lambda prediction, data: (g, h))
    tree = trees(model)[0]
    assert audit_tree(tree, x, d.get_quantile_cut(), ["q", "q"], g, h, p) == [0, 1]
    np.testing.assert_array_equal(np.float32(tree["sum_hessian"][0]), np.float32(2.**24 + targets-1))
    after = xgb.metal_info()
    assert after["split_finalize_dispatches"] > before["split_finalize_dispatches"]
    assert after["split_bounds_dispatches"] == before["split_bounds_dispatches"]


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_finalize_many_features(metal, strategy):
    rng = np.random.default_rng(217)
    x = rng.normal(size=(97, 67)).astype(np.float32)
    # The last feature wins across more than two SIMD chunks.
    y = np.column_stack([20*x[:, -1], -20*x[:, -1]]).astype(np.float32)
    d = xgb.DMatrix(x, label=y)
    p = dict(device="mps", tree_method="hist", multi_strategy=strategy, base_score=0,
             nthread=2, max_depth=2, max_bin=9)
    recorded = RecordedSquaredError()
    model = xgb.train(p, d, 1, obj=recorded)
    audit = audit_scalar if strategy == "one_output_per_tree" else audit_vector
    assert min(audit(model, d, x, recorded, p)) > 0
    assert all(tree["split_indices"][0] == x.shape[1]-1 for tree in trees(model))
