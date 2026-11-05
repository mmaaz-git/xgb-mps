"""Full-tree checks of CUDA's vector split rules using exact integer references."""
import numpy as np
import pytest
import xgboost as xgb

from cuda_split_reference import RecordedSquaredError, trees, quantize
from cuda_multi_split_reference import audit_tree, audit_recorded


@pytest.mark.parametrize("layout", ["numeric", "onehot", "partition", "pages"])
@pytest.mark.parametrize("constraints", ["none", "monotone", "regularized", "interaction"])
@pytest.mark.parametrize("growth,cache", [("depthwise", 1), ("depthwise", 64),
                                          ("lossguide", 1), ("lossguide", 64)])
def test_complete_cuda_vector_tree(metal, layout, constraints, growth, cache, tmp_path):
    rng = np.random.default_rng(7861)
    x = rng.normal(size=(257, 5)).astype(np.float32)
    x[:, 0] = rng.integers(0, 7, len(x))
    y = np.column_stack([x[:, 0] + 2*x[:, 1] - x[:, 2],
                         -x[:, 0] + x[:, 3]**2, x[:, 1] + x[:, 2]]).astype(np.float32)
    x[rng.random(x.shape) < .2] = np.nan
    x[:, -1] = np.nan
    types = ["q"] * 5
    if layout in ["onehot", "partition"]:
        types[0] = "c"
    row_weight = rng.integers(0, 5, len(x)).astype(np.float32) * .25
    if layout == "pages":
        class Pages(xgb.DataIter):
            def __init__(self):
                self.row = 0
                super().__init__(cache_prefix=str(tmp_path / "cuda_vector_pages"))

            def reset(self):
                self.row = 0

            def next(self, input_data):
                if self.row == len(x):
                    return False
                end = min(self.row + 43, len(x))
                input_data(data=x[self.row:end], label=y[self.row:end],
                           weight=row_weight[self.row:end], feature_types=types)
                self.row = end
                return True
        d = xgb.ExtMemQuantileDMatrix(Pages(), max_bin=17)
    else:
        d = xgb.DMatrix(x, label=y, weight=row_weight, feature_types=types)
    p = dict(tree_method="hist", device="mps", multi_strategy="multi_output_tree", nthread=2,
             max_bin=17, max_depth=4, max_cached_hist_node=cache, grow_policy=growth,
             base_score=0, learning_rate=.4, max_cat_to_onehot=16 if layout == "onehot" else 2,
             max_cat_threshold=6)
    if constraints == "monotone":
        p["monotone_constraints"] = "(0,1,-1,0,0)"
    elif constraints == "regularized":
        p.update(reg_alpha=.25, reg_lambda=2, max_delta_step=.75, min_child_weight=3)
    elif constraints == "interaction":
        p["interaction_constraints"] = "[[0,1],[2,3,4]]"
    recorded = RecordedSquaredError()
    before = xgb.metal_info()["cuda_integer_multi_split_dispatches"]
    model = xgb.train(p, d, 2, obj=recorded)
    assert xgb.metal_info()["cuda_integer_multi_split_dispatches"] > before
    assert min(audit_recorded(model, d, x, recorded, p)) > 0
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(d), cpu.predict(d))
    np.testing.assert_array_equal(model.inplace_predict(x), cpu.inplace_predict(x))


@pytest.mark.parametrize("sampling,fraction", [("uniform", 0.), ("uniform", .63),
                                               ("gradient_based", .63)])
@pytest.mark.parametrize("kind", ["numeric", "onehot", "partition"])
def test_sampled_cuda_vector_tree(metal, sampling, fraction, kind):
    rng = np.random.default_rng(832)
    x = rng.integers(0, 6, (193, 3)).astype(np.float32)
    y = np.column_stack([3*x[:, 0] - x[:, 1], x[:, 2]**2 - x[:, 0]]).astype(np.float32)
    x[rng.random(x.shape) < .3] = np.nan
    types = ["q"] * 3
    if kind != "numeric":
        types[0] = "c"
    d = xgb.DMatrix(x, label=y, feature_types=types,
                    weight=rng.integers(0, 5, len(x)).astype(np.float32) * .25)
    p = dict(device="mps", multi_strategy="multi_output_tree", tree_method="hist",
             base_score=0, learning_rate=.25, nthread=2, max_depth=4, max_bin=16,
             subsample=fraction, sampling_method=sampling, seed=29, seed_per_iteration=True,
             max_cat_to_onehot=16 if kind == "onehot" else 2)
    recorded = RecordedSquaredError()
    model = xgb.train(p, d, 2, obj=recorded)
    counts = audit_recorded(model, d, x, recorded, p)
    assert counts[1] > 0
    assert counts[0] > 0 if fraction else counts[0] == 0


@pytest.mark.parametrize("targets", [2, 3, 17])
@pytest.mark.parametrize("monotone", [False, True])
def test_vector_zero_hessian_target(metal, targets, monotone):
    rng = np.random.default_rng(43)
    x = rng.normal(size=(67, 2)).astype(np.float32)
    grad = rng.normal(size=(len(x), targets)).astype(np.float32)
    hess = rng.integers(0, 4, grad.shape).astype(np.float32)
    hess[:, 0] = 0
    d = xgb.DMatrix(x, label=np.zeros_like(grad))
    p = dict(device="mps", multi_strategy="multi_output_tree", tree_method="hist", base_score=0,
             nthread=2, max_depth=3, max_bin=17, min_child_weight=.25, reg_alpha=.3)
    if monotone:
        p["monotone_constraints"] = "(1,-1)"
    model = xgb.train(p, d, 1, obj=lambda prediction, data: (grad, hess))
    assert min(audit_tree(trees(model)[0], x, d.get_quantile_cut(), ["q", "q"], grad, hess, p)) > 0


@pytest.mark.parametrize("exponent", [0, 54, 60])
@pytest.mark.parametrize("growth", ["depthwise", "lossguide"])
def test_vector_integer_children_survive_cancellation(metal, exponent, growth):
    big, small = np.float32(2.**exponent), np.float32(2.**(exponent-54))
    x = np.array([[0,0],[0,1],[0,1],[0,2],[1,0],[1,1],[1,1],[1,2]], dtype=np.float32)
    base = np.array([big,small,small,small,-big,-small,-small,-small], dtype=np.float32)
    grad = np.column_stack([base, -.5*base, 2*base]).astype(np.float32)
    hess = np.ones_like(grad)
    d = xgb.DMatrix(x, label=np.zeros_like(grad))
    p = dict(device="mps", multi_strategy="multi_output_tree", tree_method="hist", base_score=0,
             learning_rate=1., nthread=1, max_depth=2, max_bin=4, max_cached_hist_node=1,
             grow_policy=growth)
    model = xgb.train(p, d, 1, obj=lambda prediction, data: (grad, hess))
    assert audit_tree(trees(model)[0], x, d.get_quantile_cut(), ["q", "q"], grad, hess, p) == [3, 4]
    expected = np.array([-big/2,-3*small/4,-3*small/4,-3*small/4,
                          big/2,3*small/4,3*small/4,3*small/4], dtype=np.float32)
    expected = np.column_stack([expected, -.5*expected, 2*expected]).astype(np.float32)
    np.testing.assert_array_equal(model.predict(d), expected)
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(cpu.predict(d), expected)


@pytest.mark.parametrize("kind", ["numeric", "onehot", "partition"])
def test_vector_feature_tie_and_leaf_limit(metal, kind):
    column = np.repeat(np.arange(7, dtype=np.float32), 5)
    x = np.column_stack([column, column, column])
    y = np.column_stack([column - 3, -(column - 3)**2]).astype(np.float32)
    types = ["q" if kind == "numeric" else "c"] * 3
    d = xgb.DMatrix(x, label=y, feature_types=types)
    p = dict(device="mps", tree_method="hist", multi_strategy="multi_output_tree", base_score=0,
             nthread=2, max_depth=0, max_leaves=4, grow_policy="lossguide", max_cached_hist_node=1,
             max_cat_to_onehot=16 if kind == "onehot" else 2)
    recorded = RecordedSquaredError()
    model = xgb.train(p, d, 1, obj=recorded)
    assert audit_recorded(model, d, x, recorded, p) == [3, 4]
    tree = trees(model)[0]
    assert all(f == 0 for f, left in zip(tree["split_indices"], tree["left_children"], strict=True) if left >= 0)


@pytest.mark.parametrize("split_targets", [1, 2])
@pytest.mark.parametrize("sampling,fraction", [("uniform", 1.), ("uniform", .63),
                                               ("gradient_based", .63)])
def test_reduced_vector_tree_integer_state(metal, split_targets, sampling, fraction):
    import math
    from cuda_uniform_reference import weights
    from cuda_gradient_reference import sample

    class Reduced(RecordedSquaredError):
        def __init__(self):
            super().__init__()
            self.splits = []

        def split_grad(self, iteration, grad, hess):
            g, h = np.mean(grad, axis=1), np.mean(hess, axis=1)
            if split_targets == 2:
                g, h = np.column_stack([g, grad[:, 0]]), np.column_stack([h, hess[:, 0]])
            self.splits.append((g.copy(), h.copy()))
            return g, h

    rng = np.random.default_rng(719)
    x = rng.normal(size=(193, 4)).astype(np.float32)
    y = np.column_stack([x[:, 0] - x[:, 1], x[:, 2]**2, 2*x[:, 3]]).astype(np.float32)
    x[rng.random(x.shape) < .2] = np.nan
    d = xgb.DMatrix(x, label=y, weight=rng.integers(0, 5, len(x)).astype(np.float32)*.25)
    p = dict(device="mps", tree_method="hist", multi_strategy="multi_output_tree", base_score=0,
             nthread=2, max_depth=4, max_bin=17, grow_policy="lossguide", max_cached_hist_node=1,
             reg_alpha=.25, reg_lambda=2, subsample=fraction, sampling_method=sampling,
             seed=29, seed_per_iteration=True)
    recorded = Reduced()
    model = xgb.train(p, d, 2, obj=recorded)
    for iteration, (tree, (vg, vh), (g, h)) in enumerate(zip(
            trees(model), recorded.gradients, recorded.splits, strict=True)):
        keep, integers = None, None
        factors = np.ones(len(x), dtype=np.float32)
        if fraction < 1:
            rng = np.random.RandomState(29 * 127 + iteration)
            row_seed = int(rng.randint(0, 1 << 32, dtype=np.uint32))
            if sampling == "uniform":
                keep = weights(row_seed, len(x)) <= np.float32(fraction)
                factors = keep.astype(np.float32)
            else:
                _, factors, _, (qg, qh, ge, he) = sample(g, h, fraction, row_seed, return_quantized=True)
                integers = ([[(int(qg[r,t]), int(qh[r,t])) for t in range(split_targets)]
                             for r in range(len(x))], list(zip(map(int, ge), map(int, he), strict=True)))
        # Value leaf scales come from ORIGINAL value gradients; replay rounds
        # each sampled value to float before requantizing with those scales.
        scales = [quantize(vg[:, t], vh[:, t])[1] for t in range(vg.shape[1])]
        qvalues = []
        for r in range(len(x)):
            row = []
            for t, (ge, he) in enumerate(scales):
                a, b = np.float32(vg[r,t] * factors[r]), np.float32(vh[r,t] * factors[r])
                qg, qh = int(math.ldexp(float(a), ge)), int(math.ldexp(float(b), he))
                row.append((qg, max(1, qh) if b > 0 else qh))
            qvalues.append(row)
        assert min(audit_tree(tree, x, d.get_quantile_cut(), ["q"]*4, g, h, p, keep,
                              integers=integers, value_integers=(qvalues, scales))) > 0
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(d), cpu.predict(d))


@pytest.mark.parametrize("targets", [2, 17])
@pytest.mark.parametrize("reason", ["no_features", "no_gain", "leaf_limit"])
def test_vector_root_only_sum_residency(metal, targets, reason):
    x = np.arange(64, dtype=np.float32).reshape(32, 2)
    if reason == "no_features":
        x[:] = np.nan
    g = np.tile(np.arange(targets, dtype=np.float32) + 1, (len(x), 1))
    h = np.ones_like(g)
    if reason == "leaf_limit":
        g[16:] *= -3
    d = xgb.DMatrix(x, label=np.zeros_like(g))
    params = dict(device="mps", tree_method="hist", multi_strategy="multi_output_tree",
                  base_score=0, learning_rate=1, max_depth=3, nthread=1)
    if reason == "leaf_limit":
        params.update(max_leaves=1, grow_policy="lossguide")
    before = xgb.metal_info()
    model = xgb.train(params, d, 1, obj=lambda pred, data: (g, h))
    after = xgb.metal_info()
    for key in ("gradient_sum_readback_bytes", "split_sum_readback_bytes", "split_sum_host_input_bytes"):
        assert after[key] == before[key], key
    assert after["split_sum_gather_dispatches"] > before["split_sum_gather_dispatches"]
    assert after["split_sum_apply_dispatches"] == before["split_sum_apply_dispatches"]
    assert len(trees(model)[0]["left_children"]) == 1
    expected = np.tile(-g.astype(np.float64).sum(axis=0) / (len(x) + 1), (len(x), 1)).astype(np.float32)
    np.testing.assert_array_equal(model.predict(d), expected)
