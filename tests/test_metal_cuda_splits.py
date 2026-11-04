"""Full-tree scalar CUDA-source checks: CPU tree identity is not the oracle."""
import numpy as np
import pytest
import xgboost as xgb

from cuda_split_reference import RecordedSquaredError, audit_tree, trees
from cuda_uniform_reference import weights


@pytest.mark.parametrize("layout", ["numeric", "onehot", "partition", "pages"])
@pytest.mark.parametrize("constraints", ["none", "monotone", "regularized", "interaction"])
@pytest.mark.parametrize("growth,cache", [("depthwise", 1), ("depthwise", 64),
                                          ("lossguide", 1), ("lossguide", 64)])
def test_complete_cuda_scalar_tree(metal, layout, constraints, growth, cache, tmp_path):
    rng = np.random.default_rng(7861)
    x = rng.normal(size=(257, 5)).astype(np.float32)
    x[:, 0] = rng.integers(0, 7, len(x))
    y = (x[:, 0] + 2 * x[:, 1] - x[:, 2] + x[:, 3]**2).astype(np.float32)
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
                super().__init__(cache_prefix=str(tmp_path / "cuda_split_pages"))

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
    p = dict(tree_method="hist", device="mps", nthread=2, max_bin=17, max_depth=4,
             max_cached_hist_node=cache, grow_policy=growth, base_score=0, learning_rate=.4,
             max_cat_to_onehot=16 if layout == "onehot" else 2, max_cat_threshold=6)
    if constraints == "monotone":
        p["monotone_constraints"] = "(0,1,-1,0,0)"
    elif constraints == "regularized":
        p.update(reg_alpha=.25, reg_lambda=2, max_delta_step=.75, min_child_weight=3)
    elif constraints == "interaction":
        p["interaction_constraints"] = "[[0,1],[2,3,4]]"
    recorded = RecordedSquaredError()
    before = xgb.metal_info()
    model = xgb.train(p, d, 2, obj=recorded)
    after = xgb.metal_info()
    assert after["cuda_integer_split_dispatches"] > before["cuda_integer_split_dispatches"]
    counts = [0, 0]
    for tree, (g, h) in zip(trees(model), recorded.gradients, strict=True):
        result = audit_tree(tree, x, d.get_quantile_cut(), types, g.reshape(-1), h.reshape(-1), p)
        counts = [a + b for a, b in zip(counts, result, strict=True)]
    assert min(counts) > 0
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(d), cpu.predict(d))
    np.testing.assert_array_equal(model.inplace_predict(x), cpu.inplace_predict(x))


@pytest.mark.parametrize("method", ["hist", "approx"])
@pytest.mark.parametrize("kind", ["numeric", "onehot", "partition"])
@pytest.mark.parametrize("sampling", [False, True])
def test_sampled_scalar_cuda_tree(metal, method, kind, sampling):
    rng = np.random.default_rng(832)
    x = rng.integers(0, 6, (193, 3)).astype(np.float32)
    y = (3*x[:,0] - x[:,1] + x[:,2]**2).astype(np.float32)
    x[rng.random(x.shape) < .3] = np.nan
    types = ["q"] * 3
    if kind != "numeric":
        types[0] = "c"
    d = xgb.DMatrix(x, label=y, feature_types=types,
                    weight=rng.integers(0, 5, len(x)).astype(np.float32) * .25)
    p = dict(device="mps", tree_method=method, base_score=0, learning_rate=.25, nthread=2,
             max_depth=4, max_bin=16, subsample=.63 if sampling else 1., seed=29,
             seed_per_iteration=True, max_cat_to_onehot=16 if kind == "onehot" else 2)
    recorded = RecordedSquaredError()
    model = xgb.train(p, d, 2, obj=recorded)
    for iteration, (tree, (g,h)) in enumerate(zip(trees(model),recorded.gradients,strict=True)):
        keep = None
        if sampling:
            rng = np.random.RandomState(29 * 127 + iteration)
            seed = int(rng.randint(0, 1 << 32, dtype=np.uint32))
            keep = weights(seed, len(x)) <= np.float32(.63)
        result = audit_tree(tree,x,d.get_quantile_cut(),types,g.reshape(-1),h.reshape(-1),p,keep)
        assert min(result) > 0


@pytest.mark.parametrize("kind", ["numeric", "onehot", "partition"])
def test_cuda_missing_direction_tie(metal, kind):
    # No missing rows: CUDA's scalar loss comparison ties and chooses RIGHT.
    # CPU one-hot/partition enumeration can select a different default direction.
    x = np.repeat(np.arange(4, dtype=np.float32), 4).reshape(-1, 1)
    y = np.repeat(np.array([-2,-2,2,2],dtype=np.float32),4)
    types = ["q" if kind == "numeric" else "c"]
    d = xgb.DMatrix(x,label=y,feature_types=types)
    p = dict(device="mps",tree_method="hist",base_score=0,nthread=1,max_depth=1,
             max_cat_to_onehot=16 if kind == "onehot" else 2)
    recorded = RecordedSquaredError()
    model=xgb.train(p,d,1,obj=recorded)
    g,h=recorded.gradients[0]
    audit_tree(trees(model)[0],x,d.get_quantile_cut(),types,g.reshape(-1),h.reshape(-1),p)
    if kind != "partition":
        assert trees(model)[0]["default_left"][0] == 0


@pytest.mark.parametrize("exponent", [0, 54, 60])
@pytest.mark.parametrize("external", [False, True])
def test_integer_children_survive_cancellation(metal, exponent, external, tmp_path):
    # Each root child rounds to +/-big in binary64. Its next split must recover
    # the small sibling through INTEGER parent-minus-child, including across pages.
    big, small = np.float32(2.**exponent), np.float32(2.**(exponent-54))
    x = np.array([[0,0],[0,1],[0,1],[0,2],[1,0],[1,1],[1,1],[1,2]], dtype=np.float32)
    grad = np.array([big,small,small,small,-big,-small,-small,-small],dtype=np.float32)
    hess = np.ones_like(grad)
    if external:
        class Pages(xgb.DataIter):
            def __init__(self):
                self.row=0
                super().__init__(cache_prefix=str(tmp_path / "integer_children"))

            def reset(self):
                self.row=0

            def next(self,input_data):
                if self.row == len(x):
                    return False
                end=min(self.row+3,len(x))
                input_data(data=x[self.row:end],label=np.zeros(end-self.row,dtype=np.float32))
                self.row=end
                return True
        d=xgb.ExtMemQuantileDMatrix(Pages(),max_bin=4)
    else:
        d=xgb.DMatrix(x,label=np.zeros(len(x)))
    p=dict(device="mps",tree_method="hist",base_score=0,learning_rate=1.,nthread=1,
           max_depth=2,max_bin=4,max_cached_hist_node=1)
    model=xgb.train(p,d,1,obj=lambda prediction,data:(grad,hess))
    result=audit_tree(trees(model)[0],x,d.get_quantile_cut(),["q","q"],grad,hess,p)
    assert result == (3,4)
    expected=np.array([-big/2,-3*small/4,-3*small/4,-3*small/4,
                       big/2,3*small/4,3*small/4,3*small/4],dtype=np.float32)
    np.testing.assert_array_equal(model.predict(d),expected)
    np.testing.assert_array_equal(model.inplace_predict(x),expected)
    cpu=xgb.Booster(params={"device":"cpu"},model_file=model.save_raw())
    np.testing.assert_array_equal(cpu.predict(d),expected)
