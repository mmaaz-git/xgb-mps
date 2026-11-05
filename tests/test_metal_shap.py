"""Same-model CPU comparisons plus explicit CUDA-reference deep-tree cases."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xgboost as xgb


def metal_copy(model):
    gpu = xgb.Booster(model_file=model.save_raw())
    gpu.set_param({"device": "mps", "nthread": 2})
    return gpu


def compare_shap(cpu, data, interactions, **options):
    gpu = metal_copy(cpu)
    args = {"pred_interactions" if interactions else "pred_contribs": True, "strict_shape": True}
    expected = cpu.predict(data, **args, **options)
    before = xgb.metal_info()
    actual = gpu.predict(data, **args, **options)
    after = xgb.metal_info()
    counter = "shap_interaction_dispatches" if interactions else "shap_dispatches"
    assert (after[counter] > before[counter]) == (data.num_row() > 0)
    if data.num_row():
        assert after["shap_gpu_time_ns"] > before["shap_gpu_time_ns"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    if interactions:
        np.testing.assert_array_equal(actual, actual.swapaxes(-1, -2))
        # Compare to CPU additive values: the upstream recurrence itself has
        # roundoff differences between additive and interaction accumulation.
        additive = cpu.predict(data, pred_contribs=True, strict_shape=True, **options)
        np.testing.assert_allclose(actual.sum(axis=-1), additive, rtol=2e-5, atol=2e-5)
    return actual


@pytest.mark.parametrize("layout", ["dense", "missing", "sparse", "quantile"])
@pytest.mark.parametrize("kind", ["regression", "multiclass", "vector", "dart", "forest"])
@pytest.mark.parametrize("interactions", [False, True])
def test_shap_models(metal, layout, kind, interactions):
    rng = np.random.default_rng(456)
    x = rng.normal(size=(193, 7)).astype(np.float32)
    y = x[:, 0] * x[:, 1] - x[:, 2] ** 2 + x[:, 3]
    if layout in {"missing", "quantile"}:
        x[rng.random(x.shape) < .23] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < .7] = 0
        x = sp.csr_matrix(x)
    params = dict(nthread=2, max_depth=4, max_bin=32, base_score=.4, seed=45)
    groups = 1
    if kind == "multiclass":
        y = np.digitize(y, [-1, 1])
        groups = 3
        params.update(objective="multi:softprob", num_class=groups)
    elif kind == "vector":
        y = np.column_stack([y, -y, y / 2])
        groups = 3
        params.update(multi_strategy="multi_output_tree")
    elif kind == "dart":
        params.update(booster="dart", rate_drop=.4, skip_drop=0)
    elif kind == "forest":
        params.update(num_parallel_tree=3, colsample_bynode=.7, subsample=.7)
    data = (xgb.QuantileDMatrix(x, label=y, max_bin=32, nthread=2) if layout == "quantile"
            else xgb.DMatrix(x, label=y, nthread=2))
    cpu = xgb.train(params, data, 6)
    compare_shap(cpu, data, interactions)
    data.set_base_margin(rng.normal(size=(len(y), groups)).astype(np.float32))
    compare_shap(cpu, data, interactions, iteration_range=(0, 3))


@pytest.mark.parametrize("interactions", [False, True])
@pytest.mark.parametrize("onehot", [1, 64])
def test_shap_categorical_recoding(metal, interactions, onehot):
    rng = np.random.default_rng(3)
    categories = [f"cat-{i}" for i in range(41)]
    x = pd.DataFrame({"cat": pd.Categorical(rng.choice(categories, 257), categories=categories),
                      "num": rng.normal(size=257)})
    y = (x.cat.cat.codes.to_numpy() % 7) * x.num.to_numpy()
    cpu = xgb.train(dict(nthread=2, max_depth=5, max_cat_to_onehot=onehot),
                    xgb.DMatrix(x, label=y, enable_categorical=True, nthread=2), 6)
    x["cat"] = x.cat.cat.reorder_categories(categories[::-1])
    x.loc[::13, "cat"] = np.nan
    compare_shap(cpu, xgb.DMatrix(x, enable_categorical=True, nthread=2), interactions)


@pytest.mark.parametrize("rows", [0, 1, 4097])
@pytest.mark.parametrize("rounds", [0, 1])
@pytest.mark.parametrize("interactions", [False, True])
def test_shap_empty_stump_and_batches(metal, rows, rounds, interactions):
    train = xgb.DMatrix(np.ones((17, 3)), label=np.ones(17), nthread=2)
    cpu = xgb.train(dict(nthread=2, base_score=.25, min_child_weight=100), train, rounds)
    margin = np.linspace(-1, 1, rows, dtype=np.float32)
    data = xgb.DMatrix(np.ones((rows, 3)), base_margin=margin, nthread=2)
    actual = compare_shap(cpu, data, interactions)
    if rows:
        expected = cpu.predict(data, output_margin=True, strict_shape=True)
        total = actual.sum(axis=(-1, -2) if interactions else -1)
        np.testing.assert_array_equal(total, expected)


def handcrafted_tree(depth=2, zero_cover=False):
    """A chain revisits three features; each right child is a leaf."""
    d = xgb.DMatrix(np.ones((4, 3)), label=np.ones(4), nthread=2)
    model = xgb.train(dict(nthread=2, base_score=.2, min_child_weight=100), d, 1)
    raw = json.loads(model.save_raw(raw_format="json"))
    tree = raw["learner"]["gradient_booster"]["model"]["trees"][0]
    n = 2 * depth + 1
    left = [-1] * n
    right = [-1] * n
    parents = [2147483647] + [0] * (n - 1)
    split = [0] * n
    values = [.25] * n
    covers = [0. if zero_cover else 1.] * n
    for i in range(depth):
        at, l, r = 2 * i, 2 * i + 1, 2 * i + 2
        # Continue on the right, keeping construction independent of traversal.
        left[at], right[at] = l, r
        parents[l] = parents[r] = at
        split[at] = i % 3
        values[at] = (i % 5 - 2) / 3
        values[l] = (i % 7 - 3) / 4
        covers[at] = 0. if zero_cover else float(depth + 1 - i)
    tree.update(left_children=left, right_children=right, parents=parents,
                split_indices=split, split_conditions=values, sum_hessian=covers,
                default_left=[i % 2 for i in range(n)], split_type=[0] * n,
                base_weights=[0.] * n, loss_changes=[0.] * n)
    tree["tree_param"]["num_nodes"] = str(n)
    model = xgb.Booster(model_file=bytearray(json.dumps(raw).encode()))
    model.set_param({"device": "cpu", "nthread": 2})
    return model, tree


@pytest.mark.parametrize("depth", [2, 17, 65, 127, 130])
@pytest.mark.parametrize("zero_cover", [False, True])
@pytest.mark.parametrize("interactions", [False, True])
def test_shap_repeated_features_and_depth(metal, depth, zero_cover, interactions):
    cpu, _ = handcrafted_tree(depth, zero_cover)
    x = np.array([[-1, -1, -1], [1, 1, 1], [np.nan] * 3, [.5, -1, np.nan]], dtype=np.float32)
    if zero_cover and depth >= 127:
        from cuda_shap_reference import predict as source_predict
        data = xgb.DMatrix(x, nthread=2)
        args = {"pred_interactions" if interactions else "pred_contribs": True, "strict_shape": True}
        cpu_result = cpu.predict(data, **args)
        assert np.isnan(cpu_result).any()  # Separate CPU basis overflows here.
        expected = source_predict(json.loads(cpu.save_raw(raw_format="json")), x, interactions)
        before = xgb.metal_info()
        actual = metal_copy(cpu).predict(data, **args)
        key = "shap_interaction_dispatches" if interactions else "shap_dispatches"
        assert xgb.metal_info()[key] > before[key]
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        finite = np.isfinite(cpu_result)
        np.testing.assert_allclose(actual[finite], cpu_result[finite], rtol=2e-5, atol=2e-5)
    else:
        compare_shap(cpu, xgb.DMatrix(x, nthread=2), interactions)


@pytest.mark.parametrize("interactions", [False, True])
def test_shap_independent_coalitions(metal, interactions):
    cpu, tree = handcrafted_tree(2)
    x = np.array([[-1, -1, 0], [1, 1, 1], [np.nan, .5, 0]], dtype=np.float32)
    data = xgb.DMatrix(x, nthread=2)
    actual = compare_shap(cpu, data, interactions)
    n = 3

    def conditional(row, subset, node=0):
        left, right = tree["left_children"][node], tree["right_children"][node]
        if left < 0:
            return tree["split_conditions"][node]
        feature = tree["split_indices"][node]
        if subset & (1 << feature):
            value = row[feature]
            go_left = bool(tree["default_left"][node]) if np.isnan(value) else value < tree["split_conditions"][node]
            return conditional(row, subset, left if go_left else right)
        cover = tree["sum_hessian"]
        return (cover[left] * conditional(row, subset, left) +
                cover[right] * conditional(row, subset, right)) / cover[node]

    for r, row in enumerate(x):
        v = [conditional(row, subset) for subset in range(1 << n)]
        phi = np.zeros(n + 1)
        phi[-1] = v[0] + .2
        for i in range(n):
            for subset in range(1 << n):
                if subset & (1 << i):
                    continue
                k = subset.bit_count()
                w = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
                phi[i] += w * (v[subset | (1 << i)] - v[subset])
        if not interactions:
            np.testing.assert_allclose(actual[r, 0], phi, atol=2e-6, rtol=2e-6)
            continue
        matrix = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for subset in range(1 << n):
                    if subset & ((1 << i) | (1 << j)):
                        continue
                    k = subset.bit_count()
                    w = math.factorial(k) * math.factorial(n-k-2) / (2 * math.factorial(n-1))
                    matrix[i, j] += w * (v[subset | (1 << i) | (1 << j)] -
                                          v[subset | (1 << i)] - v[subset | (1 << j)] + v[subset])
        for i in range(n+1):
            matrix[i, i] = phi[i] - matrix[i].sum()
        np.testing.assert_allclose(actual[r, 0], matrix, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("interactions", [False, True])
def test_shap_external_pages(metal, tmp_path, interactions):
    rng = np.random.default_rng(85)
    x = rng.normal(size=(4124, 5)).astype(np.float32)
    y = x[:, 0] * x[:, 1] + x[:, 2]
    x[rng.random(x.shape) < .3] = np.nan
    margins = rng.normal(size=len(x)).astype(np.float32)

    class Pages(xgb.DataIter):
        def __init__(self):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / "shap-pages"))

        def reset(self):
            self.page = 0

        def next(self, input_data):
            if self.page == 4:
                return False
            begin = 1031 * self.page
            end = begin + 1031
            input_data(data=x[begin:end], label=y[begin:end], base_margin=margins[begin:end])
            self.page += 1
            return True

    data = xgb.DMatrix(Pages(), nthread=2)
    cpu = xgb.train(dict(nthread=2, max_depth=3, max_bin=32), data, 3)
    before = xgb.metal_info()
    compare_shap(cpu, data, interactions)
    key = "shap_interaction_dispatches" if interactions else "shap_dispatches"
    assert xgb.metal_info()[key] - before[key] >= 4


def test_shap_wide_interaction_batches(metal):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(257, 257)).astype(np.float32)
    y = x[:, 0] * x[:, 1]
    data = xgb.DMatrix(x, label=y, nthread=2)
    cpu = xgb.train(dict(nthread=2, max_depth=2), data, 2)
    before = xgb.metal_info()["shap_interaction_dispatches"]
    compare_shap(cpu, data, True)
    assert xgb.metal_info()["shap_interaction_dispatches"] - before > 1


@pytest.mark.parametrize("interactions", [False, True])
@pytest.mark.parametrize("cover", [-1., 0., 1e-35])
def test_shap_cover_boundaries(metal, cover, interactions):
    cpu, _ = handcrafted_tree(2)
    raw = json.loads(cpu.save_raw(raw_format="json"))
    tree = raw["learner"]["gradient_booster"]["model"]["trees"][0]
    tree["sum_hessian"][1] = cover
    cpu = xgb.Booster(model_file=bytearray(json.dumps(raw).encode()))
    cpu.set_param({"device": "cpu", "nthread": 2})
    data = xgb.DMatrix(np.array([[-1, 0, 0], [1, 0, 0], [np.nan, np.nan, 0]]), nthread=2)
    if cover < 0:
        args = {"pred_interactions" if interactions else "pred_contribs": True}
        for model in [cpu, metal_copy(cpu)]:
            with pytest.raises(xgb.core.XGBoostError, match="negative child cover"):
                model.predict(data, **args)
    else:
        compare_shap(cpu, data, interactions)
