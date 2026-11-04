"""Approx integration and CUDA updater ordering, without a CUDA execution claim.

CPU model comparisons are limited to common flow (no row/column sampling).
The current Metal sketch/sampler implementations still differ from CUDA's.
"""
import json

import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb


def matrices(layout, kind, tmp_path):
    rng = np.random.default_rng(845)
    x = rng.normal(size=(521, 7)).astype(np.float32)
    y = x[:, 0] * 2 + x[:, 1] ** 2 - x[:, 2]
    params = dict(objective="reg:squarederror", base_score=0.)
    groups = 1
    if kind == "binary":
        y = (y > 0).astype(np.float32)
        params.update(objective="binary:logistic", base_score=.5)
    elif kind == "multiclass":
        y = np.digitize(y, [-1, 1])
        params.update(objective="multi:softprob", num_class=3, base_score=[.5] * 3)
        groups = 3
    elif kind == "multitarget":
        y = np.column_stack([y, -y, y / 2])
        params.update(base_score=[0.] * 3)
        groups = 3
    types = ["q"] * x.shape[1]
    if layout == "categorical":
        x[:, 4] = rng.integers(0, 6, len(x))
        types[4] = "c"
    if layout in {"missing", "external"}:
        x[rng.random(x.shape) < .25] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < .65] = 0
        x = sp.csr_matrix(x)
    weight = rng.uniform(.1, 3, len(y)).astype(np.float32)
    weight[::19] = 0

    class Pages(xgb.DataIter):
        def __init__(self, device):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / device))

        def reset(self):
            self.page = 0

        def next(self, input_data):
            if self.page == 4:
                return False
            begin, end = self.page * 131, min((self.page + 1) * 131, len(x))
            input_data(data=x[begin:end], label=y[begin:end], weight=weight[begin:end])
            self.page += 1
            return True

    data = {}
    for device in ["cpu", "mps"]:
        if layout == "external":
            data[device] = xgb.DMatrix(Pages(device), nthread=2)
        else:
            data[device] = xgb.DMatrix(x, label=y, weight=weight, feature_types=types, nthread=2)
    return data, params, groups


def compare(cpu, gpu, data):
    for args in [{}, {"output_margin": True}]:
        np.testing.assert_allclose(gpu.predict(data["mps"], **args), cpu.predict(data["cpu"], **args),
                                   rtol=2e-5, atol=2e-5)
    for actual, expected in zip(data["mps"].get_quantile_cut(), data["cpu"].get_quantile_cut()):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("layout", ["dense", "missing", "sparse", "categorical", "external"])
@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass", "multitarget"])
@pytest.mark.parametrize("subsample,sampling", [(1., "uniform"), (.7, "gradient_based")])
@pytest.mark.parametrize("growth", ["depthwise", "lossguide"])
def test_approx_training(metal, layout, kind, subsample, sampling, growth, tmp_path):
    data, objective, groups = matrices(layout, kind, tmp_path)
    params = dict(tree_method="approx", max_bin=17, nthread=2, eta=.3, seed=37,
                  seed_per_iteration=True, subsample=subsample, sampling_method=sampling,
                  grow_policy=growth, max_depth=3 if growth == "depthwise" else 0,
                  max_leaves=0 if growth == "depthwise" else 7, reg_lambda=2., **objective)
    before_cpu = xgb.metal_info()
    cpu = xgb.train(params | {"device": "cpu"}, data["cpu"], 3) if subsample == 1 else None
    assert xgb.metal_info()["hessian_extract_dispatches"] == before_cpu["hessian_extract_dispatches"]
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data["mps"], 3)
    after = xgb.metal_info()
    assert after["gradient_copy_dispatches"] - before["gradient_copy_dispatches"] >= 3 * groups
    assert after["hessian_extract_dispatches"] - before["hessian_extract_dispatches"] == 3 * groups
    for key in ["gradient_sum_dispatches", "sketch_weight_dispatches", "binning_dispatches",
                "sketch_update_dispatches", "histogram_dispatches", "split_dispatches",
                "partition_dispatches", "leaf_partition_dispatches", "gradient_state_gpu_time_ns"]:
        assert after[key] > before[key], key
    # Upstream selects sorted sketches only when Hessians require regeneration.
    if kind in {"binary", "multiclass"}:
        assert after["sketch_sorted_dispatches"] > before["sketch_sorted_dispatches"]
    else:
        assert after["sketch_sorted_dispatches"] == before["sketch_sorted_dispatches"]
    assert json.loads(gpu.save_config())["learner"]["gradient_booster"]["gbtree_train_param"]["tree_method"] == "approx"
    if cpu is not None:
        compare(cpu, gpu, data)
        cpu = xgb.train(params | {"device": "cpu"}, data["cpu"], 2, xgb_model=cpu.save_raw())
    saved = gpu.save_raw()
    reloaded = xgb.Booster(model_file=saved)
    reloaded.set_param({"device": "mps"})
    np.testing.assert_array_equal(reloaded.predict(data["mps"]), gpu.predict(data["mps"]))
    gpu = xgb.train(params | {"device": "mps"}, data["mps"], 2, xgb_model=saved)
    repeat = xgb.train(params | {"device": "mps"}, data["mps"], 2, xgb_model=saved)
    np.testing.assert_array_equal(repeat.predict(data["mps"]), gpu.predict(data["mps"]))
    assert np.isfinite(gpu.predict(data["mps"])).all()
    if cpu is not None:
        compare(cpu, gpu, data)


@pytest.mark.parametrize("kind", ["regression", "binary"])
@pytest.mark.parametrize("options", [dict(), dict(monotone_constraints="(1,0,0,0,0,0,0)"),
    dict(interaction_constraints="[[0,1],[2,3,4,5,6]]"),
    dict(reg_alpha=.3, reg_lambda=3., max_delta_step=.25), dict(num_parallel_tree=3),
    dict(colsample_bytree=.8, colsample_bylevel=.8, colsample_bynode=.8)])
def test_approx_constraints_and_forests(metal, kind, options, tmp_path):
    data, objective, _ = matrices("missing", kind, tmp_path)
    params = dict(tree_method="approx", max_bin=33, nthread=2, max_depth=3,
                  seed=37, seed_per_iteration=True, **objective, **options)
    gpu = xgb.train(params | {"device": "mps"}, data["mps"], 3)
    if "colsample_bytree" not in options:
        cpu = xgb.train(params | {"device": "cpu"}, data["cpu"], 3)
        compare(cpu, gpu, data)
    else:
        # CUDA resets feature sampling after row sampling for each tree; CPU
        # approx initializes its sampler before row sampling and shares forests.
        repeat = xgb.train(params | {"device": "mps"}, data["mps"], 3)
        np.testing.assert_array_equal(repeat.predict(data["mps"]), gpu.predict(data["mps"]))


@pytest.mark.parametrize("kind,expected", [("regression", 1), ("binary", 4)])
def test_approx_resketch_policy(metal, kind, expected, tmp_path):
    data, objective, _ = matrices("dense", kind, tmp_path)
    before = xgb.metal_info()
    xgb.train(dict(tree_method="approx", device="mps", nthread=2, max_depth=2,
                   max_bin=17, **objective), data["mps"], 4)
    after = xgb.metal_info()
    assert after["sketch_finalize_dispatches"] - before["sketch_finalize_dispatches"] == expected
    assert after["sketch_sorted_dispatches"] - before["sketch_sorted_dispatches"] == (0 if kind == "regression" else expected)
    assert after["hessian_extract_dispatches"] - before["hessian_extract_dispatches"] == 4


@pytest.mark.parametrize("rows", [0, 1, 257])
@pytest.mark.parametrize("sampling", ["uniform", "gradient_based"])
@pytest.mark.parametrize("subsample", [0., 1.])
def test_approx_empty_and_zero_sampling(metal, rows, sampling, subsample):
    x = sp.csr_matrix((rows, 5), dtype=np.float32)
    data = {d: xgb.DMatrix(x, label=np.ones(rows), nthread=1) for d in ["cpu", "mps"]}
    params = dict(tree_method="approx", nthread=1, base_score=.25, max_depth=2,
                  subsample=subsample, sampling_method=sampling)
    cpu = xgb.train(params | {"device": "cpu"}, data["cpu"], 3)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data["mps"], 3)
    after = xgb.metal_info()
    assert after["gradient_copy_dispatches"] - before["gradient_copy_dispatches"] == 3 * bool(rows)
    assert after["hessian_extract_dispatches"] - before["hessian_extract_dispatches"] == 3 * bool(rows)
    np.testing.assert_allclose(gpu.predict(data["mps"]), cpu.predict(data["cpu"]), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("options", [dict(multi_strategy="one_output_per_tree"),
    dict(multi_strategy="multi_output_tree"), dict(num_parallel_tree=3)])
def test_hist_gradient_layout_copies(metal, options, tmp_path):
    data, objective, _ = matrices("missing", "multitarget", tmp_path)
    params = dict(tree_method="hist", nthread=2, max_depth=2, max_bin=17,
                  seed=37, subsample=.7, **objective, **options)
    cpu = xgb.train(params | {"device": "cpu"}, data["cpu"], 3)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data["mps"], 3)
    after = xgb.metal_info()
    assert after["gradient_copy_dispatches"] > before["gradient_copy_dispatches"]
    assert after["hessian_extract_dispatches"] == before["hessian_extract_dispatches"]
    compare(cpu, gpu, data)


@pytest.mark.parametrize("layout", ["dense", "missing", "sparse", "categorical", "external"])
@pytest.mark.parametrize("sampling", ["uniform", "gradient_based"])
@pytest.mark.parametrize("subsample", [0., .3, .8])
def test_cuda_approx_cuts_precede_sampling(metal, layout, sampling, subsample, tmp_path):
    """Source oracle: GPUGlobalApproxMaker::Update extracts original Hessians.

    Supply fixed, nonconstant Hessians so gradients cannot obscure ordering.
    CPU approx with NO sampling is a reference for the existing sketch only;
    this does not validate CUDA's separate GPU sketch algorithm.
    """
    data, _, _ = matrices(layout, "regression", tmp_path)
    rows = data["cpu"].num_row()
    rng = np.random.default_rng(990)
    grad = rng.normal(size=rows).astype(np.float32)
    hess = rng.uniform(.01, 7., rows).astype(np.float32)
    hess[::13] = 0

    def objective(pred, dtrain):
        return grad.copy(), hess.copy()

    params = dict(tree_method="approx", nthread=2, max_depth=3, max_bin=17,
                  base_score=0., seed=39, objective="binary:logitraw", num_parallel_tree=3)
    xgb.train(params | {"device": "cpu", "subsample": 1.}, data["cpu"], 1, obj=objective)
    expected_cuts = data["cpu"].get_quantile_cut()
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps", "subsample": subsample,
                             "sampling_method": sampling}, data["mps"], 2, obj=objective)
    after = xgb.metal_info()
    # CUDA rebuilds these cuts once per updater call, not once per forest tree.
    assert after["sketch_finalize_dispatches"] - before["sketch_finalize_dispatches"] == 2
    assert after["hessian_extract_dispatches"] - before["hessian_extract_dispatches"] == 2
    assert after["gradient_copy_dispatches"] - before["gradient_copy_dispatches"] == 6
    assert after["sampling_dispatches"] - before["sampling_dispatches"] == 6
    assert after["column_init_dispatches"] - before["column_init_dispatches"] == 6
    for actual, expected in zip(data["mps"].get_quantile_cut(), expected_cuts):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    forest = gpu.get_dump(dump_format="json")
    if subsample > 0:
        # Fresh masks for each tree, unlike CPU approx's shared sampling buffer.
        assert len(set(forest[:3])) > 1
    else:
        np.testing.assert_array_equal(gpu.predict(data["mps"], output_margin=True), np.zeros(rows))


@pytest.mark.parametrize("trees", [1, 3])
def test_approx_root_matches_unsampled_formula(metal, trees):
    # Independent scalar oracle for the shared no-sampling leaf formula.
    x = np.zeros((259, 3), dtype=np.float32)
    grad = np.tile(np.array([-.5, 1., .25, -2.], dtype=np.float32), 65)[:len(x)]
    hess = np.arange(len(x), dtype=np.float32) % 7 / 8
    dtrain = xgb.DMatrix(x, label=np.zeros(len(x)))
    params = dict(device="mps", tree_method="approx", base_score=0., eta=.3,
                  reg_lambda=2., num_parallel_tree=trees, nthread=2)
    gpu = xgb.train(params, dtrain, 1, obj=lambda p, d: (grad.copy(), hess.copy()))
    expected = -.3 * np.sum(grad, dtype=np.float64) / (np.sum(hess, dtype=np.float64) + 2.)
    np.testing.assert_allclose(gpu.predict(dtrain), expected, atol=1e-7, rtol=1e-6)
