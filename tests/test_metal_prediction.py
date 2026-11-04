"""Exercise actual Metal tree traversal using the same CPU-trained model."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xgboost as xgb
from cuda_uniform_reference import UniformSquaredError


def metal_copy(model):
    result = xgb.Booster(model_file=model.save_raw())
    result.set_param({"device": "mps", "nthread": 2})
    return result

@pytest.mark.parametrize("layout", ["dense", "missing", "sparse", "quantile"])
@pytest.mark.parametrize("kind", ["regression", "multiclass", "vector", "dart", "forest"])
def test_gpu_batch_leaf_and_inplace(metal, layout, kind):
    rng = np.random.default_rng(852)
    x = rng.normal(size=(1031, 9)).astype(np.float32)
    y = x[:, 0] * 2 - x[:, 1] + rng.normal(scale=0.1, size=len(x))
    if layout in {"missing", "quantile"}:
        x[rng.random(x.shape) < 0.25] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < 0.7] = 0
        x = sp.csr_matrix(x)
    p = dict(tree_method="hist", nthread=2, max_depth=4, max_bin=32, base_score=0.4)
    if kind == "multiclass":
        y = np.digitize(y, [-1, 1])
        p.update(objective="multi:softprob", num_class=3)
    elif kind == "vector":
        y = np.column_stack([y, -y, y * 0.5])
        p.update(multi_strategy="multi_output_tree")
    elif kind == "dart":
        p.update(booster="dart", rate_drop=0.3, skip_drop=0, seed=53)
    elif kind == "forest":
        p.update(num_parallel_tree=3, subsample=0.7, colsample_bynode=0.7)
    matrix = xgb.QuantileDMatrix if layout == "quantile" else xgb.DMatrix
    d = matrix(x, label=y, max_bin=32) if layout == "quantile" else matrix(x, label=y)
    cpu = xgb.train(p, d, 8)
    gpu = metal_copy(cpu)
    for options in [{}, {"output_margin": True}, {"pred_leaf": True}, {"iteration_range": (2, 6)}]:
        expected = cpu.predict(d, **options)
        gpu = metal_copy(cpu)  # Fresh cache: every option must exercise traversal.
        before = xgb.metal_info()["prediction_dispatches"]
        actual = gpu.predict(d, **options)
        assert xgb.metal_info()["prediction_dispatches"] > before
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    # Inplace prediction accepts raw input even for a model trained from quantized data.
    for options in [{}, {"predict_type": "margin"}, {"iteration_range": (2, 6)}]:
        expected = cpu.inplace_predict(x, **options)
        before = xgb.metal_info()["prediction_dispatches"]
        actual = gpu.inplace_predict(x, **options)
        assert xgb.metal_info()["prediction_dispatches"] > before
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

@pytest.mark.parametrize("onehot", [1, 64])
def test_categorical_recode_prediction(metal, onehot):
    rng = np.random.default_rng(3)
    categories = [f"category-{i}" for i in range(41)]
    x = pd.DataFrame({"cat": pd.Categorical(rng.choice(categories, 1025), categories=categories),
                      "num": rng.normal(size=1025)})
    y = x.cat.cat.codes.to_numpy() % 7 + x.num.to_numpy()
    cpu = xgb.train(dict(tree_method="hist", nthread=2, max_depth=4,
                         max_cat_to_onehot=onehot), xgb.DMatrix(x, label=y, enable_categorical=True), 8)
    gpu = metal_copy(cpu)
    x["cat"] = x.cat.cat.reorder_categories(list(reversed(categories)))
    x.loc[::13, "cat"] = np.nan
    d = xgb.DMatrix(x, enable_categorical=True)
    for leaf in [False, True]:
        before = xgb.metal_info()["prediction_dispatches"]
        actual = gpu.predict(d, pred_leaf=leaf)
        assert xgb.metal_info()["prediction_dispatches"] > before
        np.testing.assert_array_equal(actual, cpu.predict(d, pred_leaf=leaf))
    before = xgb.metal_info()["prediction_dispatches"]
    np.testing.assert_array_equal(gpu.inplace_predict(x), cpu.inplace_predict(x))
    assert xgb.metal_info()["prediction_dispatches"] > before

@pytest.mark.parametrize("rows", [0, 1, 65537])
def test_prediction_chunks_and_base_margin(metal, rows):
    rng = np.random.default_rng(43)
    train = rng.normal(size=(512, 3)).astype(np.float32)
    cpu = xgb.train(dict(nthread=2, max_depth=3), xgb.DMatrix(train, label=train[:, 0]), 3)
    gpu = metal_copy(cpu)
    x = rng.normal(size=(rows, 3)).astype(np.float32)
    margin = rng.normal(size=rows).astype(np.float32)
    d = xgb.DMatrix(x, base_margin=margin)
    before = xgb.metal_info()["prediction_dispatches"]
    np.testing.assert_array_equal(gpu.predict(d), cpu.predict(d))
    np.testing.assert_array_equal(gpu.inplace_predict(x, base_margin=margin),
                                  cpu.inplace_predict(x, base_margin=margin))
    assert (xgb.metal_info()["prediction_dispatches"] > before) == (rows > 0)

@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_gpu_leaf_cache_update(metal, strategy):
    rng = np.random.default_rng(21)
    x = rng.normal(size=(1031, 5)).astype(np.float32)
    y = np.column_stack([x[:, 0], -x[:, 1], x[:, 2] ** 2])
    d = xgb.DMatrix(x, label=y)
    p = dict(nthread=2, max_depth=3, multi_strategy=strategy, subsample=0.7,
             base_score=0., seed_per_iteration=True)
    expected = xgb.train(p | {"device": "cpu", "subsample": 1.}, d, 5,
                         obj=UniformSquaredError(p)).predict(d)
    before = xgb.metal_info()["leaf_update_dispatches"]
    actual = xgb.train(p | {"device": "mps"}, d, 5).predict(d)
    assert xgb.metal_info()["leaf_update_dispatches"] > before
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
