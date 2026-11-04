"""Public training/prediction uses typed GPU dictionary sorting and recoding."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


def levels(kind):
    if kind == "strings":
        return ["", "a", "a\0", "a\0b", "café", "β", "🙂", "prefix" * 31]
    if kind == "uint64":
        return np.array([0, 1, 2**53, 2**53 + 1, 2**53 + 2, 2**53 + 3,
                         2**63 - 2, 2**63 - 1], dtype=np.uint64)
    return np.array([-2**63, -2**63 + 1, -2**53 - 1, -2**53, 0,
                     2**53, 2**53 + 1, 2**63 - 1], dtype=np.int64)


def data(kind):
    rng = np.random.default_rng(971)
    codes = rng.integers(0, 8, 513)
    codes[::17] = -1
    cats = levels(kind)
    frame = pd.DataFrame({"a": pd.Categorical.from_codes(codes, categories=cats),
                          "b": pd.Categorical.from_codes((codes + 3) % 8, categories=cats),
                          "x": rng.normal(size=codes.size).astype(np.float32)})
    y = (codes % 3).astype(np.float32) + .1 * frame.x.to_numpy()
    return frame, y


def clone(model, device):
    result = xgb.Booster(model_file=model.save_raw())
    result.set_param({"device": device, "nthread": 2})
    return result


@pytest.mark.parametrize("kind", ["strings", "uint64", "int64"])
@pytest.mark.parametrize("surface", ["batch", "leaf", "contrib", "interaction", "inplace"])
def test_category_names_gpu_roundtrip(metal, kind, surface):
    frame, y = data(kind)
    dtrain = xgb.DMatrix(frame, label=y, enable_categorical=True)
    before = xgb.metal_info()
    model = xgb.train({"device": "mps", "tree_method": "hist", "max_depth": 3,
                       "max_cat_to_onehot": 4, "nthread": 2}, dtrain, 5)
    assert xgb.metal_info()["category_sort_dispatches"] > before["category_sort_dispatches"]
    reordered = frame.copy()
    reordered["a"] = reordered.a.cat.reorder_categories(frame.a.cat.categories[::-1])
    reordered["b"] = reordered.b.cat.reorder_categories(frame.b.cat.categories.take(np.roll(np.arange(8), 3)))
    dtest = xgb.DMatrix(reordered, enable_categorical=True)
    options = {"leaf": {"pred_leaf": True}, "contrib": {"pred_contribs": True},
               "interaction": {"pred_interactions": True}}.get(surface, {})
    # Same row names with changed codes must preserve this model's result.
    expected = model.inplace_predict(frame) if surface == "inplace" else model.predict(dtrain, **options)
    cpu = clone(model, "cpu")
    restored = clone(model, "mps")
    before = xgb.metal_info()
    if surface == "inplace":
        actual, reference = restored.inplace_predict(reordered), cpu.inplace_predict(reordered)
    else:
        actual, reference = restored.predict(dtest, **options), cpu.predict(dtest, **options)
    assert xgb.metal_info()["category_recode_dispatches"] > before["category_recode_dispatches"]
    after = xgb.metal_info()
    assert after["input_prepare_dispatches"] > before["input_prepare_dispatches"]
    assert after["input_recode_dispatches"] > before["input_recode_dispatches"]
    assert after["input_debug_readback_bytes"] == before["input_debug_readback_bytes"]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual, reference, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("kind", ["strings", "uint64", "int64"])
def test_category_subset_and_unknown(metal, kind):
    frame, y = data(kind)
    model = xgb.train({"device": "mps", "tree_method": "hist", "nthread": 2},
                      xgb.DMatrix(frame, label=y, enable_categorical=True), 3)
    subset = frame.loc[frame.a.cat.codes.isin([1, 3])].copy()
    subset.a = subset.a.cat.remove_unused_categories()
    subset.b = subset.b.cat.remove_unused_categories()
    cpu = clone(model, "cpu")
    before = xgb.metal_info()["category_recode_dispatches"]
    np.testing.assert_array_equal(model.inplace_predict(subset), cpu.inplace_predict(subset))
    assert xgb.metal_info()["category_recode_dispatches"] > before
    unknown = "unknown-category" if kind == "strings" else 23
    bad = frame.copy()
    bad.a = bad.a.cat.add_categories(pd.Index([unknown], dtype=frame.a.cat.categories.dtype))
    bad.loc[0, "a"] = unknown
    before = xgb.metal_info()["category_recode_dispatches"]
    with pytest.raises(xgb.core.XGBoostError, match="Found a category not in the training set"):
        model.inplace_predict(bad)
    assert xgb.metal_info()["category_recode_dispatches"] > before


def test_category_schema_validation_precedes_name_search(metal):
    frame, y = data("uint64")
    model = xgb.train({"device": "mps", "nthread": 2},
                      xgb.DMatrix(frame, label=y, enable_categorical=True), 2)
    bad = frame.copy()
    bad.a = bad.a.cat.add_categories([np.uint64(23)])
    bad.loc[0, "a"] = 23
    # Later type mismatch must win over the earlier unknown category, as CUDA's
    # whole-dictionary validation runs before its parallel name searches.
    bad.b = pd.Categorical.from_codes(bad.b.cat.codes, categories=levels("int64"))
    before = xgb.metal_info()["category_recode_dispatches"]
    with pytest.raises(xgb.core.XGBoostError, match="data type doesn't match"):
        model.inplace_predict(bad)
    assert xgb.metal_info()["category_recode_dispatches"] > before
