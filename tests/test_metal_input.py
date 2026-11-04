"""Production prediction/SHAP consume GPU-prepared input without dense readback."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xgboost as xgb


def copy_to_metal(model):
    result = xgb.Booster(model_file=model.save_raw())
    result.set_param({"device": "mps", "nthread": 2})
    return result


def check_dispatch(before):
    after = xgb.metal_info()
    assert after["input_prepare_dispatches"] > before["input_prepare_dispatches"]
    assert after["input_debug_readback_bytes"] == before["input_debug_readback_bytes"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.uint8,
                                  np.int16, np.uint16, np.int32, np.uint32,
                                  np.int64, np.uint64])
@pytest.mark.parametrize("layout", ["c", "fortran", "reverse", "stride", "broadcast"])
def test_typed_inplace(metal, dtype, layout):
    rng = np.random.default_rng(375)
    x = rng.integers(0, 83, (1031, 7)).astype(dtype)
    x[::13, 2] = 17  # Explicit missing sentinel.
    cpu = xgb.train({"nthread": 2, "max_depth": 3},
                    xgb.DMatrix(x, label=x[:, 0] % 7, missing=17), 4)
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1, ::-1]
    elif layout == "stride":
        storage = np.empty((2062, 14), dtype=dtype)
        storage[::2, ::2] = x
        x = storage[::2, ::2]
    elif layout == "broadcast":
        x = np.broadcast_to(x[:1], x.shape)
    # Contiguous reference avoids relying on the CPU's unsigned negative strides.
    expected = cpu.inplace_predict(np.ascontiguousarray(x), missing=17)
    gpu = copy_to_metal(cpu)
    before = xgb.metal_info()
    actual = gpu.inplace_predict(x, missing=17)
    check_dispatch(before)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("layout", ["dense", "csr", "quantile", "sparse_quantile", "nullable"])
@pytest.mark.parametrize("kind", ["batch", "leaf", "contrib", "interaction"])
def test_prepared_prediction_surfaces(metal, layout, kind):
    rng = np.random.default_rng(493)
    x = rng.normal(size=(1031, 5)).astype(np.float32)
    y = x[:, 0].copy()
    x[rng.random(x.shape) < .6] = np.nan
    raw = x
    if layout in {"csr", "sparse_quantile"}:
        rows, cols = np.where(~np.isnan(x))
        raw = sp.csr_matrix((x[rows, cols], (rows, cols)), shape=x.shape)
    elif layout == "nullable":
        raw = pd.DataFrame(x).astype("Float64")
    matrix = xgb.QuantileDMatrix if "quantile" in layout else xgb.DMatrix
    kwargs = {"max_bin": 32} if "quantile" in layout else {}
    d = matrix(raw, label=y, **kwargs)
    cpu = xgb.train({"nthread": 2, "max_depth": 3, "max_bin": 32}, d, 4)
    options = {"leaf": {"pred_leaf": True}, "contrib": {"pred_contribs": True},
               "interaction": {"pred_interactions": True}}.get(kind, {})
    gpu = copy_to_metal(cpu)
    before = xgb.metal_info()
    actual = gpu.predict(d, **options)
    check_dispatch(before)
    np.testing.assert_allclose(actual, cpu.predict(d, **options), rtol=3e-6, atol=3e-6)
    if layout == "nullable":
        before = xgb.metal_info()
        actual = gpu.inplace_predict(raw)
        check_dispatch(before)
        np.testing.assert_array_equal(actual, cpu.inplace_predict(raw))


def test_chunked_masked_columns(metal):
    rng = np.random.default_rng(539)
    x = pd.DataFrame({"a": pd.array(rng.integers(-17, 18, 65539), dtype="Int64"),
                      "b": pd.array(rng.normal(size=65539), dtype="Float64")})
    x.loc[::7, "a"] = pd.NA
    x.loc[::13, "b"] = pd.NA
    cpu = xgb.train({"nthread": 2, "max_depth": 3},
                    xgb.DMatrix(x.iloc[:512], label=np.arange(512) % 3), 3)
    gpu = copy_to_metal(cpu)
    before = xgb.metal_info()
    actual = gpu.inplace_predict(x)
    check_dispatch(before)
    assert xgb.metal_info()["input_prepare_dispatches"] - before["input_prepare_dispatches"] == 2
    np.testing.assert_array_equal(actual, cpu.inplace_predict(x))
