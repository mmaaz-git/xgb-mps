"""Explicit Metal quantile construction follows the CUDA adapter sketch frontend."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xgboost as xgb
from cuda_sketch_reference import assert_cuts, assert_same_model_cpu


class Pages(xgb.DataIter):
    def __init__(self, batches, **metadata):
        self.batches, self.metadata, self.page = batches, metadata, 0
        super().__init__()

    def reset(self):
        self.page = 0

    def next(self, input_data):
        if self.page == len(self.batches):
            return False
        input_data(data=self.batches[self.page], **{k: v[self.page] for k, v in self.metadata.items()})
        self.page += 1
        return True


@pytest.mark.parametrize("layout", ["dense", "reverse", "columnar", "masked", "categorical"])
@pytest.mark.parametrize("weight_kind", ["none", "row", "group", "zero"])
@pytest.mark.parametrize("stream", [False, True])
def test_adapter_cuts_and_training(metal, layout, weight_kind, stream):
    rng = np.random.default_rng(581)
    values = rng.normal(size=(1031, 5)).astype(np.float32)
    values[::13] = np.nan
    values[:, 4] = np.nan  # Empty numeric feature.
    if layout == "reverse":
        values = values[::-1, ::-1]
    x = values
    types = ["q"] * 5
    if layout in {"columnar", "masked", "categorical"}:
        x = pd.DataFrame(values.copy(), columns=list("abcde"))
        if layout == "masked":
            x = x.astype("Float64")
        if layout == "categorical":
            x["b"] = pd.Categorical.from_codes(np.arange(1031) % 7, categories=list("abcdefg"))
            x.loc[::11, "b"] = np.nan
            types[1] = "c"
    y = (np.arange(1031) % 11).astype(np.float32)
    split = [0, 0, 257, 777, 1031] if stream else [0, 1031]
    batches, labels, weights, groups = [], [], [], []
    for begin, end in zip(split[:-1], split[1:]):
        batches.append(x.iloc[begin:end] if isinstance(x, pd.DataFrame) else x[begin:end])
        labels.append(y[begin:end])
        if weight_kind == "group":
            groups.append(np.array([end - begin], dtype=np.uint32) if end > begin else np.array([], dtype=np.uint32))
            weights.append(np.array([.5], dtype=np.float32) if end > begin else np.array([], dtype=np.float32))
        elif weight_kind != "none":
            weights.append(np.zeros(end - begin, dtype=np.float32) if weight_kind == "zero" else
                           ((np.arange(begin, end) % 9) / 4).astype(np.float32))
    meta = {"label": labels, "feature_types": [types] * len(batches)}
    if weights:
        meta["weight"] = weights
    if groups:
        meta["group"] = groups
    before = xgb.metal_info()
    q = xgb.QuantileDMatrix(Pages(batches, **meta), device="mps", max_bin=17, nthread=2)
    after = xgb.metal_info()
    assert after["adapter_prepare_dispatches"] > before["adapter_prepare_dispatches"]
    assert after["ellpack_adapter_dispatches"] > before["ellpack_adapter_dispatches"]
    assert after["ellpack_adapter_values"] - before["ellpack_adapter_values"] == q.num_nonmissing()
    for name in ["adapter_index_dispatches", "ellpack_pack_input_bytes", "ellpack_restore_dispatches",
                 "ellpack_host_materializations", "ellpack_host_readback_bytes"]:
        assert after[name] == before[name], name
    assert after["adapter_debug_readback_bytes"] == before["adapter_debug_readback_bytes"]
    for name in ["sketch_finalize_dispatches", "sketch_update_dispatches", "sketch_sorted_dispatches"]:
        assert after[name] == before[name]
    raw = xgb.DMatrix(x, feature_types=types, nthread=2)
    assert q.num_nonmissing() == raw.num_nonmissing()
    np.testing.assert_array_equal(q.get_label(), y)
    assert_cuts(q, 17, csr=raw.get_data(), page_rows=np.diff(split).tolist(), adapter=True)
    model = xgb.train(dict(device="mps", nthread=2, max_depth=3, max_bin=17, base_score=0,
                           objective="rank:pairwise" if weight_kind == "group" else "reg:squarederror"), q, 3)
    trained = xgb.metal_info()
    assert trained["ellpack_histogram_dispatches"] > after["ellpack_histogram_dispatches"]
    for name in ["index_uploads", "ellpack_pack_input_bytes", "ellpack_restore_dispatches",
                 "ellpack_host_materializations", "ellpack_host_readback_bytes"]:
        assert trained[name] == after[name], name
    assert_same_model_cpu(model, q)
    assert xgb.metal_info()["cuda_sketch_batches"] == after["cuda_sketch_batches"]


@pytest.mark.parametrize("reference_device", [None, "mps"])
def test_adapter_reference_and_model_reload(metal, reference_device):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(513, 4)).astype(np.float32)
    ref = xgb.QuantileDMatrix(x, label=x[:, 0], max_bin=33, nthread=2, device=reference_device)
    y = x[::3].copy()
    y[::5, 1] = np.nan
    before = xgb.metal_info()
    q = xgb.QuantileDMatrix(y, label=y[:, 0], ref=ref, max_bin=33, nthread=2, device="mps")
    assert xgb.metal_info()["cuda_sketch_batches"] == before["cuda_sketch_batches"]
    for a, b in zip(q.get_quantile_cut(), ref.get_quantile_cut()):
        np.testing.assert_array_equal(a, b)
    cpu_q = xgb.QuantileDMatrix(y, label=y[:, 0], ref=ref, max_bin=33, nthread=2)
    np.testing.assert_array_equal(q.get_data().indptr, cpu_q.get_data().indptr)
    np.testing.assert_array_equal(q.get_data().data, cpu_q.get_data().data)
    model = xgb.train(dict(device="mps", max_bin=33, nthread=2, max_depth=3), ref, 3)
    assert_same_model_cpu(model, q)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.uint8, np.int16,
                                  np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_adapter_types_and_missing(metal, dtype):
    x = (np.arange(2565).reshape(513, 5) % 83).astype(dtype)[::-1, ::-1]
    raw = xgb.DMatrix(np.ascontiguousarray(x), missing=17, nthread=2)
    q = xgb.QuantileDMatrix(x, missing=17, device="mps", nthread=2, max_bin=17)
    assert q.num_nonmissing() == raw.num_nonmissing()
    assert_cuts(q, 17, csr=raw.get_data(), adapter=True)


def test_adapter_empty_window_boundary(metal):
    # Physical storage is one scalar, but the logical window crosses CUDA's 2^26
    # raw-slot boundary in the middle of a seven-feature row.
    rows = (1 << 26) // 7 + 3
    x = np.broadcast_to(np.array(np.nan, dtype=np.float32), (rows, 7))
    before = xgb.metal_info()
    q = xgb.QuantileDMatrix(x, device="mps", nthread=2, max_bin=8)
    after = xgb.metal_info()
    assert q.num_row() == rows and q.num_nonmissing() == 0
    assert after["adapter_prepare_dispatches"] - before["adapter_prepare_dispatches"] == 4
    assert after["ellpack_adapter_dispatches"] - before["ellpack_adapter_dispatches"] == 2
    for name in ["adapter_index_dispatches", "ellpack_pack_input_bytes", "ellpack_restore_dispatches",
                 "ellpack_host_materializations", "ellpack_host_readback_bytes"]:
        assert after[name] == before[name], name
    assert after["adapter_debug_readback_bytes"] == before["adapter_debug_readback_bytes"]


def test_adapter_default_and_errors(metal):
    x = np.ones((17, 3), dtype=np.float32)
    before = xgb.metal_info()
    xgb.QuantileDMatrix(x, nthread=2)
    assert xgb.metal_info()["adapter_prepare_dispatches"] == before["adapter_prepare_dispatches"]
    with pytest.raises(xgb.core.XGBoostError, match="dense or columnar"):
        xgb.QuantileDMatrix(sp.csr_matrix(x), device="mps", nthread=2)
    bad = x.copy(); bad[5, 1] = np.inf
    with pytest.raises(xgb.core.XGBoostError, match="inf"):
        xgb.QuantileDMatrix(bad, device="mps", nthread=2)
    q = xgb.QuantileDMatrix(bad, missing=np.inf, device="mps", nthread=2)
    assert q.num_nonmissing() == x.size - 1
    empty = xgb.QuantileDMatrix(np.empty((0, 3), dtype=np.float32), device="mps", nthread=2)
    assert empty.num_row() == 0


def test_adapter_reference_category_encoding(metal):
    rng = np.random.default_rng(611)
    frame = pd.DataFrame({"a": pd.Categorical.from_codes(np.arange(513) % 7, categories=list("abcdefg")),
                          "x": rng.normal(size=513)})
    frame.loc[::17, "a"] = np.nan
    ref = xgb.QuantileDMatrix(frame, label=np.arange(513) % 3, device="mps", nthread=2, max_bin=17)
    changed = frame.copy()
    changed.a = changed.a.cat.reorder_categories(list("gfedcba"))
    before = xgb.metal_info()
    q = xgb.QuantileDMatrix(changed, ref=ref, feature_types=ref.get_categories(),
                           device="mps", nthread=2, max_bin=17)
    after = xgb.metal_info()
    assert after["category_recode_dispatches"] > before["category_recode_dispatches"]
    assert after["adapter_debug_readback_bytes"] == before["adapter_debug_readback_bytes"]
    np.testing.assert_array_equal(q.get_data().data, ref.get_data().data)
    np.testing.assert_array_equal(q.get_data().indptr, ref.get_data().indptr)
    model = xgb.train(dict(device="mps", nthread=2, max_bin=17, max_depth=3), ref, 3)
    np.testing.assert_array_equal(model.predict(q), model.predict(ref))
    bad = changed.copy()
    bad.a = bad.a.cat.add_categories(["unknown"])
    bad.loc[0, "a"] = "unknown"
    with pytest.raises(xgb.core.XGBoostError, match="Found a category not in the training set"):
        xgb.QuantileDMatrix(bad, ref=ref, feature_types=ref.get_categories(),
                           device="mps", nthread=2, max_bin=17)


@pytest.mark.parametrize("change", ["rows", "missing", "batches", "columns", "width"])
def test_adapter_rejects_changed_iterator(metal, change):
    class Changing(xgb.DataIter):
        def __init__(self):
            self.pass_number, self.page = 0, 0
            super().__init__()

        def reset(self):
            self.pass_number += 1
            self.page = 0

        def next(self, input_data):
            limit = 1 if self.pass_number > 1 and change == "batches" else 2
            if self.page == limit:
                return False
            rows = 3 if self.pass_number > 1 and change == "rows" else 4
            cols = 4 if self.pass_number > 1 and change == "columns" else 3
            x = np.ones((rows, cols), dtype=np.float32)
            if change == "width":
                x[:, 1] = np.nan
                if self.pass_number > 1:
                    x[0, 1], x[1, 2] = 1, np.nan  # Same nnz, larger maximum row width.
            if self.pass_number > 1 and change == "missing":
                x[0, 0] = np.nan
            input_data(data=x)
            self.page += 1
            return True

    with pytest.raises(xgb.core.XGBoostError):
        xgb.QuantileDMatrix(Changing(), device="mps", nthread=2)
