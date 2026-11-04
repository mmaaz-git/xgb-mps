"""Metal adapter construction writes and reuses bounded disk-cache pages."""
import gc
import weakref

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xgboost as xgb

from cuda_sketch_reference import assert_cuts, assert_same_model_cpu


class DiskPages(xgb.DataIter):
    def __init__(self, prefix, batches, *, on_host=False, **metadata):
        self.batches, self.metadata, self.page = batches, metadata, 0
        self.calls = 0
        self.closed = False
        super().__init__(cache_prefix=str(prefix), on_host=on_host)

    def reset(self):
        assert not self.closed, "Cache reading must not reset the input iterator"
        self.page = 0

    def next(self, input_data):
        assert not self.closed, "Cache reading must not reread raw input"
        self.calls += 1
        if self.page == len(self.batches):
            return False
        input_data(data=self.batches[self.page],
                   **{k: v[self.page] for k, v in self.metadata.items()})
        self.page += 1
        return True


def assert_no_reingestion(before, after):
    for name in ["adapter_prepare_dispatches", "adapter_index_dispatches",
                 "adapter_debug_readback_bytes", "cuda_sketch_batches"]:
        assert after[name] == before[name], name


@pytest.mark.parametrize("layout", ["dense", "reverse", "columnar", "masked", "categorical"])
@pytest.mark.parametrize("weight_kind", ["none", "row", "group", "zero"])
def test_extmem_adapter_cuts_cache_training(metal, tmp_path, layout, weight_kind):
    rng = np.random.default_rng(1201)
    x = rng.normal(size=(1031, 5)).astype(np.float32)
    types = ["q"] * 5
    if layout != "dense":
        x[::13] = np.nan
        x[:, 4] = np.nan
    if layout == "reverse":
        x = x[::-1, ::-1]
    if layout in {"columnar", "masked", "categorical"}:
        x = pd.DataFrame(x.copy(), columns=list("abcde"))
        if layout == "masked":
            x = x.astype("Float64")
        if layout == "categorical":
            x.b = pd.Categorical.from_codes(np.arange(1031) % 7, categories=list("abcdefg"))
            x.loc[::11, "b"] = np.nan
            types[1] = "c"
    # Empty first, middle and final pages exercise cache page boundaries and row bases.
    split = [0, 0, 257, 257, 777, 1031, 1031]
    batches, labels, weights, groups = [], [], [], []
    y = (np.arange(1031) % 11).astype(np.float32)
    for begin, end in zip(split[:-1], split[1:]):
        batches.append(x.iloc[begin:end] if isinstance(x, pd.DataFrame) else x[begin:end])
        labels.append(y[begin:end])
        if weight_kind == "group":
            groups.append(np.array([end - begin], dtype=np.uint32))
            weights.append(np.array([.5], dtype=np.float32))
        elif weight_kind != "none":
            weights.append(np.zeros(end - begin, dtype=np.float32) if weight_kind == "zero"
                           else ((np.arange(begin, end) % 9) / 4).astype(np.float32))
    meta = dict(label=labels, feature_types=[types] * len(batches))
    if weights:
        meta["weight"] = weights
    if groups:
        meta["group"] = groups
    it = DiskPages(tmp_path / "cache", batches, **meta)
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(it, device="mps", max_bin=17, nthread=2)
    after = xgb.metal_info()
    assert after["adapter_prepare_dispatches"] > before["adapter_prepare_dispatches"]
    assert after["adapter_index_dispatches"] - before["adapter_index_dispatches"] == 3
    assert after["adapter_debug_readback_bytes"] == before["adapter_debug_readback_bytes"]
    for name in ["sketch_finalize_dispatches", "sketch_update_dispatches", "sketch_sorted_dispatches"]:
        assert after[name] == before[name]
    cache = list(tmp_path.glob("cache-*.gradient_index.page"))
    assert len(cache) == 1 and cache[0].stat().st_size > 0
    raw = xgb.DMatrix(x, feature_types=types, nthread=2)
    assert q.num_nonmissing() == raw.num_nonmissing()
    np.testing.assert_array_equal(q.get_label(), y)
    if weights:
        np.testing.assert_array_equal(q.get_weight(), np.concatenate(weights))
    if groups:
        np.testing.assert_array_equal(q.get_uint_info("group_ptr"), split)
    assert_cuts(q, 17, csr=raw.get_data(), page_rows=np.diff(split).tolist(), adapter=True)
    # Fixed reference cuts make CPU bin assignment an independent exact oracle.
    cpu = xgb.QuantileDMatrix(x, ref=q, max_bin=17, nthread=2, feature_types=types)
    calls = it.calls
    it.closed = True
    model = xgb.train(dict(device="mps", nthread=2, max_depth=3, max_bin=17, base_score=0,
                           objective="rank:pairwise" if weight_kind == "group" else "reg:squarederror"), q, 3)
    assert_same_model_cpu(model, q)
    for _ in range(3):
        np.testing.assert_array_equal(model.predict(q), model.predict(cpu))
    assert it.calls == calls
    assert_no_reingestion(after, xgb.metal_info())
    # Input callbacks and arrays are not retained by the completed cache.
    input_ref = weakref.ref(it)
    del it
    gc.collect()
    assert input_ref() is None
    del q, cpu, model
    gc.collect()
    assert not list(tmp_path.glob("*.gradient_index.page"))


@pytest.mark.parametrize("reference_kind", ["cpu", "mps", "external"])
def test_extmem_adapter_reference(metal, tmp_path, reference_kind):
    rng = np.random.default_rng(1212)
    x = rng.normal(size=(513, 4)).astype(np.float32)
    if reference_kind == "external":
        ref = xgb.ExtMemQuantileDMatrix(DiskPages(tmp_path / "ref", [x[:231], x[231:]]),
                                        device="mps", max_bin=33, nthread=2)
    else:
        ref = xgb.QuantileDMatrix(x, max_bin=33, nthread=2,
                                  device="mps" if reference_kind == "mps" else None)
    y = x[::3].copy()
    y[::5, 1] = np.nan
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(DiskPages(tmp_path / "validation", [y[:17], y[17:]],
                                           label=[y[:17, 0], y[17:, 0]]),
                                   ref=ref, max_bin=33, nthread=2, device="mps")
    assert xgb.metal_info()["cuda_sketch_batches"] == before["cuda_sketch_batches"]
    for a, b in zip(q.get_quantile_cut(), ref.get_quantile_cut()):
        np.testing.assert_array_equal(a, b)
    cpu = xgb.QuantileDMatrix(y, ref=ref, max_bin=33, nthread=2)
    model = xgb.train(dict(device="mps", nthread=2, max_bin=33, max_depth=3), q, 3)
    assert_same_model_cpu(model, q)
    np.testing.assert_array_equal(model.predict(q), model.predict(cpu))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.uint8, np.int16,
                                  np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_extmem_adapter_types(metal, tmp_path, dtype):
    x = (np.arange(2565).reshape(513, 5) % 83).astype(dtype)[::-1, ::-1]
    raw = xgb.DMatrix(x, missing=17, nthread=2)
    q = xgb.ExtMemQuantileDMatrix(DiskPages(tmp_path / "types", [x]), missing=17,
                                   device="mps", nthread=2, max_bin=17)
    assert q.num_nonmissing() == raw.num_nonmissing()
    assert_cuts(q, 17, csr=raw.get_data(), adapter=True)
    cpu = xgb.QuantileDMatrix(x, missing=17, ref=q, max_bin=17, nthread=2)
    q.set_label(np.arange(q.num_row()) % 5)
    model = xgb.train(dict(device="mps", nthread=2, max_bin=17, max_depth=3), q, 2)
    np.testing.assert_array_equal(model.predict(q), model.predict(cpu))


@pytest.mark.parametrize("change", ["rows", "missing", "extra_missing", "less_batches", "more_batches"])
def test_extmem_adapter_failed_construction_cleans_cache(metal, tmp_path, change):
    class Changing(DiskPages):
        def __init__(self):
            self.pass_number = 0
            super().__init__(tmp_path / "changing", [])

        def reset(self):
            self.pass_number += 1
            self.page = 0

        def next(self, input_data):
            changed = self.pass_number > 1
            limit = (1 if change == "less_batches" else 3) if changed and "batches" in change else 2
            if self.page == limit:
                return False
            rows = 3 if changed and self.page == 1 and change == "rows" else 4
            x = np.ones((rows, 3), dtype=np.float32)
            if self.page == 1 and ((changed and change == "missing") or
                                   (not changed and change == "extra_missing")):
                x[0, 0] = np.nan
            input_data(data=x)
            self.page += 1
            return True

    with pytest.raises(xgb.core.XGBoostError):
        xgb.ExtMemQuantileDMatrix(Changing(), device="mps", nthread=2)
    gc.collect()
    assert not list(tmp_path.glob("*.gradient_index.page"))


def test_extmem_adapter_defaults_errors_empty(metal, tmp_path):
    x = np.ones((17, 3), dtype=np.float32)
    def pages(values, **kw):
        return DiskPages(tmp_path / "edge", values, **kw)
    before = xgb.metal_info()
    cpu = xgb.ExtMemQuantileDMatrix(pages([x]), nthread=2)
    assert xgb.metal_info()["adapter_prepare_dispatches"] == before["adapter_prepare_dispatches"]
    del cpu
    hosted = xgb.ExtMemQuantileDMatrix(pages([x], on_host=True), device="mps", nthread=2)
    assert hosted.num_nonmissing() == x.size
    del hosted
    assert not list(tmp_path.iterdir())
    with pytest.raises(xgb.core.XGBoostError, match="dense or columnar"):
        xgb.ExtMemQuantileDMatrix(pages([sp.csr_matrix(x)]), device="mps", nthread=2)
    with pytest.raises(xgb.core.XGBoostError, match="cache_host_ratio"):
        xgb.ExtMemQuantileDMatrix(pages([x]), cache_host_ratio=.5, device="mps", nthread=2)
    bad = x.copy(); bad[5, 1] = np.inf
    with pytest.raises(xgb.core.XGBoostError, match="inf"):
        xgb.ExtMemQuantileDMatrix(pages([x, bad]), device="mps", nthread=2)
    assert not list(tmp_path.glob("*.gradient_index.page"))
    valid = xgb.ExtMemQuantileDMatrix(pages([bad]), missing=np.inf, device="mps", nthread=2)
    assert valid.num_nonmissing() == x.size - 1
    empty = xgb.ExtMemQuantileDMatrix(pages([x[:0], x[:0]]), device="mps", nthread=2)
    assert empty.num_row() == 0 and empty.num_col() == 3
    missing = xgb.ExtMemQuantileDMatrix(pages([np.full((33, 3), np.nan, dtype=np.float32)]),
                                        device="mps", nthread=2)
    assert missing.num_row() == 33 and missing.num_nonmissing() == 0


def test_extmem_adapter_category_references(metal, tmp_path):
    rng = np.random.default_rng(1221)
    frame = pd.DataFrame({"a": pd.Categorical.from_codes(np.arange(513) % 7, categories=list("abcdefg")),
                          "x": rng.normal(size=513)})
    frame.loc[::17, "a"] = np.nan
    ref = xgb.QuantileDMatrix(frame, label=np.arange(513) % 3, device="mps", nthread=2, max_bin=17)
    changed = frame.copy()
    changed.a = changed.a.cat.reorder_categories(list("gfedcba"))
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(DiskPages(tmp_path / "cats", [changed[:211], changed[211:]],
                                           feature_types=[ref.get_categories()] * 2),
                                   ref=ref, device="mps", nthread=2, max_bin=17)
    after = xgb.metal_info()
    assert after["category_recode_dispatches"] > before["category_recode_dispatches"]
    assert after["adapter_debug_readback_bytes"] == before["adapter_debug_readback_bytes"]
    model = xgb.train(dict(device="mps", nthread=2, max_bin=17, max_depth=3), ref, 3)
    np.testing.assert_array_equal(model.predict(q), model.predict(ref))
    bad = changed.copy()
    bad.a = bad.a.cat.add_categories(["unknown"])
    bad.loc[0, "a"] = "unknown"
    with pytest.raises(xgb.core.XGBoostError, match="Found a category not in the training set"):
        xgb.ExtMemQuantileDMatrix(DiskPages(tmp_path / "bad-cats", [changed, bad],
                                           feature_types=[ref.get_categories()] * 2),
                                   ref=ref, device="mps", nthread=2, max_bin=17)


@pytest.mark.parametrize("sparse", [False, True])
def test_extmem_cpu_training_column_interop(metal, tmp_path, sparse):
    rng = np.random.default_rng(1231)
    x = rng.normal(size=(777, 9)).astype(np.float32)
    y = (x[:, 0] * 2 - x[:, 3]).copy()
    if sparse:
        x[rng.random(x.shape) < .85] = np.nan
    it = DiskPages(tmp_path / "cpu-train", [x[:231], x[231:]], label=[y[:231], y[231:]])
    q = xgb.ExtMemQuantileDMatrix(it, device="mps", max_bin=31, nthread=2)
    cpu = xgb.QuantileDMatrix(x, y, ref=q, max_bin=31, nthread=2)
    params = dict(device="cpu", tree_method="hist", max_bin=31, max_depth=3, nthread=2)
    from_pages = xgb.train(params, q, 4)
    from_memory = xgb.train(params, cpu, 4)
    np.testing.assert_allclose(from_pages.predict(q), from_memory.predict(cpu), rtol=2e-5, atol=2e-5)


def test_extmem_adapter_vector_metadata(metal, tmp_path):
    rng = np.random.default_rng(1237)
    x = rng.normal(size=(257, 4)).astype(np.float32)
    y = np.column_stack([x[:, 0], x[:, 1] * 3])
    margin = rng.normal(size=(257, 2)).astype(np.float32)
    bounds = np.arange(257, dtype=np.float32)
    split = [0, 131, 257]
    def chunks(values):
        return [values[a:b] for a, b in zip(split[:-1], split[1:])]
    q = xgb.ExtMemQuantileDMatrix(
        DiskPages(tmp_path / "metadata", chunks(x), label=chunks(y), base_margin=chunks(margin),
                  label_lower_bound=chunks(bounds), label_upper_bound=chunks(bounds + 1),
                  feature_names=[list("abcd")] * 2),
        device="mps", max_bin=17, nthread=2)
    np.testing.assert_array_equal(q.get_label(), y)
    np.testing.assert_array_equal(q.get_base_margin(), margin)
    np.testing.assert_array_equal(q.get_float_info("label_lower_bound"), bounds)
    np.testing.assert_array_equal(q.get_float_info("label_upper_bound"), bounds + 1)
    assert q.feature_names == list("abcd")
    model = xgb.train(dict(device="mps", nthread=2, max_bin=17, max_depth=3,
                           multi_strategy="multi_output_tree"), q, 3)
    assert_same_model_cpu(model, q)
