"""CUDA-style compressed external cache, concatenation and split storage on Metal."""
import gc
import weakref

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from cuda_sketch_reference import assert_cuts, assert_same_model_cpu
from test_metal_extmem_adapter import DiskPages, assert_no_reingestion


class HostPages(DiskPages):
    def __init__(self, prefix, batches, minimum, **metadata):
        super().__init__(prefix, batches, on_host=True, **metadata)
        self.min_cache_page_bytes = minimum


@pytest.mark.parametrize("layout", ["dense", "relative", "sparse", "categorical", "missing"])
@pytest.mark.parametrize("minimum", [0, 128, None])
@pytest.mark.parametrize("ratio", [None, 0., .0001, .25, 1.])
def test_host_cache_training(metal, tmp_path, layout, minimum, ratio):
    rng = np.random.default_rng(1301)
    x = rng.normal(size=(777, 7)).astype(np.float32)
    y = (np.arange(777) % 9).astype(np.float32)
    if layout in {"relative", "sparse", "missing"}:
        x[rng.random(x.shape) < (.9 if layout == "sparse" else .15)] = np.nan
        if layout == "sparse":
            x[:, 6] = np.nan
        elif layout == "relative":
            x[0] = np.arange(7)
        else:
            x[:] = np.nan
    if layout == "categorical":
        x = pd.DataFrame(x, columns=list("abcdefg"))
        x.b = pd.Categorical.from_codes(np.arange(777) % 9, categories=list("abcdefghi"))
        x.loc[::13, "b"] = np.nan
    weights = ((np.arange(777) % 7) / 4).astype(np.float32)
    split = [0, 0, 131, 388, 388, 777, 777]
    def chunks(a):
        return [a.iloc[b:e] if isinstance(a, pd.DataFrame) else a[b:e]
                for b, e in zip(split[:-1], split[1:])]
    it = HostPages(tmp_path / "unused", chunks(x), minimum, label=chunks(y), weight=chunks(weights))
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(it, device="mps", max_bin=17, nthread=2, cache_host_ratio=ratio)
    after = xgb.metal_info()
    assert not list(tmp_path.iterdir())
    assert after["ellpack_pack_dispatches"] > before["ellpack_pack_dispatches"]
    if layout != "missing":
        assert after["ellpack_copy_dispatches"] > before["ellpack_copy_dispatches"]
    assert after["ellpack_restore_dispatches"] > before["ellpack_restore_dispatches"]
    assert after["ellpack_debug_readback_bytes"] == before["ellpack_debug_readback_bytes"]
    if ratio == 0 or minimum is None:
        # The default groups this small dataset into one all-device cache page.
        assert after["ellpack_cached_host_bytes"] == before["ellpack_cached_host_bytes"]
        assert after["ellpack_cached_device_bytes"] > before["ellpack_cached_device_bytes"]
    elif ratio is not None and ratio > 0 and minimum == 0:
        assert after["ellpack_cached_host_bytes"] > before["ellpack_cached_host_bytes"]
        if ratio == 1:
            assert after["ellpack_cached_device_bytes"] == before["ellpack_cached_device_bytes"]
    np.testing.assert_array_equal(q.get_label(), y)
    np.testing.assert_array_equal(q.get_weight(), weights)
    raw = xgb.DMatrix(x, nthread=2)
    assert q.num_nonmissing() == raw.num_nonmissing()
    assert_cuts(q, 17, csr=raw.get_data(), page_rows=np.diff(split).tolist(), adapter=True)
    cpu = xgb.QuantileDMatrix(x, y, weight=weights, ref=q, max_bin=17, nthread=2)
    it.closed = True
    calls = it.calls
    before_train = xgb.metal_info()
    model = xgb.train(dict(device="mps", max_bin=17, max_depth=3, base_score=0, nthread=2), q, 3)
    assert xgb.metal_info()["index_uploads"] == before_train["index_uploads"]
    assert_same_model_cpu(model, q)
    for _ in range(2):
        np.testing.assert_array_equal(model.predict(q), model.predict(cpu))
    assert it.calls == calls
    assert_no_reingestion(after, xgb.metal_info())
    assert xgb.metal_info()["ellpack_debug_readback_bytes"] == after["ellpack_debug_readback_bytes"]
    ref = weakref.ref(it)
    del it
    gc.collect()
    assert ref() is None
    del q, cpu, model
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("ratio", [None, 0., .23, 1.])
@pytest.mark.parametrize("reference_device", ["cpu", "mps"])
def test_host_cache_reference_categories(metal, tmp_path, ratio, reference_device):
    frame = pd.DataFrame({"a": pd.Categorical.from_codes(np.arange(257) % 7, categories=list("abcdefg")),
                          "x": np.arange(257, dtype=np.float32) / 13})
    ref = xgb.QuantileDMatrix(frame, label=np.arange(257) % 5, device=reference_device, max_bin=17, nthread=2)
    reordered = frame.copy()
    reordered.a = reordered.a.cat.reorder_categories(list("gfedcba"))
    it = HostPages(tmp_path / "validation", [reordered[:137], reordered[137:]], 0,
                   feature_types=[ref.get_categories()] * 2)
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(it, ref=ref, device="mps", max_bin=17, nthread=2, cache_host_ratio=ratio)
    assert xgb.metal_info()["cuda_sketch_batches"] == before["cuda_sketch_batches"]
    for a, b in zip(q.get_quantile_cut(), ref.get_quantile_cut()):
        np.testing.assert_array_equal(a, b)
    model = xgb.train(dict(device="mps", max_bin=17, max_depth=3, nthread=2), ref, 3)
    np.testing.assert_array_equal(model.predict(q), model.predict(ref))
    assert_same_model_cpu(model, q)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("ratio", [-.1, 1.1, np.inf, -np.inf])
def test_host_cache_invalid_ratio(metal, tmp_path, ratio):
    it = HostPages(tmp_path / "invalid", [np.ones((17, 3), dtype=np.float32)], 0)
    with pytest.raises(xgb.core.XGBoostError):
        xgb.ExtMemQuantileDMatrix(it, device="mps", nthread=2, cache_host_ratio=ratio)
    assert not list(tmp_path.iterdir())


def test_host_cache_empty_and_cpu_training(metal, tmp_path):
    empty = xgb.ExtMemQuantileDMatrix(HostPages(tmp_path / "empty", [np.empty((0, 3), dtype=np.float32)] * 2, 0),
                                     device="mps", nthread=2, cache_host_ratio=.23)
    assert empty.num_row() == 0 and empty.num_col() == 3
    x = np.arange(777 * 3, dtype=np.float32).reshape(777, 3) % 31
    x[::7, 1] = np.nan
    y = np.arange(777, dtype=np.float32) % 5
    q = xgb.ExtMemQuantileDMatrix(HostPages(tmp_path / "cpu", [x[:131], x[131:]], 0,
                                          label=[y[:131], y[131:]]), device="mps", nthread=2, max_bin=17,
                                  cache_host_ratio=.23)
    cpu = xgb.QuantileDMatrix(x, y, ref=q, max_bin=17, nthread=2)
    params = dict(device="cpu", max_bin=17, max_depth=3, nthread=2)
    a, b = xgb.train(params, q, 3), xgb.train(params, cpu, 3)
    np.testing.assert_allclose(a.predict(q), b.predict(cpu), atol=2e-5, rtol=2e-5)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ["rows", "missing", "extra_missing", "less_batches", "more_batches"])
def test_host_cache_changed_iterator(metal, tmp_path, change):
    class Changing(HostPages):
        def __init__(self):
            self.pass_number = 0
            super().__init__(tmp_path / "changing", [], 0)

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
        xgb.ExtMemQuantileDMatrix(Changing(), device="mps", nthread=2, cache_host_ratio=.23)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["ranking", "vector", "survival"])
def test_host_cache_metadata(metal, tmp_path, kind):
    rng = np.random.default_rng(1311)
    x = rng.normal(size=(257, 4)).astype(np.float32)
    split = [0, 131, 257]
    def chunks(a):
        return [a[b:e] for b, e in zip(split[:-1], split[1:])]
    metadata, params = {}, dict(device="mps", max_bin=17, max_depth=3, nthread=2)
    if kind == "ranking":
        y = (np.arange(257) % 5).astype(np.float32)
        metadata = dict(label=chunks(y), group=[np.array([131]), np.array([126])],
                         weight=[np.array([.5], dtype=np.float32), np.array([.75], dtype=np.float32)])
        params["objective"] = "rank:pairwise"
    elif kind == "vector":
        y = x[:, :2].copy()
        margin = rng.normal(size=(257, 2)).astype(np.float32)
        metadata = dict(label=chunks(y), base_margin=chunks(margin))
        params["multi_strategy"] = "multi_output_tree"
    else:
        lower = np.exp(x[:, 0])
        upper = lower.copy(); upper[::7] = np.inf
        metadata = dict(label_lower_bound=chunks(lower), label_upper_bound=chunks(upper))
        params["objective"] = "survival:aft"
    q = xgb.ExtMemQuantileDMatrix(HostPages(tmp_path / "metadata", chunks(x), 0, **metadata),
                                  device="mps", max_bin=17, nthread=2, cache_host_ratio=.23)
    if kind == "ranking":
        np.testing.assert_array_equal(q.get_uint_info("group_ptr"), split)
        np.testing.assert_array_equal(q.get_weight(), [.5, .75])
    elif kind == "vector":
        np.testing.assert_array_equal(q.get_label(), y)
        np.testing.assert_array_equal(q.get_base_margin(), margin)
    else:
        np.testing.assert_array_equal(q.get_float_info("label_lower_bound"), lower)
        np.testing.assert_array_equal(q.get_float_info("label_upper_bound"), upper)
    model = xgb.train(params, q, 3)
    assert_same_model_cpu(model, q)
