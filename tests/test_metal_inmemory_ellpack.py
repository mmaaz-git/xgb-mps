"""In-memory ELLPACK construction, bidirectional conversion and CPU data export."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from test_metal_adapter import Pages
from test_metal_lazy_host import assert_no_host


class ClosedPages(Pages):
    def __init__(self, *args, **kwargs):
        self.closed = False
        super().__init__(*args, **kwargs)

    def reset(self):
        assert not self.closed, "Conversion must not reset input"
        super().reset()

    def next(self, input_data):
        assert not self.closed, "Conversion must not reread input"
        return super().next(input_data)


def data(layout, rows=513):
    rng = np.random.default_rng(1901)
    x = rng.normal(size=(rows, 7)).astype(np.float32)
    y = (np.arange(rows) % 7).astype(np.float32) / 4
    if layout != "dense":
        x[rng.random(x.shape) < (.8 if layout == "sparse" else .2)] = np.nan
        if layout == "relative":
            x[0] = 1
        if layout in {"sparse", "missing"}:
            x[:, 6] = np.nan
        if layout == "missing":
            x[:] = np.nan
    if layout == "categorical":
        x = pd.DataFrame(x, columns=list("abcdefg"))
        x.b = pd.Categorical.from_codes(np.arange(rows) % 7, categories=list("abcdefg"))
        x.loc[::13, "b"] = np.nan
    return x, y


def assert_export(actual, expected):
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("layout", ["dense", "relative", "sparse", "categorical", "missing"])
@pytest.mark.parametrize("stream", [False, True])
def test_inmemory_compressed_consumers(metal, device, layout, stream):
    x, y = data(layout)
    split = [0, 0, 137, 137, 513, 513] if stream else [0, len(x)]
    chunks = [x.iloc[b:e] if isinstance(x, pd.DataFrame) else x[b:e]
              for b, e in zip(split[:-1], split[1:])]
    it = ClosedPages(chunks, label=[y[b:e] for b, e in zip(split[:-1], split[1:])])
    start = xgb.metal_info()
    q = xgb.QuantileDMatrix(it, device=device, nthread=2, max_bin=17)
    it.closed = True
    q.get_quantile_cut()
    reference = xgb.QuantileDMatrix(x, y, ref=q, device="cpu", nthread=2, max_bin=17)
    ready = xgb.metal_info()
    assert_no_host(start)
    assert ready["ellpack_pack_input_bytes"] == start["ellpack_pack_input_bytes"]
    if device == "mps":
        assert ready["ellpack_adapter_values"] - start["ellpack_adapter_values"] == q.num_nonmissing()
        assert ready["ellpack_view_pages"] - start["ellpack_view_pages"] == 1
        assert ready["adapter_index_dispatches"] == start["adapter_index_dispatches"]
    params = dict(device="mps", nthread=2, max_bin=17, max_depth=3, base_score=.5)
    model = xgb.train(params, q, 3)
    options = ["output_margin", "pred_leaf", "pred_contribs", "pred_interactions"]
    results = [model.predict(q, strict_shape=True, **{option: True}) for option in options]
    after_gpu = xgb.metal_info()
    assert_no_host(start)
    assert after_gpu["index_uploads"] == start["index_uploads"]
    assert after_gpu["ellpack_histogram_dispatches"] > ready["ellpack_histogram_dispatches"]
    assert after_gpu["ellpack_input_dispatches"] > ready["ellpack_input_dispatches"]
    uploaded = (q.num_nonmissing() * (1 if layout == "dense" else 4) + (len(x) + 1) * 8
                if device == "cpu" and q.num_nonmissing() else 0)
    assert after_gpu["ellpack_pack_input_bytes"] - start["ellpack_pack_input_bytes"] == uploaded
    assert after_gpu["ellpack_view_pages"] - start["ellpack_view_pages"] == 1
    cpu = xgb.Booster(model_file=model.save_raw())
    cpu.set_param(dict(device="cpu", nthread=2))
    for option, expected in zip(options, results):
        actual = cpu.predict(q, strict_shape=True, **{option: True})
        oracle = cpu.predict(reference, strict_shape=True, **{option: True})
        np.testing.assert_array_equal(actual, oracle)
        if option in {"output_margin", "pred_leaf"}:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    after_cpu = xgb.metal_info()
    assert after_cpu["ellpack_host_materializations"] - start["ellpack_host_materializations"] == int(device == "mps")
    for _ in range(2):
        assert_export(q.get_data(), reference.get_data())
        fresh = xgb.Booster(model_file=model.save_raw())
        fresh.set_param(params)
        np.testing.assert_array_equal(fresh.predict(q, output_margin=True, strict_shape=True), results[0])
    final = xgb.metal_info()
    for key in ["ellpack_host_materializations", "ellpack_host_readback_bytes", "ellpack_pack_input_bytes",
                "ellpack_count_dispatches", "ellpack_view_pages", "index_uploads"]:
        assert final[key] == after_cpu[key], key


@pytest.mark.parametrize("layout", ["dense", "sparse", "categorical"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_inmemory_export_is_first_host_request(metal, layout, device):
    x, y = data(layout, 137)
    q = xgb.QuantileDMatrix(x, y, device=device, nthread=2, max_bin=17)
    reference = xgb.QuantileDMatrix(x, y, ref=q, device="cpu", nthread=2, max_bin=17)
    before = xgb.metal_info()
    assert_export(q.get_data(), reference.get_data())
    after = xgb.metal_info()
    assert after["ellpack_host_materializations"] - before["ellpack_host_materializations"] == int(device == "mps")
    assert after["ellpack_pack_input_bytes"] == before["ellpack_pack_input_bytes"]
    assert_export(q.get_data(), reference.get_data())
    assert_no_host(after)


def test_inmemory_compressed_real_window_boundary(metal):
    rows = (1 << 26) // 7 + 3
    pattern = np.array([1, np.nan, 2, np.nan, 3, np.nan, np.nan], dtype=np.float32)
    reference = xgb.QuantileDMatrix(pattern.reshape(1, 7), device="cpu", max_bin=8, nthread=2)
    x = np.broadcast_to(pattern, (rows, 7))
    before = xgb.metal_info()
    q = xgb.QuantileDMatrix(x, ref=reference, device="mps", max_bin=8, nthread=2)
    after = xgb.metal_info()
    assert q.num_row() == rows and q.num_nonmissing() == rows * 3
    assert after["ellpack_adapter_dispatches"] - before["ellpack_adapter_dispatches"] == 3
    assert after["ellpack_adapter_values"] - before["ellpack_adapter_values"] == rows * 3
    for key in ["ellpack_pack_input_bytes", "adapter_index_dispatches", "adapter_debug_readback_bytes",
                "ellpack_debug_readback_bytes", "cuda_sketch_batches"]:
        assert after[key] == before[key], key
    assert_no_host(before)
