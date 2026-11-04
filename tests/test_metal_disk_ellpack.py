"""Compressed disk ownership, cache regeneration and public CPU/GPU consumers."""
import gc
import struct

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from test_metal_extmem_adapter import DiskPages
from test_metal_inmemory_ellpack import data
from test_metal_lazy_host import assert_no_host


def read_layout(path):
    """Read CUDA's aligned raw-page fields independently of the native reader."""
    blob = path.read_bytes()
    offset, pages = 0, []
    while offset < len(blob):
        rows, dense, stride, size = struct.unpack_from("<Q?7xQQ", blob, offset)
        payload = blob[offset + 32:offset + 32 + size]
        assert len(payload) == size
        end = offset + 32 + (size + 7) // 8 * 8
        assert not any(blob[offset + 32 + size:end])
        base, symbols = struct.unpack_from("<QQ", blob, end)
        bits = max(1, (symbols - 1).bit_length())
        compressed = (rows * stride * bits + 7) // 8
        assert size == max(9, (compressed + 8 + 3) // 4 * 4)
        assert payload[:8] == bytes(8)
        pages.append((rows, base, dense, stride, symbols))
        offset = end + 16
    assert offset == len(blob)
    return pages


def make_pages(tmp_path, layout, kind):
    x, y = data(layout)
    if kind == "vector":
        y = np.column_stack([y, y * -.7 + .1])
    if kind == "multiclass":
        y = (np.arange(len(x)) % 3).astype(np.float32)
    split = [0, 0, 137, 137, 312, 513, 513]
    def chunks(a):
        return [a.iloc[b:e] if isinstance(a, pd.DataFrame) else a[b:e]
                for b, e in zip(split[:-1], split[1:])]
    weight = (np.arange(len(x)) % 7 + 1).astype(np.float32) / 4
    it = DiskPages(tmp_path / "cache", chunks(x), label=chunks(y), weight=chunks(weight))
    return x, y, weight, it, split


@pytest.mark.parametrize("layout", ["dense", "relative", "sparse", "categorical", "missing"])
@pytest.mark.parametrize("method,kind", [
    ("hist", "scalar"), ("approx", "scalar"), ("approx_variable", "scalar"),
    ("hist", "vector"), ("hist", "dart"), ("approx_variable", "dart"),
])
def test_ordinary_disk_generations(metal, tmp_path, layout, method, kind):
    x, y, weight, it, split = make_pages(tmp_path, layout, kind)
    matrix = xgb.DMatrix(it, nthread=2)
    it.closed = True
    params = dict(device="mps", tree_method="hist" if method == "hist" else "approx",
                  nthread=2, max_bin=17, max_depth=2, base_score=.5)
    if method == "approx_variable":
        params["objective"] = "reg:pseudohubererror"
    if kind == "vector":
        params["multi_strategy"] = "multi_output_tree"
    if kind == "dart":
        params.update(booster="dart", rate_drop=.3)
    before = xgb.metal_info()
    model = xgb.train(params, matrix, 3)
    after = xgb.metal_info()
    generations = 3 if method == "approx_variable" else 1
    assert after["ellpack_io_write_pages"] - before["ellpack_io_write_pages"] == (len(split) - 1) * generations
    assert after["ellpack_csr_values"] - before["ellpack_csr_values"] == matrix.num_nonmissing() * generations
    for key in ["adapter_index_dispatches", "binning_dispatches", "ellpack_pack_input_bytes", "index_uploads"]:
        assert after[key] == before[key], key
    assert_no_host(before)
    files = list(tmp_path.glob("*.gradient_index.page"))
    assert len(files) == 1
    pages = read_layout(files[0])
    assert [p[:2] for p in pages] == list(zip(np.diff(split), split[:-1]))
    cuts = matrix.get_quantile_cut()
    if method == "hist":
        reference = xgb.QuantileDMatrix(x, y, weight=weight, ref=matrix, nthread=2, max_bin=17)
        for a, b in zip(reference.get_quantile_cut(), cuts):
            np.testing.assert_array_equal(a, b)
    assert_no_host(before)
    cpu = xgb.Booster(model_file=model.save_raw())
    cpu.set_param(dict(device="cpu", nthread=2))
    for option in [dict(output_margin=True), dict(pred_leaf=True), dict(pred_contribs=True),
                   dict(pred_interactions=True)]:
        strict = not (kind == "vector" and "pred_leaf" in option)
        a = model.predict(matrix, strict_shape=strict, **option)
        b = cpu.predict(matrix, strict_shape=strict, **option)
        if "output_margin" in option or "pred_leaf" in option:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
    assert_no_host(before)  # Ordinary CPU prediction reads raw CSR.
    if method == "hist":
        # Mutable device/max_bin changes must replace the correct cache generation.
        for bins in [33, 17]:
            xgb.train(params | dict(device="cpu", max_bin=bins), matrix, 2)
            start = xgb.metal_info()
            xgb.train(params | dict(max_bin=bins), matrix, 2)
            assert xgb.metal_info()["ellpack_io_write_pages"] - start["ellpack_io_write_pages"] == len(pages)
            read_layout(files[0])
            assert_no_host(start)
    del matrix, model, cpu
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("layout", ["dense", "relative", "sparse", "categorical", "missing"])
@pytest.mark.parametrize("kind", ["scalar", "vector", "multiclass", "dart"])
def test_quantile_disk_cpu_boundaries(metal, tmp_path, layout, kind):
    x, y, weight, it, split = make_pages(tmp_path, layout, kind)
    start = xgb.metal_info()
    matrix = xgb.ExtMemQuantileDMatrix(it, device="mps", max_bin=17, nthread=2)
    it.closed = True
    files = list(tmp_path.glob("*.metal-ellpack.page"))
    assert len(files) == 1
    assert [p[:2] for p in read_layout(files[0])] == list(zip(np.diff(split), split[:-1]))
    matrix.get_quantile_cut()
    reference = xgb.QuantileDMatrix(x, y, weight=weight, ref=matrix, nthread=2, max_bin=17)
    assert_no_host(start)
    params = dict(device="mps", nthread=2, max_bin=17, max_depth=2, base_score=.5)
    if kind == "vector":
        params["multi_strategy"] = "multi_output_tree"
    if kind == "multiclass":
        params.update(objective="multi:softprob", num_class=3)
    if kind == "dart":
        params.update(booster="dart", rate_drop=.3)
    model = xgb.train(params, matrix, 3)
    options = [dict(output_margin=True), dict(pred_leaf=True), dict(pred_contribs=True),
               dict(pred_interactions=True)]
    outputs = [model.predict(matrix, strict_shape=not (kind == "vector" and "pred_leaf" in o), **o) for o in options]
    assert_no_host(start)
    cpu = xgb.Booster(model_file=model.save_raw())
    cpu.set_param(dict(device="cpu", nthread=2))
    for option, expected in zip(options, outputs):
        strict = not (kind == "vector" and "pred_leaf" in option)
        a = cpu.predict(matrix, strict_shape=strict, **option)
        b = cpu.predict(reference, strict_shape=strict, **option)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, expected, rtol=2e-5, atol=2e-5)
        start = xgb.metal_info()
        np.testing.assert_array_equal(model.predict(matrix, strict_shape=strict, **option), expected)
        assert_no_host(start)
    start = xgb.metal_info()
    reloaded = xgb.Booster(model_file=model.save_raw())
    reloaded.set_param(dict(device="mps", nthread=2))
    np.testing.assert_array_equal(reloaded.predict(matrix, output_margin=True),
                                  model.predict(matrix, output_margin=True))
    assert_no_host(start)
    del matrix, model, cpu, reloaded
    gc.collect()
    assert not list(tmp_path.iterdir())
