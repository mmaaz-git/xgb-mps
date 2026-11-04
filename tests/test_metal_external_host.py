"""Ordinary external cache defaults, empty inputs and existing parameter errors."""
import gc

import numpy as np
import pytest
import xgboost as xgb

from test_metal_extmem_adapter import DiskPages
from test_metal_lazy_host import assert_no_host


@pytest.mark.parametrize("on_host", [False, True])
@pytest.mark.parametrize("rows,batches", [(0, 1), (0, 3), (37, 1), (37, 3)])
@pytest.mark.parametrize("method", ["hist", "approx"])
def test_external_cache_empty_and_single(metal, tmp_path, on_host, rows, batches, method):
    x = np.arange(rows * 5, dtype=np.float32).reshape(rows, 5) / 7
    y = np.arange(rows, dtype=np.float32) % 7
    chunks = [x] if batches == 1 else [x[:0], x, x[:0]]
    labels = [y] if batches == 1 else [y[:0], y, y[:0]]
    it = DiskPages(tmp_path / "edge", chunks, on_host=on_host, label=labels)
    matrix = xgb.DMatrix(it, nthread=2)
    it.closed = True
    params = dict(device="mps", nthread=2, tree_method=method, max_bin=17, max_depth=2, base_score=.5)
    before = xgb.metal_info()
    model = xgb.train(params, matrix, 2)
    after = xgb.metal_info()
    assert_no_host(before)
    assert after["ellpack_io_write_pages"] - before["ellpack_io_write_pages"] == (0 if on_host else batches)
    if on_host:
        assert all(p.name.endswith(".row.page") for p in tmp_path.iterdir())
        host = after["ellpack_cached_host_bytes"] - before["ellpack_cached_host_bytes"]
        device = after["ellpack_cached_device_bytes"] - before["ellpack_cached_device_bytes"]
        assert (host == 0 and device > 0) if batches == 1 else (host > 0 and device == 0)
    actual = model.predict(matrix)
    cpu = xgb.Booster(model_file=model.save_raw())
    cpu.set_param(dict(device="cpu", nthread=2))
    np.testing.assert_array_equal(actual, cpu.predict(matrix))
    assert actual.shape[0] == rows
    if rows:
        assert actual.shape == (rows,)
    else:
        assert actual.size == 0  # Upstream empty predictions have shape (0, 0).
    # A repeated constant-Hessian fit reuses the initialized generation.
    xgb.train(params, matrix, 2)
    final = xgb.metal_info()
    for key in ["ellpack_cached_host_bytes", "ellpack_cached_device_bytes", "ellpack_io_write_pages"]:
        assert final[key] == after[key], key
    assert_no_host(before)
    del matrix, model, cpu
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("on_host", [False, True])
@pytest.mark.parametrize("minimum", [-1, 1, 4096])
def test_ordinary_rejects_page_concatenation(metal, tmp_path, on_host, minimum):
    it = DiskPages(tmp_path / "bad", [np.ones((3, 2), dtype=np.float32)], on_host=on_host)
    it.min_cache_page_bytes = minimum
    with pytest.raises(xgb.core.XGBoostError, match="Page concatenation is not supported"):
        xgb.DMatrix(it, nthread=2)
    gc.collect()
    assert not list(tmp_path.iterdir())
