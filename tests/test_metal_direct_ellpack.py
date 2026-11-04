"""Direct compressed construction: typed input, real window seams and error guards."""
import numpy as np
import pytest
import xgboost as xgb

from cuda_sketch_reference import assert_same_model_cpu
from test_metal_host_cache import HostPages


def assert_direct(before, after, nnz):
    assert after["ellpack_adapter_values"] - before["ellpack_adapter_values"] == nnz
    assert after["ellpack_adapter_dispatches"] > before["ellpack_adapter_dispatches"]
    for key in ["adapter_index_dispatches", "binning_dispatches", "ellpack_pack_input_bytes",
                "adapter_debug_readback_bytes", "ellpack_debug_readback_bytes"]:
        assert after[key] == before[key], key


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.int16, np.int32,
                                   np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_direct_typed_strided_input(metal, tmp_path, dtype):
    x = (np.arange(93).reshape(31, 3) % 19).astype(dtype)[::-1, ::-1]
    y = np.arange(31, dtype=np.float32) % 5
    it = HostPages(tmp_path / "typed", [x[:13], x[13:]], 0, label=[y[:13], y[13:]])
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(it, device="mps", max_bin=8, nthread=2, cache_host_ratio=.23)
    assert_direct(before, xgb.metal_info(), x.size)
    cpu = xgb.QuantileDMatrix(x, y, ref=q, max_bin=8, nthread=2)
    model = xgb.train(dict(device="mps", max_bin=8, max_depth=2, nthread=2), q, 3)
    np.testing.assert_array_equal(model.predict(q), model.predict(cpu))
    assert_same_model_cpu(model, q)
    assert not list(tmp_path.iterdir())


def test_direct_sparse_real_window_boundary(metal, tmp_path):
    # 2^26 % 7 == 4: the final valid value of the boundary row belongs to the
    # second window. A reference avoids a large sketch in this construction test.
    rows = (1 << 26) // 7 + 3
    pattern = np.array([1, np.nan, 2, np.nan, 3, np.nan, np.nan], dtype=np.float32)
    ref = xgb.QuantileDMatrix(pattern.reshape(1, 7), device="cpu", max_bin=8, nthread=2)
    x = np.broadcast_to(pattern, (rows, 7))
    it = HostPages(tmp_path / "boundary", [x], 0)
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(it, ref=ref, device="mps", max_bin=8, nthread=2)
    after = xgb.metal_info()
    assert_direct(before, after, rows * 3)
    assert q.num_row() == rows and q.num_nonmissing() == rows * 3
    assert after["ellpack_adapter_dispatches"] - before["ellpack_adapter_dispatches"] == 3
    assert after["cuda_sketch_batches"] == before["cuda_sketch_batches"]
    for actual, expected in zip(q.get_quantile_cut(), ref.get_quantile_cut()):
        np.testing.assert_array_equal(actual, expected)
    assert not list(tmp_path.iterdir())


def test_direct_rejects_changed_row_width(metal, tmp_path):
    class Changing(HostPages):
        def __init__(self):
            self.pass_number = 0
            super().__init__(tmp_path / "changed", [], 0)

        def reset(self):
            self.pass_number += 1
            self.page = 0

        def next(self, input_data):
            if self.page:
                return False
            x = np.array([[1, 1, np.nan, np.nan], [np.nan, np.nan, 1, 1]], dtype=np.float32)
            if self.pass_number > 1:
                x[0, 2], x[1, 2] = 1, np.nan  # Same total nnz, different maximum row width.
            input_data(data=x)
            self.page += 1
            return True

    with pytest.raises(xgb.core.XGBoostError, match="row width changed"):
        xgb.ExtMemQuantileDMatrix(Changing(), device="mps", nthread=2)
    assert not list(tmp_path.iterdir())


def test_direct_infinity_and_missing_sentinel(metal, tmp_path):
    x = np.array([[1, np.inf], [2, 3]], dtype=np.float32)
    with pytest.raises(xgb.core.XGBoostError, match="inf"):
        xgb.ExtMemQuantileDMatrix(HostPages(tmp_path / "bad", [x], 0), device="mps", nthread=2)
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(HostPages(tmp_path / "sentinel", [x], 0), missing=np.inf,
                                  device="mps", nthread=2)
    assert q.num_nonmissing() == 3
    assert_direct(before, xgb.metal_info(), 3)
    assert not list(tmp_path.iterdir())
