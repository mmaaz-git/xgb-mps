"""Require resident Metal training buffers and real row partition dispatches."""
import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb

@pytest.mark.parametrize("layout", ["dense", "missing", "sparse"])
@pytest.mark.parametrize("growth", ["depthwise", "lossguide"])
def test_resident_rows_and_page_cache(metal, layout, growth):
    rng = np.random.default_rng(701)
    x = rng.normal(size=(8193, 11)).astype(np.float32)
    y = x[:, 0] * 2 - x[:, 1] + x[:, 2] ** 2
    if layout == "missing":
        x[rng.random(x.shape) < 0.4] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < 0.85] = 0
        x = sp.csr_matrix(x)
    d = xgb.DMatrix(x, label=y)
    p = dict(tree_method="hist", max_depth=5, max_leaves=17, grow_policy=growth,
             max_bin=33, nthread=2, subsample=0.7, seed=23)
    cpu = xgb.train(p | {"device": "cpu"}, d, 7)
    before = xgb.metal_info()
    gpu = xgb.train(p | {"device": "mps"}, d, 7)
    after = xgb.metal_info()
    assert after["index_uploads"] - before["index_uploads"] == 1
    assert after["row_init_dispatches"] - before["row_init_dispatches"] == 7
    assert after["leaf_partition_dispatches"] - before["leaf_partition_dispatches"] == 7
    assert after["partition_dispatches"] > before["partition_dispatches"]
    assert after["split_dispatches"] > before["split_dispatches"]
    assert after["resident_histogram_dispatches"] - before["resident_histogram_dispatches"] == after["histogram_dispatches"] - before["histogram_dispatches"]
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)
    # Reuse the source page in a different booster without reuploading immutable bins.
    before = xgb.metal_info()["index_uploads"]
    fresh = xgb.train(p | {"device": "mps"}, d, 7)
    assert xgb.metal_info()["index_uploads"] == before
    np.testing.assert_array_equal(fresh.predict(d), gpu.predict(d))


def test_quantized_page_invalidation_and_device_switch(metal):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4097, 8)).astype(np.float32)
    d = xgb.DMatrix(x, label=x[:, 0] - x[:, 1])
    for bins in [16, 257, 64, 16]:
        p = dict(tree_method="hist", max_bin=bins, max_depth=3, nthread=2)
        expected = xgb.train(p | {"device": "cpu"}, d, 3).predict(d)
        before = xgb.metal_info()["index_uploads"]
        model = xgb.train(p | {"device": "mps"}, d, 3)
        assert xgb.metal_info()["index_uploads"] - before == 1
        np.testing.assert_allclose(model.predict(d), expected, rtol=2e-5, atol=2e-5)
        model.set_param({"device": "cpu"})
        np.testing.assert_allclose(model.predict(d), expected, rtol=2e-5, atol=2e-5)

@pytest.mark.parametrize("size", [1, 255, 256, 257, 511, 512, 513, 65537])
def test_partition_block_boundaries(metal, size):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 7, size=(size, 3)).astype(np.float32)
    y = x[:, 0] + x[:, 1] * 2
    x[::7, 0] = np.nan
    d = xgb.DMatrix(x, label=y)
    p = dict(nthread=2, max_depth=4, min_child_weight=0, max_bin=8)
    expected = xgb.train(p | {"device": "cpu"}, d, 3).predict(d)
    before = xgb.metal_info()["leaf_partition_dispatches"]
    model = xgb.train(p | {"device": "mps"}, d, 3)
    assert xgb.metal_info()["leaf_partition_dispatches"] - before == 3
    np.testing.assert_allclose(model.predict(d), expected, rtol=2e-5, atol=2e-5)
