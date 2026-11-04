"""Fresh matrices must assign their bins on Metal before GPU training."""
import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb


@pytest.mark.parametrize("layout", ["dense", "missing", "sparse"])
@pytest.mark.parametrize("bins", [3, 256, 257, 65537])
@pytest.mark.parametrize("categorical", [False, True])
def test_bin_assignment_matches_cpu(metal, layout, bins, categorical):
    rng = np.random.default_rng(149)
    x = rng.normal(size=(1027, 7)).astype(np.float32)
    y = x[:, 0] * 2 - x[:, 1] + x[:, 2] ** 2
    ft = ["q"] * x.shape[1]
    if categorical:
        x[:, 1] = rng.integers(0, 9, len(x)) * 2
        x[:, 5] = rng.integers(0, 5, len(x))
        ft[1] = ft[5] = "c"
    else:
        # Exercise bin comparisons around zero and FP32 subnormal cut values.
        x[:, 4] = (np.arange(len(x)) % 17 - 8) * np.nextafter(np.float32(0), np.float32(1))
    if layout == "missing":
        x[rng.random(x.shape) < .25] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < .65] = 0
        x = sp.csr_matrix(x)
    gpu_data = xgb.DMatrix(x, label=y, feature_types=ft, nthread=2)
    p = dict(max_bin=bins, max_depth=3, nthread=2, seed=19)
    before = xgb.metal_info()
    gpu = xgb.train(p | {"device": "mps"}, gpu_data, 3)
    after = xgb.metal_info()
    assert after["ellpack_csr_dispatches"] - before["ellpack_csr_dispatches"] == 1
    assert after["ellpack_csr_values"] - before["ellpack_csr_values"] == gpu_data.num_nonmissing()
    for key in ["binning_dispatches", "index_uploads", "ellpack_host_materializations", "ellpack_pack_input_bytes"]:
        assert after[key] == before[key], key
    assert after["ellpack_gpu_time_ns"] > before["ellpack_gpu_time_ns"]
    # Bin identical cut candidates on the CPU; also check the ordinary Metal
    # cuts against the CUDA sketch equations independently.
    from cuda_sketch_reference import assert_cuts
    assert_cuts(gpu_data, bins)
    cpu_data = xgb.QuantileDMatrix(x, label=y, feature_types=ft, nthread=2, max_bin=bins, ref=gpu_data)
    cpu = xgb.train(p | {"device": "cpu"}, cpu_data, 3)
    for actual, expected in zip(gpu_data.get_quantile_cut(), cpu_data.get_quantile_cut()):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(gpu.predict(gpu_data), cpu.predict(cpu_data), atol=2e-5, rtol=2e-5)
