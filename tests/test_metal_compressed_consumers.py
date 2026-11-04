"""Compressed cache consumers: training, traversal, SHAP and model persistence."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from test_metal_host_cache import HostPages


@pytest.mark.parametrize("layout", ["dense", "relative", "sparse", "categorical"])
@pytest.mark.parametrize("ratio", [0., .25, 1.])
def test_compressed_consumers(metal, tmp_path, layout, ratio):
    rng = np.random.default_rng(1601)
    x = rng.normal(size=(513, 5)).astype(np.float32)
    y = x[:, 0] * x[:, 1] - x[:, 2] + x[:, 4]
    if layout != "dense":
        x[rng.random(x.shape) < (.7 if layout == "sparse" else .2)] = np.nan
        if layout == "sparse":
            x[:, 4] = np.nan
        else:
            x[0] = np.arange(5)
    if layout == "categorical":
        x = pd.DataFrame(x, columns=list("abcde"))
        x.b = pd.Categorical.from_codes(np.arange(513) % 7, categories=list("abcdefg"))
        x.loc[::17, "b"] = np.nan
    split = [0, 0, 131, 259, 513, 513]
    chunks = [x.iloc[b:e] if isinstance(x, pd.DataFrame) else x[b:e]
              for b, e in zip(split[:-1], split[1:])]
    labels = [y[b:e] for b, e in zip(split[:-1], split[1:])]
    q = xgb.ExtMemQuantileDMatrix(HostPages(tmp_path / "cache", chunks, 0, label=labels),
                                  device="mps", nthread=2, max_bin=17, cache_host_ratio=ratio)
    before = xgb.metal_info()
    model = xgb.train(dict(device="mps", nthread=2, max_bin=17, max_depth=3), q, 4)
    after = xgb.metal_info()
    assert after["ellpack_histogram_dispatches"] > before["ellpack_histogram_dispatches"]
    assert after["ellpack_partition_dispatches"] > before["ellpack_partition_dispatches"]
    assert after["index_uploads"] == before["index_uploads"]
    assert after["index_gpu_time_ns"] == before["index_gpu_time_ns"]
    for option in ["output_margin", "pred_leaf", "pred_contribs", "pred_interactions"]:
        # Reloading gives a fresh prediction cache and checks saved-model integration.
        cpu = xgb.Booster(model_file=model.save_raw())
        cpu.set_param(dict(device="cpu", nthread=2))
        gpu = xgb.Booster(model_file=model.save_raw())
        gpu.set_param(dict(device="mps", nthread=2))
        kwargs = {option: True, "strict_shape": True}
        expected = cpu.predict(q, **kwargs)
        before = xgb.metal_info()
        actual = gpu.predict(q, **kwargs)
        after = xgb.metal_info()
        assert after["ellpack_input_dispatches"] > before["ellpack_input_dispatches"]
        assert after["index_uploads"] == before["index_uploads"]
        assert after["ellpack_debug_readback_bytes"] == before["ellpack_debug_readback_bytes"]
        if option in {"pred_leaf", "output_margin"}:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert not list(tmp_path.iterdir())
