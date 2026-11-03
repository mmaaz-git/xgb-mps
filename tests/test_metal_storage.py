"""Exercise the actual Metal histogram cache across eviction, pages, and targets."""
import numpy as np
import pytest
import xgboost as xgb


def compare_storage(d, p):
    before = xgb.metal_info()
    cpu = xgb.train(p | {"device": "cpu"}, d, 4)
    assert xgb.metal_info()["histogram_conversion_dispatches"] == before["histogram_conversion_dispatches"]
    gpu = xgb.train(p | {"device": "mps"}, d, 4)
    after = xgb.metal_info()
    assert after["histogram_conversion_dispatches"] - before["histogram_conversion_dispatches"] == after["histogram_dispatches"] - before["histogram_dispatches"] > 0
    assert after["resident_split_dispatches"] - before["resident_split_dispatches"] == after["split_dispatches"] - before["split_dispatches"] > 0
    assert after["readback_time_ns"] == before["readback_time_ns"]
    if p["max_cached_hist_node"] == 64:
        assert after["histogram_subtraction_dispatches"] > before["histogram_subtraction_dispatches"]
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("cache", [1, 2, 7, 64])
@pytest.mark.parametrize("growth", ["depthwise", "lossguide"])
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_histogram_cache_growth_and_eviction(metal, cache, growth, strategy):
    rng = np.random.default_rng(703)
    x = rng.normal(size=(4097, 9)).astype(np.float32)
    y = np.column_stack([2 * x[:, 0] - x[:, 1], x[:, 2] ** 2 + x[:, 3]])
    x[rng.random(x.shape) < 0.2] = np.nan
    d = xgb.DMatrix(x, label=y)
    p = dict(nthread=2, tree_method="hist", max_depth=5, max_leaves=21,
             max_cached_hist_node=cache, grow_policy=growth,
             multi_strategy=strategy, max_bin=65)
    compare_storage(d, p)


@pytest.mark.parametrize("cache", [2, 64])
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_histogram_external_page_accumulation(metal, cache, strategy, tmp_path):
    rng = np.random.default_rng(971)
    x = rng.normal(size=(2057, 6)).astype(np.float32)
    y = np.column_stack([x[:, 0] * 2 - x[:, 1], x[:, 2] ** 2])
    x[rng.random(x.shape) < 0.4] = np.nan

    class Pages(xgb.DataIter):
        def __init__(self):
            self.offset = 0
            super().__init__(cache_prefix=str(tmp_path / "cache"))

        def reset(self):
            self.offset = 0

        def next(self, input_data):
            if self.offset == len(x):
                return False
            end = min(len(x), self.offset + 257)
            input_data(data=x[self.offset:end], label=y[self.offset:end])
            self.offset = end
            return True

    d = xgb.ExtMemQuantileDMatrix(Pages(), max_bin=33)
    p = dict(nthread=2, tree_method="hist", max_depth=4, max_bin=33,
             max_cached_hist_node=cache, multi_strategy=strategy)
    compare_storage(d, p)
