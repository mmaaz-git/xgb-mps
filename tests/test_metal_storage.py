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
    assert after["histogram_integer_store_dispatches"] - before["histogram_integer_store_dispatches"] == after["histogram_dispatches"] - before["histogram_dispatches"]
    assert after["histogram_integer_subtraction_dispatches"] - before["histogram_integer_subtraction_dispatches"] == after["histogram_subtraction_dispatches"] - before["histogram_subtraction_dispatches"]
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


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("exponent", [-80, 0, 54, 100])
def test_integer_cache_preserves_cancellation(metal, external, exponent, tmp_path):
    """CUDA integer sums retain terms lost by page-wise binary64 addition."""
    big = np.float32(2.**exponent)
    small = np.float32(2.**(exponent - 54))
    x = np.zeros((4, 1), dtype=np.float32)
    grad = np.array([big, small, small, -big], dtype=np.float32)
    hess = np.array([0., 1., 1., 0.], dtype=np.float32)

    class Pages(xgb.DataIter):
        def __init__(self):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / "cancellation"))

        def reset(self):
            self.page = 0

        def next(self, input_data):
            if self.page == 2:
                return False
            begin, end = (0, 3) if self.page == 0 else (3, 4)
            input_data(data=x[begin:end], label=np.zeros(end - begin))
            self.page += 1
            return True

    data = xgb.DMatrix(Pages() if external else x, nthread=1)
    if not external:
        data.set_label(np.zeros(4))
    before = xgb.metal_info()
    model = xgb.train(dict(device="mps", tree_method="hist", max_bin=2, max_depth=1,
                           base_score=0., eta=1., reg_lambda=0., min_child_weight=0., nthread=1),
                      data, 1, obj=lambda p, d: (grad.copy(), hess.copy()))
    after = xgb.metal_info()
    assert after["histogram_integer_store_dispatches"] > before["histogram_integer_store_dispatches"]
    assert after["histogram_integer_sum_dispatches"] > before["histogram_integer_sum_dispatches"]
    if external:
        assert after["histogram_integer_accumulation_dispatches"] > before["histogram_integer_accumulation_dispatches"]
    # CUDA quantization represents these dyadic inputs exactly. The cancellation
    # leaves gradient 2*small and Hessian 2; lambda=0 and eta=1 give leaf=-small.
    expected = np.full(4, -small, dtype=np.float32)
    np.testing.assert_array_equal(model.predict(data), expected)
    np.testing.assert_array_equal(model.inplace_predict(x), expected)
    restored = xgb.Booster(model_file=model.save_raw())
    restored.set_param({"device": "mps"})
    np.testing.assert_array_equal(restored.predict(xgb.DMatrix(x)), expected)
