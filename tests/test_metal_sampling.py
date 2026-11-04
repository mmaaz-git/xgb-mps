"""GPU sampling retains CPU decisions, rescaling and reduced-gradient replay."""
import numpy as np
import pytest
import xgboost as xgb
from xgboost.objective import TreeObjective


@pytest.mark.parametrize("subsample", [0.0, 0.1, 0.7, 1.0])
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("threads", [1, 3])
def test_uniform_gpu_sampling(metal, subsample, strategy, threads):
    rng = np.random.default_rng(393)
    x = rng.normal(size=(1027, 7)).astype(np.float32)
    y = np.column_stack([x[:, 0] * 2 - x[:, 1], x[:, 2] ** 2 + x[:, 3]])
    x[rng.random(x.shape) < 0.1] = np.nan
    d = xgb.DMatrix(x, label=y)
    p = dict(tree_method="hist", multi_strategy=strategy, nthread=threads,
             max_depth=3, max_bin=33, subsample=subsample, seed=21, base_score=0)
    before = xgb.metal_info()["sampling_dispatches"]
    cpu = xgb.train(p | {"device": "cpu"}, d, 4)
    assert xgb.metal_info()["sampling_dispatches"] == before
    gpu = xgb.train(p | {"device": "mps"}, d, 4)
    delta = xgb.metal_info()["sampling_dispatches"] - before
    assert (delta > 0) == (subsample < 1)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)
    # A fresh booster with the same seed must replay the same sampling decisions.
    again = xgb.train(p | {"device": "mps"}, d, 4)
    np.testing.assert_array_equal(gpu.predict(d), again.predict(d))


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
def test_replay_sampling_for_reduced_gradient_leaves(metal, threads, sampling_method):
    class Reduced(TreeObjective):
        def __call__(self, iteration, prediction, data):
            label = data.get_label().reshape(prediction.shape)
            return prediction - label, np.ones_like(prediction)

        def split_grad(self, iteration, grad, hess):
            return np.mean(grad, axis=1), np.mean(hess, axis=1)

    rng = np.random.default_rng(777)
    x = rng.normal(size=(1027, 7)).astype(np.float32)
    y = np.column_stack([x[:, 0] * 2, x[:, 1] - x[:, 2], x[:, 3] ** 2])
    d = xgb.DMatrix(x, label=y)
    p = dict(tree_method="hist", multi_strategy="multi_output_tree", nthread=threads,
             max_depth=4, max_bin=33, subsample=0.6, seed=77, base_score=0,
             sampling_method=sampling_method)
    cpu = xgb.train(p | {"device": "cpu"}, d, 4, obj=Reduced())
    before = xgb.metal_info()["sampling_dispatches"]
    gpu = xgb.train(p | {"device": "mps"}, d, 4, obj=Reduced())
    # Each tree samples the split gradient and replays that mask on value gradients.
    assert xgb.metal_info()["sampling_dispatches"] - before == 8
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("subsample", [0.0, 0.1, 0.7, 1.0])
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("threads", [1, 3])
def test_gradient_gpu_sampling(metal, subsample, strategy, threads):
    rng = np.random.default_rng(1123)
    x = rng.normal(size=(1027, 7)).astype(np.float32)
    y = np.column_stack([x[:, 0] * 2 - x[:, 1], x[:, 2] ** 2 + x[:, 3]])
    x[rng.random(x.shape) < 0.1] = np.nan
    d = xgb.DMatrix(x, label=y)
    p = dict(tree_method="hist", multi_strategy=strategy, nthread=threads,
             max_depth=3, max_bin=33, subsample=subsample, seed=21, base_score=0,
             sampling_method="gradient_based")
    before = xgb.metal_info()["gradient_sampling_dispatches"]
    cpu = xgb.train(p | {"device": "cpu"}, d, 4)
    assert xgb.metal_info()["gradient_sampling_dispatches"] == before
    gpu = xgb.train(p | {"device": "mps"}, d, 4)
    after = xgb.metal_info()["gradient_sampling_dispatches"]
    assert (after > before) == (subsample < 1)
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)
    again = xgb.train(p | {"device": "mps"}, d, 4)
    np.testing.assert_array_equal(gpu.predict(d), again.predict(d))
