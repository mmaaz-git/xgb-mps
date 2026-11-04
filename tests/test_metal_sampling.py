"""CUDA-source sampling decisions and replay, with CPU downstream validation."""
import numpy as np
import pytest
import xgboost as xgb
from cuda_uniform_reference import UniformSquaredError
from cuda_gradient_reference import GradientSquaredError
from xgboost.objective import TreeObjective
from cuda_split_reference import RecordedSquaredError, audit_recorded


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
             max_depth=3, max_bin=33, subsample=subsample, seed=21, base_score=0,
             seed_per_iteration=True)
    before = xgb.metal_info()["sampling_dispatches"]
    cpu = xgb.train(p | {"device": "cpu", "subsample": 1.}, d, 4,
                    obj=UniformSquaredError(p))
    assert xgb.metal_info()["sampling_dispatches"] == before
    recorded = RecordedSquaredError() if strategy == "one_output_per_tree" else None
    gpu = xgb.train(p | {"device": "mps"}, d, 4, obj=recorded)
    delta = xgb.metal_info()["sampling_dispatches"] - before
    assert (delta > 0) == (subsample < 1)
    if recorded is not None:
        audit_recorded(gpu,d,x,recorded,p)
    else:
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
             sampling_method=sampling_method, seed_per_iteration=True)
    cpu_params = p | {"device": "cpu", "subsample": 1.}
    reference = UniformSquaredError if sampling_method == "uniform" else GradientSquaredError
    cpu_objective = reference(p, Reduced())
    cpu = xgb.train(cpu_params, d, 4, obj=cpu_objective)
    before = xgb.metal_info()["sampling_dispatches"]
    gpu = xgb.train(p | {"device": "mps"}, d, 4, obj=Reduced())
    # Each tree samples the split gradient and replays that mask on value gradients.
    # The shared state counts integer sampling once; replay shares saved decisions.
    assert xgb.metal_info()["sampling_dispatches"] - before == 4
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
             sampling_method="gradient_based", seed_per_iteration=True)
    before = xgb.metal_info()["gradient_sampling_dispatches"]
    cpu = xgb.train(p | {"device": "cpu", "subsample": 1.}, d, 4,
                    obj=GradientSquaredError(p))
    assert xgb.metal_info()["gradient_sampling_dispatches"] == before
    recorded = RecordedSquaredError() if strategy == "one_output_per_tree" else None
    gpu = xgb.train(p | {"device": "mps"}, d, 4, obj=recorded)
    after = xgb.metal_info()["gradient_sampling_dispatches"]
    assert (after > before) == (subsample < 1)
    if recorded is not None:
        audit_recorded(gpu,d,x,recorded,p)
    else:
        np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)
    again = xgb.train(p | {"device": "mps"}, d, 4)
    np.testing.assert_array_equal(gpu.predict(d), again.predict(d))


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.parametrize("trees", [2, 3])
@pytest.mark.parametrize("targets", [1, 3])
@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
def test_forests_use_fresh_original_state(metal, tree_method, trees, targets, sampling_method):
    from cuda_uniform_reference import weights

    rows, rounds, seed = 257, 2, 113
    x = np.zeros((rows, 2), dtype=np.float32)
    grad = (np.arange(rows, dtype=np.float32) % 11 - 5)[:, None] / 8
    grad = grad * np.arange(1, targets + 1, dtype=np.float32)[None, :]
    hess = np.broadcast_to((np.arange(rows, dtype=np.float32) % 5 + 1)[:, None] / 4,
                          grad.shape).copy()
    data = xgb.DMatrix(x, label=np.zeros(grad.shape), nthread=1)
    params = dict(device="mps", tree_method=tree_method, num_parallel_tree=trees,
                  multi_strategy="one_output_per_tree", base_score=0., eta=.3,
                  subsample=.6, seed=seed, max_depth=1, max_bin=2, reg_lambda=1., nthread=1,
                  sampling_method=sampling_method)
    before = xgb.metal_info()
    model = xgb.train(params, data, rounds, obj=lambda p, d: (grad.copy(), hess.copy()))
    after = xgb.metal_info()
    counter = ("cuda_uniform_sampling_dispatches" if sampling_method == "uniform"
               else "cuda_gradient_sampling_dispatches")
    assert after[counter] - before[counter] == rounds * trees * targets
    assert after["quantization_dispatches"] - before["quantization_dispatches"] == rounds * trees * targets
    rng = np.random.RandomState(seed)
    expected = np.zeros(targets, dtype=np.float32)
    learning_rate = np.float32(np.float32(.3) / np.float32(trees))
    for _ in range(rounds):
        for target in range(targets):
            for _ in range(trees):
                row_seed = int(rng.randint(0, 1 << 32, dtype=np.uint32))
                if sampling_method == "uniform":
                    keep = weights(row_seed, rows) <= np.float32(.6)
                    # Dyadic inputs are represented exactly by the original quantizer.
                    g = grad[keep, target].sum(dtype=np.float64)
                    h = hess[keep, target].sum(dtype=np.float64)
                else:
                    from cuda_gradient_reference import sample
                    _, _, roots = sample(grad[:, target], hess[:, target], np.float32(.6), row_seed)
                    g, h = roots[0][0], roots[1][0]
                expected[target] += np.float32(-g / (h + 1.)) * learning_rate
    expected = np.tile(expected, (rows, 1))
    if targets == 1:
        expected = expected[:, 0]
    np.testing.assert_array_equal(model.predict(data), expected)
    # Original custom gradients remain intact across trees/rounds and both
    # histogram methods, regardless of which rows a previous tree rejected.
    repeated = xgb.train(params, data, rounds, obj=lambda p, d: (grad.copy(), hess.copy()))
    np.testing.assert_array_equal(repeated.predict(data), expected)
