"""Reduced-gradient training must refit final vector leaves on Metal."""
import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb
from cuda_uniform_reference import UniformSquaredError
from cuda_gradient_reference import GradientSquaredError
from xgboost.objective import TreeObjective


class Reduced(TreeObjective):
    def __call__(self, iteration, prediction, data):
        label = data.get_label().reshape(prediction.shape)
        weights = data.get_weight()
        weight = weights[:, None] if weights.size else 1.
        return (prediction - label) * weight, np.ones_like(prediction) * weight

    def split_grad(self, iteration, grad, hess):
        return np.mean(grad, axis=1), np.mean(hess, axis=1)


def dataset(layout, tmp_path):
    rng = np.random.default_rng(183)
    x = rng.normal(size=(1031, 7)).astype(np.float32)
    y = np.column_stack([x[:, 0] + x[:, 1], x[:, 1] ** 2, -2 * x[:, 2]])
    weights = rng.uniform(.1, 3, len(x)).astype(np.float32)
    weights[::19] = 0
    if layout in {"missing", "external"}:
        x[rng.random(x.shape) < .25] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < .65] = 0
        x = sp.csr_matrix(x)
    if layout != "external":
        return xgb.DMatrix(x, label=y, weight=weights, nthread=2)

    class Pages(xgb.DataIter):
        def __init__(self):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / "leaf-pages"))

        def reset(self):
            self.page = 0

        def next(self, input_data):
            if self.page == 4:
                return False
            begin, end = self.page * 263, min((self.page + 1) * 263, len(x))
            input_data(data=x[begin:end], label=y[begin:end], weight=weights[begin:end])
            self.page += 1
            return True
    return xgb.DMatrix(Pages(), nthread=2)


@pytest.mark.parametrize("layout", ["dense", "missing", "sparse", "external"])
@pytest.mark.parametrize("sampling,subsample", [("uniform", 0.), ("uniform", .6),
    ("uniform", 1.), ("gradient_based", .6)])
@pytest.mark.parametrize("regularization", ["none", "penalties", "clipped", "root_only"])
def test_gpu_reduced_leaf_fitting(metal, layout, sampling, subsample, regularization, tmp_path):
    data = dataset(layout, tmp_path)
    params = dict(nthread=2, max_depth=3, max_bin=33, base_score=[0., 0., 0.], eta=.3,
                  multi_strategy="multi_output_tree", subsample=subsample,
                  sampling_method=sampling, seed=71, debug_synchronize=True, seed_per_iteration=True)
    if regularization == "none":
        params.update(reg_lambda=0.)
    elif regularization == "penalties":
        params.update(reg_lambda=3., reg_alpha=.7)
    elif regularization == "clipped":
        params.update(reg_alpha=.3, max_delta_step=.125)
    elif regularization == "root_only":
        params.update(min_child_weight=1e9, reg_lambda=.1)
    cpu_params = params | {"device": "cpu", "subsample": 1.}
    def cpu_objective(rounds_before=0):
        reference = UniformSquaredError if sampling == "uniform" else GradientSquaredError
        return reference(params, Reduced(), rounds_before=rounds_before)
    cpu = xgb.train(cpu_params, data, 3, obj=cpu_objective())
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data, 3, obj=Reduced())
    after = xgb.metal_info()
    # Split copy, value copy and the original value-bound column layout per round.
    assert after["gradient_copy_dispatches"] - before["gradient_copy_dispatches"] == 9
    pages = 4 if layout == "external" else 1
    assert after["leaf_sum_init_dispatches"] - before["leaf_sum_init_dispatches"] == 3
    assert after["leaf_sum_dispatches"] - before["leaf_sum_dispatches"] == 3 * pages
    assert after["leaf_integer_sum_dispatches"] - before["leaf_integer_sum_dispatches"] == 3 * pages
    assert after["leaf_weight_dispatches"] - before["leaf_weight_dispatches"] == 3
    assert after["leaf_fitting_gpu_time_ns"] > before["leaf_fitting_gpu_time_ns"]
    np.testing.assert_allclose(gpu.predict(data), cpu.predict(data), atol=2e-5, rtol=2e-5)
    # Both CPU and CUDA reject monotonic constraints for reduced gradients; the
    # low-level weight API nevertheless tests the shared bound formula natively.
    # Reload both sides from the same format so RNG configuration has the same
    # lifecycle; a live booster and a saved model do not carry identical state.
    cpu_raw, gpu_raw = cpu.save_raw(), gpu.save_raw()
    cpu = xgb.train(cpu_params, data, 1, obj=cpu_objective(3), xgb_model=cpu_raw)
    gpu = xgb.train(params | {"device": "mps"}, data, 1, obj=Reduced(), xgb_model=gpu_raw)
    np.testing.assert_allclose(gpu.predict(data), cpu.predict(data), atol=2e-5, rtol=2e-5)


def test_reduced_monotone_error_matches_cpu(metal, tmp_path):
    data = dataset("dense", tmp_path)
    for device in ["cpu", "mps"]:
        with pytest.raises(xgb.core.XGBoostError, match="Monotonic constraints are not supported"):
            xgb.train(dict(device=device, nthread=2, multi_strategy="multi_output_tree",
                           monotone_constraints="(1,0,0,0,0,0,0)"), data, 1, obj=Reduced())


def test_regular_vector_training_skips_refitting(metal, tmp_path):
    data = dataset("dense", tmp_path)
    before = xgb.metal_info()["leaf_weight_dispatches"]
    xgb.train(dict(device="mps", nthread=2, multi_strategy="multi_output_tree"), data, 2)
    assert xgb.metal_info()["leaf_weight_dispatches"] == before


def integer_leaf_data(x, targets, layout, tmp_path):
    if layout == "missing":
        x[:, 0], x[1, 1] = np.nan, np.nan
    elif layout == "empty":
        x[:] = np.nan
    label = np.zeros((len(x), targets), dtype=np.float32)
    if layout != "external":
        return xgb.DMatrix(x, label=label, nthread=1)

    class Pages(xgb.DataIter):
        def __init__(self):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / "integer-leaf-pages"))
        def reset(self):
            self.page = 0
        def next(self, input_data):
            if self.page == 2:
                return False
            begin, end = (0, 3) if self.page == 0 else (3, 4)
            input_data(data=x[begin:end], label=label[begin:end])
            self.page += 1
            return True
    return xgb.DMatrix(Pages(), nthread=1)


@pytest.mark.parametrize("layout", ["dense", "missing", "empty", "external"])
@pytest.mark.parametrize("exponent", [-80, 0, 54, 100])
def test_reduced_leaf_integer_cancellation(metal, layout, exponent, tmp_path):
    """The first page's rounded sum loses small terms; integer state must keep them."""
    big, small = np.float32(2.**exponent), np.float32(2.**(exponent-54))
    multipliers = np.array([1., 2., 4.], dtype=np.float32)
    grad = np.array([big, small, small, -big], dtype=np.float32)[:, None] * multipliers
    hess = np.tile(np.array([0., 1., 1., 0.], dtype=np.float32)[:, None], (1, 3))
    class Objective(TreeObjective):
        def __call__(self, iteration, pred, data):
            return grad.copy(), hess.copy()
        def split_grad(self, iteration, g, h):
            return np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    x = np.zeros((4, 2), dtype=np.float32)
    data = integer_leaf_data(x, 3, layout, tmp_path)
    before = xgb.metal_info()
    model = xgb.train(dict(device="mps", multi_strategy="multi_output_tree", nthread=1,
                          base_score=0., eta=1., reg_lambda=0., min_child_weight=0.,
                          max_depth=1, max_bin=2), data, 1, obj=Objective())
    after = xgb.metal_info()
    assert after["leaf_integer_sum_dispatches"] - before["leaf_integer_sum_dispatches"] == (2 if layout == "external" else 1)
    assert after["gradient_bound_dispatches"] - before["gradient_bound_dispatches"] == 4
    assert after["quantization_dispatches"] - before["quantization_dispatches"] == 1
    expected = np.tile(-small * multipliers, (4, 1))
    np.testing.assert_array_equal(model.predict(data), expected)
    np.testing.assert_array_equal(model.inplace_predict(x), expected)
    restored = xgb.Booster(model_file=model.save_raw())
    np.testing.assert_array_equal(restored.predict(data), expected)


@pytest.mark.parametrize("layout", ["dense", "missing", "empty", "external"])
@pytest.mark.parametrize("sampling", ["uniform", "gradient_based"])
def test_reduced_leaf_uses_original_value_scales(metal, layout, sampling, tmp_path):
    from cuda_uniform_reference import weights
    from cuda_gradient_reference import exponents, sample

    split_g = np.array([-.125, .5, 1.5, 4.], dtype=np.float32)
    split_h = np.ones(4, dtype=np.float32)
    # Sampling is driven by split gradients, independently of the value-gradient
    # outlier. Reject that outlier, retaining at least two tiny-curvature rows.
    for seed in range(1000):
        row_seed = int(np.random.RandomState(seed).randint(0, 1 << 32, dtype=np.uint32))
        if sampling == "uniform":
            factors = (weights(row_seed, 4) <= np.float32(.6)).astype(np.float32)
        else:
            _, factors, _ = sample(split_g, split_h, np.float32(.6), row_seed)
        if factors[0] == 0 and np.count_nonzero(factors[1:]) >= 2:
            break
    else:
        raise AssertionError("No reference seed found")
    multipliers = np.array([1., 2., 4.], dtype=np.float32)
    grad = np.array([.33333334, 1.00000012, -1.2345678, .7654321], dtype=np.float32)[:, None] * multipliers
    hess = np.tile(np.array([2.**65, 2.**-149, 2.**-100, 2.**-50], dtype=np.float32)[:, None], (1, 3)) * multipliers
    # CUDA value path: original bounds, FLOAT replay, then fixed-point conversion.
    ge, he = exponents(grad), exponents(hess)
    sampled_g, sampled_h = grad * factors[:, None], hess * factors[:, None]
    qg = np.ldexp(sampled_g.astype(np.float64), ge).astype(np.int64)
    qh = np.ldexp(sampled_h.astype(np.float64), he).astype(np.int64)
    qh[(sampled_h > 0) & (qh == 0)] = 1
    total_g = np.ldexp(qg.sum(axis=0).astype(np.float64), -ge)
    total_h = np.ldexp(qh.sum(axis=0).astype(np.float64), -he)
    assert np.all(total_h == np.count_nonzero(factors) * 16. * multipliers)
    expected = np.tile((-total_g / (total_h + 1.)).astype(np.float32), (4, 1))
    class Objective(TreeObjective):
        def __call__(self, iteration, pred, data):
            return grad.copy(), hess.copy()
        def split_grad(self, iteration, g, h):
            return split_g.copy(), split_h.copy()
    x = np.zeros((4, 2), dtype=np.float32)
    data = integer_leaf_data(x, 3, layout, tmp_path)
    before = xgb.metal_info()
    model = xgb.train(dict(device="mps", multi_strategy="multi_output_tree", nthread=1,
                          seed=seed, subsample=.6, sampling_method=sampling,
                          # Keep a single leaf even when missingness can split rows.
                          gamma=1e30, base_score=0., eta=1., reg_lambda=1., min_child_weight=0.,
                          max_depth=1, max_bin=2), data, 1, obj=Objective())
    after = xgb.metal_info()
    assert after["gradient_bound_dispatches"] - before["gradient_bound_dispatches"] == 4
    assert after["leaf_integer_sum_dispatches"] - before["leaf_integer_sum_dispatches"] == (2 if layout == "external" else 1)
    np.testing.assert_array_equal(model.predict(data), expected)
    np.testing.assert_array_equal(model.inplace_predict(x), expected)
