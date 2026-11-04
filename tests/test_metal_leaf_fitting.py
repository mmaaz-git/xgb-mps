"""Reduced-gradient training must refit final vector leaves on Metal."""
import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb
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
                  sampling_method=sampling, seed=71, debug_synchronize=True)
    if regularization == "none":
        params.update(reg_lambda=0.)
    elif regularization == "penalties":
        params.update(reg_lambda=3., reg_alpha=.7)
    elif regularization == "clipped":
        params.update(reg_alpha=.3, max_delta_step=.125)
    elif regularization == "root_only":
        params.update(min_child_weight=1e9, reg_lambda=.1)
    cpu = xgb.train(params | {"device": "cpu"}, data, 3, obj=Reduced())
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data, 3, obj=Reduced())
    after = xgb.metal_info()
    pages = 4 if layout == "external" else 1
    assert after["leaf_sum_init_dispatches"] - before["leaf_sum_init_dispatches"] == 3
    assert after["leaf_sum_dispatches"] - before["leaf_sum_dispatches"] == 3 * pages
    assert after["leaf_weight_dispatches"] - before["leaf_weight_dispatches"] == 3
    assert after["leaf_fitting_gpu_time_ns"] > before["leaf_fitting_gpu_time_ns"]
    np.testing.assert_allclose(gpu.predict(data), cpu.predict(data), atol=2e-5, rtol=2e-5)
    # Both CPU and CUDA reject monotonic constraints for reduced gradients; the
    # low-level weight API nevertheless tests the shared bound formula natively.
    # Reload both sides from the same format so RNG configuration has the same
    # lifecycle; a live booster and a saved model do not carry identical state.
    cpu_raw, gpu_raw = cpu.save_raw(), gpu.save_raw()
    cpu = xgb.train(params | {"device": "cpu"}, data, 1, obj=Reduced(), xgb_model=cpu_raw)
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
