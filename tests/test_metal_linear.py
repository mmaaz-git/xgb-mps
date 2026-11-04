"""CUDA-capable linear coordinate descent now dispatches Metal operations."""
import json

import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb

SELECTORS = ["cyclic", "shuffle", "random", "greedy", "thrifty"]


def data_and_params(layout, kind):
    rng = np.random.default_rng(731)
    x = rng.normal(size=(257, 7)).astype(np.float32)
    y = x[:, 0] * 2 - x[:, 1] + .3 * x[:, 2]
    params = {}
    if kind == "binary":
        y = (y > 0).astype(np.float32)
        params.update(objective="binary:logistic", base_score=.5)
    elif kind == "multiclass":
        y = np.digitize(y, [-1, 1])
        params.update(objective="multi:softprob", num_class=3, base_score=[.5] * 3)
    elif kind == "multitarget":
        y = np.column_stack([y, -y, y / 2])
        params.update(base_score=[0.] * 3)
    else:
        params.update(base_score=0.)
    if layout == "missing":
        x[rng.random(x.shape) < .25] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < .7] = 0
        x = sp.csr_matrix(x)
    weights = rng.uniform(.1, 3, len(y)).astype(np.float32)
    weights[::19] = 0
    return xgb.DMatrix(x, label=y, weight=weights, nthread=2), params


def check_models(cpu, gpu, data):
    for args in [{}, {"output_margin": True}, {"pred_contribs": True}]:
        np.testing.assert_allclose(gpu.predict(data, **args), cpu.predict(data, **args),
                                   rtol=2e-5, atol=2e-5)
    weights = [json.loads(m.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["weights"]
               for m in [cpu, gpu]]
    np.testing.assert_allclose(weights[1], weights[0], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("layout", ["dense", "missing", "sparse"])
@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass", "multitarget"])
@pytest.mark.parametrize("penalized", [False, True])
def test_metal_coordinate_descent(metal, selector, layout, kind, penalized):
    data, objective = data_and_params(layout, kind)
    params = dict(booster="gblinear", updater="coord_descent", feature_selector=selector,
                  nthread=2, eta=.3, seed=91, seed_per_iteration=True, **objective)
    if penalized:
        params.update(reg_alpha=.03, reg_lambda=.2)
    cpu = xgb.train(params | {"device": "cpu"}, data, 5)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data, 5)
    after = xgb.metal_info()
    assert after["linear_gradient_dispatches"] > before["linear_gradient_dispatches"]
    assert after["linear_residual_dispatches"] > before["linear_residual_dispatches"]
    assert after["linear_gpu_time_ns"] > before["linear_gpu_time_ns"]
    check_models(cpu, gpu, data)
    cpu_raw, gpu_raw = cpu.save_raw(), gpu.save_raw()
    cpu = xgb.train(params | {"device": "cpu"}, data, 2, xgb_model=cpu_raw)
    gpu = xgb.train(params | {"device": "mps"}, data, 2, xgb_model=gpu_raw)
    check_models(cpu, gpu, data)


@pytest.mark.parametrize("selector", ["greedy", "thrifty"])
@pytest.mark.parametrize("top_k", [1, 3, 20])
def test_linear_top_k(metal, selector, top_k):
    data, objective = data_and_params("sparse", "multiclass")
    params = dict(booster="gblinear", updater="coord_descent", feature_selector=selector,
                  top_k=top_k, nthread=2, eta=.2, **objective)
    cpu = xgb.train(params | {"device": "cpu"}, data, 4)
    gpu = xgb.train(params | {"device": "mps"}, data, 4)
    check_models(cpu, gpu, data)


@pytest.mark.parametrize("rows", [0, 1, 257])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_linear_empty_columns_rows_and_weights(metal, rows, zero_weight):
    x = sp.csr_matrix((rows, 5), dtype=np.float32)
    data = xgb.DMatrix(x, label=np.ones(rows), weight=np.zeros(rows) if zero_weight else None, nthread=1)
    params = dict(booster="gblinear", updater="coord_descent", nthread=1, base_score=.25)
    cpu = xgb.train(params | {"device": "cpu"}, data, 3)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data, 3)
    after = xgb.metal_info()
    assert after["linear_gradient_dispatches"] - before["linear_gradient_dispatches"] == 3 * bool(rows)
    check_models(cpu, gpu, data)


def test_linear_convergence_and_zero_step(metal):
    data, objective = data_and_params("dense", "regression")
    for options in [dict(eta=0.), dict(tolerance=1e6)]:
        params = dict(booster="gblinear", updater="coord_descent", nthread=2, **objective, **options)
        cpu = xgb.train(params | {"device": "cpu"}, data, 5)
        before = xgb.metal_info()
        gpu = xgb.train(params | {"device": "mps"}, data, 5)
        after = xgb.metal_info()
        check_models(cpu, gpu, data)
        assert after["linear_gradient_dispatches"] > before["linear_gradient_dispatches"]
        if options.get("eta") == 0:
            assert after["linear_residual_dispatches"] == before["linear_residual_dispatches"]


def test_shotgun_retains_host_updater(metal):
    # Upstream CUDA also leaves the default shotgun updater on the host.
    data, objective = data_and_params("sparse", "regression")
    params = dict(booster="gblinear", nthread=1, **objective)
    cpu = xgb.train(params | {"device": "cpu"}, data, 3)
    before = xgb.metal_info()["linear_gradient_dispatches"]
    gpu = xgb.train(params | {"device": "mps"}, data, 3)
    assert xgb.metal_info()["linear_gradient_dispatches"] == before
    check_models(cpu, gpu, data)


def test_linear_external_pages_rejected_like_cuda(metal, tmp_path):
    class Pages(xgb.DataIter):
        def __init__(self):
            self.page = 0
            super().__init__(cache_prefix=str(tmp_path / "linear-pages"))

        def reset(self):
            self.page = 0

        def next(self, input_data):
            if self.page == 2:
                return False
            x = np.arange(45, dtype=np.float32).reshape(15, 3) + self.page
            input_data(data=x, label=x[:, 0])
            self.page += 1
            return True
    data = xgb.DMatrix(Pages(), nthread=1)
    before = xgb.metal_info()["linear_gradient_dispatches"]
    with pytest.raises(xgb.core.XGBoostError, match="single CSC block"):
        xgb.train(dict(device="mps", booster="gblinear", updater="coord_descent", nthread=1), data, 1)
    assert xgb.metal_info()["linear_gradient_dispatches"] == before


def test_linear_deprecated_cuda_updater_name(metal):
    data, objective = data_and_params("dense", "regression")
    for device in ["cpu", "mps"]:
        with pytest.raises(xgb.core.XGBoostError, match="gpu_coord_descent.*deprecated"):
            xgb.train(dict(device=device, booster="gblinear", updater="gpu_coord_descent",
                           nthread=1, **objective), data, 1)
