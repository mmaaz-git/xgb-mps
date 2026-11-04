"""GPU column sampling preserves tree/level/node selection and training models."""
import json

import numpy as np
import pytest
import xgboost as xgb
from cuda_split_reference import RecordedSquaredError, audit_recorded
from cuda_multi_split_reference import audit_recorded as audit_vector
from cuda_column_reference import sample


@pytest.mark.parametrize("rates", [(1., 1., 1.), (.7, 1., 1.), (1., .7, 1.),
                                    (1., 1., .7), (.9, .7, .5)])
@pytest.mark.parametrize("weights", ["none", "equal", "varied", "zeros"])
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_column_sampling_training(metal, rates, weights, strategy):
    rng = np.random.default_rng(591)
    x = rng.normal(size=(513, 33)).astype(np.float32)
    y = np.column_stack((x[:, 0] + 2 * x[:, 7], x[:, 13] ** 2 - x[:, 23])).astype(np.float32)
    feature_weights = None
    if weights != "none":
        feature_weights = np.ones(x.shape[1], dtype=np.float32)
        if weights == "varied":
            feature_weights = np.exp2(np.linspace(-18, 18, x.shape[1])).astype(np.float32)
        if weights == "zeros":
            feature_weights[::2] = 0
    x[rng.random(x.shape) < .1] = np.nan
    data = xgb.DMatrix(x, label=y, feature_weights=feature_weights, nthread=2)
    params = dict(tree_method="hist", max_depth=3, max_bin=33, nthread=2, seed=49,
                  multi_strategy=strategy, colsample_bytree=rates[0],
                  colsample_bylevel=rates[1], colsample_bynode=rates[2])
    before = xgb.metal_info()
    cpu = xgb.train(params | {"device": "cpu"}, data, 4)
    after_cpu = xgb.metal_info()
    assert after_cpu["column_sampling_dispatches"] == before["column_sampling_dispatches"]
    assert after_cpu["column_init_dispatches"] == before["column_init_dispatches"]
    # Keep the original built-in-objective baseline for unsampled vector trees;
    # a custom objective skips automatic intercept estimation.
    recorded = (None if strategy == "multi_output_tree" and rates == (1., 1., 1.)
                else RecordedSquaredError())
    gpu = xgb.train(params | {"device": "mps"}, data, 4, obj=recorded)
    after = xgb.metal_info()
    assert after["column_init_dispatches"] > before["column_init_dispatches"]
    assert (after["column_sampling_dispatches"] > before["column_sampling_dispatches"]) == (rates != (1., 1., 1.))
    if strategy == "one_output_per_tree":
        # Column selection has its own native ID/RNG checks. Scalar split ties
        # follow CUDA, so audit chosen features' candidates and exact child sums.
        assert min(audit_recorded(gpu,data,x,recorded,params)) > 0
    else:
        if recorded is not None:
            assert min(audit_vector(gpu, data, x, recorded, params)) > 0
        # CPU deliberately uses another random stream. Retain training identity
        # checks where sampling is disabled; source selections are tested below.
        if rates == (1., 1., 1.):
            np.testing.assert_allclose(gpu.predict(data), cpu.predict(data), rtol=2e-5, atol=2e-5)
            models = [json.loads(m.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
                      for m in [cpu, gpu]]
            for a, b in zip(*models, strict=True):
                for key in ["left_children", "right_children", "split_indices", "default_left"]:
                    assert a[key] == b[key]
    cpu_loaded = xgb.Booster(model_file=gpu.save_raw())
    cpu_loaded.set_param({"device": "cpu"})
    np.testing.assert_allclose(cpu_loaded.predict(data), gpu.predict(data), rtol=2e-5, atol=2e-5)
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps"})
    np.testing.assert_array_equal(loaded.predict(data), gpu.predict(data))


@pytest.mark.parametrize("rates", [(1 / 33, 1., 1.), (1., 1 / 33, 1.),
                                    (1., 1., 1 / 33), (.5, .25, .1)])
@pytest.mark.parametrize("weighted", ["none", "varied", "zero"])
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("seed", [0, 7])
def test_cuda_column_selection_in_training(metal, rates, weighted, strategy, seed):
    rng = np.random.default_rng(193)
    x = rng.normal(size=(257, 33)).astype(np.float32)
    y = x.sum(axis=1)
    if strategy == "multi_output_tree":
        y = np.column_stack((y, x[:, ::2].sum(axis=1)))
    fw = None if weighted == "none" else np.zeros(33, dtype=np.float32)
    if weighted == "varied":
        fw[:] = np.exp2(np.linspace(-6, 6, 33))
    data = xgb.DMatrix(x, label=y, feature_weights=fw, nthread=2)
    params = dict(device="mps", tree_method="hist", base_score=0, max_depth=1,
                  nthread=2, seed=seed, seed_per_iteration=True, multi_strategy=strategy,
                  colsample_bytree=rates[0], colsample_bylevel=rates[1], colsample_bynode=rates[2])
    before = xgb.metal_info()["column_sampling_dispatches"]
    model = xgb.train(params, data, 3)
    assert xgb.metal_info()["column_sampling_dispatches"] > before
    trees = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
    for iteration, tree in enumerate(trees):
        parent = np.random.RandomState((seed * 127 + iteration) & 0xffffffff)
        features = np.arange(33, dtype=np.uint32)
        for rate in rates:
            if rate != 1.:
                draw = int(parent.randint(0, 1 << 32, dtype=np.uint32))
                features = sample(features, fw, draw, rate)
        assert len(features) == 1
        assert tree["left_children"][0] != -1
        assert tree["split_indices"][0] == int(features[0])
    cpu = xgb.Booster(model_file=model.save_raw())
    cpu.set_param({"device": "cpu"})
    np.testing.assert_allclose(cpu.predict(data), model.predict(data), rtol=2e-5, atol=2e-5)
