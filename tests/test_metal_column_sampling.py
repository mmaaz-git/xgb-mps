"""GPU column sampling preserves tree/level/node selection and training models."""
import json

import numpy as np
import pytest
import xgboost as xgb
from cuda_split_reference import RecordedSquaredError, audit_recorded


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
    recorded = RecordedSquaredError() if strategy == "one_output_per_tree" else None
    gpu = xgb.train(params | {"device": "mps"}, data, 4, obj=recorded)
    after = xgb.metal_info()
    assert after["column_init_dispatches"] > before["column_init_dispatches"]
    assert (after["column_sampling_dispatches"] > before["column_sampling_dispatches"]) == (rates != (1., 1., 1.))
    if recorded is not None:
        # Column selection has its own native ID/RNG checks. Scalar split ties
        # follow CUDA, so audit chosen features' candidates and exact child sums.
        assert min(audit_recorded(gpu,data,x,recorded,params)) > 0
    else:
        np.testing.assert_allclose(gpu.predict(data), cpu.predict(data), rtol=2e-5, atol=2e-5)
        models = [json.loads(m.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
                  for m in [cpu, gpu]]
        for a, b in zip(*models, strict=True):
            for key in ["left_children", "right_children", "split_indices", "default_left"]:
                assert a[key] == b[key]
    loaded = xgb.Booster(model_file=gpu.save_raw())
    loaded.set_param({"device": "mps"})
    np.testing.assert_array_equal(loaded.predict(data), gpu.predict(data))
