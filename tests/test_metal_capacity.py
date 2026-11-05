"""Wide-target training, with fixed cuts and exact analytic predictions."""
import json

import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("targets", [65536])
def test_wide_vector_tree(metal, targets):
    x = np.array([3, 0, 2, 1], dtype=np.float32)[:, None]
    base = (np.arange(targets) % 17 - 8).astype(np.float32)
    y = base[None, :] + 2 * x
    data = xgb.QuantileDMatrix(x, label=y, max_bin=8, nthread=2)
    params = dict(
        tree_method="hist", multi_strategy="multi_output_tree", max_depth=1,
        max_bin=8, reg_lambda=0, min_child_weight=0, learning_rate=1,
        nthread=2, seed=0,
    )
    cpu = xgb.train(params | {"device": "cpu"}, data, 1)
    before = xgb.metal_info()
    gpu = xgb.train(params | {"device": "mps"}, data, 1)
    after = xgb.metal_info()
    assert after["mean_dispatches"] > before["mean_dispatches"]
    assert after["quantization_dispatches"] >= before["quantization_dispatches"] + targets
    expected = base[None, :] + np.where(x < 2, 1, 5).astype(np.float32)
    np.testing.assert_array_equal(cpu.predict(data), expected)
    np.testing.assert_array_equal(gpu.predict(data), expected)
    # A saved model also retains its wide vector leaves across backend changes.
    restored = xgb.Booster(model_file=gpu.save_raw())
    restored.set_param({"device": "cpu", "nthread": 2})
    np.testing.assert_array_equal(restored.predict(data), expected)
    config = json.loads(gpu.save_config())["learner"]["learner_model_param"]
    assert int(config["num_target"]) == targets
