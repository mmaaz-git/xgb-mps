"""GPU split-selection coverage independent of the histogram dispatch checks."""
import json

import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("layout", ["numeric", "categorical_onehot", "categorical_partition"])
@pytest.mark.parametrize("constraints", ["none", "monotone", "interaction", "regularized"])
def test_gpu_split_selection(metal, strategy, layout, constraints, tmp_path):
    rng = np.random.default_rng(453)
    x = rng.normal(size=(2053, 7)).astype(np.float32)
    x[:, 0] = rng.integers(0, 9, len(x))
    y = np.column_stack([x[:, 0] + x[:, 1] ** 2 - x[:, 2],
                         2 * x[:, 0] - x[:, 1] - x[:, 2],
                         x[:, 0] + x[:, 3] * 2]).astype(np.float32)
    # Include missing categorical/numeric values and an entirely empty feature.
    x[rng.random(x.shape) < 0.15] = np.nan
    x[:, -1] = np.nan
    w = rng.uniform(0.1, 2, len(x)).astype(np.float32)
    w[::11] = 0
    types = ["q"] * x.shape[1]
    if layout != "numeric":
        types[0] = "c"
    d = xgb.DMatrix(x, label=y, weight=w, feature_types=types)
    # Row RNGs intentionally differ. Isolate splits/column sampling here;
    # test_metal_sampling supplies explicit CUDA row masks for sampled training.
    p = dict(tree_method="hist", multi_strategy=strategy, max_depth=4,
             max_bin=33, nthread=2, seed=7, subsample=1.,
             colsample_bytree=0.9, colsample_bylevel=0.9,
             max_cat_to_onehot=16 if layout == "categorical_onehot" else 2,
             max_cat_threshold=6)
    if constraints == "monotone":
        # Keep the categorical feature unconstrained.
        p["monotone_constraints"] = "(0,1,-1,0,0,0,0)"
    elif constraints == "interaction":
        p["interaction_constraints"] = "[[0,1,2],[3,4,5,6]]"
    elif constraints == "regularized":
        p.update(reg_alpha=0.3, reg_lambda=2, max_delta_step=0.5, min_child_weight=3)
    before = xgb.metal_info()["split_dispatches"]
    cpu = xgb.train(p | {"device": "cpu"}, d, 5)
    assert xgb.metal_info()["split_dispatches"] == before
    gpu = xgb.train(p | {"device": "mps"}, d, 5)
    assert xgb.metal_info()["split_dispatches"] > before
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)
    cpu_trees = json.loads(cpu.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
    gpu_trees = json.loads(gpu.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
    # Verify selected topology, features, directions and categories, not only loss.
    for a, b in zip(cpu_trees, gpu_trees, strict=True):
        for key in ["left_children", "right_children", "split_indices", "default_left",
                    "categories", "categories_nodes", "categories_segments", "categories_sizes"]:
            assert a[key] == b[key]
    path = tmp_path / "model.ubj"
    gpu.save_model(path)
    loaded = xgb.Booster(params={"device": "cpu"}, model_file=path)
    np.testing.assert_allclose(loaded.predict(d), cpu.predict(d), rtol=2e-5, atol=2e-5)
