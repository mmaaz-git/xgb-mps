"""Vector model growth must copy and scale on Metal, including after reload."""
import numpy as np
import pytest
import xgboost as xgb

from cuda_split_reference import RecordedSquaredError, trees
from cuda_multi_split_reference import audit_recorded


@pytest.mark.parametrize("kind", ["numeric", "onehot", "partition"])
@pytest.mark.parametrize("growth", ["depthwise", "lossguide"])
@pytest.mark.parametrize("eta", [0., .3, 1.])
def test_model_copy_scale_and_reload(metal, kind, growth, eta):
    rng = np.random.default_rng(174)
    x = rng.normal(size=(193, 4)).astype(np.float32)
    x[:, 0] = rng.integers(0, 7, len(x))
    y = np.column_stack([20*x[:, 0] + x[:, 1], -10*x[:, 0] + x[:, 2],
                         5*x[:, 0] - x[:, 3]]).astype(np.float32)
    x[rng.random(x.shape) < .15] = np.nan
    types = ["q"] * x.shape[1]
    if kind != "numeric":
        types[0] = "c"
    d = xgb.DMatrix(x, label=y, feature_types=types)
    params = dict(device="mps", multi_strategy="multi_output_tree", tree_method="hist",
                  nthread=2, base_score=0, learning_rate=eta, max_depth=3, max_bin=17,
                  max_cached_hist_node=1, grow_policy=growth,
                  max_cat_to_onehot=16 if kind == "onehot" else 2)
    recorded = RecordedSquaredError()
    before = xgb.metal_info()
    model = xgb.train(params, d, 2, obj=recorded)
    saved = model.save_raw()
    # Resume from serialized state, rather than reusing the live GPU workspace.
    # Preserve the saved three-target intercept; a scalar base_score override
    # during continuation is rejected by upstream model validation.
    resume_params = {key: value for key, value in params.items() if key != "base_score"}
    model = xgb.train(resume_params, d, 1, obj=recorded, xgb_model=saved)
    after = xgb.metal_info()
    splits, leaves = audit_recorded(model, d, x, recorded, params)
    assert splits > 0 and leaves > 0
    category_words = sum(len(tree["categories"]) for tree in trees(model))
    # JSON stores category IDs, not packed bitset words. With categories 0..6,
    # each categorical split has exactly one storage word.
    category_splits = sum(sum(tree["split_type"]) for tree in trees(model))
    assert (category_words > 0) == (kind != "numeric")
    roots = len(trees(model))
    assert after["model_copy_words"] - before["model_copy_words"] == (
        (3 * splits + 2 * roots) * y.shape[1] + category_splits)
    copies = after["model_weight_copy_dispatches"] - before["model_weight_copy_dispatches"]
    scales = after["model_scale_dispatches"] - before["model_scale_dispatches"]
    categories = after["model_category_copy_dispatches"] - before["model_category_copy_dispatches"]
    assert copies == scales + roots and scales > roots
    assert 0 < categories <= copies if kind != "numeric" else categories == 0
    assert after["split_weight_store_dispatches"] - before["split_weight_store_dispatches"] > roots
    for key in ["split_weight_readback_bytes", "model_host_copy_bytes"]:
        assert after[key] == before[key]
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(d), cpu.predict(d))
    np.testing.assert_array_equal(model.inplace_predict(x), cpu.inplace_predict(x))


@pytest.mark.parametrize("targets", [3, 17])
@pytest.mark.parametrize("eta", [0., .3, 1.])
def test_root_only_model_copies_and_scales_resident_weights(metal, targets, eta):
    x = np.full((17, 2), np.nan, dtype=np.float32)
    y = np.ones((17, targets), dtype=np.float32)
    d = xgb.DMatrix(x, label=y)
    params = dict(device="mps", tree_method="hist", nthread=1, base_score=0,
                  learning_rate=eta, multi_strategy="multi_output_tree")
    before = xgb.metal_info()
    model = xgb.train(params, d, 2)
    after = xgb.metal_info()
    assert all(len(tree["left_children"]) == 1 for tree in trees(model))
    assert after["model_weight_copy_dispatches"] - before["model_weight_copy_dispatches"] == 4
    assert after["model_copy_words"] - before["model_copy_words"] == 4 * targets
    assert after["model_scale_dispatches"] - before["model_scale_dispatches"] == 2
    assert after["split_weight_store_dispatches"] - before["split_weight_store_dispatches"] == 2
    for key in ["model_category_copy_dispatches", "split_weight_readback_bytes",
                "model_host_copy_bytes"]:
        assert after[key] == before[key]
    cpu = xgb.train({**params, "device": "cpu"}, d, 2)
    np.testing.assert_allclose(model.predict(d), cpu.predict(d), rtol=2e-6, atol=2e-6)
