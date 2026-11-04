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
    assert after["model_copy_words"] - before["model_copy_words"] == 3 * y.shape[1] * splits + category_splits
    copies = after["model_weight_copy_dispatches"] - before["model_weight_copy_dispatches"]
    scales = after["model_scale_dispatches"] - before["model_scale_dispatches"]
    categories = after["model_category_copy_dispatches"] - before["model_category_copy_dispatches"]
    assert copies == scales and copies > 0
    assert 0 < categories <= copies if kind != "numeric" else categories == 0
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(d), cpu.predict(d))
    np.testing.assert_array_equal(model.inplace_predict(x), cpu.inplace_predict(x))


def test_root_only_model_does_not_copy_or_scale(metal):
    x = np.full((17, 2), np.nan, dtype=np.float32)
    y = np.ones((17, 3), dtype=np.float32)
    d = xgb.DMatrix(x, label=y)
    before = xgb.metal_info()
    model = xgb.train(dict(device="mps", tree_method="hist", nthread=1, base_score=0,
                           multi_strategy="multi_output_tree"), d, 2)
    after = xgb.metal_info()
    assert all(len(tree["left_children"]) == 1 for tree in trees(model))
    for key in ["model_weight_copy_dispatches", "model_category_copy_dispatches",
                "model_copy_words", "model_scale_dispatches"]:
        assert after[key] == before[key]
