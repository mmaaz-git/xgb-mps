"""Categorical winners must build CUDA-layout bitsets on Metal."""
import numpy as np
import pytest
import xgboost as xgb

from cuda_split_reference import RecordedSquaredError, trees, audit_recorded as scalar_audit
from cuda_multi_split_reference import audit_recorded as vector_audit


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("growth", ["depthwise", "lossguide"])
@pytest.mark.parametrize("onehot", [False, True])
@pytest.mark.parametrize("sparse_ids", [False, True])
def test_categorical_gpu_assembly(metal, strategy, growth, onehot, sparse_ids):
    rng = np.random.default_rng(241)
    ids = np.array([0, 1, 31, 32, 63, 64, 127, 255, 1023] if sparse_ids else range(9),
                   dtype=np.float32)
    choices = rng.integers(0, len(ids), 257)
    x = rng.normal(size=(len(choices), 3)).astype(np.float32)
    x[:, 0] = ids[choices]
    y = np.column_stack([choices**2 + x[:, 1], -3*choices + x[:, 2]]).astype(np.float32)
    x[rng.random(x.shape) < .17] = np.nan
    d = xgb.DMatrix(x, label=y, feature_types=["c", "q", "q"],
                    weight=rng.integers(0, 5, len(x)).astype(np.float32) * .25)
    params = dict(device="mps", tree_method="hist", multi_strategy=strategy, nthread=2,
                  max_depth=4, max_bin=17, max_cached_hist_node=1, grow_policy=growth,
                  base_score=0, learning_rate=.3,
                  max_cat_to_onehot=int(ids[-1]) + 2 if onehot else 2,
                  max_cat_threshold=6)
    recorded = RecordedSquaredError()
    before = xgb.metal_info()
    model = xgb.train(params, d, 3, obj=recorded)
    after = xgb.metal_info()
    audit = scalar_audit if strategy == "one_output_per_tree" else vector_audit
    assert min(audit(model, d, x, recorded, params)) > 0
    categorical = sum(sum(tree["split_type"]) for tree in trees(model))
    assert categorical > 0
    assert after["split_category_dispatches"] > before["split_category_dispatches"]
    assert after["split_category_nodes"] - before["split_category_nodes"] >= categorical
    if strategy == "multi_output_tree":
        assert after["split_category_store_dispatches"] > before["split_category_store_dispatches"]
        assert after["split_category_readback_bytes"] == before["split_category_readback_bytes"]
        assert after["partition_category_host_copy_bytes"] == before["partition_category_host_copy_bytes"]
        assert after["partition_category_shared_input_bytes"] > before["partition_category_shared_input_bytes"]
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(d), cpu.predict(d))
    np.testing.assert_array_equal(model.inplace_predict(x), cpu.inplace_predict(x))


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("onehot", [False, True])
def test_high_category_word_and_tail(metal, strategy, onehot):
    # Public cut construction emits every ID through the maximum. The native
    # test covers 2^24-1 directly without creating 2^24 training histogram bins.
    x = np.tile(np.array([0, 31, 32, 63, 64, 4095], dtype=np.float32), 4)[:, None]
    y = np.column_stack([(x[:, 0] == 4095)*10, (x[:, 0] == 4095)*-7]).astype(np.float32)
    d = xgb.DMatrix(x, label=y, feature_types=["c"])
    params = dict(device="mps", tree_method="hist", multi_strategy=strategy, nthread=1,
                  base_score=0, max_depth=1, max_cat_to_onehot=4097 if onehot else 2)
    recorded = RecordedSquaredError()
    model = xgb.train(params, d, 2, obj=recorded)
    audit = scalar_audit if strategy == "one_output_per_tree" else vector_audit
    assert min(audit(model, d, x, recorded, params)) > 0
    # One-hot must select the highest category. Partition may store its complement.
    if onehot:
        assert all(tree["categories"] == [4095] for tree in trees(model))
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(d), cpu.predict(d))


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("categorical", [False, True])
def test_no_category_dispatch_without_categorical_winner(metal, strategy, categorical):
    x = np.tile(np.arange(8, dtype=np.float32), 8)[:, None]
    y = np.column_stack([x[:, 0], -x[:, 0]])
    d = xgb.DMatrix(x, label=y, feature_types=["c" if categorical else "q"])
    params = dict(device="mps", tree_method="hist", multi_strategy=strategy, nthread=1,
                  max_depth=2, base_score=0, max_cat_to_onehot=1, max_cat_threshold=1)
    before = xgb.metal_info()
    model = xgb.train(params, d, 2)
    after = xgb.metal_info()
    # Partition threshold one enumerates no CUDA candidate; numeric trees split.
    assert all((len(tree["left_children"]) == 1) == categorical for tree in trees(model))
    for key in ["split_category_dispatches", "split_category_nodes"]:
        assert after[key] == before[key]


@pytest.mark.parametrize("empty", [False, True])
def test_vector_numeric_winners_clear_category_storage(metal, empty):
    # CUDA clears node category storage even when a numeric feature wins.
    # An empty categorical column must not create a zero-width invalid store.
    x = np.column_stack([np.arange(97, dtype=np.float32),
                         np.full(97, np.nan if empty else 0, dtype=np.float32)])
    y = np.column_stack([x[:, 0], -x[:, 0]])
    data = xgb.DMatrix(x, label=y, feature_types=["q", "c"])
    params = dict(device="mps", tree_method="hist", multi_strategy="multi_output_tree",
                  nthread=1, base_score=0, max_depth=3, max_bin=17)
    before = xgb.metal_info()
    model = xgb.train(params, data, 2)
    after = xgb.metal_info()
    assert all(len(tree["left_children"]) > 1 and not any(tree["split_type"])
               for tree in trees(model))
    assert after["split_category_store_dispatches"] > before["split_category_store_dispatches"]
    assert after["split_category_readback_bytes"] == before["split_category_readback_bytes"]
    cpu = xgb.Booster(params={"device": "cpu"}, model_file=model.save_raw())
    np.testing.assert_array_equal(model.predict(data), cpu.predict(data))
