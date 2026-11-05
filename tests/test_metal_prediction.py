"""Exercise actual Metal tree traversal using the same CPU-trained model."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
from cuda_uniform_reference import UniformSquaredError


def metal_copy(model):
    result = xgb.Booster(model_file=model.save_raw())
    result.set_param({"device": "mps", "nthread": 2})
    return result


@pytest.mark.parametrize("kind", ["regression", "multiclass", "vector", "dart"])
@pytest.mark.parametrize("method", ["batch", "inplace", "leaf"])
def test_prediction_model_publication_reuses_device_arrays(metal, kind, method):
    rng = np.random.default_rng(709)
    x = rng.normal(size=(257, 5)).astype(np.float32)
    y = x[:, 0] - x[:, 1]
    params = dict(nthread=2, max_depth=3, base_score=.25)
    if kind == "multiclass":
        params.update(objective="multi:softprob", num_class=3)
        y = np.digitize(y, [-.5, .5])
    elif kind == "vector":
        params["multi_strategy"] = "multi_output_tree"
        y = np.column_stack([y, -y, y * .5])
    elif kind == "dart":
        params.update(booster="dart", rate_drop=.5, seed=19)
    data = xgb.DMatrix(x, label=y, nthread=2)
    cpu = xgb.train(params, data, 4)
    gpu = metal_copy(cpu)

    def predict(model, end):
        if method == "inplace":
            return model.inplace_predict(x, iteration_range=(0, end))
        return model.predict(data, pred_leaf=method == "leaf", iteration_range=(0, end))

    # Publish all tree mirrors once. Changing the range then forces real model
    # packing/traversal instead of merely retrieving an existing prediction cache.
    predict(gpu, 4)
    for end in [1, 2, 3, 4, 1]:
        expected = predict(cpu, end)
        before = xgb.metal_info()
        actual = predict(gpu, end)
        after = xgb.metal_info()
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        assert after["prediction_model_pack_dispatches"] > before["prediction_model_pack_dispatches"]
        assert after["prediction_model_shared_input_bytes"] > before["prediction_model_shared_input_bytes"]
        assert after["prediction_model_staging_bytes"] == before["prediction_model_staging_bytes"]
        assert after["vector_readback_bytes"] == before["vector_readback_bytes"]
        assert after["vector_upload_bytes"] == before["vector_upload_bytes"]


@pytest.mark.parametrize("vector", [False, True])
def test_concurrent_first_model_publication(metal, vector):
    rng = np.random.default_rng(234)
    x = rng.normal(size=(257, 3)).astype(np.float32)
    y = x[:, 0] - x[:, 1]
    if vector:
        y = np.column_stack([y, -y, y * .25])
    data = xgb.DMatrix(x, label=y, nthread=2)
    cpu = xgb.train(dict(nthread=2, max_depth=3, base_score=.25,
                        multi_strategy="multi_output_tree" if vector else "one_output_per_tree"),
                    data, 4)
    gpu = metal_copy(cpu)
    gpu.save_config()  # Configure without publishing tree arrays or predictions.
    expected = cpu.inplace_predict(x)
    counts = [1, 33, 257, 17] * 2
    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(lambda n: gpu.inplace_predict(x[:n]), counts))
    for n, result in zip(counts, results):
        np.testing.assert_allclose(result, expected[:n], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("objective", ["reg:expectileerror", "binary:logistic", "multi:softmax"])
@pytest.mark.parametrize("margin", [False, True])
@pytest.mark.parametrize("rounds", [0, 3])
def test_prediction_cache_device_handoff(metal, objective, margin, rounds):
    rng = np.random.default_rng(765)
    x = rng.normal(size=(33, 3)).astype(np.float32)
    params = dict(objective=objective, base_score=.5, nthread=2, max_depth=2)
    groups = 1 if objective == "binary:logistic" else 3
    if objective == "reg:expectileerror":
        params["expectile_alpha"] = [.1, .5, .9]
        y = x[:, 0]
    elif objective == "multi:softmax":
        params["num_class"] = 3
        y = np.digitize(x[:, 0], [-.5, .5])
    else:
        y = (x[:, 0] > 0).astype(np.float32)
    base_margin = rng.normal(size=(len(x), groups)).astype(np.float32) if margin else None
    data = xgb.DMatrix(x, label=y, base_margin=base_margin, nthread=2)
    cpu = xgb.train(params, data, rounds)
    expected = cpu.predict(data)
    expected_margin = cpu.predict(data, output_margin=True)
    expected_eval = cpu.eval(data)
    model = metal_copy(cpu)
    for _ in range(2):
        model.set_param({"device": "mps"})
        np.testing.assert_allclose(model.predict(data), expected, rtol=2e-6, atol=2e-6)
        # The complete cache is reused after this switch; no traversal is needed.
        model.set_param({"device": "cpu"})
        before = xgb.metal_info()
        np.testing.assert_allclose(model.predict(data), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(model.predict(data, output_margin=True), expected_margin,
                                   rtol=2e-6, atol=2e-6)
        assert model.eval(data) == expected_eval
        np.testing.assert_allclose(model.inplace_predict(x, base_margin=base_margin), expected,
                                   rtol=2e-6, atol=2e-6)
        after = xgb.metal_info()
        for key in ("transform_dispatches", "prediction_dispatches", "prediction_init_dispatches"):
            assert after[key] == before[key]


@pytest.mark.parametrize("layout", ["dense", "missing", "sparse", "quantile"])
@pytest.mark.parametrize("kind", ["regression", "multiclass", "vector", "dart", "forest"])
def test_gpu_batch_leaf_and_inplace(metal, layout, kind):
    rng = np.random.default_rng(852)
    x = rng.normal(size=(1031, 9)).astype(np.float32)
    y = x[:, 0] * 2 - x[:, 1] + rng.normal(scale=0.1, size=len(x))
    if layout in {"missing", "quantile"}:
        x[rng.random(x.shape) < 0.25] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < 0.7] = 0
        x = sp.csr_matrix(x)
    p = dict(tree_method="hist", nthread=2, max_depth=4, max_bin=32, base_score=0.4)
    if kind == "multiclass":
        y = np.digitize(y, [-1, 1])
        p.update(objective="multi:softprob", num_class=3)
    elif kind == "vector":
        y = np.column_stack([y, -y, y * 0.5])
        p.update(multi_strategy="multi_output_tree")
    elif kind == "dart":
        p.update(booster="dart", rate_drop=0.3, skip_drop=0, seed=53)
    elif kind == "forest":
        p.update(num_parallel_tree=3, subsample=0.7, colsample_bynode=0.7)
    matrix = xgb.QuantileDMatrix if layout == "quantile" else xgb.DMatrix
    d = matrix(x, label=y, max_bin=32) if layout == "quantile" else matrix(x, label=y)
    cpu = xgb.train(p, d, 8)
    gpu = metal_copy(cpu)
    for options in [{}, {"output_margin": True}, {"pred_leaf": True}, {"iteration_range": (2, 6)}]:
        expected = cpu.predict(d, **options)
        gpu = metal_copy(cpu)  # Fresh cache: every option must exercise traversal.
        before = xgb.metal_info()["prediction_dispatches"]
        actual = gpu.predict(d, **options)
        assert xgb.metal_info()["prediction_dispatches"] > before
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    # Inplace prediction accepts raw input even for a model trained from quantized data.
    for options in [{}, {"predict_type": "margin"}, {"iteration_range": (2, 6)}]:
        expected = cpu.inplace_predict(x, **options)
        before = xgb.metal_info()["prediction_dispatches"]
        actual = gpu.inplace_predict(x, **options)
        assert xgb.metal_info()["prediction_dispatches"] > before
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

@pytest.mark.parametrize("onehot", [1, 64])
def test_categorical_recode_prediction(metal, onehot):
    rng = np.random.default_rng(3)
    categories = [f"category-{i}" for i in range(41)]
    x = pd.DataFrame({"cat": pd.Categorical(rng.choice(categories, 1025), categories=categories),
                      "num": rng.normal(size=1025)})
    y = x.cat.cat.codes.to_numpy() % 7 + x.num.to_numpy()
    cpu = xgb.train(dict(tree_method="hist", nthread=2, max_depth=4,
                         max_cat_to_onehot=onehot), xgb.DMatrix(x, label=y, enable_categorical=True), 8)
    gpu = metal_copy(cpu)
    x["cat"] = x.cat.cat.reorder_categories(list(reversed(categories)))
    x.loc[::13, "cat"] = np.nan
    d = xgb.DMatrix(x, enable_categorical=True)
    for leaf in [False, True]:
        before = xgb.metal_info()["prediction_dispatches"]
        actual = gpu.predict(d, pred_leaf=leaf)
        assert xgb.metal_info()["prediction_dispatches"] > before
        np.testing.assert_array_equal(actual, cpu.predict(d, pred_leaf=leaf))
    before = xgb.metal_info()["prediction_dispatches"]
    np.testing.assert_array_equal(gpu.inplace_predict(x), cpu.inplace_predict(x))
    assert xgb.metal_info()["prediction_dispatches"] > before

@pytest.mark.parametrize("rows", [0, 1, 65537])
def test_prediction_chunks_and_base_margin(metal, rows):
    rng = np.random.default_rng(43)
    train = rng.normal(size=(512, 3)).astype(np.float32)
    cpu = xgb.train(dict(nthread=2, max_depth=3), xgb.DMatrix(train, label=train[:, 0]), 3)
    gpu = metal_copy(cpu)
    x = rng.normal(size=(rows, 3)).astype(np.float32)
    margin = rng.normal(size=rows).astype(np.float32)
    d = xgb.DMatrix(x, base_margin=margin)
    before = xgb.metal_info()["prediction_dispatches"]
    np.testing.assert_array_equal(gpu.predict(d), cpu.predict(d))
    np.testing.assert_array_equal(gpu.inplace_predict(x, base_margin=margin),
                                  cpu.inplace_predict(x, base_margin=margin))
    assert (xgb.metal_info()["prediction_dispatches"] > before) == (rows > 0)

@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_gpu_leaf_cache_update(metal, strategy):
    rng = np.random.default_rng(21)
    x = rng.normal(size=(1031, 5)).astype(np.float32)
    y = np.column_stack([x[:, 0], -x[:, 1], x[:, 2] ** 2])
    d = xgb.DMatrix(x, label=y)
    p = dict(nthread=2, max_depth=3, multi_strategy=strategy, subsample=0.7,
             base_score=0., seed_per_iteration=True)
    expected = xgb.train(p | {"device": "cpu", "subsample": 1.}, d, 5,
                         obj=UniformSquaredError(p)).predict(d)
    before = xgb.metal_info()["leaf_update_dispatches"]
    actual = xgb.train(p | {"device": "mps"}, d, 5).predict(d)
    assert xgb.metal_info()["leaf_update_dispatches"] > before
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("threshold", [-8, -1, 0, 1, 8])
def test_prediction_subnormal_split_boundaries(metal, threshold):
    import json
    tiny = np.finfo(np.float32).smallest_subnormal
    cpu = xgb.train(dict(nthread=1, max_depth=1, min_child_weight=0, reg_lambda=0,
                         eta=1, base_score=0), xgb.DMatrix([[0.], [1.]], label=[-1., 1.]), 1)
    model = json.loads(cpu.save_raw(raw_format="json"))
    tree = model["learner"]["gradient_booster"]["model"]["trees"][0]
    assert tree["left_children"][0] >= 0
    tree["split_conditions"][0] = float(np.float32(threshold) * tiny)
    cpu = xgb.Booster(model_file=bytearray(json.dumps(model), "utf-8"))
    cpu.set_param({"device": "cpu"})
    x = (np.arange(-10, 11, dtype=np.float32) * tiny).reshape(-1, 1)
    x = np.vstack([x, [[-0.], [0.], [np.nan]]]).astype(np.float32)
    data = xgb.DMatrix(x, nthread=1)
    for leaf in [False, True]:
        gpu = metal_copy(cpu)
        before = xgb.metal_info()["prediction_dispatches"]
        np.testing.assert_array_equal(gpu.predict(data, pred_leaf=leaf), cpu.predict(data, pred_leaf=leaf))
        assert xgb.metal_info()["prediction_dispatches"] == before + 1
    gpu = metal_copy(cpu)
    np.testing.assert_array_equal(gpu.inplace_predict(x), cpu.inplace_predict(x))


def test_prediction_negative_subnormal_category(metal):
    train = xgb.DMatrix([[0.], [1.]], label=[-1., 1.], feature_types=["c"], nthread=1)
    cpu = xgb.train(dict(nthread=1, max_depth=1, min_child_weight=0, reg_lambda=0,
                         eta=1, base_score=0), train, 1)
    tiny = np.finfo(np.float32).smallest_subnormal
    data = xgb.DMatrix(np.array([-tiny, -0., 0., tiny, 1., np.nan]).reshape(-1, 1),
                       feature_types=["c"], nthread=1)
    for leaf in [False, True]:
        np.testing.assert_array_equal(metal_copy(cpu).predict(data, pred_leaf=leaf),
                                       cpu.predict(data, pred_leaf=leaf))
