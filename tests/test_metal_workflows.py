"""Small end-to-end Python workflows; timings belong in scripts/, not these tests."""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, ndcg_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


def assert_metal_work(before):
    after = xgb.metal_info()
    assert after["histogram_dispatches"] > before["histogram_dispatches"]
    assert after["prediction_dispatches"] > before["prediction_dispatches"]


def assert_model_roundtrip(model, data, path):
    expected = model.predict(data)
    model.save_model(path)
    restored = type(model)(device="mps", enable_categorical=model.enable_categorical)
    restored.load_model(path)
    np.testing.assert_array_equal(restored.predict(data), expected)
    restored.set_params(device="cpu")
    np.testing.assert_array_equal(restored.predict(data), expected)


def test_pipeline_model_selection_and_persistence(metal, tmp_path):
    """Mixed columns, sparse preprocessing, CV cloning, refit and whole-pipeline I/O."""
    rng = np.random.default_rng(931)
    values = rng.normal(size=(900, 4)).astype(np.float32)
    categories = rng.integers(0, 3, size=len(values))
    target = 3 * values[:, 0] - 2 * values[:, 1] + categories
    target += rng.normal(scale=0.1, size=len(values))
    frame = pd.DataFrame(values, columns=["a", "b", "c", "d"])
    frame["segment"] = np.array(["small", "medium", "large"])[categories]
    frame.loc[rng.choice(len(frame), 25, replace=False), "c"] = np.nan

    scores = {}
    before = xgb.metal_info()
    for device in ["cpu", "mps"]:
        transform = ColumnTransformer(
            [("numeric", SimpleImputer(), ["a", "b", "c", "d"]),
             ("segment", OneHotEncoder(handle_unknown="ignore"), ["segment"])],
            sparse_threshold=1.0,
        )
        pipeline = Pipeline([
            ("prepare", transform),
            ("regressor", xgb.XGBRegressor(
                device=device, tree_method="hist", n_estimators=35,
                learning_rate=0.2, n_jobs=2, random_state=17,
            )),
        ])
        search = GridSearchCV(
            pipeline, {"regressor__max_depth": [2, 3]}, cv=3,
            scoring="neg_mean_squared_error", n_jobs=1, error_score="raise",
        )
        search.fit(frame.iloc[:700], target[:700])
        assert np.isfinite(search.cv_results_["mean_test_score"]).all()
        prediction = search.predict(frame.iloc[700:])
        scores[device] = r2_score(target[700:], prediction)
        assert scores[device] > 0.90
        if device == "mps":
            path = tmp_path / "pipeline.joblib"
            joblib.dump(search.best_estimator_, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict(frame.iloc[700:]), prediction)
            restored.set_params(regressor__device="cpu")
            np.testing.assert_array_equal(restored.predict(frame.iloc[700:]), prediction)
    assert abs(scores["cpu"] - scores["mps"]) < 0.03
    assert_metal_work(before)


def test_categorical_classifier_early_stopping_and_reload(metal, tmp_path):
    """Weighted DataFrames with missing values, a validation set and a held-out test set."""
    rng = np.random.default_rng(267)
    values = rng.normal(size=(1200, 5)).astype(np.float32)
    categories = rng.integers(0, 4, size=len(values))
    target = (3 * values[:, 0] - values[:, 1] + categories - 1.5 > 0).astype(np.int32)
    frame = pd.DataFrame(values, columns=["a", "b", "c", "d", "e"])
    frame["segment"] = pd.Categorical(categories)
    frame.loc[rng.choice(len(frame), 40, replace=False), "d"] = np.nan
    weights = rng.uniform(0.5, 2, size=len(frame))
    scores = {}
    before = xgb.metal_info()
    for device in ["cpu", "mps"]:
        model = xgb.XGBClassifier(
            device=device, tree_method="hist", enable_categorical=True,
            n_estimators=80, max_depth=3, learning_rate=0.2, n_jobs=2,
            random_state=13, early_stopping_rounds=8, eval_metric="logloss",
        )
        model.fit(
            frame.iloc[:800], target[:800], sample_weight=weights[:800],
            eval_set=[(frame.iloc[800:1000], target[800:1000])],
            sample_weight_eval_set=[weights[800:1000]], verbose=False,
        )
        held_out = frame.iloc[1000:].copy()
        held_out["segment"] = held_out["segment"].cat.reorder_categories([3, 1, 0, 2])
        probability = model.predict_proba(held_out)
        assert np.isfinite(probability).all()
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(
            probability,
            model.predict_proba(held_out, iteration_range=(0, model.best_iteration + 1)),
        )
        scores[device] = accuracy_score(target[1000:], model.predict(held_out))
        assert scores[device] > 0.90
        if device == "mps":
            assert_model_roundtrip(model, held_out, tmp_path / "classifier.ubj")
    assert abs(scores["cpu"] - scores["mps"]) < 0.03
    assert_metal_work(before)


def test_ranker_query_split_early_stopping_and_reload(metal, tmp_path):
    """Split whole queries, fit the sklearn ranker and evaluate unseen query groups."""
    rng = np.random.default_rng(751)
    query_size = 16
    values = rng.normal(size=(48 * query_size, 4)).astype(np.float32)
    relevance = np.digitize(3 * values[:, 0] - values[:, 1], [-2, 0, 2]).astype(np.float32)
    qid = np.repeat(np.arange(48), query_size)
    train, valid, test = qid < 32, (qid >= 32) & (qid < 40), qid >= 40
    scores = {}
    before = xgb.metal_info()
    for device in ["cpu", "mps"]:
        model = xgb.XGBRanker(
            device=device, tree_method="hist", objective="rank:ndcg",
            n_estimators=40, max_depth=3, learning_rate=0.2, n_jobs=2,
            random_state=19, eval_metric="ndcg@5", early_stopping_rounds=6,
        )
        model.fit(
            values[train], relevance[train], qid=qid[train],
            eval_set=[(values[valid], relevance[valid])], eval_qid=[qid[valid]],
            verbose=False,
        )
        prediction = model.predict(values[test])
        scores[device] = ndcg_score(
            relevance[test].reshape(-1, query_size),
            prediction.reshape(-1, query_size), k=5,
        )
        assert scores[device] > 0.95
        if device == "mps":
            assert_model_roundtrip(model, values[test], tmp_path / "ranker.ubj")
    assert abs(scores["cpu"] - scores["mps"]) < 0.03
    assert_metal_work(before)


def test_threads_share_one_dmatrix(metal):
    """Concurrent boosters must not race on a shared DMatrix's lazily synced metadata."""
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(1)
    x = rng.normal(size=(20000, 32)).astype(np.float32)
    y = x[:, 0] - x[:, 1]
    weight = rng.random(len(x)).astype(np.float32)
    query = xgb.DMatrix(x[:100])
    params = dict(device="mps", seed=0)
    expected = xgb.train(params, xgb.DMatrix(x, label=y, weight=weight), 5).predict(query)
    with ThreadPoolExecutor(8) as pool:
        for _ in range(30):
            shared = xgb.DMatrix(x, label=y, weight=weight)
            results = pool.map(lambda _: xgb.train(params, shared, 5).predict(query), range(8))
            for result in results:
                np.testing.assert_array_equal(result, expected)
