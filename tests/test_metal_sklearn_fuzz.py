"""Estimator state transitions and classification output contracts."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from hypothesis import HealthCheck, given, note, settings, strategies as st
from sklearn.base import clone

import xgboost as xgb


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("producer", ["torch", "mlx"])
@pytest.mark.parametrize("kind", ["binary", "multiclass", "regression", "multitarget"])
def test_score_with_native_labels_and_weights(metal, device, producer, kind):
    rng = np.random.default_rng(713)
    x = rng.normal(size=(48, 3)).astype(np.float32)
    y = x[:, 0] - x[:, 1]
    if kind == "binary":
        y = (y > 0).astype(np.float32)
    elif kind == "multiclass":
        y = np.digitize(y, [-0.5, 0.5]).astype(np.float32)
    elif kind == "multitarget":
        y = np.column_stack([y, -y])
    weight = rng.uniform(0.1, 2, len(x)).astype(np.float32)
    if producer == "torch":
        torch = pytest.importorskip("torch")

        def native(values):
            return torch.tensor(values, device="mps")
    else:
        mx = pytest.importorskip("mlx.core")
        native = mx.array
    x_native, y_native, weight_native = map(native, [x, y, weight])
    estimator = xgb.XGBClassifier if kind in ["binary", "multiclass"] else xgb.XGBRegressor
    model = estimator(device=device, n_estimators=3, max_depth=2, n_jobs=1)
    model.fit(x_native if device == "mps" else x, y_native if device == "mps" else y)
    expected = model.score(x, y, sample_weight=weight)
    # Each native metadata input must work independently, as well as together.
    for labels, weights in [(y_native, weight), (y, weight_native), (y_native, weight_native)]:
        assert model.score(x_native, labels, sample_weight=weights) == pytest.approx(expected)
    assert model.score(x_native, y_native) == pytest.approx(model.score(x, y))


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("objective", ["multi:softprob", "multi:softmax"])
@pytest.mark.parametrize("n_classes", [1, 2, 4])
def test_explicit_multiclass_outputs(metal, tmp_path, device, objective, n_classes):
    rng = np.random.default_rng(357)
    x = rng.normal(size=(48, 4)).astype(np.float32)
    y = np.arange(len(x)) % n_classes
    model = xgb.XGBClassifier(
        device=device, objective=objective, n_estimators=3, n_jobs=1, max_depth=2,
    ).fit(x, y)
    matrix = xgb.DMatrix(x)
    raw = model.get_booster().predict(matrix, output_margin=True, strict_shape=True)
    expected = np.exp(raw - raw.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)

    def check(estimator):
        assert estimator.objective == objective
        probability = estimator.predict_proba(x)
        assert probability.shape == (len(x), n_classes)
        np.testing.assert_allclose(probability, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(estimator.predict(x), probability.argmax(axis=1))
        np.testing.assert_array_equal(estimator.classes_, np.arange(n_classes))

    check(model)
    # Updating an unrelated parameter must not change the effective objective.
    model.set_params(learning_rate=0.2)
    check(model)
    for suffix in ["json", "ubj"]:
        path = tmp_path / f"classifier.{suffix}"
        model.save_model(path)
        loaded = xgb.XGBClassifier(device=device, n_estimators=3, n_jobs=1, max_depth=2)
        loaded.load_model(path)
        check(loaded)
        # Class counts belong to this fit, even when the old model had 1 or 4.
        # A loaded intercept is an explicit parameter; let the new fit estimate it.
        loaded.set_params(base_score=None)
        binary = np.arange(len(x)) % 2
        reference = clone(loaded).fit(x, binary)
        loaded.fit(x, binary)
        assert loaded.predict(x).shape == (len(x),)
        np.testing.assert_array_equal(loaded.predict_proba(x), reference.predict_proba(x))


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("outputs", [2, 3])
def test_multilabel_predictions_keep_indicator_shape(metal, device, outputs):
    rng = np.random.default_rng(375)
    x = rng.normal(size=(48, 4)).astype(np.float32)
    y = (rng.normal(size=(len(x), outputs)) > 0).astype(np.int32)
    model = xgb.XGBClassifier(
        device=device, n_estimators=3, n_jobs=1, max_depth=2,
    ).fit(x, y)
    probability = model.predict_proba(x)
    assert model.objective == "binary:logistic"
    assert probability.shape == y.shape
    np.testing.assert_array_equal(model.predict(x), probability > 0.5)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("kind", ["classifier", "regressor", "ranker"])
def test_refit_clears_evaluation_history(metal, device, kind):
    rng = np.random.default_rng(167)
    x = rng.normal(size=(48, 4)).astype(np.float32)
    y = np.arange(len(x)) % 2
    options = dict(device=device, n_estimators=2, n_jobs=1, max_depth=2)
    fit_options, eval_options = {}, {}
    if kind == "ranker":
        model = xgb.XGBRanker(**options)
        qid = np.repeat(np.arange(8), 6)
        fit_options = dict(qid=qid)
        eval_options = dict(eval_qid=[qid])
    elif kind == "regressor":
        model = xgb.XGBRegressor(**options)
    else:
        model = xgb.XGBClassifier(**options)
    model.fit(x, y, eval_set=[(x, y)], verbose=False, **fit_options, **eval_options)
    assert model.evals_result()
    model.fit(x, y, **fit_options)
    with pytest.raises(xgb.core.XGBoostError, match="No evaluation result"):
        model.evals_result()
    assert not hasattr(model, "evals_result_")


@pytest.mark.parametrize("device", ["cpu", "mps"])
@settings(
    max_examples=24, deadline=None, derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    seed=st.integers(0, 10000),
    rows=st.integers(12, 65),
    columns=st.integers(1, 7),
    objective=st.sampled_from(["binary:logistic", "multi:softprob", "multi:softmax"]),
    layout=st.sampled_from(["numpy", "pandas", "csr"]),
    steps=st.lists(st.tuples(st.integers(2, 5), st.booleans()), min_size=2, max_size=4),
)
def test_classifier_refit_matches_fresh_clone(
    metal, device, seed, rows, columns, objective, layout, steps,
):
    note(f"device={device}, layout={layout}, objective={objective}, steps={steps}")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, columns)).astype(np.float32)
    weight = rng.uniform(0.1, 2, rows).astype(np.float32)
    data = pd.DataFrame(x) if layout == "pandas" else sp.csr_matrix(x) if layout == "csr" else x
    model = xgb.XGBClassifier(
        device=device, objective=objective, n_estimators=2, n_jobs=1,
        max_depth=2, random_state=17,
    )
    for n_classes, evaluate in steps:
        y = rng.permutation(np.arange(rows) % n_classes)
        options = dict(sample_weight=weight, verbose=False)
        if evaluate:
            options.update(eval_set=[(data, y)], sample_weight_eval_set=[weight])
        reference = clone(model).fit(data, y, **options)
        model.fit(data, y, **options)
        assert model.n_classes_ == n_classes
        probability = model.predict_proba(data)
        assert probability.shape == (rows, n_classes)
        np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(model.predict(data), probability.argmax(axis=1))
        np.testing.assert_array_equal(probability, reference.predict_proba(data))
        if evaluate:
            assert model.evals_result() == reference.evals_result()
        else:
            with pytest.raises(xgb.core.XGBoostError, match="No evaluation result"):
                model.evals_result()
