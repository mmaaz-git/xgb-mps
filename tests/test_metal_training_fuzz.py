"""Generated training workflows and regression cases minimized from fuzzing."""

import json
import re

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra import numpy as hnp

import xgboost as xgb


@pytest.mark.parametrize("feature_type", [None, "q", "int"])
@pytest.mark.parametrize("targets", [1, 3])
@pytest.mark.parametrize("with_stats", [False, True])
def test_dump_missing_only_split(metal, feature_type, targets, with_stats):
    x = np.array([[np.nan], [np.nan], [0], [0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    if targets > 1:
        y = np.column_stack([y, -y, y * 2])
    matrix = xgb.DMatrix(
        x, label=y, feature_types=[feature_type] if feature_type else None, nthread=2
    )
    model = xgb.train(
        dict(device="mps", max_depth=1, base_score=0.5, nthread=2, reg_lambda=0,
             multi_strategy="multi_output_tree"), matrix, 1
    )
    # CUDA-derived missing-only splits legitimately use negative infinity.
    # The inspection dump must use the same number spelling as saved models.
    tree = json.loads(model.get_dump(dump_format="json", with_stats=with_stats)[0])
    saved = json.loads(model.save_raw(raw_format="json"))
    expected = saved["learner"]["gradient_booster"]["model"]["trees"][0]
    condition = expected["split_conditions"][0]
    if targets == 1:
        assert np.isneginf(condition)
    # Vector trees may choose an equivalent finite split/default direction.
    # Integer annotations render its ceiling, preserving integer-row routing.
    threshold = np.ceil(condition) if feature_type == "int" else condition
    assert np.float32(tree["split_condition"]) == np.float32(threshold)
    assert tree["missing"] == (tree["yes"] if expected["default_left"][0] else tree["no"])
    # The text format must not narrow an infinite integer-feature threshold.
    text = model.get_dump(dump_format="text")[0]
    assert np.float32(re.search(r"<([^\]]+)\]", text).group(1)) == np.float32(threshold)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_dump_large_integer_threshold(metal, device, sign):
    values = np.array([sign * 2**34, sign * 2**34 + 2**20], dtype=np.float32)
    x = np.repeat(values, 4).reshape(-1, 1)
    matrix = xgb.DMatrix(x, label=np.repeat([0., 1.], 4), feature_types=["int"], nthread=2)
    model = xgb.train(dict(device=device, max_depth=1, base_score=.5, nthread=2), matrix, 1)
    saved = json.loads(model.save_raw(raw_format="json"))
    expected = saved["learner"]["gradient_booster"]["model"]["trees"][0]["split_conditions"][0]
    tree = json.loads(model.get_dump(dump_format="json")[0])
    assert np.float32(tree["split_condition"]) == np.float32(expected)
    text = model.get_dump(dump_format="text")[0]
    assert np.float32(re.search(r"<([^\]]+)\]", text).group(1)) == np.float32(expected)


@pytest.mark.parametrize("name", ["{left}", "{right}", "{missing}", "{stat}", "{value}",
                                  "{left}/{right}/{stat}", "prefix{children}{nodes}suffix"])
@pytest.mark.parametrize("with_stats", [False, True])
def test_dump_preserves_template_like_feature_names(metal, name, with_stats):
    x = np.arange(16, dtype=np.float32).reshape(-1, 1)
    matrix = xgb.DMatrix(x, label=x[:, 0] ** 2, feature_names=[name], nthread=2)
    model = xgb.train(dict(device="mps", max_depth=3, nthread=2), matrix, 1)
    tree = json.loads(model.get_dump(dump_format="json", with_stats=with_stats)[0])

    def check(node):
        if "leaf" in node:
            return 0
        assert node["split"] == name
        return 1 + sum(check(child) for child in node["children"])

    splits = check(tree)
    assert splits > 1  # Tokens inside a subtree must survive outer templates too.
    text = model.get_dump(dump_format="text", with_stats=with_stats)[0]
    assert text.count(f"[{name}<") == splits
    dot = model.get_dump(dump_format="dot", with_stats=with_stats)[0]
    assert dot.count(f'label="{name}<') == splits


@st.composite
def training_inputs(draw):
    rows = draw(st.sampled_from([1, 2, 7, 17, 33]))
    columns = draw(st.integers(1, 5))
    x = draw(hnp.arrays(np.float32, (rows, columns),
                       elements=st.sampled_from([np.nan, -4., -1., 0., 1., 4.])))
    seed = draw(st.integers(0, 2**16))
    objective = draw(st.sampled_from(["reg:squarederror", "binary:logistic", "multi:softprob"]))
    return x, seed, objective


@settings(max_examples=50, deadline=None, derandomize=True,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(case=training_inputs())
def test_generated_training_workflows(metal, case):
    x, seed, objective = case
    rng = np.random.default_rng(seed)
    rows = len(x)
    y = rng.normal(size=rows).astype(np.float32)
    params = dict(device="mps", objective=objective, max_depth=3, max_bin=16,
                  min_child_weight=0, base_score=.5, nthread=2, seed=seed)
    if objective == "binary:logistic":
        y = (y > 0).astype(np.float32)
    elif objective == "multi:softprob":
        y = np.digitize(y, [-.5, .5]).astype(np.float32)
        params["num_class"] = 3
    weights = rng.choice([0., .2, 1., 3.], size=rows).astype(np.float32)
    weights[0] = 1.
    margin = rng.normal(scale=.2, size=(rows, 3) if objective == "multi:softprob" else rows).astype(np.float32)
    matrix = xgb.QuantileDMatrix(x, label=y, weight=weights, base_margin=margin,
                                max_bin=16, nthread=2)
    model = xgb.train(params, matrix, 3)
    query = xgb.DMatrix(x, base_margin=margin, nthread=2)
    expected = model.predict(query, output_margin=True, strict_shape=True)
    assert np.isfinite(expected).all()
    for dump in model.get_dump(dump_format="json", with_stats=True):
        assert "nodeid" in json.loads(dump)
    for device in ["cpu", "mps"]:
        for format in ["ubj", "json"]:
            restored = xgb.Booster(params={"device": device, "nthread": 2},
                                   model_file=model.save_raw(raw_format=format))
            np.testing.assert_array_equal(
                restored.predict(query, output_margin=True, strict_shape=True), expected
            )
            np.testing.assert_array_equal(
                restored.inplace_predict(x, predict_type="margin", base_margin=margin,
                                         strict_shape=True), expected
            )
    order = rng.permutation(rows)
    reordered = xgb.DMatrix(x[order], base_margin=margin[order], nthread=2)
    np.testing.assert_array_equal(
        model.predict(reordered, output_margin=True, strict_shape=True), expected[order]
    )
    # Explicitly stored zeros are values; only absent/NaN entries are missing.
    row, col = np.where(~np.isnan(x))
    sparse = sp.csr_matrix((x[row, col], (row, col)), shape=x.shape)
    sparse_query = xgb.DMatrix(sparse, base_margin=margin, nthread=2)
    np.testing.assert_array_equal(
        model.predict(sparse_query, output_margin=True, strict_shape=True), expected
    )
    first = xgb.train(params, matrix, 1)
    continuation = {k: v for k, v in params.items() if k != "base_score"}
    resumed = xgb.train(continuation, matrix, 2, xgb_model=first.save_raw())
    np.testing.assert_array_equal(
        resumed.predict(query, output_margin=True, strict_shape=True), expected
    )
