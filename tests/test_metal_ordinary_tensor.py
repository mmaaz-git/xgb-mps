"""Ordinary matrices from native tensors match CPU CSR data and existing models."""

import gc

import numpy as np
import pytest
from test_metal_extmem_adapter import DiskPages
from test_metal_gradient_import import DeviceOnly
from test_metal_tensor import producer

import xgboost as xgb


def same_csr(actual, expected):
    a, b = actual.get_data(), expected.get_data()
    assert a.shape == b.shape
    np.testing.assert_array_equal(a.indptr, b.indptr)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.data.view(np.uint32), b.data.view(np.uint32))
    assert actual.num_nonmissing() == expected.num_nonmissing()


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.int8, np.int16, np.int32, np.int64, np.uint8, np.bool_],
)
@pytest.mark.parametrize("layout", ["dense", "slice", "transpose", "broadcast"])
@pytest.mark.parametrize("missing", [np.nan, 0.0])
def test_native_ordinary_csr(metal, name, dtype, layout, missing):
    values = (np.arange(137 * 10).reshape(137, 10) % 17 - 8).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        values[::5, 3] = np.nan
        values[::7] = np.nan
        values[:, 9] = np.nan
    raw = producer(name, values)
    if layout == "slice":
        raw, values = raw[::2, 1::2], values[::2, 1::2]
    elif layout == "transpose":
        raw, values = raw.T, values.T
    elif layout == "broadcast":
        if name == "torch":
            raw = raw[1:2].expand(137, 10)
        else:
            import mlx.core as mx

            raw = mx.broadcast_to(raw[1:2], (137, 10))
        values = np.broadcast_to(values[1:2], (137, 10))
    before = xgb.metal_info()
    actual = xgb.DMatrix(DeviceOnly(raw), missing=missing, nthread=2)
    after = xgb.metal_info()
    assert after["tensor_csr_dispatches"] == before["tensor_csr_dispatches"] + 1
    assert (
        after["tensor_csr_values"] - before["tensor_csr_values"]
        == actual.num_nonmissing()
    )
    assert (
        after["adapter_debug_readback_bytes"] == before["adapter_debug_readback_bytes"]
    )
    assert after["input_staging_bytes"] == before["input_staging_bytes"]
    assert (
        after["vector_readback_bytes"] - before["vector_readback_bytes"]
        == (actual.num_row() + 1) * 8
    )
    assert after["tensor_csr_host_copy_bytes"] == before["tensor_csr_host_copy_bytes"]
    # Captured source data must outlive neither the producer nor its allocation.
    del raw
    gc.collect()
    expected = xgb.DMatrix(values.astype(np.float32), missing=missing, nthread=2)
    same_csr(actual, expected)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (7, 0), (1, 1), (1, 7), (7, 1)])
@pytest.mark.parametrize("versioned", [False, True])
def test_native_ordinary_shapes_capsules(metal, name, shape, versioned):
    values = np.ones(shape, dtype=np.float32)
    raw = producer(name, values)
    capsule = raw.__dlpack__(max_version=(1, 0)) if versioned else raw.__dlpack__()
    actual = xgb.DMatrix(capsule, nthread=2)
    with pytest.raises(ValueError, match="consumed"):
        xgb.DMatrix(capsule, nthread=2)
    del raw, capsule
    gc.collect()
    same_csr(actual, xgb.DMatrix(values, nthread=2))


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("method", ["hist", "approx"])
@pytest.mark.parametrize("kind", ["regression", "multiclass", "categorical", "vector"])
def test_native_ordinary_training_and_reuse(metal, name, method, kind, tmp_path):
    rng = np.random.default_rng(4431)
    values = rng.normal(size=(257, 5)).astype(np.float32)
    values[::5, 0] = np.nan
    values[::17] = np.nan
    label = np.nan_to_num(values[:, 1]) * 2 + np.nan_to_num(values[:, 2])
    kwargs = {}
    params = dict(
        nthread=2,
        device="mps",
        tree_method=method,
        max_bin=17,
        max_depth=3,
        base_score=0.5,
    )
    if kind == "multiclass":
        label = (np.arange(257) % 3).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    elif kind == "categorical":
        values[:, 2] = np.arange(257) % 7
        kwargs.update(feature_types=["q", "q", "c", "q", "q"], enable_categorical=True)
    elif kind == "vector":
        label = np.column_stack([label, label * 2])
        params.update(multi_strategy="multi_output_tree")
    weights = (np.arange(257) % 7 + 1).astype(np.float32) / 4
    raw = producer(name, values)
    actual = xgb.DMatrix(
        raw, producer(name, label), weight=producer(name, weights), nthread=2, **kwargs
    )
    expected = xgb.DMatrix(values, label, weight=weights, nthread=2, **kwargs)
    if kind == "vector" and method == "approx":
        # This combination is rejected by the common CPU/CUDA training gate.
        for device in ["cpu", "mps"]:
            for data in [actual, expected]:
                with pytest.raises(
                    xgb.core.XGBoostError, match="Only the hist tree method"
                ):
                    xgb.train(params | {"device": device}, data, 1)
        return
    before = xgb.metal_info()
    model = xgb.train(params, actual, 3)
    after = xgb.metal_info()
    assert after["csr_shared_input_bytes"] > before["csr_shared_input_bytes"]
    assert after["csr_input_copy_bytes"] == before["csr_input_copy_bytes"]
    reference = xgb.train(params, expected, 3)
    assert model.save_raw() == reference.save_raw()
    np.testing.assert_array_equal(model.predict(actual), reference.predict(expected))
    same_csr(actual, expected)
    for bins in [9, 33, 17]:
        config = params | {"max_bin": bins}
        a = xgb.train(config, actual, 1)
        b = xgb.train(config, expected, 1)
        assert a.save_raw() == b.save_raw()
    # CPU fitting stays independent of MPS's quantile algorithm: compare two
    # CPU fits using equal sparse data, rather than forcing CPU/GPU tree equality.
    cpu_config = params | {"device": "cpu"}
    a = xgb.train(cpu_config, actual, 2)
    b = xgb.train(cpu_config, expected, 2)
    assert a.save_raw() == b.save_raw()
    # Native inplace input to a CPU tree model uses the ordinary-matrix fallback.
    np.testing.assert_array_equal(a.inplace_predict(raw).numpy(), a.predict(expected))
    selection = np.array([7, 2, 7, 9, 0, 256], dtype=np.int32)
    same_csr(actual.slice(selection), expected.slice(selection))
    path = tmp_path / "matrix.buffer"
    actual.save_binary(path)
    restored = xgb.DMatrix(path, nthread=2)
    same_csr(restored, expected)
    np.testing.assert_array_equal(restored.get_label(), expected.get_label())
    np.testing.assert_array_equal(restored.get_weight(), expected.get_weight())
    assert restored.feature_types == expected.feature_types
    np.testing.assert_array_equal(a.predict(restored), a.predict(expected))


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("on_host", [False, True])
@pytest.mark.parametrize("method", ["hist", "approx"])
def test_native_ordinary_external(metal, name, on_host, method, tmp_path):
    values = (np.arange(519 * 5).reshape(519, 5) % 31).astype(np.float32) / 4
    values[::7] = np.nan
    label = np.nan_to_num(values[:, 0]) * 2
    bounds = [0, 0, 137, 137, 391, 519, 519]
    raw = producer(name, values)
    parts = [raw[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    ys = [producer(name, label[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    iterator = DiskPages(tmp_path / "native", parts, on_host=on_host, label=ys)
    matrix = xgb.DMatrix(iterator, nthread=2)
    iterator.closed = True
    host = DiskPages(
        tmp_path / "host",
        [values[a:b] for a, b in zip(bounds[:-1], bounds[1:])],
        on_host=on_host,
        label=[label[a:b] for a, b in zip(bounds[:-1], bounds[1:])],
    )
    expected = xgb.DMatrix(host, nthread=2)
    host.closed = True
    del raw, parts, ys, iterator, host
    gc.collect()
    assert matrix.num_row() == expected.num_row() == len(values)
    assert matrix.num_col() == expected.num_col() == values.shape[1]
    assert matrix.num_nonmissing() == expected.num_nonmissing()
    np.testing.assert_array_equal(matrix.get_label(), expected.get_label())
    # Upstream ordinary external matrices cannot export a single CSR page.
    for data in [matrix, expected]:
        with pytest.raises(xgb.core.XGBoostError, match="single CSR page"):
            data.get_data()
    del data
    before = xgb.metal_info()["tensor_csr_dispatches"]
    for bins in [17, 9, 17]:
        params = dict(
            device="mps",
            nthread=2,
            max_depth=3,
            max_bin=bins,
            tree_method=method,
            base_score=0.5,
        )
        actual_model = xgb.train(params, matrix, 2)
        expected_model = xgb.train(params, expected, 2)
        assert actual_model.save_raw() == expected_model.save_raw()
        np.testing.assert_array_equal(
            actual_model.predict(matrix), expected_model.predict(expected)
        )
    assert xgb.metal_info()["tensor_csr_dispatches"] == before
    del matrix, expected, actual_model, expected_model
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_native_ordinary_predicates(metal, name):
    for missing in [np.inf, -np.inf, 0.0, -0.0, np.nan]:
        values = np.array([[0, -0.0, np.nan], [1e-40, -1e-40, missing]], np.float32)
        actual = xgb.DMatrix(producer(name, values), missing=missing, nthread=2)
        expected = xgb.DMatrix(values, missing=missing, nthread=2)
        same_csr(actual, expected)
    for value in [np.inf, -np.inf]:
        with pytest.raises(xgb.core.XGBoostError, match="contains `inf`"):
            xgb.DMatrix(producer(name, np.array([[value]], np.float32)), nthread=2)
    with pytest.raises(ValueError, match="two-dimensional"):
        xgb.DMatrix(producer(name, np.ones(7, np.float32)), nthread=2)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("booster", ["dart", "gblinear"])
def test_native_ordinary_other_boosters(metal, name, booster):
    values = (np.arange(257 * 5).reshape(257, 5) % 31).astype(np.float32) / 4
    values[::11, 2] = np.nan
    label = values[:, 0] - values[:, 1]
    native = xgb.DMatrix(producer(name, values), label, nthread=2)
    host = xgb.DMatrix(values, label, nthread=2)
    params = dict(device="mps", booster=booster, nthread=2, base_score=0.5, seed=19)
    if booster == "gblinear":
        params.update(updater="coord_descent", feature_selector="cyclic")
    else:
        params.update(max_depth=3, rate_drop=0.5, max_bin=17)
    actual = xgb.train(params, native, 4)
    expected = xgb.train(params, host, 4)
    assert actual.save_raw() == expected.save_raw()
    np.testing.assert_array_equal(actual.predict(native), expected.predict(host))


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("targets", [1, 3])
def test_native_cpu_fallback_margin_and_range(metal, name, targets):
    values = (np.arange(137 * 5).reshape(137, 5) % 31).astype(np.float32) / 4
    labels = np.repeat(values[:, :1], targets, axis=1)
    host = xgb.DMatrix(values, labels, nthread=2)
    params = dict(device="cpu", nthread=2, base_score=0.5, max_depth=3)
    if targets > 1:
        params.update(multi_strategy="multi_output_tree")
    model = xgb.train(params, host, 4)
    margin = (np.arange(137 * targets).reshape(137, targets) % 7).astype(np.float32)
    host.set_base_margin(margin)
    before = xgb.metal_info()
    actual = model.inplace_predict(
        producer(name, values),
        base_margin=producer(name, margin),
        predict_type="margin",
        iteration_range=(1, 3),
        strict_shape=True,
    )
    after = xgb.metal_info()
    assert after["tensor_csr_dispatches"] == before["tensor_csr_dispatches"] + 1
    assert after["tensor_prediction_windows"] == before["tensor_prediction_windows"]
    assert isinstance(actual, xgb.MetalOutput)
    actual = actual.numpy()
    expected = model.predict(
        host, output_margin=True, iteration_range=(1, 3), strict_shape=True
    )
    np.testing.assert_array_equal(actual, expected)
