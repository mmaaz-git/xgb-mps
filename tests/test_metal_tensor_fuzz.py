"""Metamorphic checks for native tensor layouts and metadata snapshots."""

import gc
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from hypothesis import HealthCheck, given, note, settings, strategies as st

import xgboost as xgb


def tensor(name, values):
    if name == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(values, device="mps")
    mx = pytest.importorskip("mlx.core")
    return mx.array(values)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@settings(
    max_examples=100,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    rows=st.integers(1, 65),
    columns=st.integers(1, 9),
    seed=st.integers(0, 10000),
    dtype=st.sampled_from(["float16", "float32", "int32", "uint8", "bool"]),
    layout=st.sampled_from(
        ["dense", "transpose", "slice", "offset", "broadcast", "reverse"]
    ),
    missing=st.sampled_from([np.nan, -3.0]),
)
def test_native_tensor_layouts(metal, name, rows, columns, seed, dtype, layout, missing):
    rng = np.random.default_rng(seed)
    shape = (rows, columns)
    if layout == "transpose":
        shape = (columns, rows)
    elif layout == "slice":
        shape = (2 * rows + 1, 2 * columns + 1)
    elif layout == "offset":
        shape = (rows + 3, columns + 2)
    elif layout == "broadcast":
        shape = (1, columns)
    host = rng.integers(-7, 8, shape).astype(dtype)
    if dtype.startswith("float"):
        host[rng.random(shape) < 0.1] = np.nan
    native = tensor(name, host)
    if layout == "transpose":
        host, native = host.T, native.T
    elif layout == "slice":
        host, native = host[1::2, 1::2], native[1::2, 1::2]
    elif layout == "offset":
        host = host[2 : rows + 2, 1 : columns + 1]
        native = native[2 : rows + 2, 1 : columns + 1]
    elif layout == "reverse":
        host = host[::-1, ::-1]
        native = native.flip((0, 1)) if name == "torch" else native[::-1, ::-1]
    elif layout == "broadcast":
        host = np.broadcast_to(host, (rows, columns))
        if name == "torch":
            native = native.expand(rows, columns)
        else:
            import mlx.core as mx

            native = mx.broadcast_to(native, (rows, columns))
    y = rng.normal(size=rows).astype(np.float32)
    weight = rng.uniform(0.1, 2, rows).astype(np.float32)
    margin = rng.normal(size=rows).astype(np.float32)
    labels = tensor(name, np.column_stack([y, y]))[:, 1]
    weights = tensor(name, np.column_stack([weight, weight]))[:, 1]
    margins = tensor(name, np.column_stack([margin, margin]))[:, 1]
    actual = xgb.DMatrix(
        native, label=labels, weight=weights, base_margin=margins,
        missing=missing, nthread=2,
    )
    expected = xgb.DMatrix(
        host, label=y, weight=weight, base_margin=margin, missing=missing, nthread=2,
    )
    for attribute in ["get_label", "get_weight", "get_base_margin"]:
        np.testing.assert_array_equal(
            getattr(actual, attribute)(), getattr(expected, attribute)()
        )
    a, b = actual.get_data(), expected.get_data()
    for attribute in ["data", "indices", "indptr"]:
        np.testing.assert_array_equal(getattr(a, attribute), getattr(b, attribute))
    params = dict(device="mps", nthread=2, max_depth=2, max_bin=16, base_score=0.0)
    model = xgb.train(params, actual, 2)
    reference = xgb.train(params, expected, 2)
    note(f"producer={name}, shape={host.shape}, dtype={dtype}, layout={layout}, seed={seed}")
    np.testing.assert_array_equal(model.predict(actual), reference.predict(expected))
    prediction = model.inplace_predict(native, base_margin=margins, missing=missing)
    np.testing.assert_array_equal(
        prediction.numpy(),
        model.inplace_predict(host, base_margin=margin, missing=missing),
    )


@settings(
    max_examples=24,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    seed=st.integers(0, 10000),
    rows=st.integers(17, 129),
    kind=st.sampled_from(["scalar", "vector", "multiclass"]),
    method=st.sampled_from(["predict", "inplace"]),
    versioned=st.booleans(),
    copy=st.booleans(),
)
def test_cold_prediction_export_lifetime(metal, seed, rows, kind, method, versioned, copy):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 5)).astype(np.float32)
    y = x[:, 0] - x[:, 1]
    params = dict(nthread=2, max_depth=2, base_score=0.5)
    if kind == "vector":
        y = np.column_stack([y, -y, y / 2])
        params["multi_strategy"] = "multi_output_tree"
    elif kind == "multiclass":
        y = np.arange(rows) % 3
        params.update(objective="multi:softprob", num_class=3)
    data = xgb.DMatrix(x, label=y, nthread=2)
    raw_model = xgb.train(params, data, 3).save_raw()
    reference = xgb.Booster(model_file=raw_model)
    reference.set_param(dict(device="mps", nthread=2))
    expected = reference.predict(data, strict_shape=True)
    model = xgb.Booster(model_file=raw_model)
    model.set_param(dict(device="mps", nthread=2))
    native = tensor("torch", x)
    barrier = threading.Barrier(8)

    def predict(worker):
        barrier.wait()
        count = 1 + (worker * 17) % rows
        if method == "inplace":
            result = model.inplace_predict(native[:count], strict_shape=True)
        else:
            count = rows
            result = model.predict(data, output_device="mps", strict_shape=True)
        capsule = result.__dlpack__(max_version=(1, 0) if versioned else None, copy=copy)
        exported = torch.from_dlpack(capsule)
        return count, exported

    with ThreadPoolExecutor(8) as executor:
        results = list(executor.map(predict, range(8)))
    # Workers, result owners, model and original features are all gone. Only
    # the consumers keep the exported allocations alive.
    del model, reference, data, native
    gc.collect()
    for count, exported in results:
        np.testing.assert_array_equal(exported.cpu().numpy(), expected[:count])


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("dtype", ["float16", "float32", "int64"])
def test_native_tensor_edge_values(metal, name, dtype):
    rng = np.random.default_rng(918)
    if dtype == "float16":
        # Exhaust every finite half payload, including subnormals and -0.
        values = np.arange(65536, dtype=np.uint16).view(np.float16)
    elif dtype == "float32":
        values = rng.integers(0, 2**32, 65536, dtype=np.uint32).view(np.float32)
    else:
        values = rng.integers(-(2**63), 2**63 - 1, 65536, dtype=np.int64)
    values = values[np.isfinite(values)]
    expected = values.astype(np.float32).view(np.uint32)
    native = tensor(name, values)
    matrix = xgb.DMatrix(native.reshape(-1, 1), nthread=2)
    np.testing.assert_array_equal(matrix.get_data().data.view(np.uint32), expected)
    matrix.set_base_margin(native)
    np.testing.assert_array_equal(matrix.get_base_margin().view(np.uint32), expected)
