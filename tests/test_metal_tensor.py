"""Native producer tensors must enter GPU quantization/prediction without CPU staging."""

import ctypes as C
import gc

import numpy as np
import pytest
from test_metal_extmem_adapter import DiskPages

import xgboost as xgb
from xgboost._metal import MetalData, _capsule, _Legacy, _Version, _Versioned, is_metal


def producer(name, values):
    if name == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(values, device="mps")
    mx = pytest.importorskip("mlx.core")
    return mx.array(values)


def no_staging(before):
    after = xgb.metal_info()
    for key in (
        "input_staging_bytes",
        "adapter_debug_readback_bytes",
        "input_debug_readback_bytes",
        "ellpack_pack_input_bytes",
        "ellpack_host_readback_bytes",
    ):
        assert after[key] == before[key], key


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.int8, np.int16, np.int32, np.int64, np.uint8, np.bool_],
)
@pytest.mark.parametrize("layout", ["dense", "slice", "transpose", "broadcast"])
def test_tensor_quantiles_and_prediction(metal, name, dtype, layout):
    values = (np.arange(257 * 14).reshape(257, 14) % 19 - 9).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        values[::3, 2] = np.nan
        values[::5, 7] = -99
    raw = producer(name, values)
    if layout == "slice":
        raw, values = raw[1::2, 1::2], values[1::2, 1::2]
    elif layout == "transpose":
        raw, values = raw.T, values.T
    elif layout == "broadcast":
        if name == "torch":
            raw = raw[:1].expand(257, 14)
        else:
            import mlx.core as mx

            raw = mx.broadcast_to(raw[:1], (257, 14))
        values = np.broadcast_to(values[:1], (257, 14))
    labels = np.arange(values.shape[0], dtype=np.float32) % 7
    before = xgb.metal_info()
    matrix = xgb.QuantileDMatrix(raw, labels, max_bin=17, missing=-99, nthread=2)
    no_staging(before)
    after = xgb.metal_info()
    assert after["tensor_imports"] > before["tensor_imports"]
    assert after["tensor_adapter_windows"] > before["tensor_adapter_windows"]
    host = xgb.QuantileDMatrix(
        values.astype(np.float32),
        labels,
        max_bin=17,
        missing=-99,
        nthread=2,
        device="mps",
    )
    for actual, expected in zip(matrix.get_quantile_cut(), host.get_quantile_cut()):
        np.testing.assert_array_equal(actual, expected)
    params = dict(device="mps", nthread=2, max_bin=17, max_depth=3, base_score=0.5)
    model = xgb.train(params, matrix, 3)
    reference = xgb.train(params, host, 3)
    np.testing.assert_array_equal(model.predict(matrix), reference.predict(host))
    before = xgb.metal_info()
    predicted = model.inplace_predict(raw, missing=-99)
    no_staging(before)
    assert (
        xgb.metal_info()["tensor_prediction_windows"]
        > before["tensor_prediction_windows"]
    )
    assert isinstance(predicted, xgb.MetalOutput)
    predicted = predicted.numpy()
    np.testing.assert_array_equal(
        predicted, model.inplace_predict(values.astype(np.float32), missing=-99)
    )
    model.set_param(dict(device="cpu"))
    np.testing.assert_array_equal(predicted, model.predict(matrix))


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("on_host", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
def test_tensor_iterator(metal, tmp_path, name, on_host, dtype):
    values = (np.arange(519 * 7).reshape(519, 7) % 23).astype(dtype)
    values[::3, ::2] = 0
    raw = producer(name, values)
    parts = [raw[:0], raw[:131], raw[131:258], raw[258:], raw[:0]]
    labels = np.arange(519, dtype=np.float32) % 11
    ys = [labels[:0], labels[:131], labels[131:258], labels[258:], labels[:0]]
    it = DiskPages(tmp_path / "metal", parts, on_host=on_host, label=ys)
    before = xgb.metal_info()
    matrix = xgb.ExtMemQuantileDMatrix(
        it, max_bin=17, missing=0, nthread=2, cache_host_ratio=0.5 if on_host else None
    )
    no_staging(before)
    it.closed = True
    del raw, parts, it
    gc.collect()
    model = xgb.train(dict(device="mps", nthread=2, max_bin=17, max_depth=3), matrix, 3)
    actual = model.predict(matrix)
    np.testing.assert_array_equal(
        actual, model.inplace_predict(values.astype(np.float32), missing=0)
    )
    model.set_param(dict(device="cpu"))
    np.testing.assert_array_equal(actual, model.predict(matrix))
    del matrix, model
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_tensor_capsule_and_snapshot(metal, name):
    values = np.arange(217, dtype=np.float32).reshape(31, 7) / 7
    raw = producer(name, values)
    capsule = raw.__dlpack__()
    data = MetalData(capsule)
    proxy = xgb.core._ProxyDMatrix()
    xgb.data.dispatch_proxy_set_data(proxy, data)
    assert proxy.num_row() == 31 and proxy.num_col() == 7
    assert not is_metal(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="consumed"):
        MetalData(capsule)
    # Resetting/changing the producer after import cannot alter the proxy snapshot.
    if name == "torch":
        raw.fill_(0)
    else:
        raw[:] = 0
    del data, raw, capsule
    gc.collect()
    # The owned proxy still describes the original tensor and needs no producer.
    assert proxy.num_row() == 31


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_tensor_empty_and_versions(metal, name):
    for rows in [0, 1, 37]:
        raw = producer(name, np.ones((rows, 3), dtype=np.float16))
        # The versioned producer protocol is consumed when actually supplied.
        capsule = raw.__dlpack__(max_version=(1, 0))
        q = xgb.QuantileDMatrix(capsule, np.ones(rows), max_bin=17, nthread=2)
        assert q.num_row() == rows and q.num_col() == 3


@pytest.mark.parametrize("versioned", [False, True])
def test_capsule_deleter_once(metal, versioned):
    raw = producer("torch", np.ones((13, 5), dtype=np.float32))
    owner = raw.__dlpack__()
    _, managed, _ = _capsule(owner)
    calls = []
    callback = C.CFUNCTYPE(None, C.c_void_p)(lambda p: calls.append(p))
    pointer = C.cast(callback, C.c_void_p)
    # Read-only is safe: imports only read the producer allocation.
    exported = (
        _Versioned(_Version(1, 0), None, pointer, 1, managed.tensor)
        if versioned
        else _Legacy(managed.tensor, None, pointer)
    )
    new_capsule = C.pythonapi.PyCapsule_New
    new_capsule.argtypes, new_capsule.restype = (
        [C.c_void_p, C.c_char_p, C.c_void_p],
        C.py_object,
    )
    name = b"dltensor_versioned" if versioned else b"dltensor"
    capsule = new_capsule(C.addressof(exported), name, None)
    data = MetalData(capsule)
    proxy = xgb.core._ProxyDMatrix()
    xgb.data.dispatch_proxy_set_data(proxy, data)
    assert calls == []
    del data
    gc.collect()
    assert calls == [C.addressof(exported)]
    del capsule
    gc.collect()
    assert len(calls) == 1


def test_unsupported_capsule_version(metal):
    # No DLTensor is present: consumers must reject the version before reading it.
    header = _Version(2, 0)
    new_capsule = C.pythonapi.PyCapsule_New
    new_capsule.argtypes, new_capsule.restype = (
        [C.c_void_p, C.c_char_p, C.c_void_p],
        C.py_object,
    )
    capsule = new_capsule(C.addressof(header), b"dltensor_versioned", None)
    with pytest.raises(BufferError, match="major version: 2"):
        MetalData(capsule)


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_cached_export_completion(metal, name):
    # DataIter can reuse its transformed export. Writes still need completion at
    # every import, including writes issued after MetalData was first created.
    raw = producer(name, np.zeros((117, 9), dtype=np.float32))
    data = MetalData(raw)
    expected = np.arange(117 * 9, dtype=np.float32).reshape(117, 9) % 31
    replacement = producer(name, expected)
    raw[:] = replacement
    del replacement
    if name == "mlx":
        import mlx.core as mx

        # Export must also complete lazy work from a nondefault producer stream.
        with mx.stream(mx.new_stream(mx.gpu)):
            raw = (raw + 7) * 2
            mx.async_eval(raw)
        # MLX assignment may replace the allocation; a DLPack export is a snapshot
        # of its allocation, not a promise to follow later array replacements.
        data = MetalData(raw)
    else:
        raw.add_(7).mul_(2)
    expected = (expected + 7) * 2
    q = xgb.QuantileDMatrix(data, max_bin=19, nthread=2)
    host = xgb.QuantileDMatrix(expected, max_bin=19, nthread=2, device="mps")
    for actual, reference in zip(q.get_quantile_cut(), host.get_quantile_cut()):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("kind", ["multi_class", "multi_target", "categorical"])
def test_tensor_prediction_shapes_and_reference(metal, name, kind):
    rng = np.random.default_rng(1123)
    values = rng.normal(size=(257, 9)).astype(np.float32)
    label = np.arange(257, dtype=np.float32) % 3
    params = dict(device="mps", nthread=2, max_bin=19, max_depth=3, base_score=0.5)
    feature_types = ["q"] * 9
    if kind == "multi_class":
        params.update(objective="multi:softprob", num_class=3)
    elif kind == "multi_target":
        label = np.stack([label, label / 2], axis=1)
        params.update(multi_strategy="multi_output_tree")
    else:
        values[:, 0] = np.arange(257) % 5
        feature_types[0] = "c"
    raw = producer(name, values)
    q = xgb.QuantileDMatrix(
        raw, label, feature_types=feature_types, max_bin=19, nthread=2
    )
    validation = xgb.QuantileDMatrix(
        raw, label, feature_types=feature_types, ref=q, max_bin=19, nthread=2
    )
    model = xgb.train(params, q, 4)
    for margin in [False, True]:
        expected = model.predict(
            validation, output_margin=margin, iteration_range=(1, 3), strict_shape=True
        )
        before = xgb.metal_info()
        actual = model.inplace_predict(
            raw,
            predict_type="margin" if margin else "value",
            iteration_range=(1, 3),
            strict_shape=True,
        )
        no_staging(before)
        assert isinstance(actual, xgb.MetalOutput)
        actual = actual.numpy()
        np.testing.assert_array_equal(actual, expected)
    model.set_param(dict(device="cpu"))
    np.testing.assert_array_equal(
        actual,
        model.predict(
            validation, output_margin=True, iteration_range=(1, 3), strict_shape=True
        ),
    )


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_tensor_errors(metal, name):
    raw = producer(name, np.ones((11, 3), dtype=np.float32))
    with pytest.raises(xgb.core.XGBoostError, match="Construction device"):
        xgb.QuantileDMatrix(raw, device="cpu")
    with pytest.raises(ValueError, match="two-dimensional|2-dimensional"):
        xgb.QuantileDMatrix(raw[:, 0])
    ordinary = xgb.DMatrix(raw, nthread=2)
    assert ordinary.num_row() == 11 and ordinary.num_col() == 3
    q = xgb.QuantileDMatrix(raw, raw[:, 0], nthread=2)
    np.testing.assert_array_equal(q.get_label(), np.ones(11, dtype=np.float32))
    complex_data = producer(name, np.ones((11, 3), dtype=np.complex64))
    with pytest.raises(xgb.core.XGBoostError, match="dtype"):
        xgb.QuantileDMatrix(complex_data)
