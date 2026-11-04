"""Owned prediction output: real DLPack consumers, lifetimes and explicit copies."""

import gc
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from test_metal_tensor import producer

import xgboost as xgb
from xgboost._metal import _capsule


def fixture(kind="scalar", rows=131):
    rng = np.random.default_rng(481)
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    y = x[:, 0] - 2 * x[:, 1]
    params = dict(nthread=2, max_depth=3, max_bin=17, base_score=0.5)
    if kind in ("multi:softprob", "multi:softmax"):
        y = np.digitize(y, [-1, 1])
        params.update(objective=kind, num_class=3)
    elif kind == "vector":
        y = np.column_stack([y, -y, y / 2])
        params.update(multi_strategy="multi_output_tree")
    elif kind == "dart":
        params.update(booster="dart", rate_drop=0.4, seed=73)
    elif kind == "linear":
        params.update(booster="gblinear", updater="coord_descent")
    elif kind == "binary:logistic":
        y = (y > 0).astype(np.float32)
        params.update(objective=kind)
    elif kind in ("count:poisson", "reg:gamma", "reg:tweedie", "survival:aft"):
        y = np.exp(y / 4)
        params.update(objective=kind)
    elif kind in ("reg:quantileerror", "reg:expectileerror"):
        params.update(objective=kind)
        params[
            "quantile_alpha" if kind == "reg:quantileerror" else "expectile_alpha"
        ] = [0.2, 0.8]
    data = xgb.DMatrix(x, label=y, nthread=2)
    if kind == "survival:aft":
        data.set_info(label_lower_bound=y, label_upper_bound=y)
    cpu = xgb.train(params, data, 3)
    gpu = xgb.Booster(model_file=cpu.save_raw())
    gpu.set_param({"device": "mps", "nthread": 2})
    return x, data, cpu, gpu


def consume(name, result):
    module = pytest.importorskip("torch" if name == "torch" else "mlx.core")
    return module.from_dlpack(result)


def host(name, tensor):
    if name == "torch":
        assert tensor.device.type == "mps"
        return tensor.cpu().numpy()
    return np.array(tensor)


@pytest.mark.parametrize("kind", ["scalar", "multi:softprob", "vector", "dart"])
@pytest.mark.parametrize("mode", ["value", "margin", "leaf", "contrib", "interaction"])
@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_prediction_modes(metal, kind, mode, name):
    _, data, cpu, gpu = fixture(kind)
    options = dict(strict_shape=True, iteration_range=(0, 2))
    options.update(
        {
            "value": {},
            "margin": dict(output_margin=True),
            "leaf": dict(pred_leaf=True),
            "contrib": dict(pred_contribs=True),
            "interaction": dict(pred_interactions=True),
        }[mode]
    )
    if kind == "vector" and mode == "leaf":
        # Upstream rejects strict leaf shapes for vector trees on every device.
        for model in (cpu, gpu):
            with pytest.raises(
                xgb.core.XGBoostError, match="strict_shape.*vector leaf"
            ):
                model.predict(data, output_device="mps", **options)
        options["strict_shape"] = False
    expected = gpu.predict(data, **options)
    assert isinstance(expected, np.ndarray)
    before = xgb.metal_info()
    result = gpu.predict(data, output_device="mps", **options)
    assert isinstance(result, xgb.MetalOutput)
    assert result.shape == expected.shape and result.size == expected.size
    assert result.ndim == expected.ndim and result.dtype == np.float32
    assert result.__dlpack_device__() == (8, 0)
    tensor = consume(name, result)
    after = xgb.metal_info()
    assert after["output_host_readback_bytes"] == before["output_host_readback_bytes"]
    assert (
        after["output_snapshot_host_copy_bytes"]
        == before["output_snapshot_host_copy_bytes"]
    )
    assert (
        after["output_snapshot_shared_input_bytes"]
        - before["output_snapshot_shared_input_bytes"]
        == expected.nbytes
    )
    np.testing.assert_array_equal(host(name, tensor), expected)
    np.testing.assert_allclose(
        result.numpy(), cpu.predict(data, **options), rtol=2e-5, atol=2e-5
    )
    assert (
        xgb.metal_info()["output_host_readback_bytes"]
        - after["output_host_readback_bytes"]
        == expected.nbytes
    )
    # All exports own storage independently of the cache/model/source handle.
    gpu.predict(data, output_margin=True)
    del result, gpu, data
    gc.collect()
    np.testing.assert_array_equal(host(name, tensor), expected)


@pytest.mark.parametrize(
    "kind",
    [
        "binary:logistic",
        "multi:softmax",
        "count:poisson",
        "reg:gamma",
        "reg:tweedie",
        "survival:aft",
        "reg:quantileerror",
        "reg:expectileerror",
        "linear",
    ],
)
def test_transforms(metal, kind):
    _, data, cpu, gpu = fixture(kind)
    before = xgb.metal_info()
    result = gpu.predict(data, output_device="mps")
    after = xgb.metal_info()
    assert (
        after["prediction_host_output_bytes"] == before["prediction_host_output_bytes"]
    )
    assert after["prediction_host_input_bytes"] == before["prediction_host_input_bytes"]
    if kind != "linear":
        assert (
            after["prediction_shared_output_bytes"]
            > before["prediction_shared_output_bytes"]
        )
    if kind == "multi:softmax":
        assert (
            after["prediction_device_copy_bytes"]
            > before["prediction_device_copy_bytes"]
        )
    np.testing.assert_array_equal(result.numpy(), gpu.predict(data))
    np.testing.assert_allclose(result.numpy(), cpu.predict(data), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "layout", ["dense", "csr", "pandas", "torch", "mlx", "columnar"]
)
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("margin", [False, True])
def test_inplace_sources(metal, layout, device, margin):
    x, _, cpu, gpu = fixture()
    x[::7, 2] = -99
    if layout == "dense":
        values = x
    elif layout == "csr":
        values = sp.csr_matrix(x)
    elif layout == "pandas":
        values = pd.DataFrame(x)
    elif layout == "columnar":
        values = xgb.MetalColumnar(
            [producer("torch", x[:, i]) for i in range(x.shape[1])]
        )
    else:
        values = producer(layout, x)
    model = gpu if device == "mps" else cpu
    base = np.linspace(-1, 1, len(x), dtype=np.float32)
    options = dict(
        missing=-99,
        base_margin=base,
        iteration_range=(1, 3),
        predict_type="margin" if margin else "value",
        strict_shape=True,
    )
    expected = cpu.inplace_predict(x, **options)
    before = xgb.metal_info()
    result = model.inplace_predict(values, output_device="mps", **options)
    after = xgb.metal_info()
    assert after["output_host_readback_bytes"] == before["output_host_readback_bytes"]
    np.testing.assert_array_equal(result.numpy(), expected)
    np.testing.assert_array_equal(
        model.inplace_predict(values, output_device="cpu", **options), expected
    )


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("rows", [0, 1, 65537])
def test_empty_and_chunked_output(metal, name, rows):
    x, _, cpu, gpu = fixture()
    values = np.resize(x, (rows, x.shape[1])).copy()
    native = producer(name, values)
    for strict in [False, True]:
        expected = cpu.inplace_predict(values, strict_shape=strict)
        result = gpu.inplace_predict(native, output_device="mps", strict_shape=strict)
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(host(name, consume(name, result)), expected)


@pytest.mark.parametrize("version", [None, (0, 8), (1, 0), (1, 1)])
@pytest.mark.parametrize("copy", [None, False, True])
def test_exports_and_cleanup(metal, version, copy):
    gc.collect()
    before = xgb.metal_info()
    _, data, _, gpu = fixture()
    result = gpu.predict(data, output_device="mps")
    expected = result.numpy()
    capsules = [result.__dlpack__(max_version=version, copy=copy) for _ in range(3)]
    records = [_capsule(c) for c in capsules]
    assert len({r[0] for r in records}) == 3
    is_versioned = version is not None and version >= (1, 0)
    for _, record, _ in records:
        if is_versioned:
            assert record.version.major == 1 and record.version.minor == 0
            assert record.flags == (2 if copy else 0)
        assert (
            tuple(record.tensor.shape[i] for i in range(record.tensor.ndim))
            == result.shape
        )
    # Native MetalData consumes both capsule ABIs and frees them exactly once.
    matrix = xgb.DMatrix(
        np.zeros((len(expected), 1), dtype=np.float32), label=capsules.pop(), nthread=2
    )
    np.testing.assert_array_equal(matrix.get_label(), expected)
    del result, capsules, records, matrix
    gc.collect()
    after = xgb.metal_info()
    assert after["output_live_buffers"] == before["output_live_buffers"]
    assert after["output_live_exports"] == before["output_live_exports"]


def test_shared_writes_copy_and_lifetimes(metal):
    torch = pytest.importorskip("torch")
    gc.collect()
    before = xgb.metal_info()
    _, data, _, gpu = fixture()
    result = gpu.predict(data, output_device="mps")
    expected = result.numpy()
    first, second = torch.from_dlpack(result), torch.from_dlpack(result)
    clone = torch.from_dlpack(result.__dlpack__(copy=True))
    assert (
        first.data_ptr() == second.data_ptr() and first.data_ptr() != clone.data_ptr()
    )
    first.add_(2)
    # numpy/re-export synchronize the known consumer's queue before reading writes.
    np.testing.assert_array_equal(result.numpy(), expected + 2)
    np.testing.assert_array_equal(host("torch", second), expected + 2)
    np.testing.assert_array_equal(host("torch", clone), expected)
    np.testing.assert_array_equal(gpu.predict(data), expected)
    del result, first, gpu, data
    gc.collect()
    np.testing.assert_array_equal(host("torch", second), expected + 2)
    del second, clone
    gc.collect()
    after = xgb.metal_info()
    assert after["output_live_buffers"] == before["output_live_buffers"]
    assert after["output_live_exports"] == before["output_live_exports"]


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_gpu_output_chaining(metal, name):
    x, data, _, gpu = fixture()
    result = gpu.predict(
        data, output_margin=True, strict_shape=True, output_device="mps"
    )
    expected = gpu.predict(data, output_margin=True, strict_shape=True)
    before = xgb.metal_info()
    matrix = xgb.DMatrix(result, label=result, nthread=2)
    tensor = consume(name, result)
    gradient = tensor * 2
    hessian = tensor * 0 + 1
    a = xgb.Booster(dict(device="mps", nthread=2, max_depth=2, base_score=0.5), [data])
    b = xgb.Booster(dict(device="mps", nthread=2, max_depth=2, base_score=0.5), [data])
    a.boost(data, 0, grad=gradient, hess=hessian)
    b.boost(data, 0, grad=expected * 2, hess=np.ones_like(expected))
    np.testing.assert_array_equal(a.save_raw(), b.save_raw())
    after = xgb.metal_info()
    assert after["output_host_readback_bytes"] == before["output_host_readback_bytes"]
    assert after["input_staging_bytes"] == before["input_staging_bytes"]
    np.testing.assert_array_equal(matrix.get_label(), expected[:, 0])
    np.testing.assert_array_equal(matrix.get_data().toarray(), expected)
    assert matrix.num_row() == len(x)


def test_threaded_exports(metal):
    gc.collect()
    _, data, _, gpu = fixture()
    result = gpu.predict(data, output_device="mps")
    expected = result.numpy()
    before = xgb.metal_info()

    def run(i):
        capsule = result.__dlpack__(max_version=(1, 0), copy=bool(i % 2))
        matrix = xgb.DMatrix(
            np.zeros((len(expected), 1), dtype=np.float32), label=capsule, nthread=1
        )
        return matrix.get_label()

    with ThreadPoolExecutor(4) as executor:
        for values in executor.map(run, range(12)):
            np.testing.assert_array_equal(values, expected)
    gc.collect()
    after = xgb.metal_info()
    assert after["output_live_buffers"] == before["output_live_buffers"]
    assert after["output_live_exports"] == before["output_live_exports"]


@pytest.mark.parametrize("consumed", [False, True])
@pytest.mark.parametrize("versioned", [False, True])
def test_interpreter_shutdown(metal, consumed, versioned):
    # Native capsule names/destructors must survive module/global teardown.
    code = f"""
import ctypes as C
import numpy as np
import xgboost as xgb
from xgboost import _metal
shape = (C.c_uint64 * 1)(3)
values = np.array([1, 2, 3], dtype=np.float32)
output = xgb.MetalOutput._from_prediction(shape, C.c_uint64(1), values.ctypes.data_as(C.POINTER(C.c_float)))
capsule = output.__dlpack__(max_version={(1, 0) if versioned else None!r})
owner = _metal.MetalData(capsule, allow_tensor_shape=True) if {consumed!r} else None
del output
# Keep the capsule alive in a cycle until finalization.
cycle = [capsule, owner]
cycle.append(cycle)
del capsule, owner, cycle
"""
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Exception ignored" not in completed.stderr


def test_output_validation(metal):
    x, data, _, gpu = fixture()
    result = gpu.predict(data, output_device="mps")
    with pytest.raises(TypeError, match="explicit host copy"):
        np.asarray(result)
    with pytest.raises(TypeError, match="created by XGBoost"):
        xgb.MetalOutput()
    assert "shape=" in repr(result) and "device=mps" in repr(result)
    for method, arg in [(gpu.predict, data), (gpu.inplace_predict, x)]:
        for invalid in ("cuda", "metal", "mps:1", 8):
            with pytest.raises(ValueError, match="output_device"):
                method(arg, output_device=invalid)
    for args, error in [
        ({"stream": 1}, BufferError),
        ({"dl_device": (1, 0)}, BufferError),
        ({"dl_device": (8, 1)}, BufferError),
        ({"copy": 1}, TypeError),
        ({"max_version": (1,)}, TypeError),
        ({"max_version": (-1, 0)}, TypeError),
        ({"max_version": [1, 0]}, TypeError),
    ]:
        with pytest.raises(error):
            result.__dlpack__(**args)
    # Positional stream=None is part of the producer protocol.
    del_capsule = result.__dlpack__(None, dl_device=(8, 0), copy=False)
    del del_capsule
