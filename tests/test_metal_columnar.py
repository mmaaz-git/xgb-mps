"""Native column buffers, validity bits and exact dictionary names versus host input."""

import ctypes as C
import gc
import json

import numpy as np
import pandas as pd
import pytest
from test_metal_extmem_adapter import DiskPages
from test_metal_gradient_import import DeviceOnly
from test_metal_ordinary_tensor import same_csr
from test_metal_tensor import no_staging, producer

import xgboost as xgb
from xgboost._metal import _capsule, _Legacy, _Version, _Versioned


def native(name, values):
    return DeviceOnly(producer(name, values))


def packed(name, valid, bit):
    bits = np.concatenate([np.zeros(bit, dtype=np.uint8), valid.astype(np.uint8)])
    return native(name, np.packbits(bits, bitorder="little"))


def string_names(name, names):
    buffers = [s.encode("utf-8") for s in names]
    offsets = np.cumsum([0] + [len(s) for s in buffers], dtype=np.int32)
    values = np.frombuffer(b"".join(buffers), dtype=np.int8).copy()
    return native(name, offsets), native(name, values)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("bit", range(8))
@pytest.mark.parametrize("kind", ["ordinary", "quantile"])
def test_mixed_columns_masks(metal, name, bit, kind):
    n = 259
    types = [
        np.float16,
        np.float32,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.bool_,
    ]
    columns, expected = {}, {}
    for i, dtype in enumerate(types):
        host = (np.arange(n * 2) % 31 - 15).astype(dtype)
        raw = producer(name, host)[1::2]
        valid = (np.arange(n) + i) % 7 != 0
        valid[99] = False  # entirely missing row
        values = host[1::2].astype(np.float32)
        values[~valid] = np.nan
        columns[str(i)] = xgb.MetalColumn(
            DeviceOnly(raw), mask=packed(name, valid, bit), mask_bit=bit
        )
        expected[str(i)] = values
    table = xgb.MetalColumnar(columns)
    frame = pd.DataFrame(expected)
    y = np.arange(n, dtype=np.float32) % 5
    cls = xgb.DMatrix if kind == "ordinary" else xgb.QuantileDMatrix
    kwargs = {} if kind == "ordinary" else dict(max_bin=17)
    before = xgb.metal_info()
    actual = cls(table, y, missing=0, nthread=2, **kwargs)
    no_staging(before)
    assert xgb.metal_info()["columnar_imports"] > before["columnar_imports"]
    reference = (
        cls(frame, y, missing=0, nthread=2, device="mps", **kwargs)
        if kind == "quantile"
        else cls(frame, y, missing=0, nthread=2)
    )
    if kind == "ordinary":
        same_csr(actual, reference)
    else:
        for a, b in zip(actual.get_quantile_cut(), reference.get_quantile_cut()):
            np.testing.assert_array_equal(a, b)
    assert actual.feature_names == list(columns)
    params = dict(device="mps", nthread=2, max_bin=17, max_depth=3, base_score=0.5)
    model = xgb.train(params, actual, 2)
    host_model = xgb.train(params, reference, 2)
    np.testing.assert_array_equal(model.predict(actual), host_model.predict(reference))
    before = xgb.metal_info()
    prediction = model.inplace_predict(table, missing=0)
    no_staging(before)
    np.testing.assert_array_equal(prediction, model.inplace_predict(frame, missing=0))
    model.set_param({"device": "cpu"})
    np.testing.assert_array_equal(model.inplace_predict(table, missing=0), prediction)


def categorical(name, names, *, order=None, n=271, begin=0):
    order = np.arange(len(names)) if order is None else np.asarray(order)
    codes = np.arange(begin, begin + n, dtype=np.int32) % len(order)
    valid = np.arange(begin, begin + n) % 11 != 0
    ordered = np.asarray(names)[order]
    category_buffer = (
        string_names(name, ordered)
        if isinstance(names[0], str)
        else native(name, ordered)
    )
    codes_buffer = native(name, codes)
    table = xgb.MetalColumnar(
        {
            "number": native(
                name, (np.arange(begin, begin + n) % 23).astype(np.float32)
            ),
            "category": xgb.MetalColumn(
                codes_buffer,
                categories=category_buffer,
                mask=packed(name, valid, 3),
                mask_bit=3,
            ),
        }
    )
    host_codes = codes.copy()
    host_codes[~valid] = -1
    frame = pd.DataFrame(
        {
            "number": (np.arange(begin, begin + n) % 23).astype(np.float32),
            "category": pd.Categorical.from_codes(host_codes, categories=ordered),
        }
    )
    return table, frame


NAMES = [
    np.array([-(2**63), -1, 2**53, 2**53 + 1, 2**63 - 1], dtype=np.int64),
    ["", "a", "a\0b", "café", "東京"],
]


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("names", NAMES, ids=["int64", "utf8"])
@pytest.mark.parametrize("kind", ["ordinary", "quantile"])
def test_dictionary_prediction_references(metal, name, names, kind, tmp_path):
    table, frame = categorical(name, names)
    new_table, new_frame = categorical(name, names, order=[3, 0, 4, 1, 2])
    cls = xgb.DMatrix if kind == "ordinary" else xgb.QuantileDMatrix
    kwargs = dict(enable_categorical=True, nthread=2)
    if kind == "quantile":
        kwargs.update(max_bin=17)
    y = np.arange(len(frame), dtype=np.float32) % 7
    before = xgb.metal_info()
    actual = cls(table, y, **kwargs)
    assert (
        xgb.metal_info()["category_import_dispatches"]
        > before["category_import_dispatches"]
    )
    assert (
        xgb.metal_info()["category_import_copy_bytes"]
        == before["category_import_copy_bytes"]
    )
    no_staging(before)
    reference = cls(
        frame, y, **(kwargs | (dict(device="mps") if kind == "quantile" else {}))
    )
    reference.feature_types = actual.feature_types
    if kind == "ordinary":
        same_csr(actual, reference)
    for device in ["mps", "cpu"]:
        params = dict(device=device, nthread=2, max_bin=17, max_depth=3, base_score=0.5)
        model = xgb.train(params, actual, 3)
        host_model = xgb.train(params, reference, 3)
        assert model.save_raw() == host_model.save_raw()
        for input_data, expected in [(table, frame), (new_table, new_frame)]:
            np.testing.assert_array_equal(
                model.inplace_predict(input_data), model.inplace_predict(expected)
            )
        ref = model.get_categories()
        extra = dict(ref=actual) if kind == "quantile" else {}
        recoded = cls(new_table, feature_types=ref, **kwargs, **extra)
        host_recoded = cls(new_frame, feature_types=ref, **kwargs, **extra)
        np.testing.assert_array_equal(
            model.predict(recoded), model.predict(host_recoded)
        )
        np.testing.assert_array_equal(
            model.predict(recoded), model.inplace_predict(new_frame)
        )
        if kind == "ordinary":
            same_csr(recoded, host_recoded)
        else:
            for a, b in zip(recoded.get_quantile_cut(), actual.get_quantile_cut()):
                np.testing.assert_array_equal(a, b)
        path = tmp_path / f"{device}.json"
        model.save_model(path)
        restored = xgb.Booster(model_file=path)
        restored.set_param({"device": device})
        np.testing.assert_array_equal(
            restored.inplace_predict(new_table), model.inplace_predict(new_frame)
        )
        del ref, recoded, host_recoded
    # Reject missing categories instead of silently assigning another code.
    unseen = list(names[:-1]) + (["unseen"] if isinstance(names[0], str) else [7])
    unseen = unseen if isinstance(names[0], str) else np.asarray(unseen, dtype=np.int64)
    bad, _ = categorical(name, unseen)
    model.set_param({"device": "mps"})
    with pytest.raises(xgb.core.XGBoostError, match="not in the training set"):
        model.inplace_predict(bad)
    with pytest.raises(xgb.core.XGBoostError, match="not in the training set"):
        cls(bad, feature_types=model.get_categories(), **kwargs)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("kind", ["ordinary", "quantile"])
@pytest.mark.parametrize("on_host", [False, True])
def test_columnar_external(metal, name, kind, on_host, tmp_path):
    parts, frames, labels = [], [], []
    bounds = [0, 0, 117, 117, 391, 519, 519]
    for begin, end in zip(bounds[:-1], bounds[1:]):
        table, frame = categorical(name, NAMES[1], n=end - begin, begin=begin)
        parts.append(table)
        frames.append(frame)
        labels.append((np.arange(begin, end) % 7).astype(np.float32))
    iterator = DiskPages(tmp_path / "native", parts, on_host=on_host, label=labels)
    host_iterator = DiskPages(
        tmp_path / "host",
        frames,
        on_host=on_host,
        label=labels,
        feature_types=[["q", "c"]] * len(frames),
    )
    cls = xgb.DMatrix if kind == "ordinary" else xgb.ExtMemQuantileDMatrix
    kwargs = dict(enable_categorical=True, nthread=2)
    if kind == "quantile":
        kwargs.update(max_bin=17, device="mps")
    actual, reference = cls(iterator, **kwargs), cls(host_iterator, **kwargs)
    iterator.closed = host_iterator.closed = True
    del parts, frames, iterator, host_iterator, table, frame
    gc.collect()
    params = dict(device="mps", nthread=2, max_bin=17, max_depth=3, base_score=0.5)
    model, host_model = xgb.train(params, actual, 2), xgb.train(params, reference, 2)
    assert model.save_raw() == host_model.save_raw()
    np.testing.assert_array_equal(model.predict(actual), host_model.predict(reference))
    test, expected = categorical(name, NAMES[1], order=[4, 2, 0, 3, 1])
    np.testing.assert_array_equal(
        model.inplace_predict(test), model.inplace_predict(expected)
    )
    del actual, reference, model, host_model
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_columnar_validation_capsules(metal, name):
    raw = producer(name, np.arange(17, dtype=np.int32))
    capsule = raw.__dlpack__()
    table = xgb.MetalColumnar([capsule, capsule])
    matrix = xgb.DMatrix(table)
    xgb.DMatrix(table)  # retained aliases can be reimported, but not reconsumed
    assert matrix.num_col() == 2
    with pytest.raises(ValueError, match="consumed"):
        xgb.MetalColumnar([capsule])
    with pytest.raises(ValueError, match="must not equal to 0"):
        xgb.MetalColumnar([])
    with pytest.raises(ValueError, match="vectors"):
        xgb.MetalColumnar([producer(name, np.ones((17, 2), dtype=np.float32))])
    with pytest.raises(ValueError, match="same number of rows"):
        xgb.MetalColumnar([raw, raw[:1]])
    with pytest.raises(TypeError, match="metadata or gradients"):
        matrix.set_label(table)
    table, _ = categorical(name, NAMES[1])
    with pytest.raises(ValueError, match="enable_categorical"):
        xgb.DMatrix(table, enable_categorical=False)
    for mask, bit, error in [
        (np.ones(1, dtype=np.uint8), 0, "too short"),
        (np.ones(3, dtype=np.int8), 0, "uint8"),
    ]:
        table = xgb.MetalColumnar(
            [xgb.MetalColumn(raw, mask=native(name, mask), mask_bit=bit)]
        )
        with pytest.raises(xgb.core.XGBoostError, match=error):
            xgb.DMatrix(table)
    table = xgb.MetalColumnar(
        [
            xgb.MetalColumn(
                raw, categories=native(name, np.array([1.0, 2.0], dtype=np.float32))
            )
        ]
    )
    with pytest.raises(xgb.core.XGBoostError, match="floating point"):
        xgb.DMatrix(table, enable_categorical=True)
    empty = xgb.MetalColumnar({"x": producer(name, np.empty(0, dtype=np.int32))})
    assert xgb.DMatrix(empty).num_row() == 0
    assert xgb.QuantileDMatrix(empty).num_row() == 0
    assert "columns" in json.loads(empty.descriptor)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("kind", ["ordinary", "quantile"])
@pytest.mark.parametrize("on_host", [False, True])
def test_reference_dictionaries_across_batches(metal, name, kind, on_host, tmp_path):
    initial, _ = categorical(name, NAMES[1])
    train = xgb.QuantileDMatrix(initial, np.arange(271) % 7, nthread=2, max_bin=17)
    params = dict(device="mps", max_depth=3, max_bin=17, nthread=2, base_score=0.5)
    booster = xgb.train(params, train, 3)
    ref = booster.get_categories()
    parts, frames, labels = [], [], []
    for i, (begin, end) in enumerate(zip([0, 0, 37, 97, 97], [0, 37, 97, 97, 137])):
        table, frame = categorical(
            name, NAMES[1], order=np.roll(np.arange(5), i), n=end - begin, begin=begin
        )
        parts.append(table)
        frames.append(frame)
        labels.append(np.arange(begin, end) % 7)
    meta = dict(label=labels, feature_types=[ref] * len(parts))
    a_iter = DiskPages(tmp_path / "native", parts, on_host=on_host, **meta)
    b_iter = DiskPages(tmp_path / "host", frames, on_host=on_host, **meta)
    cls = xgb.DMatrix if kind == "ordinary" else xgb.ExtMemQuantileDMatrix
    extra = {} if kind == "ordinary" else dict(ref=train, device="mps", max_bin=17)
    actual, expected = cls(a_iter, nthread=2, **extra), cls(b_iter, nthread=2, **extra)
    a_iter.closed = b_iter.closed = True
    expected.feature_types = actual.feature_types
    np.testing.assert_array_equal(booster.predict(actual), booster.predict(expected))
    # Common inplace output reports (0, 0) for empty scalar batches. Check
    # native/host shapes per batch, then concatenate scalar values explicitly.
    per_batch = []
    for part, frame in zip(parts, frames):
        a = booster.inplace_predict(part, strict_shape=True)
        b = booster.inplace_predict(frame, strict_shape=True)
        np.testing.assert_array_equal(a, b)
        per_batch.append(b.reshape(-1))
    np.testing.assert_array_equal(booster.predict(actual), np.concatenate(per_batch))
    model, reference = xgb.train(params, actual, 2), xgb.train(params, expected, 2)
    assert model.save_raw() == reference.save_raw()
    del a_iter, b_iter, actual, expected, model, reference
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("versioned", [False, True])
def test_columnar_capsule_alias_lifetime(metal, versioned):
    raw = producer("torch", np.arange(17, dtype=np.int32))
    owner = raw.__dlpack__()
    _, managed, _ = _capsule(owner)
    calls = []
    callback = C.CFUNCTYPE(None, C.c_void_p)(lambda p: calls.append(p))
    pointer = C.cast(callback, C.c_void_p)
    exported = (
        _Versioned(_Version(1, 0), None, pointer, 1, managed.tensor)
        if versioned
        else _Legacy(managed.tensor, None, pointer)
    )
    new = C.pythonapi.PyCapsule_New
    new.argtypes, new.restype = [C.c_void_p, C.c_char_p, C.c_void_p], C.py_object
    capsule = new(
        C.addressof(exported), b"dltensor_versioned" if versioned else b"dltensor", None
    )
    # The same capsule is both a dictionary and two columns' code buffer.
    table = xgb.MetalColumnar([xgb.MetalColumn(capsule, categories=capsule), capsule])
    expected = xgb.DMatrix(np.column_stack([np.arange(17), np.arange(17)]))
    actual = xgb.DMatrix(table)
    same_csr(actual, expected)
    assert calls == []
    raw[:] = 0  # synchronous import must preserve the matrix's owned snapshot
    del table
    gc.collect()
    assert calls == [C.addressof(exported)]
    same_csr(actual, expected)
    del capsule
    gc.collect()
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_columnar_feature_names(metal, name):
    table, frame = categorical(name, NAMES[1])
    d = xgb.DMatrix(table, np.arange(271) % 7)
    booster = xgb.train(dict(device="mps", nthread=2, base_score=0.5), d, 1)
    columns = {
        "wrong": native(name, np.ones(271, dtype=np.float32)),
        "category": native(name, np.zeros(271, dtype=np.int32)),
    }
    with pytest.raises(ValueError, match="feature_names mismatch"):
        booster.inplace_predict(xgb.MetalColumnar(columns))
    # Explicit caller names override wrapper names during construction.
    overridden = xgb.DMatrix(table, feature_names=["a", "b"])
    assert overridden.feature_names == ["a", "b"]
