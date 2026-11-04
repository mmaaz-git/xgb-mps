"""Native custom gradients follow CUDA packing and existing objective dispatch."""

import ctypes as C
import gc

import numpy as np
import pytest
from test_metal_tensor import producer

import xgboost as xgb
from xgboost._metal import _capsule, _Legacy, _Version, _Versioned
from xgboost.objective import Objective, TreeObjective


class DeviceOnly:
    """DLPack provider that forbids an accidental NumPy payload conversion."""

    def __init__(self, array):
        self.array = array

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        return self.array.__dlpack__(**kwargs)

    def __array__(self, *args, **kwargs):
        raise AssertionError("Native gradients must not be converted through NumPy")


def matrix(rows=137, targets=3):
    values = (np.arange(rows * 7).reshape(rows, 7) % 23).astype(np.float32)
    labels = np.zeros((rows, targets), dtype=np.float32)
    return xgb.QuantileDMatrix(values, labels, max_bin=17, nthread=2)


def params(device, targets):
    return dict(
        device=device,
        max_depth=3,
        max_bin=17,
        nthread=2,
        base_score=0.5,
        num_target=targets,
        multi_strategy="multi_output_tree" if targets > 1 else "one_output_per_tree",
    )


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.int8, np.int32, np.int64, np.bool_]
)
@pytest.mark.parametrize(
    "layout", ["vector", "matrix", "slice", "transpose", "broadcast"]
)
def test_native_boost(metal, name, device, dtype, layout):
    rows, targets = 137, 1 if layout == "vector" else 3
    g = (np.arange(rows * targets).reshape(rows, targets) % 9 - 4).astype(dtype)
    h = (np.arange(rows * targets).reshape(rows, targets) % 3 + 1).astype(np.float32)
    if layout == "vector":
        g, h = g[:, 0], h[:, 0]
    if layout == "transpose":
        ng, nh = producer(name, g.T.copy()).T, producer(name, h.T.copy()).T
    elif layout == "slice":
        ng = producer(name, np.repeat(g, 2, axis=0))[::2]
        nh = producer(name, np.repeat(h, 2, axis=1))[:, ::2]
    elif layout == "broadcast":
        g = np.broadcast_to(g[:1], g.shape)
        h = np.broadcast_to(h[:1], h.shape)
        if name == "torch":
            ng, nh = (
                producer(name, g[:1]).expand(rows, targets),
                producer(name, h[:1]).expand(rows, targets),
            )
        else:
            import mlx.core as mx

            ng, nh = (
                mx.broadcast_to(producer(name, g[:1]), g.shape),
                mx.broadcast_to(producer(name, h[:1]), h.shape),
            )
    else:
        ng, nh = producer(name, g), producer(name, h)
    data = matrix(rows, targets)
    actual = xgb.Booster(params(device, targets), [data])
    expected = xgb.Booster(params(device, targets), [data])
    before = xgb.metal_info()
    for iteration in range(2):
        actual.boost(data, iteration, grad=DeviceOnly(ng), hess=DeviceOnly(nh))
        expected.boost(data, iteration, grad=g.astype(np.float32), hess=h.copy())
    after = xgb.metal_info()
    assert (
        after["gradient_import_pairs"] - before["gradient_import_pairs"]
        == 2 * rows * targets
    )
    assert (
        after["gradient_import_dispatches"] - before["gradient_import_dispatches"] == 2
    )
    assert actual.save_raw() == expected.save_raw()
    np.testing.assert_array_equal(actual.predict(data), expected.predict(data))


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("column", [False, True])
def test_legacy_flattened_gradients(metal, name, column):
    data = matrix()
    g = (np.arange(411) % 9 - 4).astype(np.float32)
    h = np.ones_like(g)
    native = [producer(name, np.repeat(v, 2))[::2] for v in (g, h)]
    if column:
        native = [v.reshape(411, 1) for v in native]
    actual = xgb.Booster(params("mps", 3), [data])
    expected = xgb.Booster(params("mps", 3), [data])
    with pytest.warns(FutureWarning, match="shape of the gradient") as warnings:
        actual.boost(data, 0, grad=native[0], hess=native[1])
    assert len(warnings) == 2
    expected.boost(data, 0, grad=g.reshape(137, 3), hess=h.reshape(137, 3))
    assert actual.save_raw() == expected.save_raw()


class NativeObjective(Objective):
    def __init__(self, name):
        self.name = name

    def __call__(self, iteration, prediction, data):
        # Multiplication/subtraction by powers of two are exact here, allowing
        # comparison with the host callback without a loose training tolerance.
        if self.name:
            pred = producer(self.name, prediction)
            return pred * 2 - 4, pred * 0 + 2
        return prediction * 2 - 4, prediction * 0 + 2


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("kind", ["plain", "objective", "tree_without_split"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_objective_dispatch(metal, name, kind, device):
    data = matrix(targets=1)

    class NoSplit(NativeObjective, TreeObjective):
        def split_grad(self, iteration, grad, hess):
            return None

    def objective(source):
        if kind == "tree_without_split":
            return NoSplit(source)
        obj = NativeObjective(source)
        return (lambda p, d: obj(0, p, d)) if kind == "plain" else obj

    actual = xgb.train(params(device, 1), data, 3, obj=objective(name))
    expected = xgb.train(params(device, 1), data, 3, obj=objective(None))
    assert actual.save_raw() == expected.save_raw()


class Separate(TreeObjective):
    def __init__(self, name, placement):
        self.name, self.placement = name, placement

    def pair(self, values, native):
        return tuple(producer(self.name, v) if native else v for v in values)

    def __call__(self, iteration, prediction, data):
        # Distinct values and widths prevent the importer from silently reusing
        # the split pair as the leaf-value pair.
        values = np.arange(411).reshape(137, 3).astype(np.float32) % 7 - 3
        return self.pair(
            (values, np.full_like(values, 2)), self.placement in {"both", "value"}
        )

    def split_grad(self, iteration, grad, hess):
        values = np.arange(137).astype(np.float32) % 3 - 1
        return self.pair(
            (values, np.ones_like(values)), self.placement in {"both", "split"}
        )


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("placement", ["both", "split", "value"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_independent_split_value_pairs(metal, name, placement, device):
    data = matrix()
    before = xgb.metal_info()
    actual = xgb.train(params(device, 3), data, 2, obj=Separate(name, placement))
    after = xgb.metal_info()
    pairs = 4 if placement == "both" else 1 if placement == "split" else 3
    assert (
        after["gradient_import_pairs"] - before["gradient_import_pairs"]
        == 2 * 137 * pairs
    )
    expected = xgb.train(params(device, 3), data, 2, obj=Separate(None, "host"))
    assert actual.save_raw() == expected.save_raw()
    np.testing.assert_array_equal(actual.predict(data), expected.predict(data))


@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_aliased_capsule_consumed_once(metal, versioned, split):
    raw = producer("torch", np.ones((137, 1), dtype=np.float32))
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
    data = matrix(targets=1)
    config = params("mps", 1) | {"multi_strategy": "multi_output_tree"}
    actual = xgb.Booster(config, [data])
    expected = xgb.Booster(config, [data])

    class Alias(TreeObjective):
        def __init__(self, value):
            self.value = value

        def __call__(self, iteration, prediction, data):
            return self.value, self.value

        def split_grad(self, iteration, grad, hess):
            return self.value, self.value

    host = np.ones((137, 1), dtype=np.float32)
    if split:
        actual.boost(data, 0, fobj=Alias(capsule))
        expected.boost(data, 0, fobj=Alias(host))
    else:
        actual.boost(data, 0, grad=capsule, hess=capsule)
        expected.boost(data, 0, grad=host, hess=host)
    gc.collect()
    assert calls == [C.addressof(exported)]
    assert actual.save_raw() == expected.save_raw()
    del capsule
    gc.collect()
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_invalid_input_preserves_model(metal, name):
    data = matrix()
    model = xgb.Booster(params("mps", 3), [data])
    g, h = np.ones((137, 3), np.float32), np.ones((137, 3), np.float32)
    model.boost(data, 0, grad=g, hess=h)
    original = model.save_raw()
    bad = [
        (producer(name, g), h, "same device"),
        (g, producer(name, h), "same device"),
        (producer(name, g), producer(name, h[:, :2]), "Mismatched shape"),
        (producer(name, g[:136]), producer(name, h[:136]), "Mismatched size"),
        (
            producer(name, g.reshape(1, 411)),
            producer(name, h.reshape(1, 411)),
            "Mismatched size",
        ),
    ]
    for grad, hess, message in bad:
        with pytest.raises(xgb.core.XGBoostError, match=message):
            model.boost(data, 1, grad=grad, hess=hess)
        assert model.save_raw() == original
    with pytest.raises(ValueError, match="at most two"):
        model.boost(data, 1, grad=producer(name, g[:, :, None]), hess=producer(name, h))
    with (
        pytest.warns(FutureWarning),
        pytest.raises(ValueError, match="cannot be reshaped"),
    ):
        model.boost(
            data,
            1,
            grad=producer(name, np.ones(138, np.float32)),
            hess=producer(name, h),
        )
    assert model.save_raw() == original


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("rows", [0, 1])
def test_empty_and_scalar_inputs(metal, name, device, rows):
    data = matrix(rows, 1)
    actual = xgb.Booster(params(device, 1), [data])
    expected = xgb.Booster(params(device, 1), [data])
    host = np.ones(rows, dtype=np.float32)
    native = producer(name, host.reshape(()) if rows else host)
    actual.boost(data, 0, grad=native, hess=native)
    expected.boost(data, 0, grad=host, hess=host)
    assert actual.save_raw() == expected.save_raw()
