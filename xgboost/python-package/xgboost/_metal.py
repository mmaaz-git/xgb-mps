"""Native Metal tensor ownership, descriptors and explicit output readback."""

import ctypes as C
import json
import sys
import warnings
import weakref
from collections.abc import Mapping
from typing import Any

import numpy as np

from ._c_api import _LIB, _check_call


class _Device(C.Structure):
    _fields_ = [("kind", C.c_int32), ("index", C.c_int32)]


class _DType(C.Structure):
    _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]


class _Tensor(C.Structure):
    _fields_ = [
        ("data", C.c_void_p),
        ("device", _Device),
        ("ndim", C.c_int32),
        ("dtype", _DType),
        ("shape", C.POINTER(C.c_int64)),
        ("strides", C.POINTER(C.c_int64)),
        ("offset", C.c_uint64),
    ]


class _Legacy(C.Structure):
    _fields_ = [("tensor", _Tensor), ("manager", C.c_void_p), ("deleter", C.c_void_p)]


class _Version(C.Structure):
    _fields_ = [("major", C.c_uint32), ("minor", C.c_uint32)]


class _Versioned(C.Structure):
    _fields_ = [
        ("version", _Version),
        ("manager", C.c_void_p),
        ("deleter", C.c_void_p),
        ("flags", C.c_uint64),
        ("tensor", _Tensor),
    ]


_valid = C.pythonapi.PyCapsule_IsValid
_valid.argtypes, _valid.restype = [C.py_object, C.c_char_p], C.c_int
_pointer = C.pythonapi.PyCapsule_GetPointer
_pointer.argtypes, _pointer.restype = [C.py_object, C.c_char_p], C.c_void_p
_rename = C.pythonapi.PyCapsule_SetName
_rename.argtypes, _rename.restype = [C.py_object, C.c_char_p], C.c_int
# Capsule names must stay alive; these are module constants, never temporary strings.
_USED = b"used_dltensor"
_USED_VERSIONED = b"used_dltensor_versioned"


def _capsule(data: Any) -> Any:
    if _valid(data, b"dltensor"):
        ptr = _pointer(data, b"dltensor")
        value = C.cast(ptr, C.POINTER(_Legacy)).contents
        return ptr, value, _USED
    if _valid(data, b"dltensor_versioned"):
        ptr = _pointer(data, b"dltensor_versioned")
        version = C.cast(ptr, C.POINTER(_Version)).contents
        if version.major != 1:
            raise BufferError(f"Unsupported DLPack major version: {version.major}")
        value = C.cast(ptr, C.POINTER(_Versioned)).contents
        return ptr, value, _USED_VERSIONED
    if _valid(data, _USED) or _valid(data, _USED_VERSIONED):
        raise ValueError("DLPack capsule has already been consumed.")
    return None


def is_metal(data: Any) -> bool:
    if isinstance(data, (MetalData, MetalColumnar)):
        return True
    if type(data).__name__ == "PyCapsule":
        capsule = _capsule(data)
        return capsule is not None and capsule[1].tensor.device.kind == 8
    device = getattr(data, "__dlpack_device__", None)
    return device is not None and device()[0] == 8


def _synchronize() -> None:
    # DLPack's stream argument does not define a portable Metal queue handshake.
    # Known producers are explicitly completed; foreign/raw-capsule producers
    # must supply ready buffers. Do not import either optional runtime eagerly.
    torch = sys.modules.get("torch")
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()
    mlx = sys.modules.get("mlx.core")
    if mlx is not None and mlx.metal.is_available():
        mlx.synchronize()


class MetalData:
    """Own a single DLPack export until synchronous Metal import has completed.

    Producer tensors are not converted through NumPy or copied to host memory.
    Generic Metal DLPack providers must finish their writes before exporting.
    """

    def __init__(self, data: Any, *, allow_tensor_shape: bool = False) -> None:
        if type(data).__name__ == "PyCapsule":
            capsule = data
        else:
            # Legacy export remains supported by both current and older producers.
            capsule = data.__dlpack__(stream=None)
        record = _capsule(capsule)
        if record is None:
            raise TypeError("Expected a DLPack tensor capsule.")
        pointer, managed, used_name = record
        tensor = managed.tensor
        if tensor.device.kind != 8 or tensor.device.index != 0:
            raise BufferError("Expected a Metal DLPack tensor on device 0.")
        valid_dim = tensor.ndim in (0, 1, 2) if allow_tensor_shape else tensor.ndim == 2
        if not valid_dim or (tensor.ndim and not tensor.shape):
            raise ValueError(
                "Metal tensor must have at most two dimensions."
                if allow_tensor_shape
                else "Metal feature input must be two-dimensional."
            )
        shape = tuple(tensor.shape[i] for i in range(tensor.ndim))
        if any(size < 0 for size in shape):
            raise ValueError("Invalid Metal tensor shape.")
        strides = (
            tuple(tensor.strides[i] for i in range(tensor.ndim))
            if tensor.strides
            else (
                () if tensor.ndim == 0 else (1,) if tensor.ndim == 1 else (shape[1], 1)
            )
        )
        self.shape = shape
        self._descriptor = json.dumps(
            dict(
                device_type=8,
                device_id=0,
                buffer=tensor.data or 0,
                byte_offset=tensor.offset,
                shape=shape,
                strides=strides,
                dtype_code=tensor.dtype.code,
                dtype_bits=tensor.dtype.bits,
                dtype_lanes=tensor.dtype.lanes,
            )
        ).encode("utf-8")
        # Keep metadata and the producer allocation alive across iterator resets.
        # Rename exactly once; the producer capsule destructor then becomes a no-op.
        self._capsule = capsule
        self._producer = data
        if managed.deleter:
            deleter = C.PYFUNCTYPE(None, C.c_void_p)(managed.deleter)
            self._release = weakref.finalize(self, deleter, pointer)
        try:
            _rename(capsule, used_name)
        except BaseException:
            # Until renaming succeeds, the capsule destructor still owns release.
            if managed.deleter:
                self._release.detach()
            raise

    @property
    def descriptor(self) -> bytes:
        # Iterators may retain this export while updating the producer in place.
        # Complete writes immediately before each snapshot, including cached exports.
        _synchronize()
        return self._descriptor

    def gradient_descriptor(self, n_samples: int) -> bytes:
        """Preserve the existing flattened-gradient compatibility path on GPU."""
        descriptor = self.descriptor
        if (
            self.shape
            and self.shape[0] != n_samples
            and (len(self.shape) == 1 or self.shape[1] == 1)
        ):
            warnings.warn(
                "Since 2.1.0, the shape of the gradient and hessian is required to"
                " be (n_samples, n_targets) or (n_samples, n_classes).",
                FutureWarning,
            )
            size = self.shape[0]
            if not n_samples or size % n_samples:
                raise ValueError(
                    "Gradient size cannot be reshaped to the training row count."
                )
            desc = json.loads(descriptor)
            targets = size // n_samples
            stride = desc["strides"][0]
            desc["shape"] = (n_samples, targets)
            desc["strides"] = (stride * targets, stride)
            descriptor = json.dumps(desc).encode("utf-8")
        return descriptor


class MetalColumn:
    """One native Metal column, optionally masked or dictionary encoded.

    ``values`` is a one-dimensional Metal DLPack tensor (PyTorch MPS and MLX
    are supported). ``mask`` is an optional contiguous uint8 Metal tensor of
    packed validity bits: the least significant bit is first, and 1 means valid.
    ``mask_bit`` selects the first row's bit within the first byte (0 through 7).

    For a categorical column, ``values`` contains dictionary codes and
    ``categories`` is either an integer Metal tensor of names, or a pair
    ``(offsets, values)`` of int32 offsets and int8 UTF-8 bytes on Metal. Category
    names must be nonempty. Use ``enable_categorical=True`` when making a matrix.
    These are experimental interfaces specific to this Metal fork.
    """

    def __init__(
        self,
        values: Any,
        *,
        mask: Any = None,
        mask_bit: int = 0,
        categories: Any = None,
    ) -> None:
        self.values = values
        self.mask = mask
        self.mask_bit = mask_bit
        self.categories = categories


class MetalColumnar:
    """A native Metal table with independently typed one-dimensional columns.

    Pass a mapping from feature names to tensors or :py:class:`MetalColumn`
    objects, or a sequence of unnamed columns. All columns must have equal row
    counts. The table retains DLPack exports across iterator resets; each C API
    import makes an owned GPU snapshot. Payloads are never converted via NumPy.
    Known PyTorch/MLX producers are synchronized before each import. Other
    producers and raw capsules must supply completed buffers.

    Example::

        table = xgboost.MetalColumnar({"size": mps_float_tensor,
                                      "count": mps_integer_tensor})
        matrix = xgboost.QuantileDMatrix(table, labels)
    """

    def __init__(self, columns: Any) -> None:
        self.feature_names = list(columns) if isinstance(columns, Mapping) else None
        if self.feature_names is not None:
            if not all(isinstance(name, str) for name in self.feature_names):
                raise TypeError("Metal feature names must be strings.")
            columns = list(columns.values())
        else:
            columns = list(columns)
        if not columns:
            raise ValueError("The number of columns must not equal to 0.")
        captured = {}

        def vector(value: Any) -> MetalData:
            key = id(value)
            if key not in captured:
                if not is_metal(value) or isinstance(value, MetalColumnar):
                    raise TypeError(
                        "Metal columns require native Metal DLPack tensors."
                    )
                captured[key] = transform(value, allow_tensor_shape=True)
            tensor = captured[key]
            if len(tensor.shape) != 1:
                raise ValueError("Metal columns must be vectors.")
            return tensor

        self._columns = []
        self.feature_types = []
        rows = None
        for column in columns:
            column = column if isinstance(column, MetalColumn) else MetalColumn(column)
            values = vector(column.values)
            if rows is None:
                rows = values.shape[0]
            if rows != values.shape[0]:
                raise ValueError("All columns should have the same number of rows.")
            entry = {"values": values}
            if column.mask is not None:
                if not isinstance(column.mask_bit, int) or not 0 <= column.mask_bit < 8:
                    raise ValueError("mask_bit must be an integer from 0 through 7.")
                entry.update(mask=vector(column.mask), mask_bit=column.mask_bit)
            elif column.mask_bit:
                raise ValueError("mask_bit requires a validity mask.")
            if column.categories is not None:
                names = column.categories
                if isinstance(names, tuple) and len(names) == 2:
                    entry["categories"] = dict(
                        offsets=vector(names[0]), values=vector(names[1])
                    )
                else:
                    entry["categories"] = vector(names)
            self._columns.append(entry)
            self.feature_types.append("c" if column.categories is not None else "q")
        self.shape = (rows, len(columns))
        self._reference = None

    def _with_reference(self, reference: Any) -> "MetalColumnar":
        # Reuse retained exports without mutating a caller's table or re-consuming
        # its capsules when two matrices use different reference dictionaries.
        result = object.__new__(MetalColumnar)
        result.__dict__ = self.__dict__.copy()
        result._reference = reference
        return result

    @property
    def descriptor(self) -> bytes:
        """Synchronize producers and describe the retained native buffers."""
        _synchronize()

        def encode(value: Any) -> Any:
            if isinstance(value, MetalData):
                return json.loads(value._descriptor)
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            return value

        result = dict(columns=[encode(column) for column in self._columns])
        if self._reference is not None and not self._reference.empty():
            result["ref_categories"] = self._reference.get_handle()
        return json.dumps(result).encode("utf-8")


def transform_features(data, feature_names, feature_types, enable_categorical):
    """Infer a native table's schema and retain optional reference categories."""
    from ._data_utils import Categories, get_ref_categories

    data = transform(data)
    if isinstance(data, MetalColumnar):
        feature_types, reference = get_ref_categories(feature_types)
        if "c" in data.feature_types and not enable_categorical:
            raise ValueError(
                "Categorical Metal columns require enable_categorical=True."
            )
        feature_names = data.feature_names if feature_names is None else feature_names
        feature_types = data.feature_types if feature_types is None else feature_types
        return data._with_reference(reference), feature_names, feature_types
    if isinstance(feature_types, Categories):
        raise ValueError("Reference category requires columnar input.")
    return data, feature_names, feature_types


def transform(data: Any, *, allow_tensor_shape: bool = False) -> Any:
    if isinstance(data, MetalColumnar):
        if allow_tensor_shape:
            raise TypeError(
                "Metal columnar tables cannot be used as metadata or gradients."
            )
        return data
    return (
        data
        if isinstance(data, MetalData)
        else MetalData(data, allow_tensor_shape=allow_tensor_shape)
    )


def capture_tensors(*arrays: Any) -> tuple:
    """Retain managed exports through a C call, consuming aliased capsules once."""
    captured = {}
    result = []
    for array in arrays:
        key = id(array)
        if key in captured:
            result.append(captured[key])
        elif is_metal(array):
            tensor = transform(array, allow_tensor_shape=True)
            captured[key] = tensor
            result.append(tensor)
        else:
            result.append(array)
    return tuple(result)


class MetalOutput:
    """An owned float32 Metal prediction, exported through DLPack.

    Request with ``predict(..., output_device="mps")`` or
    ``inplace_predict(..., output_device="mps")``. Use ``torch.from_dlpack(result)``
    or ``mlx.core.from_dlpack(result)`` to consume the buffer on GPU, or
    :py:meth:`numpy` for an explicit host copy. There is no implicit NumPy
    conversion. The allocation survives model deletion and subsequent predictions.

    Each DLPack export retains the allocation independently. Default exports
    share writable storage; consumer writes are visible through other exports.
    ``__dlpack__(copy=True)`` instead makes an independent GPU copy. Generic
    consumers must finish writes before re-export or host readback; loaded
    PyTorch/MLX runtimes are synchronized automatically. Metal stream handles
    have no portable DLPack handshake, so only ``stream=None`` is supported.
    """

    def __init__(self) -> None:
        raise TypeError("MetalOutput is created by XGBoost prediction methods.")

    @classmethod
    def _from_prediction(cls, shape: Any, dims: Any, values: Any) -> "MetalOutput":
        result = object.__new__(cls)
        result._shape = tuple(int(shape[i]) for i in range(dims.value))
        result._size = 1
        for length in result._shape:
            result._size *= length
        handle = C.c_void_p()
        _check_call(_LIB.XGMetalOutputCreate(values, shape, dims, C.byref(handle)))
        result._handle = handle
        result._release = weakref.finalize(result, _LIB.XGMetalOutputFree, handle)
        return result

    @property
    def shape(self) -> tuple:
        """Prediction shape, identical to the corresponding NumPy result."""
        return self._shape

    @property
    def size(self) -> int:
        """Number of float32 values."""
        return self._size

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def dtype(self) -> np.dtype:
        """Float32, as in XGBoost's prediction C API."""
        return np.dtype(np.float32)

    def __dlpack_device__(self) -> tuple:
        return (8, 0)

    def __dlpack__(
        self, stream=None, *, max_version=None, dl_device=None, copy=None
    ) -> Any:
        if stream is not None:
            raise BufferError("Metal DLPack export requires stream=None.")
        if dl_device is not None and tuple(dl_device) != (8, 0):
            raise BufferError(
                "MetalOutput supports DLPack exports only on Metal device 0."
            )
        if copy is not None and not isinstance(copy, bool):
            raise TypeError("copy must be None, True or False.")
        if max_version is not None and (
            not isinstance(max_version, tuple)
            or len(max_version) != 2
            or any(not isinstance(v, int) or v < 0 for v in max_version)
        ):
            raise TypeError("max_version must be a pair of nonnegative integers.")
        versioned = max_version is not None and max_version >= (1, 0)
        _synchronize()
        destructor = C.c_void_p()
        _check_call(_LIB.XGMetalOutputCapsuleDestructor(C.byref(destructor)))
        managed, name = C.c_void_p(), C.c_char_p()
        _check_call(
            _LIB.XGMetalOutputExportDLPack(
                self._handle,
                int(versioned),
                int(copy is True),
                C.byref(managed),
                C.byref(name),
            )
        )
        new_capsule = C.pythonapi.PyCapsule_New
        new_capsule.argtypes, new_capsule.restype = (
            [C.c_void_p, C.c_char_p, C.c_void_p],
            C.py_object,
        )
        try:
            # Use the native static name pointer, not a temporary Python bytes
            # buffer; capsules can outlive Python module/global teardown.
            return new_capsule(managed, name, destructor)
        except BaseException:
            typ = _Versioned if versioned else _Legacy
            value = C.cast(managed, C.POINTER(typ)).contents
            C.CFUNCTYPE(None, C.c_void_p)(value.deleter)(managed)
            raise

    def numpy(self) -> np.ndarray:
        """Synchronize consumers and copy the result to a new host array."""
        _synchronize()
        result = np.empty(self.shape, dtype=np.float32)
        _check_call(
            _LIB.XGMetalOutputRead(
                self._handle,
                result.ctypes.data_as(C.POINTER(C.c_float)),
                C.c_uint64(self.size),
            )
        )
        return result

    def __repr__(self) -> str:
        return f"MetalOutput(shape={self.shape}, dtype=float32, device=mps)"

    def __array__(self, *args, **kwargs):
        raise TypeError("Use MetalOutput.numpy() for an explicit host copy.")
