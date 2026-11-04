"""Native Metal DLPack ownership and descriptors (no tensor payload readback)."""

import ctypes as C
import json
import sys
import warnings
import weakref
from typing import Any


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
    if isinstance(data, MetalData):
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


def transform(data: Any, *, allow_tensor_shape: bool = False) -> MetalData:
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
