"""Verify the optional Metal API and clear failure in a non-Metal build."""
import ctypes
import json
import tempfile
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "build-cpu/lib/libxgboost.dylib"
lib = ctypes.CDLL(str(path))
lib.XGBGetLastError.restype = ctypes.c_char_p


def check(status):
    assert status == 0, lib.XGBGetLastError().decode()


out = ctypes.c_char_p()
check(lib.XGBuildInfo(ctypes.byref(out)))
assert json.loads(out.value)["USE_METAL"] is False
check(lib.XGBMetalInfo(ctypes.byref(out)))
info = json.loads(out.value)
assert info["built"] is False and info["available"] is False
assert all(value == 0 for key, value in info.items() if key not in {"built", "available"})
booster = ctypes.c_void_p()
check(lib.XGBoosterCreate(None, ctypes.c_uint64(0), ctypes.byref(booster)))
try:
    check(lib.XGBoosterSetParam(booster, b"device", b"mps"))
    check(lib.XGBoosterSetParam(booster, b"num_feature", b"2"))
    length = ctypes.c_uint64()
    status = lib.XGBoosterSaveJsonConfig(booster, ctypes.byref(length), ctypes.byref(out))
    assert status != 0
    assert b"USE_METAL=ON" in lib.XGBGetLastError()
finally:
    check(lib.XGBoosterFree(booster))
print("CPU-only build verified: Metal reports unavailable and device=mps raises a clear error.")

# The new construction selector must be usable for CPU and fail clearly for MPS.
proxy = ctypes.c_void_p()
check(lib.XGProxyDMatrixCreate(ctypes.byref(proxy)))
values = (ctypes.c_float * 2)(1, 2)
interface = json.dumps({"version": 3, "typestr": "<f4", "shape": [1, 2],
                        "data": [ctypes.addressof(values), False], "strides": None}).encode()
state = [False]
@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def reset(_):
    state[0] = False
@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
def next_batch(_):
    if state[0]:
        return 0
    state[0] = True
    return int(lib.XGProxyDMatrixSetDataDense(proxy, interface) == 0)
try:
    with tempfile.TemporaryDirectory(prefix="xgb-cpu-quantile-") as cache_dir:
        for constructor in [lib.XGQuantileDMatrixCreateFromCallback,
                            lib.XGExtMemQuantileDMatrixCreateFromCallback]:
            for device in ["cpu", "mps"]:
                output = ctypes.c_void_p()
                config = json.dumps({"device": device, "max_bin": 8, "nthread": 2,
                                     "missing": float("nan"), "on_host": False,
                                     "cache_prefix": str(Path(cache_dir) / "pages")}).encode()
                status = constructor(None, proxy, None, reset, next_batch, config,
                                     ctypes.byref(output))
                if device == "cpu":
                    check(status)
                    check(lib.XGDMatrixFree(output))
                else:
                    assert status != 0 and b"USE_METAL=ON" in lib.XGBGetLastError()
        assert not list(Path(cache_dir).iterdir())
finally:
    check(lib.XGDMatrixFree(proxy))
print("CPU-only in-memory/external quantile construction accepts CPU and rejects MPS explicitly.")
