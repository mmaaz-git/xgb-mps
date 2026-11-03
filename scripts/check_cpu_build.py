"""Verify the optional Metal API and clear failure in a non-Metal build."""
import ctypes
import json
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
assert info["histogram_dispatches"] == 0
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
