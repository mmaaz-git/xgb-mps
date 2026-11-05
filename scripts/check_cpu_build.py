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
for key in ("prediction_model_pack_dispatches", "prediction_model_shared_input_bytes",
            "prediction_model_staging_bytes"):
    assert info[key] == 0
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

# Native entry points exist in CPU builds but reject before touching Metal handles.
proxy = ctypes.c_void_p()
booster = ctypes.c_void_p()
check(lib.XGProxyDMatrixCreate(ctypes.byref(proxy)))
check(lib.XGBoosterCreate(None, ctypes.c_uint64(0), ctypes.byref(booster)))
try:
    result = ctypes.c_void_p()
    assert lib.XGDMatrixCreateFromMetal(b"{}", b"{}", ctypes.byref(result)) != 0
    assert b"USE_METAL=ON" in lib.XGBGetLastError()
    assert not result.value
    assert lib.XGProxyDMatrixSetDataMetal(proxy, b"{}") != 0
    assert b"USE_METAL=ON" in lib.XGBGetLastError()
    assert lib.XGDMatrixSetInfoFromMetal(proxy, b"label", b"{}") != 0
    assert b"USE_METAL=ON" in lib.XGBGetLastError()
    native = b'{"device_type":8}'
    assert lib.XGBoosterTrainOneIter(booster, proxy, 0, native, native) != 0
    assert b"USE_METAL=ON" in lib.XGBGetLastError()
    assert lib.XGBoosterTrainOneIterWithSplitGrad(booster, proxy, 0, native, native, native, native) != 0
    assert b"USE_METAL=ON" in lib.XGBGetLastError()
    assert lib.XGBoosterPredictFromMetal(booster, b"{}", b"{}", None, None, None, None) != 0
    assert b"USE_METAL=ON" in lib.XGBGetLastError()
finally:
    check(lib.XGDMatrixFree(proxy))
    check(lib.XGBoosterFree(booster))
print("CPU-only native Metal tensor entry points reject input explicitly.")

# Output APIs remain exported without Metal, with deterministic unset handles.
result, managed, destructor = ctypes.c_void_p(1), ctypes.c_void_p(1), ctypes.c_void_p(1)
name = ctypes.c_char_p(b"sentinel")
for call, output in [
    (lambda: lib.XGMetalOutputCreate(None, None, ctypes.c_uint64(0), ctypes.byref(result)), result),
    (lambda: lib.XGMetalOutputExportDLPack(None, 0, 0, ctypes.byref(managed), ctypes.byref(name)), managed),
    (lambda: lib.XGMetalOutputCapsuleDestructor(ctypes.byref(destructor)), destructor),
]:
    assert call() != 0 and b"USE_METAL=ON" in lib.XGBGetLastError()
    assert output.value is None
assert name.value is None
assert lib.XGMetalOutputRead(None, None, ctypes.c_uint64(0)) != 0
assert b"USE_METAL=ON" in lib.XGBGetLastError()
check(lib.XGMetalOutputFree(None))
print("CPU-only Metal output APIs reject explicitly and leave output handles unset.")

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
            for on_host in [False, True]:
                for device in ["cpu", "mps"]:
                    output = ctypes.c_void_p()
                    config = json.dumps({"device": device, "max_bin": 8, "nthread": 2,
                                         "missing": float("nan"), "on_host": on_host,
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

# Ordinary external caches remain disk-backed in a CPU-only build, including when
# the iterator's GPU host-cache option is true. Exercise regeneration of that path.
proxy = ctypes.c_void_p()
check(lib.XGProxyDMatrixCreate(ctypes.byref(proxy)))
try:
    with tempfile.TemporaryDirectory(prefix="xgb-cpu-ordinary-") as cache_dir:
        for on_host in [False, True]:
            matrix = ctypes.c_void_p()
            config = json.dumps({"nthread": 2, "missing": float("nan"), "on_host": on_host,
                                 "cache_prefix": str(Path(cache_dir) / "ordinary")}).encode()
            check(lib.XGDMatrixCreateFromCallback(None, proxy, reset, next_batch, config,
                                                  ctypes.byref(matrix)))
            booster = ctypes.c_void_p()
            try:
                label = (ctypes.c_float * 1)(1.)
                check(lib.XGDMatrixSetFloatInfo(matrix, b"label", label, ctypes.c_uint64(1)))
                handles = (ctypes.c_void_p * 1)(matrix.value)
                check(lib.XGBoosterCreate(handles, ctypes.c_uint64(1), ctypes.byref(booster)))
                for key, value in [(b"device", b"cpu"), (b"nthread", b"2"),
                                   (b"tree_method", b"hist"), (b"base_score", b"0.5")]:
                    check(lib.XGBoosterSetParam(booster, key, value))
                for iteration, bins in enumerate([8, 17, 8]):
                    check(lib.XGBoosterSetParam(booster, b"max_bin", str(bins).encode()))
                    check(lib.XGBoosterUpdateOneIter(booster, iteration, matrix))
                assert len(list(Path(cache_dir).glob("*.row.page"))) == 1
                assert len(list(Path(cache_dir).glob("*.gradient_index.page"))) == 1
                assert not list(Path(cache_dir).glob("*.metal-ellpack.page"))
            finally:
                if booster.value:
                    check(lib.XGBoosterFree(booster))
                check(lib.XGDMatrixFree(matrix))
            assert not list(Path(cache_dir).iterdir())
finally:
    check(lib.XGDMatrixFree(proxy))
print("CPU-only ordinary external caches train/regenerate/clean up for both host-cache options.")
