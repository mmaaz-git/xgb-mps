"""Synthetic AFT interchange checks; none of these tests execute NVIDIA code."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("aft_compare", Path(__file__).parents[1] / "scripts/compare_aft_contracts.py")
aft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aft)


@lru_cache
def encoded(kind):
    parts = [aft.MAGIC, struct.pack("<II", kind, aft.COUNT)]
    for i in range(aft.COUNT):
        identity = aft.layout(i)
        rows, weights = identity[-2:]
        parts.extend([struct.pack("<IIIII", *identity), np.ones(rows, dtype="<f4").tobytes(),
                      np.ones(rows, dtype="<f4").tobytes(), bytes(4*rows),
                      np.ones(weights, dtype="<f4").tobytes()])
        if kind:
            parts.append(bytes(aft.OUTPUT.itemsize*rows))
    return b"".join(parts) + b"AFTE" + struct.pack("<I", aft.COUNT)


def reports():
    return [aft.decode(bytearray(encoded(kind))) for kind in (1, 2)]


def compare(a, b):
    return aft.compare(aft.decode(a["data"]), aft.decode(b["data"]))


def test_exact_then_ordinary_failure():
    metal, cuda = reports()
    result = compare(metal, cuda)
    assert result["status"] == "exact-on-recorded-inputs" and result["rows"] == 316683
    cuda["cases"][0]["observations"]["weighted_gradient"][0] = 0x3f800000
    result = compare(metal, cuda)
    assert result["status"] == "ordinary-tolerance-failed"
    assert result["ordinary_tolerance_failures"] == 1


@pytest.mark.parametrize("region", ["sensitive", "guard"])
def test_tail_and_guard_changes_are_retained_for_review(region):
    metal, cuda = reports()
    index = 0 if region == "sensitive" else 25
    for report in (metal, cuda):
        report["cases"][index]["inputs"]["upper"][0] = 0x40000000
    cuda["cases"][index]["observations"]["weighted_hessian"][0] = 0x3f800000
    result = compare(metal, cuda)
    assert result["status"] == "differences-require-review"
    assert result["ordinary_tolerance_failures"] == 0
    assert result["differences"][0]["fields"]["weighted_hessian"]["first_differences"][0]["region"] == region


def test_nan_payload_and_classification_are_distinct():
    metal, cuda = reports()
    metal["cases"][25]["observations"]["gradient"][0] = 0x7ff8000000000000
    cuda["cases"][25]["observations"]["gradient"][0] = 0x7ff8000000000001
    field = compare(metal, cuda)["differences"][0]["fields"]["gradient"]
    assert field["nan_payload_differences"] == 1 and field["classification_differences"] == 0
    cuda["cases"][25]["observations"]["gradient"][0] = 0
    field = compare(metal, cuda)["differences"][0]["fields"]["gradient"]
    assert field["classification_differences"] == 1


@pytest.mark.parametrize("corruption", ["header", "count", "shape", "missing_case", "footer", "trailing"])
def test_invalid_framing_is_rejected(corruption):
    data = bytearray(encoded(1))
    if corruption == "header":
        data[0] = 0
    elif corruption == "count":
        struct.pack_into("<I", data, 12, 95)
    elif corruption == "shape":
        struct.pack_into("<I", data, 28, 2**32-1)
    elif corruption == "missing_case":
        del data[-1000:]
    elif corruption == "footer":
        data[-8] = 0
    else:
        data.append(0)
    with pytest.raises(ValueError):
        aft.decode(data)


def test_mismatched_inputs_and_backend_rejected():
    metal, cuda = reports()
    cuda["cases"][0]["inputs"]["prediction"][0] = 1
    with pytest.raises(ValueError, match="input bits"):
        compare(metal, cuda)
    with pytest.raises(ValueError, match="actual CUDA"):
        aft.compare(metal, metal)
    inputs = aft.decode(encoded(0))
    assert inputs["kind"] == 0 and all(c["observations"] is None for c in inputs["cases"])
    with pytest.raises(ValueError, match="actual CUDA"):
        aft.compare(inputs, cuda)
