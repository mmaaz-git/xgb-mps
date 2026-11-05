"""Synthetic report integrity/policy checks; none executes NVIDIA code."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("math_compare", Path(__file__).parents[1]/"scripts/compare_math_contracts.py")
math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math)


@lru_cache
def encoded(kind):
    groups = {i: {"input": np.zeros((rows, inputs), dtype="<u8"),
                  "output": np.zeros((rows, outputs), dtype="<u8")}
              for i, (rows, inputs, outputs) in enumerate(math.SHAPES)}
    return math.encode(kind, groups)


def reports():
    return [math.decode(bytearray(encoded(kind))) for kind in (1, 2)]


def compare(a, b):
    return math.compare(math.decode(a["data"]), math.decode(b["data"]))


def test_exact_then_single_bit_arithmetic_failure():
    metal, cuda = reports()
    result = compare(metal, cuda)
    assert result["status"] == "matches-declared-contract-on-recorded-inputs"
    assert result["output_fields"] == 1942163
    cuda["groups"][0]["output"][0, 0] = 1
    result = compare(metal, cuda)
    assert result["status"] == "contract-failed"
    assert result["exact_arithmetic_cast_failures"] == 1


def test_cast_signed_zero_requires_exact_bits():
    metal, cuda = reports()
    cuda["groups"][0]["output"][0, 4] = 0x80000000
    result = compare(metal, cuda)
    assert result["exact_arithmetic_cast_failures"] == 1
    assert result["differences"][0]["signed_zero_differences"] == 1


def test_math_one_ulp_requires_review_and_classification_change_fails():
    metal, cuda = reports()
    metal["groups"][2]["output"][0, 0] = 0x3ff0000000000000
    cuda["groups"][2]["output"][0, 0] = 0x3ff0000000000001
    result = compare(metal, cuda)
    assert result["status"] == "math-differences-require-review"
    assert result["differences"][0]["max_ulp_on_finite_pairs"] == 1
    cuda["groups"][2]["output"][0, 0] = 0x7ff0000000000000
    result = compare(metal, cuda)
    assert result["status"] == "contract-failed"
    assert result["math_classification_failures"] == 1


@pytest.mark.parametrize("group", range(4))
def test_nan_payloads_retained_under_existing_classification_policy(group):
    metal, cuda = reports()
    word = 0x7fc00000 if group == 3 else 0x7ff8000000000000
    metal["groups"][group]["output"][0, 0] = word
    cuda["groups"][group]["output"][0, 0] = word+1
    result = compare(metal, cuda)
    assert result["status"] == "matches-declared-contract-on-recorded-inputs"
    assert result["differences"][0]["nan_payload_differences"] == 1
    assert result["differences"][0]["classification_differences"] == 0


@pytest.mark.parametrize("corruption", ["magic", "mask", "shape", "truncated", "footer", "trailing", "fp32_upper"])
def test_invalid_framing_and_widths_rejected(corruption):
    report = math.decode(bytearray(encoded(1)))
    data = report["data"]
    if corruption == "magic":
        data[0] = 0
    elif corruption == "mask":
        struct.pack_into("<I", data, 12, 0)
    elif corruption == "shape":
        struct.pack_into("<I", data, 20, 0xffffffff)
    elif corruption == "truncated":
        data = data[:-100]
    elif corruption == "footer":
        data[-8] = 0
    elif corruption == "trailing":
        data = data+b"x"
    else:
        report["groups"][3]["input"][0, 0] = 1 << 32
    with pytest.raises(ValueError):
        math.decode(data)


def test_input_identity_backend_and_complete_group_requirements():
    metal, cuda = reports()
    cuda["groups"][1]["input"][0, 0] = 1
    with pytest.raises(ValueError, match="input bits"):
        compare(metal, cuda)
    with pytest.raises(ValueError, match="complete Metal"):
        math.compare(metal, metal)
    partial = math.decode(math.encode(1, {2: metal["groups"][2]}))
    with pytest.raises(ValueError, match="complete Metal"):
        math.compare(partial, cuda)
    inputs = math.decode(encoded(0))
    assert all(group["output"] is None for group in inputs["groups"].values())
    with pytest.raises(ValueError, match="complete Metal"):
        math.compare(inputs, cuda)


def test_partial_groups_reassemble_losslessly_and_outputs_are_exclusive(tmp_path):
    full = math.decode(encoded(1))
    groups = {}
    for ids in ((0, 1), (2,), (3,)):
        part = math.decode(math.encode(1, {i: full["groups"][i] for i in ids}))
        groups.update(part["groups"])
    assert math.encode(1, groups) == encoded(1)
    assert math.encode(0, groups) == encoded(0)
    path = tmp_path/"existing.bin"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        math.write(path, b"replacement")
    assert path.read_bytes() == b"keep"
