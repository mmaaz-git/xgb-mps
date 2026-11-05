"""Synthetic framing and comparison-policy tests; no NVIDIA execution."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("ranking_compare", Path(__file__).parents[1]/"scripts/compare_ranking_contracts.py")
rr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rr)


@lru_cache
def encoded(kind):
    parts = [rr.MAGIC, struct.pack("<II", kind, 64)]
    for index in range(64):
        identity = rr.layout(index)
        rows, positions = identity[5:]
        count = rr.repetitions(kind, identity)
        parts.extend((struct.pack("<8I", *identity, count), bytes(rows*8+20+positions*16)))
        output = bytes(16+120+rows*8+(rows*16 if identity[0] else 0)+positions*16)
        for rep in range(count): parts.extend((struct.pack("<I", rep), output))
    return b"".join(parts)+b"RNDE"+struct.pack("<I", 64)


def reports():
    return [rr.decode(bytearray(encoded(k))) for k in (1, 2)]


def compare(a, b):
    return rr.compare(rr.decode(a["data"]), rr.decode(b["data"]))


def bits(value, width):
    return np.asarray(value, dtype="<f4" if width == 32 else "<f8").view("<u4" if width == 32 else "<u8").item()


def test_full_case_repeat_counts_and_exact_inputs():
    m, c = reports()
    assert len(m["cases"]) == len(c["cases"]) == 64
    assert sum(len(x["observations"]) for x in c["cases"]) == 1056
    assert compare(m, c)["status"] == "identical-on-recorded-inputs"
    c["cases"][63]["inputs"]["positive"][0] = 1
    with pytest.raises(ValueError, match="positive bits differ"):
        compare(m, c)
    with pytest.raises(ValueError, match="Metal and actual CUDA"):
        rr.compare(m, m)


def test_global_race_and_dependent_bias_are_reviewed():
    m, c = reports()
    obs = c["cases"][0]["observations"][0]
    obs["global_cost_rounding"][0] = bits(1., 64)
    obs["positive"][0] = bits(20., 64)
    result = compare(m, c)
    assert result["status"] == "differences-require-review"
    assert result["contract_failures"] == 0
    assert result["race_observations"][0]["cuda_distinct_complete_observations"] == 2
    assert len(result["race_observations"][0]["cuda_distinct_global_factor_bits"]) == 2
    assert {d["field"] for d in result["differences"]} == {"global_cost_rounding", "positive"}
    obs["global_cost_rounding"][0] = 0
    result = compare(m, c)
    assert result["status"] == "contract-failed" and result["contract_failures"] == 1


def test_original_gradient_budget_and_nan_classification():
    m, c = reports()
    gradient = c["cases"][0]["observations"][0]["gradients"]
    gradient[0, 0] = bits(1e-5, 32)
    assert compare(m, c)["status"] == "differences-require-review"
    gradient[0, 0] = bits(3e-5, 32)
    assert compare(m, c)["contract_failures"] == 1
    gradient[0, 0] = bits(np.nan, 32)
    assert compare(m, c)["differences"][0]["classification_differences"] == 1


def test_nonfinite_exponent_guard_is_separate_from_gradients():
    m, c = reports()
    obs = c["cases"][2]["observations"][0]
    obs["groups"]["cost_rounding"][0] = bits(np.inf, 64)
    obs["positive"][0] = bits(np.nan, 64)
    result = compare(m, c)
    assert result["status"] == "differences-require-review"
    assert all(d["domain_guard"] for d in result["differences"])
    obs["gradients"][0, 0] = bits(np.inf, 32)
    assert compare(m, c)["contract_failures"] == 1


def test_exact_weight_and_rounding_fields_keep_signed_zero():
    m, c = reports()
    obs = c["cases"][0]["observations"][0]
    obs["groups"]["weight"][0] = 0x80000000
    obs["groups"]["gradient_rounding"][0] = 1
    result = compare(m, c)
    assert result["contract_failures"] == 2
    assert sum(d["signed_zero_differences"] for d in result["differences"]) == 1
    m["cases"][0]["observations"][0]["groups"]["weight"][0] = 0x7fc00000
    obs["groups"]["weight"][0] = 0x7fc00001
    # Changing the single Metal baseline affects all 32 CUDA repetitions,
    # plus the independently changed gradient-rounding field in repetition 0.
    assert compare(m, c)["contract_failures"] == 33


@pytest.mark.parametrize("corruption", ["magic", "count", "shape", "repeats", "footer", "truncated", "trailing", "repeat_order"])
def test_bad_framing_cannot_be_a_completed_report(corruption):
    data = bytearray(encoded(2))
    if corruption == "magic": data[0] = 0
    elif corruption == "count": struct.pack_into("<I", data, 12, 63)
    elif corruption == "shape": struct.pack_into("<I", data, 36, 0xffffffff)
    elif corruption == "repeats": struct.pack_into("<I", data, 44, 1)
    elif corruption == "footer": data[-8] = 0
    elif corruption == "truncated": data = data[:-100]
    elif corruption == "trailing": data += b"x"
    else: struct.pack_into("<I", data, 48+386*8+20+8*16, 1)
    with pytest.raises(ValueError): rr.decode(data)


def test_unpacked_output_cannot_overwrite_existing_file(tmp_path, monkeypatch):
    source, target = tmp_path/"inputs.bin", tmp_path/"keep.bin"
    source.write_bytes(encoded(0)); target.write_bytes(b"keep")
    monkeypatch.setattr(sys, "argv", ["compare", "unpack", str(source), "--output", str(target)])
    assert rr.main() == 1
    assert target.read_bytes() == b"keep"
