"""Synthetic report framing, state and tolerance checks; no NVIDIA execution."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("objective_compare", Path(__file__).parents[1]/"scripts/compare_ranking_objectives.py")
ro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ro)


@lru_cache
def encoded(kind, suite):
    specs = ro.specs(suite)
    parts = [ro.MAGIC, struct.pack("<III", kind, suite, len(specs))]
    for spec in specs:
        identity = spec["identity"]; rows, groups, weights, positions = identity[12:]
        parts.extend((struct.pack("<16IQ", *identity, spec["bias_norm"]), spec["groups"].tobytes(),
                      bytes((2*rows+weights)*4)))
        if kind: parts.append(bytes((2*rows+2*positions+1)*4))
    parts.extend((b"ROBE", struct.pack("<II", suite, len(specs))))
    return b"".join(parts)


def reports(suite=0):
    return [ro.decode(bytearray(encoded(kind, suite))) for kind in (1, 2)]


def compare(m, c):
    return ro.compare(ro.decode(m["data"]), ro.decode(c["data"]))


def words(values):
    return np.asarray(values, dtype="<f4").view("<u4")


@pytest.mark.parametrize("suite,cases,sequences,rows", [(0, 1442, 1442, 2444200), (1, 2596, 866, 2949920)])
def test_complete_matrices_and_identical_reports(suite, cases, sequences, rows):
    m, c = reports(suite)
    assert len(m["cases"]) == cases
    assert sum(x["identity"][1] == 0 for x in m["cases"]) == sequences
    assert sum(x["identity"][12] for x in m["cases"]) == rows
    assert compare(m, c)["status"] == "identical-on-recorded-inputs"


def test_rng_bit_changes_fail_and_input_bits_must_match():
    m, c = reports()
    c["cases"][1441]["observed"]["next_rng"][0] = 1
    result = compare(m, c)
    assert result["failures"] == 1 and result["differences"][0]["field"] == "next_rng"
    c["cases"][1441]["inputs"]["prediction"][0] = 1
    with pytest.raises(ValueError, match="prediction input bits differ"): compare(m, c)
    with pytest.raises(ValueError, match="Metal and actual CUDA"): ro.compare(m, m)


def test_subnormal_loss_cannot_hide_under_ordinary_absolute_budget():
    a = np.asarray([1000], dtype="<u4")
    assert ro.difference(a, np.asarray([0], dtype="<u4"), strict_subnormal=True)["subnormal_failures"] == 1
    assert ro.difference(np.asarray([0], dtype="<u4"), a, strict_subnormal=True)["subnormal_failures"] == 1
    assert ro.difference(a, np.asarray([999], dtype="<u4"), strict_subnormal=True)["tolerance_or_exact_failures"] == 0


def test_original_ordinary_budget_classification_and_signed_zero():
    assert ro.difference(words([0]), words([1e-6]))["tolerance_or_exact_failures"] == 0
    assert ro.difference(words([0]), words([3e-6]))["tolerance_or_exact_failures"] == 1
    assert ro.difference(words([0]), words([np.nan]))["classification_differences"] == 1
    assert ro.difference(words([np.inf]), words([-np.inf]))["tolerance_or_exact_failures"] == 1
    diff = ro.difference(words([0]), words([-0.]))
    assert diff["bit_differences"] == 1 and diff["tolerance_or_exact_failures"] == 0


def test_bias_differences_propagate_in_report_without_reset_or_waiver():
    m, c = reports(1)
    c["cases"][4]["observed"]["positive"][0] = words([1e-6])[0]
    c["cases"][5]["observed"]["gradient"][0, 0] = words([1e-6])[0]
    result = compare(m, c)
    assert result["status"] == "differences-require-review"
    assert result["first_differing_iteration_by_sequence"] == {"2": 0}
    assert result["differences"][1]["prior_serialized_bias_difference"]
    c["cases"][5]["observed"]["gradient"][0, 0] = words([3e-6])[0]
    assert compare(m, c)["failures"] == 1


def test_sequence_labels_cannot_change_between_calls():
    report = ro.decode(bytearray(encoded(0, 1)))
    report["cases"][5]["inputs"]["label"][0] = 1
    with pytest.raises(ValueError, match="sequence metadata changed"):
        ro.decode(report["data"])


@pytest.mark.parametrize("corruption", ["magic", "count", "iteration", "seed", "bias_norm", "group_boundary", "truncated", "footer", "trailing"])
def test_corrupt_or_partial_objective_reports_rejected(corruption):
    data = bytearray(encoded(2, 0))
    if corruption == "magic": data[0] = 0
    elif corruption == "count": struct.pack_into("<I", data, 16, 1441)
    elif corruption == "iteration": struct.pack_into("<I", data, 24, 1)
    elif corruption == "seed": struct.pack_into("<I", data, 64, 0)
    elif corruption == "bias_norm": struct.pack_into("<Q", data, 84, 0)
    elif corruption == "group_boundary": struct.pack_into("<I", data, 96, 2)
    elif corruption == "truncated": data = data[:-100]
    elif corruption == "footer": data[-12] = 0
    else: data += b"x"
    with pytest.raises(ValueError): ro.decode(data)


def test_unpacked_input_file_is_exclusive(tmp_path, monkeypatch):
    source, output = tmp_path/"inputs.bin", tmp_path/"existing.bin"
    source.write_bytes(encoded(0, 0)); output.write_bytes(b"keep")
    monkeypatch.setattr(sys, "argv", ["compare", "unpack", str(source), "--output", str(output)])
    assert ro.main() == 1
    assert output.read_bytes() == b"keep"
