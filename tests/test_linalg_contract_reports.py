"""Synthetic report/reference checks; these are never NVIDIA execution evidence."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("linalg_compare", Path(__file__).parents[1]/"scripts/compare_linalg_contracts.py")
lc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lc)


@lru_cache
def encoded(kind):
    parts = [lc.MAGIC, struct.pack("<II", kind, lc.COUNT)]
    for s in lc.specs():
        values, weights, initial = [np.zeros(n, dtype="<u4") for n in (s.values, s.weights, s.initial)]
        actual = np.zeros(s.outputs, dtype="<u4")
        oracle = np.zeros(s.columns if s.kind in (0, 2, 4) else 0, dtype=lc.ORACLE)
        if s.kind in (0, 4):
            oracle["count"][0] = s.rows
            result = s.rows*(1. if s.kind == 0 else .5) if s.mode == 0 else 0.
            oracle["sum"][0] = oracle["magnitude"][0] = result
            oracle["rounded"][0] = actual[0] = lc.bits(result)
        if s.kind == 2:
            oracle["count"] = s.rows
            if s.mode == 6:
                values[[0, 64, 128]] = [lc.bits(2.**24), lc.bits(-2.**24), lc.bits(1.)]
                oracle["sum"] = 1/256; oracle["magnitude"] = 131072+1/256
                oracle["rounded"] = lc.bits(1/256)
        if s.kind == 1:
            p = struct.unpack("<d", struct.pack("<Q", s.parameter))[0]
            with np.errstate(all="ignore"):
                v = np.float32(0.)
                if s.op == 0: result = np.float32(np.float64(v)*p)
                elif s.op == 1: result = np.float32(np.float64(v)*(np.float64(1.)/p))
                elif s.op == 2: result = np.float32(np.log(np.float64(np.float32(v+np.float32(p)))))
                else: result = np.float32(v-np.float32(p))
            actual[:s.rows*s.stride0:s.stride0] = lc.bits(result)
        parts += [struct.pack("<13IQ", *s), values.tobytes(), weights.tobytes(), initial.tobytes()]
        if kind:
            status = (2 if kind == 2 else 1) if s.kind == 3 else 0
            parts.append(struct.pack("<I", status))
            if status != 2: parts.append(actual.tobytes())
        if kind == 1:
            if s.kind == 1: parts.append(actual.tobytes())
            parts.append(oracle.tobytes())
    return b"".join(parts)+b"LINE"+struct.pack("<I", lc.COUNT)


def reports(): return [lc.decode(bytearray(encoded(k))) for k in (1, 2)]


def compare(m, c): return lc.compare(lc.decode(m["data"]), lc.decode(c["data"]))


def find(**fields):
    return next(s.id for s in lc.specs() if all(getattr(s, name) == val for name, val in fields.items()))


def test_complete_corpus_includes_domain_guards_and_empty_operations():
    m, c = reports()
    assert [sum(s.kind == k for s in lc.specs()) for k in range(7)] == [141, 80, 33, 3, 1, 1, 1]
    assert sum(s.values+s.weights+s.initial for s in lc.specs()) == 21611969
    assert sum(s.outputs for s in lc.specs()) == 10508701
    assert sum(s.outputs for s in lc.specs() if s.kind == 1) == 10487920
    assert sum(s.columns for s in lc.specs() if s.kind in (0, 2, 4)) == 11360
    lc.match_inputs(lc.decode(encoded(0)), m)
    result = compare(m, c)
    assert result["cuda_executed_cases"] == 257 and len(result["domain_exclusions"]) == 3
    assert not result["reference_failures"] and not result["different_fields"]
    assert result["status"] == "common-domain-identical-with-exclusions"
    assert len(c["cases"][254]["actual"]) == 0
    assert not lc.validate_metal(m)
    with pytest.raises(ValueError, match="Metal and actual CUDA"): lc.compare(m, m)


def test_identical_wrong_outputs_fail_independent_references():
    m, c = reports()
    for report in (m, c): report["cases"][0]["actual"][0] = lc.bits(1.)
    result = compare(m, c)
    assert len(result["reference_failures"]) == 2 and not result["different_fields"]


def test_scaling_requires_exact_original_ulps():
    m, c = reports(); index = find(kind=1, op=0)
    c["cases"][index]["actual"][0] = 1  # Smallest subnormal versus zero.
    assert compare(m, c)["reference_failures"][0]["policy"] == "original-vector-ulp"


def test_log_budget_retained_and_all_differences_counted():
    m, c = reports(); index = find(kind=1, op=2, parameter=lc.bits(1., 64))
    c["cases"][index]["actual"][:50] = 1
    result = compare(m, c)
    assert not result["reference_failures"] and result["different_fields"] == 50
    assert len(result["differences"][0]["first_fields"]) == 32
    c["cases"][index]["actual"][0] = 2
    assert compare(m, c)["reference_failures"][0]["count"] == 1


def test_signed_zero_and_nan_payloads_remain_visible():
    m, c = reports(); index = find(kind=1, op=0)
    c["cases"][index]["actual"][0] = lc.bits(-0.)
    result = compare(m, c)
    assert not result["reference_failures"] and result["different_fields"] == 1
    for report in (m, c): report["cases"][index]["actual"][1] = 0x7fc00001 if report is m else 0xffc00002
    m["cases"][index]["reference"][1] = 0x7fc00000
    result = compare(m, c)
    assert not result["reference_failures"] and result["different_fields"] == 2
    c["cases"][index]["actual"][1] = lc.bits(float("inf"))
    assert compare(m, c)["reference_failures"]


def test_log_guard_storage_is_bit_exact_even_with_one_ulp_budget():
    m, c = reports(); index = find(kind=1, op=2, stride0=3, parameter=lc.bits(1., 64))
    c["cases"][index]["actual"][1] = 1
    result = compare(m, c)
    assert [e["policy"] for e in result["reference_failures"]] == ["untouched-vector-guard"]


def test_reduction_guard_and_zero_mode_remain_exact():
    m, c = reports(); index = find(kind=0, rows=257, mode=4)
    c["cases"][index]["actual"][:2] = [1, 1]
    result = compare(m, c)
    assert {e["policy"] for e in result["reference_failures"]} == {"original-reduction-bound", "untouched-output-guard"}


def test_association_bound_not_widened_and_within_bound_differences_reviewed():
    m, c = reports(); index = find(kind=2, rows=257, mode=0)
    m["cases"][index]["reductions"]["magnitude"] = 1.
    c["cases"][index]["actual"][0] = lc.bits(1e-7)
    result = compare(m, c)
    assert not result["reference_failures"] and result["different_fields"] == 1
    c["cases"][index]["actual"][0] = lc.bits(2e-6)
    assert compare(m, c)["reference_failures"]


def test_cancellation_is_a_metal_witness_not_cub_zero_oracle():
    m, c = reports(); index = find(kind=2, mode=6)
    c["cases"][index]["actual"][0] = lc.bits(1/256)
    result = compare(m, c)
    assert not result["reference_failures"] and result["different_fields"] == 1
    m["cases"][index]["actual"][0] = lc.bits(1/256)
    result = compare(m, c)
    assert [e["policy"] for e in result["reference_failures"]] == ["original-metal-cancellation"]


def test_infinite_float_sum_requires_correct_sign_and_classification():
    m, c = reports(); index = find(kind=0, rows=257, mode=6)
    ref = m["cases"][index]["reductions"]
    ref["sum"][0] = ref["magnitude"][0] = 1e40; ref["rounded"][0] = lc.bits(float("inf"))
    for report in (m, c): report["cases"][index]["actual"][0] = lc.bits(float("inf"))
    assert not compare(m, c)["reference_failures"]
    c["cases"][index]["actual"][0] = lc.bits(-float("inf"))
    assert compare(m, c)["reference_failures"]


def test_domain_rejection_keeps_metal_output_unchanged():
    m, c = reports(); m["cases"][254]["actual"][0] = 1
    result = compare(m, c)
    assert [e["policy"] for e in result["reference_failures"]] == ["metal-rejection-modified-output"]
    assert result["domain_exclusions"][0]["cuda_status"] == "outside-domain-not-executed"


def test_false_cuda_guard_execution_status_rejected():
    c = lc.decode(bytearray(encoded(2)))
    # The zero-length output starts immediately after the status word.
    position = c["cases"][254]["actual"].__array_interface__["data"][0] - np.frombuffer(c["data"], dtype="u1").__array_interface__["data"][0]
    struct.pack_into("<I", c["data"], position-4, 0)
    with pytest.raises(ValueError, match="domain status"): lc.decode(c["data"])


def test_each_original_input_and_guard_buffer_is_compared_bitwise():
    for key, index in (("values", 1), ("weights", 28), ("initial", 1)):
        m, c = reports(); c["cases"][index]["inputs"][key][0] ^= np.uint32(1)
        with pytest.raises(ValueError, match="input bits differ"): compare(m, c)


def test_invalid_oracle_counts_rounding_and_magnitude_rejected():
    for field, val in (("count", 999999), ("rounded", 1), ("magnitude", -1.), ("sum", float("nan"))):
        m, _ = reports(); m["cases"][0]["reductions"][field][0] = val
        with pytest.raises(ValueError): lc.decode(m["data"])


def test_framing_identity_parameters_completion_and_no_overwrite(tmp_path, monkeypatch):
    raw = encoded(0)
    for bad in (raw[:-1], raw+b"!", raw[:8]+struct.pack("<I", 3)+raw[12:]):
        with pytest.raises(ValueError): lc.decode(bad)
    for offset in (16, 16+52):
        bad = bytearray(raw); bad[offset] ^= 1
        with pytest.raises(ValueError, match="identity"): lc.decode(bad)
    output = tmp_path/"existing"; output.write_bytes(b"retain")
    monkeypatch.setattr(lc, "load", lambda _: lc.decode(raw))
    monkeypatch.setattr(sys, "argv", ["compare", "unpack", "input", "--output", str(output)])
    assert lc.main() == 1 and output.read_bytes() == b"retain"


def test_domain_exclusions_require_review_exit_even_without_bit_differences(tmp_path, monkeypatch):
    m, c = reports()
    monkeypatch.setattr(lc, "load", lambda path: m if str(path) == "metal" else c)
    monkeypatch.setattr(sys, "argv", ["compare", "compare", "metal", "cuda", "--output", str(tmp_path/"report.json")])
    assert lc.main() == 2
