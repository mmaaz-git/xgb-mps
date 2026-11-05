"""Synthetic statistics reference/framing checks, never NVIDIA evidence."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("stats_compare", Path(__file__).parents[1]/"scripts/compare_stats_contracts.py")
sc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sc)


@lru_cache
def encoded(kind):
    parts = [sc.MAGIC, struct.pack("<II", kind, sc.COUNT)]
    for s in sc.specs():
        inputs = np.zeros(s[5], dtype="<u4")
        observed = np.zeros(s[7], dtype="<u8")
        if s[1] == 0: observed[-1] = 2 if s[4] == 0 else 1 if not s[2] or s[4] == 2 else 0
        elif s[1] == 2:
            inputs[:] = [sc.bits(1e8, 32), sc.bits(1., 32)] if s[4] == 0 else [0x7f7fffff]*2
            observed[:] = [sc.bits(1e8), sc.bits(100000001.)] if s[4] == 0 else [
                sc.bits(float("inf")), sc.bits(2*sc.value(0x7f7fffff, 32))]
        elif s[1] == 3: observed[0] = sc.bits(38.5)
        parts += [struct.pack("<9I", *s), inputs.tobytes(), bytes(s[6]*4)]
        if kind: parts.append(observed.tobytes())
        if kind == 1: parts += [observed.tobytes(), bytes(16)]
    return b"".join(parts)+b"STAE"+struct.pack("<I", sc.COUNT)


def reports():
    return [sc.decode(bytearray(encoded(kind))) for kind in (1, 2)]


def compare(m, c):
    return sc.compare(sc.decode(m["data"]), sc.decode(c["data"]))


def test_complete_statistics_matrix_and_original_reference_acceptance():
    m, c = reports()
    assert [sum(s[1] == k for s in sc.specs()) for k in range(4)] == [56, 63, 2, 1]
    assert sum(s[5] for s in sc.specs()) == 7565596 and sum(s[6] for s in sc.specs()) == 27864
    result = compare(m, c)
    assert result["compared_fields"] == 515 and result["status"] == "identical-on-recorded-inputs"
    sc.match_inputs(sc.decode(encoded(0)), m)
    with pytest.raises(ValueError, match="Metal and actual CUDA"): sc.compare(m, m)


def test_identical_wrong_results_cannot_pass():
    m, c = reports()
    for report in (m, c): report["cases"][0]["actual"][0] = sc.bits(1., 32)
    result = compare(m, c)
    assert result["failures"] == 2 and not result["differences"]


def test_median_signed_zero_is_exact_but_nan_payloads_are_reviewed():
    m, c = reports()
    c["cases"][0]["actual"][0] = sc.bits(-0., 32)
    assert compare(m, c)["failures"] == 1
    m["cases"][0]["reference"][0] = 0x7fc00000
    m["cases"][0]["actual"][0] = 0x7fc00001
    c["cases"][0]["actual"][0] = 0x7fc00002
    assert compare(m, c)["status"] == "differences-require-review"


def test_original_mean_budget_and_within_bound_bits_are_visible():
    m, c = reports()
    c["cases"][0]["actual"][1] = sc.bits(1e-7, 32)
    result = compare(m, c)
    assert result["failures"] == 0 and result["status"] == "differences-require-review"
    c["cases"][0]["actual"][1] = sc.bits(2e-6, 32)
    assert compare(m, c)["failures"] == 1


def test_both_reduction_outputs_retain_exact_cases():
    m, c = reports()
    for i in (0, 1): c["cases"][56+7]["actual"][i] = sc.bits(2.**-149)
    assert compare(m, c)["failures"] == 2
    for i in (0, 1): c["cases"][56+7]["actual"][i] = sc.bits(-0.)
    result = compare(m, c)
    assert result["failures"] == 0 and len(result["differences"]) == 2


def test_mode_two_forward_bounds_are_not_widened():
    m, c = reports(); index = 56+7*5+2  # 257 rows, wide exponent mode.
    m["cases"][index]["oracle"][1] = sc.bits(1.)
    c["cases"][index]["actual"][:] = [sc.bits(1e-7), sc.bits(1e-15)]
    # Generic source output must be a widened FP32, even if the error is small.
    result = compare(m, c)
    assert result["failures"] == 1 and "generic-result-not-widened-fp32" in result["reference_failures"][0]["reasons"]
    c["cases"][index]["actual"][0] = sc.bits(float(np.float32(1e-7)))
    assert compare(m, c)["failures"] == 0
    c["cases"][index]["actual"][:] = [sc.bits(1.), sc.bits(1e-12)]
    assert compare(m, c)["failures"] == 2


def test_compensated_check_is_independent_of_cpu_reference():
    m, c = reports(); index = 56+7*5+2
    m["cases"][index]["oracle"][0] = sc.bits(1.)
    result = compare(m, c)
    assert result["failures"] == 2
    assert all("compensated-reduction-bound" in f["reasons"] for f in result["reference_failures"])


def test_nonfinite_reduction_classification_changes_fail():
    m, c = reports(); c["cases"][56+7+4]["actual"][0] = sc.bits(float("nan"))
    assert compare(m, c)["failures"] == 1


@pytest.mark.parametrize("index,field,bad", [(119, 0, 100000001.), (119, 1, 1e8),
                                           (120, 0, 2*sc.value(0x7f7fffff, 32)), (120, 1, 1.), (121, 0, 3.5)])
def test_precision_and_scalar_witnesses_cannot_be_waived(index, field, bad):
    m, c = reports(); c["cases"][index]["actual"][field] = sc.bits(bad)
    assert compare(m, c)["failures"] == 1


def test_weighted_error_and_not_called_statuses_are_checked():
    m, c = reports()
    c["cases"][1]["actual"][-1] = 0  # Empty weighted input should reject.
    c["cases"][0]["actual"][-1] = 1  # Unweighted mode should not call weighted API.
    assert compare(m, c)["failures"] == 2


def test_input_bits_order_and_references_are_checked():
    m, c = reports(); c["cases"][10]["inputs"]["values"][0] = 1
    with pytest.raises(ValueError, match="input bits differ"): compare(m, c)
    c["cases"][10]["inputs"]["values"][0] = 0
    c["cases"][10]["inputs"]["weights"][0] = 1
    with pytest.raises(ValueError, match="input bits differ"): compare(m, c)
    m["cases"][119]["reference"][0] = 0
    c["cases"][10]["inputs"]["weights"][0] = 0
    assert compare(m, c)["failures"] == 2


def test_malformed_complete_shape_upper_bits_and_oracles():
    original = encoded(1)
    bad = [b"", original[:-1], original+b"x"]
    for offset, payload in [(0, b"BADMAGIC"), (8, struct.pack("<I", 3)), (12, struct.pack("<I", 121)),
                            (16, struct.pack("<I", 1)), (len(original)-8, b"FAIL")]:
        data = bytearray(original); data[offset:offset+len(payload)] = payload; bad.append(data)
    for data in bad:
        with pytest.raises(ValueError): sc.decode(data)
    m, _ = reports(); m["cases"][0]["actual"][0] = 1 << 32
    with pytest.raises(ValueError, match="upper FP32"): sc.decode(m["data"])
    m, _ = reports(); m["cases"][58]["oracle"][1] = sc.bits(float("nan"))
    with pytest.raises(ValueError, match="compensated"): sc.decode(m["data"])


def test_cli_refuses_existing_output(tmp_path, monkeypatch):
    m = tmp_path/"metal.bin"; c = tmp_path/"cuda.bin"; out = tmp_path/"output"
    m.write_bytes(encoded(1)); c.write_bytes(encoded(2)); out.write_bytes(b"preserve")
    for args in [["unpack", str(m)], ["compare", str(m), str(c)]]:
        monkeypatch.setattr(sys, "argv", ["stats", *args, "--output", str(out)])
        assert sc.main() == 1 and out.read_bytes() == b"preserve"
