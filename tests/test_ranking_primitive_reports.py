"""Synthetic exact/domain/framing checks; these are not NVIDIA observations."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("primitive_compare", Path(__file__).parents[1]/"scripts/compare_ranking_primitives.py")
rp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rp)


@lru_cache
def encoded(kind):
    return (rp.MAGIC + struct.pack("<II", kind, rp.COUNT) + bytes(rp.QUERY.itemsize*rp.COUNT) +
            (bytes(rp.RESULT.itemsize*rp.COUNT) if kind else b"") + b"RPRE" + struct.pack("<I", rp.COUNT))


def reports():
    return [rp.decode(bytearray(encoded(kind))) for kind in (1, 2)]


def compare(m, c):
    return rp.compare(rp.decode(m["data"]), rp.decode(c["data"]))


def bits(value, double=False):
    return struct.unpack("<Q" if double else "<I", struct.pack("<d" if double else "<f", value))[0]


def test_full_corpus_and_identical_reports():
    assert rp.QUERY.itemsize == 36 and rp.RESULT.itemsize == 48
    assert rp.COUNT == 162851
    m, c = reports()
    result = compare(m, c)
    assert result["status"] == "identical-on-recorded-inputs"
    assert result["result_fields"] == 977106 and result["diagnostic_fields"] == 325702
    assert result["compared_fields"] == 1302808
    rp.match_inputs(rp.decode(encoded(0)), m)
    with pytest.raises(ValueError, match="Metal and actual CUDA"): rp.compare(m, m)


@pytest.mark.parametrize("field", ["factor", "positive", "negative", "cost_factor", "cost_positive", "cost_negative"])
def test_every_finite_domain_output_is_exact(field):
    m, c = reports()
    c["observed"][field][-1] = 1  # Smallest subnormal cannot disappear under an absolute budget.
    result = compare(m, c)
    assert result["failures"] == 1 and result["differences"][0]["field"] == field


def test_signed_zero_is_exact_and_nan_payloads_remain_review_items():
    m, c = reports()
    c["observed"]["positive"][0] = bits(-0.)
    assert compare(m, c)["failures"] == 1
    for r, payload in [(m, 0x7fc00001), (c, 0x7fc00002)]: r["observed"]["positive"][0] = payload
    result = compare(m, c)
    assert result["failures"] == 0 and result["status"] == "differences-require-review"
    assert result["differences"][0]["nan_payload_or_sign_differences"] == 1


@pytest.mark.parametrize("prefix", ["", "cost_"])
def test_nonfinite_factor_policy_does_not_waive_equal_factor_truncations(prefix):
    m, c = reports(); double = bool(prefix)
    for r in (m, c):
        r["observed"][prefix+"delta"][0] = bits(float("inf"), double)
        r["observed"][prefix+"factor"][0] = bits(1., double)
    c["observed"][prefix+"positive"][0] = bits(1., double)
    assert compare(m, c)["failures"] == 1
    c["observed"][prefix+"factor"][0] = bits(2., double)
    result = compare(m, c)
    assert result["failures"] == 0 and result["status"] == "differences-require-review"
    assert sum(x["nonfinite_factor_policy_differences"] for x in result["differences"]) == 2


@pytest.mark.parametrize("value", [1., float("inf"), float("nan")])
def test_delta_diagnostics_are_not_waived(value):
    m, c = reports()
    c["observed"]["delta"][0] = bits(value)
    result = compare(m, c)
    assert result["failures"] == 1
    assert result["differences"][0]["nonfinite_factor_policy_differences"] == 0
    if not np.isfinite(value): assert result["domains"]["float"]["finite_nonfinite_disagreement"] == 1


def test_nonfinite_diagnostic_classification_change_still_fails():
    m, c = reports()
    m["observed"]["delta"][0] = bits(float("inf"))
    c["observed"]["delta"][0] = bits(float("nan"))
    c["observed"]["factor"][0] = bits(1.)
    result = compare(m, c)
    assert result["failures"] == 1 and result["domains"]["float"]["both_nonfinite"] == 1


def test_all_input_fields_and_order_are_exact():
    m, c = reports()
    for field in rp.QUERY.names:
        c["inputs"][field][-1] = 1
        with pytest.raises(ValueError, match="input bits/order"): compare(m, c)
        c["inputs"][field][-1] = 0


def test_incomplete_wrong_kind_shape_footer_and_trailing_data():
    original = encoded(1)
    bad = [b"", original[:-1], original+b"x"]
    for offset, payload in [(0, b"BADMAGIC"), (8, struct.pack("<I", 3)),
                            (12, struct.pack("<I", rp.COUNT-1)), (len(original)-8, b"FAIL"),
                            (len(original)-4, struct.pack("<I", rp.COUNT-1))]:
        data = bytearray(original); data[offset:offset+len(payload)] = payload; bad.append(data)
    for data in bad:
        with pytest.raises(ValueError): rp.decode(data)


def test_cli_unpack_and_compare_refuse_existing_output(tmp_path, monkeypatch):
    m = tmp_path/"metal.bin"; c = tmp_path/"cuda.bin"; out = tmp_path/"output"
    m.write_bytes(encoded(1)); c.write_bytes(encoded(2)); out.write_bytes(b"preserve")
    monkeypatch.setattr(sys, "argv", ["compare", "unpack", str(m), "--output", str(out)])
    assert rp.main() == 1 and out.read_bytes() == b"preserve"
    monkeypatch.setattr(sys, "argv", ["compare", "compare", str(m), str(c), "--output", str(out)])
    assert rp.main() == 1 and out.read_bytes() == b"preserve"
