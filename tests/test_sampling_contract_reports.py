"""Synthetic report integrity tests; these do not execute CUDA or validate its math."""
import gzip
import importlib.util
import json
import lzma
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("sampling_compare", ROOT / "scripts/compare_sampling_contracts.py")
sc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sc)


def test_complete_original_matrix_and_layout_sizes():
    specs = sc.specs()
    assert len(specs) == 199
    assert [sum(s.family == f for s in specs) for f in range(3)] == [143, 54, 2]
    assert {s.id for s in specs if s.domain} == {114, 117, 126, 129, 132, 135, 138, 198}
    assert sum(s.rows * s.targets for s in specs if s.family == 0 and not s.domain) == 4499447
    assert sum(2 * (sc.layout(s).storage + sc.layout(s).replay_storage) for s in specs) == 30820562
    assert sum(n for s in specs for _, _, n in sc.sections(s)) == 19522746
    for sequence in range(18):
        calls = [s for s in specs if s.family == 1 and s.sequence == sequence]
        assert [s.step for s in calls] == [0, 1, 2]
        assert all(s.seed == 701 for s in calls)
    # The source generator advances this counter only for successful cases.
    accepted = 0
    for s in specs[:112]:
        assert (s.layout, s.fraction, s.seed) == (accepted % 4, sc.bits((0, .001, .37, .999, 1)[accepted % 5]), accepted * 7919)
        accepted += not s.domain


@pytest.fixture
def reports(monkeypatch):
    # Compact synthetic cases exercise the same format and field policies. The
    # complete real 199-case Metal corpus has a separate recorded validation.
    specs = [
        sc.Spec(0, 0, 0, 3, 1, 0, sc.bits(.6), 1, 0, 0, 0, 0),
        sc.Spec(1, 0, 5, 3, 2, 0, sc.bits(.6), 2, 0, 0, 0, 0),
        sc.Spec(2, 0, 5, 1, 3, 0, sc.bits(1), 3, 0, 0, 0, 0),
        sc.Spec(3, 1, 5, 1, 0, 0, sc.bits(0), 701, 3, 0, 0, 0),
        sc.Spec(4, 1, 5, 1, 0, 0, sc.bits(.6), 701, 3, 0, 1, 0),
        sc.Spec(5, 0, 33, 17, 0, 3, sc.bits(.6), 793, 0, 0, 0, 1),
        sc.Spec(6, 2, 4, 1, 0, 0, sc.bits(0), 1, 0, 0, 0, 0),
        sc.Spec(7, 2, 4, 1, 0, 0, sc.bits(.5), 1, 0, 0, 0, 1),
    ]
    monkeypatch.setattr(sc, "COUNT", len(specs))
    monkeypatch.setattr(sc, "EXCLUDED", {5, 7})
    monkeypatch.setattr(sc, "specs", lambda: specs)
    result = []
    for kind in (1, 2):
        cases = []
        for s in specs:
            shape = sc.layout(s)
            case = {"spec": s, "storage": np.zeros(2 * shape.storage, dtype="<u4"),
                    "replay_input": np.zeros(2 * shape.replay_storage, dtype="<u4"),
                    "status": (1 if kind == 1 else 2) if s.domain else 0,
                    "actual": np.zeros(sum(n for _, _, n in sc.sections(s)), dtype="<u8")}
            if sc.sections(s):
                parts = sc.split(s, case["actual"])
                mask = sc.guards(s)
                case["replay_input"][mask] = sc.bits(71.)
                parts["replay"][mask] = case["replay_input"][mask]
                if "threshold" in parts:
                    parts["threshold"][:] = sc.bits(float("inf") if int(s.rows * sc.fraction(s)) == 0 else 1.)
            if kind == 1:
                case["reference"] = case["actual"].copy()
            cases.append(case)
        result.append({"kind": kind, "cases": cases})
    return result


def parts(report, index, key="actual"):
    case = report["cases"][index]
    return sc.split(case["spec"], case[key])


def encode(report, kind=None):
    kind = report["kind"] if kind is None else kind
    chunks = [sc.MAGIC, struct.pack("<II", kind, sc.COUNT)]
    for case in report["cases"]:
        chunks.extend((struct.pack("<12I", *case["spec"]), case["storage"].tobytes(), case["replay_input"].tobytes()))
        if kind:
            chunks.extend((struct.pack("<I", case["status"]), case["actual"].tobytes()))
        if kind == 1:
            chunks.append(case["reference"].tobytes())
    return b"".join(chunks) + b"SMPE" + struct.pack("<I", sc.COUNT)


def test_valid_reports_preserve_exclusions_without_claiming_parity(reports):
    m, c = reports
    result = sc.compare(sc.decode(encode(m)), sc.decode(encode(c)))
    assert result["status"] == "common-domain-identical-with-exclusions"
    assert result["cuda_executed_cases"] == 6 and len(result["domain_exclusions"]) == 2
    assert not result["reference_failures"] and not sc.validate_metal(m)
    with pytest.raises(ValueError, match="actual CUDA"):
        sc.compare(m, m)


@pytest.mark.parametrize("backend", [0, 1])
def test_each_backend_checked_against_reference(reports, backend):
    parts(reports[backend], 1)["integers_before"][0] = 2**63 + 1
    failures = sc.compare(*reports)["reference_failures"]
    assert len(failures) == 1 and failures[0]["backend"] == ("metal", "cuda")[backend]


def test_identical_wrong_outputs_do_not_pass(reports):
    for report in reports:
        parts(report, 1)["roots_after"][0] = np.array([1.], dtype="<f8").view("<u8")[0]
    result = sc.compare(*reports)
    assert not result["different_fields"]
    assert {r["backend"] for r in result["reference_failures"]} == {"metal", "cuda"}


def test_signed_zero_root_is_numerically_equal_but_still_reported(reports):
    parts(reports[1], 1)["roots_before"][0] = 1 << 63
    result = sc.compare(*reports)
    assert not result["reference_failures"] and result["different_fields"] == 1
    assert result["status"] == "review-differences-and-domains"


def test_nan_policy_and_exact_replay_are_distinct(reports):
    m, c = reports
    parts(m, 1, "reference")["norms"][0] = 0x7FC00001
    parts(m, 1)["norms"][0] = 0x7FC00002
    parts(c, 1)["norms"][0] = 0xFFC00003
    assert not sc.compare(m, c)["reference_failures"]
    logical = np.flatnonzero(~sc.guards(m["cases"][1]["spec"]))[0]
    parts(m, 1, "reference")["replay"][logical] = 0x7FC00001
    parts(m, 1)["replay"][logical] = 0x7FC00001
    parts(c, 1)["replay"][logical] = 0x7FC00002
    failures = sc.compare(m, c)["reference_failures"]
    assert len(failures) == 1 and failures[0]["policy"] == "original-float-exact"


def test_integer_and_rng_words_never_round_through_floats(reports):
    m, c = reports
    for key in ("actual", "reference"):
        parts(m, 1, key)["integers_before"][0] = 2**63 + 1
    parts(c, 1)["integers_before"][0] = 2**63
    parts(c, 4)["next_rng"][0] = 1
    failures = sc.compare(m, c)["reference_failures"]
    assert {f["policy"] for f in failures} == {"original-integer", "original-rng"}


@pytest.mark.parametrize("case,field,index,value,policy", [
    (1, "replay", 0, 0, "untouched-replay-padding"),
    (2, "replay", 8, 1, "unsampled-input-identity"),
    (3, "sampled", 0, 1, "zero-budget-sampled"),
    (3, "roots_before", 0, 1, "empty-or-zero-budget-root"),
    (0, "threshold", 0, 0, "zero-budget-threshold"),
])
def test_input_invariants_reject_even_forged_matching_references(reports, case, field, index, value, policy):
    m, c = reports
    for report in reports:
        parts(report, case)[field][index] = value
    parts(m, case, "reference")[field][index] = value
    result = sc.compare(m, c)
    assert not result["different_fields"]
    assert policy in {f["policy"] for f in result["reference_failures"]}


@pytest.mark.parametrize("key", ["storage", "replay_input"])
def test_input_padding_bits_must_match(reports, key):
    reports[1]["cases"][1][key][-1] ^= np.uint32(1)
    with pytest.raises(ValueError, match="input bits"):
        sc.compare(*reports)


def test_wrong_domain_status_and_upper_bits_rejected(reports):
    m, c = reports
    c["cases"][5]["status"] = 1  # Metal rejection must not masquerade as CUDA output.
    with pytest.raises(ValueError, match="exclusion status"):
        sc.decode(encode(c))
    c["cases"][5]["status"] = 2
    parts(c, 4)["next_rng"][0] = 1 << 32
    with pytest.raises(ValueError, match="upper sampling"):
        sc.decode(encode(c))
    parts(m, 1, "reference")["norms"][0] = 1 << 32
    with pytest.raises(ValueError, match="upper sampling"):
        sc.decode(encode(m))


def test_framing_and_lossless_compression(reports, tmp_path):
    raw = encode(reports[0], 0)
    assert sc.decode(raw)["kind"] == 0
    for bad in (raw[:-1], raw + b"x", b"BADMAGIC" + raw[8:],
                raw[:8] + struct.pack("<I", 9) + raw[12:],
                raw[:12] + struct.pack("<I", sc.COUNT - 1) + raw[16:],
                raw[:16] + struct.pack("<I", 99) + raw[20:]):
        with pytest.raises(ValueError):
            sc.decode(bad)
    for suffix, compress in ((".bin", lambda b: b), (".gz", gzip.compress), (".xz", lzma.compress)):
        path = tmp_path / ("inputs" + suffix)
        path.write_bytes(compress(raw))
        assert sc.load(path)["data"] == raw


def test_cli_requires_input_only_and_never_overwrites(reports, tmp_path, monkeypatch):
    source, output = tmp_path / "source.bin", tmp_path / "input.bin"
    source.write_bytes(encode(reports[0]))
    monkeypatch.setattr(sys, "argv", ["compare", "unpack", str(source), str(output)])
    with pytest.raises(ValueError, match="input-only"):
        sc.main()
    assert not output.exists()
    raw = encode(reports[0], 0)
    source.write_bytes(raw)
    assert sc.main() == 0 and output.read_bytes() == raw
    with pytest.raises(FileExistsError):
        sc.main()


def test_cli_retains_review_and_failure_reports(reports, tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "load", lambda p: reports[int(str(p) == "cuda")])
    output = tmp_path / "review.json"
    monkeypatch.setattr(sys, "argv", ["compare", "compare", "metal", "cuda", "--output", str(output)])
    assert sc.main() == 2  # Even identical common-domain output retains exclusions.
    assert json.loads(output.read_text())["status"] == "common-domain-identical-with-exclusions"
    with pytest.raises(ValueError, match="already exists"):
        sc.main()
    parts(reports[1], 1)["integers_after"][0] = 1
    monkeypatch.setattr(sys, "argv", ["compare", "compare", "metal", "cuda", "--output", str(tmp_path / "failure.json")])
    assert sc.main() == 1
