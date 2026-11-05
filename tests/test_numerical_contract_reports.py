"""Integrity checks for comparison tooling. All CUDA data here is synthetic.

These tests exercise incomplete/mismatched reports and review outcomes, not
CUDA execution or numerical equivalence.
"""
import copy
import json
from pathlib import Path
import runpy

import pytest

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/compare_numerical_contracts.py"))
load = api["load_report"]
compare = api["compare"]
ReportError = api["ReportError"]


def synthetic(backend, suite="numerical-v1"):
    records = [{"kind": "metadata", "schema_version": 1, "backend": backend,
                "fixture_suite": suite,
                "synthetic_test_data": "yes; no GPU execution"}]
    for name, sizes in api["case_rows"](suite).items():
        for rows in sizes:
            for exponent in api["SCALES"]:
                if name == "weighted_threshold":
                    inputs = {"values": [0]*rows, "weights": [0]*rows, "alpha": [0]}
                    outputs = {"api_selected": [0]}
                elif name == "weighted_cdf":
                    inputs = {"values": [0]*rows, "weights": [0]*rows}
                    outputs = {"cdf": [0]*rows, "selected": [0], "api_selected": [0]}
                elif name == "raw_prefix":
                    inputs = {"weights": [0]*rows, "ids": [0]*rows}
                    outputs = {"cdf": [0]*rows}
                else:
                    inputs = {"entries": [0]*(4*rows), "ptr": [0, 0, 4, 5, 9]}
                    outputs = {"entries": [0]*(4*(rows if name == "repair" else 4)),
                               "ptr": [0, 0, 4, 5, 9] if name == "repair" else [0, 0, 1, 3, 4]}
                for iteration in range(api["iterations"](backend, name)):
                    if name == "weighted_threshold":
                        # Deliberately distinct synthetic variants detect accidental
                        # comparison against variant zero instead of matching inputs.
                        inputs["alpha"] = [iteration]
                        outputs["api_selected"] = [iteration]
                    records.append(copy.deepcopy({"kind": "case", "id": name, "rows": rows,
                                                  "exponent": exponent, "iteration": iteration,
                                                  "input": inputs, "output": outputs}))
    records.append({"kind": "summary", "status": "completed", "cases": len(records)-1})
    return records


def write(path, records):
    path.write_text("".join(json.dumps(r)+"\n" for r in records))
    return path


def test_exact_and_order_differences_require_review(tmp_path):
    metal = load(write(tmp_path/"metal.jsonl", synthetic("metal")))
    records = synthetic("cuda")
    cuda = load(write(tmp_path/"cuda.jsonl", records))
    result = compare(metal, cuda)
    assert result["status"] == "exact-on-these-inputs"
    assert result["input_cases"] == 21 and result["cuda_executions"] == 114
    # A changed selected observation and a changed repair rank must stay visible,
    # even though both operations have documented order-dependent contracts.
    records[1]["output"]["api_selected"][0] = 0x3F800000
    repair = next(r for r in records if r.get("id") == "repair")
    repair["output"]["entries"][0] = 0x40000000
    result = compare(metal, load(write(tmp_path/"different.jsonl", records)))
    assert result["status"] == "differences-require-review"
    assert result["differing_executions"] == 2
    assert {r["review_contract"] for r in result["differences"]} == {
        "P5-N08 CDF association", "P5-N13 source race"}
    assert any(r["frequencies"] == [31, 1] for r in result["repeat_observations"])


@pytest.mark.parametrize("corruption", ["truncated", "missing_case", "duplicate", "bad_bits", "short_output"])
def test_reject_incomplete_or_malformed(tmp_path, corruption):
    records = synthetic("cuda")
    if corruption == "truncated":
        records.pop()
    elif corruption == "missing_case":
        records.pop(1)
        records[-1]["cases"] -= 1
    elif corruption == "duplicate":
        records[2] = copy.deepcopy(records[1])
    elif corruption == "bad_bits":
        records[1]["output"]["cdf"][0] = 1 << 32
    else:
        records[1]["output"]["cdf"].pop()
    with pytest.raises(ReportError):
        load(write(tmp_path/"invalid.jsonl", records))


@pytest.mark.parametrize("corruption", ["input", "ptr", "repair_weight", "reverse_value"])
def test_reject_mismatched_fixtures_and_structural_outputs(tmp_path, corruption):
    metal = load(write(tmp_path/"metal.jsonl", synthetic("metal")))
    records = synthetic("cuda")
    if corruption == "input":
        records[1]["input"]["weights"][0] = 1
    elif corruption == "ptr":
        next(r for r in records if r.get("id") == "repair")["output"]["ptr"][1] = 1
    elif corruption == "repair_weight":
        next(r for r in records if r.get("id") == "repair")["output"]["entries"][2] = 1
    else:
        next(r for r in records if r.get("id") == "reverse_unique")["output"]["entries"][3] = 1
    with pytest.raises(ReportError):
        compare(metal, load(write(tmp_path/"invalid.jsonl", records)))


def test_threshold_variants_are_distinct_exact_contracts(tmp_path):
    metal = load(write(tmp_path/"metal.jsonl", synthetic("metal", "numerical-v2")))
    records = synthetic("cuda", "numerical-v2")
    result = compare(metal, load(write(tmp_path/"cuda.jsonl", records)))
    assert result["input_cases"] == 39 and result["cuda_executions"] == 132
    assert result["status"] == "exact-on-these-inputs"
    threshold = [v for v in result["repeat_observations"] if v["id"] == "weighted_threshold"]
    assert len(threshold) == 18 and all(v["frequencies"] == [1] for v in threshold)
    next(r for r in records if r.get("id") == "weighted_threshold")["output"]["api_selected"] = [0x3f800000]
    with pytest.raises(ReportError, match="exact binary64 threshold"):
        compare(metal, load(write(tmp_path/"bad-threshold.jsonl", records)))


def test_reject_missing_threshold_and_mixed_suite(tmp_path):
    metal = load(write(tmp_path/"metal.jsonl", synthetic("metal", "numerical-v2")))
    old_cuda = load(write(tmp_path/"old.jsonl", synthetic("cuda")))
    with pytest.raises(ReportError, match="suites differ"):
        compare(metal, old_cuda)
    records = synthetic("cuda", "numerical-v2")
    records.pop(next(i for i, r in enumerate(records) if r.get("id") == "weighted_threshold"))
    records[-1]["cases"] -= 1
    with pytest.raises(ReportError, match="Missing cases"):
        load(write(tmp_path/"missing-threshold.jsonl", records))
