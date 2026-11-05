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


def synthetic(backend):
    records = [{"kind": "metadata", "schema_version": 1, "backend": backend,
                "synthetic_test_data": "yes; no GPU execution"}]
    for name, sizes in api["ROWS"].items():
        for rows in sizes:
            for exponent in api["SCALES"]:
                if name == "weighted_cdf":
                    inputs = {"values": [0]*rows, "weights": [0]*rows}
                    outputs = {"cdf": [0]*rows, "selected": [0], "api_selected": [0]}
                elif name == "raw_prefix":
                    inputs = {"weights": [0]*rows, "ids": [0]*rows}
                    outputs = {"cdf": [0]*rows}
                else:
                    inputs = {"entries": [0]*(4*rows), "ptr": [0, 0, 4, 5, 9]}
                    outputs = {"entries": [0]*(4*(rows if name == "repair" else 4)),
                               "ptr": [0, 0, 4, 5, 9] if name == "repair" else [0, 0, 1, 3, 4]}
                for iteration in range(32 if backend == "cuda" and name == "repair" else 1):
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
