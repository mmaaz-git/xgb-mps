#!/usr/bin/env python3
"""Validate complete v1/v2 observation files and compare shared witness inputs.

Exit 0: files are valid and the compared outputs match on these inputs only.
Exit 2: numerical/read-order differences were recorded and require review.
Exit 1: incomplete/mismatched inputs, structural mismatch, or invalid reports.
No result from this script establishes full CUDA parity.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

SCALES = (0, -149, 100)
ROWS = {"weighted_cdf": (3, 257), "raw_prefix": (3, 261, 513),
        "reverse_unique": (9,), "repair": (9,)}


def case_rows(suite):
    return ROWS | ({"weighted_threshold": (2,)} if suite == "numerical-v2" else {})


def iterations(backend, name):
    if name == "weighted_threshold":
        return 6  # Three neighboring alphas, each in forward/reversed input order.
    return 32 if backend == "cuda" and name == "repair" else 1


class ReportError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReportError(message)


def arrays(fields, shape, context):
    require(isinstance(fields, dict) and fields.keys() == shape.keys(), f"{context}: wrong fields")
    for key, (length, bits) in shape.items():
        values = fields[key]
        require(isinstance(values, list) and len(values) == length, f"{context}.{key}: wrong length")
        require(all(type(v) is int and 0 <= v < (1 << bits) for v in values),
                f"{context}.{key}: expected unsigned {bits}-bit words")


def load_report(path):
    path = Path(path)
    try:
        lines = [json.loads(line) for line in path.read_text().splitlines()]
    except (ValueError, OSError) as error:
        raise ReportError(f"Cannot read {path}: {error}") from error
    require(len(lines) >= 2 and all(isinstance(line, dict) for line in lines), "Missing report framing")
    metadata, summary = lines[0], lines[-1]
    require(metadata.get("kind") == "metadata" and type(metadata.get("schema_version")) is int
            and metadata["schema_version"] == 1,
            "Missing metadata/schema version")
    backend = metadata.get("backend")
    require(backend in ("metal", "cuda"), "Unknown backend")
    suite = metadata.get("fixture_suite", "numerical-v1")
    require(suite in ("numerical-v1", "numerical-v2"), "Unknown fixture suite")
    require(summary.get("kind") == "summary" and summary.get("status") == "completed",
            "Missing successful completion record")
    require(type(summary.get("cases")) is int and summary["cases"] == len(lines)-2,
            "Summary count does not match the observations")
    expected = {(name, rows, exponent, iteration)
                for name, sizes in case_rows(suite).items() for rows in sizes for exponent in SCALES
                for iteration in range(iterations(backend, name))}
    cases = {}
    for case in lines[1:-1]:
        require(case.get("kind") == "case", "Unexpected non-case record")
        require(isinstance(case.get("id"), str) and all(type(case.get(k)) is int
                for k in ("rows", "exponent", "iteration")), "Invalid case identity")
        key = tuple(case[k] for k in ("id", "rows", "exponent", "iteration"))
        require(key in expected, f"Unexpected case {key}")
        require(key not in cases, f"Duplicate case {key}")
        name, rows, _, _ = key
        if name == "weighted_threshold":
            inputs = {"values": (2, 32), "weights": (2, 32), "alpha": (1, 32)}
            outputs = {"api_selected": (1, 32)}
        elif name == "weighted_cdf":
            inputs = {"values": (rows, 32), "weights": (rows, 32)}
            outputs = {"cdf": (rows, 32), "selected": (1, 32), "api_selected": (1, 32)}
        elif name == "raw_prefix":
            inputs = {"weights": (rows, 32), "ids": (rows, 32)}
            outputs = {"cdf": (rows, 32)}
        else:
            inputs = {"entries": (4*rows, 32), "ptr": (5, 64)}
            outputs = {"entries": (4*(rows if name == "repair" else 4), 32), "ptr": (5, 64)}
        arrays(case.get("input"), inputs, f"{key}.input")
        arrays(case.get("output"), outputs, f"{key}.output")
        if name in ("reverse_unique", "repair"):
            for side in ("input", "output"):
                ptr = case[side]["ptr"]
                require(ptr[0] == 0 and ptr[-1]*4 == len(case[side]["entries"])
                        and ptr == sorted(ptr), f"{key}.{side}: invalid column pointers")
        cases[key] = case
    require(cases.keys() == expected, f"Missing cases: {sorted(expected-cases.keys())}")
    return {"metadata": metadata, "cases": cases, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(metal, cuda):
    require(metal["metadata"]["backend"] == "metal" and cuda["metadata"]["backend"] == "cuda",
            "Expected Metal report followed by CUDA report")
    suite = metal["metadata"].get("fixture_suite", "numerical-v1")
    require(suite == cuda["metadata"].get("fixture_suite", "numerical-v1"), "Fixture suites differ")
    differences = []
    patterns = {}
    for key, observed in cuda["cases"].items():
        base = key[:3] + (0 if key[0] == "repair" else key[3],)
        reference = metal["cases"][base]
        require(observed["input"] == reference["input"], f"{key}: fixture inputs differ")
        counts = patterns.setdefault(base, Counter())
        counts[json.dumps(observed["output"], sort_keys=True)] += 1
        changed = {}
        for field, actual in observed["output"].items():
            expected = reference["output"][field]
            indices = [i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]]
            if not indices:
                continue
            require(field != "ptr", f"{key}: structural column-pointer mismatch")
            require(key[0] != "weighted_threshold", f"{key}: exact binary64 threshold contract mismatch")
            if field == "entries":
                invariant = (2, 3) if key[0] == "repair" else (0, 3)
                require(all(i % 4 not in invariant for i in indices),
                        f"{key}: source-preserved entry field changed")
            changed[field] = {"count": len(indices), "first_differences": [
                {"index": i, "metal_bits": f"{expected[i]:08x}", "cuda_bits": f"{actual[i]:08x}"}
                for i in indices[:8]]}
        if changed:
            differences.append({"id": key[0], "rows": key[1], "exponent": key[2], "iteration": key[3],
                                "review_contract": "P5-N13 source race" if key[0] == "repair" else
                                "P5-N08 CDF association" if key[0] == "weighted_cdf" else
                                "P5-N14 scan association", "fields": changed})
    return {
        "status": "differences-require-review" if differences else "exact-on-these-inputs",
        "scope": ("P5-N01/08/13/14: 39 shared inputs only; not full CUDA parity" if suite == "numerical-v2" else
                  "P5-N08/13/14: 21 shared inputs only; not full CUDA parity"),
        "metal": {k: v for k, v in metal.items() if k != "cases"},
        "cuda": {k: v for k, v in cuda.items() if k != "cases"},
        "input_cases": len(metal["cases"]), "cuda_executions": len(cuda["cases"]),
        "differing_executions": len(differences), "differences": differences,
        "repeat_observations": [
            {"id": key[0], "rows": key[1], "exponent": key[2], "variant": key[3],
             "distinct_cuda_outputs": len(counts), "frequencies": sorted(counts.values(), reverse=True)}
            for key, counts in patterns.items()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Check completeness/shape without claiming equivalence")
    inspect.add_argument("report", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("metal", type=Path)
    comparison.add_argument("cuda", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            report = load_report(args.report)
            print(json.dumps({"metadata": report["metadata"], "execution_records": len(report["cases"]),
                              "status": "complete-observation-file; no equivalence claim"}, indent=2))
            return 0
        result = compare(load_report(args.metal), load_report(args.cuda))
        with args.output.open("x") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        print(f"{result['status']}: {result['input_cases']} inputs, "
              f"{result['cuda_executions']} CUDA executions, "
              f"{result['differing_executions']} differing executions; {args.output}")
        return 2 if result["differences"] else 0
    except (ReportError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
