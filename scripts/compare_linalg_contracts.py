#!/usr/bin/env python3
"""Compare all linalg observations without treating CUDA-undefined guards as passes.

Metal observations contain original vector references and reduction bounds.
Input-only files never contain expected outputs. All observed bit differences
require review, including differences allowed by the original reference checks.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import NamedTuple

import numpy as np

MAGIC, COUNT = b"XGBLIN1\0", 260
ORACLE = np.dtype([("sum", "<f8"), ("magnitude", "<f8"), ("count", "<u8"), ("rounded", "<u4")])


def require(ok, message):
    if not ok: raise ValueError(message)


def bits(x, width=32):
    return struct.unpack("<I" if width == 32 else "<Q", struct.pack("<f" if width == 32 else "<d", x))[0]


def value(x):
    return struct.unpack("<f", struct.pack("<I", int(x)))[0]


class Spec(NamedTuple):
    id: int
    kind: int
    rows: int
    columns: int
    mode: int
    op: int
    stride0: int
    stride1: int
    out_stride: int
    values: int
    weights: int
    initial: int
    outputs: int
    parameter: int


def specs():
    out = []
    def add(kind, n, c, mode, op, s0, s1, so, nv, nw, ni, no, p):
        out.append(Spec(len(out), kind, n, c, mode, op, s0, s1, so, nv, nw, ni, no, bits(p, 64)))
    def hist(n, c, mode):
        st = 1+mode%3
        add(0, n, c, mode, 0, st, 1, st, n*st+4, n, c*st+4, c*st+4, 1.)
    for n in (0, 1, 257, 4097, 65537):
        for c in (1, 3, 33, 257):
            for mode in range(7): hist(n, c, mode)
    hist(262145, 1025, 3)
    for st in (1, 3):
        for op in range(4):
            for p in (0., -0., 1., -1., 1/3, 1e-40, 1e-300, 1e300, math.inf, math.nan):
                add(1, 65548, 1, 0, op, st, 1, st, 65548*st+3, 0, 0, 65548*st+3, p)
    for n in (1, 32, 257, 4097, 65537, 262145):
        for mode in range(5): add(2, n, 1, mode, 0, 3, 1, 1, n*3+4, 0, 0, 1, 0.)
    for c in (3, 7): add(2, 257, c, 5, 0, 2*c+1, 2, 3, 257*(2*c+1)+4, 0, c*3+4, c*3+4, 0.)
    add(2, 256, 1, 6, 0, 1, 1, 1, 256, 0, 0, 1, 0.)
    for n in (4, 2, 1): add(3, n, 3, 0, 0, 1, 1, 1, n, 0, 3, 3, 1.)
    add(4, 5, 4, 0, 0, 1, 1, 1, 5, 0, 4, 4, .5)
    add(5, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2.)
    add(6, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1.)
    require(len(out) == COUNT, "Incomplete linalg control matrix")
    return out


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, "Invalid linalg interchange header")
    kind, count = struct.unpack_from("<II", data, 8)
    require(kind in (0, 1, 2) and count == COUNT, "Wrong linalg backend/count")
    offset, cases = 16, []
    def array(n, dtype="<u4"):
        nonlocal offset
        size = np.dtype(dtype).itemsize*n
        require(offset+size <= len(data), "Truncated linalg array")
        out = np.frombuffer(data, dtype=dtype, count=n, offset=offset); offset += size
        return out
    for spec in specs():
        require(offset+60 <= len(data), "Truncated linalg case header")
        identity = struct.unpack_from("<13IQ", data, offset); offset += 60
        require(identity == spec, "Wrong linalg identity/order/shape/parameter")
        inputs = {"values": array(spec.values), "weights": array(spec.weights), "initial": array(spec.initial)}
        status, actual, reference, reductions = None, None, None, None
        if kind:
            status = int(array(1)[0])
            expected = (2 if kind == 2 else 1) if spec.kind == 3 else 0
            require(status == expected, "Wrong linalg execution/domain status")
            actual = array(0 if status == 2 else spec.outputs)
        if kind == 1:
            reference = array(spec.outputs if spec.kind == 1 else 0)
            reductions = array(spec.columns if spec.kind in (0, 2, 4) else 0, ORACLE)
            require(np.all(np.isfinite(reductions["sum"])) and np.all(np.isfinite(reductions["magnitude"])) and
                    np.all(reductions["magnitude"] >= 0), "Invalid linalg sum/magnitude oracle")
            require(np.all(reductions["count"] <= spec.rows), "Invalid reduction term count")
            if spec.kind in (0, 4):
                require(int(reductions["count"].sum()) == spec.rows, "Incomplete histogram term counts")
            elif spec.kind == 2:
                require(np.all(reductions["count"] == spec.rows), "Wrong mean term count")
            with np.errstate(over="ignore"):
                rounded = reductions["sum"].astype("<f4").view("<u4")
            require(np.array_equal(rounded, reductions["rounded"]), "Inconsistent rounded reduction oracle")
        cases.append({"spec": spec, "inputs": inputs, "status": status, "actual": actual,
                      "reference": reference, "reductions": reductions})
    require(len(data) == offset+8 and data[offset:offset+4] == b"LINE" and
            struct.unpack_from("<I", data, offset+4)[0] == COUNT,
            "Missing linalg completion marker or trailing bytes")
    return {"kind": kind, "cases": cases, "data": data, "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as stream: return decode(stream.read())


def match_inputs(a, b):
    require(len(a["cases"]) == len(b["cases"]) == COUNT, "Incomplete linalg corpus")
    for x, y in zip(a["cases"], b["cases"]):
        require(x["spec"] == y["spec"], "Different linalg parameters")
        for name, words in x["inputs"].items():
            require(np.array_equal(words, y["inputs"][name]), f"Case {x['spec'].id}: {name} input bits differ")


def ulp_failures(actual, reference, budget):
    a, b = actual.view("<f4"), reference.view("<f4")
    equal = (a == b) | (np.isnan(a) & np.isnan(b))
    finite = np.isfinite(a) & np.isfinite(b)
    ak = np.where(actual >> 31, ~actual, actual ^ np.uint32(0x80000000)).astype(np.int64)
    bk = np.where(reference >> 31, ~reference, reference ^ np.uint32(0x80000000)).astype(np.int64)
    return ~equal & (~finite | (np.abs(ak-bk) > budget))


def reduction_ok(word, oracle, exact):
    actual, expected = value(word), float(oracle["sum"])
    rounded = value(oracle["rounded"])
    if math.isnan(expected): return math.isnan(actual)
    if math.isinf(rounded): return actual == rounded
    if exact: return actual == rounded
    k = (int(oracle["count"])+255)//256+8
    eps = 2.**-23
    bound = k*eps/(1-k*eps)*float(oracle["magnitude"])
    return math.isfinite(actual) and abs(actual-expected)/max(bound, 1e-300) <= 1


def failures(case, reference, backend):
    s, actual = case["spec"], case["actual"]
    if case["status"] == 2: return []
    results = []
    def record(policy, mask):
        indices = np.flatnonzero(mask)
        if indices.size:
            results.append({"case": s.id, "backend": backend, "policy": policy, "count": int(indices.size),
                            "first_indices": indices[:32].tolist()})
    if s.kind == 1:
        record("original-vector-ulp", ulp_failures(actual, reference["reference"], 1 if s.op == 2 else 0))
        guard = np.ones(s.outputs, dtype=bool); guard[:s.rows*s.stride0:s.stride0] = False
        record("untouched-vector-guard", guard & (actual != case["inputs"]["values"]))
    elif s.kind in (0, 2, 4):
        exact = (s.kind == 0 and (s.mode <= 2 or s.mode == 4)) or s.kind == 4 or (s.kind == 2 and s.mode == 4)
        bad = np.zeros(s.outputs, dtype=bool)
        for t, oracle in enumerate(reference["reductions"]):
            bad[t*s.out_stride] = not reduction_ok(actual[t*s.out_stride], oracle, exact)
        record("original-reduction-bound", bad)
        if s.initial:
            guard = np.ones(s.outputs, dtype=bool); guard[:s.columns*s.out_stride:s.out_stride] = False
            record("untouched-output-guard", guard & (actual != case["inputs"]["initial"]))
        if s.kind == 2 and s.mode == 6 and backend == "metal":
            record("original-metal-cancellation", actual.view("<f4") != 0)
    elif s.kind == 3:
        record("metal-rejection-modified-output", actual != case["inputs"]["initial"])
    return results


def validate_metal(metal):
    require(metal["kind"] == 1, "Expected Metal linalg observations")
    return [failure for case in metal["cases"] for failure in failures(case, case, "metal")]


def compare(metal, cuda):
    require(metal["kind"] == 1 and cuda["kind"] == 2, "Expected Metal and actual CUDA linalg outputs")
    match_inputs(metal, cuda)
    errors, differences, domains, compared = [], [], [], 0
    for m, c in zip(metal["cases"], cuda["cases"]):
        s = m["spec"]
        errors.extend(failures(m, m, "metal")); errors.extend(failures(c, m, "cuda"))
        if c["status"] == 2:
            domains.append({"case": s.id, "metal_status": "rejected", "cuda_status": "outside-domain-not-executed"})
            continue
        a, b = m["actual"], c["actual"]; compared += a.size
        indices = np.flatnonzero(a != b)
        if indices.size:
            differences.append({"case": s.id, "count": int(indices.size),
                                "first_fields": [{"index": int(i), "metal_bits": f"{a[i]:08x}",
                                                  "cuda_bits": f"{b[i]:08x}"} for i in indices[:32]]})
    return {"suite": "linalg-v1", "cases": COUNT, "cuda_executed_cases": COUNT-len(domains),
            "compared_fields": int(compared), "failure_fields_by_policy": sum(e["count"] for e in errors),
            "reference_failures": errors, "different_fields": sum(d["count"] for d in differences),
            "differences": differences, "domain_exclusions": domains,
            "status": "reference-check-failed" if errors else "differences-require-review" if differences else "common-domain-identical-with-exclusions",
            "metal_sha256": metal["sha256"], "cuda_sha256": cuda["sha256"],
            "limits": ["Three Metal guard cases are not executed on CUDA and are never counted as CUDA passes; exit 2 requires domain review even without numerical differences.",
                       "Histogram CUDA inputs use an explicit bit-checked GPU gather to contiguous logical labels; Metal strided-input support remains local evidence.",
                       "Multi-column means replay one CUDA Mean per column; subtraction replays the multiclass transform expression, not the full objective.",
                       "Original ULP and reduction checks apply independently to both GPUs. Guard storage is additionally required bit-exact; Metal cancellation zero is not imposed on CUB.",
                       "Every bit difference is counted; up to 32 examples per case/policy are printed. Complete bits remain in the reports.",
                       "Hardware provenance requires matching build/runtime evidence, not report framing."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "unpack", "validate-metal"):
        p = sub.add_parser(name); p.add_argument("report", type=Path)
        if name == "unpack": p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare"); p.add_argument("metal", type=Path); p.add_argument("cuda", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command != "compare":
            report = load(args.report)
            if args.command == "unpack":
                with args.output.open("xb") as stream: stream.write(report["data"])
            errors = validate_metal(report) if args.command == "validate-metal" else []
            print(json.dumps({"kind": report["kind"], "cases": COUNT, "raw_sha256": report["sha256"],
                              "reference_failures": errors, "status": "failed" if errors else "complete; no CUDA equivalence claim"}))
            return 1 if errors else 0
        result = compare(load(args.metal), load(args.cuda))
        with args.output.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
        print(result["status"])
        return 1 if result["reference_failures"] else 2
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
