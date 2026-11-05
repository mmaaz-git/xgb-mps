#!/usr/bin/env python3
"""Compare complete metric observations with original row and reduction checks.

Each backend's reduction is checked against its own observed FP32 row terms,
using the original Neumaier sum and unchanged 32*epsilon64*magnitude budget.
All GPU-to-GPU output bit differences remain review items.
"""
import argparse
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
import sys
from typing import NamedTuple
import numpy as np

MAGIC, COUNT = b"XGBMET1\0", 375


def require(ok, message):
    if not ok: raise ValueError(message)


def bits(x, width=32):
    return struct.unpack("<I" if width == 32 else "<Q", struct.pack("<f" if width == 32 else "<d", x))[0]


def value(word, width=64):
    return struct.unpack("<f" if width == 32 else "<d", struct.pack("<I" if width == 32 else "<Q", int(word)))[0]


class Spec(NamedTuple):
    id: int
    family: int
    kind: int
    rows: int
    targets: int
    levels: int
    mode: int
    accuracy: int
    predictions: int
    labels: int
    upper: int
    weights: int
    alphas: int
    fields: int
    parameter: int


def specs():
    result = []
    def add(f, k, n, t, l, m, ac, p, y, u, w, a, fields, param):
        result.append(Spec(len(result), f, k, n, t, l, m, ac, p, y, u, w, a, fields, bits(param)))
    add(0, 0, 100057, 1, 1, 0, 0, 100057, 0, 0, 0, 0, 500285, 0.)
    def ordinary(k, n, m):
        t = 7 if k >= 11 else 3; y = n if k >= 11 else n*t
        add(1, k, n, t, 1, m, 0, n*t, y, 0, n if m else 0, 0, 2*y,
            1.3712345 if k == 9 else 1.3 if k == 10 else .371234567)
    for k in range(13):
        for n in (1, 257, 4097):
            for m in range(7): ordinary(k, n, m)
        ordinary(k, 65537, 0)
    ordinary(2, 262145, 3)
    for expectile in (0, 1):
        for n in (1, 257, 4097):
            for m in range(5): add(2, expectile, n, 3, 7, m, 0, n*21, n*3, 0, n, 7, n*42, 0.)
        add(2, expectile, 65537, 3, 7, 0, 0, 65537*21, 65537*3, 0, 65537, 7, 65537*42, 0.)
    for d in range(3):
        for m in range(4):
            for ac in (0, 1): add(3, d, 65537, 1, 1, m, ac, 65537, 65537, 65537, 65537, 0, 0, 1.)
    for d in range(3):
        for m in (0, 1): add(4, d, 5, 1, 1, m, 0, 5, 5, 5, 5, 0, 0, 1.)
    for k in (11, 12):
        for m in range(6): add(5, k, 1, 3, 1, m, 0, 3, 1, 0, 0, 0, 0, .5)
    for k in range(13): add(6, k, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, .5)
    require(len(result) == COUNT, "Incomplete metric control matrix")
    return result


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, "Invalid metric interchange header")
    kind, count = struct.unpack_from("<II", data, 8)
    require(kind in (0, 1, 2) and count == COUNT, "Wrong metric backend/count")
    offset, cases = 16, []
    def array(n, width=4):
        nonlocal offset
        require(offset+n*width <= len(data), "Truncated metric array")
        out = np.frombuffer(data, dtype=f"<u{width}", count=n, offset=offset); offset += n*width
        return out
    for spec in specs():
        require(offset+60 <= len(data), "Truncated metric identity")
        identity = struct.unpack_from("<15I", data, offset); offset += 60
        require(identity == spec, "Wrong metric identity/order/shape/parameters")
        inputs = {name: array(getattr(spec, name)) for name in ("predictions", "labels", "upper", "weights", "alphas")}
        status, actual, sums, reductions, reference, analytic = None, None, None, None, None, None
        if kind:
            status = int(array(1)[0])
            excluded = kind == 2 and spec.family == 5 and spec.mode >= 2
            require(status == 2 if excluded else status <= (1 if spec.family == 5 else 0),
                    "Unknown metric execution/domain status")
            actual = array(spec.fields)
            sums = array(2 if spec.family and not status else 0, 8)
            reductions = array(4 if spec.family in (1, 2) else 0, 8)
        if kind == 1:
            reference = array(spec.fields)
            analytic = array(2 if spec.family in (3, 4) else 0, 8)
            require(np.all(np.isfinite(analytic.view("<f8"))), "Invalid analytic metric reference")
        cases.append({"spec": spec, "inputs": inputs, "status": status, "actual": actual, "sums": sums,
                      "reductions": reductions, "reference": reference, "analytic": analytic})
    require(len(data) == offset+8 and data[offset:offset+4] == b"METE" and
            struct.unpack_from("<I", data, offset+4)[0] == COUNT,
            "Missing metric completion marker or trailing bytes")
    return {"kind": kind, "cases": cases, "data": data, "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    opener = lzma.open if path.suffix == ".xz" else gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream: return decode(stream.read())


def match_inputs(a, b):
    require(len(a["cases"]) == len(b["cases"]) == COUNT, "Incomplete metric corpus")
    for x, y in zip(a["cases"], b["cases"]):
        require(x["spec"] == y["spec"], "Different metric parameters")
        for name, words in x["inputs"].items():
            require(np.array_equal(words, y["inputs"][name]), f"Case {x['spec'].id}: {name} input bits differ")


def equal(a, b): return a == b or (math.isnan(a) and math.isnan(b))


def neumaier(words):
    values = words.view("<f4")
    total, correction, magnitude = 0., 0., 0.
    # Zero additions do not change the numerical sum/magnitude. Skipping them
    # accelerates empty/zero-weight cases; diagnostic zero signs are not oracles.
    with np.errstate(invalid="ignore"):
        nonzero = values[values != 0].astype(np.float64).tolist()
    for x in nonzero:
        magnitude += abs(x)
        new = total+x
        if math.isfinite(new) and math.isfinite(total) and math.isfinite(x):
            correction += (total-new)+x if abs(total) >= abs(x) else (x-new)+total
        total = new
    return total+correction, magnitude


def ulp_failures(actual, reference, budget):
    a, b = actual.view("<f4"), reference.view("<f4")
    same = (a == b) | (np.isnan(a) & np.isnan(b)); finite = np.isfinite(a) & np.isfinite(b)
    ak = np.where(actual >> 31, ~actual, actual ^ np.uint32(0x80000000)).astype(np.int64)
    bk = np.where(reference >> 31, ~reference, reference ^ np.uint32(0x80000000)).astype(np.int64)
    return ~same & (~finite | (np.abs(ak-bk) > budget))


def failures(case, reference, backend, diagnostics=None):
    s = case["spec"]; errors = []
    if case["status"] == 2: return errors
    def record(policy, mask):
        indices = np.flatnonzero(mask)
        if indices.size: errors.append({"case": s.id, "backend": backend, "policy": policy,
                                       "count": int(indices.size), "first_indices": indices[:32].tolist()})
    if s.family == 5:
        record("original-invalid-label-rejection", [case["status"] != 1]); return errors
    if s.fields:
        if s.family == 0: budget = np.resize(np.array([1, 1, 1, 0, 6]), s.fields)
        else:
            loss_budget = 0 if s.family == 2 or s.kind in (0, 2, 3, 5, 10, 12) else 4
            budget = np.resize(np.array([loss_budget, 0]), s.fields)
        record("original-row-ulp", ulp_failures(case["actual"], reference["reference"], budget))
        if s.family == 0:
            zero = case["inputs"]["predictions"].view("<f4") == 0
            for k in (1, 3):
                record(f"original-math-signed-zero-{k}", zero & (case["actual"][k::5] != reference["reference"][k::5]))
    if s.family in (1, 2):
        if diagnostics is None: diagnostics = [neumaier(case["actual"][c::2]) for c in (0, 1)]
        for channel, (expected, magnitude) in enumerate(diagnostics):
            recorded = case["reductions"][2*channel:2*channel+2]
            record(f"own-terms-diagnostic-{channel}", [not equal(value(recorded[0]), expected), not equal(value(recorded[1]), magnitude)])
            actual = value(case["sums"][channel]); ok = equal(actual, expected)
            if not ok and math.isfinite(actual) and math.isfinite(expected):
                bound = 32*2.**-52*magnitude
                ok = abs(actual-expected)/max(bound, 1e-300) <= 1
            record(f"original-own-terms-reduction-{channel}", [not ok])
    elif s.family in (3, 4):
        a, w = map(value, case["sums"]); expected, expected_weight = map(value, reference["analytic"])
        budget = 2e-14*max(abs(expected), 1e-300 if s.family == 3 else 1.)
        record("original-survival-analytic", [not abs(a-expected) <= budget, w != expected_weight])
    elif s.family == 6:
        record("original-empty-sums", case["sums"].view("<f8") != 0)
    return errors


def validate_metal(report, progress=False):
    require(report["kind"] == 1, "Expected Metal metric observations")
    errors = []
    for case in report["cases"]:
        errors.extend(failures(case, case, "metal"))
        if progress and (case["spec"].id+1)%25 == 0:
            print(f"Validated Metal metrics {case['spec'].id+1}/{COUNT}", file=sys.stderr, flush=True)
    return errors


def compare(metal, cuda, progress=False):
    require(metal["kind"] == 1 and cuda["kind"] == 2, "Expected Metal and actual CUDA metric outputs")
    match_inputs(metal, cuda)
    errors, differences, exclusions, compared = [], [], [], 0
    def difference(case, name, a, b, width):
        require(a.size == b.size, "Different successful metric output shapes")
        indices = np.flatnonzero(a != b)
        if indices.size:
            differences.append({"case": case, "field": name, "count": int(indices.size),
                                "first_fields": [{"index": int(i), "metal_bits": f"{a[i]:0{width*2}x}",
                                                  "cuda_bits": f"{b[i]:0{width*2}x}"} for i in indices[:32]]})
    for m, c in zip(metal["cases"], cuda["cases"]):
        s = m["spec"]; md = None
        if s.family in (1, 2): md = [neumaier(m["actual"][i::2]) for i in (0, 1)]
        cd = md if np.array_equal(m["actual"], c["actual"]) else None
        errors.extend(failures(m, m, "metal", md)); errors.extend(failures(c, m, "cuda", cd))
        if c["status"] == 2:
            exclusions.append({"case": s.id, "metal_status": "rejected", "cuda_status": "outside-domain-not-executed"})
            continue
        compared += s.fields+1
        difference(s.id, "row_fields", m["actual"], c["actual"], 4)
        difference(s.id, "status", np.array([m["status"]], dtype="<u4"), np.array([c["status"]], dtype="<u4"), 4)
        if m["status"] == c["status"]:
            compared += m["sums"].size; difference(s.id, "sums", m["sums"], c["sums"], 8)
        if progress and (s.id+1)%25 == 0: print(f"Compared metrics {s.id+1}/{COUNT}", file=sys.stderr, flush=True)
    return {"suite": "metric-v1", "cases": COUNT, "compared_fields_including_status": int(compared),
            "cuda_executed_cases": COUNT-len(exclusions), "domain_exclusions": exclusions,
            "reference_failures": errors, "failure_fields_by_policy": sum(e["count"] for e in errors),
            "different_fields": sum(d["count"] for d in differences), "differences": differences,
            "status": "reference-check-failed" if errors else "differences-require-review" if differences else "common-domain-identical-with-exclusions",
            "metal_sha256": metal["sha256"], "cuda_sha256": cuda["sha256"],
            "limits": ["Original row/math/analytic/reduction checks apply independently to each GPU; all output bit differences require review.",
                       "Eight nonfinite/out-of-int32 label casts are excluded by NUM-OP-36. Their Metal checks remain; CUDA does not execute them or fabricate sums. Exclusions require domain review, never full-parity success.",
                       "Reduction diagnostics are independently recomputed from each GPU's actual FP32 terms using Neumaier arithmetic; another GPU's terms are not a reduction oracle.",
                       "Pseudo-Huber/quantile/expectile lambdas are source-expression probes; reducer bodies and other row policies are compiled from source. Empty observations cover reducer boundaries, not constructor parameter validation.",
                       "Every differing output field is counted; at most 32 examples per case/field are printed. Full raw reports retain all values.",
                       "Hardware provenance requires matching build/runtime evidence, not report framing."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
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
            errors = validate_metal(report, True) if args.command == "validate-metal" else []
            print(json.dumps({"kind": report["kind"], "cases": COUNT, "raw_sha256": report["sha256"],
                              "reference_failures": errors, "status": "failed" if errors else "complete; no CUDA equivalence claim"}))
            return 1 if errors else 0
        result = compare(load(args.metal), load(args.cuda), True)
        with args.output.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
        print(result["status"]); return 1 if result["reference_failures"] else 2
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
