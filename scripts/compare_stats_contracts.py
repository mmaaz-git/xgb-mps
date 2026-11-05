#!/usr/bin/env python3
"""Compare the complete statistics corpus using its original per-operation checks.

The Metal report carries the existing CPU/reference observations. CUDA inputs
contain only parameters and input values. Each GPU result is checked against
the references separately; all GPU-to-GPU bit differences remain review items.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

MAGIC, COUNT = b"XGBSTA1\0", 122


def require(ok, message):
    if not ok: raise ValueError(message)


def bits(value, width=64):
    return struct.unpack("<Q" if width == 64 else "<I", struct.pack("<d" if width == 64 else "<f", value))[0]


def value(word, width=64):
    return struct.unpack("<d" if width == 64 else "<f", struct.pack("<Q" if width == 64 else "<I", int(word)))[0]


def specs():
    result = []
    for rows in (0, 1, 2, 31, 256, 257, 4097):
        for columns in (1, 3):
            for mode in range(4):
                weighted = mode and rows and mode != 2
                result.append((len(result), 0, rows, columns, mode, rows*columns,
                               rows if mode else 0, 2*columns+2+(columns if weighted else 0), bits(1., 32)))
    for rows in (0, 1, 2, 31, 256, 257, 4097, 65537, 1000003):
        for mode in range(7): result.append((len(result), 1, rows, 1, mode, rows, 0, 2, bits(1., 32)))
    for mode in (0, 1): result.append((len(result), 2, 2, 1, mode, 2, 0, 2, bits(1., 32)))
    result.append((len(result), 3, 11, 1, 0, 0, 0, 1, bits(3.5, 32)))
    require(len(result) == COUNT, "Incomplete statistics control matrix")
    return result


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, "Invalid statistics interchange header")
    kind, count = struct.unpack_from("<II", data, 8)
    require(kind in (0, 1, 2) and count == COUNT, "Wrong statistics backend/count")
    offset, cases = 16, []

    def array(n, width):
        nonlocal offset
        require(offset+n*width <= len(data), "Truncated statistics array")
        out = np.frombuffer(data, dtype=f"<u{width}", count=n, offset=offset); offset += n*width
        return out

    for spec in specs():
        require(offset+36 <= len(data), "Truncated statistics case header")
        identity = struct.unpack_from("<9I", data, offset); offset += 36
        require(identity == spec, "Wrong statistics identity/order/shape/parameters")
        actual, reference, oracle = None, None, None
        inputs = {"values": array(spec[5], 4), "weights": array(spec[6], 4)}
        if kind: actual = array(spec[7], 8)
        if kind == 1: reference, oracle = array(spec[7], 8), array(2, 8)
        if spec[1] == 0:
            for words in (actual, reference):
                if words is not None:
                    require(np.all(words <= 0xffffffff), "Nonzero upper FP32/status bits")
                    require(words[-1] <= 2, "Unknown weighted-mean status")
        if kind == 1 and spec[1] == 1 and spec[4] == 2:
            require(math.isfinite(value(oracle[0])) and math.isfinite(value(oracle[1])) and value(oracle[1]) >= 0,
                    "Invalid compensated reference/magnitude")
        cases.append({"spec": spec, "inputs": inputs, "actual": actual, "reference": reference, "oracle": oracle})
    require(len(data) == offset+8 and data[offset:offset+4] == b"STAE" and
            struct.unpack_from("<I", data, offset+4)[0] == count,
            "Missing statistics completion marker or trailing bytes")
    return {"kind": kind, "cases": cases, "data": data, "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as stream:
        return decode(stream.read())


def match_inputs(a, b):
    require(len(a["cases"]) == len(b["cases"]) == COUNT, "Incomplete statistics corpus")
    for x, y in zip(a["cases"], b["cases"]):
        require(x["spec"] == y["spec"], "Different statistics parameters")
        for name, words in x["inputs"].items():
            require(np.array_equal(words, y["inputs"][name]), f"Case {x['spec'][0]}: {name} input bits differ")


def fields(spec):
    _, kind, rows, columns, mode, *_ = spec
    if kind == 0:
        out = [(f"median_{i}", 32, "exact") for i in range(columns)]
        out += [(f"sample_mean_{i}", 32, "mean") for i in range(columns)] + [("mean", 32, "mean")]
        if mode and rows and mode != 2: out += [(f"weighted_mean_{i}", 32, "mean") for i in range(columns)]
        return out + [("weighted_status", 32, "status")]
    if kind in (1, 2): return [("generic", 64, "reduce32"), ("optional", 64, "reduce64")]
    return [("scalar_optional", 64, "scalar")]


def equal_numeric(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


def validate(spec, field, word, reference, oracle):
    name, width, policy = field
    _, kind, rows, _, mode, *_ = spec
    reasons = []
    if policy == "status":
        expected = 2 if mode == 0 else 1 if not rows or mode == 2 else 0
        if int(reference) != expected: reasons.append("invalid-reference-status")
        if int(word) != expected: reasons.append("weighted-mean-status")
        return reasons
    actual, ref = value(word, width), value(reference, width)
    if kind == 2:
        # Known source precision witnesses, including the stronger exact double
        # sum for two FLT_MAX terms (the original native check asserts finite).
        expected = (1e8 if name == "generic" else 100000001.) if mode == 0 else (
            math.inf if name == "generic" else 2*value(0x7f7fffff, 32))
        if not equal_numeric(ref, expected): reasons.append("invalid-precision-reference")
        if not equal_numeric(actual, expected): reasons.append("precision-witness")
        return reasons
    if policy == "scalar":
        if ref != 38.5: reasons.append("invalid-scalar-reference")
        if actual != 38.5: reasons.append("scalar-optional-weight")
        return reasons
    if policy == "exact":
        if word != reference and not (math.isnan(actual) and math.isnan(ref)): reasons.append("exact-median")
    elif policy == "mean":
        # Preserve Close(float,float), including FP32 subtraction and budget.
        with np.errstate(invalid="ignore", over="ignore"):
            error = abs(np.float32(actual)-np.float32(ref))
            budget = np.float32(1e-6)+np.float32(2e-5)*abs(np.float32(ref))
        if not error <= budget: reasons.append("original-mean-tolerance")
    else:
        if policy == "reduce32":
            with np.errstate(invalid="ignore", over="ignore"):
                widened = float(np.float32(actual))
            if not equal_numeric(actual, widened): reasons.append("generic-result-not-widened-fp32")
        magnitude, compensated = value(oracle[1]), value(oracle[0])
        steps, n = 0, rows
        while n > 1: steps += 8; n = (n+255)//256
        eps = 2.**-23 if policy == "reduce32" else 2.**-52
        budget = steps*eps/(1-steps*eps)*magnitude if policy == "reduce32" else 32*eps*magnitude
        for target, label in [(ref, "cpu")] + ([(compensated, "compensated")] if mode < 4 and policy == "reduce32" else []):
            if equal_numeric(actual, target): continue
            if not math.isfinite(actual) or not math.isfinite(target): reasons.append(label+"-classification")
            elif mode != 2: reasons.append(label+"-exact-reduction")
            elif abs(actual-target) > budget: reasons.append(label+"-reduction-bound")
    return reasons


def compare(metal, cuda):
    require(metal["kind"] == 1 and cuda["kind"] == 2, "Expected Metal and actual CUDA statistics outputs")
    match_inputs(metal, cuda)
    differences, failures, compared = [], [], 0
    for m, c in zip(metal["cases"], cuda["cases"]):
        spec = m["spec"]
        for i, field in enumerate(fields(spec)):
            name, width, policy = field; compared += 1
            for backend, observation in [("metal", m), ("cuda", c)]:
                reasons = validate(spec, field, observation["actual"][i], m["reference"][i], m["oracle"])
                if reasons: failures.append({"case": spec[0], "field": name, "backend": backend, "reasons": reasons,
                                             "actual_bits": f"{int(observation['actual'][i]):016x}",
                                             "reference_bits": f"{int(m['reference'][i]):016x}"})
            a, b = m["actual"][i], c["actual"][i]
            if a != b:
                differences.append({"case": spec[0], "field": name, "policy": policy,
                                    "metal_bits": f"{int(a):016x}", "cuda_bits": f"{int(b):016x}"})
    return {"suite": "stats-v1", "cases": COUNT, "compared_fields": compared,
            "status": "reference-check-failed" if failures else "differences-require-review" if differences else "identical-on-recorded-inputs",
            "failures": len(failures), "reference_failures": failures, "differences": differences,
            "metal_sha256": metal["sha256"], "cuda_sha256": cuda["sha256"],
            "limits": ["Both GPUs are checked independently against recorded references and unchanged operation-specific bounds; no universal bitwise reduction claim.",
                       "All GPU-to-GPU bit differences remain review items, including signed zeros and NaN payloads permitted by a reference check.",
                       "Two-FLT_MAX optional-weight sum has an additional exact source witness beyond the original native finite classification check.",
                       "References are only in Metal observations, never in CUDA input records.",
                       "No other linalg, metric or training-reduction corpus is covered by this statistics replay.",
                       "Hardware provenance requires matching build/runtime evidence, not report framing."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "unpack"):
        p = sub.add_parser(name); p.add_argument("report", type=Path)
        if name == "unpack": p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare"); p.add_argument("metal", type=Path); p.add_argument("cuda", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command in ("inspect", "unpack"):
            report = load(args.report)
            if args.command == "unpack":
                with args.output.open("xb") as stream: stream.write(report["data"])
            print(json.dumps({"kind": report["kind"], "cases": COUNT, "raw_sha256": report["sha256"],
                              "status": "complete; no equivalence claim"})); return 0
        result = compare(load(args.metal), load(args.cuda))
        with args.output.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
        print(result["status"])
        return 1 if result["failures"] else 2 if result["differences"] else 0
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
