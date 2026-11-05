#!/usr/bin/env python3
"""Full root/stump corpus, with the original exact/8e-14 sum and exact weight checks.

CUDA sums and diagnostic weights are a separate replay of its source reducer;
actual production API weights are recorded separately. Neither is a capture of
private API sums. Each backend is independently checked against original input
terms; all cross-backend and API/diagnostic bit differences require review.
"""
import argparse
from collections import namedtuple
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
import sys

import numpy as np

MAGIC, COUNT = b"XGBSTP1\0", 405
Spec = namedtuple("Spec", "id rows targets layout mode stride0 stride1 pairs")


def require(ok, message):
    if not ok: raise ValueError(message)


def specs():
    result = []
    for rows in (1, 31, 32, 255, 256, 257, 2048, 4097, 65537):
        for targets in (1, 3, 7):
            for layout in range(3):
                a = 1 if layout == 1 else targets*(2 if layout == 2 else 1)
                b = rows+3 if layout == 1 else (2 if layout == 2 else 1)
                for mode in range(5):
                    result.append(Spec(len(result), rows, targets, layout, mode, a, b,
                                       (rows-1)*a+(targets-1)*b+4))
    require(len(result) == COUNT, "Incomplete stump matrix")
    return result


def bits(value, width=64):
    return struct.unpack("<Q" if width == 64 else "<I", struct.pack("<d" if width == 64 else "<f", value))[0]


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, "Invalid stump header")
    kind, count = struct.unpack_from("<II", data, 8)
    require(kind in (0, 1, 2) and count == COUNT, "Wrong stump backend/count")
    offset, cases = 16, []

    def array(n, width):
        nonlocal offset
        require(offset+n*width <= len(data), "Truncated stump array")
        out = np.frombuffer(data, dtype=f"<u{width}", count=n, offset=offset)
        offset += n*width
        return out

    for s in specs():
        require(offset+32 <= len(data), "Truncated stump identity")
        identity = struct.unpack_from("<8I", data, offset); offset += 32
        require(identity == s, "Wrong stump identity/order/shape")
        record = {"spec": s, "storage": array(2*s.pairs, 4)}
        if kind: record.update(sums=array(2*s.targets, 8), weights=array(s.targets, 4))
        if kind == 2: record["api_weights"] = array(s.targets, 4)
        if kind == 1: record["references"] = array(4*s.targets, 8)
        cases.append(record)
    require(len(data) == offset+8 and data[offset:offset+4] == b"STPE" and
            struct.unpack_from("<I", data, offset+4)[0] == COUNT,
            "Missing stump completion marker or trailing bytes")
    return {"kind": kind, "cases": cases, "data": data, "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    opener = lzma.open if path.suffix == ".xz" else gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream: return decode(stream.read())


def match_inputs(a, b):
    require(len(a["cases"]) == len(b["cases"]) == COUNT, "Incomplete stump corpus")
    for x, y in zip(a["cases"], b["cases"]):
        require(x["spec"] == y["spec"] and np.array_equal(x["storage"], y["storage"]),
                f"Stump case {x['spec'].id}: input bits/parameters differ")


def compensated(values):
    # Same independent Neumaier reference and sequential magnitude as native.
    total = correction = magnitude = 0.
    for value in values:
        value = float(value)
        next_sum = total+value
        correction += (total-next_sum)+value if abs(total) >= abs(value) else (value-next_sum)+total
        total = next_sum
        magnitude += abs(value)
    return total+correction, magnitude


def references(case):
    s = case["spec"]
    data = case["storage"].view("<f4")
    result = []
    for t in range(s.targets):
        for channel in (0, 1):
            start = 2*t*s.stride1+channel
            terms = data[start:start+2*s.rows*s.stride0:2*s.stride0]
            require(len(terms) == s.rows and np.all(np.isfinite(terms)), "Nonfinite/wrong stump input terms")
            result.extend(compensated(terms) if np.any(terms) else (0., 0.))
    return np.array(result, dtype="<f8")


def weight(g, h):
    with np.errstate(over="ignore"):
        return np.float32(-g/max(h, float(np.float32(1e-6))))


def validate(case, reference, backend):
    s = case["spec"]; failures = []
    sums, weights = case["sums"].view("<f8"), case["weights"]
    exact = s.mode in (0, 1, 4)
    for channel, actual in enumerate(sums):
        ref, magnitude = reference[2*channel:2*channel+2]
        budget = 0. if exact else magnitude*8e-14
        if not math.isfinite(actual) or abs(actual-ref) > budget:
            failures.append({"case": s.id, "backend": backend, "field": channel,
                             "policy": "original-exact-sum" if exact else "original-sum-bound"})
    for target in range(s.targets):
        g, h = map(float, sums[2*target:2*target+2])
        if not math.isfinite(g) or not math.isfinite(h): continue
        if int(weights[target]) != bits(weight(g, h), 32):
            failures.append({"case": s.id, "backend": backend, "field": target,
                             "policy": "original-weight-from-own-sums"})
        if "api_weights" not in case: continue
        # The actual API's private sums are not observable. Propagate exactly
        # the original allowed sum intervals through the source division and
        # FP32 rounding; add no new tolerance. Bit differences still require review.
        rg, mg, rh, mh = map(float, reference[4*target:4*target+4])
        word = int(case["api_weights"][target])
        actual = float(case["api_weights"].view("<f4")[target])
        if exact:
            valid = word == bits(weight(rg, rh), 32)
        else:
            eg, eh = mg*8e-14, mh*8e-14
            corners = [float(weight(gv, hv)) for gv in (rg-eg, rg+eg) for hv in (rh-eh, rh+eh)]
            valid = not math.isnan(actual) and min(corners) <= actual <= max(corners)
        if not valid:
            failures.append({"case": s.id, "backend": backend, "field": target,
                             "policy": "api-weight-original-sum-interval"})
    return failures


def validate_metal(metal, progress=False):
    require(metal["kind"] == 1, "Need Metal observations")
    failures = []
    for case in metal["cases"]:
        reference = references(case)
        if not np.array_equal(reference.view("<u8"), case["references"]):
            failures.append({"case": case["spec"].id, "backend": "metal", "policy": "original-input-reference"})
        failures.extend(validate(case, reference, "metal"))
        if progress and (case["spec"].id+1)%45 == 0:
            print(f"Validated Metal stumps {case['spec'].id+1}/{COUNT}", flush=True)
    return failures


def compare(metal, cuda):
    require(metal["kind"] == 1 and cuda["kind"] == 2, "Need Metal and actual CUDA observation kinds")
    match_inputs(metal, cuda)
    failures, differences, api_differences = [], [], []
    fields = different = api_different = 0

    def difference(case, field, a, b):
        indices = np.flatnonzero(a != b)
        return ({"case": case, "field": field, "count": len(indices),
                 "first_fields": [{"index": int(i), "a": int(a[i]), "b": int(b[i])} for i in indices[:32]]}
                if len(indices) else None)

    for m, c in zip(metal["cases"], cuda["cases"]):
        reference = references(m)
        if not np.array_equal(reference.view("<u8"), m["references"]):
            failures.append({"case": m["spec"].id, "backend": "metal", "policy": "original-input-reference"})
        failures.extend(validate(m, reference, "metal")); failures.extend(validate(c, reference, "cuda"))
        for field in ("sums", "weights", "api_weights"):
            a, b = m["weights" if field == "api_weights" else field], c[field]
            fields += len(a)
            diff = difference(m["spec"].id, field, a, b)
            if diff: differences.append(diff); different += diff["count"]
        diff = difference(m["spec"].id, "api-versus-source-probe", c["weights"], c["api_weights"])
        if diff: api_differences.append(diff); api_different += diff["count"]
    return {"suite": "stump-v1", "cases": COUNT, "compared_fields": fields,
            "reference_failures": failures, "different_fields": different, "differences": differences,
            "api_probe_different_fields": api_different, "api_probe_differences": api_differences,
            "status": "reference-failure" if failures else "review-differences" if different or api_different else "observed-bit-identical",
            "scope": "405 existing strided root/stump cases; separately executed CUDA sum probe and actual CUDA API weights; not full CUDA parity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "unpack", "validate-metal"):
        p = commands.add_parser(name); p.add_argument("input", type=Path)
        if name == "unpack": p.add_argument("output", type=Path)
    p = commands.add_parser("compare"); p.add_argument("metal", type=Path); p.add_argument("cuda", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        require(not args.output.exists(), "Output already exists")
        result = compare(load(args.metal), load(args.cuda))
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("differences", "api_probe_differences")}))
        return 1 if result["reference_failures"] else 2 if result["different_fields"] or result["api_probe_different_fields"] else 0
    report = load(args.input)
    if args.command == "unpack":
        require(report["kind"] == 0, "Only input-only records can be unpacked for CUDA")
        with args.output.open("xb") as stream: stream.write(report["data"])
    result = {"kind": report["kind"], "cases": COUNT, "raw_sha256": report["sha256"]}
    if args.command == "validate-metal":
        result["reference_failures"] = validate_metal(report, progress=True)
        result["status"] = "reference-failure" if result["reference_failures"] else "complete; no CUDA equivalence claim"
    print(json.dumps(result))
    return 1 if result.get("reference_failures") else 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL stump report: {error}", file=sys.stderr); sys.exit(1)
