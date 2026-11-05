#!/usr/bin/env python3
"""Inspect/compare the complete exported AFT corpus without discarding tail cases.

Exit 0: identical observations on these inputs. Exit 2: differences need review.
Exit 1: invalid evidence or an ordinary gradient/Hessian tolerance failure.
The ordinary tolerance remains atol=1e-6, rtol=2e-5; guards and sensitive tails
are reported separately, never silently accepted as equivalent.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

MAGIC = b"XGBAFT1\0"
COUNT = 96
OUTPUT = np.dtype([("gradient", "<u8"), ("hessian", "<u8"), ("likelihood", "<u8"),
                   ("weighted_gradient", "<u4"), ("weighted_hessian", "<u4")])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def layout(index):
    within = index % 32
    scale = ([.1, .5, 1, 2, 10][within//5] if within < 25 else
             [.1, 1, 0, -1, 1e-40, np.inf, np.nan][within-25])
    mode, rows = (within % 5, 4097) if within < 25 else (5, 448)
    scale_bits = int(np.asarray(scale, dtype="<f4").view("<u4"))
    return index//32, mode, scale_bits, rows, rows if mode else 0


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, "Invalid AFT interchange header")
    kind, count = struct.unpack_from("<II", data, 8)
    require(kind in (0, 1, 2) and count == COUNT, "Wrong AFT backend or case count")
    offset, cases = 16, []
    for index in range(COUNT):
        require(offset+20 <= len(data), "Truncated AFT case header")
        identity = struct.unpack_from("<IIIII", data, offset)
        offset += 20
        require(identity == layout(index), f"Case {index}: wrong identity/order/shape")
        distribution, mode, scale_bits, rows, weights = identity
        inputs = {}
        for name, size in (("lower", rows), ("upper", rows), ("prediction", rows), ("weights", weights)):
            require(offset+4*size <= len(data), "Truncated AFT inputs")
            inputs[name] = np.frombuffer(data, dtype="<u4", count=size, offset=offset)
            offset += 4*size
        observations = None
        if kind:
            require(offset+OUTPUT.itemsize*rows <= len(data), "Truncated AFT observations")
            observations = np.frombuffer(data, dtype=OUTPUT, count=rows, offset=offset)
            offset += OUTPUT.itemsize*rows
        cases.append({"identity": identity, "inputs": inputs, "observations": observations})
    require(len(data) == offset+8 and data[offset:offset+4] == b"AFTE"
            and struct.unpack_from("<I", data, offset+4)[0] == COUNT,
            "Missing AFT completion marker or trailing bytes")
    return {"kind": kind, "cases": cases, "sha256": hashlib.sha256(data).hexdigest(), "data": data}


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        result = decode(stream.read())
    result["path"] = str(path)
    return result


def floats(words):
    return words.view("<f8" if words.dtype.itemsize == 8 else "<f4")


def sensitive(case):
    different_bounds = floats(case["inputs"]["lower"]) != floats(case["inputs"]["upper"])
    return different_bounds & ~(floats(case["observations"]["likelihood"]) > float(np.float32(1e-5)))


def compare(metal, cuda):
    require(metal["kind"] == 1 and cuda["kind"] == 2, "Expected Metal and actual CUDA observation kinds")
    differences, ordinary_failures, total_rows = [], 0, 0
    sensitive_counts = {"metal": 0, "cuda": 0}
    for index, (a, b) in enumerate(zip(metal["cases"], cuda["cases"])):
        require(a["identity"] == b["identity"] and all(np.array_equal(a["inputs"][k], b["inputs"][k])
                for k in a["inputs"]), f"Case {index}: input bits differ")
        sa, sb = sensitive(a), sensitive(b)
        sensitive_counts["metal"] += int(sa.sum()); sensitive_counts["cuda"] += int(sb.sum())
        guard = a["identity"][1] == 5
        ordinary = ~(sa | sb) if not guard else np.zeros_like(sa)
        total_rows += a["identity"][3]
        fields = {}
        for field in OUTPUT.names:
            aw, bw = a["observations"][field], b["observations"][field]
            changed = np.flatnonzero(aw != bw)
            if not len(changed):
                continue
            av, bv = floats(aw).astype(np.float64), floats(bw).astype(np.float64)
            finite = np.isfinite(av) & np.isfinite(bv)
            both_nan = np.isnan(av) & np.isnan(bv)
            same = (av == bv) | both_nan
            delta = np.zeros_like(av)
            with np.errstate(over="ignore", invalid="ignore"):
                delta[finite] = np.abs(av[finite]-bv[finite])
                close = same | (finite & (delta <= 1e-6+2e-5*np.abs(bv)))
            failures = ordinary & ~close if field.startswith("weighted_") else np.zeros_like(ordinary)
            ordinary_failures += int(failures.sum())
            maximum = float(delta[finite].max()) if finite.any() else None
            width = aw.dtype.itemsize*2
            fields[field] = {"bit_differences": int(len(changed)),
                "ordinary_tolerance_failures": int(failures.sum()),
                "nan_payload_differences": int(np.count_nonzero(both_nan & (aw != bw))),
                "classification_differences": int(np.count_nonzero(
                    (np.isnan(av) != np.isnan(bv)) | (np.isposinf(av) != np.isposinf(bv)) |
                    (np.isneginf(av) != np.isneginf(bv)))),
                "max_abs_on_finite_pairs": maximum if maximum is None or np.isfinite(maximum) else "overflow",
                "first_differences": [{"row": int(i), "metal_bits": f"{int(aw[i]):0{width}x}",
                    "cuda_bits": f"{int(bw[i]):0{width}x}",
                    "region": "guard" if guard else "sensitive" if sa[i] or sb[i] else "ordinary"}
                    for i in changed[:8]]}
        if fields:
            differences.append({"case": index, "distribution": a["identity"][0],
                                "mode": a["identity"][1], "scale_bits": a["identity"][2], "fields": fields})
    return {"status": "ordinary-tolerance-failed" if ordinary_failures else
            "differences-require-review" if differences else "exact-on-recorded-inputs",
            "scope": "P5-N02 corpus only; not full CUDA parity or a universal math-error bound",
            "cases": COUNT, "rows": total_rows, "sensitive_rows": sensitive_counts,
            "ordinary_tolerance_failures": ordinary_failures, "differences": differences,
            "metal_sha256": metal["sha256"], "cuda_sha256": cuda["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "unpack"):
        p = commands.add_parser(name)
        p.add_argument("report", type=Path)
        if name == "unpack":
            p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("compare")
    p.add_argument("metal", type=Path); p.add_argument("cuda", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command in ("inspect", "unpack"):
            report = load(args.report)
            if args.command == "unpack":
                with args.output.open("xb") as output:
                    output.write(report["data"])
            print(json.dumps({"kind": ("input-only", "metal", "cuda")[report["kind"]],
                              "cases": COUNT, "rows": sum(c["identity"][3] for c in report["cases"]),
                              "uncompressed_sha256": report["sha256"],
                              "status": "complete; no equivalence claim"}))
            return 0
        result = compare(load(args.metal), load(args.cuda))
        with args.output.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False); output.write("\n")
        print(result["status"])
        return 1 if result["ordinary_tolerance_failures"] else 2 if result["differences"] else 0
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
