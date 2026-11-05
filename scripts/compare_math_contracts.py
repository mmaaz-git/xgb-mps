#!/usr/bin/env python3
"""Lossless primitive-math corpus packing and actual Metal/CUDA comparison.

Arithmetic/casts require exact results except both-NaN payload differences, as
in the existing Metal tests. Math classification changes fail; other math bit
differences require review. Existing CPU ULP budgets are not widened or reused
as an automatic GPU-to-GPU allowance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

MAGIC = b"XGBMAT1\0"
SHAPES = ((100196, 2, 6), (100007, 1, 1), (80060, 1, 3), (500400, 2, 2))
FIELDS = (("add", "subtract", "multiply", "divide", "narrow_a", "narrow_b"),
          ("widen",), ("exp", "log", "erf"), ("hypotf", "tanhf"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def width(group, column, output):
    return 32 if group == 3 or (group == 1 and not output) or (
        group == 0 and output and column >= 4) else 64


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, "Invalid math interchange header")
    kind, mask = struct.unpack_from("<II", data, 8)
    require(kind in (0, 1, 2) and 0 < mask <= 15, "Invalid math kind/group mask")
    offset, groups = 16, {}
    for group, shape in enumerate(SHAPES):
        if not mask & (1 << group):
            continue
        require(offset+16 <= len(data), "Truncated math group header")
        require(struct.unpack_from("<IIII", data, offset) == (group, *shape),
                "Wrong math identity/order/shape")
        offset += 16
        arrays = []
        for output, columns in enumerate(shape[1:]):
            if output and kind == 0:
                arrays.append(None)
                continue
            cells = shape[0]*columns
            require(offset+8*cells <= len(data), "Truncated math words")
            array = np.frombuffer(data, dtype="<u8", count=cells, offset=offset).reshape(shape[0], columns)
            offset += 8*cells
            for col in range(columns):
                require(width(group, col, output) != 32 or np.all(array[:, col] <= 0xffffffff),
                        "Nonzero upper bits in FP32 math cell")
            arrays.append(array)
        groups[group] = {"input": arrays[0], "output": arrays[1]}
    require(len(data) == offset+8 and data[offset:offset+4] == b"MEND" and
            struct.unpack_from("<I", data, offset+4)[0] == mask,
            "Missing math completion marker or trailing bytes")
    return {"kind": kind, "mask": mask, "groups": groups, "data": data,
            "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        return decode(stream.read())


def encode(kind, groups):
    mask = sum(1 << group for group in groups)
    parts = [MAGIC, struct.pack("<II", kind, mask)]
    for group, arrays in sorted(groups.items()):
        parts.extend((struct.pack("<IIII", group, *SHAPES[group]), arrays["input"].tobytes()))
        if kind:
            require(arrays["output"] is not None, "Missing math observations")
            parts.append(arrays["output"].tobytes())
    parts.extend((b"MEND", struct.pack("<I", mask)))
    data = b"".join(parts)
    decode(data)  # Validate before publication, including integer widths.
    return data


def write(path, data):
    with path.open("xb") as stream:
        stream.write(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def floating(words, bits):
    return words.view("<f8") if bits == 64 else words.astype("<u4").view("<f4")


def compare(metal, cuda):
    require(metal["kind"] == 1 and cuda["kind"] == 2 and metal["mask"] == cuda["mask"] == 15,
            "Expected complete Metal and actual CUDA math observations")
    differences, exact_failures, classification_failures, math_changes = [], 0, 0, 0
    for group, a in metal["groups"].items():
        b = cuda["groups"][group]
        require(np.array_equal(a["input"], b["input"]), f"Math group {group}: input bits differ")
        for col, name in enumerate(FIELDS[group]):
            aw, bw = a["output"][:, col], b["output"][:, col]
            changed = aw != bw
            if not changed.any():
                continue
            bits = width(group, col, True)
            av, bv = floating(aw, bits), floating(bw, bits)
            both_nan = np.isnan(av) & np.isnan(bv)
            meaningful = changed & ~both_nan
            classification = ((np.isnan(av) != np.isnan(bv)) |
                              (np.isposinf(av) != np.isposinf(bv)) |
                              (np.isneginf(av) != np.isneginf(bv)))
            finite = np.isfinite(av) & np.isfinite(bv)
            if group < 2:
                exact_failures += int(meaningful.sum())
            else:
                math_changes += int(meaningful.sum())
                classification_failures += int(classification.sum())
            sign, mask = np.uint64(1 << (bits-1)), np.uint64((1 << bits)-1)
            ak = np.where(aw & sign, (~aw) & mask, aw ^ sign)
            bk = np.where(bw & sign, (~bw) & mask, bw ^ sign)
            ulp = np.maximum(ak, bk)-np.minimum(ak, bk)
            indices = np.flatnonzero(changed)
            differences.append({"group": group, "field": name,
                "bit_differences": int(changed.sum()),
                "nan_payload_differences": int((changed & both_nan).sum()),
                "classification_differences": int(classification.sum()),
                "signed_zero_differences": int((changed & (av == 0) & (bv == 0)).sum()),
                "max_ulp_on_finite_pairs": int(ulp[finite].max()) if finite.any() else None,
                "first_differences": [{"row": int(i), "metal_bits": f"{int(aw[i]):0{bits//4}x}",
                                       "cuda_bits": f"{int(bw[i]):0{bits//4}x}"} for i in indices[:8]]})
    failures = exact_failures+classification_failures
    return {"suite": "math-v1", "status": "contract-failed" if failures else
            "math-differences-require-review" if math_changes else "matches-declared-contract-on-recorded-inputs",
            "output_fields": sum(rows*outputs for rows, _, outputs in SHAPES),
            "exact_arithmetic_cast_failures": exact_failures,
            "math_classification_failures": classification_failures,
            "math_non_nan_bit_differences": math_changes, "differences": differences,
            "nan_policy": "Both-NaN payload differences are reported and allowed by the existing primitive contract",
            "scope": "Recorded primitive inputs only; no universal error bound or full CUDA parity claim",
            "metal_sha256": metal["sha256"], "cuda_sha256": cuda["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "unpack"):
        p = commands.add_parser(name)
        p.add_argument("report", type=Path)
        if name == "unpack":
            p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("pack")
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--metal", type=Path, required=True)
    p = commands.add_parser("compare")
    p.add_argument("metal", type=Path); p.add_argument("cuda", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "pack":
            require(args.inputs.resolve() != args.metal.resolve() and not args.inputs.exists()
                    and not args.metal.exists(), "Use two new distinct math output paths")
            groups = {}
            for path in args.reports:
                report = load(path)
                require(report["kind"] == 1 and not groups.keys() & report["groups"].keys(),
                        "Packing requires disjoint Metal observation groups")
                groups.update(report["groups"])
            require(set(groups) == set(range(4)), "Packing requires every math group")
            inputs, observed = encode(0, groups), encode(1, groups)
            write(args.inputs, inputs); write(args.metal, observed)
            print("Packed all four groups with unchanged input/output bits; no CUDA comparison")
            return 0
        if args.command in ("inspect", "unpack"):
            report = load(args.report)
            if args.command == "unpack":
                with args.output.open("xb") as stream:
                    stream.write(report["data"])
            print(json.dumps({"kind": ("input-only", "metal", "cuda")[report["kind"]],
                              "groups": list(report["groups"]), "uncompressed_sha256": report["sha256"],
                              "status": "complete for declared groups; no equivalence claim"}))
            return 0
        result = compare(load(args.metal), load(args.cuda))
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
        print(result["status"])
        return 1 if result["status"] == "contract-failed" else 2 if result["math_non_nan_bit_differences"] else 0
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
