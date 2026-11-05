#!/usr/bin/env python3
"""Compare actual ranking outputs without disguising CUDA's cost-factor race.

Original local tolerances remain unchanged. Every bit difference is retained for
review, including within-tolerance differences and source-race-dependent bias.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

MAGIC, COUNT = b"XGBRND1\0", 64
GROUP = np.dtype([("normalization", "<u4"), ("gradient_rounding", "<u4"),
                  ("hessian_rounding", "<u4"), ("weight", "<u4"), ("cost_rounding", "<u8")])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def layout(index):
    largest = 65537 if index//4 % 2 else 257
    mean = index//16 % 2
    return index//32, mean, index//8 % 2, largest, index % 4, 129+largest, 32 if mean else 8


def repetitions(kind, identity):
    return 0 if kind == 0 else 32 if kind == 2 and identity[3] == 257 else 1


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, "Invalid ranking interchange header")
    kind, count = struct.unpack_from("<II", data, 8)
    require(kind in (0, 1, 2) and count == COUNT, "Invalid ranking backend/case count")
    offset, cases = 16, []

    def array(dtype, n):
        nonlocal offset
        dtype = np.dtype(dtype)
        require(offset+dtype.itemsize*n <= len(data), "Truncated ranking array")
        values = np.frombuffer(data, dtype=dtype, count=n, offset=offset)
        offset += dtype.itemsize*n
        return values

    for index in range(COUNT):
        require(offset+32 <= len(data), "Truncated ranking case header")
        header = struct.unpack_from("<8I", data, offset); offset += 32
        identity, repeats = header[:7], header[7]
        require(identity == layout(index) and repeats == repetitions(kind, identity),
                "Wrong ranking identity/order/shape/repetitions")
        rows, positions = identity[5:]
        inputs = {name: array(dtype, n) for name, dtype, n in (
            ("prediction", "<u4", rows), ("label", "<u4", rows), ("weights", "<u4", 5),
            ("positive", "<u8", positions), ("negative", "<u8", positions))}
        observations = []
        for rep in range(repeats):
            require(int(array("<u4", 1)[0]) == rep, "Missing/reordered ranking repetition")
            start = offset
            observed = {"global_cost_rounding": array("<u8", 1), "weight_norm": array("<u8", 1),
                        "groups": array(GROUP, 5), "gradients": array("<u4", rows*2).reshape(rows, 2),
                        "prefix": array("<u8", rows*2 if identity[0] else 0).reshape(-1, 2),
                        "positive": array("<u8", positions), "negative": array("<u8", positions)}
            observed["sha256"] = hashlib.sha256(data[start:offset]).hexdigest()
            observations.append(observed)
        cases.append({"identity": identity, "inputs": inputs, "observations": observations})
    require(len(data) == offset+8 and data[offset:offset+4] == b"RNDE" and
            struct.unpack_from("<I", data, offset+4)[0] == COUNT,
            "Missing ranking completion marker or trailing bytes")
    return {"kind": kind, "cases": cases, "data": data, "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as stream:
        return decode(stream.read())


def match_inputs(a, b):
    for index, (ca, cb) in enumerate(zip(a["cases"], b["cases"])):
        require(ca["identity"] == cb["identity"], "Ranking case identities differ")
        for name, values in ca["inputs"].items():
            require(np.array_equal(values, cb["inputs"][name]), f"Ranking case {index}: {name} bits differ")


def floats(words):
    return words.view("<f4" if words.dtype.itemsize == 4 else "<f8").astype(np.float64)


def difference(a, b, tolerance):
    a, b = a.reshape(-1), b.reshape(-1)
    changed = a != b
    if not changed.any():
        return None
    x, y = floats(a), floats(b)
    both_nan = np.isnan(x) & np.isnan(y)
    classification = ((np.isnan(x) != np.isnan(y)) | (np.isposinf(x) != np.isposinf(y)) |
                      (np.isneginf(x) != np.isneginf(y)))
    with np.errstate(invalid="ignore", over="ignore"):
        error = np.abs(y-x)/np.maximum(1., np.abs(x))
    if tolerance is None:
        failed = np.zeros(a.size, dtype=bool)
    elif tolerance == 0:
        failed = changed  # Exact fields retain signed-zero and NaN payload distinctions.
    else:
        failed = classification | (np.isfinite(x) & np.isfinite(y) & (error > tolerance))
    finite_error = error[np.isfinite(error)]
    ids = np.flatnonzero(changed)[:8]
    width = 2*a.dtype.itemsize
    return {"bit_differences": int(changed.sum()), "contract_failures": int(failed.sum()),
            "classification_differences": int(classification.sum()),
            "nan_payload_differences": int((changed & both_nan).sum()),
            "signed_zero_differences": int((changed & (x == 0) & (y == 0)).sum()),
            "max_finite_scaled_error": float(finite_error.max()) if finite_error.size else None,
            "nonfinite_scaled_errors": int((changed & ~np.isfinite(error) & ~both_nan).sum()),
            "tolerance_times_max_one_abs_metal": tolerance,
            "samples": [{"flat_index": int(i), "metal_bits": f"{int(a[i]):0{width}x}",
                         "cuda_bits": f"{int(b[i]):0{width}x}"} for i in ids]}


def compare(metal, cuda):
    require(metal["kind"] == 1 and cuda["kind"] == 2, "Expected Metal and actual CUDA observations")
    match_inputs(metal, cuda)
    differences, race_cases, failures, fields = [], [], 0, 0
    for index, (mc, cc) in enumerate(zip(metal["cases"], cuda["cases"])):
        m = mc["observations"][0]
        domain = mc["identity"][4] == 2  # Zero bias gives nonfinite frexp inputs.
        winners, observed_hashes = set(), set()
        winner_checks = []
        for rep, c in enumerate(cc["observations"]):
            winners.add(int(c["global_cost_rounding"][0])); observed_hashes.add(c["sha256"])
            winner_checks.append(bool(np.any(c["groups"]["cost_rounding"] == c["global_cost_rounding"][0])))
            race_changed = bool(m["global_cost_rounding"][0] != c["global_cost_rounding"][0])
            pairs = [("global_cost_rounding", m["global_cost_rounding"], c["global_cost_rounding"], None),
                     ("weight_norm", m["weight_norm"], c["weight_norm"], None),
                     ("gradient", m["gradients"][:, 0], c["gradients"][:, 0], 2e-5),
                     ("hessian", m["gradients"][:, 1], c["gradients"][:, 1], 2e-5),
                     ("map_hits", m["prefix"][:, 0], c["prefix"][:, 0], 2e-12),
                     ("map_precision", m["prefix"][:, 1], c["prefix"][:, 1], 2e-12)]
            for field, tol in (("normalization", 2e-7), ("gradient_rounding", 0),
                               ("hessian_rounding", 0), ("weight", 0),
                               ("cost_rounding", None if domain else 0)):
                pairs.append((field, m["groups"][field], c["groups"][field], tol))
            for field in ("positive", "negative"):
                pairs.append((field, m[field], c[field], None if domain or race_changed else 2e-12))
            for field, a, b, tol in pairs:
                fields += a.size
                diff = difference(a, b, tol)
                if diff:
                    diff.update(case=index, repetition=rep, field=field,
                                domain_guard=domain, different_global_factor=race_changed)
                    failures += diff["contract_failures"]; differences.append(diff)
        race_cases.append({"case": index, "repetitions": len(cc["observations"]),
            "metal_global_factor_bits": f"{int(m['global_cost_rounding'][0]):016x}",
            "cuda_distinct_global_factor_bits": [f"{v:016x}" for v in sorted(winners)],
            "cuda_distinct_complete_observations": len(observed_hashes),
            "winner_matches_replayed_query_factor": winner_checks, "domain_guard": domain})
    return {"suite": "ranking-rounding-v1", "status": "contract-failed" if failures else
            "differences-require-review" if differences else "identical-on-recorded-inputs",
            "contract_failures": failures, "compared_fields_including_repetitions": fields,
            "differences": differences, "race_observations": race_cases,
            "metal_sha256": metal["sha256"], "cuda_sha256": cuda["sha256"],
            "limits": ["Repeated observations do not enumerate every legal race outcome.",
                "Per-query CUDA cost factors are separate source evaluations from cached maxima.",
                "Bias with a different global factor and nonfinite-exponent guards require review; never silently accepted.",
                "This corpus does not replace the broader ranking gradient/unbiased objective suites."]}


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
            print(json.dumps({"kind": report["kind"], "cases": len(report["cases"]),
                "executions": sum(len(c["observations"]) for c in report["cases"]),
                "raw_sha256": report["sha256"], "status": "complete; no equivalence claim"}))
            return 0
        result = compare(load(args.metal), load(args.cuda))
        with args.output.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
        print(result["status"])
        return 1 if result["contract_failures"] else 2 if result["differences"] else 0
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
