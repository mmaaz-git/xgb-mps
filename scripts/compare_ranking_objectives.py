#!/usr/bin/env python3
"""Compare complete public ranking objective sequences, preserving strict subnormals.

Exit 1: invalid evidence, RNG mismatch or original tolerance exceeded. Exit 2:
other bit differences require review. Stateful CUDA race effects remain visible;
no observed Metal bias state is supplied as a CUDA input.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

MAGIC = b"XGBROB1\0"


def require(ok, message):
    if not ok: raise ValueError(message)


def specs(suite):
    require(suite in (0, 1), "Unknown objective suite")
    result = []

    def add(sequence, iteration, objective, mean, k, norm, score, exp, unbiased,
            batched, mode, seed, weighted, bias_norm, sizes):
        groups = np.concatenate((np.zeros(1, dtype="<u4"), np.cumsum(sizes, dtype="<u4")))
        rows, count = int(groups[-1]), len(sizes)
        positions = 0 if not unbiased else min(32, max(sizes)) if mean else k
        identity = (sequence, iteration, objective, mean, k, norm, score, exp, unbiased,
                    batched, mode, seed, rows, count, count if weighted else 0, positions)
        bias_bits = struct.unpack("<Q", struct.pack("<d", bias_norm))[0]
        result.append({"identity": identity, "bias_norm": bias_bits, "groups": groups})

    if suite == 0:
        for mean in (0, 1):
            for objective in range(3):
                for sizes in ([1], [7, 31, 33], [255, 256, 257], [4097, 31, 1]):
                    for k in (1, 8, 32):
                        for norm in (0, 1):
                            for score in (0, 1):
                                for mode in range(5):
                                    index = len(result)
                                    add(index, 0, objective, mean, k, norm, score, int(mode != 3),
                                        0, 0, mode, 17891+index, mode >= 2, 1., sizes)
            index = len(result)
            add(index, 0, 0, mean, 32, 1, 1, 1, 0, 1, 0, 17891+index, False, 1., [128]*2560)
    else:
        for mean in (0, 1):
            for iteration in range(2):
                add(mean, iteration, 0, mean, 8, 1, 1, 1, 1, 1, 0, 12345+iteration, False, 1., [128]*4096)
        sequence, calls = 2, 0
        for objective in range(3):
            for mean in (0, 1):
                for k in (1, 8, 32):
                    for norm in (0, 1):
                        for score in (0, 1):
                            for bias in (0., 1., 2.5):
                                for mode in range(4):
                                    for iteration in range(3):
                                        add(sequence, iteration, objective, mean, k, norm, score, 1,
                                            1, 0, mode, 8912+calls, mode == 3, bias, [1, 7, 31, 33, 257])
                                        calls += 1
                                    sequence += 1
    require(len(result) == (1442 if suite == 0 else 2596), "Incomplete objective matrix")
    return result


def decode(data):
    require(len(data) >= 20 and data[:8] == MAGIC, "Invalid objective interchange header")
    kind, suite, count = struct.unpack_from("<III", data, 8)
    matrix = specs(suite)
    require(kind in (0, 1, 2) and count == len(matrix), "Wrong objective backend/case count")
    offset, cases = 20, []

    def array(n):
        nonlocal offset
        require(offset+4*n <= len(data), "Truncated objective array")
        value = np.frombuffer(data, dtype="<u4", count=n, offset=offset); offset += 4*n
        return value

    for spec in matrix:
        require(offset+72 <= len(data), "Truncated objective case header")
        identity = struct.unpack_from("<16I", data, offset)
        bias = struct.unpack_from("<Q", data, offset+64)[0]; offset += 72
        require(identity == spec["identity"] and bias == spec["bias_norm"],
                "Wrong objective identity/order/parameters")
        rows, groups, weights, positions = identity[12:]
        boundaries = array(groups+1)
        require(np.array_equal(boundaries, spec["groups"]), "Wrong objective query boundaries")
        inputs = {"groups": boundaries, "prediction": array(rows), "label": array(rows), "weights": array(weights)}
        if identity[1]:
            previous = cases[-1]
            require(identity[0] == previous["identity"][0] and identity[1] == previous["identity"][1]+1,
                    "Missing objective sequence state")
            for name in ("label", "weights"):
                require(np.array_equal(inputs[name], previous["inputs"][name]),
                        "Ranking sequence metadata changed between objective calls")
        observed = None
        if kind:
            observed = {"gradient": array(rows*2).reshape(rows, 2), "positive": array(positions),
                        "negative": array(positions), "next_rng": array(1)}
        cases.append({"identity": identity, "bias_norm": bias, "inputs": inputs, "observed": observed})
    require(len(data) == offset+12 and data[offset:offset+4] == b"ROBE" and
            struct.unpack_from("<II", data, offset+4) == (suite, count),
            "Missing objective completion marker or trailing bytes")
    return {"kind": kind, "suite": suite, "cases": cases, "data": data,
            "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as stream:
        return decode(stream.read())


def match_inputs(a, b):
    require(a["suite"] == b["suite"] and len(a["cases"]) == len(b["cases"]), "Different objective corpora")
    for index, (ca, cb) in enumerate(zip(a["cases"], b["cases"])):
        require(ca["identity"] == cb["identity"] and ca["bias_norm"] == cb["bias_norm"], "Objective parameters differ")
        for name, values in ca["inputs"].items():
            require(np.array_equal(values, cb["inputs"][name]), f"Case {index}: {name} input bits differ")


def difference(a, b, strict_subnormal=False, exact=False):
    a, b = a.reshape(-1), b.reshape(-1)
    changed = a != b
    if not changed.any(): return None
    if exact:
        failures, subnormal = changed, np.zeros(a.size, dtype=bool)
        classification = np.zeros(a.size, dtype=bool)
        max_error = None
    else:
        x, y = a.view("<f4").astype(np.float64), b.view("<f4").astype(np.float64)
        classification = ((np.isnan(x) != np.isnan(y)) | (np.isposinf(x) != np.isposinf(y)) |
                          (np.isneginf(x) != np.isneginf(y)))
        finite = np.isfinite(x) & np.isfinite(y)
        subnormal = strict_subnormal & (((x != 0) & (np.abs(x) < 2.**-126)) |
                                       ((y != 0) & (np.abs(y) < 2.**-126)))
        with np.errstate(invalid="ignore"):
            error = np.abs(y-x)
            budget = np.where(subnormal, 2*2.**-149+2e-5*np.abs(x), 2e-6+2e-5*np.abs(x))
        failures = classification | (finite & (error > budget))
        max_error = float(error[finite].max()) if finite.any() else None
    indices = np.flatnonzero(changed)[:8]
    return {"bit_differences": int(changed.sum()), "tolerance_or_exact_failures": int(failures.sum()),
            "classification_differences": int(classification.sum()),
            "subnormal_differences": int((changed & subnormal).sum()),
            "subnormal_failures": int((failures & subnormal).sum()), "max_finite_absolute_error": max_error,
            "samples": [{"flat_index": int(i), "metal_bits": f"{int(a[i]):08x}",
                         "cuda_bits": f"{int(b[i]):08x}"} for i in indices]}


def compare(metal, cuda):
    require(metal["kind"] == 1 and cuda["kind"] == 2, "Expected Metal and actual CUDA objective outputs")
    match_inputs(metal, cuda)
    differences, failures, compared = [], 0, 0
    previous_bias_diff = False
    changed_sequences = {}
    for index, (m, c) in enumerate(zip(metal["cases"], cuda["cases"])):
        identity = m["identity"]; sequence, iteration = identity[:2]
        if iteration == 0: previous_bias_diff = False
        mo, co = m["observed"], c["observed"]
        for field, a, b in [("gradient", mo["gradient"][:, 0], co["gradient"][:, 0]),
                            ("hessian", mo["gradient"][:, 1], co["gradient"][:, 1]),
                            *[(f, mo[f], co[f]) for f in ("positive", "negative", "next_rng")]]:
            compared += a.size
            diff = difference(a, b, strict_subnormal=metal["suite"] == 0 and field in ("gradient", "hessian"),
                              exact=field == "next_rng")
            if diff:
                diff.update(case=index, sequence=sequence, iteration=iteration, field=field,
                            unbiased=bool(identity[8]), prior_serialized_bias_difference=previous_bias_diff)
                failures += diff["tolerance_or_exact_failures"]; differences.append(diff)
                changed_sequences.setdefault(str(sequence), iteration)
        previous_bias_diff = any(not np.array_equal(mo[f], co[f]) for f in ("positive", "negative"))
    return {"suite": "ranking-objective-v1", "corpus": "unbiased" if metal["suite"] else "gradient",
            "status": "tolerance-or-exact-check-failed" if failures else
                      "differences-require-review" if differences else "identical-on-recorded-inputs",
            "cases": len(metal["cases"]), "compared_fields": compared, "failures": failures,
            "differences": differences, "first_differing_iteration_by_sequence": changed_sequences,
            "metal_sha256": metal["sha256"], "cuda_sha256": cuda["sha256"],
            "limits": ["Serialized bias is FP32; matching serialized values do not establish identical hidden binary64 state.",
                       "Unbiased source shared-factor races can affect subsequent state; exceeding a tolerance needs causal review.",
                       "No numerical tolerance is widened; every bit difference remains visible.",
                       "GPU provenance requires matching build/runtime evidence, not just report framing."]}


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
            print(json.dumps({"kind": report["kind"], "corpus": "unbiased" if report["suite"] else "gradient",
                "cases": len(report["cases"]), "sequences": sum(c["identity"][1] == 0 for c in report["cases"]),
                "rows": sum(c["identity"][12] for c in report["cases"]), "raw_sha256": report["sha256"],
                "status": "complete; no equivalence claim"}))
            return 0
        result = compare(load(args.metal), load(args.cuda))
        with args.output.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
        print(result["status"])
        return 1 if result["failures"] else 2 if result["differences"] else 0
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
