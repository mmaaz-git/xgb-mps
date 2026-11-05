#!/usr/bin/env python3
"""Compare the full ranking rounding boundary corpus without hiding domain policies.

Finite-domain results retain the original exact checks (NaN payload differences
remain review items). CUDA's unspecified exponent on nonfinite frexp inputs is
separate; only truncations with different actual factors inherit that review.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

MAGIC, COUNT = b"XGBRPR1\0", 162851
QUERY = np.dtype([(n, "<u4") for n in ("maximum", "rows", "value")] +
                 [(n, "<u8") for n in ("cost", "minimum", "term")])
RESULT = np.dtype([(n, "<u4") for n in ("factor", "positive", "negative", "delta")] +
                  [(n, "<u8") for n in ("cost_factor", "cost_positive", "cost_negative", "cost_delta")])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decode(data):
    require(len(data) >= 16 and data[:8] == MAGIC, "Invalid primitive interchange header")
    kind, count = struct.unpack_from("<II", data, 8)
    require(kind in (0, 1, 2) and count == COUNT, "Wrong primitive backend/count")
    offset = 16 + QUERY.itemsize * count
    size = offset + (RESULT.itemsize * count if kind else 0)
    require(len(data) == size+8 and data[size:size+4] == b"RPRE" and
            struct.unpack_from("<I", data, size+4)[0] == count,
            "Incomplete primitive corpus/footer or trailing bytes")
    return {"kind": kind, "inputs": np.frombuffer(data, QUERY, count, 16),
            "observed": np.frombuffer(data, RESULT, count, offset) if kind else None,
            "data": data, "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as stream:
        return decode(stream.read())


def match_inputs(a, b):
    for name in QUERY.names:
        require(np.array_equal(a["inputs"][name], b["inputs"][name]),
                f"Primitive {name} input bits/order differ")


def values(words):
    return words.view("<f4" if words.dtype.itemsize == 4 else "<f8")


def difference(a, b, policy):
    changed = a != b
    if not changed.any():
        return None
    x, y = values(a), values(b)
    both_nan = np.isnan(x) & np.isnan(y)
    failure = changed & ~both_nan & ~policy
    classification = ((np.isnan(x) != np.isnan(y)) | (np.isposinf(x) != np.isposinf(y)) |
                      (np.isneginf(x) != np.isneginf(y)))
    return {"bit_differences": int(changed.sum()), "exact_failures": int(failure.sum()),
            "nonfinite_factor_policy_differences": int((changed & policy).sum()),
            "nan_payload_or_sign_differences": int((changed & both_nan).sum()),
            "classification_differences": int((changed & classification).sum()),
            "samples": [{"row": int(i), "metal_bits": f"{int(a[i]):0{a.dtype.itemsize*2}x}",
                         "cuda_bits": f"{int(b[i]):0{b.dtype.itemsize*2}x}",
                         "policy_review": bool(policy[i])} for i in np.flatnonzero(changed)[:8]]}


def compare(metal, cuda):
    require(metal["kind"] == 1 and cuda["kind"] == 2, "Expected Metal and actual CUDA primitive outputs")
    match_inputs(metal, cuda)
    m, c = metal["observed"], cuda["observed"]
    differences, domains, failures = [], {}, 0
    no_policy = np.zeros(COUNT, dtype=bool)
    for prefix in ("", "cost_"):
        delta = prefix+"delta"
        mn, cn = ~np.isfinite(values(m[delta])), ~np.isfinite(values(c[delta]))
        common_nonfinite = mn & cn
        factor_changed = m[prefix+"factor"] != c[prefix+"factor"]
        domains["float" if not prefix else "double"] = {
            "both_finite": int((~mn & ~cn).sum()), "both_nonfinite": int(common_nonfinite.sum()),
            "finite_nonfinite_disagreement": int((mn != cn).sum()),
            "nonfinite_domain_factor_differences": int((common_nonfinite & factor_changed).sum())}
        for field in ("factor", "positive", "negative", "delta"):
            # A nonfinite delta does not waive the separate arithmetic diagnostic.
            # Equal actual factors also leave the signed truncations exact.
            policy = (no_policy if field == "delta" else common_nonfinite if field == "factor"
                      else common_nonfinite & factor_changed)
            name = prefix+field
            diff = difference(m[name], c[name], policy)
            if diff:
                diff["field"] = name
                differences.append(diff); failures += diff["exact_failures"]
    return {"suite": "ranking-primitive-v1", "cases": COUNT, "result_fields": COUNT*6,
            "diagnostic_fields": COUNT*2, "compared_fields": COUNT*8,
            "status": "exact-check-failed" if failures else "differences-require-review" if differences
                      else "identical-on-recorded-inputs",
            "failures": failures, "differences": differences, "domains": domains,
            "metal_sha256": metal["sha256"], "cuda_sha256": cuda["sha256"],
            "limits": ["Delta diagnostics are separate source evaluations, not captures of private temporaries.",
                       "Nonfinite frexp exponent policy is reviewed separately only when both diagnostics are nonfinite.",
                       "Dependent truncation differences are review items only when actual factors also differ; this is not proof of causation.",
                       "No finite-domain tolerance is widened. Signed-zero differences fail exact checks; NaN payload differences remain review items.",
                       "Build/runtime provenance is required; valid framing alone does not authenticate NVIDIA execution."]}


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
            print(json.dumps({"kind": report["kind"], "cases": COUNT,
                              "raw_sha256": report["sha256"], "status": "complete; no equivalence claim"}))
            return 0
        result = compare(load(args.metal), load(args.cuda))
        with args.output.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
        print(result["status"])
        return 1 if result["failures"] else 2 if result["differences"] else 0
    except (ValueError, OSError, EOFError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
