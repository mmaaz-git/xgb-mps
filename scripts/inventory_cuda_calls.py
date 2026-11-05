"""Freeze lexical CUDA dispatch/helper sites for manual source-to-Metal audit.

This is a reproducible candidate inventory, not a C++ parser or parity proof.
It includes helper definitions and backend types; manual dispositions distinguish
those from production consumers. Comments, literals and includes are excluded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "382f78ce867bfa248cd85088c70322edca8805fb"
HELPERS = (
    "ArgSort|SegmentedArgSort|SegmentedSortKeys|SegmentedArgMergeSort|CopyIf|"
    "InclusiveScan|InclusiveSum|RunLengthEncode|SegmentedSum|AllOf|"
    "SegmentedTrapezoidThreads|UnravelTrapeziodIdx|SegmentedUniqueByKey|SegmentedUnique"
)
PATTERNS = {
    "kernel-launch": re.compile(r"<<<|\b(?:LaunchN|LaunchKernel)\s*(?:<|\()"),
    "thrust": re.compile(r"\bthrust::[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*"),
    "cub": re.compile(r"\bcub::[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*"),
    "runtime": re.compile(r"\bcurt::[A-Za-z_]\w*|\bcuda[A-Z]\w*\s*\("),
    "algorithm-helper": re.compile(r"\b(?:" + HELPERS + r")\b\s*(?:<|\()"),
    "device-helper": re.compile(r"\bdh::(?:Reduce|CopyTo|MemcpyBatchAsync)\b"),
    "generic-transform": re.compile(
        r"\bcommon::Transform\s*<|\b(?:ElementWiseKernel|TransformIdxKernel|"
        r"TransformKernel|VecScaMul|VecScaDiv|LogE|SmallHistogram)\b"
    ),
}
LITERALS = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S
)
TRAINING = {
    "gbtree",
    "prediction",
    "shap",
    "linear",
    "constraints",
    "stumps",
    "quantization-sampling",
    "histograms",
    "scalar-splits",
    "vector-splits",
    "partition",
    "leaf-refit",
    "vector-model",
    "tree-updaters",
    "random",
}
NUMERIC = {
    "generic-reductions",
    "linalg",
    "regression-objectives",
    "hinge-multiclass",
    "quantile-objective",
    "aft",
    "ranking",
    "metrics",
    "ranking-metrics",
    "auc",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def owner(path, families):
    if path.endswith(("algorithm.cuh", "algorithm.h", "threading_utils.cuh")):
        return "root"
    if set(families) & TRAINING or any(
        part in path
        for part in ("/tree/", "/gbm/", "/predictor/", "/linear/", "/random.")
    ):
        return "training"
    if set(families) & NUMERIC or any(
        part in path
        for part in (
            "/objective/",
            "/metric/",
            "/ranking_",
            "/stats.",
            "/linalg",
            "/numeric.",
            "/optional_weight.",
            "/transform.",
        )
    ):
        return "numeric"
    return "data"


def generate(revision):
    mapping = json.loads(git("show", f"{revision}:validation/cuda-operation-map.json"))
    families = {}
    for group in mapping["groups"]:
        for path in group["cuda_sources"]:
            families.setdefault(path, []).append(group["id"])
    paths = git(
        "ls-tree", "-r", "--name-only", revision, "xgboost/src", "xgboost/plugin"
    )
    sources, sites = [], []
    for path in sorted(paths.decode().splitlines()):
        if Path(path).suffix not in {".cc", ".h", ".cu", ".cuh"}:
            continue
        raw = git("show", f"{revision}:{path}")
        original = raw.decode()
        stripped = LITERALS.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), original)
        found = []
        for lineno, (line, code) in enumerate(
            zip(original.splitlines(), stripped.splitlines()), 1
        ):
            if code.lstrip().startswith("#"):
                continue
            matches = {
                kind: [m[0].strip() for m in pattern.finditer(code)]
                for kind, pattern in PATTERNS.items()
            }
            matches = {kind: values for kind, values in matches.items() if values}
            if matches:
                found.append(
                    {
                        "id": f"{path}:{lineno}",
                        "path": path,
                        "line": lineno,
                        "owner": owner(path, families.get(path, [])),
                        "families": families.get(path, []),
                        "matches": matches,
                        "source": line.strip(),
                    }
                )
        if found or path in families:
            sources.append(
                {
                    "path": path,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "families": families.get(path, []),
                    "sites": len(found),
                }
            )
            sites.extend(found)
    return {
        "schema_version": 1,
        "revision": revision,
        "upstream_import": mapping["upstream_import"],
        "method": "Lexical candidate sites plus manual operation/caller classification; not exhaustive C++ semantic analysis.",
        "patterns": {kind: pattern.pattern for kind, pattern in PATTERNS.items()},
        "sources": sources,
        "sites": sites,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=BASELINE)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "validation/cuda-calls-inventory.json"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = generate(args.revision)
    if args.check:
        assert json.loads(args.output.read_text()) == result, "Frozen inventory differs"
    else:
        # Keep one candidate/source per line so the frozen ledger remains
        # inspectable without thousands of lines of nested JSON whitespace.
        arrays = ("sources", "sites")
        output = json.dumps(
            {k: v for k, v in result.items() if k not in arrays}, indent=2
        )
        for key in arrays:
            rows = ",\n".join("    " + json.dumps(row) for row in result[key])
            output = output[:-2] + f',\n  "{key}": [\n{rows}\n  ]\n}}'
        args.output.write_text(output + "\n")
    counts = {
        who: sum(site["owner"] == who for site in result["sites"])
        for who in ("root", "training", "data", "numeric")
    }
    print(
        f"{'Verified' if args.check else 'Wrote'} {len(result['sites'])} candidate sites: {counts}"
    )


if __name__ == "__main__":
    main()
