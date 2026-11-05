#!/usr/bin/env python3
"""Observe production GPU sketches/training and compare complete Metal/CUDA runs.

This is a correctness experiment, not a benchmark. Comparisons return 0 for
identical observed outputs, 2 for differences requiring review, and 1 for invalid
evidence. No result establishes parity outside the recorded cases.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xgboost/python-package"))
import numpy as np
import xgboost as xgb

CASES = ("dense", "missing", "categorical", "row_weight", "tiny_weight",
         "zero_weight", "group_weight", "approx_hessian")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", *args],
                                   cwd=ROOT, text=True).strip()


def write(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def words(values):
    return np.asarray(values, dtype="<f4").view("<u4").ravel().tolist()


def fingerprint(values):
    values = np.asarray(values)
    canonical = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<"))
    return {"shape": list(values.shape), "dtype": canonical.dtype.str,
            "sha256": hashlib.sha256(canonical.view(np.uint8)).hexdigest()}


def fixture_rows(name):
    rows = 513 if name in ("categorical", "tiny_weight", "zero_weight") else 8193
    return 8192 if name == "group_weight" else rows


def fixture(name):
    rows = fixture_rows(name)
    # Integer arithmetic and binary fractions avoid RNG/library-math variation.
    r = np.arange(rows, dtype=np.int64)[:, None]
    f = np.arange(9, dtype=np.int64)[None, :]
    x = (((r*7919 + f*104729 + r*r*17) % 65521)-32760).astype(np.float32) / np.float32(32)
    y = ((r[:, 0]*13 % 17) > 8).astype(np.float32)
    metadata = {}
    types = ["q"] * 9
    if name == "missing":
        x[(r+f*3) % 7 == 0] = np.nan
        x[:, -1] = np.nan
    if name == "categorical":
        x[:, 2] = (r[:, 0] % 7)*2
        types[2] = "c"
    if name in ("row_weight", "tiny_weight", "zero_weight", "approx_hessian"):
        weight = ((r[:, 0]*3) % 8).astype(np.float32) / np.float32(4)
        if name == "tiny_weight":
            weight *= np.float32(2.**-140)
        if name == "zero_weight":
            weight[:] = 0
        metadata["weight"] = weight
    if name == "group_weight":
        y = (r[:, 0] % 4).astype(np.float32)
        metadata.update(group=np.full(32, 256, dtype=np.uint32),
                        weight=(np.arange(32) % 8).astype(np.float32)/np.float32(4))
    return x, y, x[:513].copy(), metadata, types


def predictive_model(model):
    raw = json.loads(model.save_raw(raw_format="json"))["learner"]
    result = {"base_score_bits": words(json.loads(raw["learner_model_param"]["base_score"])), "trees": []}
    for tree in raw["gradient_booster"]["model"]["trees"]:
        categories = {node: tree["categories"][start:start+size] for node, start, size in zip(
            tree["categories_nodes"], tree["categories_segments"], tree["categories_sizes"])}
        paths, pending = {}, [(0, "root")]
        while pending:
            node, path = pending.pop()
            left, right = tree["left_children"][node], tree["right_children"][node]
            if left == -1:
                paths[path] = {"leaf_bits": words([tree["split_conditions"][node]])[0]}
            else:
                paths[path] = {"feature": tree["split_indices"][node],
                               "default_left": tree["default_left"][node],
                               "split_type": tree["split_type"][node]}
                if tree["split_type"][node]:
                    paths[path]["categories"] = categories[node]
                else:
                    paths[path]["threshold_bits"] = words([tree["split_conditions"][node]])[0]
                pending.extend([(right, path+"R"), (left, path+"L")])
        result["trees"].append(paths)
    return result


def observe(name, device, cache):
    if name == "higgs":
        x = np.load(cache / "X.npy", mmap_mode="r")
        y = np.load(cache / "y.npy", mmap_mode="r")
        require(x.shape == (11_000_000, 28) and y.shape == (11_000_000,), "Wrong HIGGS cache shape")
        test = x[10_500_000:10_501_024]
        x, y = x[:10_500_000], y[:10_500_000]
        metadata, types, rounds, bins = {}, ["q"]*28, 1, 256
    else:
        x, y, test, metadata, types = fixture(name)
        rounds, bins = 3, 256 if name == "dense" else 33
    inputs = {k: fingerprint(v) for k, v in {"x": x, "y": y, "test": test, **metadata}.items()}
    method = "approx" if name == "approx_hessian" else "hist"
    params = dict(device=device, tree_method=method, max_bin=bins, max_depth=6, nthread=8,
                  seed=0, base_score=.5, objective="rank:pairwise" if name == "group_weight" else "binary:logitraw",
                  eta=.3, min_child_weight=1, reg_lambda=1, reg_alpha=0, subsample=1,
                  colsample_bytree=1)
    matrix = xgb.DMatrix(x, label=y, feature_types=types, nthread=8, **metadata)
    before = xgb.metal_info() if device == "mps" else {}
    model = xgb.train(params, matrix, rounds)
    config = json.loads(model.save_config())["learner"]
    actual_device = config["generic_param"]["device"]
    require(actual_device == ("mps" if device == "mps" else "cuda:0"), "GPU request fell back to another device")
    updater = config["gradient_booster"]["gbtree_train_param"]["updater"]
    if device == "cuda":
        require(updater == ("grow_gpu_approx" if method == "approx" else "grow_gpu_hist"), "CUDA updater not selected")
    ptr, cuts = matrix.get_quantile_cut()
    data = xgb.DMatrix(test, feature_types=types, nthread=8)
    margins = model.predict(data, output_margin=True)
    after = xgb.metal_info() if device == "mps" else {}
    counts = {k: after[k]-before[k] for k in after if k not in ("built", "available")}
    if device == "mps":
        require(counts["cuda_sketch_batches"] > 0 and counts["histogram_dispatches"] > 0
                and counts["prediction_dispatches"] > 0, "Missing Metal sketch/training/prediction execution")
        require(counts["sketch_finalize_dispatches"] == 0, "Unexpected host-sketch path")
    cpu = xgb.Booster(params={"device": "cpu", "nthread": 8}, model_file=model.save_raw())
    expected = cpu.predict(data, output_margin=True)
    delta = np.abs(margins.astype(np.float64)-expected.astype(np.float64))
    require(np.isfinite(margins).all() and np.isfinite(expected).all(), "Nonfinite predictions")
    outside = int(np.count_nonzero(delta > 2e-5+2e-5*np.abs(expected)))
    require(outside == 0, "GPU prediction disagrees with CPU evaluation of the same saved model")
    params.pop("device")
    return {"name": name, "inputs": inputs, "feature_types": types, "params": params, "rounds": rounds,
            "cuts": {"ptr": ptr.tolist(), "bits": words(cuts)}, "model": predictive_model(model),
            "margins_bits": words(margins), "cpu_prediction_check": {"outside_tolerance": outside,
                "max_abs": float(delta.max()), "exact_values": int(np.count_nonzero(delta == 0))},
            "execution": {"device": actual_device, "updater": updater, "metal_counters": counts}}


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x"):
        pass
    record = {"schema_version": 1, "kind": "gpu-pipeline", "status": "running", "device": args.device,
              "suite": args.suite, "started_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    try:
        build = json.loads(args.build_record.read_text())
        info = xgb.build_info()
        library = Path(info["libxgboost"])
        require(not git("diff", "HEAD", "--", "xgboost"), "Production sources have uncommitted changes")
        tree = git("rev-parse", "HEAD:xgboost")
        require(tree == build["production_tree"] and sha(library) == build["library_sha256"],
                "Loaded library/source do not match the supplied build record")
        require(build["status"] == "passed-compile-link" or (build["status"] == "finished"
                and build.get("production_unchanged_after") and build.get("library_unchanged_after")),
                "Build record is not a successful frozen build/run")
        record.update(revision=git("rev-parse", "HEAD"), production_tree=tree,
                      library_sha256=sha(library), build_record_sha256=sha(args.build_record),
                      script_sha256=sha(Path(__file__)), platform=platform.platform(),
                      numpy_version=np.__version__, xgboost_version=xgb.__version__, build_info=info)
        if args.device == "mps":
            require(info["USE_METAL"] and xgb.metal_info()["available"], "Metal unavailable")
            record["hardware"] = json.loads(subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"], text=True))
        else:
            require(info["USE_CUDA"], "Loaded XGBoost is not a CUDA build")
            record["hardware"] = subprocess.check_output(["nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
            record["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        names = CASES if args.suite == "fixtures-v1" else ("higgs",)
        for i, name in enumerate(names):
            record["active_case"] = name
            write(args.output, record)
            print(f"{i+1}/{len(names)} {args.device}: {name}", flush=True)
            record["cases"].append(observe(name, args.device, args.cache))
        require(sha(library) == record["library_sha256"] and sha(Path(__file__)) == record["script_sha256"]
                and git("rev-parse", "HEAD:xgboost") == tree
                and not git("diff", "HEAD", "--", "xgboost"), "Library/source changed during observation")
        record.pop("active_case")
        record["status"] = "completed"
        validate(record)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output, record)


def validate(record):
    require(record.get("schema_version") == 1 and record.get("kind") == "gpu-pipeline"
            and record.get("status") == "completed", "Incomplete pipeline observation")
    require(record.get("device") in ("mps", "cuda"), "Invalid GPU backend")
    require(record.get("suite") in ("fixtures-v1", "higgs-v1"), "Unknown input suite")
    names = CASES if record["suite"] == "fixtures-v1" else ("higgs",)
    require([c["name"] for c in record["cases"]] == list(names), "Missing, duplicated or reordered cases")

    def floating_bits(values, allow_infinite=False):
        require(isinstance(values, list) and all(type(b) is int and 0 <= b < 2**32 for b in values),
                "Invalid FP32 words")
        floats = np.asarray(values, dtype="<u4").view("<f4")
        require((~np.isnan(floats) if allow_infinite else np.isfinite(floats)).all(),
                "Invalid nonfinite FP32 observation")

    for case in record["cases"]:
        columns = len(case["feature_types"])
        rows, expected_columns, samples = ((10_500_000, 28, 1024) if case["name"] == "higgs" else
                                          (fixture_rows(case["name"]), 9, 513))
        require(columns == expected_columns and case["inputs"]["x"]["shape"] == [rows, columns]
                and case["inputs"]["y"]["shape"] == [rows]
                and case["inputs"]["test"]["shape"] == [samples, columns]
                and all(t in ("q", "c") for t in case["feature_types"]), "Invalid feature metadata")
        fields = {"x", "y", "test"}
        if case["name"] in ("row_weight", "tiny_weight", "zero_weight", "approx_hessian", "group_weight"):
            fields.add("weight")
        if case["name"] == "group_weight":
            fields.add("group")
        require(case["inputs"].keys() == fields, "Missing input metadata")
        for item in case["inputs"].values():
            digest = item["sha256"]
            require(isinstance(digest, str) and len(digest) == 64
                    and all(c in "0123456789abcdef" for c in digest), "Invalid input fingerprint")
        ptr, bits = case["cuts"]["ptr"], case["cuts"]["bits"]
        require(len(ptr) == columns+1 and ptr[0] == 0 and ptr[-1] == len(bits)
                and all(type(p) is int for p in ptr) and ptr == sorted(ptr), "Invalid cut column pointers")
        floating_bits(bits, allow_infinite=True)
        for i, kind in enumerate(case["feature_types"]):
            column = bits[ptr[i]:ptr[i+1]]
            if kind == "q":
                # C API GetCutImpl prepends NumericBinLowerBound = -infinity.
                require(len(column) >= 2 and column[0] == 0xff800000, "Missing numeric lower boundary")
                floating_bits(column[1:])
            else:
                floating_bits(column)
        floating_bits(case["margins_bits"])
        require(len(case["margins_bits"]) == case["inputs"]["test"]["shape"][0], "Missing prediction values")
        floating_bits(case["model"]["base_score_bits"])
        require(len(case["model"]["base_score_bits"]) == 1, "Missing model intercept")
        require(case["rounds"] == (1 if case["name"] == "higgs" else 3)
                and len(case["model"]["trees"]) == case["rounds"], "Missing predictive trees")
        for tree in case["model"]["trees"]:
            pending, visited = ["root"], set()
            while pending:
                path = pending.pop()
                require(path in tree, "Missing tree path")
                visited.add(path)
                node = tree[path]
                if "leaf_bits" in node:
                    require(set(node) == {"leaf_bits"}, "Ambiguous leaf")
                    floating_bits([node["leaf_bits"]])
                else:
                    require(type(node["feature"]) is int and 0 <= node["feature"] < columns
                            and node["default_left"] in (0, 1) and node["split_type"] in (0, 1),
                            "Invalid split metadata")
                    if node["split_type"]:
                        require(isinstance(node["categories"], list)
                                and all(type(c) is int and c >= 0 for c in node["categories"]),
                                "Invalid category partition")
                    else:
                        # Missing/nonmissing partitions can split at -infinity.
                        floating_bits([node["threshold_bits"]], allow_infinite=True)
                    pending.extend([path+"L", path+"R"])
            require(visited == tree.keys(), "Unreachable tree paths")
        require(case["cpu_prediction_check"]["outside_tolerance"] == 0, "Failed CPU predictor control")
        require(case["execution"]["device"] == ("mps" if record["device"] == "mps" else "cuda:0"),
                "Observation used another device")
        if record["device"] == "mps":
            counts = case["execution"]["metal_counters"]
            require(all(counts[k] > 0 for k in ("cuda_sketch_batches", "histogram_dispatches", "prediction_dispatches"))
                    and counts["sketch_finalize_dispatches"] == 0, "Missing Metal GPU execution evidence")
        else:
            expected = "grow_gpu_approx" if case["params"]["tree_method"] == "approx" else "grow_gpu_hist"
            require(case["execution"]["updater"] == expected, "Missing CUDA updater execution evidence")


def compare(metal, cuda):
    validate(metal)
    validate(cuda)
    require(metal["device"] == "mps" and cuda["device"] == "cuda", "Expected Metal followed by CUDA")
    for field in ("production_tree", "script_sha256", "suite"):
        require(metal[field] == cuda[field], f"Mismatched {field}")
    differences = []
    for a, b in zip(metal["cases"], cuda["cases"]):
        for field in ("inputs", "feature_types", "params", "rounds"):
            require(a[field] == b[field], f"{a['name']}: mismatched {field}")
        changed = [f for f in ("cuts", "model", "margins_bits") if a[f] != b[f]]
        if changed:
            features = [i for i in range(len(a["feature_types"])) if
                a["cuts"]["bits"][a["cuts"]["ptr"][i]:a["cuts"]["ptr"][i+1]] !=
                b["cuts"]["bits"][b["cuts"]["ptr"][i]:b["cuts"]["ptr"][i+1]]]
            first = None
            for i, (at, bt) in enumerate(zip(a["model"]["trees"], b["model"]["trees"])):
                for path in sorted(at.keys() | bt.keys(), key=lambda p: (len(p), p)):
                    if at.get(path) != bt.get(path):
                        first = {"tree": i, "path": path, "metal": at.get(path), "cuda": bt.get(path)}
                        break
                if first:
                    break
            margins = [np.asarray(c["margins_bits"], dtype="<u4").view("<f4").astype(np.float64) for c in (a, b)]
            differences.append({"name": a["name"], "changed": changed, "features_with_different_cuts": features,
                                "first_tree_difference": first,
                                "margin_max_abs": float(np.max(np.abs(margins[0]-margins[1])))})
    return {"status": "differences-require-review" if differences else "identical-on-recorded-cases",
            "cases": len(metal["cases"]), "differences": differences,
            "scope": "Recorded quantile/training/prediction cases only; not full CUDA parity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    run_parser.add_argument("--suite", choices=("fixtures-v1", "higgs-v1"), default="fixtures-v1")
    run_parser.add_argument("--build-record", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--cache", type=Path, default=ROOT / ".cache/datasets/higgs")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("metal", type=Path)
    compare_parser.add_argument("cuda", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args)
            print(f"Recorded {args.suite} on {args.device}; NVIDIA/Metal comparison remains separate")
            return 0
        result = compare(json.loads(args.metal.read_text()), json.loads(args.cuda.read_text()))
        result["reports"] = {str(p): sha(p) for p in (args.metal, args.cuda)}
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(result["status"])
        return 2 if result["differences"] else 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
