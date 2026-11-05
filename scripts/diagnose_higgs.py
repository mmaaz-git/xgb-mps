#!/usr/bin/env python3
"""Diagnose a completed full-HIGGS replay without changing production behavior.

Compare saved models by tree path, check the saved Metal model on the CPU
predictor, and reproduce each backend's first round with exported quantile cuts.
An optional common-cut run separates quantization from downstream training.
These are correctness diagnostics, not replacement benchmark timings.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xgboost/python-package"))
import numpy as np
import xgboost as xgb


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def trees(model):
    return json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]


def tree_paths(tree):
    """Ignore allocation-specific node IDs and non-predictive training statistics."""
    result = {}
    pending = [(0, "root")]
    while pending:
        node, path = pending.pop()
        left, right = tree["left_children"][node], tree["right_children"][node]
        if left == -1:
            result[path] = {"kind": "leaf", "weight_bits": bits(tree["split_conditions"][node])}
        else:
            if tree["split_type"][node] != 0:
                raise ValueError("This HIGGS diagnostic expects numeric scalar trees")
            result[path] = {"kind": "split", "feature": tree["split_indices"][node],
                            "threshold": tree["split_conditions"][node],
                            "threshold_bits": bits(tree["split_conditions"][node]),
                            "default_left": tree["default_left"][node]}
            pending.extend([(right, path+"R"), (left, path+"L")])
    return result


def model_difference(cpu, mps):
    result = {"cpu_trees": len(cpu), "mps_trees": len(mps), "different_trees": 0,
              "first_prediction_field_difference": None, "first_split_difference": None,
              "first_tree_roots": {"cpu": tree_paths(cpu[0])["root"], "mps": tree_paths(mps[0])["root"]}}
    if len(cpu) != len(mps):
        raise ValueError("The saved models have different tree counts")
    for i, (a, b) in enumerate(zip(cpu, mps)):
        a, b = tree_paths(a), tree_paths(b)
        different = [p for p in sorted(a.keys() | b.keys(), key=lambda p: (len(p), p)) if a.get(p) != b.get(p)]
        if not different:
            continue
        result["different_trees"] += 1
        for path in different:
            observation = {"tree": i, "path": path, "cpu": a.get(path), "mps": b.get(path)}
            if result["first_prediction_field_difference"] is None:
                result["first_prediction_field_difference"] = observation
            if (a.get(path, {}).get("kind") != "leaf" or b.get(path, {}).get("kind") != "leaf"):
                if result["first_split_difference"] is None:
                    result["first_split_difference"] = observation
    return result


def margin_difference(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("Prediction shapes differ")
    difference = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    mismatch = difference > 2e-5+2e-5*np.abs(expected)
    return {"finite": finite, "strict_equivalence": bool(finite and not mismatch.any()),
            "values": int(expected.size), "outside_tolerance": int(mismatch.sum()),
            "max_abs": float(difference.max()), "rmse": float(np.sqrt(np.mean(difference**2))),
            "exact_bits": int(np.count_nonzero(actual.view(np.uint32) == expected.view(np.uint32)))}


def common_cut_worker(args, benchmark, x, y):
    device = args.worker.removeprefix("common-")
    params = benchmark["runs"][device][0]["params"]
    rows = benchmark["runs"][device][0]["shapes"]["train"][0]
    threads, bins = params["nthread"], params["max_bin"]
    print(f"{device}: constructing a CPU reference cut set from {rows} rows", flush=True)
    reference = xgb.QuantileDMatrix(x[:rows], label=y[:rows], nthread=threads, max_bin=bins)
    if device == "cpu":
        matrix = reference
    else:
        matrix = xgb.QuantileDMatrix(x[:rows], label=y[:rows], nthread=threads,
                                    max_bin=bins, ref=reference, device="mps")
    ptr, cuts = matrix.get_quantile_cut()
    for actual, expected in zip((ptr, cuts), reference.get_quantile_cut()):
        np.testing.assert_array_equal(actual, expected)
    del reference
    np.savez(args.output_dir / f"{device}-cuts.npz", ptr=ptr, cuts=cuts)
    test = xgb.DMatrix(x[10_500_000:11_000_000], label=y[10_500_000:11_000_000], nthread=threads)

    class Progress(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if (epoch+1) % 10 == 0 or epoch+1 == args.rounds:
                print(f"{device}: common-cut round {epoch+1}/{args.rounds}", flush=True)
            return False

    history = {}
    before = xgb.metal_info()
    model = xgb.train(params, matrix, num_boost_round=args.rounds, evals=[(test, "test")],
                      evals_result=history, verbose_eval=False, callbacks=[Progress()])
    after = xgb.metal_info()
    assert (after["histogram_dispatches"] > before["histogram_dispatches"]) == (device == "mps")
    model.save_model(args.output_dir / f"{device}.ubj")
    np.save(args.output_dir / f"{device}.margins.npy", model.predict(test, output_margin=True))
    write(args.output_dir / f"{device}.json", {
        "status": "recorded", "params": params, "rounds": args.rounds,
        "cut_count": len(cuts), "metric_history": history,
        "model_sha256": sha(args.output_dir / f"{device}.ubj"),
        "margins_sha256": sha(args.output_dir / f"{device}.margins.npy"),
        "dispatch_statistics": {k: after[k]-before[k] for k in after if k not in ("built", "available")}})


def worker(args, benchmark):
    artifacts = args.benchmark.with_suffix("")
    x = np.load(args.cache / "X.npy", mmap_mode="r")
    y = np.load(args.cache / "y.npy", mmap_mode="r")
    if args.worker.startswith("common-"):
        common_cut_worker(args, benchmark, x, y)
        return
    rows = benchmark["runs"]["cpu"][0]["shapes"]["train"][0]
    threads = benchmark["cpu_threads"]
    if args.worker == "predict":
        print("CPU predictor: loading the saved MPS model and all 500,000 test rows", flush=True)
        model = xgb.Booster(model_file=artifacts / "mps-1.ubj")
        model.set_param({"device": "cpu", "nthread": threads})
        test = xgb.DMatrix(x[10_500_000:11_000_000], nthread=threads)
        before = xgb.metal_info()
        actual = model.predict(test, output_margin=True)
        after = xgb.metal_info()
        assert after["prediction_dispatches"] == before["prediction_dispatches"]
        expected = np.load(artifacts / "mps-1.margins.npy")
        result = margin_difference(actual, expected)
        result["metal_prediction_dispatches"] = 0
        np.save(args.output_dir / "saved-mps-model-cpu-margins.npy", actual)
        write(args.output_dir / "predict.json", result)
        print("CPU prediction of saved MPS model:", result, flush=True)
        return

    device = args.worker
    params = benchmark["runs"][device][0]["params"]
    # Reproduce the benchmark's warmup, both fresh ordinary matrices and test
    # evaluation. Check exact first-tree replay before attributing the saved
    # benchmark model's behavior to these newly exported cuts.
    warm = xgb.DMatrix(np.arange(64, dtype=np.float32).reshape(32, 2),
                       label=np.arange(32, dtype=np.float32) % 2, nthread=threads)
    xgb.train(params, warm, num_boost_round=1, evals=[(warm, "warm")], verbose_eval=False)
    del warm
    print(f"{device}: constructing {rows} rows, then one original-path round", flush=True)
    matrix = xgb.DMatrix(x[:rows], label=y[:rows], nthread=threads)
    test = xgb.DMatrix(x[10_500_000:11_000_000], label=y[10_500_000:11_000_000], nthread=threads)
    before = xgb.metal_info()
    model = xgb.train(params, matrix, num_boost_round=1, evals=[(test, "test")], verbose_eval=False)
    after = xgb.metal_info()
    assert (after["histogram_dispatches"] > before["histogram_dispatches"]) == (device == "mps")
    ptr, cuts = matrix.get_quantile_cut()
    np.savez(args.output_dir / f"{device}-cuts.npz", ptr=ptr, cuts=cuts)
    model.save_model(args.output_dir / f"{device}-first.json")
    previous = xgb.Booster(model_file=artifacts / f"{device}-1.ubj")
    replay = tree_paths(trees(model)[0]) == tree_paths(trees(previous)[0])
    result = {"status": "recorded", "first_tree_exact_replay": replay,
              "root": tree_paths(trees(model)[0])["root"], "cut_count": len(cuts),
              "dispatch_statistics": {k: after[k]-before[k] for k in after if k not in ("built", "available")}}
    write(args.output_dir / f"{device}.json", result)
    print(f"{device}: exported {len(cuts)} cuts; first tree exact replay={replay}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/datasets/higgs")
    parser.add_argument("--common-cuts", action="store_true",
                        help="Train fresh CPU/Metal quantile matrices with identical CPU reference cuts")
    parser.add_argument("--rounds", type=int, default=100, help="Rounds for the common-cut diagnostic")
    parser.add_argument("--worker", choices=("cpu", "mps", "predict", "common-cpu", "common-mps"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    provenance = json.loads(args.provenance.read_text())
    if benchmark["preset"] != "nvidia2017-higgs" or benchmark["status"] not in ("complete", "prediction_equivalence_failed"):
        parser.error("Use a terminal HIGGS replay; never run this during benchmark timing")
    if not provenance.get("production_unchanged_after") or not provenance.get("library_unchanged_after"):
        parser.error("The benchmark did not finish with verified source/library provenance")
    if sha(ROOT / "xgboost/lib/libxgboost.dylib") != provenance["library_sha256"]:
        parser.error("The production library changed since the benchmark")
    if not 1 <= args.rounds <= benchmark["rounds"]:
        parser.error("--rounds must be positive and no greater than the recorded benchmark")
    if args.worker:
        worker(args, benchmark)
        return

    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "scope": "HIGGS correctness diagnosis; not a timing benchmark",
              "mode": "common-cuts" if args.common_cuts else "first-divergence",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "benchmark_sha256": sha(args.benchmark), "provenance_sha256": sha(args.provenance),
              "library_sha256": provenance["library_sha256"], "script_sha256": sha(Path(__file__)),
              "production_source_commit": provenance["production_source_commit"], "steps": []}
    output = args.output_dir / "diagnosis.json"
    write(output, record)
    try:
        for filename in ("X.npy", "y.npy"):
            original = provenance["dataset_sha256"][f".cache/datasets/higgs/{filename}"]
            if sha(args.cache / filename) != original:
                raise ValueError(f"Dataset changed: {filename}")
        if not args.common_cuts:
            artifacts = args.benchmark.with_suffix("")
            models = [xgb.Booster(model_file=artifacts / f"{device}-1.ubj") for device in ("cpu", "mps")]
            record["saved_model_difference"] = model_difference(*(trees(m) for m in models))
            record["model_sha256"] = {device: sha(artifacts / f"{device}-1.ubj") for device in ("cpu", "mps")}
            write(output, record)
            print("Saved model comparison:", record["saved_model_difference"], flush=True)
        stages = ("common-cpu", "common-mps") if args.common_cuts else ("predict", "cpu", "mps")
        for stage in stages:
            command = [sys.executable, str(Path(__file__).resolve()), "--benchmark", str(args.benchmark),
                       "--provenance", str(args.provenance), "--output-dir", str(args.output_dir),
                       "--cache", str(args.cache), "--worker", stage, "--rounds", str(args.rounds)]
            print("Starting diagnostic stage:", stage, flush=True)
            record["steps"].append({"stage": stage, "command": command, "status": "running"})
            write(output, record)
            result = subprocess.run(command, cwd=ROOT)
            record["steps"][-1].update(exit_code=result.returncode, status="passed" if result.returncode == 0 else "failed")
            write(output, record)
            result.check_returncode()
        observations = {device: json.loads((args.output_dir / f"{device}.json").read_text()) for device in ("cpu", "mps")}
        if args.common_cuts:
            record["rounds"] = args.rounds
            record["training"] = observations
            models = [xgb.Booster(model_file=args.output_dir / f"{device}.ubj") for device in ("cpu", "mps")]
            record["common_cut_model_difference"] = model_difference(*(trees(m) for m in models))
            record["common_cut_margins"] = margin_difference(
                np.load(args.output_dir / "mps.margins.npy"), np.load(args.output_dir / "cpu.margins.npy"))
            record["metric_history_max_abs_difference"] = float(np.max(np.abs(
                np.array(observations["cpu"]["metric_history"]["test"]["error"])
                - np.array(observations["mps"]["metric_history"]["test"]["error"]))))
        else:
            record["cpu_prediction_of_mps_model"] = json.loads((args.output_dir / "predict.json").read_text())
            record["first_round"] = observations
        cuts = {device: np.load(args.output_dir / f"{device}-cuts.npz") for device in ("cpu", "mps")}
        comparisons = []
        for feature in range(len(cuts["cpu"]["ptr"])-1):
            columns = {d: v["cuts"][v["ptr"][feature]:v["ptr"][feature+1]] for d, v in cuts.items()}
            a, b = columns["cpu"], columns["mps"]
            exact = a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))
            same_length = len(a) == len(b)
            indices = np.flatnonzero(a.view(np.uint32) != b.view(np.uint32)) if same_length else []
            comparisons.append({"feature": feature, "cpu_count": len(a), "mps_count": len(b),
                                "exact": bool(exact), "same_index_differences": len(indices) if same_length else None,
                                "first_differences": [{"index": int(i), "cpu": float(a[i]), "mps": float(b[i])} for i in indices[:5]]})
        record["cut_comparison"] = comparisons
        record["features_with_different_cuts"] = sum(not c["exact"] for c in comparisons)
        if args.common_cuts and record["features_with_different_cuts"] != 0:
            raise ValueError("Common-cut workers produced different cuts; comparison is invalid")
        if sha(ROOT / "xgboost/lib/libxgboost.dylib") != record["library_sha256"]:
            raise ValueError("The library changed during the diagnostic")
        record["status"] = "recorded; interpretation-required"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(output, record)


if __name__ == "__main__":
    main()
