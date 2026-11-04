"""Replay published histogram workloads with fresh CPU/MPS worker processes.

Published CUDA timings are context, not measurements of this Mac. Download and
prepare inputs with prepare_published_data.py before starting timed experiments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xgboost/python-package"))
import numpy as np
import xgboost as xgb
from benchmark import memory_snapshot


PRESETS = {
    "nvidia2017-higgs": {
        "dataset": "higgs", "rounds": 1000,
        "source": "https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/",
        "params": {"objective": "binary:logitraw", "eval_metric": "error", "eta": 0.3},
        "split": "First 10,500,000 train; last 500,000 test; evaluate test every round",
    },
    "gbm2018-covtype": {
        "dataset": "covtype", "rounds": 500,
        "source": "https://arxiv.org/abs/1806.11248",
        "source_code": "https://github.com/RAMitchell/GBM-Benchmarks/blob/a0bbed08c918b0a82e9a5e2207d1f43134b445e0/benchmark.py",
        # The reference loader keeps labels 1..7 and computes max(label)+1.
        # Preserve its unused class 0 instead of silently changing the workload.
        "params": {"objective": "multi:softmax", "num_class": 8,
                   "eval_metric": "merror", "eta": 0.1},
        "split": "60/20/20 random_state=0 in two train_test_split calls; evaluate train and val",
    },
}


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    temporary.replace(path)


class Progress(xgb.callback.TrainingCallback):
    def __init__(self, device, rounds, every, output):
        self.device, self.rounds, self.every, self.output = device, rounds, every, output
        self.start = time.perf_counter()
        self.last = self.start
        self.boundaries = []

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.every == 0 or epoch + 1 == self.rounds:
            now = time.perf_counter()
            item = {"round": epoch + 1, "elapsed_seconds": now - self.start,
                    "segment_seconds": now - self.last}
            self.last = now
            self.boundaries.append(item)
            print(f"{self.device}: round {epoch+1}/{self.rounds}, {now-self.start:.2f}s", flush=True)
            write_json(self.output.with_suffix(".progress.json"), {"status": "training", **item})
        return False


def data_views(args):
    preset = PRESETS[args.preset]
    base = args.cache / preset["dataset"]
    x = np.load(base / "X.npy", mmap_mode="r")
    y = np.load(base / "y.npy", mmap_mode="r")
    if preset["dataset"] == "higgs":
        rows = args.train_rows or 10_500_000
        if not 0 < rows <= 10_500_000:
            raise ValueError("HIGGS train-rows must be in [1, 10500000]")
        indices = {"train": slice(0, rows), "test": slice(10_500_000, 11_000_000)}
    else:
        from sklearn.model_selection import train_test_split
        if args.train_rows is not None:
            raise ValueError("train-rows is only supported for HIGGS")
        train, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0)
        train, val = train_test_split(train, test_size=0.2 / (1 - 0.2), random_state=0)
        indices = {"train": train, "val": val, "test": test}
    arrays = {name: (x[idx], y[idx]) for name, idx in indices.items()}
    fingerprint = {name: hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()
                   for name, idx in indices.items() if not isinstance(idx, slice)}
    return arrays, fingerprint


def worker(args):
    device = args.worker
    preset = PRESETS[args.preset]
    rounds = args.rounds or preset["rounds"]
    params = dict(tree_method="hist", device=device, nthread=args.threads, seed=0,
                  max_depth=6, max_bin=256, min_child_weight=1, gamma=0,
                  reg_alpha=0, reg_lambda=1, subsample=1, colsample_bytree=1,
                  base_score=0.5, **preset["params"])
    if device == "mps":
        assert xgb.metal_info()["available"]
    warm_y = np.arange(32, dtype=np.float32) % (8 if preset["dataset"] == "covtype" else 2)
    warm = xgb.DMatrix(np.arange(64, dtype=np.float32).reshape(32, 2), label=warm_y,
                       nthread=args.threads)
    xgb.train(params, warm, num_boost_round=1, evals=[(warm, "warm")], verbose_eval=False)
    del warm
    arrays, fingerprint = data_views(args)
    print(f"{device}: constructing fresh matrices {[(k, v[0].shape) for k,v in arrays.items()]}", flush=True)
    prepare_start = time.perf_counter()
    matrices = {k: xgb.DMatrix(x, label=y, nthread=args.threads) for k, (x, y) in arrays.items()}
    setup = time.perf_counter() - prepare_start
    test_y = arrays["test"][1]
    shapes = {k: list(v[0].shape) for k, v in arrays.items()}
    del arrays
    evals = ([(matrices["test"], "test")] if preset["dataset"] == "higgs" else
             [(matrices["train"], "train"), (matrices["val"], "val")])
    history = {}
    before, mem_before = xgb.metal_info(), memory_snapshot()
    progress = Progress(device, rounds, args.progress_every, args.output)
    wall_start = time.time()
    start = time.perf_counter()
    model = xgb.train(params, matrices["train"], num_boost_round=rounds,
                      evals=evals, evals_result=history, verbose_eval=False, callbacks=[progress])
    elapsed = time.perf_counter() - start
    wall_elapsed = time.time() - wall_start
    after = xgb.metal_info()
    counters = {k: after[k] - before[k] for k in after if k not in {"built", "available"}}
    report = dict(status="trained", device=device, params=params, rounds=rounds, shapes=shapes,
                  split_index_sha256=fingerprint, dmatrix_seconds=setup, training_seconds=elapsed,
                  dmatrix_plus_training_seconds=setup + elapsed,
                  clock={"started_utc": datetime.fromtimestamp(wall_start, timezone.utc).isoformat(),
                         "wall_seconds": wall_elapsed, "monotonic_seconds": elapsed},
                  memory={"before": mem_before, "after_training": memory_snapshot()},
                  dispatch_statistics=counters, metric_history=history,
                  training_segments=progress.boundaries)
    write_json(args.output, report)
    assert abs(wall_elapsed - elapsed) <= 1, "Sleep or wall-clock discontinuity invalidates timing"
    assert (counters["histogram_dispatches"] > 0) == (device == "mps")
    prediction_start = time.perf_counter()
    margins = model.predict(matrices["test"], output_margin=True)
    prediction = model.predict(matrices["test"])
    report["test_prediction_seconds"] = time.perf_counter() - prediction_start
    if margins.ndim == 1:
        # Conventional binary quality uses margin 0 / probability 0.5. The old
        # logitraw+error metric uses a different threshold; retain both values.
        classes = margins > 0
        margin64 = margins.astype(np.float64)
        loss = np.logaddexp(0, margin64) - test_y * margin64
    else:
        classes = prediction
        z = margins.astype(np.float64)
        z -= np.max(z, axis=1, keepdims=True)
        loss = np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(test_y)), test_y.astype(int)]
    report["test_accuracy"] = float(np.mean(classes == test_y))
    report["test_logloss"] = float(np.mean(loss))
    np.save(args.output.with_suffix(".margins.npy"), margins)
    np.save(args.output.with_suffix(".classes.npy"), classes)
    model.save_model(args.output.with_suffix(".ubj"))
    report["status"] = "complete"
    write_json(args.output, report)
    print(f"{device}: train {elapsed:.3f}s; DMatrix {setup:.3f}s; accuracy {report['test_accuracy']:.6f}", flush=True)


def main(args):
    if args.output.exists():
        raise FileExistsError(f"Inspect existing result or select a new name: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    preset = PRESETS[args.preset]
    provenance = json.loads((args.cache / preset["dataset"] / "manifest.json").read_text())
    report = dict(status="in_progress", preset=args.preset, published=preset,
                  dataset=provenance, platform=platform.platform(), cpu_threads=args.threads,
                  xgboost_version=xgb.__version__, numpy_version=np.__version__,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  working_tree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
                  rounds=args.rounds or preset["rounds"], repeats=args.repeats,
                  train_rows_override=args.train_rows, runs={"cpu": [], "mps": []}, comparisons=[],
                  timing="Fresh process/matrices per run; warmed runtime; training includes first quantization and per-round evaluations; DMatrix/loading/download excluded and setup reported separately",
                  equivalence="All test margins: rtol=2e-5, atol=2e-5 against CPU; class disagreement and accuracy/logloss differences reported independently",
                  deviations=["Apple M3 CPU/Metal and current XGBoost, not the historical CUDA/software/hardware",
                              "Fresh matrices for each run, alternating CPU/MPS order, runtime warmup",
                              "Progress every N rounds instead of printing evaluation every round",
                              "FP32 input cache; explicitly fixed historical base_score=0.5 and histogram defaults"])
    write_json(args.output, report)
    artifacts = args.output.with_suffix("")
    artifacts.mkdir(exist_ok=False)
    try:
        for repeat in range(args.repeats):
            paths = {}
            for device in (["cpu", "mps"] if repeat % 2 == 0 else ["mps", "cpu"]):
                path = artifacts / f"{device}-{repeat+1}.json"
                paths[device] = path
                command = [sys.executable, str(Path(__file__).resolve()), "--preset", args.preset,
                           "--cache", str(args.cache), "--output", str(path), "--worker", device,
                           "--rounds", str(report["rounds"]), "--threads", str(args.threads),
                           "--progress-every", str(args.progress_every)]
                if args.train_rows is not None:
                    command += ["--train-rows", str(args.train_rows)]
                print(f"Repeat {repeat+1}/{args.repeats}: {device}; artifact {path}", flush=True)
                result = subprocess.run(command)
                if result.returncode:
                    report["failed_worker"] = {"device": device, "repeat": repeat+1,
                                               "returncode": result.returncode, "artifact": str(path)}
                    raise RuntimeError("Worker failed; partial logs/results retained")
                report["runs"][device].append(json.loads(path.read_text()))
                write_json(args.output, report)
            cpu, mps = [np.load(paths[d].with_suffix(".margins.npy")) for d in ("cpu", "mps")]
            difference = np.abs(cpu.astype(np.float64) - mps.astype(np.float64))
            cclass, mclass = [np.load(paths[d].with_suffix(".classes.npy")) for d in ("cpu", "mps")]
            mismatch = difference > 2e-5 + 2e-5 * np.abs(cpu)
            finite = np.isfinite(cpu).all() and np.isfinite(mps).all()
            compare = dict(repeat=repeat+1, strict_margin_equivalence=bool(finite and not mismatch.any()),
                           margin_max_abs_difference=float(np.max(difference)),
                           margin_rmse_difference=float(np.sqrt(np.mean(difference ** 2))),
                           margin_mismatch_count=int(np.count_nonzero(mismatch)),
                           margin_values=int(cpu.size), class_disagreement_count=int(np.count_nonzero(cclass != mclass)),
                           test_rows=len(cclass))
            report["comparisons"].append(compare)
            print(f"Equivalence repeat {repeat+1}: {compare}", flush=True)
            write_json(args.output, report)
        medians = {d: statistics.median(r["training_seconds"] for r in report["runs"][d]) for d in ("cpu", "mps")}
        setup_medians = {d: statistics.median(r["dmatrix_plus_training_seconds"] for r in report["runs"][d]) for d in ("cpu", "mps")}
        report.update(training_median_seconds=medians, training_cpu_over_mps=medians["cpu"]/medians["mps"],
                      dmatrix_plus_training_median_seconds=setup_medians,
                      dmatrix_plus_training_cpu_over_mps=setup_medians["cpu"]/setup_medians["mps"],
                      status="complete" if all(c["strict_margin_equivalence"] for c in report["comparisons"]) else "prediction_equivalence_failed")
    except BaseException:
        report["status"] = "worker_failed"
        write_json(args.output, report)
        raise
    write_json(args.output, report)
    print(f"{report['status']}: CPU/MPS={report['training_cpu_over_mps']:.3f}x; {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/datasets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--train-rows", type=int, help="HIGGS scaling case; keep the original last-500k test set")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--worker", choices=["cpu", "mps"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds or PRESETS[args.preset]["rounds"], args.repeats, args.threads, args.progress_every) < 1:
        parser.error("rounds, repeats, threads and progress-every must be positive")
    main(args)
