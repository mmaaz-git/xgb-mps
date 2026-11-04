"""Benchmark public grouped ROC/PR evaluation on identical fixed predictions.

This isolates metric evaluation. Training and DMatrix construction are outside
the timer; first evaluation and warm prediction-cache timings are separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
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


def save(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def evaluate(model, data, device, metric):
    before = xgb.metal_info()
    memory_before = memory_snapshot()
    wall_start = time.time()
    start = time.perf_counter()
    text = model.eval(data)
    elapsed = time.perf_counter() - start
    wall_elapsed = time.time() - wall_start
    after = xgb.metal_info()
    counters = {key: after[key] - before[key] for key in after if key not in {"built", "available"}}
    score = float(text.rsplit(":", 1)[1])
    return dict(device=device, metric=metric, seconds=elapsed, score=score, dispatch_statistics=counters,
                clock=dict(started_utc=datetime.fromtimestamp(wall_start, timezone.utc).isoformat(),
                           wall_seconds=wall_elapsed, monotonic_seconds=elapsed),
                memory=dict(before=memory_before, after=memory_snapshot()))


def validate(run, expected):
    assert abs(run["clock"]["wall_seconds"] - run["seconds"]) <= 1, "Sleep/clock discontinuity invalidates timing"
    family = "pr" if run["metric"] == "aucpr" else "roc"
    assert run["dispatch_statistics"][f"ranking_{family}_auc_dispatches"] == int(run["device"] == "mps")
    assert run["dispatch_statistics"][f"{family}_auc_dispatches"] == int(run["device"] == "mps")
    np.testing.assert_allclose(run["score"], expected, rtol=2e-12, atol=2e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["auc", "aucpr"], default="auc")
    parser.add_argument("--rows", type=int, nargs="+", default=[100000, 1000000])
    parser.add_argument("--query-sizes", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--max-cpu-pairs", type=int, default=2_000_000_000,
                        help="Refuse accidental quadratic CPU workloads exceeding this many pairs per evaluation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Inspect existing result or choose another output: {args.output}")
    if min(args.rows) < 3 or min(args.query_sizes) < 3 or args.repeats < 1 or args.threads < 1:
        parser.error("Rows/query-sizes must be at least three; repeats/threads must be positive")
    for n in args.rows:
        for size in args.query_sizes:
            whole, tail = divmod(n, size)
            pairs = whole * size * (size - 1) // 2 + tail * (tail - 1) // 2
            if args.metric == "auc" and pairs > args.max_cpu_pairs:
                parser.error(f"{n} rows/query-size {size} would enumerate {pairs} CPU pairs; adjust --max-cpu-pairs explicitly")
    assert xgb.metal_info()["available"], "An accessible Metal device is required"
    params = dict(objective="reg:squarederror", eval_metric=args.metric, base_score=.5, nthread=args.threads)
    label_count = 2 if args.metric == "aucpr" else 8
    warm = xgb.DMatrix(np.ones((8, 1), dtype=np.float32), label=np.arange(8) % 2,
                       base_margin=np.arange(8), group=[4, 4], nthread=args.threads)
    for device in ["cpu", "mps"]:
        xgb.train(params | {"device": device}, warm, 0).eval(warm)
    del warm
    report = dict(status="in_progress", platform=platform.platform(), cpu_threads=args.threads,
                  xgboost_version=xgb.__version__, build_info=xgb.build_info(), params=params,
                  weighted=args.weighted, repeats=args.repeats, cases=[],
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  working_tree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
                  timing="Public model.eval including metric text; zero-tree models with identical fixed base margins; warmed runtime; first evaluation and cached repeats separate; excludes data generation, DMatrix construction and training",
                  memory_note="Process lifetime peak RSS; swap is system-wide, not attributable solely to this benchmark")
    if sys.platform == "darwin":
        report["hardware"] = {key: subprocess.check_output(["sysctl", "-n", key], text=True).strip()
                              for key in ("machdep.cpu.brand_string", "hw.memsize", "hw.ncpu")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    try:
        for n in args.rows:
            for query_size in args.query_sizes:
                print(f"Preparing {n:,} rows, queries of {query_size:,}", flush=True)
                rng = np.random.default_rng(839)
                labels = rng.integers(0, label_count, n).astype(np.float32)
                margin = rng.standard_normal(n, dtype=np.float32)
                groups = np.diff(np.r_[np.arange(0, n, query_size), n]).astype(np.uint32)
                weights = rng.uniform(.1, 3, len(groups)).astype(np.float32) if args.weighted else None
                start = time.perf_counter()
                data = xgb.DMatrix(np.ones((n, 1), dtype=np.float32), label=labels,
                                   base_margin=margin, group=groups, weight=weights, nthread=args.threads)
                case = dict(rows=n, query_size=query_size, groups=len(groups),
                            cpu_pair_comparisons=(int(sum(int(g) * (int(g) - 1) // 2 for g in groups if g >= 3))
                                                  if args.metric == "auc" else None),
                            dmatrix_seconds=time.perf_counter() - start,
                            labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                            margins_sha256=hashlib.sha256(margin.tobytes()).hexdigest(),
                            first_evaluation={}, runs={"cpu": [], "mps": []})
                report["cases"].append(case)
                models = {d: xgb.train(params | {"device": d}, data, 0) for d in ["cpu", "mps"]}
                for device in ["cpu", "mps"]:
                    run = evaluate(models[device], data, device, args.metric)
                    case["first_evaluation"][device] = run
                    save(args.output, report)
                    validate(run, case["first_evaluation"]["cpu"]["score"])
                    print(f"  {device} first evaluation: {run['seconds']:.6f}s", flush=True)
                for repeat in range(args.repeats):
                    for device in (["cpu", "mps"] if repeat % 2 == 0 else ["mps", "cpu"]):
                        run = evaluate(models[device], data, device, args.metric)
                        case["runs"][device].append(run)
                        save(args.output, report)
                        validate(run, case["first_evaluation"]["cpu"]["score"])
                        print(f"  {device} repeat {repeat + 1}/{args.repeats}: {run['seconds']:.6f}s", flush=True)
                cpu = statistics.median(run["seconds"] for run in case["runs"]["cpu"])
                mps = statistics.median(run["seconds"] for run in case["runs"]["mps"])
                case.update(cpu_median=cpu, mps_median=mps, speedup=cpu / mps,
                            max_score_difference=max(abs(run["score"] - case["first_evaluation"]["cpu"]["score"])
                                                     for run in case["runs"]["mps"]))
                save(args.output, report)
                print(f"  Metric CPU/Metal ratio: {cpu / mps:.3f}x; saved {args.output}", flush=True)
                del models, data, labels, margin, weights
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()
