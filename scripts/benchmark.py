"""Compare fresh CPU and Metal training jobs; write measured timings as JSON."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xgboost/python-package"))
import numpy as np
import xgboost as xgb


def memory_snapshot():
    # Peak RSS is process-wide and includes preceding runs; swap is system-wide.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": int(rss * (1 if sys.platform == "darwin" else 1024))}
    if sys.platform == "darwin":
        result["system_swap"] = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                                                 capture_output=True, text=True, check=True).stdout.strip()
    return result


class Progress(xgb.callback.TrainingCallback):
    def __init__(self, device, rounds, every, start):
        self.device, self.rounds, self.every, self.start = device, rounds, every, start

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.every == 0 or epoch + 1 == self.rounds:
            print(f"    {self.device} round {epoch+1}/{self.rounds}, "
                  f"{time.perf_counter()-self.start:.1f}s elapsed", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--objective", choices=["reg:squarederror", "reg:absoluteerror", "reg:quantileerror", "binary:logistic", "rank:pairwise", "rank:ndcg", "rank:map"],
                        default="reg:squarederror")
    parser.add_argument("--quantile-levels", type=float, nargs="+", default=[0.5])
    parser.add_argument("--eval-metric", help="Evaluate this training metric after every round")
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--sampling-method", choices=["uniform", "gradient_based"], default="uniform")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--query-size", type=int, default=128, help="Rows per query for ranking objectives")
    parser.add_argument("--rank-pairs", type=int, default=32)
    parser.add_argument("--unbiased", action="store_true", help="Enable ranking position-bias correction")
    parser.add_argument("--rank-method", choices=["topk", "mean"], default="topk")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--float32-generation", action="store_true",
                        help="Generate large feature arrays directly as FP32, avoiding FP64 temporaries")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; inspect it or choose a new output name")
    if args.query_size < 1 or args.rank_pairs < 1:
        parser.error("query-size and rank-pairs must be positive")
    is_ranking = args.objective.startswith("rank:")
    assert xgb.metal_info()["available"], "A visible Metal GPU is required"
    p = dict(tree_method="hist", objective=args.objective, max_depth=args.max_depth,
             max_bin=256, nthread=args.threads, seed=7, subsample=args.subsample,
             sampling_method=args.sampling_method)
    if is_ranking:
        p.update(lambdarank_pair_method=args.rank_method, lambdarank_num_pair_per_sample=args.rank_pairs,
                 lambdarank_unbiased=args.unbiased)
    if args.eval_metric:
        p["eval_metric"] = args.eval_metric
    if args.objective == "reg:quantileerror":
        p["quantile_alpha"] = args.quantile_levels
    # Compile the shader and initialize the runtime before timing either device.
    warm_labels = np.arange(32) % 2 if args.objective in {"binary:logistic", "rank:map"} else np.arange(32)
    warm = xgb.DMatrix(np.arange(64, dtype=np.float32).reshape(32, 2), label=warm_labels,
                       weight=np.ones(1 if is_ranking else 32) if args.weighted else None,
                       group=[32] if is_ranking else None)
    for device in ["cpu", "mps"]:
        xgb.train(p | {"device": device}, warm, num_boost_round=1)
    report = dict(status="in_progress", platform=platform.platform(), cpu_threads=args.threads, features=args.features,
                  rounds=args.rounds, repeats=args.repeats, weighted=args.weighted, params=p, cases=[],
                  xgboost_version=xgb.__version__, numpy_version=np.__version__, build_info=xgb.build_info(),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  working_tree=subprocess.check_output(["git", "status", "--short"], text=True).strip(),
                  float32_generation=args.float32_generation, progress_every=args.progress_every,
                  query_size=args.query_size if is_ranking else None,
                  memory_note="RSS is the process lifetime peak; swap is system-wide, not isolated to this benchmark",
                  timing="DMatrix construction + training with quantization; warmed Metal runtime; data generation excluded")
    if sys.platform == "darwin":
        report["hardware"] = {
            key: subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            for key in ("machdep.cpu.brand_string", "hw.memsize", "hw.ncpu")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n in args.rows:
        print(f"Generating {n:,} rows x {args.features} features", flush=True)
        rng = np.random.default_rng(17)
        x = (rng.standard_normal(size=(n, args.features), dtype=np.float32) if args.float32_generation
             else rng.normal(size=(n, args.features)).astype(np.float32))
        y = (x[:, 0] * 2 - x[:, 1] + x[:, 2] ** 2 + rng.normal(scale=0.1, size=n)).astype(np.float32)
        if args.objective == "binary:logistic":
            y = (rng.random(n) < 1 / (1 + np.exp(-.5 * (y - 1)))).astype(np.float32)
        groups = None
        if is_ranking:
            groups = np.diff(np.r_[np.arange(0, n, args.query_size), n]).astype(np.uint32)
            y = ((y > 1).astype(np.float32) if args.objective == "rank:map" else
                 np.digitize(y, [-2, -1, 0, 1, 2, 4, 6]).astype(np.float32))
        weights = rng.uniform(0.1, 3, len(groups) if is_ranking else n).astype(np.float32) if args.weighted else None
        timings = {"cpu": [], "mps": []}
        preds = {}
        metric_histories = {"cpu": [], "mps": []}
        memory = {"cpu": [], "mps": []}
        dispatch_statistics = {"cpu": [], "mps": []}
        clock_checks = {"cpu": [], "mps": []}
        for repeat in range(args.repeats):
            order = ["cpu", "mps"] if repeat % 2 == 0 else ["mps", "cpu"]
            for device in order:
                before = xgb.metal_info()
                memory_before = memory_snapshot()
                wall_start = time.time()
                start = time.perf_counter()
                d = xgb.DMatrix(x, label=y, weight=weights, group=groups, nthread=args.threads)
                history = {}
                model = xgb.train(p | {"device": device}, d, num_boost_round=args.rounds,
                                  evals=[(d, "train")] if args.eval_metric else [],
                                  evals_result=history, verbose_eval=False,
                                  callbacks=[Progress(device, args.rounds, args.progress_every, start)]
                                  if args.progress_every else None)
                elapsed = time.perf_counter() - start
                wall_elapsed = time.time() - wall_start
                clock_checks[device].append(dict(
                    started_utc=datetime.fromtimestamp(wall_start, timezone.utc).isoformat(),
                    wall_seconds=wall_elapsed, monotonic_seconds=elapsed))
                # macOS's monotonic clock can stop during sleep. Do not silently
                # publish a measurement spanning sleep or a wall-clock adjustment.
                if abs(wall_elapsed - elapsed) > 1:
                    report["status"] = "invalid_clock_discontinuity"
                    report["in_progress"] = dict(rows=n, seconds=timings, clock_checks=clock_checks)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    raise RuntimeError("Benchmark crossed sleep or a clock adjustment; timings invalid")
                memory[device].append({"before": memory_before, "after": memory_snapshot()})
                after = xgb.metal_info()
                stats = {k: after[k] - before[k] for k in after if k not in {"built", "available"}}
                dispatch_statistics[device].append(stats)
                metric_histories[device].append(history)
                if args.eval_metric == "auc":
                    assert stats["roc_auc_dispatches"] == (args.rounds if device == "mps" else 0)
                if args.sampling_method == "gradient_based":
                    targets = len(args.quantile_levels) if args.objective == "reg:quantileerror" else 1
                    assert stats["gradient_sampling_dispatches"] == (
                        args.rounds * targets if device == "mps" and args.subsample < 1 else 0)
                if is_ranking:
                    assert stats["ranking_gradient_dispatches"] == (args.rounds + 1 if device == "mps" else 0)
                    assert stats["ranking_bias_dispatches"] == (args.rounds + 1 if device == "mps" and args.unbiased else 0)
                completed = after["histogram_dispatches"] - before["histogram_dispatches"]
                gpu_seconds = (after.get("gpu_time_ns", 0) - before.get("gpu_time_ns", 0)) / 1e9
                assert (completed > 0) == (device == "mps")
                timings[device].append(elapsed)
                preds[device] = model.predict(d)
                print(f"  {device} repeat {repeat+1}/{args.repeats}: {elapsed:.3f}s, {completed} GPU dispatches, {gpu_seconds:.3f}s GPU time", flush=True)
                report["in_progress"] = dict(rows=n, seconds=timings, memory=memory,
                                              dispatch_statistics=dispatch_statistics, clock_checks=clock_checks,
                                              metric_histories=metric_histories)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                del d, model
        cpu = statistics.median(timings["cpu"])
        mps = statistics.median(timings["mps"])
        try:
            np.testing.assert_allclose(preds["mps"], preds["cpu"], rtol=2e-5, atol=2e-5)
        except AssertionError:
            report["status"] = "prediction_equivalence_failed"
            difference = np.abs(preds["cpu"] - preds["mps"])
            report["in_progress"]["max_prediction_difference"] = float(np.max(difference))
            report["in_progress"]["prediction_rmse_difference"] = float(np.sqrt(np.mean(difference.astype(np.float64) ** 2)))
            report["in_progress"]["prediction_mismatch_count"] = int(np.count_nonzero(difference > 2e-5 + 2e-5 * np.abs(preds["cpu"])))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise
        if args.eval_metric:
            np.testing.assert_allclose(metric_histories["mps"][-1]["train"][args.eval_metric],
                                       metric_histories["cpu"][-1]["train"][args.eval_metric],
                                       rtol=2e-5, atol=2e-5)
        truth = y[:, None] if preds["cpu"].ndim == 2 else y
        case = dict(rows=n, seconds=timings, memory=memory, dispatch_statistics=dispatch_statistics,
                    clock_checks=clock_checks, cpu_median=cpu, mps_median=mps,
                    speedup=cpu/mps, max_prediction_difference=float(np.max(np.abs(preds["cpu"]-preds["mps"]))),
                    cpu_rmse=float(np.sqrt(np.mean((preds["cpu"]-truth)**2))),
                    mps_rmse=float(np.sqrt(np.mean((preds["mps"]-truth)**2))))
        if args.eval_metric:
            case["metric_histories"] = metric_histories
        report.pop("in_progress", None)
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  CPU/Metal speedup: {cpu/mps:.3f}x; wrote {args.output}", flush=True)

    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
