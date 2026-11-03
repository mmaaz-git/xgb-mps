"""Compare fresh CPU and Metal training jobs; write measured timings as JSON."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xgboost/python-package"))
import numpy as np
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; inspect it or choose a new output name")
    assert xgb.metal_info()["available"], "A visible Metal GPU is required"
    p = dict(tree_method="hist", objective="reg:squarederror", max_depth=6,
             max_bin=256, nthread=args.threads, seed=7)
    # Compile the shader and initialize the runtime before timing either device.
    warm = xgb.DMatrix(np.arange(64, dtype=np.float32).reshape(32, 2), label=np.arange(32))
    for device in ["cpu", "mps"]:
        xgb.train(p | {"device": device}, warm, num_boost_round=1)
    report = dict(platform=platform.platform(), cpu_threads=args.threads, features=args.features,
                  rounds=args.rounds, repeats=args.repeats, params=p, cases=[],
                  timing="DMatrix construction + training with quantization; warmed Metal runtime; data generation excluded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n in args.rows:
        print(f"Generating {n:,} rows x {args.features} features", flush=True)
        rng = np.random.default_rng(17)
        x = rng.normal(size=(n, args.features)).astype(np.float32)
        y = (x[:, 0] * 2 - x[:, 1] + x[:, 2] ** 2 + rng.normal(scale=0.1, size=n)).astype(np.float32)
        timings = {"cpu": [], "mps": []}
        preds = {}
        dispatch_statistics = {"cpu": [], "mps": []}
        for repeat in range(args.repeats):
            order = ["cpu", "mps"] if repeat % 2 == 0 else ["mps", "cpu"]
            for device in order:
                before = xgb.metal_info()
                start = time.perf_counter()
                d = xgb.DMatrix(x, label=y, nthread=args.threads)
                model = xgb.train(p | {"device": device}, d, num_boost_round=args.rounds)
                elapsed = time.perf_counter() - start
                after = xgb.metal_info()
                stats = {k: after[k] - before[k] for k in after if k not in {"built", "available"}}
                dispatch_statistics[device].append(stats)
                completed = after["histogram_dispatches"] - before["histogram_dispatches"]
                gpu_seconds = (after.get("gpu_time_ns", 0) - before.get("gpu_time_ns", 0)) / 1e9
                assert (completed > 0) == (device == "mps")
                timings[device].append(elapsed)
                preds[device] = model.predict(d)
                print(f"  {device} repeat {repeat+1}/{args.repeats}: {elapsed:.3f}s, {completed} GPU dispatches, {gpu_seconds:.3f}s GPU time", flush=True)
                del d, model
        cpu = statistics.median(timings["cpu"])
        mps = statistics.median(timings["mps"])
        np.testing.assert_allclose(preds["mps"], preds["cpu"], rtol=2e-5, atol=2e-5)
        case = dict(rows=n, seconds=timings, dispatch_statistics=dispatch_statistics, cpu_median=cpu, mps_median=mps,
                    speedup=cpu/mps, max_prediction_difference=float(np.max(np.abs(preds["cpu"]-preds["mps"]))),
                    cpu_rmse=float(np.sqrt(np.mean((preds["cpu"]-y)**2))),
                    mps_rmse=float(np.sqrt(np.mean((preds["mps"]-y)**2))))
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  CPU/Metal speedup: {cpu/mps:.3f}x; wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
