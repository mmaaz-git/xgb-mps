"""Repeated uncached prediction on identical CPU-trained models and inputs."""
import argparse
import json
import platform
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xgboost/python-package"))
import numpy as np
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1000, 100_000, 1_000_000])
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; inspect it or choose another path")
    rng = np.random.default_rng(91)
    train = rng.normal(size=(10_000, args.features)).astype(np.float32)
    label = train[:, 0] * 2 - train[:, 1] + train[:, 2] ** 2 + rng.normal(size=len(train)) * 0.1
    print(f"Training shared CPU model: {args.trees} trees, depth 6", flush=True)
    cpu = xgb.train(dict(nthread=args.threads, max_depth=6, tree_method="hist"),
                    xgb.DMatrix(train, label=label), args.trees)
    gpu = xgb.Booster(model_file=cpu.save_raw())
    gpu.set_param(dict(device="mps", nthread=args.threads))
    for model in [cpu, gpu]:
        model.inplace_predict(train[:10])
    report = dict(platform=platform.platform(), params=vars(args) | {"output": str(args.output)},
                  timing="Uncached inplace_predict; includes host input staging and GPU transfers; excludes training and data generation; warmed runtime",
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        print(f"Generating {rows:,} prediction rows", flush=True)
        x = rng.normal(size=(rows, args.features)).astype(np.float32)
        timings = {"cpu": [], "mps": []}
        gpu_times = []
        predictions = {}
        for repeat in range(args.repeats):
            for device in (["cpu", "mps"] if repeat % 2 == 0 else ["mps", "cpu"]):
                model = cpu if device == "cpu" else gpu
                before = xgb.metal_info()
                start = time.perf_counter()
                predictions[device] = model.inplace_predict(x)
                elapsed = time.perf_counter() - start
                after = xgb.metal_info()
                count = after["prediction_dispatches"] - before["prediction_dispatches"]
                assert (count > 0) == (device == "mps")
                timings[device].append(elapsed)
                if device == "mps":
                    gpu_times.append((after["prediction_gpu_time_ns"] - before["prediction_gpu_time_ns"]) / 1e9)
                print(f"  {device} {repeat+1}/{args.repeats}: {elapsed:.4f}s; {count} prediction dispatches", flush=True)
        np.testing.assert_allclose(predictions["mps"], predictions["cpu"], rtol=2e-6, atol=2e-6)
        speedup = statistics.median(timings["cpu"]) / statistics.median(timings["mps"])
        report["cases"].append(dict(rows=rows, seconds=timings, gpu_seconds=gpu_times,
                                     speedup=speedup, max_prediction_difference=float(np.max(np.abs(predictions["cpu"] - predictions["mps"])))))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  CPU/Metal: {speedup:.3f}x; wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
