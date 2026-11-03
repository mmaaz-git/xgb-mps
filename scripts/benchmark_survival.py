"""Measure AFT training with evaluation, including GPU work and CPU corrections."""
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


def matrix(x, y, weights, threads):
    lo, hi = y.copy(), y.copy()
    lo[1::4] = 0
    hi[2::4] = np.inf
    lo[3::4] *= .7
    hi[3::4] *= 1.4
    return xgb.DMatrix(x, label_lower_bound=lo, label_upper_bound=hi,
                       weight=weights, nthread=threads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--distributions", nargs="+", choices=["normal", "logistic", "extreme"],
                        default=["normal", "logistic", "extreme"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; inspect it or choose a new output name")
    assert xgb.metal_info()["available"], "A visible GPU is required"
    params = dict(objective="survival:aft", max_depth=6, max_bin=256,
                   tree_method="hist", nthread=args.threads, seed=7,
                   eval_metric=["aft-nloglik", "interval-regression-accuracy"])
    warm = matrix(np.zeros((32, 2)), np.ones(32), None, args.threads)
    for device in ["cpu", "mps"]:
        xgb.train(params | {"device": device}, warm, 1, evals=[(warm, "train")], verbose_eval=False)
    report = dict(platform=platform.platform(), cpu_threads=args.threads, features=32,
                   rounds=args.rounds, repeats=args.repeats, weighted=True, params=params, cases=[],
                   timing="Fresh DMatrix + training + two metrics each round; warmed runtime; generation excluded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        rng = np.random.default_rng(193)
        x = rng.normal(size=(rows, 32)).astype(np.float32)
        y = np.exp(.3*x[:, 0] - .2*x[:, 1] + rng.normal(scale=.5, size=rows)).astype(np.float32)
        weights = rng.uniform(.1, 3, rows).astype(np.float32)
        for distribution in args.distributions:
            print(f"Measuring {distribution}, {rows:,} rows, 32 features", flush=True)
            timings = {"cpu": [], "mps": []}
            counters = {"cpu": [], "mps": []}
            predictions, metrics = {}, {}
            for repeat in range(args.repeats):
                for device in (["cpu", "mps"] if repeat % 2 == 0 else ["mps", "cpu"]):
                    before = xgb.metal_info()
                    start = time.perf_counter()
                    d = matrix(x, y, weights, args.threads)
                    history = {}
                    model = xgb.train(params | dict(device=device, aft_loss_distribution=distribution),
                                       d, args.rounds, evals=[(d, "train")], evals_result=history, verbose_eval=False)
                    elapsed = time.perf_counter() - start
                    after = xgb.metal_info()
                    delta = {key: after[key] - before[key] for key in after if key not in {"built", "available"}}
                    assert delta["survival_gradient_dispatches"] == (args.rounds if device == "mps" else 0)
                    assert delta["survival_metric_dispatches"] == (2*args.rounds if device == "mps" else 0)
                    timings[device].append(elapsed)
                    counters[device].append(delta)
                    predictions[device] = model.predict(d)
                    metrics[device] = history["train"]
                    print(f"  {device} {repeat+1}/{args.repeats}: {elapsed:.3f}s; CPU corrections {delta['survival_cpu_corrections']:,}", flush=True)
                    del model, d
            np.testing.assert_allclose(predictions["mps"], predictions["cpu"], rtol=2e-5, atol=2e-5)
            for name in metrics["cpu"]:
                np.testing.assert_allclose(metrics["mps"][name], metrics["cpu"][name], rtol=2e-6, atol=2e-6)
            cpu, mps = [statistics.median(timings[d]) for d in ["cpu", "mps"]]
            report["cases"].append(dict(rows=rows, distribution=distribution, seconds=timings,
                 dispatch_statistics=counters, cpu_median=cpu, mps_median=mps, speedup=cpu/mps,
                 max_prediction_difference=float(np.max(np.abs(predictions["mps"]-predictions["cpu"]))),
                 final_metrics={d: {k: v[-1] for k, v in m.items()} for d, m in metrics.items()}))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"  CPU/Metal {cpu/mps:.3f}x; wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
