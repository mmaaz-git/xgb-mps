"""Warm metric-only ROC-AUC timings on identical cached predictions and labels."""
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
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--targets", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--metric", choices=["auc", "aucpr"], default="auc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; inspect it or choose a new output name")
    assert xgb.metal_info()["available"]
    report = dict(platform=platform.platform(), cpu_threads=args.threads, repeats=args.repeats,
                  weighted=args.weighted, metric=args.metric, cases=[],
                  timing="model.eval on fixed margins and labels, warmed prediction cache/runtime; "
                         "includes metric staging, sorting, GPU work and synchronization; "
                         "excludes DMatrix/model construction, data generation and training")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        for targets in args.targets:
            print(f"Generating {rows:,} rows x {targets} AUC targets", flush=True)
            rng = np.random.default_rng(371)
            scores = rng.normal(size=(rows, targets)).astype(np.float32)
            labels = rng.integers(0, 2, size=(rows, targets)).astype(np.float32)
            weights = rng.uniform(0, 3, rows).astype(np.float32) if args.weighted else None
            d = xgb.DMatrix(np.zeros((rows, 1), dtype=np.float32), label=labels,
                             base_margin=scores, weight=weights, nthread=args.threads)
            params = dict(objective="reg:squarederror", eval_metric=args.metric, num_target=targets,
                          nthread=args.threads, base_score=.5)
            models = {device: xgb.train(params | {"device": device}, d, 0)
                      for device in ["cpu", "mps"]}
            first_eval = {}
            values = {}
            for device, model in models.items():
                start = time.perf_counter()
                values[device] = float(model.eval(d).rsplit(":", 1)[1])
                first_eval[device] = time.perf_counter() - start
            timings = {device: [] for device in models}
            counters = {device: [] for device in models}
            for repeat in range(args.repeats):
                for device in (["cpu", "mps"] if repeat % 2 == 0 else ["mps", "cpu"]):
                    before = xgb.metal_info()
                    start = time.perf_counter()
                    values[device] = float(models[device].eval(d).rsplit(":", 1)[1])
                    seconds = time.perf_counter() - start
                    after = xgb.metal_info()
                    delta = {k: after[k] - before[k] for k in after if k not in {"built", "available"}}
                    assert delta["roc_auc_dispatches" if args.metric == "auc" else "pr_auc_dispatches"] == int(device == "mps")
                    timings[device].append(seconds)
                    counters[device].append(delta)
                    print(f"  {device} {repeat+1}/{args.repeats}: {seconds:.6f} s", flush=True)
            np.testing.assert_allclose(values["mps"], values["cpu"], rtol=2e-12, atol=2e-12)
            cpu, mps = (statistics.median(timings[k]) for k in ["cpu", "mps"])
            case = dict(rows=rows, targets=targets, seconds=timings, dispatch_statistics=counters,
                        first_eval_seconds=first_eval, cpu_median=cpu, mps_median=mps,
                        speedup=cpu/mps, values=values, absolute_difference=abs(values["cpu"]-values["mps"]))
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"  CPU/Metal {cpu/mps:.3f}x, wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
