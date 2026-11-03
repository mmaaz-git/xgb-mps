"""Ranking metric timings on identical predictions; includes all staging and synchronization."""
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
    parser.add_argument("--query-sizes", type=int, nargs="+", default=[32, 256, 4096])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; inspect it or choose a new output name")
    assert xgb.metal_info()["available"]
    report = dict(platform=platform.platform(), cpu_threads=args.threads, repeats=args.repeats,
                  metric=args.metric, cases=[],
                  timing="model.eval on fixed margins and labels, warmed prediction cache/runtime; "
                         "includes metric staging, sorting, GPU work and synchronization; "
                         "excludes DMatrix/model construction, data generation and training")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        for query_size in args.query_sizes:
            print(f"Generating {rows:,} rows, queries of {query_size}, metric {args.metric}", flush=True)
            rng = np.random.default_rng(371)
            scores = rng.normal(size=rows).astype(np.float32)
            labels = rng.integers(0, 2, size=rows).astype(np.float32)
            groups = np.diff(np.r_[np.arange(0, rows, query_size), rows]).astype(np.uint32)
            d = xgb.DMatrix(np.zeros((rows, 1), dtype=np.float32), label=labels,
                             base_margin=scores, group=groups, nthread=args.threads)
            params = dict(objective="reg:squarederror", eval_metric=args.metric,
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
                    wall_start = time.time()
                    start = time.perf_counter()
                    values[device] = float(models[device].eval(d).rsplit(":", 1)[1])
                    seconds = time.perf_counter() - start
                    if abs(time.time() - wall_start - seconds) > 1:
                        raise RuntimeError("Benchmark crossed sleep or a clock adjustment")
                    after = xgb.metal_info()
                    delta = {k: after[k] - before[k] for k in after if k not in {"built", "available"}}
                    assert delta["group_sort_dispatches"] == int(device == "mps")
                    expected_score = device == "mps" and args.metric.split("@")[0].rstrip("-") in {"ndcg", "pre"}
                    assert delta["ranking_metric_dispatches"] == int(expected_score)
                    timings[device].append(seconds)
                    counters[device].append(delta)
                    print(f"  {device} {repeat+1}/{args.repeats}: {seconds:.6f} s", flush=True)
            np.testing.assert_allclose(values["mps"], values["cpu"], rtol=2e-12, atol=2e-12)
            cpu, mps = (statistics.median(timings[k]) for k in ["cpu", "mps"])
            case = dict(rows=rows, query_size=query_size, groups=len(groups), seconds=timings, dispatch_statistics=counters,
                        first_eval_seconds=first_eval, cpu_median=cpu, mps_median=mps,
                        speedup=cpu/mps, values=values, absolute_difference=abs(values["cpu"]-values["mps"]))
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"  CPU/Metal {cpu/mps:.3f}x, wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
