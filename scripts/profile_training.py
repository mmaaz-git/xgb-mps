"""Profile large training jobs separately from correctness tests.

Each CPU/MPS repeat runs in a fresh process. HIGGS and Covertype use the prepared
cache from prepare_published_data.py; wide and categorical generate deterministic data. GPU diagnostic
durations can overlap and must not be added to estimate CPU or total wall time.
--cut-device is a correctness diagnostic: it shares quantile cuts across devices
and moves their construction into setup, so its training timings are not
comparable with ordinary DMatrix runs.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power():
    if sys.platform == "darwin":
        return subprocess.check_output(["pmset", "-g", "batt"], text=True).strip()
    return "unavailable"


def worker(args):
    sys.path.insert(0, str(args.package_dir.resolve()))
    import numpy as np
    import xgboost as xgb
    from benchmark import memory_snapshot

    path = args.output / f"{args.worker}-{args.repeat_index}.json"
    if path.exists():
        raise FileExistsError(path)
    package = Path(xgb.__file__).resolve()
    assert package.is_relative_to(args.package_dir.resolve()), package
    library = Path(xgb.core._LIB._name).resolve()
    params = dict(
        device=args.worker, tree_method="hist", max_depth=6, max_bin=256,
        nthread=args.threads, seed=0, base_score=0.5, eta=0.3,
        objective="binary:logitraw" if args.case == "higgs" else "binary:logistic",
        eval_metric="error" if args.case == "higgs" else "logloss", verbosity=3,
    )
    if args.case == "covtype":
        params.update(objective="multi:softprob", num_class=7, eval_metric="mlogloss")
    if args.worker == "mps":
        assert xgb.metal_info()["available"]
    warm = xgb.DMatrix(np.arange(64, dtype=np.float32).reshape(32, 2),
                       label=np.arange(32) % (7 if args.case == "covtype" else 2), nthread=args.threads)
    xgb.train(params | {"verbosity": 0}, warm, num_boost_round=1)
    del warm

    rows = args.rows or (10_500_000 if args.case == "higgs" else 1_000_000)
    load_start = time.perf_counter()
    if args.case == "higgs":
        if rows > 10_500_000:
            raise ValueError("HIGGS train rows must not overlap the last 500,000 test rows")
        x = np.load(args.cache / "higgs/X.npy", mmap_mode="r")
        y = np.load(args.cache / "higgs/y.npy", mmap_mode="r")
        train_x, train_y = x[:rows], y[:rows]
        test_x, test_y = x[10_500_000:], y[10_500_000:]
        dataset = json.loads((args.cache / "higgs/manifest.json").read_text())
    elif args.case == "covtype":
        from sklearn.model_selection import train_test_split
        x = np.load(args.cache / "covtype/X.npy", mmap_mode="r")
        y = np.load(args.cache / "covtype/y.npy", mmap_mode="r")
        train_indices, test_indices = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=1701, stratify=y)
        if args.rows is not None:
            if args.rows > len(train_indices):
                raise ValueError("Covertype rows exceed the 80% training split")
            train_indices = train_indices[:args.rows]
        train_x, train_y = x[train_indices], y[train_indices] - 1
        test_x, test_y = x[test_indices], y[test_indices] - 1
        dataset = json.loads((args.cache / "covtype/manifest.json").read_text()) | {
            "split": "Stratified 80/20 train/test, random_state=1701; optional training prefix",
            "label_transform": "Original classes 1..7 mapped to 0..6",
            "train_indices_sha256": hashlib.sha256(train_indices.astype("<i8").tobytes()).hexdigest(),
            "test_indices_sha256": hashlib.sha256(test_indices.astype("<i8").tobytes()).hexdigest(),
        }
    else:
        rng = np.random.default_rng(1701)
        test_rows = min(100_000, max(1000, rows // 10))
        if args.case == "categorical":
            x = rng.integers(0, args.categories, size=(rows + test_rows, args.features)).astype(np.float32)
            features = np.linspace(0, args.features - 1,
                                   args.informative_features or min(16, args.features), dtype=int)
            weights = rng.standard_normal((len(features), args.categories), dtype=np.float32)
            score = np.zeros(len(x), dtype=np.float32)
            for i, feature in enumerate(features):
                score += weights[i, x[:, feature].astype(int)]
            score *= np.float32(2 / np.sqrt(len(features)))
            signal = {"formula": "2/sqrt(k) * sum(normal_lookup[i, x[f_i]])",
                      "features": features.tolist(), "categories": args.categories,
                      "weights_sha256": hashlib.sha256(memoryview(weights)).hexdigest()}
        else:
            x = rng.standard_normal((rows + test_rows, args.features), dtype=np.float32)
            if args.informative_features is None:
                score = 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] ** 2 - 0.5
                signal = {"formula": "2*x[0] - x[1] + 0.5*x[2]**2 - 0.5",
                          "features": [0, 1, 2]}
            else:
                features = np.linspace(0, args.features - 1, args.informative_features, dtype=int)
                score = np.zeros(len(x), dtype=np.float32)
                for i, feature in enumerate(features):
                    value = x[:, feature]
                    term = (value * value - 1) / np.sqrt(np.float32(2)) if i % 4 == 0 else value
                    score += term if i % 2 == 0 else -term
                score *= np.float32(2 / np.sqrt(len(features)))
                signal = {"formula": "2/sqrt(k) * sum((-1)**i * t_i); t_i=(x[f_i]**2-1)/sqrt(2) for i%4==0, otherwise x[f_i]",
                          "features": features.tolist()}
        y = (rng.random(len(x)) < 1 / (1 + np.exp(-score))).astype(np.float32)
        if args.missing_fraction:
            x[rng.random(x.shape, dtype=np.float32) < args.missing_fraction] = np.nan
        train_x, train_y = x[:rows], y[:rows]
        test_x, test_y = x[rows:], y[rows:]
        dataset = {"generator": ("numpy.default_rng(1701); uniform category IDs; random per-category effects; noisy logistic target"
                                 if args.case == "categorical" else
                                 "numpy.default_rng(1701); FP32 normal features; noisy logistic target"),
                   "numpy_version": np.__version__, "signal": signal,
                   "features_sha256": hashlib.sha256(memoryview(x)).hexdigest(),
                   "labels_sha256": hashlib.sha256(memoryview(y)).hexdigest()}
        if args.missing_fraction:
            dataset["missing_fraction"] = args.missing_fraction
    loading_seconds = time.perf_counter() - load_start
    print(f"PROFILE {args.worker}: train={train_x.shape}, test={test_x.shape}", flush=True)
    setup = time.perf_counter()
    matrix_options = ({"enable_categorical": True, "feature_types": ["c"] * args.features}
                      if args.case == "categorical" else {})
    cut_record = None
    if args.cut_device:
        options = dict(label=train_y, nthread=args.threads, max_bin=params["max_bin"], **matrix_options)
        reference = xgb.QuantileDMatrix(train_x, **options,
            **({"device": "mps"} if args.cut_device == "mps" else {}))
        train = xgb.QuantileDMatrix(train_x, ref=reference, **options,
            **({"device": "mps"} if args.worker == "mps" else {}))
        ptr, cuts = train.get_quantile_cut()
        for actual, expected in zip((ptr, cuts), reference.get_quantile_cut()):
            np.testing.assert_array_equal(actual, expected)
        np.savez(path.with_suffix(".cuts.npz"), ptr=ptr, cuts=cuts)
        cut_record = {"device": args.cut_device, "count": len(cuts),
                      "ptr_sha256": hashlib.sha256(ptr.tobytes()).hexdigest(),
                      "cuts_sha256": hashlib.sha256(cuts.tobytes()).hexdigest()}
        del reference
    else:
        train = xgb.DMatrix(train_x, label=train_y, nthread=args.threads, **matrix_options)
    test = xgb.DMatrix(test_x, label=test_y, nthread=args.threads, **matrix_options)
    setup_seconds = time.perf_counter() - setup
    segments = []
    before = xgb.metal_info()
    memory_before, power_before = memory_snapshot(), power()

    class Progress(xgb.callback.TrainingCallback):
        def __init__(self):
            self.previous, self.last = before, time.perf_counter()

        def after_iteration(self, model, epoch, evals_log):
            if epoch == 0 or (epoch + 1) % args.profile_every == 0 or epoch + 1 == args.rounds:
                now, current = time.perf_counter(), xgb.metal_info()
                counters = {k: current[k] - self.previous[k] for k in current
                            if k not in {"built", "available"}}
                segments.append({"through_round": epoch + 1, "wall_seconds": now - self.last,
                                 "counters": counters})
                print(f"PROFILE {args.worker}: round {epoch + 1}/{args.rounds}, "
                      f"segment {now - self.last:.3f}s", flush=True)
                self.previous, self.last = current, now
                write_json(path.with_suffix(".progress.json"), {"segments": segments})
            return False

    progress = Progress()
    start, wall_start = time.perf_counter(), time.time()
    history = {}
    model = xgb.train(params, train, num_boost_round=args.rounds, evals=[(test, "test")],
                      evals_result=history, verbose_eval=False, callbacks=[progress])
    training_seconds = time.perf_counter() - start
    wall_seconds = time.time() - wall_start
    after = xgb.metal_info()
    counters = {k: after[k] - before[k] for k in after if k not in {"built", "available"}}
    assert (counters["histogram_dispatches"] > 0) == (args.worker == "mps")
    clock_valid = abs(wall_seconds - training_seconds) <= 1
    power_after = power()
    prediction_start = time.perf_counter()
    margins = model.predict(test, output_margin=True)
    prediction_seconds = time.perf_counter() - prediction_start
    assert np.isfinite(margins).all()
    np.save(path.with_suffix(".margins.npy"), margins)
    model.save_model(path.with_suffix(".ubj"))
    if args.worker == "mps":
        # Reload to force CPU traversal with an empty prediction cache.
        cpu_model = xgb.Booster(model_file=path.with_suffix(".ubj"))
        cpu_model.set_param({"device": "cpu", "nthread": args.threads})
        np.testing.assert_array_equal(cpu_model.predict(test, output_margin=True), margins)
    z = margins.astype(np.float64)
    if margins.ndim == 1:
        classes = margins > 0
        loss = np.logaddexp(0, z) - test_y * z
    else:
        classes = np.argmax(margins, axis=1)
        z -= np.max(z, axis=1, keepdims=True)
        loss = np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(test_y)), test_y.astype(int)]
    report = {
        "status": "complete" if clock_valid else "invalid_clock_discontinuity",
        "started_utc": datetime.fromtimestamp(wall_start, timezone.utc).isoformat(),
        "case": args.case, "device": args.worker, "params": params, "rounds": args.rounds,
        "common_cut_diagnostic": cut_record,
        "train_shape": list(train_x.shape), "test_shape": list(test_x.shape), "dataset": dataset,
        "package": str(package), "library": str(library), "library_sha256": digest(library),
        "loading_seconds": loading_seconds, "dmatrix_seconds": setup_seconds,
        "training_seconds": training_seconds, "wall_seconds": wall_seconds,
        "prediction_seconds": prediction_seconds, "metric_history": history,
        "same_model_cpu_inference_exact": True if args.worker == "mps" else None,
        "test_accuracy": float(np.mean(classes == test_y)),
        "test_logloss": float(np.mean(loss)),
        "memory": {"before": memory_before, "after": memory_snapshot()},
        "power": {"before": power_before, "after": power_after},
        "power_source_unchanged": power_before.splitlines()[0] == power_after.splitlines()[0],
        "counters": counters, "segments": segments,
        "gpu_counter_seconds": dict(sorted(
            ((k, v / 1e9) for k, v in counters.items()
             if (k == "gpu_time_ns" or k.endswith("_gpu_time_ns")) and v),
            key=lambda item: item[1], reverse=True)),
        "model_sha256": digest(path.with_suffix(".ubj")),
        "margins_sha256": digest(path.with_suffix(".margins.npy")),
    }
    write_json(path, report)
    assert clock_valid, "Sleep or wall-clock discontinuity invalidates timing"
    assert report["power_source_unchanged"], "Power source changed during timing"
    print(f"PROFILE {args.worker}: train {training_seconds:.3f}s; "
          f"GPU counters {report['gpu_counter_seconds']}", flush=True)


def main(args):
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "case": args.case, "runs": [], "comparisons": [],
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
        "script_sha256": digest(Path(__file__)),
        "timing": ("Common-cut diagnostic: quantization is in setup; do not compare training times with ordinary runs."
                   if args.cut_device else "Fresh worker per run; warm runtime; train includes quantization/evaluation; setup and prediction separate. verbosity=3 host monitors are in worker logs."),
        "counter_note": "gpu_time_ns is histogram GPU time; family counters can overlap. Do not sum them or add host wait_time_ns to GPU time.",
        "comparison_note": "Independent CPU/MPS training may choose different cuts. Report margins and held-out quality; MPS-model CPU prediction is checked exactly in each MPS worker.",
    }
    write_json(args.output / "summary.json", report)
    try:
        for repeat in range(args.repeats):
            devices = args.devices if repeat % 2 == 0 else list(reversed(args.devices))
            pair = {}
            for device in devices:
                print(f"PROFILE repeat {repeat + 1}/{args.repeats}: {device}", flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                           "--worker", device, "--repeat-index", str(repeat)]
                with (args.output / f"{device}-{repeat}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((args.output / f"{device}-{repeat}.json").read_text())
                report["runs"].append(result)
                pair[device] = result
                print(f"PROFILE completed {device}: {result['training_seconds']:.3f}s; "
                      f"GPU {result['gpu_counter_seconds']}", flush=True)
                write_json(args.output / "summary.json", report)
            if "cpu" in pair and "mps" in pair:
                import numpy as np
                assert pair["cpu"]["dataset"] == pair["mps"]["dataset"], "CPU/MPS datasets differ"
                cpu, mps = [np.load(args.output / f"{d}-{repeat}.margins.npy") for d in ("cpu", "mps")]
                report["comparisons"].append({
                    "repeat": repeat, "margin_max_abs_difference": float(np.max(np.abs(cpu - mps))),
                    "margin_rmse_difference": float(np.sqrt(np.mean((cpu.astype(float) - mps) ** 2))),
                    "class_disagreement_fraction": float(np.mean(
                        ((cpu > 0) != (mps > 0)) if cpu.ndim == 1 else
                        (np.argmax(cpu, axis=1) != np.argmax(mps, axis=1)))),
                    "accuracy_difference": pair["mps"]["test_accuracy"] - pair["cpu"]["test_accuracy"],
                    "logloss_difference": pair["mps"]["test_logloss"] - pair["cpu"]["test_logloss"],
                })
                if args.cut_device:
                    assert pair["cpu"]["common_cut_diagnostic"] == pair["mps"]["common_cut_diagnostic"]
                    from diagnose_higgs import model_difference, trees, xgb
                    models = [xgb.Booster(model_file=args.output / f"{d}-{repeat}.ubj") for d in ("cpu", "mps")]
                    report["comparisons"][-1]["model_difference"] = model_difference(*(trees(m) for m in models))
        report["training_median_seconds"] = {
            d: statistics.median(r["training_seconds"] for r in report["runs"] if r["device"] == d)
            for d in args.devices}
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        write_json(args.output / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["higgs", "covtype", "wide", "categorical"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for reports and worker logs")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/datasets")
    parser.add_argument("--package-dir", type=Path, default=ROOT / "xgboost/python-package")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--features", type=int, default=128, help="Feature count for synthetic cases")
    parser.add_argument("--informative-features", type=int,
                        help="Synthetic cases: number of informative columns spread across the matrix; defaults to three for wide and up to sixteen for categorical")
    parser.add_argument("--categories", type=int, default=256,
                        help="Distinct values per feature for the categorical case")
    parser.add_argument("--missing-fraction", type=float, default=0,
                        help="Synthetic cases: independently replace this fraction of features with NaN after generating labels")
    parser.add_argument("--cut-device", choices=["cpu", "mps"],
                        help="Correctness diagnostic with fixed reference cuts; changes setup/training timing boundaries")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-every", type=int, default=25)
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps"], default=["cpu", "mps"])
    parser.add_argument("--worker", choices=["cpu", "mps"], help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rows or 1, args.rounds, args.threads, args.repeats, args.profile_every) < 1 or args.features < 3:
        parser.error("counts must be positive and features must be at least three")
    if len(set(args.devices)) != len(args.devices):
        parser.error("devices must be unique")
    if args.informative_features is not None and (
        args.case not in {"wide", "categorical"} or not 1 <= args.informative_features <= args.features
    ):
        parser.error("informative-features requires wide/categorical and a count in [1, features]")
    if args.cut_device and args.case == "categorical":
        parser.error("The common-cut tree comparison currently requires numeric features")
    if args.case == "categorical" and not 2 <= args.categories <= 65536:
        parser.error("categories must be in [2, 65536]")
    if not 0 <= args.missing_fraction < 1 or (
        args.missing_fraction and args.case not in {"wide", "categorical"}
    ):
        parser.error("missing-fraction requires wide/categorical and a value in [0, 1)")
    main(args)
