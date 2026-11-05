#!/usr/bin/env python3
"""Compile the CUDA observation harness against a configured vendored build.

This never executes GPU code. The runtime command is printed after a successful
link. Compiler commands, input/library hashes and terminal status are retained.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("tests/numerical_contract_fixtures.h", "tests/numerical_contract_report.h",
          "tests/cuda_numerical_contracts.cu", "tests/cuda_comparison/CMakeLists.txt",
          "tests/cuda_comparison/nvcomp_link_check.cc", "tests/cuda_aft_contracts.cu",
          "tests/cuda_comparison/nvcomp_runtime.cmake",
          "tests/aft_contract_io.h", "tests/cuda_rng_contracts.cu", "tests/rng_contract_fixtures.h",
          "tests/cuda_ranking_pairs.h", "tests/cuda_philox_reference.h",
          "tests/column_rng_contract_fixtures.h", "tests/cuda_column_reference.h",
          "tests/math_contract_io.h", "tests/cuda_math_contracts.cu",
          "tests/ranking_rounding_io.h", "tests/cuda_ranking_contracts.cu",
          "tests/ranking_objective_io.h", "tests/ranking_objective_report.h",
          "tests/ranking_primitive_io.h",
          "tests/stats_contract_io.h",
          "tests/linalg_contract_io.h",
          "tests/stump_contract_io.h", "tests/cuda_stump_contracts.cuh", "xgboost/src/tree/fit_stump.cu",
          "tests/sampling_contract_io.h", "tests/cuda_sampling_contracts.cuh",
          "tests/integer_leaf_contract_io.h", "tests/cuda_integer_leaf_contracts.cuh",
          "tests/integer_cache_contract_io.h", "tests/cuda_integer_cache_contracts.cuh",
          "tests/linear_contract_io.h", "tests/cuda_linear_contracts.cu",
          "tests/cuda_linear_source.h", "tests/linear_registry_probe.h",
          "xgboost/src/linear/updater_gpu_coordinate.cu", "xgboost/src/linear/coordinate_common.h",
          "xgboost/src/linear/param.h", "xgboost/include/xgboost/linear_updater.h",
          "tests/grouped_pr_contract_io.h", "tests/cuda_auc_contracts.cu",
          "tests/auc_capture.cuh", "scripts/prepare_auc_probe.py",
          "xgboost/src/metric/auc.cu", "xgboost/src/metric/auc.h",
          "xgboost/src/tree/gpu_hist/histogram.cu", "xgboost/src/tree/gpu_hist/histogram.cuh",
          "xgboost/src/tree/gpu_hist/feature_groups.cu", "xgboost/src/tree/gpu_hist/feature_groups.cuh",
          "xgboost/src/data/ellpack_page.cu", "xgboost/src/data/ellpack_page.cuh",
          "xgboost/src/data/gradient_index.cc", "xgboost/src/data/gradient_index.h",
          "xgboost/src/tree/gpu_hist/leaf_sum.cu", "xgboost/src/tree/gpu_hist/leaf_sum.cuh",
          "xgboost/src/tree/param.h", "xgboost/src/tree/split_evaluator.h",
          "xgboost/src/tree/gpu_hist/sampler.cu", "xgboost/src/tree/gpu_hist/sampler.cuh",
          "xgboost/src/tree/gpu_hist/quantiser.cu", "xgboost/src/tree/gpu_hist/quantiser.cuh",
          "xgboost/src/tree/updater_gpu_hist.cuh", "xgboost/src/tree/updater_gpu_hist.cu",
          "tests/metric_contract_io.h", "tests/cuda_metric_source.h",
          "tests/cuda_metric_contracts.cu", "tests/metric_registry_probe.h",
          "tests/cuda_metric_reference.h",
          "xgboost/src/metric/elementwise_metric.cu", "xgboost/src/metric/multiclass_metric.cu",
          "xgboost/src/metric/survival_metric.cu",
          "scripts/build_cuda_comparison.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    # Container checkout ownership can differ from the step's uid. Trust only
    # this explicitly selected checkout for this invocation; no global setting.
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", *args], cwd=ROOT)


def cache_values(path):
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-cuda-ci")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New for all/compile; successful object-compile record for link")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--phase", choices=("all", "compile", "link"), default="all",
                        help="CI can compile the object before building the production library; link resumes that record")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    cache = cache_values(build / "CMakeCache.txt")
    if cache.get("USE_CUDA") != "ON" or cache.get("USE_METAL") != "OFF":
        parser.error("The supplied build must use CUDA=ON and METAL=OFF")
    for key in ("HIDE_CXX_SYMBOLS", "BUILD_STATIC_LIB", "PLUGIN_RMM"):
        if cache.get(key) == "ON":
            parser.error(f"This harness currently requires {key}=OFF")
    if Path(cache["CMAKE_HOME_DIRECTORY"]).resolve() != ROOT / "xgboost":
        parser.error("The configured build belongs to a different source checkout")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.phase == "link":
        record = json.loads((output / "build.json").read_text())
        if record["status"] != "passed-object-compile":
            parser.error("--phase link requires a successful object-compile record")
        record["object_finished_utc"] = record.pop("finished_utc")
        record["link_started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record["status"] = "link-in-progress"
    else:
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "gpu_execution": False,
                  "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "commands": [], "inputs": {}}

    def save():
        (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")

    def run(command):
        record["commands"].append(command)
        save()
        print("RUN", shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    save()
    try:
        tree = git("rev-parse", "HEAD:xgboost").decode().strip()
        diff = hashlib.sha256(git("diff", "HEAD", "--", "xgboost")).hexdigest()
        inputs = {relative: sha(ROOT / relative) for relative in INPUTS}
        if args.phase == "link":
            if (record["production_tree"] != tree or record["production_working_diff_sha256"] != diff
                    or record["inputs"] != inputs or record["production_cache_sha256"] != sha(build / "CMakeCache.txt")):
                raise ValueError("Source, fixture inputs or build configuration changed after object compilation")
        else:
            record.update(revision=git("rev-parse", "HEAD").decode().strip(), production_tree=tree,
                          production_working_diff_sha256=diff, inputs=inputs,
                          production_cache_sha256=sha(build / "CMakeCache.txt"))
        libdir = build / "lib" if cache.get("KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR") == "ON" else ROOT / "xgboost/lib"
        library = libdir / "libxgboost.so"
        configure = ["cmake", "-S", str(ROOT / "tests/cuda_comparison"), "-B", str(output / "cmake"),
                     "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                     f"-DXGB_SOURCE_DIR={ROOT / 'xgboost'}", f"-DXGB_BINARY_DIR={build}",
                     f"-DXGB_LIBRARY={library}", f"-DXGB_USE_NVCOMP={cache.get('USE_NVCOMP', 'OFF')}"]
        # Use the production build's compiler, CCCL, architectures and arithmetic
        # flags, instead of silently compiling the template trace with defaults
        # from another toolkit or a different fast-math/FMA configuration.
        for key in ("CMAKE_CUDA_COMPILER", "CMAKE_CUDA_HOST_COMPILER", "CMAKE_CUDA_ARCHITECTURES",
                    "CMAKE_CUDA_FLAGS", "CMAKE_CUDA_FLAGS_RELEASE", "CCCL_DIR", "nvcomp_DIR"):
            if key in cache:
                configure.append(f"-D{key}={cache[key]}")
        if args.phase != "link":
            run(configure)
            run(["cmake", "--build", str(output / "cmake"), "--target", "cuda_comparison_objects",
                 "--parallel", str(args.jobs)])
            if cache.get("USE_NVCOMP") == "ON":
                # Address-only host probe: validates the packaged SONAME search
                # path without calling CUDA or requiring an NVIDIA device.
                run([str(output / "cmake/cuda_nvcomp_link_check")])
                record["nvcomp_loader_probe"] = {"status": "passed", "gpu_execution": False}
            record.update(status="passed-object-compile",
                          compile_commands=json.loads((output / "cmake/compile_commands.json").read_text()))
            save()
            if args.phase == "compile":
                print("PASS CUDA harness object compilation; library link and GPU execution remain.", flush=True)
                return
        # Ninja checks changed production dependencies before every final link.
        # An existing shared object alone never proves it matches current source.
        record["status"] = "link-in-progress"
        run(["cmake", "--build", str(build), "--parallel", str(args.jobs)])
        run(["cmake", "--build", str(output / "cmake"), "--parallel", str(args.jobs)])
        binary = output / "cmake/cuda_numerical_contracts"
        binaries = {name: {"path": str(output / f"cmake/cuda_{name}_contracts"),
                           "sha256": sha(output / f"cmake/cuda_{name}_contracts")}
                    for name in ("numerical", "aft", "rng", "math", "ranking", "metric", "linear", "auc")}
        # Host-only checks, runnable on compiler CI without NVIDIA hardware.
        # Verify both the real registry and dynamic symbol boundary: the metric
        # probe must neither register alternate factories nor interpose bodies.
        def capture(command, path):
            record["commands"].append(command)
            save()
            print("RUN host isolation check", shlex.join(command), flush=True)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            path.write_text(result.stdout)
            if result.stderr:
                print(result.stderr, end="", flush=True)
            if result.returncode:
                raise RuntimeError(f"Host isolation command failed with exit {result.returncode}: {shlex.join(command)}")
            return result.stdout
        baseline_path, probe_path = output/"metric-registry-baseline.txt", output/"metric-registry-probe.txt"
        baseline = capture([binaries["numerical"]["path"], "--metric-registry"], baseline_path)
        probe = capture([binaries["metric"]["path"], "--metric-registry"], probe_path)
        if baseline != probe or not any(line.startswith("rmse ") for line in baseline.splitlines()):
            raise ValueError("Metric probe changed/omitted registered names or factory types")
        symbols_path = output/"metric-defined-dynamic-symbols.txt"
        symbols = capture(["nm", "-D", "--defined-only", "-C", binaries["metric"]["path"]], symbols_path)
        exported_names = [re.sub(r"^\s*(?:[0-9a-fA-F]+\s+)?[A-Za-z?]\s+", "", line)
                          for line in symbols.splitlines()]
        metric_definition = r"^(?:(?:non-virtual|virtual) thunk to |(?:typeinfo(?: name)?|vtable|VTT) for |guard variable for )?xgboost::metric::"
        if any(re.match(metric_definition, name) for name in exported_names):
            raise ValueError("Metric source probe exports production metric definitions")
        record["metric_source_isolation"] = {
            "status": "passed", "gpu_execution": False,
            "registry_entries": len(baseline.splitlines()),
            "files": {str(path): sha(path) for path in (baseline_path, probe_path, symbols_path)}}
        print("PASS host metric registry/factory identity and hidden source definitions; no GPU execution", flush=True)
        auc_paths = (output/"auc-registry-probe.txt", output/"auc-defined-dynamic-symbols.txt")
        auc_registry = capture([binaries["auc"]["path"], "--metric-registry"], auc_paths[0])
        if auc_registry != baseline:
            raise ValueError("AUC source probe changed metric registration/factory types")
        auc_symbols = capture(["nm", "-D", "--defined-only", "-C", binaries["auc"]["path"]], auc_paths[1])
        auc_names = [re.sub(r"^\s*(?:[0-9a-fA-F]+\s+)?[A-Za-z?]\s+", "", line)
                     for line in auc_symbols.splitlines()]
        if any(re.match(metric_definition, name) for name in auc_names):
            raise ValueError("AUC source probe exports production/probe metric definitions")
        generated = output/"cmake/auc_probe.inc"
        generated_record = generated.with_suffix(".inc.json")
        source_record = json.loads(generated_record.read_text())
        probe_spec = importlib.util.spec_from_file_location("auc_source_probe", ROOT/"scripts/prepare_auc_probe.py")
        probe_module = importlib.util.module_from_spec(probe_spec)
        probe_spec.loader.exec_module(probe_module)
        expected_source, expected_record = probe_module.generate((ROOT/"xgboost/src/metric/auc.cu").read_bytes())
        if (source_record["source_sha256"] != inputs["xgboost/src/metric/auc.cu"]
                or source_record != expected_record or generated.read_bytes() != expected_source):
            raise ValueError("AUC source observation provenance differs from build inputs")
        record["auc_source_isolation"] = {
            "status": "passed", "gpu_execution": False,
            "registry_entries": len(auc_registry.splitlines()),
            "files": {str(path): sha(path) for path in (*auc_paths, generated, generated_record)}}
        print("PASS host AUC registry identity, hidden source copy and generated-source provenance; no GPU execution", flush=True)
        linear_paths = (output/"linear-registry-baseline.txt", output/"linear-registry-probe.txt",
                        output/"linear-defined-dynamic-symbols.txt")
        linear_baseline = capture([binaries["numerical"]["path"], "--linear-registry"], linear_paths[0])
        linear_probe = capture([binaries["linear"]["path"], "--linear-registry"], linear_paths[1])
        if linear_baseline != linear_probe or not any(line.startswith("gpu_coord_descent ") for line in linear_baseline.splitlines()):
            raise ValueError("Linear probe changed/omitted registered names or factory types")
        linear_symbols = capture(["nm", "-D", "--defined-only", "-C", binaries["linear"]["path"]], linear_paths[2])
        linear_names = [re.sub(r"^\s*(?:[0-9a-fA-F]+\s+)?[A-Za-z?]\s+", "", line)
                        for line in linear_symbols.splitlines()]
        linear_definition = r"^(?:(?:non-virtual|virtual) thunk to |(?:typeinfo(?: name)?|vtable|VTT) for |guard variable for )?xgboost::linear::"
        if any(re.match(linear_definition, name) for name in linear_names):
            raise ValueError("Linear source probe exports production linear definitions")
        record["linear_source_isolation"] = {
            "status": "passed", "gpu_execution": False,
            "registry_entries": len(linear_baseline.splitlines()),
            "files": {str(path): sha(path) for path in linear_paths}}
        print("PASS host linear registry/factory identity and hidden source definitions; no GPU execution", flush=True)
        rng_path = output / "sampling-rng-alignment.txt"
        rng_check = capture([binaries["numerical"]["path"], "--sampling-rng"], rng_path)
        if rng_check.strip() != "PASS sampling MT19937 row-seed alignment: 10005 cases; no GPU execution":
            raise ValueError("Sampling RNG alignment check did not complete its full host corpus")
        record["sampling_rng_alignment"] = {
            "status": "passed", "gpu_execution": False, "cases": 10005,
            "files": {str(rng_path): sha(rng_path)}}
        if (git("rev-parse", "HEAD:xgboost").decode().strip() != tree
                or hashlib.sha256(git("diff", "HEAD", "--", "xgboost")).hexdigest() != diff
                or {relative: sha(ROOT / relative) for relative in INPUTS} != inputs):
            raise ValueError("Source or fixture inputs changed during compilation/linking")
        record.update(status="passed-compile-link", library=str(library.resolve(strict=True)), library_sha256=sha(library),
                      binary=str(binary), binary_sha256=sha(binary), binaries=binaries,
                      production_cache_sha256=sha(build / "CMakeCache.txt"),
                      compile_commands=json.loads((output / "cmake/compile_commands.json").read_text()))
        print("PASS CUDA harness compile/link; NVIDIA execution has NOT occurred.", flush=True)
        print("Runtime command:", shlex.join([str(binary), str(output / "cuda-observations.jsonl")]), flush=True)
        print("AFT runtime command:", shlex.join([binaries["aft"]["path"], "results/aft-inputs.bin",
                                                str(output / "cuda-aft-observations.bin")]), flush=True)
        print("RNG runtime command:", shlex.join([binaries["rng"]["path"],
                                                str(output / "cuda-rng-report.json")]), flush=True)
        print("Math runtime command:", shlex.join([binaries["math"]["path"], "results/math-inputs.bin",
                                                 str(output / "cuda-math-observations.bin")]), flush=True)
        print("Ranking runtime command:", shlex.join([binaries["ranking"]["path"], "results/ranking-inputs.bin",
                                                    str(output / "cuda-ranking-observations.bin")]), flush=True)
        print("Ranking objective command:", shlex.join([binaries["ranking"]["path"], "--objectives",
                "results/ranking-objective-inputs.bin", str(output / "cuda-ranking-objective-observations.bin")]), flush=True)
        print("Ranking primitive command:", shlex.join([binaries["ranking"]["path"], "--primitives",
                "results/ranking-primitive-inputs.bin", str(output / "cuda-ranking-primitive-observations.bin")]), flush=True)
        print("Statistics runtime command:", shlex.join([binaries["numerical"]["path"], "--statistics",
                "results/stats-inputs.bin", str(output / "cuda-stats-observations.bin")]), flush=True)
        print("Linalg runtime command:", shlex.join([binaries["numerical"]["path"], "--linalg",
                "results/linalg-inputs.bin", str(output / "cuda-linalg-observations.bin")]), flush=True)
        print("Metric runtime command:", shlex.join([binaries["metric"]["path"], "results/metric-inputs.bin",
                str(output / "cuda-metric-observations.bin")]), flush=True)
        print("Stump runtime command:", shlex.join([binaries["numerical"]["path"], "--stumps",
                "results/stump-inputs.bin", str(output / "cuda-stump-observations.bin")]), flush=True)
        print("Sampling runtime command:", shlex.join([binaries["numerical"]["path"], "--sampling",
                "results/sampling-inputs.bin", str(output / "cuda-sampling-observations.bin")]), flush=True)
        print("Integer leaf runtime command:", shlex.join([binaries["numerical"]["path"], "--integer-leaves",
                "results/integer-leaf-inputs.bin", str(output / "cuda-integer-leaf-observations.bin")]), flush=True)
        print("Integer cache runtime command:", shlex.join([binaries["numerical"]["path"], "--integer-cache",
                "results/integer-cache-inputs.bin", str(output / "cuda-integer-cache-observations.bin")]), flush=True)
        print("Linear runtime command:", shlex.join([binaries["linear"]["path"],
                "results/linear-inputs.bin", str(output / "cuda-linear-observations.bin")]), flush=True)
        print("Grouped PR runtime command:", shlex.join([binaries["auc"]["path"], "--grouped-pr",
                "results/grouped-pr-inputs.bin", str(output / "cuda-grouped-pr-observations.bin")]), flush=True)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
