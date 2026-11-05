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
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("tests/numerical_contract_fixtures.h", "tests/numerical_contract_report.h",
          "tests/cuda_numerical_contracts.cu", "tests/cuda_comparison/CMakeLists.txt",
          "tests/cuda_comparison/nvcomp_link_check.cc", "tests/cuda_aft_contracts.cu",
          "tests/aft_contract_io.h", "tests/cuda_rng_contracts.cu", "tests/rng_contract_fixtures.h",
          "tests/cuda_ranking_pairs.h", "tests/cuda_philox_reference.h",
          "tests/column_rng_contract_fixtures.h", "tests/cuda_column_reference.h",
          "tests/math_contract_io.h", "tests/cuda_math_contracts.cu",
          "tests/ranking_rounding_io.h", "tests/cuda_ranking_contracts.cu",
          "tests/ranking_objective_io.h", "tests/ranking_objective_report.h",
          "tests/ranking_primitive_io.h",
          "tests/stats_contract_io.h",
          "tests/linalg_contract_io.h",
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
                    for name in ("numerical", "aft", "rng", "math", "ranking")}
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
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
