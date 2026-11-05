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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--jobs", type=int, default=2)
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
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "gpu_execution": False,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "production_tree": subprocess.check_output(["git", "rev-parse", "HEAD:xgboost"], cwd=ROOT, text=True).strip(),
        "working_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
        "commands": [], "inputs": {},
    }
    for relative in ("tests/numerical_contract_fixtures.h", "tests/numerical_contract_report.h",
                     "tests/cuda_numerical_contracts.cu", "tests/cuda_comparison/CMakeLists.txt",
                     "scripts/build_cuda_comparison.py"):
        record["inputs"][relative] = sha(ROOT / relative)

    def save():
        (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")

    def run(command):
        record["commands"].append(command)
        save()
        print("RUN", shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    save()
    try:
        # Ninja rebuilds only changed dependencies. Never silently link a stale
        # library just because the shared object already exists.
        run(["cmake", "--build", str(build), "--parallel", str(args.jobs)])
        libdir = build / "lib" if cache.get("KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR") == "ON" else ROOT / "xgboost/lib"
        library = (libdir / "libxgboost.so").resolve(strict=True)
        configure = ["cmake", "-S", str(ROOT / "tests/cuda_comparison"), "-B", str(output / "cmake"),
                     "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                     f"-DXGB_SOURCE_DIR={ROOT / 'xgboost'}", f"-DXGB_BINARY_DIR={build}",
                     f"-DXGB_LIBRARY={library}"]
        # Use the production build's compiler, CCCL, architectures and arithmetic
        # flags, instead of silently compiling the template trace with defaults
        # from another toolkit or a different fast-math/FMA configuration.
        for key in ("CMAKE_CUDA_COMPILER", "CMAKE_CUDA_HOST_COMPILER", "CMAKE_CUDA_ARCHITECTURES",
                    "CMAKE_CUDA_FLAGS", "CMAKE_CUDA_FLAGS_RELEASE", "CCCL_DIR"):
            if key in cache:
                configure.append(f"-D{key}={cache[key]}")
        run(configure)
        run(["cmake", "--build", str(output / "cmake"), "--parallel", str(args.jobs)])
        binary = output / "cmake/cuda_numerical_contracts"
        record.update(status="passed-compile-link", library=str(library), library_sha256=sha(library),
                      binary=str(binary), binary_sha256=sha(binary),
                      production_cache_sha256=sha(build / "CMakeCache.txt"),
                      compile_commands=json.loads((output / "cmake/compile_commands.json").read_text()))
        print("PASS CUDA harness compile/link; NVIDIA execution has NOT occurred.", flush=True)
        print("Runtime command:", shlex.join([str(binary), str(output / "cuda-observations.jsonl")]), flush=True)
    except (OSError, subprocess.CalledProcessError) as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
