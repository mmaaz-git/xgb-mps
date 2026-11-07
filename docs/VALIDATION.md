# Testing and validation

The latest backend run (`614f15a`) passed **106 native groups and 4,003 Python
tests**, with two expected skips: no-GPU behavior on a machine with a GPU, and
Linux-only loader behavior on macOS. Three installed-wheel workflows also passed
outside the checkout. All before/after MPS benchmark models, margins and metrics
were exact. [Commands, source/library hashes and results](../validation/wide-quality-and-performance-m3.json)
are revision-specific; later code changes require appropriate new checks.

## Run checks

The canonical runner builds first and recompiles selected native tests. Its
[manifest](../tests/native_tests.json) is the native-source inventory.

```sh
bash scripts/test.sh --list
bash scripts/test.sh --native 'metal_*model*' --dry-run
bash scripts/test.sh --native metal_model_state_api \
  --python tests/test_metal_model_state.py
bash scripts/test.sh --python tests/test_metal_workflows.py
bash scripts/test.sh --full
```

No selectors also means the full run. Python tests use `--require-metal`, which
fails when XGBoost's GPU is unavailable. Native Snappy checks require `snappy`;
interop acceptance also requires **both PyTorch and MLX**. Missing producer
libraries can still be skipped by `pytest.importorskip`, so inspect skips.

```sh
.venv/bin/python -m pip install torch mlx
.venv/bin/python -c 'import torch, mlx.core as mx; assert torch.backends.mps.is_available(); print(mx.default_device())'
```

Logs default to a new `results/test-runs/<UTC>-<pid>/` directory. `summary.json`
records commands, environments, exits, source diff and library hashes. A failed
step stops the run. Linear tests rebuild their independent reference when stale.
Inspect existing results before repeating expensive work; for a long job:

```sh
tmux new-session -d -s xgb-validation \
  'bash scripts/test.sh --full --log-dir results/validation-run > results/validation-run.log 2>&1'
tail -n 20 results/validation-run.log
```

Use a fresh session/path; `--log-dir` refuses to overwrite an existing directory.
For installation acceptance, install the wheel and run workflows outside the
checkout with source-path injection disabled; verify the loaded library hash.

## CPU-only and federated checks

These configurations are separate from the standard Metal run:

```sh
cmake -S xgboost -B build-cpu -G Ninja -DUSE_METAL=OFF -DUSE_CUDA=OFF \
  -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-cpu -j4
DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib" .venv/bin/python \
  scripts/check_cpu_build.py --metal-library xgboost/lib/libxgboost.dylib
bash scripts/test_federated.sh  # Optional; requires gRPC and Protobuf.
```

The CPU checker compares diagnostic schemas in separate processes and checks
disabled Metal APIs plus CPU matrix/cache behavior. It does not replace CPU
learner tests. Federated tests use a separate build/package; local workers do
not establish multi-host coverage.

## NVIDIA comparison

**No NVIDIA execution results are published.** The [CUDA CI job](../.github/workflows/cuda-compile.yml)
compiles production CUDA/NCCL/nvCOMP/federated paths and links comparison tools
on a CPU runner. Shared-source changes trigger it; the two Metal-only backend
files do not. Compiler success and synthetic parser tests are not GPU evidence.

For actual comparison, use the same source and fixture revision on both machines.
On macOS, export a fresh numerical observation file:

```sh
bash scripts/test.sh --native metal_numerical_contracts
MTL_DEBUG_LAYER=1 results/metal_numerical_contracts_test \
  xgboost/src/common/metal_histogram.metal --report results/metal-observations.jsonl
python3 scripts/compare_numerical_contracts.py inspect results/metal-observations.jsonl
```

On NVIDIA/Linux, set `CMAKE_CUDA_ARCHITECTURES` for that GPU (`80` is the CI
configuration). The harness requires a shared library with visible C++ symbols
and RMM disabled. The build driver preserves compiler flags and records input,
binary and library hashes; it prints runtime commands but does not execute them.

```sh
cmake -S xgboost -B build-cuda-run -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DUSE_CUDA=ON -DUSE_METAL=OFF -DUSE_NCCL=OFF -DUSE_NVCOMP=OFF \
  -DPLUGIN_FEDERATED=OFF -DPLUGIN_RMM=OFF \
  -DHIDE_CXX_SYMBOLS=OFF -DBUILD_STATIC_LIB=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
python3 scripts/build_cuda_comparison.py --build-dir build-cuda-run \
  --output-dir results/cuda-comparison-build
results/cuda-comparison-build/cmake/cuda_numerical_contracts \
  results/cuda-observations.jsonl
python3 scripts/compare_numerical_contracts.py compare \
  results/metal-observations.jsonl results/cuda-observations.jsonl \
  --output results/numerical-comparison.json
```

Preserve `build.json`, runtime logs, reports and binary sidecars together. Reports
retain input/output bits, subnormals and signed zeros. Comparators reject missing,
truncated or mismatched fixtures. Exit 0 means the recorded checks match; exit 1
means invalid evidence or a required-check failure; exit 2 requires review even
when outputs match, because domain or scheduling limits may remain. Consult the
[numerical contracts](CUDA_PARITY.md#numerical-contracts). An unavailable NVIDIA
device fails rather than skipping or falling back to CPU.

The other checked-in corpora use `unpack`, the indicated executable/mode, then
`compare`; run each `scripts/compare_*.py --help` for argument details. The table
uses executable suffixes after `cuda_`; all binaries are under the build output's
`cmake/` directory. [Harness sources](../tests/cuda_comparison/) and the
[build driver](../scripts/build_cuda_comparison.py) define compilation inputs.

| Corpus | Executable / mode | Comparator in `scripts/` |
| --- | --- | --- |
| AFT | `aft_contracts` | `compare_aft_contracts.py` |
| Primitive math | `math_contracts` | `compare_math_contracts.py` |
| Ranking rounding | `ranking_contracts` | `compare_ranking_contracts.py` |
| Ranking objectives / unbiased state | `ranking_contracts --objectives` | `compare_ranking_objectives.py` |
| Ranking primitives | `ranking_contracts --primitives` | `compare_ranking_primitives.py` |
| Ranking metrics | `ranking_contracts --metrics` | `compare_ranking_metrics.py` |
| Statistics / linalg | `numerical_contracts --statistics` / `--linalg` | `compare_stats_contracts.py` / `compare_linalg_contracts.py` |
| Stump weights / sampling | `numerical_contracts --stumps` / `--sampling` | `compare_stump_contracts.py` / `compare_sampling_contracts.py` |
| Integer leaves / cache | `numerical_contracts --integer-leaves` / `--integer-cache` | `compare_integer_leaf_contracts.py` / `compare_integer_cache_contracts.py` |
| Metric terms / linear updates | `metric_contracts` / `linear_contracts` | `compare_metric_contracts.py` / `compare_linear_contracts.py` |
| AUC / grouped PR / grouped ROC | `auc_contracts --auc` / `--grouped-pr` / `--grouped-roc` | `compare_auc_contracts.py` / `compare_grouped_pr_contracts.py` / `compare_grouped_roc_contracts.py` |

`cuda_rng_contracts REPORT.json` separately checks RNG streams, ranking pairs and
column sampling. `compare_gpu_pipeline.py` exercises production cuts, training
and prediction (`fixtures-v1` or the one-tree `higgs-v1` correctness workload).
Its `run` command requires a build record matching the loaded library and
production source tree on each machine. Generate both observations from the
same source tree, then compare:

```sh
python3 scripts/compare_gpu_pipeline.py compare \
  results/gpu-pipeline-fixtures-mps.json results/gpu-pipeline-fixtures-cuda.json \
  --output results/gpu-pipeline-fixtures-comparison.json
```

The stored pipeline observations are from `68c800e` and cannot be compared with
current-source CUDA results. These fixtures do not cover every external-memory, native-input,
distributed or numerical path. Missing replay modes are verification gaps, not
proof of missing MPS features. See [remaining work](CUDA_PARITY.md#remaining-work).

## Evidence conventions

Keep concise revision-specific records in [validation/](../validation/) and full
local logs in ignored `results/`. Record hardware, dependencies, exact commands,
hashes, passes, skips and failures. Counters show execution, not full residency;
ownership tests also check copy/readback behavior. CPU inference of an MPS model
tests that model's compatibility, not equality of independently trained models.
Current [performance measurements](PUBLISHED_BENCHMARKS.md) are separate from
correctness tests. Other Apple hardware and actual multi-host operation remain
unvalidated. Required [audit reports](audits/) and binary corpora in `validation/`
remain; superseded progress documents and logs are in Git history at `fc318c5`.
