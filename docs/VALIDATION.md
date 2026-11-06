# Validation commands and evidence

Use the vendored Python package and the freshly built library. The canonical
runner builds first so a changed header cannot accidentally test an old binary.
Inspect existing outputs before repeating expensive jobs.

The 2026-11-05 removal of R/JVM/Spark was checked with CPU and Metal builds,
wheel/source-archive inspection, and a fresh wheel installation outside the
checkout: 77 workflow/model-state tests passed, with one expected no-GPU skip.
Regression/classification train/predict/save/load and CPU model interchange also
passed. The Metal library hash is unchanged from `cb495ee`.
[Recorded checks](../validation/python-only-m3.json).

## Select checks

```sh
bash scripts/test.sh --list
bash scripts/test.sh --native 'metal_*model*' --dry-run
bash scripts/test.sh --native metal_model_state_api \
  --python tests/test_metal_model_state.py
bash scripts/test.sh --python tests/test_metal_metadata.py
bash scripts/test.sh --full
```

No selectors means the full standard run: 106 native groups, then pytest with
`--require-metal`. Linear tests build their independent reference only when
selected and the reference is missing or stale. The manifest is
[tests/native_tests.json](../tests/native_tests.json). The runner validates its
native-source inventory and recompiles selected test binaries. Optional federated
transport is deliberately in its separate build, not missing from this manifest.

Logs default to a new `results/test-runs/<UTC>-<pid>/` directory. `--log-dir PATH`
requires a new directory and will not overwrite an existing run. `summary.json`
records selections, exact commands/environments, exit codes, source revision/diff,
untracked-file hashes and the built library hash. A failed step stops the run.

For a long run:

```sh
tmux new-session -d -s xgb-validation \
  'bash scripts/test.sh --full --log-dir results/validation-run > results/validation-run.log 2>&1'
tail -n 20 results/validation-run.log
```

Run from the repository root; use a fresh session/path if that name already
exists. Check the existing process/log rather than starting a duplicate.

## Dependencies and skips

README describes the build dependencies. Native Snappy checks also need `snappy`.
PyTorch and MLX are optional packages for general use but **both are required for
interop acceptance**. Verify their GPU devices, then inspect the pytest skip
summary. `--require-metal` only guarantees that Metal-dependent fixtures fail
instead of skipping when the XGBoost GPU is unavailable; it does not override
`pytest.importorskip` for producer libraries.

```sh
.venv/bin/python -m pip install torch mlx
.venv/bin/python -c 'import torch, mlx.core as mx; assert torch.backends.mps.is_available(); print(mx.default_device())'
```

Counters establish execution of the counted operations. Residency/lifetime tests
must inspect ownership and copy/readback counters before host comparison. CPU
reading an MPS-trained model tests model/reader compatibility, not independent
CPU-training equivalence. CUDA-source oracles and NVIDIA execution are different
evidence classes too.

## CPU-only and federated gates

The standard runner does not silently include these configurations:

```sh
cmake -S xgboost -B build-cpu -G Ninja -DUSE_METAL=OFF -DUSE_CUDA=OFF \
  -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-cpu -j4
DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib" .venv/bin/python \
  scripts/check_cpu_build.py --metal-library xgboost/lib/libxgboost.dylib

# Optional, requires gRPC and Protobuf; separate library/Python package.
bash scripts/test_federated.sh
```

The CPU checker loads the two libraries in separate processes and compares their
diagnostic schemas, then checks disabled Metal APIs and CPU matrix/cache behavior.
It does not replace CPU learner tests. A milestone must explicitly list its CPU
learner selections and confirm `USE_METAL=false` for that package.

## CI and publication

The GitHub [CUDA workflow](../.github/workflows/cuda-compile.yml) compiles shared
source with CUDA/NCCL/nvCOMP/federated dependencies on a CPU runner. It has no
NVIDIA GPU and runs no Metal tests. An active same-branch/PR build finishes;
newer revisions replace only the pending build. The two Metal-only implementation
files do not trigger it by themselves; shared headers, stubs and build files do.
Sparse checkout includes compiler inputs, excluding recorded validation corpora.
Runtime jobs remain a separate planned gate.
The workflow now also compiles/links the [shared numerical comparison harness](NVIDIA_COMPARISON.md)
without executing it. The report parser's synthetic integrity tests and a
successful link must not be reported as NVIDIA runtime evidence.

Publish concise records in `validation/` naming revision, commands, dependencies,
hardware, passed/skipped/failed scope and limitations. Keep full local logs in
ignored `results/`. Evidence is revision-specific; later focused checks do not
turn a historical full run into a current full run.

## Recorded results

- **Fresh full regression, exact FP32 ordering and input staging:** clean
  `28908e1` passed all 106 native groups and 4,001 Python tests, with two expected
  platform skips. The subsequent candidate passed seven native groups and 427
  Python tests, followed by three installed-package workflows. Direct ordering
  checks cover 1,400,784 FP32 pairs. All 26 MPS benchmark runs reproduce baseline
  models, margins and metric histories exactly across dense, missing-value,
  categorical, Covertype and full-row 1,000-round HIGGS workloads. The full run
  predates the optimization; final acceptance is focused. No NVIDIA execution.
  [Sources, all 41 timings and checks](../validation/regression-and-benchmarks-m3.json).
- **Batched split commands, wide histogram dispatch and native conservative
  bounds:** eight native groups and 300 Python tests passed, with one expected
  empty-dataset warning, followed by three installed-package workflows. Direct
  quantized-bound containment covers 1,044,483 cases; the scalar oracle checks
  1,992 split tasks. Compressed consumers pass 318 cases and 73,234,428 exact
  fields. New mapped-parent inputs exercise queued staging ownership. All 18 MPS
  models, margin arrays and metric histories match before/after on two synthetic
  datasets and full-row HIGGS. This is focused acceptance, not a full-suite rerun
  or NVIDIA execution. [Sources, all 27 timings and checks](../validation/parallel-tuning-m3.json).
- **Conservative quantized split screening and identical-gain reuse:** five native
  groups and 178 Python tests passed, followed by three installed-package workflows.
  The expanded independent CUDA-formula oracle checks 1,992 tasks and 426,234 bins,
  including exact ties, near ties, zero curvature, cancellation and interval
  fallback ranges. Over one million gain-bound checks also pass. All 12 MPS
  models, margins and per-round metric histories in the paired large workloads
  match before/after exactly. This is focused acceptance, not a full-suite rerun
  or NVIDIA execution. [Sources, timings and checks](../validation/quantized-split-tuning-m3.json).
- **Partition-copy removal and bounded feature-key sorting:** the final build
  passed four native groups and 171 Python tests, with no skips, followed by
  three installed-package workflows outside the checkout. Partition-only checks
  had also passed three native groups and 36 Python tests. New cases exercise
  repeated full/partial/unordered partitions and weighted feature-key ties across
  radix boundaries. All 24 MPS model/margin outputs across two large-workload
  experiments match before/after. This is focused acceptance; the full standard
  run below predates these optimizations. End-to-end performance remains mixed.
  [Hashes, complete timings, kernel checks and CPU sample](../validation/partition-sketch-tuning-m3.json).
- **Four-feature histogram tiling:** four native groups and 89 Python tests
  passed, with one expected no-GPU-test skip. The final compressed-consumer
  assertions also passed separately: 316 cases and 72,700,780 exact checked
  fields. Three real Python workflows passed against the newly installed wheel
  outside the checkout. Across two large datasets, all six MPS models and margin
  arrays per dataset match exactly before/after. This is focused acceptance of
  the optimization; the full run below predates it.
  [Source hashes, checks and all repeated timings](../validation/histogram-tiling-m3.json).
- **Python workflow baseline before the next performance change:** `0dfde8b` plus
  `test_metal_workflows.py`, 106 native groups and 4,001 Python passes with two
  expected platform skips. New workflows cover mixed-column pipelines and
  cross-validation, weighted categorical classification with early stopping,
  and grouped ranking. Those three workflows also passed against the installed
  package with source-path injection disabled.
  [Revision, test hash and results](../validation/python-workflows-full-m3.json).
- **Fresh Python installation, including the optimized sigmoid build:** the wheel's own package/library
  loaded in a fresh environment outside the checkout, without `PYTHONPATH` or
  `DYLD_LIBRARY_PATH`. Regression and classification train/predict/save/load/CPU
  interchange passed, plus 77 existing workflow/model tests and one expected
  no-GPU-test skip. [Installation record](../validation/python-install-m3.json).
  The optimized wheel is also installed in the project's `.venv`, with its
  loaded library hash and Metal train/predict dispatches checked outside the checkout.
- **Sigmoid reuse optimization:** five native groups and 112 Python tests passed
  with one expected skip. The existing sigmoid corpus now also checks the
  objective entry point: 2,996,040 inputs agree exactly with its binary64 reference.
  All six full-row, 100-round HIGGS models and margin arrays match before/after.
  [Source hashes, complete timings and limits](../validation/sigmoid-reuse-m3.json).
- **HIGGS source-reference diagnosis:** all 6,184 saved MPS cuts across 28
  features exactly match CUDA-source equations; CPU cuts differ for 24 features.
  No new training or NVIDIA execution. [Scope and command](../validation/higgs-cut-source-m3.json).
- **Earlier full native/Python regression:** `41d4dc2`, 106 native groups and 3,919
  Python passes with two expected skips. See the
  [PORT-13 full-run record](../validation/output-capacity-full-m3-run.json).
- **Accepted reader optimization:** `818e11f`, six native groups and 160 Python
  passes, plus bitwise model/margin comparisons across six 100-round MPS runs.
  See [reader evidence](../validation/ellpack-read-m3-run.json).
- **This cleanup:** based on `6f3dcbd`, two native groups and 186 Python tests
  passed after removing unused host split bridges. The selected linear test
  rebuilt its stale reference automatically. The existing CPU configuration
  built and its 305-field schema/API/cache checks passed. The operation-map
  reference check passed; it does not establish behavior. Exact source hashes,
  commands and limitations are in the [cleanup record](../validation/repo-cleanup-2026-11-05.json).
- **Separate historical CPU/federated acceptance:** `68c800e`, CPU schema/API
  checks and nine updater selections; 14 federated stage/world groups including
  TLS. All workers used one M3. See [PORT-11](../validation/cuda-port11-m3.txt).
- **Latest verified CUDA compile before this cleanup:** `818e11f`,
  [run 42133058442](https://github.com/mmaaz-git/xgb-mps/actions/runs/42133058442),
  all 197 production steps and eight comparison executable links. Input hashes
  and host loader/registry/source/RNG checks were verified. This predates the
  cleanup and does not validate its new CI checkout configuration.
  [Compiler history](../validation/cuda-compile-ci-13.4.txt).
- **Cleanup CUDA workflow at `383f19d`:** GitHub reports
  [run 42138562183](https://github.com/mmaaz-git/xgb-mps/actions/runs/42138562183)
  successful, including the new sparse checkout, CUDA compile/host checks and
  artifact upload. This status inspection is not a repeat artifact-hash audit
  and does not include NVIDIA execution.

The [local acceptance report](LOCAL_ACCEPTANCE.md) reconciles the original audit
fixes and numerical contracts. No NVIDIA GPU execution is recorded. Other Apple
hardware and actual multi-host operation remain unvalidated.
