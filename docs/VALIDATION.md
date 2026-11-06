# Validation commands and evidence

Use the vendored Python package and the freshly built library. The canonical
runner builds first so a changed header cannot accidentally test an old binary.
Inspect existing outputs before repeating expensive jobs.

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

- **Last full native/Python regression:** `41d4dc2`, 106 native groups and 3,919
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

The [local acceptance report](LOCAL_ACCEPTANCE.md) reconciles the original audit
fixes and numerical contracts. No NVIDIA GPU execution is recorded. Other Apple
hardware and actual multi-host operation remain unvalidated.
