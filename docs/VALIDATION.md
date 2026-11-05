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

No selectors means the full standard run: 102 native groups, the independent
linear reference build, then pytest with `--require-metal`. The manifest is
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
NVIDIA GPU and runs no Metal tests. A newer same-branch/PR run supersedes its old
compile job. Runtime jobs remain a separate planned gate.

Publish concise records in `validation/` naming revision, commands, dependencies,
hardware, passed/skipped/failed scope and limitations. Keep full local logs in
ignored `results/`. The latest pre-review full regression is
[native CSR storage](../validation/cuda-metal-native-csr-storage-m3.txt); review
fixes have [their own record](../validation/repo-review-2026-11-05.txt).

Latest full local regression: [PORT-6](../validation/cuda-port6-m3.txt), source
`2bfb914`, 102 native groups and 3,603 Python passes with one expected skip,
plus the separate CPU build/API/schema and nine updater checks.

Subsequent focused regression: [PORT-7](../validation/cuda-port7-m3.txt), source
`796e9fa`, six native groups and 820 Python passes without skips, including
Torch/MLX constructor interop, plus the CPU build/API/schema and nine updater
checks. This narrow constructor change did not rerun the full suite.
