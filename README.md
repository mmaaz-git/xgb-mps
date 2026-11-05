# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU backend for XGBoost 3.4.2. It uses custom Metal
compute kernels behind `device="mps"`; it is an independent fork, not an upstream
XGBoost release.

**Training and prediction work. Full CUDA parity is not established.** A passing
test suite does not prove that every CUDA operation has an equivalent Metal
path. CUDA source defines the intended
behavior; CPU comparisons apply where the algorithms agree. No NVIDIA GPU
runtime comparison has been performed.

Start with the [caller audit](docs/CUDA_AUDIT.md), [review findings](docs/REVIEW.md) and
[implementation plan](docs/PLAN.md). Correctness and operation coverage take
priority over further performance tuning.

## Current scope

Implemented paths have local tests on an **Apple M3, 10-core GPU, 16 GB RAM**.
This is the tested hardware, not a claim of validation across Apple devices.

| Area | Implemented paths |
| --- | --- |
| Tree training | `hist` and `approx`, scalar and vector trees, missing/categorical data, row/column sampling, and reduced-gradient leaf fitting |
| Objectives and metrics | GPU implementations for regression, classification, ranking, survival, and adaptive objectives; numerical qualifications are recorded in the parity ledger |
| Prediction | Batch/inplace prediction, leaf outputs, and SHAP contributions/interactions |
| Data | Ordinary/quantile matrices, native Metal tensor and columnar inputs, metadata, and host/disk external-memory caches |
| Other training | Linear coordinate descent, TCP distributed trees, and optional federated gRPC trees |

The audit at `382f78c` identified **10 implementation tasks**, including weighted-
quantile precision and vector-row validity bugs, plus GPU state/data-placement
gaps. Its 174 logical rows and 1,078 classified lexical sites are an inventory,
**not a completion percentage or proof of parity**. See the
[findings and evidence](docs/CUDA_AUDIT.md) and [ordered plan](docs/PLAN.md).
All ten tasks are locally complete. The [local acceptance report](docs/LOCAL_ACCEPTANCE.md)
connects every fix and all 17 numerical contracts to passing fixtures and states
the remaining limits. The latest production change (`68c800e`) initializes sketch
pointers and categorical state on GPU; subsequent work adds numerical witnesses.
**Local verification is complete, including CUDA compilation. Next: NVIDIA
execution comparisons and current-source performance work.** Full parity remains
unproven.

The first [NVIDIA comparison harness](docs/NVIDIA_COMPARISON.md) covers shared
CDF and sketch-order witnesses. It records full output bits and keeps differences
visible for review; NVIDIA execution is still required.

`tree_method="exact"` is unsupported on MPS. For linear GPU training, use
`booster="gblinear", updater="coord_descent"`; the default `shotgun` updater
remains on CPU, as it does with CUDA. Distributed `gblinear` is unsupported.

## Build and try it

Requires an Apple GPU on macOS, Xcode command-line tools, Python **3.11+**, CMake,
Ninja, and OpenMP. The commands below use the vendored Python package directly.

```sh
git clone https://github.com/mmaaz-git/xgb-mps.git
cd xgb-mps
brew install cmake ninja libomp
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy scikit-learn pandas pytest
bash scripts/build.sh
export PYTHONPATH="$PWD/xgboost/python-package"
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Run this with `.venv/bin/python`:

```python
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
X = rng.normal(size=(10000, 32)).astype(np.float32)
y = 2 * X[:, 0] - X[:, 1]

print(xgb.metal_info())  # Check that Metal is built and available.
model = xgb.XGBRegressor(device="mps", tree_method="hist", n_estimators=20)
model.fit(X, y)
prediction = model.predict(X)
```

To build quantiles on Metal from host arrays, use
`xgb.QuantileDMatrix(X, y, device="mps")`. Native Metal inputs and outputs support
DLPack interoperability with PyTorch and MLX. Ordinary `predict(DMatrix)` returns
NumPy by default; native inplace prediction can retain GPU output.

## Validation

Run the standard native and Python suites; the runner builds the current library first:

```sh
brew install snappy  # Independent decoder used by native tests only.
bash scripts/test.sh
```

For focused work, use `bash scripts/test.sh --list`, then select native IDs and/or
Python files with `--native` / `--python`. `--dry-run` prints commands without
executing them. Logs and status are saved per run; see [validation instructions](docs/VALIDATION.md).

The suite requires an accessible GPU and uses completed-operation counters to
check GPU execution. PyTorch and MLX are additional dependencies for their
native-tensor tests. Optional federated tests require `brew install grpc protobuf`
and run with `bash scripts/test_federated.sh` in separate build directories.

Latest full regression covers `68c800e`: **103 native groups and 3,647 Python
tests passed, with one expected skip**, plus CPU build/API/schema and nine updater
checks. PyTorch and MLX GPU interop tests ran. The
[record](validation/cuda-port11-m3.txt) includes commands, source hashes and limits.
The optional federated suite also passed at this revision: 14 transport, sketch,
and learner groups, including TLS, with all workers on the same M3. A subsequent
focused run passed four native groups and 49 Python tests, including a new group
with 21 explicit numerical witnesses. There are now 104 native groups; the new
group passed separately from the 103-group full run. [Acceptance evidence](docs/LOCAL_ACCEPTANCE.md).

The [CUDA CI workflow](.github/workflows/cuda-compile.yml) separately checks that
shared-code changes still compile with NVIDIA dependencies. Successful compilation
is documented in the [compiler record](validation/cuda-compile-ci-13.4.txt);
it does **not** run CUDA kernels or validate MPS behavior.

## Performance

There is no general speedup promise. Existing M3 measurements include improvements,
CPU wins, and failed strict-equivalence checks. The
[published-workload replay notes](docs/PUBLISHED_BENCHMARKS.md) describe HIGGS and
Covertype settings, timing boundaries, hardware, and reproduction commands.
The latest full HIGGS replay covers the accepted `68c800e` production tree:
three pairs of 1,000 rounds give **288 s CPU / 469 s MPS** median training time
on this M3. MPS is slower, and strict prediction equivalence fails. The saved
MPS model predicts identically on CPU; the independently trained models use
different quantile cuts and differ from their first split. Holding cuts fixed
passes the strict comparison for a separate 100-round diagnostic, with all splits
agreeing. CUDA sketch agreement still needs NVIDIA execution.
System-wide swap use was high, and CPU times ranged from 195 to 289 seconds.
[Timings, correctness results and limits](docs/PUBLISHED_BENCHMARKS.md#accepted-production-tree-68c800e).
Older measurements are historical; raw reports remain under [validation/](validation/).

## Repository guide

- [Review](docs/REVIEW.md): concrete findings, risks, and cleanup decisions.
- [Caller audit](docs/CUDA_AUDIT.md): classified scope, two correctness findings, and ten implementation tasks.
- [Plan](docs/PLAN.md): prioritized work and acceptance criteria.
- [Implementation guide](docs/ARCHITECTURE.md): entry points, storage and module boundaries.
- [Parity ledger](docs/CUDA_PARITY.md): current behavior and known differences.
- [Operation map](docs/CUDA_OPERATION_MAP.md): CUDA sources, Metal entry points, and tests.
- [Historical progress log](docs/archive/README-progress-2026-11-04.md): archived
  implementation updates and old measurements; not current status.
- `xgboost/`: vendored upstream source and Metal implementation; `tests/` contains
  port-specific tests, `scripts/` build/test/benchmark tools, and `validation/`
  committed evidence. Generated builds, caches, and `results/` are ignored.

## License

Apache-2.0; see [LICENSE](LICENSE). Upstream XGBoost copyright notices and
third-party licenses are retained, including the fdlibm permission notice in
the Metal source.
