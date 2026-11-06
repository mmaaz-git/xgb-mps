# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU backend for XGBoost 3.4.2, using custom Metal compute
kernels behind `device="mps"`. This is an independent fork, not an upstream release.

**The core backend trains and predicts on Apple GPUs. Full XGBoost ecosystem
support and CUDA-equivalent numerical behavior are not yet established.**
The latest bounded review found no missing major core training/prediction family;
it did find unfinished binding integration, validation and maintainability work.
See the [review](docs/REVIEW.md) and [short current plan](docs/PLAN.md).

## Current scope

Tested hardware: **Apple M3, 10-core GPU, 16 GB RAM**.

| Area | Implemented paths |
| --- | --- |
| Tree training | `hist` and `approx`, scalar/vector trees, missing/categorical data, sampling, constraints and leaf fitting |
| Objectives and metrics | Regression, classification, ranking, survival and adaptive objectives |
| Prediction | Batch/inplace prediction, leaf outputs, exact SHAP contributions/interactions |
| Data | Ordinary/quantile matrices, native Metal tensors/columns, metadata, host/disk external-memory caches |
| Other training | Linear coordinate descent, distributed trees and optional federated gRPC trees |

The original audit's ten implementation tasks and two later capacity fixes have
local acceptance. That does not prove every branch or deployment works.
[Behavior, numerical and platform limits](docs/CUDA_PARITY.md).

**Remaining:** first-class Spark MPS GPU integration; standard R/JVM packaging
and runtime validation; representative NVIDIA comparisons; additional Apple
hardware and multi-host validation; and focused maintenance/performance work.
The core Python package can build/package Metal with explicit configuration.
Only Metal device ordinal 0 is supported, and native producer readiness is synchronous.

`tree_method="exact"` is unsupported on both CUDA and MPS. For linear GPU
training, use `booster="gblinear", updater="coord_descent"`; `shotgun` remains
on CPU, as with CUDA. Distributed `gblinear` is unsupported upstream.

## Build and try it

Requires an Apple GPU on macOS, Xcode command-line tools, Python **3.11+**, CMake,
Ninja and OpenMP. These commands build and install this fork into a local environment:

```sh
git clone https://github.com/mmaaz-git/xgb-mps.git
cd xgb-mps
brew install cmake ninja libomp
python3 -m venv .venv
bash scripts/build.sh
.venv/bin/python -m pip install './xgboost/python-package[scikit-learn,pandas]' pytest
```

Run with `.venv/bin/python`:

```python
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
X = rng.normal(size=(10000, 32)).astype(np.float32)
y = 2 * X[:, 0] - X[:, 1]

model = xgb.XGBRegressor(device="mps", tree_method="hist", n_estimators=20)
model.fit(X, y)
print(model.predict(X[:3]))
```

The installed package includes the built Metal library; no `PYTHONPATH` override
is needed. A fresh-environment check on this M3 exercised regression and
classification train/predict/save/load, CPU model interchange, and 77 existing
workflow/model tests. [Installation evidence](validation/python-install-m3.json).

Use `xgb.QuantileDMatrix(X, y, device="mps")` for Metal quantile construction.
PyTorch and MLX native inputs/outputs use DLPack interoperability; ordinary
`predict(DMatrix)` returns NumPy by default.

## Validation

```sh
brew install snappy  # Independent decoder for native tests.
bash scripts/test.sh --list
bash scripts/test.sh --python tests/test_metal_objectives.py
# Full standard suite, when appropriate:
bash scripts/test.sh --full
```

The runner builds the library first and supports focused selections. CPU-only
and federated configurations are separate checks. See [commands and exact
revision-specific results](docs/VALIDATION.md); a historical full suite is not
silently treated as validation of every later change.

CUDA CI compiles the NVIDIA backend and comparison programs on a CPU runner.
It does **not** execute CUDA kernels. [Advanced NVIDIA comparison instructions](docs/NVIDIA_COMPARISON.md)
are available, but unfinished replay modes are not missing MPS features.

## Performance

Performance is mixed. The latest full 1,000-round HIGGS measurement at `68c800e`
gave **288 s CPU / 469 s MPS** median training on this M3. Independently trained
CPU/MPS models have different quantile cuts. All saved MPS cuts match the
CUDA-source sketch equations; a separate fixed-cut training diagnostic passes.
A later reader optimization at `818e11f` improved a **100-round MPS-only**
comparison from **64.8 s to 55.2 s**, with identical models and predictions.
These are different experiments, not a current general CPU-speedup claim.
[Workloads, every timing and limitations](docs/PUBLISHED_BENCHMARKS.md).

## Repository guide

- [Plan](docs/PLAN.md): current scope and remaining work.
- [Review](docs/REVIEW.md): confirmed findings and cleanup decisions.
- [Behavior/limits](docs/CUDA_PARITY.md) and [architecture](docs/ARCHITECTURE.md).
- [Validation](docs/VALIDATION.md): checks and evidence boundaries.
- [Frozen audit](docs/CUDA_AUDIT.md) and [operation map](docs/CUDA_OPERATION_MAP.md).
- `xgboost/`: vendored source and Metal backend; `tests/` and `scripts/`: checks/tools;
  `validation/`: recorded evidence; `docs/archive/`: historical plans and progress.
  Builds, caches and `results/` are ignored.

## License

Apache-2.0; see [LICENSE](LICENSE). Upstream and third-party notices, including
fdlibm's permission notice in the Metal source, are retained.
