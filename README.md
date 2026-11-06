# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU backend for XGBoost 3.4.2, using custom Metal compute
kernels through `device="mps"`. This is an independent fork, not an upstream release.

**Python bindings only.** The shared C++ engine and its CPU/CUDA/Metal backends
remain, along with Python's Dask integration. R, Java/Scala and Spark bindings,
including `xgboost.spark`, have been removed and are outside the project scope.

## Build and try it

Requires an Apple GPU on macOS, Xcode command-line tools, Python **3.11+**, CMake,
Ninja and OpenMP. Local validation uses an **Apple M3, 10-core GPU, 16 GB RAM**.

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
is needed. Use `xgb.QuantileDMatrix(X, y, device="mps")` for Metal quantile
construction. PyTorch and MLX interoperability is described in the
[architecture guide](docs/ARCHITECTURE.md).

## Implemented scope and limits

| Area | Metal paths |
| --- | --- |
| Training | `hist`/`approx` trees, regression, classification, ranking, survival and linear coordinate descent |
| Prediction | Batch/inplace prediction, leaf outputs and exact SHAP contributions/interactions |
| Data | Ordinary/quantile matrices, missing/categorical data, native Metal inputs and external-memory caches |
| Distributed training | Tree training through Python; optional federated gRPC support |

Only Metal device ordinal 0 is supported. `tree_method="exact"` is unsupported;
linear GPU training requires `updater="coord_descent"`. Broader Apple hardware,
actual multi-host deployment and NVIDIA runtime comparisons need validation.
See [feature coverage and limits](docs/CUDA_PARITY.md) and the [current plan](docs/PLAN.md).

## Testing

```sh
brew install snappy  # Independent decoder used by native tests.
bash scripts/test.sh --list
bash scripts/test.sh --python tests/test_metal_objectives.py
bash scripts/test.sh --python tests/test_metal_workflows.py
```

The test runner builds first and supports focused checks or `--full`. The latest
full baseline passed 106 native groups and 4,001 Python tests, with two expected
platform skips. Workflows include preprocessing, cross-validation, categorical
classification, early stopping, ranking and model persistence. The subsequent
optimizations passed focused native/Python checks and installed-package
workflows; the latest build passed 300 Python tests, eight native groups and
three installed-package workflows. [Validation commands/results](docs/VALIDATION.md)
record the exact revisions and scope. CUDA CI compiles the NVIDIA backend;
it does not run CUDA kernels.

## Performance

The latest changes batch split-stage commands, select existing shared histogram
kernels for more wide workloads, and make conservative split screening cheaper.
Exact split scoring is unchanged. Fresh median timings across three paired
repeats, compared with the previous MPS build (`9919a31`):

| Training data | CPU | MPS before | MPS after | MPS improvement |
| --- | ---: | ---: | ---: | ---: |
| 100K rows × 1,024 columns, synthetic | 18.39 s | 16.16 s | 15.17 s | **1.06×** |
| 25K rows × 4,096 columns, synthetic | 35.61 s | 45.64 s | 31.41 s | **1.45×** |
| 10.5M rows × 28 columns, HIGGS | 33.69 s | 33.13 s | 31.24 s | **1.06×** |

Each run uses 100 rounds, depth 6, 256 bins and eight CPU threads. Synthetic data
has 64 informative features; no column sampling is used. All 18 before/after MPS
models, predictions and metric histories match exactly.

All six synthetic before/after pairs improved; one HIGGS pair was slightly slower.
CPU won two of the three 4,096-column pairs despite its higher median, so these
runs do not establish a general GPU advantage. User applications and system-wide
swap affect timing stability. Use whole-training timings for this comparison: command
batching changed the scope of the split GPU counter.
See [current results, all repeats and limitations](docs/PUBLISHED_BENCHMARKS.md#current-m3-results);
earlier experiments remain below that section.

Large-workload profiling is separate from the tests:

```sh
.venv/bin/python scripts/profile_training.py --case wide --rows 100000 \
  --features 1024 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide-1024
```

The profiler runs fresh CPU/MPS processes and records setup/training times,
GPU counters, held-out quality, memory pressure and per-round segments. HIGGS
uses `--case higgs` with the cache from `scripts/prepare_published_data.py`.
The optional `--informative-features` spreads signal across the feature matrix;
omitting it retains the original three-feature target. Independently trained
CPU/MPS models can differ; these runs record the differences rather than claim
training equivalence. Each MPS model's CPU prediction is checked exactly.

## Repository guide

`xgboost/` contains the vendored engine and Python package; `tests/` and `scripts/`
contain checks and tools. `validation/` holds recorded evidence, and `docs/archive/`
preserves historical plans. Start with the [plan](docs/PLAN.md),
[review](docs/REVIEW.md), [architecture](docs/ARCHITECTURE.md) and [validation guide](docs/VALIDATION.md).

## License

Apache-2.0; see [LICENSE](LICENSE). Upstream and third-party notices, including
fdlibm's permission notice in the Metal source, are retained.
