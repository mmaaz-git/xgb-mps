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
full run passed **106 native groups and 4,001 Python tests**, with two expected
platform skips, on `28908e1`. The subsequent optimization passed seven native
groups, 427 Python tests and three installed-package workflows. Workflows include
preprocessing, cross-validation, categorical classification, early stopping,
ranking and model persistence. [Validation commands/results](docs/VALIDATION.md)
record the revisions and scope. CUDA CI compiles the NVIDIA backend;
it does not run CUDA kernels.

## Performance

Fresh median training times across three CPU/MPS repeats on the **10-core M3,
16 GB RAM**, using all eight CPU cores, depth 6 and 256 bins:

| Training data | Rounds | CPU | MPS |
| --- | ---: | ---: | ---: |
| 100K × 32 categorical, synthetic | 100 | 1.48 s | 5.12 s |
| 100K × 128, 20% missing, synthetic | 100 | 1.78 s | 5.20 s |
| 25K × 4,096 dense, synthetic | 100 | 44.82 s | 62.44 s |
| Covertype: 464,809 × 54 | 500 | 72.54 s | 142.96 s |
| HIGGS: 10.5M × 28 | 1,000 | 389.05 s | 308.96 s |

Training includes quantization and held-out evaluation every round. Data loading,
matrix construction and final prediction are separate. Categorical data has 256
values per feature and 16 informative columns; the other synthetic cases have
64 informative columns. Covertype uses seven classes and a stratified 80/20 split.

The latest optimization replaces software-double conversions used for FP32
ordering with exact bit comparisons and removes redundant input-staging writes.
Categorical MPS training improves **9.19 → 5.12 s (1.80×)** against `28908e1`.
The missing-value case is effectively flat; the wide median is slower
(**58.73 → 62.44 s**) amid substantial paging variation. CPU remains faster on
these three synthetic cases. All before/after MPS models, predictions and metric
histories match exactly; independently trained CPU/MPS models can differ.

[All repeats, quality checks, setup times and limitations](docs/PUBLISHED_BENCHMARKS.md#current-m3-results)
are recorded. Other applications were not controlled; system-wide paging affected
the wide runs. These measurements do not establish a general GPU advantage.

Large-workload profiling stays separate from tests. Prepare the real-data caches
with `scripts/prepare_published_data.py`; then, for example:

```sh
.venv/bin/python scripts/profile_training.py --case wide --rows 25000 \
  --features 4096 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide-4096
.venv/bin/python scripts/profile_training.py --case covtype --rounds 500 \
  --threads 8 --repeats 3 --output results/profile-covtype
```

The profiler uses fresh processes, checks each MPS model's CPU predictions exactly,
and records operation counters, held-out quality, memory and power observations.
Use `--case categorical` or `--missing-fraction 0.2` for the additional synthetic
workloads; the detailed results provide every command.

## Repository guide

`xgboost/` contains the vendored engine and Python package; `tests/` and `scripts/`
contain checks and tools. `validation/` holds recorded evidence, and `docs/archive/`
preserves historical plans. Start with the [plan](docs/PLAN.md),
[review](docs/REVIEW.md), [architecture](docs/ARCHITECTURE.md) and [validation guide](docs/VALIDATION.md).

## License

Apache-2.0; see [LICENSE](LICENSE). Upstream and third-party notices, including
fdlibm's permission notice in the Metal source, are retained.
