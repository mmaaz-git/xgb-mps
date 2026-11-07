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
platform skips, on `28908e1`. The latest memory/partition optimization passed
five native groups, 148 Python tests and three installed-package workflows.
Workflows include preprocessing, cross-validation, categorical classification, early stopping,
ranking and model persistence. [Validation commands/results](docs/VALIDATION.md)
record the revisions and scope. CUDA CI compiles the NVIDIA backend;
it does not run CUDA kernels.

## Performance

Median training times across three CPU/previous-MPS/new-MPS repeats on the
**10-core M3, 16 GB RAM**, using all eight CPU cores, 100 rounds, depth 6 and
256 bins:

| Training data | CPU | Previous MPS | New MPS |
| --- | ---: | ---: | ---: |
| 25K × 4,096 dense, synthetic | 39.00 s | 44.29 s | 44.48 s |
| HIGGS: 10.5M × 28 | 37.78 s | 37.13 s | 30.50 s |
| 100K × 32 categorical, synthetic | 2.09 s | 5.24 s | 5.09 s |

HIGGS improves **1.22×** over `048dc86`. A separate **single 1,000-round run per
variant** took 271.79 s before, 250.14 s after and 296.40 s on CPU
(1.18× CPU/new-MPS); it is not a three-repeat median.

Wide training is effectively unchanged, but measured Metal buffer allocation
falls by **256 MiB**. That does not establish a lower physical-memory peak.
HIGGS's faster partition cache adds about **279 MiB** net Metal allocation;
its eligibility and per-page size are bounded. CPU still wins the synthetic
controls. Other applications and system paging were not controlled.

Training includes quantization, cache construction and held-out evaluation each
round; loading, matrix creation and final prediction are separate. All before/after
MPS models, margins and metric histories match exactly. Independently trained
CPU/MPS models can differ, including a pre-existing wide-data accuracy difference.
[All repeats, memory measurements, quality and historical benchmarks](docs/PUBLISHED_BENCHMARKS.md#current-m3-results)
are recorded. These results do not establish a general GPU advantage.

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
