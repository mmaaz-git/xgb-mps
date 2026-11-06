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
```

The test runner builds first and supports focused checks or `--full`. Fresh-wheel
validation passed 77 workflow/model-state tests, with one expected skip, plus
regression/classification train/predict/save/load and CPU model interchange.
[Packaging evidence](validation/python-only-m3.json) and [validation commands/results](docs/VALIDATION.md)
record the tested revisions. CUDA CI compiles the NVIDIA backend; it does not run CUDA kernels.

## Performance

Speedups depend on the workload. The recorded 1,000-round HIGGS benchmark at
`68c800e` took **288 s CPU / 469 s MPS** median training on this M3. A later
objective-kernel optimization improved a focused measurement, but a general CPU
speedup is not established. See [complete timings and limitations](docs/PUBLISHED_BENCHMARKS.md).

## Repository guide

`xgboost/` contains the vendored engine and Python package; `tests/` and `scripts/`
contain checks and tools. `validation/` holds recorded evidence, and `docs/archive/`
preserves historical plans. Start with the [plan](docs/PLAN.md),
[review](docs/REVIEW.md), [architecture](docs/ARCHITECTURE.md) and [validation guide](docs/VALIDATION.md).

## License

Apache-2.0; see [LICENSE](LICENSE). Upstream and third-party notices, including
fdlibm's permission notice in the Metal source, are retained.
