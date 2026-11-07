# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU backend for XGBoost 3.4.2, using custom Metal compute
kernels through `device="mps"`. This is an independent fork, not an upstream release.

**Python bindings only.** The shared C++ engine and CPU/CUDA/Metal backends remain,
along with Python's Dask integration. R, Java/Scala and Spark bindings, including
`xgboost.spark`, have been removed.

## Build and try it

Requires an Apple GPU on macOS, Xcode command-line tools and Python **3.11+**.

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

The installed package includes the Metal library; no `PYTHONPATH` override is
needed. Use `xgb.QuantileDMatrix(X, y, device="mps")` for Metal quantile construction.
See [native PyTorch/MLX inputs and outputs](docs/ARCHITECTURE.md#python-interoperability).

## Support

Metal paths cover hist/approx tree training, regression/classification/ranking/
survival objectives, coordinate-descent linear training, prediction, exact SHAP,
missing/categorical data, external memory and distributed trees through Python.

Only device ordinal 0 is supported. `tree_method="exact"` is unsupported; linear
GPU training requires `updater="coord_descent"`. NVIDIA runtime comparisons, other
Apple GPUs and actual multi-host deployment remain unvalidated. See
[coverage, numerical limits and remaining work](docs/CUDA_PARITY.md).

## Tests

```sh
brew install snappy  # Independent decoder used by native tests.
bash scripts/test.sh --list
bash scripts/test.sh --python tests/test_metal_workflows.py
bash scripts/test.sh --full
```

The latest backend (`614f15a`) passed **106 native groups and 4,003 Python tests**,
with two expected skips, plus three installed-wheel workflows outside the checkout.
The runner builds first. [Validation](docs/VALIDATION.md) describes dependencies,
focused checks and evidence. CUDA CI compiles; it does not execute NVIDIA kernels.

## Performance

Three-repeat medians on an **Apple M3, 10-core GPU, 16 GB RAM**, using eight CPU
threads and 100 training rounds. Rerun on November 7, 2026, with backend `614f15a`:

| Training data | CPU median | MPS median | CPU / MPS |
| --- | ---: | ---: | ---: |
| 125K × 1,024, dense | 23.52 s | 15.23 s | 1.54× |
| 25K × 4,096, dense | 39.65 s | 29.45 s | 1.35× |
| HIGGS, 10.5M × 28 | 38.33 s | 34.25 s | 1.12× |
| 1M × 128, dense | 9.61 s | 9.03 s | 1.06× |
| 25K × 1,024, 20% missing | 15.88 s | 26.40 s | 0.60× |

Ratios above 1 favor MPS. CPU wins on smaller cases, missing-value cases and
Covertype in this survey. Timings include quantization and held-out evaluation
after runtime warmup; matrix setup is reported separately. Independently trained
CPU/MPS models can differ. Other applications and system paging were not controlled.
[All 14 workloads and reproduction commands](docs/PUBLISHED_BENCHMARKS.md)
include every repeat, quality comparisons, full-length HIGGS/Covertype runs and
physical-memory measurements.

## Repository guide

`xgboost/` contains the engine and Python package; `tests/` and `scripts/` contain
checks and profiling tools. `validation/` holds recorded evidence. Start with
[coverage](docs/CUDA_PARITY.md), [architecture](docs/ARCHITECTURE.md),
[validation](docs/VALIDATION.md) or [benchmarks](docs/PUBLISHED_BENCHMARKS.md).
Earlier progress logs and reviews are available in Git history.

## License

Apache-2.0; see [LICENSE](LICENSE). Upstream and third-party notices, including
fdlibm's permission notice in the Metal source, are retained.
