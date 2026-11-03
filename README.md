# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU training backend for XGBoost. Custom Metal compute
kernels build gradient histograms; XGBoost's existing CPU code handles quantization,
gradients, split selection, row partitioning, prediction, and model serialization.
This is a working **hybrid backend**, not full CUDA feature parity.

## Progress

- [x] Vendor upstream XGBoost 3.4.2 with its dependencies and licenses.
- [x] Add `USE_METAL=ON`, `device="mps"`, and real Metal histogram kernels.
- [x] Build the library and compile the shader on Apple Silicon.
- [x] Pass CPU/Metal training comparisons on an Apple M3.
- [x] Verify completed GPU dispatches; CPU fallback cannot satisfy a GPU test.
- [x] Test histogram kernels against an independent integer CPU reference.
- [x] Measure end-to-end performance against multithreaded CPU XGBoost.
- [x] Add a shared-memory histogram kernel for large dense inputs.
- [ ] Reduce CPU staging and synchronization overhead.
- [x] Test multi-target trees, external-memory pages, ranking, and extreme gradient scales.
- [ ] Expand randomized testing and validation across Apple GPU generations.
- [ ] Accelerate additional training and prediction stages.

**Current result:** correctness checks pass; the first implementation is slower
than the CPU baseline. No speedup is claimed.

## Build and use

Requires macOS, Xcode command-line tools, CMake, Ninja, Python, and OpenMP
(`brew install libomp` if needed).

```sh
uv venv .venv
uv pip install --python .venv/bin/python numpy scipy pytest scikit-learn pandas
bash scripts/build.sh
export PYTHONPATH="$PWD/xgboost/python-package"
```

```python
import xgboost as xgb

print(xgb.metal_info())  # device availability and completed GPU dispatch counters
model = xgb.XGBRegressor(device="mps", tree_method="hist")
model.fit(X, y)
```

The `mps` selector currently supports histogram tree training on one Metal device.
Explicit updater selection, `exact`/`approx` tree methods, linear boosters, and
distributed training are rejected. Training with an inaccessible GPU errors rather
than silently falling back. Prediction and metrics currently run on the CPU.

## Validation

```sh
bash scripts/test.sh
```

On Apple M3, 2026-11-03:

- Python: **50 passed, 1 skipped**. All 44 GPU tests executed and passed. The
  skipped test specifically requires a machine without an accessible GPU.
- Native shader: **28 cases passed**, comparing every histogram bin exactly
  against an independent CPU integer sum. Covers dense/sparse indices, 8/16/32-bit
  storage, nonzero page offsets, high contention, negative sums, and carries,
  for both global and shared-memory kernels.
- Training comparisons cover regression, binary/multiclass classification,
  missing values, CSR input, weights, categorical splits, quantile matrices,
  sampling, constraints, cache eviction, continuation, model interchange,
  multi-target trees, external memory, ranking, custom gradient scales, held-out
  sklearn predictions, and concurrent independent models.
- Numerical tolerances are explicit in the tests; general bitwise CPU/GPU
  equivalence is not promised.
- Selected upstream configuration, model, and serialization tests: **32 passed,
  1 skipped**. A separate `USE_METAL=OFF` build also compiles; its API reports
  Metal unavailable and rejects `device="mps"` with a clear build-option error.

## Performance measurements

Apple M3, 8 CPU threads, 32 dense features, depth 6, 20 boosting rounds, 256 bins.
Median of three runs; fresh DMatrix per run; includes matrix construction,
quantization, and training, with the Metal runtime warmed first. Synthetic data
generation is excluded. CPU and Metal predictions were identical in these runs.

| Rows | CPU | Metal | CPU / Metal |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.048 s | 0.122 s | 0.39x |
| 100,000 | 0.145 s | 0.255 s | 0.57x |
| 1,000,000 | 1.457 s | 2.518 s | 0.58x |

[Current raw measurements](validation/benchmark-v2-m3.json), and the
[first baseline](validation/benchmark-v1-m3.json). These are short local runs
with substantial timing variability; they do not establish a speedup from the
shared-memory optimization. Reproduce with:

```sh
.venv/bin/python scripts/benchmark.py --output results/my-benchmark.json
```

The script refuses to overwrite prior results. GPU histogram accumulation uses
scaled integers and two 32-bit atomics per 64-bit sum, avoiding native FP64 and
64-bit atomic requirements. Optimization and broader validation are ongoing.

## Development log

### 2026-11-03

Imported XGBoost 3.4.2, implemented the first Metal histogram backend, and added
execution counters and equivalence tests. Initial sandbox restrictions prevented
GPU access; after permissions were refreshed, the full GPU suite and native
shader tests passed. Recorded the first performance baseline above. The public
repository is maintained through normal Git commits. Added a shared-memory
histogram path, GPU command-duration counters, and broader correctness tests.
CPU/GPU predictions were identical in both benchmark runs, but Metal training
remains slower overall.

## Source and license

`xgboost/` contains upstream [XGBoost](https://github.com/dmlc/xgboost), version
[3.4.2](https://github.com/dmlc/xgboost/releases/tag/v3.4.2), imported from the official
source release. Upstream copyright notices and dependency licenses are retained.
Project code is Apache-2.0 licensed; see [LICENSE](LICENSE). This is an independent
experimental port.
