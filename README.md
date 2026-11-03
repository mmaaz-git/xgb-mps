# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU backend for XGBoost 3.4.2, using custom Metal compute
kernels behind `device="mps"`. The goal is coverage of all CUDA-accelerated XGBoost
operations, with measured performance. **Full CUDA parity is not implemented yet.**

The current backend runs histograms, gradient quantization, several objectives,
prediction transforms, tree traversal, prediction-cache updates, common metrics,
row partitioning and scalar/vector split evaluation on Metal. Quantile construction,
sampling, histogram subtraction and several specialized features still run on the CPU. See the [coverage ledger](docs/CUDA_PARITY.md)
and [105-file CUDA inventory](validation/cuda-inventory.json).

**Measured on Apple M3:** large-batch prediction is faster than CPU; training is
still slower. The latest million-row prediction benchmark is **2.51× faster**,
including input staging and transfers. This does not establish CUDA-equivalent
training performance or speedups on other hardware and workloads.

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

print(xgb.metal_info())
model = xgb.XGBRegressor(device="mps", tree_method="hist")
model.fit(X, y)
prediction = model.predict(X)
```

Explicit updater selection, `exact`/`approx` tree methods, linear boosters, and
distributed training are rejected for `mps`. Inaccessible Metal devices produce an
error. Other features may use CPU implementations as recorded in the coverage ledger;
selecting `mps` does not imply that every stage runs on the GPU.

## Current Metal operations

| Operation | Implemented coverage |
| --- | --- |
| Histograms | Dense/sparse compressed indices, external-memory pages, scaled integer sums, shared-memory dense kernel |
| Split evaluation | Numeric and categorical splits, missing directions, scalar/vector leaves, regularization, monotonic bounds; CPU keeps column sampling and interaction masks |
| Row partitioning | Stable GPU partitions, numeric/categorical/missing values, GPU leaf assignment, resident row indices |
| Quantized data cache | Page-owned GPU bins reused across trees; dense indices transposed on the GPU |
| Gradient quantization | GPU integer bit conversion with CPU-selected scales; exact comparison against CPU double `llround` |
| Gradients | Squared error, squared log error, logistic/logitraw, gamma, pseudo-Huber, Poisson, Tweedie, hinge, multiclass, expectile |
| Prediction transforms | Sigmoid, exponential, hinge, softmax, argmax, cumulative expectile |
| Tree traversal | Batch/inplace/leaf prediction; numeric/categorical splits, missing values, scalar/vector leaves, tree ranges, dropout weights |
| Training prediction cache | Leaf-value accumulation, including strided multi-target output and sampled rows |
| Metrics | RMSE, RMSLE, MAE, MAPE, logloss, error thresholds, Poisson, gamma, Tweedie, pseudo-Huber, multiclass logloss/error |

Input views and categorical recoding remain on the host. GPU metric kernels produce
scaled block sums, with the final small reduction in host FP64. Metal has no native
FP64 arithmetic. Split selection uses software binary64 operations and checked exact
integer scans to preserve histogram precision and CPU tie-breaking. Tolerances and
numerical edge cases are tested explicitly.

`xgb.metal_info()` reports completed dispatch counters for each operation family.
It also reports GPU durations and histogram staging/wait/readback wall times. Wait
time overlaps GPU duration; adding those numbers would double-count time. Counters
are process-wide, so concurrent models contribute to the same totals.

## Validation

```sh
bash scripts/test.sh
```

On Apple M3, 2026-11-03:

- Python: **231 passed, 1 skipped**. The skip requires a machine without an
  accessible GPU. Positive GPU tests assert completed operation counters so CPU
  fallback cannot satisfy them.
- Native histograms: **28 cases**, with every bin compared exactly to independent
  CPU integer sums. Covers dense/sparse indices, 8/16/32-bit storage, page offsets,
  high contention, negative values and carries.
- Native gradient quantization: **713,814 values across six scales**, including
  subnormal floats, compared exactly against double-precision CPU rounding.
- Native partition/transpose kernels: **256 cases** against CPU stable partitions, including empty nodes, unselected rows, categorical/missing values, block boundaries, page offsets and all index widths.
- Native split selection: **3,780 scalar and 2,160 vector cases**, comparing selected
  splits, gains, child statistics and categorical ordering exactly against independent
  CPU references. Includes wide exponent ranges, ties, missing data, regularization,
  monotonic bounds, pooled vector weights and threadgroup boundaries.
- Software binary64: **400,784 arithmetic operations, 200,392 narrowing conversions
  and 100,007 widening conversions** checked against CPU hardware arithmetic.
- Native objectives: **20 cases / 131,968 gradient pairs** compared against
  independent CPU mathematical references.
- Integrated comparisons cover weighted/multi-target training, sampling,
  constraints, categorical recoding, external memory, model interchange,
  continuation, concurrent models, prediction chunks and base margins.
- Metric tests include zero weights, weights from `1e-30` to `1e37`, parameterized
  thresholds, multiclass reduction and log-gamma boundaries.
- General bitwise CPU/GPU equivalence is not promised. Prediction differences in
  the benchmark runs below were zero.

A separate `USE_METAL=OFF` build passes its availability/error checks. Selected
upstream configuration/model/I/O tests pass: **32 passed, 1 skipped**. Build outputs
and full logs are local under `results/`; benchmark JSON is committed under
`validation/`.

## Performance

Apple M3, 8 CPU threads, 32 dense features; medians of three runs with alternating
CPU/Metal order. Runtime compilation is warmed before timing. Synthetic data
generation is excluded. These are local measurements with timing variability.

### Training

20 boosting rounds, depth 6, 256 bins. Includes fresh DMatrix construction,
quantization, gradients, histograms and tree construction.

| Rows | CPU | Metal | CPU / Metal |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.043 s | 0.255 s | 0.17× |
| 100,000 | 0.115 s | 0.330 s | 0.35× |
| 1,000,000 | 0.816 s | 1.158 s | 0.70× |

[Latest raw training results](validation/benchmark-split-scan-m3.json). Metal training
is still slower than CPU. GPU split evaluation initially regressed million-row
training to 2.209 s. Parallel candidate scoring and checked exact integer scans
reduced this to 1.158 s; measured split-kernel time fell from 0.857 s to 0.061 s.
The [serial evaluator](validation/benchmark-split-fast-m3.json),
[parallel evaluator](validation/benchmark-split-parallel-m3.json) and latest results
are retained. These separately timed runs are subject to system variability.

The preceding hybrid version with CPU split selection measured 1.063 s Metal versus
0.824 s CPU ([raw results](validation/benchmark-transpose-v2-m3.json)). Moving a stage
to the GPU has not automatically improved training time. Histogram staging,
subtraction, quantile construction, sampling and per-level synchronization remain
optimization targets. Small workloads pay proportionately more dispatch overhead.

### Prediction

300 depth-6 trees trained on CPU; identical model and prediction input for both
backends. Uncached `inplace_predict`, including host staging and GPU transfers.
Every Metal repetition must execute GPU traversal; prediction-cache hits cannot
inflate the benchmark.

| Rows | CPU | Metal | CPU / Metal |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.00062 s | 0.00426 s | 0.14× |
| 100,000 | 0.0449 s | 0.0248 s | 1.81× |
| 1,000,000 | 0.416 s | 0.165 s | **2.51×** |

[Latest raw prediction results](validation/benchmark-prediction-v2-m3.json).
The [first run](validation/benchmark-prediction-v1-m3.json) measured 3.42× at one
million rows. Both runs are retained; neither establishes a universal speedup.
Small batches are slower because staging and dispatch overhead dominate.

```sh
.venv/bin/python scripts/benchmark.py --threads 8 --output results/my-training.json
.venv/bin/python scripts/benchmark_prediction.py --output results/my-prediction.json
```

Both scripts refuse to overwrite prior measurements. Earlier training runs remain
in `validation/benchmark-v1-m3.json` through `benchmark-v4-m3.json` for inspection.

## Progress

- [x] Vendor upstream 3.4.2, build Metal and CPU-only configurations.
- [x] Implement/test histogram kernels and shared-memory accumulation.
- [x] Move gradient quantization onto the GPU; reduce host row staging.
- [x] Port the objective and transform families listed above.
- [x] Port batch/inplace/leaf traversal and training prediction-cache updates.
- [x] Port common elementwise/multiclass metrics with bounded GPU reductions.
- [x] Measure large-batch prediction gains and retain training regressions publicly.
- [ ] Keep training data and intermediate state resident on Metal buffers.
- [x] Port stable row partitioning and leaf assignment; keep row indices on the GPU.
- [x] Port scalar/vector split evaluation, including categorical ordering and constraints.
- [ ] Port sampling and histogram subtraction; remove histogram round trips.
- [ ] Port quantile construction, remaining objectives/metrics, SHAP and linear training.
- [ ] Add Apple-device array interoperability and distributed GPU collectives.
- [ ] Validate across Apple GPU generations and broader real datasets.
- [ ] Reach complete CUDA-operation coverage and competitive training performance.

## Development log

### 2026-11-03

Imported XGBoost 3.4.2 and built the initial hybrid histogram backend. GPU tests
became runnable after execution permissions were refreshed. Added native integer
reference tests, shared-memory histograms and end-to-end benchmarks.

Expanded the scope to all CUDA operations, inventoried 105 CUDA source/header files,
and recorded remaining CPU paths. Added GPU gradient quantization, objective kernels,
transforms, tree prediction, prediction-cache updates and common metrics. Removed
per-row histogram work-item construction, reused histogram buffers and command queues,
and added per-operation profiling. Large-batch prediction now shows measured CPU
speedups; training and full parity remain open work.

## Source and license

`xgboost/` contains upstream [XGBoost](https://github.com/dmlc/xgboost), version
[3.4.2](https://github.com/dmlc/xgboost/releases/tag/v3.4.2), imported from the official
source release. Upstream copyright notices and dependency licenses are retained.
Project code is Apache-2.0 licensed; see [LICENSE](LICENSE). This is an independent
experimental port.

### Training residency update

Added stable Metal row partitioning and GPU leaf assignment. Histogram kernels
read resident row indices directly. Quantized pages own their GPU copies, with
invalidation when source data is rebuilt/read; dense indices are transposed on the
GPU for feature-wise access. The [first residency measurement](validation/benchmark-resident-v1-m3.json)
and [transposed-layout measurement](validation/benchmark-transpose-v2-m3.json) are
retained. The latter measured 1.063 s Metal versus 0.824 s CPU at one million rows
(0.78×). These measurements predate the GPU split evaluator.

### Split evaluation update

Added GPU scalar and vector split selection, stable categorical ordering, missing
directions, L1/L2 regularization and monotonic bounds. Binary64 emulation preserves
the precision used by CPU scoring; bounded integer prefix scans accelerate cases
where parallel sums are provably exact. Winning candidates retain CPU tie order.
All 231 Python tests and 5,940 native split cases pass. The training table above
includes the new evaluator and its remaining performance cost.
