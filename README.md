# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU backend for XGBoost 3.4.2, using custom Metal compute
kernels behind `device="mps"`. The goal is coverage of all CUDA-accelerated XGBoost
operations, with measured performance. **Full CUDA parity is not implemented yet.**

The current backend runs histograms, gradient quantization, several objectives,
prediction transforms, tree traversal, prediction-cache updates, common metrics,
row partitioning and scalar/vector split evaluation on Metal. Quantile construction,
gradient-based sampling and several specialized features still run on the CPU. Histogram conversion,
subtraction and split-input gathering now execute on Metal-owned shared buffers. See the [coverage ledger](docs/CUDA_PARITY.md)
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
| Histograms | Dense/sparse compressed indices, external-memory pages, scaled integer sums, shared-memory dense kernel, GPU binary64 conversion/subtraction and a resident cache |
| Split evaluation | Numeric and categorical splits, missing directions, scalar/vector leaves, regularization, monotonic bounds; CPU keeps column sampling and interaction masks |
| Row partitioning | Stable GPU partitions, numeric/categorical/missing values, GPU leaf assignment, resident row indices |
| Quantized data cache | Page-owned GPU bins reused across trees; dense indices transposed on the GPU |
| Uniform sampling | GPU LCG skip-ahead and Bernoulli masking, both matrix layouts, replay for reduced-gradient leaf fitting; host chooses the per-tree seed |
| Gradient quantization | GPU integer bit conversion with CPU-selected scales; exact comparison against CPU double `llround` |
| Gradients | Squared error, squared log error, logistic/logitraw, gamma, pseudo-Huber, Poisson, Tweedie, hinge, multiclass, expectile; adaptive absolute-error and quantile gradients with GPU residual-scale reductions |
| Prediction transforms | Sigmoid, exponential, hinge, softmax, argmax, cumulative expectile, quantile row sorting |
| Tree traversal | Batch/inplace/leaf prediction; numeric/categorical splits, missing values, scalar/vector leaves, tree ranges, dropout weights |
| Training prediction cache | Leaf-value accumulation, including strided multi-target output and sampled rows |
| Metrics | RMSE, RMSLE, MAE, MAPE, logloss, error thresholds, Poisson, gamma, Tweedie, pseudo-Huber, multiclass logloss/error, quantile and expectile |

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

- Python: **309 passed, 1 skipped**. The skip requires a machine without an
  accessible GPU. Positive GPU tests assert completed operation counters so CPU
  fallback cannot satisfy them.
- Native histograms: **84 cases**, with every bin compared exactly to independent
  CPU integer sums. Covers dense/sparse indices, 8/16/32-bit storage, page offsets,
  high contention, negative values, carries, zero words and sampled-out rows.
- Native sampling: **3,780 exact cases / 1,019,088 Bernoulli draws** against the
  CPU standard-library implementation, including zero seeds, rounding boundaries,
  strided matrices, guard regions and CPU thread partitions.
- Native gradient quantization: **713,814 values across six scales**, including
  subnormal floats, compared exactly against double-precision CPU rounding.
- Native partition/transpose kernels: **256 cases** against CPU stable partitions, including empty nodes, unselected rows, categorical/missing values, block boundaries, page offsets and all index widths.
- Native histogram storage: **168 cases**, with 55,272 converted/accumulated pairs,
  18,424 subtracted pairs and 82,908 gathered pairs checked against CPU references,
  including signed integer limits, page accumulation, strided outputs and guard regions.
- Native split selection: **3,780 scalar and 2,160 vector cases**, comparing selected
  splits, gains, child statistics and categorical ordering exactly against independent
  CPU references. Includes wide exponent ranges, ties, missing data, regularization,
  monotonic bounds, pooled vector weights and threadgroup boundaries.
- Software binary64: **400,784 arithmetic operations, 200,392 narrowing conversions
  and 100,007 widening conversions** checked against CPU hardware arithmetic.
- Native objectives: **20 cases / 131,968 gradient pairs** compared against
  independent CPU mathematical references.
- Native adaptive objectives: **180 cases / 1,260 residual scales / 452,592 gradient
  pairs**, covering zero/tiny weights, multiple targets and response scales.
- Native quantile/expectile metrics: **288 cases / 1,727,040 loss/weight pairs**,
  including multi-label layouts, extreme weights and FP32 subnormals. Rare
  underflowing metric products use software arithmetic to preserve CPU results.
- Integrated comparisons cover weighted/multi-target training, sampling,
  constraints, categorical recoding, external memory, model interchange,
  continuation, concurrent models, prediction chunks and base margins.
- Metric tests include zero weights, weights from `1e-30` to `1e37`, parameterized
  thresholds, multiclass reduction and log-gamma boundaries.
- General bitwise CPU/GPU equivalence is not promised. Prediction differences in
  the squared-error training and traversal benchmarks were zero. The adaptive
  objective benchmarks differed by at most `4.77e-7`.

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
| 100,000 | 0.114 s | 0.385 s | 0.30× |
| 1,000,000 | 0.794 s | 1.183 s | 0.67× |

[Latest raw training results](validation/benchmark-hist-skip-zero-v1-m3.json), including
resident histograms, GPU subtraction/gathering, reused split buffers and skipped
zero atomic additions. The [preceding run](validation/benchmark-hist-pool-m3.json)
and [first resident-histogram run](validation/benchmark-hist-storage-m3.json) are
retained. Dispatch overhead and substantial GPU timing variability remain.
Metal training is still slower than CPU.

Before histogram residency, [split-evaluator optimization](validation/benchmark-split-scan-m3.json)
measured 1.158 s at one million rows. GPU split evaluation initially regressed million-row
training to 2.209 s. Parallel candidate scoring and checked exact integer scans
reduced this to 1.158 s; measured split-kernel time fell from 0.857 s to 0.061 s.
The [serial evaluator](validation/benchmark-split-fast-m3.json),
[parallel evaluator](validation/benchmark-split-parallel-m3.json) and exact-scan results
are retained. These separately timed runs are subject to system variability.

The preceding hybrid version with CPU split selection measured 1.063 s Metal versus
0.824 s CPU ([raw results](validation/benchmark-transpose-v2-m3.json)). Moving a stage
to the GPU has not automatically improved training time. Quantile construction, sampling, kernel scheduling and per-level synchronization
remain optimization targets. Histogram arithmetic now stays on the GPU, but
adding GPU operations still has dispatch costs. Small workloads pay proportionately more dispatch overhead.

A wider **128-feature** squared-error workload was also measured with 20 rounds:
1,000,000 rows took **3.839 s Metal versus 3.254 s CPU** (0.85×), and 100,000 rows
took 0.850 versus 0.420 s (0.49×). Predictions matched exactly.
[Raw wider-workload results](validation/benchmark-wide-v1-m3.json) are retained;
this workload also remains below CPU training speed.

### Adaptive objectives

Same hardware, 32 features and three alternating repetitions; **10** boosting
rounds rather than the 20 used above. Includes DMatrix construction and all
training work; quantile uses `alpha=0.5`. Neither objective shows an end-to-end
training speedup yet.

| Objective / rows | CPU | Metal | CPU / Metal |
| --- | ---: | ---: | ---: |
| Absolute error / 100,000 | 0.097 s | 0.267 s | 0.36× |
| Absolute error / 1,000,000 | 0.747 s | 0.882 s | 0.85× |
| Quantile / 100,000 | 0.105 s | 0.211 s | 0.49× |
| Quantile / 1,000,000 | 0.892 s | 1.108 s | 0.81× |

[Absolute-error results](validation/benchmark-absolute-v2-m3.json) and
[quantile results](validation/benchmark-quantile-v2-m3.json). Checked exact integer
reductions accelerate common residual scales and metric sums, with software
binary64 retained for wide exponent ranges. At one million rows, objective GPU
time fell from 0.083 to 0.052 s for absolute error and from 0.068 to 0.055 s for
quantile; end-to-end Metal training fell from 0.999 to 0.882 s and from 1.164 to
1.108 s respectively. The [first absolute-error](validation/benchmark-absolute-v1-m3.json)
and [first quantile](validation/benchmark-quantile-v1-m3.json) runs remain available.
These are separate runs, subject to system timing variability.

Base-score estimation remains on CPU; per-iteration residual-scale reduction and
gradient calculation execute on Metal. These implement the adaptive losses in the
vendored source.

### Uniform sampling

With 50% uniform row sampling, 20 rounds and otherwise the same squared-error
benchmark settings, one million rows took **1.191 s Metal versus 0.817 s CPU**
(0.69×); 100,000 rows took 0.294 s versus 0.118 s (0.40×). Predictions matched
exactly. [Raw sampled-training results](validation/benchmark-sampling-v2-m3.json)
include the completed sampling dispatches and GPU duration. This port adds GPU
coverage but has not established a training speedup.

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
- [x] Keep histograms in Metal shared memory; port conversion, subtraction and gathering.
- [x] Reuse split workspaces across tree levels and boosting rounds.
- [x] Port uniform sampling, including random-stream compatibility and replay.
- [ ] Port gradient-based sampling and remaining training-state updates.
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

### Histogram residency update

Histogram storage now uses Metal-owned shared allocations. GPU kernels perform
integer-to-binary64 conversion, external-page accumulation, sibling subtraction
and feature gathering. Cache growth uses GPU buffer copies, and scalar/vector
split workspaces are reused. Host code maintains cache metadata and the model;
it can inspect completed shared results without copying the entire histogram.
Checks cover cache sizes 1/2/7/64, both growth policies, both output strategies
and external-memory pages. All 251 Python tests and the native suites pass.
The latest timing regression is retained in the performance table above.

### Adaptive loss update

Ported the residual-scale reductions and gradients for absolute-error and quantile
regression, quantile prediction sorting, and quantile/expectile metrics. Residual
scales stay on the GPU; metric kernels return small binary64 block sums for final
host reduction. Added explicit subnormal preservation after a native test exposed
Metal's FP32 underflow behavior. All 291 Python tests and native suites pass;
CPU-only compatibility checks pass. The new objective benchmarks remain slower
than CPU and are published above.

### Uniform sampling update

Moved uniform Bernoulli sampling to Metal with LCG skip-ahead. Preserved the CPU
random sequence, its floating-point conversion, zero-seed behavior and replay for
reduced-gradient leaf fitting. All 309 Python tests and the native suites pass;
the CPU-only build still passes compatibility checks. Gradient-based sampling,
column sampling and host gradient sums remain open work.

### Histogram atomic update

Histogram kernels skip zero integer words and rows with zero gradient and Hessian,
while preserving exact two-word atomic sums. Expanded the native histogram suite
to 84 cases; all 309 Python tests still pass. Sampled million-row training changed
from [1.498 s](validation/benchmark-sampling-v1-m3.json) to
[1.191 s](validation/benchmark-sampling-v2-m3.json) across these runs; CPU remains
faster. Both measurements are retained rather than treating timing variability as
a guaranteed improvement.
