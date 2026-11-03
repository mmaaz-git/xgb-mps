# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU backend for XGBoost 3.4.2, using custom Metal compute
kernels behind `device="mps"`. The goal is coverage of all CUDA-accelerated XGBoost
operations, with measured performance. **Full CUDA parity is not implemented yet.**

The current backend runs histograms, gradient quantization, several objectives,
prediction transforms, tree traversal, prediction-cache updates, common metrics,
row partitioning, scalar/vector split evaluation, root-gradient sums and stump
fitting, means, stable column sorting and quantile/median estimation on Metal. Feature quantile sketching,
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
| Bin assignment | GPU numeric/categorical cut search and hit counts for SparsePage-backed matrices; dense 8/16/32-bit compression; quantile sketches and adapter-built QuantileDMatrix bins remain CPU |
| Quantized data cache | Page-owned GPU bins reused across trees; dense indices transposed on the GPU |
| Uniform sampling | GPU LCG skip-ahead and Bernoulli masking, both matrix layouts, replay for reduced-gradient leaf fitting; host chooses the per-tree seed |
| Gradient quantization | GPU maximum/finite reduction, exact integer scale selection and bit conversion; scales stay on GPU |
| Root statistics / stump fitting | GPU scalar/vector gradient sums, dense root histogram sums and unregularized stump weights; other leaf fitting still uses CPU |
| Generic sums | GPU float-to-binary64 reductions and optional-weight sums; scalar default weights need no array dispatch |
| Means | GPU scalar, column-wise and weighted means; binary64 reductions and weight sums; other objective-specific initialization work remains CPU |
| Quantiles / medians | GPU stable radix sorting, integer prefix scans, CPU-compatible interpolation and sequential FP32 weighted CDFs; multiple quantile levels share one sort |
| Gradients | Squared error, squared log error, logistic/logitraw, gamma, pseudo-Huber, Poisson, Tweedie, hinge, multiclass, expectile; adaptive absolute-error and quantile gradients with GPU residual-scale reductions |
| AFT survival | GPU normal/logistic/extreme gradients, prediction/base-score transforms, likelihood and interval-accuracy metrics; numerically sensitive censored gradient rows are corrected on CPU and counted explicitly |
| Prediction transforms | Sigmoid, exponential, hinge, softmax, argmax, cumulative expectile, quantile row sorting |
| Tree traversal | Batch/inplace/leaf prediction; numeric/categorical splits, missing values, scalar/vector leaves, tree ranges, dropout weights |
| Training prediction cache | Leaf-value accumulation, including strided multi-target output and sampled rows |
| Metrics | RMSE, RMSLE, MAE, MAPE, logloss, error thresholds, Poisson, gamma, Tweedie, pseudo-Huber, multiclass logloss/error, quantile and expectile |

AFT gradient evaluation is still hybrid for numerically sensitive censored rows.
The GPU flags interval probabilities at or below `1e-5`; the host recomputes those
rows with the original CPU formula. `survival_cpu_corrections` and
`survival_cpu_time_ns` expose this work. Ordinary and uncensored rows stay on GPU;
this is not complete AFT GPU coverage. AFT metrics and transforms execute on GPU.

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

- Python: **421 passed, 1 skipped**. The skip requires a machine without an
  accessible GPU. Positive GPU tests assert completed operation counters so CPU
  fallback cannot satisfy them.
- Native histograms: **192 cases**, with every bin compared exactly to independent
  CPU integer sums. Covers dense/sparse indices, 8/16/32-bit storage, page offsets,
  high contention, negative values, carries, zero words, sampled-out rows and
  workgroup sizes of 1/8/32/64 rows per lane.
- Native sampling: **3,780 exact cases / 1,019,088 Bernoulli draws** against the
  CPU standard-library implementation, including zero seeds, rounding boundaries,
  strided matrices, guard regions and CPU thread partitions.
- Native bin assignment: **168 exact cases / 216,600 entries / 38,616,312 hit
  counts**, including all index widths, category truncation, subnormal/signed-zero
  boundaries, invalid values and guard regions.
- Native gradient bounds: **63 reductions / 1,015,196 quantized values** and
  **1,296 scale boundary cases**, including NaNs/infinities, subnormals and
  row-count limits. Scales are checked with independent CPU 128-bit arithmetic.
- Native root/stump reductions: **405 cases / 11,964,810 gradient pairs**, with
  both matrix layouts, padded strides, cancellation and subnormal inputs; exact
  representable sums, compensated CPU references, and bitwise stump weights.
  **168 root histogram sums** also match CPU left-to-right addition exactly.
- Native means: **1,080 cases / 34,880,670 values**, covering weights, FP32
  per-term division, multiple targets, matrix strides, constant values, zeros,
  subnormals and wide exponent ranges. The scalar mean uses binary64 accumulation
  of FP32 terms, so its rounding order can differ from CPU FP32 accumulation.
- Native quantiles/sorting: **240 cases / 4,227,960 stable records and exact CDF
  entries / 8,640 exact quantiles**, plus **2,140,938 integer prefix values**.
  Covers ties, signed zeros, subnormals, extreme values, multiple columns, strides
  and recursive scan boundaries. **56 production median/mean API cases** also pass.
- Generic/optional-weight reductions: **63 production API cases / 14,982,576
  values**, including empty inputs, signed values, exact subnormal sums, wide
  exponent ranges, infinities and NaNs. Wide-range sums use documented rounding
  bounds and an independent compensated reference.
- Native gradient quantization: **713,814 values across six scales**, including
  subnormal floats, compared exactly against double-precision CPU rounding.
- Native partition/transpose kernels: **320 cases** against CPU stable partitions, including empty nodes, unselected rows, categorical/missing values, block boundaries, page offsets and all index widths.
- Native histogram storage: **168 cases**, with 55,272 converted/accumulated pairs,
  18,424 subtracted pairs and 82,908 gathered pairs checked against CPU references,
  including signed integer limits, page accumulation, strided outputs and guard regions.
- Native split selection: **3,780 scalar and 2,160 vector cases**, comparing selected
  splits, gains, child statistics and categorical ordering exactly against independent
  CPU references. Includes wide exponent ranges, ties, missing data, regularization,
  monotonic bounds, pooled vector weights and threadgroup boundaries.
- Software binary64: **400,784 arithmetic operations, 200,392 narrowing conversions
  and 100,007 widening conversions** checked against CPU hardware arithmetic.
- Survival mathematics: **240,180 binary64 exp/log/erf results** compared with
  CPU libm; observed maximum differences are 1/3/3 representable steps respectively
  (the test permits four). This is an observed test bound, not a proof for all inputs.
- Production AFT APIs: **75 cases / 307,275 gradient pairs**, three distributions,
  five scales, all censoring modes, narrow intervals, extreme margins, zero/tiny
  weights, empty inputs, transforms and both metrics. Maximum gradient-pair
  difference is `2.33e-10`. The deliberately extreme corpus uses 93,660 CPU
  corrections; uncensored cases require none. Ordinary integration cases use none
  except a sharp extreme-distribution case, which corrects 16/3,505 row updates.
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
| 100,000 | 0.114 s | 0.323 s | 0.35× |
| 1,000,000 | 0.795 s | 1.110 s | 0.72× |

[Latest raw training results](validation/benchmark-row-workspaces-v1-m3.json), including
GPU bin assignment, gradient bounds, root reductions, resident histograms, subtraction/gathering,
reused split buffers and skipped zero atomic additions. The [preceding run](validation/benchmark-hist-pool-m3.json)
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
1,000,000 rows took **3.890 s Metal versus 3.225 s CPU** (0.83×), and 100,000 rows
took 0.722 versus 0.412 s (0.57×). Predictions matched exactly.
[Raw wider-workload results](validation/benchmark-wide-v2-m3.json) and the
[preceding run](validation/benchmark-wide-v1-m3.json) are retained;
this workload also remains below CPU training speed.

### Adaptive objectives

Same hardware, 32 features and three alternating repetitions; **10** boosting
rounds rather than the 20 used above. Includes DMatrix construction and all
training work; quantile uses `alpha=0.5`. Neither objective shows an end-to-end
training speedup yet.

| Objective / rows | CPU | Metal | CPU / Metal |
| --- | ---: | ---: | ---: |
| Absolute error / 100,000 | 0.091 s | 0.214 s | 0.43× |
| Absolute error / 1,000,000 | 0.717 s | 0.986 s | 0.73× |
| Quantile / 100,000 | 0.107 s | 0.216 s | 0.50× |
| Quantile / 1,000,000 | 0.856 s | 0.970 s | 0.88× |

[Absolute-error results](validation/benchmark-absolute-v3-m3.json) and
[quantile results](validation/benchmark-quantile-v3-m3.json). Checked exact integer
reductions accelerate common residual scales and metric sums, with software
binary64 retained for wide exponent ranges. In the preceding
[absolute-error run](validation/benchmark-absolute-v2-m3.json) and [preceding quantile run](validation/benchmark-quantile-v2-m3.json),
at one million rows objective GPU time fell from 0.083 to 0.052 s for absolute error and from 0.068 to 0.055 s for
quantile; end-to-end Metal training fell from 0.999 to 0.882 s and from 1.164 to
1.108 s respectively. The [first absolute-error](validation/benchmark-absolute-v1-m3.json)
and [first quantile](validation/benchmark-quantile-v1-m3.json) runs remain available.
These are separate runs, subject to system timing variability.

Means and initial quantile estimation now execute on Metal. Gradient aggregation and weight
calculation for Newton stump fitting now execute on Metal, as do per-iteration
residual-scale reduction and gradient calculation. These implement the adaptive losses in the
vendored source.

With three quantile levels (`0.1, 0.5, 0.9`) and one million rows, training takes
**2.107 s Metal versus 1.878 s CPU** (0.89×). Adding random sample weights takes
**2.221 s versus 1.835 s** (0.83×). Maximum prediction differences are `9.54e-7`.
The initial GPU sort serves all three levels; weighted quantiles preserve CPU
sequential FP32 cumulative sums. [Multiple-level results](validation/benchmark-quantile-multi-v1-m3.json)
and [weighted results](validation/benchmark-quantile-weighted-v2-m3.json) include
all repetitions and operation timings.

### AFT survival

Ten rounds, depth 6, 32 features, random sample weights and a mix of uncensored,
left/right/interval-censored labels. Includes fresh DMatrix construction and
**two evaluation metrics every round**. Three alternating CPU/Metal repetitions.

| Distribution / rows | CPU | Metal | CPU / Metal |
| --- | ---: | ---: | ---: |
| Normal / 100,000 | 0.126 s | 0.520 s | 0.24× |
| Logistic / 100,000 | 0.121 s | 0.457 s | 0.27× |
| Extreme / 100,000 | 0.115 s | 0.457 s | 0.25× |
| Normal / 1,000,000 | 0.976 s | 2.243 s | 0.44× |
| Logistic / 1,000,000 | 0.934 s | 1.980 s | 0.47× |
| Extreme / 1,000,000 | 1.006 s | 2.175 s | 0.46× |

[Raw AFT results](validation/benchmark-survival-v1-m3.json) record a maximum
prediction difference of `2.38e-7`. Normal/logistic workloads need no CPU gradient
corrections; extreme workloads correct 84/1,000,000 and 828/10,000,000 row updates.
Software binary64 probability calculations add substantial GPU cost, and these
AFT training runs are slower than CPU. The correction path remains an explicit
coverage gap.

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
Project code is Apache-2.0 licensed; see [LICENSE](LICENSE). The adapted
fdlibm error-function approximation retains its Sun Microsystems permission notice
in the Metal source. This is an independent
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

### Histogram scheduling update

Shared histogram dispatch now schedules only actual blocks in each node instead
of a rectangular grid sized for the largest node. Native checks include strongly
unbalanced branches and an empty middle node. All 309 Python tests and native
suites pass. [The new run](validation/benchmark-hist-blocks-v1-m3.json) measured
1.231 s Metal versus 0.803 s CPU at one million rows (0.65×); it does not establish
an end-to-end improvement over the preceding run.

### Gradient scale update

Moved maximum-gradient reduction, finite validation and power-of-two scale
selection to Metal. Scales stay in GPU buffers through quantization and histogram
conversion; the CPU no longer scans every gradient to choose them. Exact integer
bounds avoid floating-point logarithm rounding at power-of-two boundaries.
All 315 Python tests and native suites pass. The
[training run](validation/benchmark-gradient-bounds-v1-m3.json) measured 1.219 s
Metal versus 0.791 s CPU at one million rows (0.65×), so this coverage improvement
has not established an end-to-end speedup.

### Root statistics and stump fitting update

Scalar/vector root-gradient aggregation, dense root histogram sums and Newton
stump weights now execute on Metal. Dense histogram totals retain CPU summation
order for split missing-value checks; large gradient arrays use parallel binary64
reductions. Means, quantiles and other base-score preparation remain CPU work.
All 327 Python tests pass, along with 405 native gradient/stump cases and 168 exact
root histogram sums. CPU-only build checks pass.

Latest training results above include this additional GPU coverage and remain
slower than CPU. A [single diagnostic profile](validation/profile-gradient-sums-m3.txt)
records the remaining cost across quantile construction, histograms, split
evaluation, row partitions and gradient staging. Its nested timings overlap and
should not be added together.

### Bin assignment update

SparsePage-backed matrices now assign numeric/categorical bins and accumulate bin
hit counts on Metal. Integer comparisons preserve signed-zero and subnormal cut
boundaries. Quantile sketching and adapter-built QuantileDMatrix bins remain on
CPU, and host column storage still receives completed bins. All 351 Python tests
and native suites pass; CPU-only compatibility checks pass.

[32-feature](validation/benchmark-binning-v1-m3.json) and
[128-feature](validation/benchmark-wide-v2-m3.json) benchmarks retain exact CPU
prediction agreement. Their latest speed ratios are 0.69× and 0.83× respectively
at one million rows. The [diagnostic profile](validation/profile-binning-m3.txt)
is retained; this port has not achieved a training speedup.

### Histogram workgroup tuning

A [within-process kernel benchmark](validation/benchmark-hist-kernel-v2-m3.json)
compares 1/8/16/32/64 rows per lane in randomized trial order and validates every
histogram against CPU integer sums. Larger workgroups help densely populated
nodes but hurt fragmented branches. The runtime now chooses 32 rows per lane
when the largest node contains more than a quarter of the page, and 8 otherwise.
The [benchmark source](scripts/metal_histogram_benchmark.mm) is included.

All 351 Python tests and 192 native histogram cases pass. The
[fixed-32 run](validation/benchmark-hist-workgroups-v1-m3.json) and
[adaptive run](validation/benchmark-hist-adaptive-v1-m3.json) are retained. The
adaptive million-row median is 1.209 s Metal versus 0.793 s CPU (0.66×), so the
kernel tuning has not established a full training speedup.

### Mean statistics update

Ported scalar means, per-target sample means and weighted means, including GPU
weight totals, binary64 normalization and final FP32 results. CPU label validation,
quantile/median estimation and other objective-specific initialization remain.
All 369 Python tests and 1,080 native mean cases pass, as do the other native
suites and CPU-only build checks. Zero-total-weight errors are preserved.

The [latest training run](validation/benchmark-mean-v1-m3.json) measured 1.143 s
Metal versus 0.797 s CPU at one million rows (0.70×), with identical predictions.
This remains slower than CPU, with substantial run-to-run timing variation.

### Quantile statistics update

Added stable GPU radix sorting and recursive integer prefix scans, then used them
for scalar/vector medians and initial quantile estimation. Multiple quantile levels
share a sort. Weighted cumulative sums preserve CPU sequential FP32 order, including
subnormals, so rounding cannot silently choose a different observation. All 384
Python tests, 240 native sort/quantile cases and 56 production statistics API cases
pass; CPU-only compatibility checks pass. Feature quantile sketches remain CPU work.

The latest single-quantile million-row benchmark takes 0.970 s Metal versus 0.856 s
CPU (0.88×). The three-level weighted benchmark remains slower and exposes the
sequential weighted CDF as an optimization target. Raw results are linked above.

### Weighted quantile tuning

The weighted CDF now checks each gathered tile for subnormals or negative inputs
before its sequential accumulation. Ordinary nonnegative weights use direct FP32
addition, preserving CPU rounding order without per-element underflow checks.
The slow path remains for exceptional inputs. The full test suite still passes.

GPU quantile initialization at one million rows fell from 0.153 to 0.093 seconds
across the [baseline](validation/benchmark-quantile-weighted-v1-m3.json) and
[tuned](validation/benchmark-quantile-weighted-v2-m3.json) runs. Full training changed
from 2.238 to 2.221 seconds, while CPU timing also varied; this does not establish
an end-to-end training speedup. Both measurements are retained.

### AFT survival update

Added Metal gradients for all three AFT distributions, censored likelihood and
interval-accuracy metrics, exponentiation for predictions and logarithms for base
scores. GPU probability functions use software binary64 arithmetic; `erf` adapts
the permissively licensed fdlibm/OpenLibm approximation with its notice retained.
The mathematical definitions follow the vendored XGBoost implementation and its
[AFT paper](https://arxiv.org/abs/2006.04920).

Stress testing exposed large gradient differences when small libm rounding errors
are amplified by cancellation in tiny interval probabilities. The GPU now marks
those rows for exact CPU recomputation, with visible correction counts and timing.
This preserves tested CPU equivalence while leaving full AFT GPU coverage open.
All 411 Python tests, native suites and CPU-only compatibility checks pass. The
initial AFT benchmark above establishes a reproducible performance baseline,
including its regression relative to CPU.

### Row-partition prefix tuning

Large-node partition counts now use cooperative SIMD prefix scans instead of one
serial GPU lane. Nodes with at most 16 count blocks retain the serial kernel,
which is faster for small branches. The [benchmark source](scripts/metal_partition_prefix_benchmark.mm)
compares both production kernels in randomized order and checks every integer
prefix, node total and guard region against CPU references.

At a million-row root (3,907 count blocks), prefix time falls from **248.2 to
14.9 microseconds** in the [within-process comparison](validation/benchmark-prefix-kernel-v2-m3.json).
With 256 small balanced nodes, serial is faster (2.7 versus 6.3 microseconds),
which motivates the adaptive selection. The
[initial kernel measurement](validation/benchmark-prefix-kernel-v1-m3.json) is retained.

All 411 Python tests and 320 exact native partition cases pass, including empty
nodes, partial tiles, sparse/categorical splits and multiple scan tiles. Full
training remains slower than CPU: the [parallel run](validation/benchmark-prefix-v1-m3.json)
takes 1.169 seconds and the [adaptive run](validation/benchmark-prefix-v2-m3.json)
takes 1.207 seconds at one million rows. The faster prefix kernel does not establish
an end-to-end improvement; GPU timing variability remains substantial.

### Row workspace reuse

Partition metadata, count/prefix arrays and leaf-position outputs now reuse
per-model Metal buffers. Single-page training also reuses the gradient staging
allocation for leaf assignment. External-memory pages do not retain separate
full gradient matrices. The buffers grow when needed and remain isolated between
models; transfers and synchronization still occur.

The [latest training measurement](validation/benchmark-row-workspaces-v1-m3.json)
records 42 allocations and 1,238 reuses at one million rows. Its median is 1.110 s
Metal versus 0.795 s CPU (0.72×), compared with 1.207 s Metal in the preceding run.
Timing variability prevents attributing the full change to buffer reuse, and CPU
training remains faster. All 415 Python tests and CPU-only checks pass, including
new scalar/vector and dense/sparse tests that grow and shrink training matrices.

### Generic reduction update

Moved generic float sums and optional-weight array sums to the GPU, reusing the
binary64 reduction kernels. This also moves the initial weight-total check for
absolute-error regression to Metal. Empty generic reductions now return zero on
both CPU and Metal; the prior CPU path attempted to launch zero OpenMP threads.
Scalar default weights still use the existing constant-time calculation.

All 421 Python tests, 63 new production reduction cases and CPU-only checks pass.
The [weighted absolute-error benchmark](validation/benchmark-absolute-weighted-v1-m3.json)
(10 rounds, 32 features) measures 1.121 s Metal versus 0.758 s CPU at one million
rows (0.68×), with maximum prediction difference `1.91e-6`. The GPU weight sum
itself takes about 0.72 ms. At 100,000 rows, training takes 0.256 versus 0.093 s.
This expands GPU coverage; it does not establish a training speedup.
