# xgb-mps

**Porting XGBoost to MPS, by Codex.**

An experimental Apple GPU backend for XGBoost 3.4.2, using custom Metal compute
kernels behind `device="mps"`. The goal is coverage of all CUDA-accelerated XGBoost
operations, with measured performance. **Full CUDA parity is not implemented yet.**

**Current priority: finish GPU coverage and correctness before further performance
tuning.** The backend can train and predict with `device="mps"`, but training is
still a mixture of GPU operations and CPU work. Passing tests cover implemented
paths, not every CUDA operation. **CUDA source defines the target behavior; CPU
comparisons are checks where the algorithms agree.** Some earlier kernels follow
CPU-specific algorithms and still need alignment with CUDA. No NVIDIA execution
comparison has been performed; local tests cannot establish measured CUDA equivalence.

The major remaining areas are sketch ingestion and matrix preprocessing,
remaining objective initialization and training-state operations,
CUDA algorithm alignment, categorical/metadata operations, Apple GPU array
interoperability and distributed collectives. AFT still has CPU corrections;
some long-run ranking and Covertype comparisons have numerical differences.
Persistent GPU ownership and less synchronization also remain unfinished.
The [coverage ledger](docs/CUDA_PARITY.md) records the details. Performance
experiments are deferred while these coverage gaps are addressed.

The current backend runs histograms, gradient quantization, several objectives,
prediction transforms, tree traversal, SHAP contributions/interactions, prediction-cache
updates, common metrics,
row partitioning, scalar/vector split evaluation, root-gradient sums, stump
fitting, reduced-gradient leaf refitting, linear coordinate descent, means,
stable column/grouped sorting, ROC/PR-AUC, ranking metrics,
gradient-based row sampling, uniform/weighted column sampling and quantile/median
estimation on Metal. Sketch weighting, sorted-column summaries, streaming level
compaction and numeric finalization also run on Metal; row queues, value sorting
and several specialized features still run on the CPU. Histogram conversion,
subtraction and split-input gathering now execute on Metal-owned shared buffers. See the [coverage ledger](docs/CUDA_PARITY.md)
and [105-file CUDA inventory](validation/cuda-inventory.json).

**Measured on Apple M3:** million-row binary training with AUC evaluated each
round is **1.27× faster than CPU**, and large-batch prediction measured **2.51×**.
The squared-error training baseline remains slower than CPU. These are specific
workloads, not CUDA-equivalent performance or guaranteed gains on other hardware.
Raw measurements and slower cases are retained below.

Grouped ranking ROC-AUC evaluation now measures **9.62× faster than CPU** for one
million rows in 1,024-row queries. Complete ranking training with this metric each
round measures **1.29×**; the metric gain includes avoiding quadratic pair enumeration.

[Published CUDA workload replays](docs/PUBLISHED_BENCHMARKS.md) now include full
HIGGS (10.5M training rows, 1,000 rounds): **190.85 s CPU / 193.74 s MPS**, one
paired run. Covertype (500 rounds) takes **33.32 s / 70.29 s**, medians of three
pairs. Binary-column histogram optimization reduced its MPS time from 92.50 s.
All predicted test classes agree; HIGGS passes strict margin checks, while seven
Covertype margins remain outside tolerance. CPU wins are retained alongside gains.

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

Linear GPU training uses `booster="gblinear", updater="coord_descent", device="mps"`.
Its default `shotgun` updater remains on the host, as it does with CUDA. The
vendored upstream version deprecates `gblinear`, but this port supports it.

Approximate training is also available with `tree_method="approx"`. It uses
unsampled Hessians for cuts and resets sampling per tree, following CUDA's updater
order. Sketch/sampling algorithms and quantized-state ordering still require CUDA alignment.

Explicit tree updater selection, `exact` and distributed training are rejected
for `mps`. Inaccessible Metal devices produce an
error. Other features may use CPU implementations as recorded in the coverage ledger;
selecting `mps` does not imply that every stage runs on the GPU.

## Current Metal operations

| Operation | Implemented coverage |
| --- | --- |
| Approximate training | GPU Hessian extraction, copies and tree operations; cuts before sampling and fresh sampling per tree; underlying sketch/sampler and quantized-state ordering still need CUDA alignment |
| Gradient state | GPU strided target/layout copies and bit-preserving Hessian extraction; aliases use staged snapshots |
| Histograms | Dense/sparse compressed indices, external-memory pages, scaled integer sums, shared-memory dense kernel, GPU binary64 conversion/subtraction and a resident cache |
| Split evaluation | Numeric and categorical splits, missing directions, scalar/vector leaves, regularization, monotonic bounds; CPU keeps interaction masks |
| Row partitioning | Stable GPU partitions, numeric/categorical/missing values, GPU leaf assignment, resident row indices |
| Bin assignment | GPU numeric/categorical cut search and hit counts for SparsePage-backed matrices; dense 8/16/32-bit compression; adapter-built QuantileDMatrix bins remain CPU |
| Feature sketches | GPU group/Hessian weighting, sorted-column weighted summaries, streaming level compaction and final numeric merging/pruning; row queues and value sorting remain CPU; final cut materialization remains host, as in CUDA |
| Quantized data cache | Page-owned GPU bins reused across trees; dense indices transposed on the GPU |
| Uniform sampling | GPU LCG skip-ahead and Bernoulli masking, both matrix layouts, replay for reduced-gradient leaf fitting; host chooses the per-tree seed |
| Gradient-based sampling | GPU norms, stable sorting, CPU-ordered FP32 prefixes, threshold selection, random draws and gradient rescaling; includes strided multi-target replay and zero budgets; host chooses the per-tree seed |
| Gradient quantization | GPU sign-separated binary64 bounds and CUDA scale formula; truncation and minimum-one positive Hessians; scales stay on GPU. Sampling order and integer tree-state integration still need alignment |
| Root statistics / stump fitting | GPU scalar/vector gradient sums, dense root histogram sums and unregularized stump weights |
| Reduced-gradient leaf fitting | GPU segmented binary64 sums over resident row IDs, accumulation across external pages, regularized/clipped final weights; existing host tree receives the completed leaf values |
| Linear coordinate descent | GPU bias/feature gradient sums and residual updates; all five feature selectors supported; scalar weight updates, selectors, CSC preparation and linear prediction remain host, as in CUDA |
| Generic sums | GPU float-to-binary64 reductions and optional-weight sums; scalar default weights need no array dispatch |
| Means | GPU scalar, column-wise and weighted means; binary64 reductions and weight sums; other objective-specific initialization work remains CPU |
| Objective initialization | GPU expectile initial gradients/Hessians and absolute-error target-mean broadcast; small final base-score adjustments remain host, as in CUDA |
| Objective validation / base scores | GPU logistic, gamma, squared-log, Poisson/Tweedie and multiclass label checks; checked logistic/gamma intercepts and logarithm/logit conversion, including Cox base-score conversion |
| Quantiles / medians | GPU stable radix sorting, integer prefix scans, CPU-compatible interpolation and sequential FP32 weighted CDFs; multiple quantile levels share one sort |
| Gradients | Squared error, squared log error, logistic/logitraw, gamma, pseudo-Huber, Poisson, Tweedie, hinge, multiclass, expectile; adaptive absolute-error and quantile gradients with GPU residual-scale reductions |
| AFT survival | GPU normal/logistic/extreme gradients, prediction/base-score transforms, likelihood and interval-accuracy metrics; numerically sensitive censored gradient rows are corrected on CPU and counted explicitly |
| Prediction transforms | Sigmoid, exponential, hinge, softmax, argmax, cumulative expectile, quantile row sorting |
| Tree traversal | Batch/inplace/leaf prediction; numeric/categorical splits, missing values, scalar/vector leaves, tree ranges, dropout weights |
| SHAP contributions / interactions | GPU eight-point quadrature traversal, repeated-feature paths, scalar/vector leaves, categorical/missing splits, dropout weights and bias/interaction finalization; input staging and category recoding remain host |
| Training prediction cache | Leaf-value accumulation, including strided multi-target output and sampled rows |
| Ranking training | Stable grouped sorting, NDCG cache/label checks and top-k and mean-sampled pairwise/NDCG/MAP pairs, gradients and normalization, including unbiased loss corrections and position-bias updates, on Metal |
| ROC-AUC | Binary, multiclass one-vs-rest and multilabel; stable descending score sort, weighted binary64 prefixes and tied-score trapezoid integration; final per-output normalization/averaging stays on the host |
| Grouped ranking ROC-AUC | GPU stable label ordering, merge-based concordant/tied-pair counts, query normalization and aggregation; preserves label ties, short-query exclusions and FP32 squared-weight behavior |
| Column sampling | GPU feature initialization, CPU-compatible MT19937 draws, uniform/weighted selection, stable priority ordering and integer feature sorting; tree/level/node caches remain host |
| Grouped ranking PR-AUC | GPU label validation, stable prediction ordering, segmented binary64 prefixes, continuous PR integration, degenerate-query handling and aggregation |
| PR-AUC | Binary, multiclass and multilabel; GPU sort, weighted prefixes and stable continuous PR integral; final aggregation stays on the host |
| Metrics | RMSE, RMSLE, MAE, MAPE, logloss, error thresholds, Poisson, gamma, Tweedie, pseudo-Huber, multiclass logloss/error, quantile, expectile, NDCG, MAP and precision; GPU label validation and NDCG cache initialization |

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
`gradient_sampling_dispatches` and `gradient_sampling_gpu_time_ns` report the
gradient-based sampler separately; they are also included in the general sampling
counters and must not be added to those totals.
`ranking_roc_auc_dispatches` and `ranking_roc_auc_gpu_time_ns` likewise describe a
subset of ROC-AUC and generic metric work; their timings overlap those totals.
The corresponding `ranking_pr_auc_*` counters are subsets of PR-AUC and metric totals.
`sketch_finalize_dispatches`, `sketch_finalize_columns` and
`sketch_finalize_gpu_time_ns` report final numeric sketch processing, excluding
host ingestion, staging and final cut materialization.
`sketch_weight_dispatches`, `sketch_sorted_dispatches` and
`sketch_update_dispatches` report weighting, sorted-column summaries and streaming
compaction. `sketch_ingest_gpu_time_ns` covers these commands, excluding host
queue construction/sorting, allocation and staging. The sorted-column path is
used by public `tree_method="approx"` training, with the CUDA-alignment limitations
listed in the coverage ledger.
`expectile_init_dispatches` and `prediction_init_dispatches` expose the new
objective-specific initialization work. Their combined GPU duration is reported
by `initialization_gpu_time_ns`, separately from ordinary objective GPU time.
`label_validation_dispatches` / `label_validation_gpu_time_ns` and
`base_score_dispatches` / `base_score_gpu_time_ns` report objective label checking
and intercept conversion separately. Invalid checked inputs still count completed
GPU work. Empty inputs, identity conversions and unconditional label predicates
require no dispatch.
`shap_dispatches` and `shap_interaction_dispatches` count completed contribution
and interaction batches; `shap_gpu_time_ns` covers both. These are separate from
ordinary prediction counters and exclude host preparation and staging.
`leaf_sum_init_dispatches`, `leaf_sum_dispatches` and `leaf_weight_dispatches`
report GPU clearing, per-page leaf reductions and final weight calculation.
`leaf_fitting_gpu_time_ns` covers those stages separately from root sums, histogram
work and row partitioning; source sharing/staging and final host tree updates are
excluded.
`linear_gradient_dispatches` and `linear_residual_dispatches` count completed
coordinate-descent reductions and updates; `linear_gpu_time_ns` measures those
commands. Host feature selection, scalar model updates and staging are excluded.
`gradient_copy_dispatches`, `hessian_extract_dispatches` and
`gradient_state_gpu_time_ns` report generic gradient copies/Hessian extraction.
`gradient_state_shared_input_bytes`, `gradient_state_shared_output_bytes` and
`gradient_state_staging_bytes` count complete view extents, including padding.
`objective_shared_input_bytes`, `objective_shared_output_bytes` and
`objective_staging_bytes` measure memory access for the ordinary gradient kernel.
Adaptive and AFT stages have separate memory paths.
`histogram_shared_gradient_bytes` and `leaf_shared_gradient_bytes` count shared
gradient-view extents; a leaf view can span the full matrix even for one page.
`leaf_shared_output_bytes` counts shared output slices.

## Validation

```sh
bash scripts/test.sh
```

On Apple M3, 2026-11-03:

- Python: **1,206 full-suite regression tests plus 4 new quantizer integration
  tests passed; 1 skipped**. The skip requires a machine without an
  accessible GPU. Positive GPU tests assert completed operation counters so CPU
  fallback cannot satisfy them.
- Gradient layout copies/Hessian extraction: **276 native cases**, with
  **25,026,384 copied pairs and 25,026,384 Hessians** checked bit for bit, plus
  aliases/read-only inputs, invalid views and guard checks. **141 new Python
  cases** cover approximate training and gradient-state integration, including
  CUDA updater ordering. These do not establish NVIDIA execution equivalence.
- Native histograms: **576 cases**, with every bin compared exactly to independent
  CPU integer sums. Covers dense/sparse indices, 8/16/32-bit storage, page offsets,
  high contention, negative values, carries, zero words, sampled-out rows and
  workgroup sizes of 1/8/32/64 rows per lane.
- Native sampling: **3,780 exact cases / 1,019,088 Bernoulli draws** against the
  CPU standard-library implementation, including zero seeds, rounding boundaries,
  strided matrices, guard regions and CPU thread partitions.
- Native gradient-based sampling: **498 exact cases / 6,748,668 rows**, comparing
  norms, thresholds, masks and rescaled gradients to production CPU routines and
  libc++ random draws. Includes subnormals, overflow, equal gradients, zero/full
  budgets, strided storage and reduced-gradient replay. Six additional production
  Sampler checks preserve the host random-engine state and replay decisions.
- Native bin assignment: **168 exact cases / 216,600 entries / 38,616,312 hit
  counts**, including all index widths, category truncation, subnormal/signed-zero
  boundaries, invalid values and guard regions.
- Native CUDA-formula quantization: **99 bound reductions / 19,305,306 quantized
  values**, including **1,507,902 minimum-one positive Hessians**, and **1,210
  scale boundary cases**. Independent scalar CUDA-formula references cover zero,
  subnormals, outliers, sign-separated sums, nonfinite rejection and row limits.
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
- Native gradient conversion: **713,814 values across six scales**, including
  subnormal floats, compared exactly against CUDA's truncation and positive-
  curvature preservation rules using independent double-precision references.
- Native partition/transpose kernels: **320 cases** against CPU stable partitions, including empty nodes, unselected rows, categorical/missing values, block boundaries, page offsets and all index widths.
- Native histogram storage: **168 cases**, with 55,272 converted/accumulated pairs,
  18,424 subtracted pairs and 82,908 gathered pairs checked against CPU references,
  including signed integer limits, page accumulation, strided outputs and guard regions.
- Native split selection: **6,300 scalar and 2,160 vector cases**, comparing selected
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
- Shared host-memory views: **36 cases / 4,313,492 values** plus six fallback
  checks under Metal API Validation. **51 production objective API cases /
  1,693,795 gradient pairs** cover offsets, guards, read-only inputs, split VM
  regions and overlapping input/output storage. **49 production leaf cases /
  1,118,257 row positions** add page offsets, strided/subnormal Hessians and empty leaves.
- ROC-AUC: **432 production cases / 6,761,856 score-label pairs** and **420
  independent weighted-pair checks**. Includes both dense layouts, padded strides,
  signed-zero ties, subnormal scores/weights, extreme weight ranges, degenerate
  inputs, and recursive scans spanning three levels. Normalized comparisons use
  `2e-12` absolute/relative tolerances; general bitwise agreement is not promised.
- Grouped ranking ROC-AUC: **272 production cases / 234,208,157 independently
  enumerated pairs**, compared with both upstream CPU and an independent pair
  reference. Three additional million-row analytic cases each represent
  **500,002,500,003 pairs**, testing 64-bit counts without a quadratic CPU run.
  Covers label/score ties, padded strides, zero/tiny/large weights, infinities/NaNs
  in scores, short queries, weight-square overflow errors and output guards.
- Column sampling: **530 native cases / 3,275,463 selected feature IDs /
  4,610,295 random draws**, exactly matching CPU; weighted priorities also match
  bit for bit, including constructed ties. Production hierarchy/cache behavior,
  parent RNG state and guards are checked. A separate **8,086,019-input** logarithm
  check has zero CPU mismatches on this M3.
- Sketch finalization: **942 native columns / 2,047,144 retained source entries**,
  with **9,731,868 exact field comparisons** across GPU summaries and independent
  CPU rank queries. Includes 28 streamed sketches through 1,000,003-row ingestion,
  zero/tiny/overflowing weights, rank repair and empty/tiny pruning budgets.
  Public tests require identical cut bits for dense, sparse, missing-value,
  weighted, grouped, categorical and external-page inputs.
- Sketch weighting/ingestion/compaction: **48 weight cases, 211 sorted columns,
  24 streamed sketches, 32 batched level updates and 8 production container cases**.
  Includes million-row input and exact CPU summary/cut comparisons; finite fields
  compare by bits, NaNs by classification. Public sketch tests require the new
  GPU dispatches while retaining their existing exact cut-bit checks.
- Exact FP32 division: **5,830,328 inputs** match CPU arithmetic, including random
  bit patterns, all exponent pairs, pruning denominators and subnormal rounding
  ties; NaNs are checked by classification. This replaces software binary64
  division in sketch pruning without changing its rounding contract.
- Reduced-gradient leaf fitting: **304 native cases / 24,327,036 source gradient
  pairs**, 46,852 exact sum fields and 156,240 bitwise CPU weight comparisons
  (NaNs checked by classification). Covers strided/broadcast inputs, paged sums,
  empty leaves, subnormals, large/nonfinite values, L1/L2 penalties, clipping,
  bounds and guards. **66 public tests** cover training and saved-model
  continuation across dense, missing, sparse and external-page inputs.
- Linear coordinate descent: **49 native cases / 4,118,809 source gradient
  pairs**, 3,159 exact sum fields and 148,277,124 residual field comparisons.
  Finite residuals match CPU bits, including signed zeros; NaNs use classification.
  Covers empty/million-row inputs, duplicate CSC rows, subnormals, nonfinite
  values, negative Hessians and guards. **136 public cases** cover all five selectors,
  dense/missing/sparse data, multiple outputs, penalties and model continuation.
- SHAP: **576 native cases / 424,350,144 output values**, including zero entries,
  compared with independent one-feature Shapley values; maximum absolute error
  `2.39e-7`. Checks include guards, overlapping storage, empty inputs and batch
  bounds. **87 public cases** cover CPU equivalence, independently enumerated
  feature subsets, scalar/vector/categorical trees, external pages, base margins,
  tree limits, deep repeated-feature paths, wide outputs and invalid covers.
- Objective initialization: **164 native cases / 7,220,670 gradient pairs /
  14,441,340 copied floats**. Gradients match CPU bits except NaNs, which are
  checked by classification; broadcasts preserve every bit, including NaN
  payloads. Checks include empty/million-row inputs, multiple targets, zero/tiny/
  overflowing weights, guards and aliased broadcast inputs.
- Objective label validation: **120,093,300 individual predicate comparisons**
  and **576 production API cases / 216,948,024 tensor values** against CPU.
  Includes subnormals, signed zeros, NaNs/infinities, integer class boundaries,
  strided/broadcast views, ignored padding and source guards.
- Base-score conversion: **74 native cases / 4,570,936 comparisons**, matching
  CPU bits except NaN classification; identity preserves NaN payloads. Covers
  all exponent ranges, logarithm table boundaries, logistic clamping, objective
  API integration, empty vectors and rejected inputs remaining unchanged.
- Grouped ranking PR-AUC: **297 production cases / 14,198,354 rows / 4,660 query
  results**, checked against CPU and independent continuous-integral references.
  Includes irregular/empty queries, recursive scan carries, widely differing query
  weights, fractional/subnormal labels, degenerate queries, strided storage and
  six GPU label-validation errors. Maximum observed reference difference is
  `9.1e-15`; tiny tied-score cases also receive relative-accuracy checks.
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
| 100,000 | 0.123 s | 0.522 s | 0.24× |
| 1,000,000 | 0.810 s | 1.176 s | 0.69× |

[Latest raw squared-error training results](validation/benchmark-borrow-training-v1-m3.json), including
GPU bin assignment, gradient bounds, root reductions, resident histograms, subtraction/gathering,
reused split buffers and skipped zero atomic additions. The [preceding run](validation/benchmark-hist-pool-m3.json)
and [first resident-histogram run](validation/benchmark-hist-storage-m3.json) are
retained. Dispatch overhead and substantial GPU timing variability remain.
Metal training is still slower than CPU. The preceding [objective-sharing run](validation/benchmark-borrow-v1-m3.json) measured 1.051 s Metal at one million rows; the latest run regresses despite avoiding more explicit copies.

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
- [x] Port gradient-based sampling, including exact CPU thresholds and replay.
- [x] Port uniform/weighted column sampling, with CPU-compatible random streams.
- [ ] Port remaining training-state updates.
- [x] Port final numeric sketch level merging and pruning.
- [x] Port sketch weighting, sorted-column summaries and streaming level compaction.
- [x] Port exact SHAP contributions and interactions.
- [x] Port CUDA coordinate-descent operations for linear training.
- [ ] Finish quantile construction, objective coverage and generic state operations.
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

### Shared objective memory update

Ordinary objective kernels can now read predictions, labels and weights and write
gradients directly through guarded Metal views over existing host arrays on a
unified-memory GPU. The implementation checks alignment, writable mapping and
single-VM-region coverage, retains storage until command and buffer references
are released, and keeps the copied path for unsuitable mappings. Overlapping
input/output spans use separate output staging to preserve snapshot behavior.
Views are scoped to synchronous calls and are never stored in caches.

The [implementation](xgboost/src/common/metal_memory.h) follows Apple's
[buffer-wrapping requirements](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)).
All 425 Python tests, native suites and CPU-only checks pass. Metal API Validation
also passes the new low-level and production memory tests, and the preceding
421-test integration suite was run with validation enabled.

The [latest training run](validation/benchmark-borrow-v1-m3.json) avoids 320 MB of
explicit objective staging copies over 20 million row updates. It measures
**1.051 s Metal versus 0.799 s CPU** (0.76×), with identical predictions. The
preceding workspace run measured 1.110 s Metal; this comparison remains subject
to timing variation. Full training-state residency and synchronization reduction
are still open work, and training remains slower than CPU.


### Shared training-input and leaf-output update

Gradient quantization and final leaf assignment now use the same guarded,
synchronous host-memory views as ordinary objectives. Views are retained only
until their command completes and its references are released. Owned staging
remains available for read-only, split-region and overlapping mappings; cached
buffers never retain borrowed host pointers. Leaf assignment also orders empty
ranges before nonempty ranges with the same starting position.

All 425 Python tests, native suites and CPU-only checks pass. The expanded
production memory test adds **49 cases / 1,118,257 row positions**, checking
strided targets, page offsets, sampled-row markers, empty leaves, read-only input,
aliasing and guard values under Metal API Validation.

The [new training measurement](validation/benchmark-borrow-training-v1-m3.json)
retains a regression: at one million rows, 32 features and 20 rounds, Metal takes
**1.176 s versus 0.810 s CPU** (0.69×), compared with 1.051 s Metal in the preceding
run. Predictions are identical. At 100,000 rows the medians are 0.522 s Metal and
0.123 s CPU; Metal repeats vary from 0.290 to 0.561 s. These measurements do not
establish a speedup from sharing more arrays. The counters confirm another
400 MB of explicit gradient/leaf staging copies avoided at one million rows,
on top of the ordinary-objective sharing, but dispatch and synchronization
costs remain and timing variability needs controlled profiling.


### ROC-AUC update

Binary, multiclass and multilabel `eval_metric="auc"` now sort predictions, compute
weighted positive/negative prefixes and integrate tied-score trapezoids on Metal.
All columns share one command and batched stable radix sort; no per-row host sort,
scan or class-indicator expansion is needed. Prefixes and integration use software
binary64; normalization, multiclass prevalence weighting and multilabel macro
averaging retain the existing host behavior. `roc_auc_dispatches` and
`roc_auc_gpu_time_ns` expose completed work. Ranking score calculation and grouped ROC/PR-AUC remain CPU.
NaN score ordering has no defined CPU equivalence contract.

The [weighted metric-only benchmark](validation/benchmark-roc-v1-m3.json) uses five
alternating repeats, fixed random margins, 8 CPU threads, and warmed runtime and
prediction caches. Timings include metric input staging, sorting, scans, reduction
and synchronization; they exclude DMatrix construction and training.

| Rows | Outputs | CPU | Metal | CPU / Metal |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 1 | 6.25 ms | 3.66 ms | 1.71× |
| 100,000 | 3 | 6.90 ms | 8.78 ms | 0.79× |
| 1,000,000 | 1 | 81.07 ms | 22.40 ms | 3.62× |
| 1,000,000 | 3 | 98.68 ms | 40.72 ms | 2.42× |

Maximum AUC difference is `1.05e-14`. These timings measure only metric evaluation.
First-evaluation times are retained in the raw JSON.

All **462 Python tests**, native suites under Metal API Validation and the CPU-only
build checks pass. Additional leaf-cache tests exposed Metal's flushing of
subnormal Hessians during zero comparisons. Leaf assignment now checks IEEE bits,
so tiny nonzero Hessians keep their rows valid. Direct leaf tests and cached-margin
training tests cover the fix; fresh model predictions alone could have hidden it.
The ROC class-indicator comparison similarly preserves tiny nonzero labels.


### Training with ROC-AUC evaluation

A separate [end-to-end benchmark](validation/benchmark-auc-training-v1-m3.json)
measures fresh DMatrix construction and 20 rounds of weighted binary logistic
training, evaluating AUC on the training set after every round. It uses 32
features, depth 6, 256 bins, 8 CPU threads and three alternating repeats.

| Rows | CPU | Metal | CPU / Metal |
| ---: | ---: | ---: | ---: |
| 100,000 | 0.236 s | 0.430 s | 0.55× |
| 1,000,000 | 2.089 s | 1.641 s | **1.27×** |

Metal repeats at one million rows range from 1.588 to 1.678 s, versus 2.056 to
2.094 s CPU. Every Metal training run records 20 completed ROC-AUC dispatches.
Maximum prediction difference is `1.19e-7`; maximum per-round AUC difference is
`6.93e-8`. Unlike the fixed-score metric benchmark, this run also includes CPU/GPU
prediction-transform rounding, which can change ordering at close scores.

This establishes a speedup for this training-and-evaluation workload. The
squared-error baseline without AUC evaluation and the smaller binary dataset
remain slower than CPU. Full CUDA coverage and broadly competitive training
remain unfinished.

```sh
.venv/bin/python scripts/benchmark.py --objective binary:logistic \
  --eval-metric auc --weighted --rows 100000 1000000 \
  --output results/my-auc-training.json
```


### Larger training baseline

The workload suite now accepts deeper trees, longer runs, direct FP32 data
generation, and periodic round-count progress. It records process peak RSS and
system swap alongside timing; these measurements are not isolated per-device
memory peaks. Larger jobs run sequentially in named `tmux` sessions.

The [one-million-row, 128-column baseline](validation/benchmark-wide-long128-v1-m3.json)
uses **200 rounds, depth 8**, 256 bins and all 8 CPU threads. Medians of three
alternating runs are **17.84 s CPU / 25.39 s Metal (0.70×)**, with identical
predictions. In the final Metal run, histogram commands account for 7.80 s and
split evaluation for 6.96 s. This exposes split scoring as a substantial cost
on wider, deeper training; increasing workload alone has not produced a speedup.

Process peak RSS reaches 2.80 GB. The machine already had approximately 16.6 GB
of system swap in use, rising to 18.5 GB during the run. These are measurements
under existing memory pressure, not a claim of isolated hardware throughput.

```sh
.venv/bin/python scripts/benchmark.py --rows 1000000 --features 128 \
  --rounds 200 --max-depth 8 --progress-every 20 --float32-generation \
  --output results/my-wide-long.json
```


### PR-AUC and split-scoring update

`eval_metric="aucpr"` now runs classification sorting, tied-score scans and
continuous PR integration on Metal. The [weighted metric-only benchmark](validation/benchmark-pr-v1-m3.json)
uses the same five alternating repeats and 8 CPU threads as ROC-AUC above:

| Rows | Outputs | CPU | Metal | CPU / Metal |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 1 | 7.75 ms | 2.20 ms | 3.52× |
| 100,000 | 3 | 7.95 ms | 5.45 ms | 1.46× |
| 1,000,000 | 1 | 96.12 ms | 18.88 ms | **5.09×** |
| 1,000,000 | 3 | 114.01 ms | 56.17 ms | 2.03× |

Maximum metric difference is `1.35e-14`. These gains cover metric evaluation,
not full training. The 432 native PR cases cover 6.76 million score-label pairs
and include 672 independent long-double integral checks. A separate extreme-weight
case exposes cancellation in upstream CPU PR-AUC: a negative example weighted
`1e30` above a positive example weighted `1` produces CPU area `1`, whereas the
stable GPU integral is approximately `5e-31`. Four other extreme-weight test cases
also differ from the unstable CPU formula; these are checked against the independent
integral instead. The public metric text rounds the `5e-31` result to zero.

Numerical split candidates now use [conservative gain bounds](docs/METAL_SPLIT_BOUNDS.md)
to skip software-binary64 scoring only when they provably cannot win. Survivors
retain the original precise evaluation and tie order. Missing-value directions
are included; categorical, constrained and unsupported numerical ranges keep
full evaluation. One million bound checks and 6,300 exact scalar split comparisons
pass, along with 2,160 vector cases and the integration suite.

The [paired kernel benchmark](validation/benchmark-split-filter-kernel-v3-m3.json)
compiles screening on and off from the same source and verifies both against CPU.
At 8,192 feature tasks, split scoring improves **1.69× without missing values** and
**1.81× with missing values**. Smaller launches improve 1.25–1.63×.

For [one million rows × 128 columns, 200 rounds, depth 8](validation/benchmark-wide-long128-filter-v3-m3.json),
current medians are **18.83 s CPU / 21.94 s Metal (0.86×)** with identical predictions.
Metal time is lower than the earlier 25.39 s baseline, but this still does not meet
the goal of faster general training. Runs occurred under substantial existing system
memory pressure; all repeats and swap telemetry are retained.

The [clean 256-column rerun](validation/benchmark-wide-long256-filter-v4-m3.json)
uses the same one million rows, 200 rounds and depth 8: **41.78 s CPU / 40.22 s
Metal (1.04×)**, with identical predictions. CPU repeats span 40.72–45.90 s;
Metal spans 39.85–41.78 s. The final GPU run spends 15.57 s in histogram commands
and 8.09 s scoring splits. The machine remains under existing memory pressure.
This is far short of the intended large training speedup. An earlier 256-column
run crossed laptop sleep and is excluded. The benchmark now records wall and
monotonic times, rejects clock discrepancies above one second and checkpoints
completed repetitions. The clean run's maximum discrepancy is below 0.6 ms.

Current validation: 498 Python tests pass (497 in the full run plus the corrected
metric-format test rerun), one skip; native tests pass with Metal API Validation,
and the CPU-only build checks pass. Full CUDA parity remains in progress.


### Grouped ranking sort

Stable per-query sorting now runs on Metal for `rank:pairwise`, `rank:ndcg` and
`rank:map` training, and for NDCG, MAP and precision metric evaluation. NDCG's ideal
label ordering is also GPU-sorted. Query weights use the existing GPU sum reduction.
Small queries use a bitonic sorting network with SIMD exchanges; larger or uneven
queries use stable score sorting followed by stable query-ID radix passes. Signed
zeros remain tied, and original row order breaks ties consistently with CPU.
`group_sort_dispatches` and `group_sort_gpu_time_ns` expose completed GPU work.

**This is partial ranking coverage:** pair construction, ranking gradients,
discounts, ideal-gain computation and metric score aggregation still run on CPU.
Grouped ROC/PR-AUC is also still CPU. No CPU sorting fallback is used for supported
non-NaN scores.

The native production-API test checks **100 cases / 7,249,460 sorted rows**, including
strided data, empty queries, query boundaries, 65,537-query layouts, all ties,
subnormals and infinities. All **520 Python tests** and native tests pass under
Metal API Validation; one Python test is skipped. The CPU-only build checks pass.

The [initial ranking benchmark](validation/benchmark-ranking-v1-m3.json) measures
warm `ndcg@10` evaluation with fixed predictions, 8 CPU threads and five alternating
repeats. It includes GPU sorting, host score aggregation, staging and synchronization,
and excludes training/model construction and initial ideal-order cache setup.

| Rows | Query size | CPU | Metal sort + host score | CPU / Metal |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 32 | 0.49 ms | 0.54 ms | 0.90× |
| 100,000 | 256 | 0.90 ms | 0.55 ms | 1.66× |
| 100,000 | 4,096 | 1.67 ms | 1.16 ms | 1.44× |
| 1,000,000 | 32 | 4.75 ms | 2.90 ms | 1.64× |
| 1,000,000 | 256 | 6.78 ms | 3.27 ms | 2.08× |
| 1,000,000 | 4,096 | 17.85 ms | 10.63 ms | 1.68× |

These are metric timings, not ranking-training speedups. Individual timings vary;
all repeats, first-evaluation times and GPU counters are retained in the JSON.


The next [ranking-sort benchmark](validation/benchmark-ranking-v2-m3.json) adds
an on-chip network for queries of 257–4,096 rows, retaining the global radix path
for larger queries. For one million rows in 4,096-row queries, warmed NDCG@10
medians are **9.92 ms CPU / 3.16 ms Metal-sort-plus-host-score (3.14×)**. The earlier
Metal median was 10.63 ms; CPU timing also changed between sessions, so the raw
repeats should be used when comparing results. Small-query workloads are unchanged
and remain variable. The expanded native test checks **120 cases / 7,518,270 rows**;
all **33 ranking integration tests** pass with Metal API Validation. The preceding
full suite and CPU-only checks remain green; this optimization changes only the
Metal sort path and its boundary tests.


### GPU precision and NDCG scoring

NDCG and precision now keep sorted records on Metal for query scoring and weighted
query reduction. One command performs sorting and scoring; only the final two
binary64 totals return to the host. The existing final normalization remains host.
NDCG still obtains discounts and inverse ideal gains from its CPU cache; its ideal
label ordering is GPU-sorted. MAP score accumulation and ranking gradients remain
CPU at this stage.

`ranking_metric_dispatches` and `ranking_metric_gpu_time_ns` count completed fused
sort-and-score commands. Their durations overlap grouped-sort and generic metric
times, so those fields must not be added together.

The [NDCG benchmark](validation/benchmark-ranking-v3-m3.json) and
[precision benchmark](validation/benchmark-pre-v3-m3.json) include sorting, scoring,
staging, synchronization and final host normalization. They use fixed predictions,
8 CPU threads and five alternating repeats; initial cache setup and training are
excluded. At one million rows:

| Metric | Query size | CPU | Metal | CPU / Metal |
| --- | ---: | ---: | ---: | ---: |
| NDCG@10 | 32 | 4.83 ms | 3.01 ms | 1.60× |
| NDCG@10 | 256 | 6.39 ms | 2.58 ms | 2.47× |
| NDCG@10 | 4,096 | 10.59 ms | 2.58 ms | **4.10×** |
| Precision@10 | 32 | 3.03 ms | 3.15 ms | 0.96× |
| Precision@10 | 256 | 5.43 ms | 2.59 ms | 2.10× |
| Precision@10 | 4,096 | 7.67 ms | 2.49 ms | 3.08× |

Smaller workloads and short-query precision can remain slower than CPU. All native
suites and **543 Python tests** pass, with one skip, under Metal API Validation;
the CPU-only checks also pass. The added **504 native score cases / 5,350,320 rows**
cover empty/degenerate queries, strided labels, exponential and linear gain,
near-binary labels, extreme query weights and top-k limits. Maximum scaled
CPU score difference is `3.61e-15`.


### MAP scoring and current ranking validation

MAP now joins precision and NDCG in the fused GPU sort-and-score path. Exact binary
labels use parallel integer hit counts and prefixes. For the near-binary labels
accepted by upstream validation, GPU evaluation preserves CPU summation order so
cancellation cannot silently change the zero-hit fallback. Weighted zero-relevance
behavior, signed near-zero labels and invalid-result handling are covered explicitly.
NDCG cache calculations, ranking losses and grouped ROC/PR-AUC remain unfinished.

All **557 Python tests** pass, with one skip, along with native tests under Metal
API Validation and the CPU-only build checks. Native ranking validation now covers
**120 sort cases / 7,518,270 rows** and **896 score cases / 9,511,680 rows**. The
maximum scaled CPU score difference is `4.88e-15`.

Current [NDCG@10](validation/benchmark-ranking-v4-m3.json),
[precision@10](validation/benchmark-pre-v4-m3.json) and
[MAP@10](validation/benchmark-map-v4-m3.json) measurements include all sort/score work
and use five alternating repetitions. At one million rows, with 4,096-row queries:

| Metric | CPU | Metal | CPU / Metal |
| --- | ---: | ---: | ---: |
| NDCG@10 | 10.45 ms | 2.66 ms | 3.94× |
| Precision@10 | 7.36 ms | 2.58 ms | 2.86× |
| MAP@10 | 7.98 ms | 2.94 ms | 2.72× |
| [Uncapped NDCG](validation/benchmark-ndcg-full-v4-m3.json) | 11.03 ms | 2.63 ms | **4.19×** |
| [Uncapped MAP](validation/benchmark-map-full-v4-m3.json) | 9.03 ms | 2.90 ms | 3.11× |

Short queries can still favor CPU: at 32 rows per query, million-row precision@10
is 0.90× and uncapped MAP is 0.90×. Maximum absolute metric difference across these
benchmarks is below `3.5e-15`. Initial cache setup, data/model construction and
training are excluded; first-evaluation times and every repeat are retained.
These metric improvements do not change the **1.04×** wide-training result above.


### Ranking validation and NDCG cache initialization

Ranking label checks now execute on Metal. NDCG label ordering, discounts and
inverse ideal gains also run on GPU; their arrays return to the existing host-owned
cache. Linear-gain NDCG retains upstream's validation behavior. The new counters
`ranking_label_dispatches` and `ranking_cache_dispatches`, with corresponding GPU
time fields, expose this work. Cache-command time overlaps its grouped-sort time.

The [fresh-cache benchmark](validation/benchmark-ndcg-init-v2-m3.json) includes
ranking validation, ideal-gain cache construction, prediction caching and NDCG@10
evaluation on every timed repeat. Runtime is warmed; data and model construction
are excluded. One million rows, 8 CPU threads, five alternating repeats:

| Query size | CPU | Metal | CPU / Metal |
| ---: | ---: | ---: | ---: |
| 32 | 7.30 ms | 6.38 ms | 1.14× |
| 256 | 10.55 ms | 5.68 ms | 1.86× |
| 4,096 | 16.30 ms | 5.36 ms | **3.04×** |
| 65,537 | 19.93 ms | 15.25 ms | 1.31× |

The [preceding hybrid-cache measurements](validation/benchmark-ndcg-init-baseline-v1-m3.json)
are retained. At 4,096 rows per query, Metal time changes from 6.65 to 5.36 ms;
CPU timing varies between sessions. Maximum metric difference in the new benchmark
is `1.34e-15`. These remain evaluation timings, not training speedups.

All **570 Python tests**, native suites under Metal API Validation and CPU-only
checks pass; one Python test is skipped. Added native tests cover **380 label
validation cases / 1,858,512 values** and **336 NDCG cache cases / 3,566,880 rows**,
checking **3,551,424 discounts**. Discount relative error is at most `3.34e-16`;
ideal-gain error scaled by absolute term sums is at most `2.73e-14`.

Nearly zero linear ideal gains can amplify tiny rounding differences when inverted;
the [numerical notes](docs/CUDA_PARITY.md#ranking-metric-numerical-behavior) state this
limit explicitly. The cancellation test uses an independent 80-digit Decimal
reference. On this Apple arm64 build, `long double` itself has only 53 significant
bits, so the earlier long-double PR checks should not be read as higher-precision
oracles. Ranking pair construction, gradients and grouped ROC/PR-AUC remain to port.


### Top-k ranking gradients

Biased top-k pairwise, NDCG and MAP objectives now construct pairs, compute
per-pair gradients, normalize queries and accumulate gradients on Metal. MAP
prefix statistics also execute on GPU. Pair storage is reused in 64 MiB batches;
a single larger query can exceed that budget up to the device buffer limit.
Mean-sampled and unbiased ranking gradients still use the explicit CPU path.

The full suite passes **573 Python tests**, the native suites under Metal API
Validation and CPU-only build checks; one Python test is skipped. The new native
objective tests compare **721 cases / 1,222,100 rows / 21,101,380 candidate pairs**
with CPU, including normalization switches, ties, long queries, query weights,
linear NDCG, near-binary MAP, extreme scores and multiple storage batches.
**1,122,600 gradient pairs are bit-exact**, with maximum scaled error `1.91e-6`.
A further check covers **257,952 subnormal gradient/Hessian fields** with an
absolute allowance of two FP32 subnormal ULPs plus a relative tolerance.

Clean training benchmarks use one million rows, 32 features, query size 128,
32 top-k pairs, 20 rounds, depth 6, eight CPU threads and three alternating
repeats. Timings include DMatrix construction and training; data generation is
excluded and the Metal runtime is warmed.

| Objective | CPU median | MPS median | CPU / MPS |
| --- | ---: | ---: | ---: |
| [Pairwise](validation/benchmark-ranking-training-pairwise-v2-m3.json) | 2.282 s | 2.508 s | 0.91× |
| [NDCG](validation/benchmark-ranking-training-ndcg-v1-m3.json) | 2.322 s | 3.203 s | 0.73× |
| [MAP](validation/benchmark-ranking-training-map-v1-m3.json) | 2.069 s | 3.439 s | 0.60× |

These training cases are slower on MPS. Maximum prediction differences are
`2.63e-6`, `1.23e-5` and `4.77e-7`, respectively. The NDCG/MAP reports also retain
the slower 100,000-row cases. System-wide swap was approximately 17.3 GiB;
other applications were left running. The wall/monotonic clock discrepancy is
below 0.04 ms in these published runs. A previous run interrupted by laptop
sleep was rejected by the timing guard and is not used for this table.

`ranking_gradient_dispatches` counts completed gradient calls;
`ranking_pairs` counts candidate pairs including equal-label pairs that contribute
zero; `ranking_pair_batches` counts storage batches. `ranking_gradient_gpu_time_ns`
measures the fused command and overlaps grouped-sort and objective durations.
Do not add those timings. Broad CUDA parity and large training speedups remain
unfinished.


### Mean-sampled ranking and sigmoid accuracy

Mean-sampled pairwise/NDCG/MAP now run pair generation and gradients on Metal.
Stable label sorting follows prediction rank; a GPU generator per query preserves
Apple libc++'s `minstd_rand` and rejection-sampling sequence. Ordered incidence
lists let rows accumulate independently in CPU pair order. Candidate pair,
gradient and incidence buffers share a 64 MiB batch budget, except when one query
requires more. Sampling within a query is sequential and remains a performance
limit for long queries. Unbiased gradients and position-bias updates remain CPU.

Native checks cover **1,054,720 bit-exact random draws and engine states** across
4,120 seed/range cases, including 31/32-bit ranges. Combined top-k/mean objective
tests cover **1,442 cases / 2,444,200 rows / 43,810,880 candidate pairs**;
**2,441,375 gradient pairs are bit-exact**, with maximum scaled error `6.30e-7`.
There are 551,079 subnormal field checks. The full suite passes **574 Python tests**
and native tests under Metal API Validation; one test is skipped. CPU-only checks
also pass. A separate strict prediction benchmark still has a failure described below.

The first large mean-pairwise benchmark exposed rounding feedback: native Metal
exponential/division errors produced a 0.567 maximum prediction difference after
20 rounds on 100,000 rows. Ranking now uses a two-float exponential expansion,
explicit FP32 rounding boundaries and a binary64 reciprocal. On **2,996,040**
inputs it matches both GPU software-binary64 and CPU binary64-exp sigmoid oracles
exactly. Apple's dynamic CPU `expf` differs on 4,962 of those inputs, by up to four
sigmoid ULPs, so exact CPU-libm identity is not claimed. The original failing
100,000-row case is now a regression test and has maximum prediction difference
`2.39e-6`; the benchmark tolerance was kept unchanged.

Mean training measurements use 8 samples per row, query size 128, one million
rows, 32 features, 20 rounds, depth 6, eight CPU threads, three alternating repeats
and per-round NDCG@10 evaluation. They include DMatrix construction and warmed
training, excluding data generation.

| Objective | CPU median | MPS median | CPU / MPS | Maximum prediction difference |
| --- | ---: | ---: | ---: | ---: |
| [NDCG](validation/benchmark-ranking-mean-ndcg-v2-m3.json) | 1.804 s | 2.409 s | 0.75× | `4.77e-7` |
| [MAP](validation/benchmark-ranking-mean-map-v2-m3.json) | 2.065 s | 2.499 s | 0.83× | `2.39e-7` |

Both pass the strict prediction/evaluation checks; the reports retain their slower
100,000-row cases too. MPS still loses these training comparisons.

**Known strict-equivalence failure:** the [million-row mean-pairwise run](validation/benchmark-ranking-mean-pairwise-v3-m3.json)
exceeds `atol=rtol=2e-5` on 1,544 of one million predictions. Maximum difference
is `0.000506`, RMS difference is `7.09e-6`, and every round's NDCG@10 agrees within
`4e-15` (final score approximately 0.994296). It remains explicitly marked
`prediction_equivalence_failed`, not a passed speedup result. The preceding
[run without evaluation](validation/benchmark-ranking-mean-pairwise-v2-m3.json)
retains the passing 100,000-row case and the same million-row failure. Gradient
and ranking-score equivalence do not guarantee identical long-run predictions.


### Unbiased ranking losses and position bias

Unbiased pairwise/NDCG/MAP now calculate loss corrections, per-pair bias costs,
ordered per-row bias statistics, cross-query sums and regularized bias updates
on Metal. Both top-k and mean sampling are supported. Only the small updated
bias vectors return to the objective's host-owned state for serialization;
large CPU bias-statistic arrays are no longer allocated for MPS. The new
`ranking_bias_dispatches` counter verifies completed GPU bias updates. Their GPU
duration is included in the fused ranking-gradient command.

Validation passes **583 Python tests**, all native suites under Metal API
Validation and CPU-only checks, with one skipped Python test. A new native suite
compares **2,592 successive objective calls / 852,768 rows / 118,368 serialized
bias values** across objectives, sampling methods, normalization switches and
bias regularizers. Maximum scaled gradient error is `5.73e-6`, bias error
`2.51e-7`. Four additional calls on **524,288 rows each** exercise two top-k or
three mean storage batches, including a second iteration using the updated bias.
Their maximum scaled gradient error is `3.19e-7`. Integration tests check that
bias state changes and agrees with CPU for short and 4,097-row queries.

The [NDCG](validation/benchmark-ranking-unbiased-ndcg-v1-m3.json) and
[MAP](validation/benchmark-ranking-unbiased-map-v1-m3.json) million-row benchmarks
pass strict prediction checks (maximum difference `2.39e-7`). With the same
mean-sampling configuration and NDCG@10 evaluation as above, NDCG takes **2.105 s
CPU / 3.648 s MPS (0.58×)** and MAP **2.286 s / 3.765 s (0.61×)**. These are slower
than CPU; the reports also retain slower 100,000-row cases.

**The [million-row unbiased mean-pairwise benchmark](validation/benchmark-ranking-unbiased-pairwise-v1-m3.json)
fails strict prediction equivalence:** maximum difference `0.421501`, RMS
`0.047376`, affecting 999,723 predictions under the configured tolerance.
Its NDCG@10 histories differ by at most `8.11e-5`; final CPU/MPS scores are
0.994216/0.994297. The 100,000-row case passes with maximum difference `4.48e-6`.
This failed result remains published and is not treated as a validated speedup.
Long-run mean-pairwise numerical agreement remains open work. Grouped ROC/PR-AUC
is still the remaining ranking-metric CPU path.


### Faster unbiased ranking

Eligible unbiased pairs are now compacted on GPU and corrected with an indirect
dispatch. This keeps software binary64 logarithms and bias arithmetic out of the
ordinary pair kernel. Gradient gathering retains the original pair order.
Cross-query bias sums use ordered 128-query blocks followed by a small reduction;
this changes the association of double additions across blocks, with the same
native gradient/state checks still passing.

For the million-row, 20-round mean-sampled workloads above, median MPS training
falls from **3.648 to 2.670 s for NDCG** and **3.765 to 2.665 s for MAP**—27% and
29% less wall time. CPU remains faster: **2.109 s (0.79×)** and **2.334 s (0.88×)**.
Maximum prediction differences remain `2.39e-7`. Raw intermediate and final
[NDCG](validation/benchmark-ranking-unbiased-ndcg-v3-m3.json) and
[MAP](validation/benchmark-ranking-unbiased-map-v3-m3.json) reports are retained.
The [unbiased mean-pairwise failure](validation/benchmark-ranking-unbiased-pairwise-v2-m3.json)
is unchanged; this optimization does not fix its numerical feedback issue.
All **583 Python tests and native suites** pass under Metal API Validation.

`XGBOOST_METAL_PROFILE_RANKING=1` enables optional per-encoder hardware timestamps
on supported devices. These are reported as raw GPU ticks; grouped-sort encoders
are not individually included. The [before](validation/profile-ranking-unbiased-v2.log)
and [after](validation/profile-ranking-unbiased-v3.log) profiles identified the
pair kernel as the main bottleneck. Instrumented runs are separate from the wall
timings above.


### Published CUDA workload replays and binary-column histograms

The [published benchmark notes](docs/PUBLISHED_BENCHMARKS.md) document NVIDIA's
4.15× single-P100 HIGGS result and the later eight-V100 benchmarks, including
source-code settings and discrepancies between the paper and its example results.
New scripts download/cache UCI data and run isolated CPU/MPS workers with fresh
matrices, alternating order, progress logs, clock checks and strict test-margin
comparisons. Both DMatrix setup and training time are recorded separately.

The initial **10.5-million-row HIGGS, 100-round pilot** takes **23.168 s CPU /
24.579 s MPS (0.94×)** and passes all prediction checks. The full **500-round
Covertype workload** takes **30.194 s / 92.504 s (0.33×)**, medians of three pairs.
All predicted test classes agree, but seven output margins fail strict comparison;
the reports retain that failure. Large CUDA-style training gains are still absent.

One- and two-bin dense features now accumulate exact integer gradients in registers
and reduce within SIMD groups before global atomic updates. This reduces contention
on binary columns. Validation passes **576 native histogram cases**, all other
native suites and **583 Python tests (one skipped)** under Metal API Validation.
The end-to-end performance is recorded below; isolated kernel gains are not
presented as training gains.


The full published **HIGGS workload (10.5M training rows, 1,000 rounds)** now
completes in **190.852 s CPU / 193.740 s MPS (0.985×)** in one paired run. Both
models reach **76.121% test accuracy**, with identical predicted classes and all
500,000 margins passing the strict check (maximum difference `3.34e-6`). This is
approximately a tie on the M3. See the [raw report](validation/published-higgs-1000-m3.json)
and [methodology](docs/PUBLISHED_BENCHMARKS.md) for setup time, memory, GPU timings
and differences from the historical hardware/software. The published CUDA ratios
serve to select workloads, not as promised speedups on an Apple chip.


The optimized **500-round Covertype** replay completes in median **33.320 s CPU /
70.289 s MPS (0.474×)** over three alternating pairs. MPS wall time is **24.0% lower**
than its 92.504 s baseline; median histogram GPU time falls from **35.172 to
15.616 s** with the same 21,500 histogram dispatches per run. CPU timing also
shifted between sessions, so the observed wall reduction is not a guaranteed gain.
All six before/after CPU and MPS prediction arrays are bit-for-bit identical.
The same seven CPU/MPS margin mismatches remain, with zero class disagreements and
89.4564% test accuracy. CPU remains about twice as fast on this workload.
[Raw runs](validation/published-covtype-500-lowbin-m3.json) and
[verification hashes](validation/published-covtype-lowbin-verification-m3.json)
are retained with the [source benchmarks and reproduction commands](docs/PUBLISHED_BENCHMARKS.md).

### Gradient-based sampling update

Moved gradient-based row sampling to Metal: regularized norms, stable sorting,
ordered FP32 prefixes, threshold search, CPU-compatible random draws and gradient
rescaling. Reduced-gradient leaf fitting replays sampling from the original split
gradients, including different target counts. Empty budgets are zeroed on GPU;
full budgets require no dispatch. The host still chooses one per-tree seed and
stages gradient storage.

All **601 Python tests (one skipped)** and native suites pass under Metal API
Validation; the independent CPU-only build also passes its compatibility checks.
The [native sampling report](validation/gradient-sampling-m3.txt) covers **498 exact
cases / 6.75 million rows**, plus six production sampler/replay checks. The
[coverage ledger](docs/CUDA_PARITY.md#gradient-based-sampling-behavior) explains the
CPU numerical contract and remaining synchronization/ordered-prefix costs.

### Sampling prefix optimization

The initial gradient sampler spent most GPU time on its ordered FP32 prefix.
Vector loads into registers now avoid scalar shared-memory stalls. Sampling stores
one prefix checkpoint per 256 rows; threshold search reconstructs only the queried
entries. Weighted quantiles also use vector loads/stores for their nonnegative,
normal-input prefix path. All additions retain CPU order, and all **601 Python
tests and native suites** still pass under Metal API Validation.

The [isolated kernel comparison](validation/sampling-prefix-kernels-m3.json) checks
every output against CPU in randomized trial order. At one million entries, the
full prefix falls from **80.79 ms to 24.98 ms**; sampling's checkpoint-only prefix
takes **7.89 ms**. These exclude sorting, staging and the rest of training.

End-to-end gradient-sampled training, 20 rounds, depth 6, 32 features,
`subsample=0.5`, eight CPU threads, medians of three alternating CPU/MPS pairs:

| Rows | CPU, latest | MPS, initial | MPS, optimized | CPU / optimized MPS |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0.166 s | 0.448 s | 0.307 s | 0.54× |
| 1,000,000 | 1.397 s | 2.972 s | 1.448 s | 0.96× |

The million-row MPS run is **51.3% shorter** than its initial measurement;
sampling GPU time falls from **1.684 s to 0.282 s**. Its CPU baseline changes from
1.405 s to 1.397 s across sessions. Both reports have zero CPU/MPS prediction
difference. CPU remains slightly faster; this is not a large GPU-over-CPU speedup.
[Initial measurements](validation/benchmark-gradient-sampling-v1-m3.json) and
[optimized measurements](validation/benchmark-gradient-sampling-v2-m3.json) retain
all timings, dispatch counters and clock checks.

```sh
.venv/bin/python scripts/benchmark.py --rows 100000 1000000 --features 32 \
  --rounds 20 --threads 8 --repeats 3 --subsample 0.5 \
  --sampling-method gradient_based --output results/my-gradient-sampling.json
git show 91e1c99:xgboost/src/common/metal_histogram.metal > results/sampling-baseline.metal
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc \
  scripts/metal_sampling_prefix_benchmark.mm -framework Metal -framework Foundation \
  -o results/metal_sampling_prefix_benchmark
results/metal_sampling_prefix_benchmark xgboost/src/common/metal_histogram.metal \
  results/sampling-baseline.metal > results/my-prefix-comparison.json
```

The [larger 3M-row × 128-feature run](validation/benchmark-gradient-sampling-wide-v1-m3.json)
also has zero CPU/MPS prediction difference, with medians **13.817 s CPU /
15.115 s MPS (0.91×)** over three pairs. Individual timings vary substantially:
CPU 13.78–19.33 s and MPS 12.38–17.25 s. The process lifetime peak RSS reaches
5.98 GB; system-wide swap usage is roughly 20–22 GiB during this session and cannot
be attributed solely to the benchmark. These measurements include fresh matrix
construction and first quantization, and do not establish a clean hardware limit
or a large-workload speedup. All wall/monotonic clock checks pass; no recorded run
crosses a detected sleep discontinuity.

```sh
.venv/bin/python scripts/benchmark.py --rows 3000000 --features 128 \
  --rounds 20 --threads 8 --repeats 3 --subsample 0.5 --float32-generation \
  --sampling-method gradient_based --output results/my-wide-gradient-sampling.json
```

### Grouped ranking ROC-AUC update

Grouped ROC now runs label sorting, pair counting, weighting, query normalization
and final aggregation on GPU. Parallel merges count concordant and tied pairs
without enumerating the quadratic pair matrix. Equal labels retain input order;
short queries, zero weights, squared-weight underflow/overflow and exceptional
prediction values retain the tested CPU behavior. Grouped PR-AUC remains CPU.

All **616 Python tests (one skipped)** and native suites pass, including
[272 grouped ROC cases](validation/ranking-roc-m3.txt) against upstream CPU and
234 million independently enumerated reference pairs. Three million-row analytic
cases each represent 500 billion pairs and exercise 64-bit counts. The separate
CPU-only build also passes its compatibility checks.

Public metric evaluation on fixed base margins, warm prediction caches, eight CPU
threads, medians of three alternating pairs:

| Rows | Rows per query | CPU | MPS | CPU / MPS |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 128 | 1.758 ms | 1.221 ms | 1.44× |
| 100,000 | 1,024 | 9.331 ms | 1.304 ms | 7.15× |
| 1,000,000 | 128 | 17.172 ms | 7.922 ms | 2.17× |
| 1,000,000 | 1,024 | 84.650 ms | 8.796 ms | **9.62×** |

All reported CPU/MPS scores agree exactly. These measurements exclude training,
matrix construction and runtime compilation. First-evaluation times, inputs'
hashes, memory snapshots, counters and clock checks are in the
[metric report](validation/benchmark-ranking-roc-metric-v1-m3.json). The gain includes
the different pair-counting algorithm; it is not a general hardware speed ratio.

The [complete training benchmark](validation/benchmark-ranking-roc-training-v1-m3.json)
uses one million rows, 32 features, 1,024-row queries, pairwise top-k ranking with
32 pairs per sample, depth 6 and 20 rounds, evaluating grouped ROC every round.
Fresh DMatrix construction and quantization are included. Median time is
**4.585 s CPU / 3.544 s MPS (1.29×)** over three alternating pairs; the maximum
prediction difference is **7.15e-7** and metric histories pass the existing strict
comparison. This is the observed training gain for that workload.

```sh
.venv/bin/python scripts/benchmark_ranking_auc.py --rows 100000 1000000 \
  --query-sizes 128 1024 --threads 8 --repeats 3 --output results/my-ranking-roc.json
.venv/bin/python scripts/benchmark.py --rows 1000000 --features 32 \
  --objective rank:pairwise --query-size 1024 --rank-pairs 32 --rank-method topk \
  --eval-metric auc --rounds 20 --threads 8 --repeats 3 \
  --output results/my-ranking-roc-training.json
```

### Grouped ranking PR-AUC update

Grouped PR-AUC now performs label validation, stable prediction sorting,
segmented binary64 prefix sums, continuous precision/recall integration and
query aggregation on Metal. Prefix scans reset at query boundaries, including
boundaries inside SIMD groups, threadgroups and recursive scan levels. Empty,
one-class and zero-weight queries retain CPU behavior. This completes the grouped
PR metric entry left open in the preceding updates.

[Native validation](validation/ranking-pr-m3.txt) covers **297 cases / 14,198,354
rows / 4,660 query results**, comparing upstream CPU and an independent continuous
integral. The maximum observed absolute difference is **9.10e-15**. Tests include
strided inputs, ties, extreme query weights, subnormal labels and weights,
million-row queries and six invalid-label checks. All **633 Python tests (one
skipped)** and native suites pass with Metal API Validation; the separate CPU-only
build also passes its compatibility checks. Performance measurements follow
separately; full CUDA parity remains incomplete.

Grouped PR performance on this M3 (eight CPU threads, three alternating pairs):

| Rows | Rows per query | CPU metric | MPS metric | CPU / MPS |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 128 | 1.314 ms | 2.376 ms | 0.55× |
| 100,000 | 1,024 | 1.751 ms | 2.408 ms | 0.73× |
| 1,000,000 | 128 | 11.953 ms | 18.251 ms | 0.65× |
| 1,000,000 | 1,024 | 14.673 ms | 14.992 ms | 0.98× |

The [metric report](validation/benchmark-ranking-pr-metric-v1-m3.json) times public
evaluation of fixed base margins with warm prediction caches; matrix construction,
training and runtime compilation are excluded. The [complete training report](validation/benchmark-ranking-pr-training-v1-m3.json)
uses one million rows, 32 features, binary relevance, 1,024-row queries, pairwise
top-k ranking with 32 pairs per sample, depth 6 and 20 rounds, evaluating PR each
round. Including fresh DMatrix construction and quantization, median time is
**2.152 s CPU / 2.976 s MPS (0.72×)**. CPU wins these PR workloads; the earlier ROC
speedup does not extend to PR. All existing prediction and metric-history checks
pass, as do dispatch and wall/monotonic-clock checks. Raw repeats and memory
snapshots are retained; these results do not establish an isolated hardware limit.

```sh
.venv/bin/python scripts/benchmark_ranking_auc.py --metric aucpr \
  --rows 100000 1000000 --query-sizes 128 1024 --threads 8 --repeats 3 \
  --output results/my-ranking-pr.json
.venv/bin/python scripts/benchmark.py --rows 1000000 --features 32 \
  --objective rank:pairwise --query-size 1024 --rank-pairs 32 --rank-method topk \
  --eval-metric aucpr --rounds 20 --threads 8 --repeats 3 \
  --output results/my-ranking-pr-training.json
```

### Column sampling update

Column sampling now runs feature initialization, random generation, uniform and
weighted selection, stable priority ordering and final feature-ID sorting on Metal.
The existing tree/level/node sampling rules and parent random stream are preserved.
Weights chosen to create exact CPU score ties are included in native checks;
weighted keys and selected IDs agree exactly in the tested cases. The logarithm
compatibility check covers 8.09 million inputs, including zero, one and table
boundaries. See the [behavior and limits](docs/CUDA_PARITY.md#column-sampling-behavior).

The host still owns feature vectors and cached level sets. GPU random generation
is sequential to retain CPU random draws; small feature sets can be slower due
to dispatch and staging. `column_init_dispatches`, `column_sampling_dispatches`
and `column_sampling_gpu_time_ns` expose actual GPU work. Wide-workload timings
will be published separately from this coverage update.

[Validation report](validation/column-sampling-m3.txt): **673 Python tests passed,
one skipped**, all native suites passed with Metal API Validation, and the
independent CPU-only build passed its compatibility checks.

Column-sampled training before the fused-kernel optimization (eight CPU threads,
20 depth-6 rounds, three alternating CPU/MPS pairs, including fresh matrix
construction and quantization):

| Workload | Column sampling rates: tree / level / node | CPU | MPS | CPU / MPS |
| --- | --- | ---: | ---: | ---: |
| 1M rows × 128 features, uniform | 0.8 / 0.8 / 0.8 | 3.430 s | 4.001 s | 0.86× |
| 100K rows × 2,048 features, weighted | 0.8 / 0.8 / 0.5 | 7.508 s | 7.800 s | 0.96× |

Both have zero CPU/MPS prediction difference. The [million-row report](validation/benchmark-column-million-v1-m3.json)
and [wide report](validation/benchmark-column-wide-v1-m3.json) retain all timings,
clock checks and counters. Each run completes 1,400 or 1,356 column samples,
respectively. Their median column-sampling GPU time is 59.1 ms / 353.4 ms; these
intervals exclude host staging and synchronization. Peak process RSS is 3.18 GB /
5.33 GB. System-wide swap usage is about 18–21 GiB and cannot be attributed solely
to the benchmark. Results describe this session, not an isolated hardware limit.

The common case of at most 2,048 candidate columns now fuses random generation,
weighted priority ordering, selection and final ID sorting into one compute
kernel, removing intermediate permutation/sort buffers and separate encoders.
The larger-column path remains GPU-based. Native tests exercise both sides of
that boundary; all selected IDs, random draws and weighted scores still match CPU
exactly in the tested cases.

The [fused-path validation](validation/column-sampling-fused-m3.txt) passes all
**673 Python tests (one skipped)** and native suites with Metal API Validation,
plus the unchanged CPU-only build checks.

The first fusion experiment reduced million-row MPS time from **4.001 s to
3.767 s**, with CPU at 3.363 s, but regressed weighted wide training from
**7.800 s to 8.458 s**, with CPU at 7.411 s. Sampling GPU time changed from
59.1 to 55.1 ms for uniform sampling and from 353.4 to 437.4 ms for weighted
sampling. Reports retain both [the modest improvement](validation/benchmark-column-million-v2-m3.json)
and [the regression](validation/benchmark-column-wide-v2-m3.json); predictions
still match CPU exactly. CPU timings also vary between runs, so these are not
isolated measurements of the full-training effect of fusion.

The production dispatch now restricts fusion to **uniform** sampling. Weighted
sampling retains the earlier staged GPU implementation. The specialized uniform
kernel also keeps its MT19937 seed recurrence in a register and avoids unused
scratch buffers and the flag-clearing encoder. The current path therefore differs
from the first fusion experiment above; final timings follow separately.

[Specialized-path validation](validation/column-sampling-uniform-m3.txt) passes
all 530 native column cases and all 673 Python tests (one skipped), plus CPU-only
compatibility checks. Other native implementations are unchanged from the preceding
full validation.

Final dispatch policy, medians of three alternating pairs on the same workloads:

| Workload | CPU | MPS | CPU / MPS | Column-sampling GPU time |
| --- | ---: | ---: | ---: | ---: |
| 1M × 128, uniform sampling | 3.443 s | 3.811 s | 0.90× | 44.2 ms |
| 100K × 2,048, weighted sampling | 7.326 s | 8.320 s | 0.88× | 348.1 ms |

The [final million-row report](validation/benchmark-column-million-v3-m3.json)
and [final weighted report](validation/benchmark-column-wide-v3-m3.json) have zero
reported CPU/MPS prediction difference. The uniform sampler's GPU time is **25%
lower** than the first implementation (59.1 → 44.2 ms); its whole MPS training
median falls about **4.7%** (4.001 → 3.811 s). Restoring the staged weighted path
removes the measured sampling-kernel regression (437.4 → 348.1 ms), although its
whole-training median remains above the original session's result. Neither final
workload beats CPU. Broader training costs and session variability remain; system
swap is still about 20 GiB in the final wide run. This is a targeted sampling
improvement, not a claim of general GPU training acceleration.

```sh
.venv/bin/python scripts/benchmark.py --rows 1000000 --features 128 \
  --rounds 20 --threads 8 --repeats 3 --float32-generation \
  --colsample-bytree .8 --colsample-bylevel .8 --colsample-bynode .8 \
  --output results/my-column-million.json
.venv/bin/python scripts/benchmark.py --rows 100000 --features 2048 \
  --rounds 20 --threads 8 --repeats 3 --float32-generation --feature-weights \
  --colsample-bytree .8 --colsample-bylevel .8 --colsample-bynode .5 \
  --output results/my-column-wide.json
```

### Numeric sketch finalization update

Final merging and pruning of retained numeric sketch levels now execute on Metal.
This expands preprocessing coverage; ingestion, queue sorting and streaming
compaction remain CPU work. Final cut materialization remains on the host in both
this backend and the vendored CUDA implementation. The
[native/integration validation](validation/sketch-finalization-m3.txt) checks exact
summary fields and cut bits, including weighted and external-page inputs.

Initial measurements use the same column-sampling workloads immediately above,
with fresh matrices and three alternating CPU/MPS pairs:

| Workload | CPU | MPS | CPU / MPS | Sketch GPU time |
| --- | ---: | ---: | ---: | ---: |
| 1M × 128 | 4.305 s | 4.667 s | 0.92× | 148.6 ms |
| 100K × 2,048 | 8.724 s | 8.466 s | 1.03× | 55.5 ms |

The [million-row report](validation/benchmark-sketch-million-v1-m3.json) and
[wide report](validation/benchmark-sketch-wide-v1-m3.json) record zero final-pair
prediction differences. Wide CPU times range from 7.39 to 10.15 s and MPS times
from 8.04 to 9.36 s. With this variability and substantial system swap, the small
median lead does not establish a reliable training speedup. The preceding
column-sampling reports remain the baseline before GPU sketch finalization.

Pruning initially used software binary64 division rounded to FP32. It now uses
a native FP32 estimate corrected by exact integer residuals, retaining one final
round-to-nearest-even step including subnormals. The
[division validation](validation/sketch-fp32-division-m3.txt) checks 5,830,328
arithmetic inputs, all 942 sketch cases and all 700 Python tests (one skipped).
The first measurements above precede this optimization.

After exact FP32 division, the [million-row report](validation/benchmark-sketch-million-v2-m3.json)
measures **4.501 s CPU / 5.111 s MPS**, with sketch GPU time **121.9 ms**;
the [wide report](validation/benchmark-sketch-wide-v2-m3.json) measures
**10.014 s / 9.758 s**, with sketch GPU time **54.9 ms**. The million-row kernel
improves about 18%, but these full-training timings do not establish a speedup
from the optimization. The reports retain the variability and memory conditions.

The next implementation assigns one SIMD group to each column and parallelizes
pruning's rank searches and removal of duplicate selections. Merge/rank repair
retain their original order. GPU validation of monotonic rank keys selects a
serial GPU path for irregular summaries. Native summaries and all 700 Python
tests still pass; measurements of this change follow separately.

Completed measurements for parallel pruning: [1M × 128](validation/benchmark-sketch-million-v3-m3.json)
takes **4.300 s CPU / 4.789 s MPS**, with sketch GPU time **47.2 ms**;
[100K × 2,048](validation/benchmark-sketch-wide-v3-m3.json) takes
**7.680 s / 8.322 s**, with sketch GPU time **79.8 ms**. Both report zero
final-pair prediction difference. The million-row sketch stage is about 3.15×
faster than its first GPU version, while the wide sketch stage regresses from
55.5 to 79.8 ms. Neither final training median beats CPU. Further optimization
is deferred in favor of completing missing GPU operations; all three experiments
are retained.

### Objective initialization coverage update

Expectile's initial per-row gradients/Hessians now run on Metal, as does the
absolute-error initializer's broadcast of each target mean to its prediction
column. These use separate completed-dispatch counters. Means, weight sums and
stump fitting were already on Metal; the small final intercept additions and
ordering stay on the host, as they do in the CUDA source.

Public checks compare estimated intercepts and predictions with CPU, including
weighted/constant/multiple-target data and continued training from saved models.
Explicit intercepts and zero-total-weight absolute-error inputs must skip the
new initialization operations. Full CUDA coverage remains incomplete; this update
closes two specific initialization gaps. No new performance experiments accompany
this coverage work.

[Validation](validation/objective-initialization-m3.txt): all native suites,
**715 Python tests (one skipped)** and the independent CPU-only build pass with
Metal API Validation enabled for GPU tests.


### Objective labels and base-score conversion coverage

Objective label predicates now run on Metal for logistic, gamma, squared-log,
Poisson/Tweedie and multiclass objectives. Logistic/gamma intercept validation
and log/logit base-score conversion, including Cox base scores, also run on
Metal. Existing error messages and checked-input behavior are preserved.

[Validation](validation/objective-domain-m3.txt): **778 Python tests pass (one
skipped)**, all native suites pass with Metal API Validation, and the independent
CPU-only build passes. New native checks cover 120,093,300 individual label
predicates, 576 strided validation cases and 4,570,936 base-score comparisons.
Finite margin conversions match the tested CPU's bits; exceptional values and
invalid-input handling are checked separately. These results are correctness
checks, not performance measurements. Generic metadata validation and other
coverage gaps remain listed in the ledger.


### SHAP coverage update

SHAP feature contributions and interaction matrices now execute on Metal using
the vendored eight-point quadrature algorithm. This includes repeated-feature
paths, scalar/vector leaves, categorical and missing-value decisions, dropout
weights, base margins and interaction finalization. Model preparation, input
staging and categorical recoding remain host work and are recorded in the ledger.

[Validation](validation/shap-m3.txt): **865 Python tests pass (one skipped)**,
all native suites pass with Metal API Validation, and the independent CPU-only
build passes. The 87 new public cases include independent feature-subset
expectations and CPU comparisons through depth 130. Native checks cover 576 cases,
buffer guards, overlapping storage and bounded batches. No SHAP speedup is claimed;
coverage and correctness remain the priority.

The next concrete training gap is final leaf refitting when custom objectives
provide separate split and value gradients: leaf-gradient accumulation and
regularized leaf-weight calculation still use the CPU.


### Reduced-gradient leaf fitting coverage update

Final vector-leaf fitting now runs on Metal when an objective supplies separate
split and value gradients. The GPU sums gradients using resident row partitions,
keeps those sums across external-memory pages and computes regularized/clipped
weights. The host receives only the final leaf values for the model update.
Earlier generic gradient copies and allocations still need coverage work.

[Validation](validation/leaf-fitting-m3.txt): **931 Python tests pass (one
skipped)**, all native suites pass with Metal API Validation, and the independent
CPU-only build passes. New coverage includes 66 integration tests and 304 native
cases covering 24,327,036 source gradient pairs. The tested sum fields and finite
weight results match CPU exactly; exceptional values are checked separately.
Sampling, zero/empty leaves, strided inputs, external pages, L1/L2 penalties,
clipping and saved-model continuation are covered. No performance benchmark was
added. Linear coordinate descent remains a major missing CUDA operation.


### Linear coordinate descent coverage update

The bias/feature gradient reductions and residual updates used by CUDA's linear
coordinate updater now run on Metal. Use
`booster="gblinear", updater="coord_descent", device="mps"`. All five feature
selectors are supported. Selectors, scalar weight updates, CSC preparation and
linear prediction remain host work, matching CUDA's placement. The default
`shotgun` algorithm also remains host. The numerical reference is CPU, including
updated residuals for greedy/thrifty selection; CUDA intermediate bits can differ.

[Validation](validation/linear-coordinate-m3.txt): **1,066 Python tests pass (one
skipped)**, all native suites pass, and the separate CPU-only build/check passes.
New coverage includes 136 public cases and 49 native cases using 4,118,809 source
gradient pairs, with Metal API Validation enabled for the native linear tests.
The tests cover selectors, multiple outputs, missing/sparse inputs, L1/L2
penalties, saved-model continuation, million-row reductions, duplicate CSC rows,
exceptional arithmetic and buffer guards.

Coverage and correctness remain the priority; no new performance measurements
were run. Sketch ingestion/preprocessing, `approx` training, sensitive AFT
corrections, generic state/metadata operations, device interoperability and
distributed execution still need work. Full CUDA parity is not complete.


### Sketch weighting and streaming compaction coverage update

Metal now computes group/Hessian weights, weighted summaries of sorted numeric
columns, and streaming level merges/pruning/carry propagation. The existing GPU
finalizer consumes those retained levels. Row collection, queue sorting and
queue-to-summary formation still run on CPU, as does category collection. The
sorted-column implementation is validated through production cut creation; the
public `approx` updater still needs its remaining GPU integration.

The [validation record](validation/sketch-ingestion-m3.txt) includes **1,066 Python
tests passing (one skipped)** and all native suites. New native checks cover
**48 weight cases, 211 sorted columns, 24 streamed sketches, 32 batched updates
and 8 three-page container cases**. Public sketch tests preserve exact CPU cut
bits and now assert the new completed-operation counters. Performance experiments
remain deferred; state allocation and staging costs are not optimized here.
The independent CPU-only build/check also passes. After restoring the original
row-weight size check, the final native ingestion suite and all 27 public sketch
tests were rechecked successfully.


### Approximate training and CUDA alignment update

`tree_method="approx", device="mps"` now runs with GPU Hessian extraction, gradient
copies, root statistics and the existing Metal tree operations. Cuts use original
Hessians before row sampling, once per updater call, and each tree gets fresh row
and column sampling. This follows CUDA's updater order, which differs from CPU
approx. Fresh external-page gradient indices now preserve MPS for GPU binning.

CUDA source is the implementation authority. Existing CPU-based sketch,
sampling/RNG, quantization, leaf sums and linear-selector choices still need
alignment; they are explicitly listed in the [ledger](docs/CUDA_PARITY.md).
CPU equality is tested where algorithms agree. Fixed-gradient tests check the
CUDA cut/sampling order independently of model-prediction equality. No tests have
yet run against actual NVIDIA execution, and full CUDA parity remains unfinished.

[Validation](validation/approx-gradient-state-m3.txt) adds 141 integration cases
and 276 native gradient-state cases. The full run passes **1,206 Python tests
(one skipped)**, every native suite and the independent CPU-only build/check.
The native checks include over 25 million
copied pairs and extracted Hessians, raw NaN bits, layouts, padding, aliases and
read-only inputs. Coverage and correctness remain the priority; no new performance
benchmark accompanies this change.


### CUDA gradient quantization alignment

Histogram quantization now uses CUDA's separate positive/negative magnitude sums,
`CreateRoundingFactor<double>` scale formula, truncation toward zero, and minimum
integer one for tiny positive Hessians. This replaces the earlier maximum-bound,
round-to-nearest scheme. Bounds and scales execute on Metal; no CPU gradient
reduction is introduced.

[Validation](validation/cuda-quantizer-m3.txt) checks over 19.3 million quantized
values against scalar CUDA formulas, 1,210 scale boundaries, and the separate
713,814-conversion/576-histogram suite. All 1,206 Python regression tests pass
(one skipped), plus four new training tests that intentionally differ from CPU:
`hist` and `approx` preserve tiny positive Hessians according to CUDA's rule.
All native suites and the CPU-only build/check pass, without tolerance changes. This is formula alignment, not an
NVIDIA execution comparison. Quantization still needs to move before sampling;
integer gradient state must also feed sampling, root/leaf statistics and the
histogram cache. No new performance benchmark was run.
