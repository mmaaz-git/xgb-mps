# CUDA → Metal coverage

The project goal is behavioral coverage of **all XGBoost operations accelerated by
CUDA**, with measured performance. Passing training tests alone does not establish
that those operations execute on the GPU. Metal compute is the implementation
technology; `mps` is the public device selector. Apple's Metal Performance Shaders
library is not a replacement for XGBoost's custom CUDA kernels.

Current architecture: CPU owns XGBoost's data structures and orchestrates training;
selected stages dispatch Metal kernels. Remaining CPU execution is explicitly
listed below. This is not yet a complete GPU backend.

| CUDA subsystem | Principal upstream source | Metal status |
| --- | --- | --- |
| Gradient histogram accumulation | `tree/gpu_hist/histogram.cu`, `histogram.cuh` | Implemented: dense/sparse, quantized 64-bit sums, shared-memory dense path |
| Gradient quantization | `tree/gpu_hist/quantiser.cuh` | Implemented with a different bounded fixed-point scheme; GPU maximum/finite reduction, exact integer scale selection and bit conversion; scales remain in GPU buffers; eligible raw gradients are read through guarded shared host views |
| Histogram subtraction, cache management | `tree/gpu_hist/histogram.cuh` | Metal-owned shared cache, GPU integer-to-binary64 conversion, external-page accumulation, subtraction and GPU feature gathering; host cache policy/metadata |
| Split evaluation, categorical partitions, constraints | `tree/gpu_hist/evaluate_splits.cu`, `evaluate_splits.cuh`, `multi_evaluate_splits.cu` | Metal scalar/vector candidate scoring, category ordering and missing directions; software binary64, exact bounded scans, conservative numerical-candidate screening, monotonic bounds and pooled vector weights; host interaction masks and tree updates |
| Row partition and final leaf assignment | `tree/gpu_hist/row_partitioner.cu` | Metal stable classify/prefix/scatter with cooperative scans for large nodes, resident row IDs, GPU leaf positions with guarded shared gradient inputs/output slices; supports numerical/categorical splits and missing values |
| Sampling and gradient sums | `tree/gpu_hist/sampler.cu`, `tree/updater_gpu_hist.cu`, `common/random.cu` | Metal uniform and gradient-based row sampling/replay, including CPU-compatible draws, norms, sorted thresholds and rescaling; scalar/vector root-gradient sums and dense root histogram sums on Metal; column initialization, uniform/weighted selection and final feature ordering now use Metal |
| Squared/logistic/squared-log/gamma/pseudo-Huber/Poisson/Tweedie/hinge/multiclass/expectile gradients | `objective/regression_obj.cu`, `hinge.cu`, `multiclass_obj.cu` | Metal kernels with guarded synchronous host-memory sharing where eligible; means/weighted means and Newton stump fitting use Metal; label validation and other objective-specific initialization remain CPU |
| Sigmoid/exp/hinge/softmax/argmax/expectile prediction transforms | Same objective files | Metal kernels; identity transforms require no dispatch |
| Absolute-error and quantile objectives, associated statistics | `objective/regression_obj.cu`, `quantile_obj.cu` | Metal per-iteration residual scales, gradients and quantile prediction sorting; absolute-error initial weight sums, means/stump fitting and initial quantile estimation use Metal; other initialization work remains CPU |
| Ranking losses, pair construction, normalization | `objective/lambdarank_obj.cu` | Prediction order and NDCG ideal-label order use stable grouped Metal sorting; query-weight sums use Metal. NDCG discounts, ideal-gain calculation and ranking label validation also use Metal; biased top-k/mean pair construction, gradients, MAP prefixes and query normalization execute on Metal. Unbiased loss corrections, per-row bias costs, cross-query aggregation and regularized bias updates also execute on GPU; small bias vectors return to host for serialization |
| AFT survival loss | `objective/aft_obj.cu`, `common/survival_util.h` | Metal normal/logistic/extreme gradients and prediction/base-score transforms; GPU identifies small censored probabilities for original CPU gradient recomputation, exposed through correction counters/time. Full GPU coverage remains incomplete |
| Base score estimation, leaf value fitting | `tree/fit_stump.cu`, `common/stats.cu` | Means/weighted means, quantiles/medians, Newton stump gradient aggregation and unregularized weights on Metal; other initialization work and leaf fitting remain CPU |
| Batch/inplace/leaf tree traversal | `predictor/gpu_predictor.cu` | Metal: scalar/vector leaves, category bitsets, missing values, tree ranges, dropout weights; CPU input views stage dense chunks and recode categories |
| Prediction cache from leaf IDs | `predictor/predictor.cu` | Metal; supports strided target slices and negated sampling positions |
| SHAP contributions and interactions | `predictor/gpu_predictor.cu`, `predictor/interpretability/shap.cu` | CPU |
| RMSE/RMSLE/MAE/MAPE/logloss/error/Poisson/gamma/Tweedie/pseudo-Huber metrics | `metric/elementwise_metric.cu` | Metal loss calculation and scaled block reductions; final small reduction in host FP64 |
| Multiclass error and logloss | `metric/multiclass_metric.cu` | Metal loss calculation and scaled block reductions; final small reduction in host FP64 |
| Quantile and expectile metrics | `metric/elementwise_metric.cu` | Metal loss calculation, subnormal-preserving products and binary64 block reductions; small final host sum |
| AFT likelihood and interval-regression accuracy | `metric/survival_metric.cu` | GPU probability calculations and binary64 block sums; small final host reduction |
| ROC-AUC | `metric/auc.cu` | Binary, multiclass and multilabel stable batched sorting, weighted binary64 prefixes and tied-score trapezoids on Metal; per-output normalization and aggregation remain host |
| PR-AUC | `metric/auc.cu` | Binary, multiclass and multilabel stable batched sorting, weighted binary64 prefixes and stable continuous integration on Metal; final per-output degeneracy checks and aggregation remain host |
| Ranking metrics | `metric/rank_metric.cu`, `metric/auc.cu` | NDCG/MAP/precision sorting, weighted query scoring and query-sum reduction execute in one Metal command. NDCG discounts/inverse ideal gains and ranking label validation also use Metal; cache values are read back to existing host-owned arrays. Grouped ROC uses GPU stable label order, merge-based pair counts, query normalization and final aggregation. Grouped PR uses GPU label validation, stable score order, segmented binary64 prefixes, continuous PR integration, valid-query checks and aggregation |
| Quantile sketching, cuts, weighted/category handling | `common/quantile.cu`, `data/gradient_index.cu` | Final numeric sketch level merging, rank repair and pruning use Metal; ingestion, queue sorting and streaming compaction remain CPU. Final cut materialization is host-side in both the CUDA and Metal implementations. Numeric/categorical bin assignment and hit counts for SparsePage-backed matrices use Metal; adapter-built QuantileDMatrix bins remain CPU |
| Compressed GPU matrix, external memory, iterators | `data/ellpack_page.cu`, `iterative_dmatrix.cu`, `quantile_dmatrix.cu` | Host GHistIndexMatrix remains authoritative; page-owned Metal copies and dense GPU transpose cache persist for each source page lifetime |
| Categorical recoding, metadata manipulation, validation | `data/cat_container.cu`, `metainfo.cu`, `array_interface.cu` | CPU |
| Linear coordinate-descent booster | `linear/updater_gpu_coordinate.cu` | `gblinear` rejected for `mps` |
| Device arrays, scans, sorts, reductions, random generation | `common/*.cu`, `common/*.cuh` | Histogram/metric/gradient reductions, generic float and optional-weight sums, checked exact scans, category/quantile-row sorting, stable column radix sorting, stable grouped ranking sorting, recursive integer/binary64 prefix scans and sampling-specific random generation implemented; other general Metal primitives missing |
| Distributed GPU collectives | `collective/comm.cu`, `collective/coll.cu`, `collective/aggregator.cuh` | Distributed `mps` training rejected |
| CUDA array interface / CuPy / cuDF / CUDA streams | `c_api/c_api.cu`, `data/device_adapter.cu` | NVIDIA-specific interfaces cannot consume Metal memory; corresponding Apple-device interoperability is missing |
| CUDA contexts, allocators, stream lifetimes | `common/cuda_*`, `common/device_*`, `common/host_device_vector.cu` | Shared-memory Metal runtime, guarded host-memory views for ordinary gradients, quantization and leaf assignment, model-local row workspaces, reused command queue and histogram buffers; persistent end-to-end device ownership and asynchronous scheduling missing |

Paths in the table are relative to `xgboost/src` and denote starting points, not
an exhaustive symbol audit. [The generated source inventory](../validation/cuda-inventory.json)
contains **all 105 `.cu`/`.cuh` files** under that directory; the suffix alone does not
prove that every function in a file is CUDA-accelerated (Cox gradients, for example,
are CPU-only upstream). Header-only CUDA and vendored GPU dependencies also require
review. This inventory is a scope ledger, not a claim of full coverage.

## Acceptance criteria

For each ported operation:

1. Test the actual GPU output against independent mathematical references and/or
   upstream CPU behavior, including edge cases and documented tolerances.
2. Check a completed-operation counter so a CPU fallback cannot pass the test.
3. Validate the integrated API, model persistence, and CPU-only build.
4. Measure cold and warm time, transfers/staging, dispatches and GPU duration where
   useful. Compare identical inputs and parameters, including preprocessing.
5. Keep limitations visible. A feature's training output matching CPU does not mean
   its gradients, prediction, metrics, or preprocessing have been ported.

Performance work must reduce the full training/prediction time. Apple GPU and
NVIDIA GPU speedups cannot be assumed equal; publish hardware, workload, CPU thread
count, repeated measurements, and accuracy alongside any speedup claim.

## Next major training work

Quantized pages, row indices and histogram caches now persist on Metal buffers.
Split selection, histogram conversion/accumulation/subtraction, uniform/gradient-based sampling
and root-gradient/stump reductions also execute on Metal. Host orchestration still synchronizes at each tree level;
ordinary gradient calls can share host arrays directly, while other gradient and prediction-state paths still stage data between host arrays and GPU buffers.
Keeping those arrays resident and reducing synchronization are the next architectural
steps. Host model construction and serialization can remain on the CPU.

Split tests include categorical partitions, missing directions, monotonic and
interaction constraints, vector leaves and external-memory pages. Column sampling
now uses GPU initialization, random draws, selection and ordering. Remaining
training-state updates and more efficient external-memory staging are still required.

Sensitive AFT gradient corrections, SHAP, remaining sketch
construction, linear training and collectives are separate coverage requirements.
The observed prediction speedups do not establish any of those as complete.

## Gradient-based sampling behavior

The Metal implementation follows the upstream CPU sampler: sequential FP32
target accumulation, sorted FP32 prefix sums, the same threshold search,
Apple libc++ random draws and reciprocal rescaling. The tested Apple Clang CPU
build contracts the regularized squared-gradient expression as an FMA; the Metal
norm kernel preserves that rounding and implements subnormal handling and correctly
rounded square roots. Native tests compare actual CPU-library norms and thresholds,
then independently replay the standard-library draws, checking results bit for bit.
This is a CPU behavioral contract, not a claim of identical CUDA masks: the CUDA
sampler uses quantized gradients and a different random stream.

All per-row sampling calculations execute on GPU. The host selects one seed per
tree, supplies matrix metadata and waits for the result. Input/output storage is
still staged; reduced-gradient replay recomputes the GPU norms/sort/threshold from
the original split gradients. The ordered FP32 prefix preserves CPU decisions but
limits parallelism. The optimized kernel reads four sorted values into registers
at a time and writes one checkpoint per 256 rows. Threshold search reconstructs
only its queried entries from those checkpoints, preserving the same addition
order. Persistent sampling state and further threshold/sort optimization remain
performance work.

## Survival precision limitation

AFT expressions subtract nearby CDF values. A few binary64 ULP of difference
between CPU libm and GPU transcendental approximations can change gradient/Hessian
clipping drastically in very small probabilities. Censored rows with computed
probability at or below `1e-5` are compacted by the GPU and recomputed with the
original CPU formula. `survival_cpu_corrections` counts rows; `survival_cpu_time_ns`
measures that host work. These counters are separate from completed GPU dispatches.
No hidden whole-objective CPU fallback is used. Removing the correction path while
preserving a defensible numerical contract remains required for full AFT coverage.


## Ranking metric numerical behavior

Precision preserves CPU FP32 label × query-weight products before binary64 summation.
NDCG preserves the CPU's unweighted fallback when inverse ideal gain is nonpositive;
MAP's zero-hit fallback is multiplied by the query weight. These details can differ
from upstream CUDA degeneracy handling. The reference contract here is upstream CPU.

MAP uses parallel integer hit counts/prefixes for exact binary labels. Upstream
validation also admits labels within epsilon of zero or one, including tiny negatives.
For those queries, the GPU uses the CPU's sequential binary64 evaluation order:
reordering near-zero signed terms could otherwise change whether the zero-hit fallback
is selected. This path remains on the GPU and is slower for long nonbinary queries.
Empty-query and top-k-zero native tests preserve CPU NaN/fallback behavior; the
existing public metric finalization still reports invalid scores.

The fused metric command reads back only score and weight totals. NDCG cache
creation and grouped ROC/PR calculation also execute on Metal; their completion
does not establish coverage of the other training, data and prediction operations.

Grouped ranking ROC now sorts labels stably on GPU and counts concordant/tied
pairs during parallel merges of prediction keys. This preserves the CPU rule that
equal-label pairs participate in input order. It avoids the CPU's quadratic pair
enumeration: counts require `O(N log² m)` work for `N` rows and maximum query size
`m`, followed by integer reductions and software-binary64 normalization. Query
weights are squared in FP32, and half-weight products retain FP32 subnormal
rounding. Queries shorter than three rows are excluded; zero-weight queries retain
CPU's zero score and valid count. Overflowing squared weights raise an error.

CPU comparison uses a `2e-12` absolute/relative tolerance in the tested cases;
binary64 weighting/aggregation can differ from repeated CPU additions for very
large, irregularly weighted queries. Bitwise equivalence is not a universal claim.
Million-row analytic cases use unit weights and exact expected scores, exercising
pair counts beyond 32 bits without running a trillion-comparison CPU oracle.
Input arrays remain staged and host-owned. Only the aggregate score/valid count
normally returns to the host; optional native-test diagnostics expose per-query
results. The public metric retains distributed/final ratio handling on the host.

Grouped PR-AUC uses stable prediction sorting and recursive segmented binary64
scans. Query IDs accompany block totals so carries stop at every query boundary,
including empty queries and boundaries inside SIMD groups or recursive scan
blocks. The GPU preserves FP32 label/weight products and CPU's binary64 first
negative term. A separate query flag checks positivity of the original FP32
negative terms, so underflow cannot incorrectly make a degenerate query valid.
Range validation executes on GPU, including negative subnormal labels. Invalid
queries contribute neither score nor valid count; zero-weight PR queries remain
invalid, unlike the CPU's zero-weight grouped ROC convention.

The continuous PR integral uses the same stable `log1p`-equivalent formulation
as the existing Metal classification metric. Native production tests cover 14.2
million rows, query weights spanning large exponent ranges, fractional labels,
subnormal products, nonfinite/degenerate queries, strides and three scan levels.
The observed maximum difference from CPU/independent integral references is
`9.1e-15`; the comparison tolerance is `2e-12` absolute/relative. This is tested
agreement rather than a universal bitwise guarantee. Scores/valid counts are
aggregated on GPU; the public metric keeps the final host ratio. Arrays still
stage through host-owned storage.


NDCG cache initialization uses software-binary64 logarithms for discounts and
parallel ideal-gain sums. Native tests compare 3.55 million discounts and include
zero and nearly cancelling linear gains. Discount relative error is at most
`3.34e-16`; ideal-gain error scaled by the sum of absolute terms is at most
`2.73e-14` in the tested cases. Inverse gains near zero are ill-conditioned: their
relative difference can be much larger, so no uniform relative equivalence is
claimed there. An 80-digit Decimal oracle checks the cancellation example; Apple
arm64 `long double` has only 53 significand bits. Earlier PR integral tests are
independent formulas implemented in `long double`, not extra-precision oracles on
this machine. Linear NDCG labels retain upstream's lack of relevance validation;
exponential NDCG and binary ranking labels are checked on GPU, including negative
subnormals and tolerance boundaries.


## Ranking gradient numerical behavior

Biased top-k/mean pairwise/NDCG/MAP gradients preserve FP32 sigmoid and CPU score
subtraction, binary64 delta/Hessian calculations, per-row FP32 accumulation order,
and separate FP32 normalization/weight multiplications. Software arithmetic
preserves subnormal inputs/results that native Metal arithmetic can flush.
Parallel query normalization and MAP prefixes can change low-order rounding;
CPU equivalence is tested with FP32 tolerances, not a universal bitwise guarantee.
Candidate-pair buffers are batched, and output gradients return to existing
host-owned storage. Fully resident training state remains open work.


Mean sampling matches Apple libc++ engine/distribution sequences and stable label
bucket order. Query generators build ordered incidence lists on GPU; generation
within a query is sequential. The ranking sigmoid uses a two-float exponential
expansion with FP32 intermediate rounding and an emulated binary64 reciprocal.
It matches CPU/GPU binary64-exp oracles on 2,996,040 test inputs; dynamic Apple
`expf` itself differs on 4,962 of these (up to four sigmoid ULPs). Rounding feedback
can amplify small differences in mean-pairwise training. A million-row, 20-round
benchmark fails the strict per-prediction check on 0.1544% of rows (maximum
0.000506, RMS 7.09e-6), while NDCG@10 histories agree within 4e-15. This limitation
is retained in the published failed report; strict checks were not weakened.


Unbiased loss corrections and bias updates now execute on GPU. Per-row cost
accumulation retains CPU order; cross-query sums use ordered 128-query blocks
and a final block reduction, changing addition association; logarithms/powers use software
binary64. Small bias vectors return to host for serialization. Native multi-step
and multi-batch tests validate gradients and state. A large unbiased mean-pairwise
training case still fails strict prediction equivalence (maximum 0.421501, RMS
0.047376), although NDCG@10 histories agree within 8.11e-5. Its failed report is
published; unbiased NDCG/MAP cases pass strict prediction checks. This is an
explicit numerical limitation, not an unported CPU-gradient fallback.

## Column sampling behavior

Feature initialization, uniform and weighted sampling without replacement, and
final feature-ID sorting execute on Metal for `colsample_bytree`,
`colsample_bylevel` and `colsample_bynode`. Existing host caches retain the same
per-tree/per-level reuse and per-node resampling rules. The host consumes one
parent RNG seed per actual sample, exactly as the CPU path does; rates of one
reuse the feature set without consuming a seed or dispatching a sampler.

The GPU implements MT19937 and Apple libc++ uniform-real and rejection-based
uniform-integer draws. Uniform selection uses forward Fisher-Yates and stops
when its selected prefix is fixed; unused local-generator draws cannot affect
later samples. Weighted sampling gathers feature weights, applies the CPU's
`1e-6` floor, computes `log(u)/weight`, stably orders priorities and sorts selected
integer IDs. Unweighted candidate sets of at most 2,048 columns use a fused kernel with
cooperative sorting, subject to available threadgroup memory. Weighted and larger
sets use separate GPU stages, with integer radix sorting for large selections. Both paths preserve IDs
above the exact-integer range of FP32.

Weighted score ties are sensitive to libm rounding. A sampling-specific table
and cubic logarithm matches the tested Apple CPU rounding; generic binary64 log
rounding differed by up to two FP32 ULP in weighted keys. Tests now require exact
CPU keys, including weights deliberately chosen to produce tied scores. A separate
8,086,019-input check covers random draws, all exponent ranges in the draw domain,
range-reduction boundaries, zero and one with zero observed mismatches. This is a
local Apple libc++/libm compatibility result, not a guarantee about other library
versions or every possible weighted priority.

Native checks cover 530 cases, 3,275,463 selected IDs and 4,610,295 exact random
draws, including million-feature samples and the production sampler hierarchy.
Inputs and output guards, weight-index bounds, parent RNG consumption, cached
levels and GPU dispatch counts are checked. Public tests cover scalar/vector trees,
zero/nonuniform feature weights, missing values and model persistence.

RNG generation is sequential on one GPU lane to preserve the CPU stream. Host
feature vectors and cache metadata remain authoritative; sampling uploads inputs,
waits for completion and reads back selected IDs. These are explicit performance
limits. `column_init_dispatches`, `column_sampling_dispatches` and
`column_sampling_gpu_time_ns` expose completed work; the sampling time excludes
the separate initialization command and host staging/wait overhead.

## Numeric sketch finalization

Fresh MPS histogram matrices batch the retained levels of all numeric columns
into one Metal command. GPU merging preserves level order and equal-value rank
combination, repairs cumulative rank bounds, prunes after each merge and prunes
again to the requested final budget. FP32 operation order, subnormal ranks and
comparison behavior match the CPU sketch. Pruning uses a native FP32 quotient
estimate corrected by exact integer residuals and rounded once with
guard/round/sticky bits. A 5,830,328-input check matches CPU division exactly for
finite/infinite/zero results and checks NaN classification. Each column has one GPU thread; columns run
in parallel, while operations within a column remain sequential.

This is partial sketch coverage. CPU ingestion still builds and sorts queues,
flushes them into levels and performs streaming compaction. The GPU stage copies
those levels into Metal buffers and reads back completed summaries. Final cut
materialization remains on the host; the vendored CUDA
`SketchContainer::MakeCuts` also copies summaries back and materializes cuts on
the host, so that step is not itself missing CUDA parity. Category collection
and adapter-built QuantileDMatrix bins remain CPU work.

Native validation covers 942 columns, including 28 complete streamed sketches
through 1,000,003 input rows. Exact comparisons cover CPU merge/prune output,
independently queried rank bounds, source fingerprints, guards and final cuts.
Inputs include empty levels, duplicate values, zero/subnormal/overflowing weights,
inconsistent ranks requiring repair and pruning budgets from zero to 65,537.
Public tests compare exact CPU cut bits and training predictions for fresh dense,
sparse, missing-value, weighted, grouped and external-page matrices. Cached
gradient indices do not redispatch finalization; categorical-only matrices do not
dispatch numeric sketch work. See [validation](../validation/sketch-finalization-m3.txt).

`sketch_finalize_dispatches`, `sketch_finalize_columns` and
`sketch_finalize_gpu_time_ns` expose completed work. The GPU duration excludes
host ingestion, staging, waiting and cut materialization; it cannot be used as
an end-to-end preprocessing time or speedup.
