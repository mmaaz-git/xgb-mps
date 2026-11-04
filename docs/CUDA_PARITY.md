# CUDA → Metal coverage

The project goal is behavioral coverage of **all XGBoost operations accelerated by
CUDA**, with measured performance. Passing training tests alone does not establish
that those operations execute on the GPU. Metal compute is the implementation
technology; `mps` is the public device selector. Apple's Metal Performance Shaders
library is not a replacement for XGBoost's custom CUDA kernels.

Current architecture: CPU owns XGBoost's data structures and orchestrates training;
selected stages dispatch Metal kernels. Remaining CPU execution is explicitly
listed below. This is not yet a complete GPU backend.

Current priority is complete operation coverage and correctness. Performance
experiments are deferred until the missing GPU paths are implemented; existing
measurements, including regressions, remain available.

| CUDA subsystem | Principal upstream source | Metal status |
| --- | --- | --- |
| Gradient histogram accumulation | `tree/gpu_hist/histogram.cu`, `histogram.cuh` | Implemented: dense/sparse, quantized 64-bit sums, shared-memory dense path |
| Approximate tree method | `tree/updater_gpu_hist.cu` (`grow_gpu_approx`) | `tree_method=approx` is rejected for MPS; Hessian-weighted resketching and the corresponding GPU training path remain unimplemented |
| Gradient quantization | `tree/gpu_hist/quantiser.cuh` | Implemented with a different bounded fixed-point scheme; GPU maximum/finite reduction, exact integer scale selection and bit conversion; scales remain in GPU buffers; eligible raw gradients are read through guarded shared host views |
| Histogram subtraction, cache management | `tree/gpu_hist/histogram.cuh` | Metal-owned shared cache, GPU integer-to-binary64 conversion, external-page accumulation, subtraction and GPU feature gathering; host cache policy/metadata |
| Split evaluation, categorical partitions, constraints | `tree/gpu_hist/evaluate_splits.cu`, `evaluate_splits.cuh`, `multi_evaluate_splits.cu` | Metal scalar/vector candidate scoring, category ordering and missing directions; software binary64, exact bounded scans, conservative numerical-candidate screening, monotonic bounds and pooled vector weights; host interaction masks and tree updates |
| Row partition and final leaf assignment | `tree/gpu_hist/row_partitioner.cu` | Metal stable classify/prefix/scatter with cooperative scans for large nodes, resident row IDs, GPU leaf positions with guarded shared gradient inputs/output slices; supports numerical/categorical splits and missing values |
| Sampling and gradient sums | `tree/gpu_hist/sampler.cu`, `tree/updater_gpu_hist.cu`, `common/random.cu` | Metal uniform and gradient-based row sampling/replay, including CPU-compatible draws, norms, sorted thresholds and rescaling; scalar/vector root-gradient sums and dense root histogram sums on Metal; column initialization, uniform/weighted selection and final feature ordering now use Metal |
| Squared/logistic/squared-log/gamma/pseudo-Huber/Poisson/Tweedie/hinge/multiclass/expectile gradients | `objective/regression_obj.cu`, `hinge.cu`, `multiclass_obj.cu` | Metal kernels with guarded synchronous host-memory sharing where eligible; means/weighted means, expectile initial gradients/Hessians and Newton stump fitting use Metal; regression and multiclass objective label predicates now use Metal; generic metadata validation and other initialization work remain CPU |
| Sigmoid/exp/hinge/softmax/argmax/expectile prediction transforms | Same objective files | Metal kernels; identity transforms require no dispatch |
| Base-score validation and margin conversion | `objective/regression_obj.cu` | Metal checked logistic/gamma intercepts, clamped FP32 logit and logarithms for gamma/Poisson/Tweedie/Cox. Identity mappings require no dispatch; expectile's host margin loop matches upstream host execution |
| Absolute-error and quantile objectives, associated statistics | `objective/regression_obj.cu`, `quantile_obj.cu` | Metal per-iteration residual scales, gradients and quantile prediction sorting; absolute-error initial weight sums, means, target-mean broadcast and stump fitting, and initial quantile estimation use Metal; other initialization work remains CPU |
| Ranking losses, pair construction, normalization | `objective/lambdarank_obj.cu` | Prediction order and NDCG ideal-label order use stable grouped Metal sorting; query-weight sums use Metal. NDCG discounts, ideal-gain calculation and ranking label validation also use Metal; biased top-k/mean pair construction, gradients, MAP prefixes and query normalization execute on Metal. Unbiased loss corrections, per-row bias costs, cross-query aggregation and regularized bias updates also execute on GPU; small bias vectors return to host for serialization |
| AFT survival loss | `objective/aft_obj.cu`, `common/survival_util.h` | Metal normal/logistic/extreme gradients and prediction/base-score transforms; GPU identifies small censored probabilities for original CPU gradient recomputation, exposed through correction counters/time. Full GPU coverage remains incomplete |
| Base score estimation, leaf value fitting | `tree/fit_stump.cu`, `common/stats.cu`, `tree/gpu_hist/leaf_sum.cu` | Means/weighted means, quantiles/medians, Newton stump gradient aggregation and unregularized weights on Metal; final vector-leaf refitting with separate value gradients now uses GPU segmented sums over resident row IDs, cross-page accumulation and regularized/clipped weights; final tree updates stay host as in CUDA. Other initialization/state work still requires porting/audit |
| Batch/inplace/leaf tree traversal | `predictor/gpu_predictor.cu` | Metal: scalar/vector leaves, category bitsets, missing values, tree ranges, dropout weights; CPU input views stage dense chunks and recode categories |
| Prediction cache from leaf IDs | `predictor/predictor.cu` | Metal; supports strided target slices and negated sampling positions |
| SHAP contributions and interactions | `predictor/gpu_predictor.cu`, `predictor/interpretability/shap.cu` | Metal fixed eight-point quadrature, repeated-feature paths, scalar/vector leaves, categorical/missing decisions, dropout weights, bias addition, symmetry and diagonal finalization; model compression/root means remain host as in CUDA; dense input staging/category recoding remain host |
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
| Linear coordinate-descent booster | `linear/updater_gpu_coordinate.cu` | GPU bias/feature gradient reductions and residual updates with `updater=coord_descent`; host selectors, scalar weight/model updates, CSC preparation and linear predictions match CUDA's placement; default `shotgun` remains host |
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
4. Keep dispatch instrumentation for coverage checks. When performance work
   resumes, measure cold/warm time and staging costs on identical inputs and
   parameters, including preprocessing.
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

Sensitive AFT gradient corrections, remaining sketch
construction, GPU `approx` training and collectives are separate coverage requirements.
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
finite/infinite/zero results and checks NaN classification. Each column uses one
32-lane SIMD group. Merge and rank repair retain their serial order; pruning
performs independent binary searches and removes repeated selected entries with
SIMD prefix counts. GPU checks require monotonic search keys and a finite,
nonnegative rank range; irregular summaries use the original serial GPU pruning.

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

## Objective-specific initialization

Expectile initialization now computes per-row/per-level gradient and Hessian
pairs on Metal using the previously computed label mean. This is a distinct
formula from the ordinary transformed-margin expectile gradient kernel. FP32
subtraction, multiplication order, subnormal comparisons and underflow behavior
are preserved. Absolute-error initialization now broadcasts target means into
its prediction matrix with a GPU bitwise-copy kernel. Inputs are staged before
output is written, including when broadcast source and destination overlap.

Native checks cover 164 cases, 7,220,670 gradient pairs and 14,441,340 copied
floats. Gradients match CPU bits, with NaNs checked by classification; copied
NaN payloads and signed zeros must match exactly. Empty inputs skip dispatch,
and tests include million-row inputs, multiple levels/targets, tiny/overflowing
weights and output guards. Public checks compare intercepts, predictions,
saved-model continuation and the conditions that skip initialization.

`expectile_init_dispatches` and `prediction_init_dispatches` count completed
commands. `initialization_gpu_time_ns` measures these commands separately from
ordinary objective/transform GPU durations. The small final intercept additions
and expectile ordering remain on the host, matching the CUDA source. Generic
state allocation/filling, metadata validation and other initialization paths still
require audit; this is not a claim that all initialization is GPU-resident.

## Objective labels and base-score conversion

Objective label checks for logistic regression/classification/logitraw, squared
log error, gamma, Poisson/Tweedie and multiclass classification now execute on
Metal. The GPU reduces predicate failures to one status flag; existing host code
retains the same error messages and weight-shape checks. Integer-bit comparisons
preserve subnormal values, signed-zero equality and NaN comparison behavior.
Multiclass integrality uses exponent/mantissa bits, avoiding subnormal flushing
in a native floor/equality comparison. The class bound is converted to FP32 just
as it is in the CPU comparison. Empty inputs and square loss's unconditional
predicate return true without scanning or dispatching.

The production API supports row-major, column-major, padded and broadcast tensor
views. Tests compare 120,093,300 individual predicate results with the upstream
CPU predicates and 576 API cases covering 216,948,024 selected tensor values.
Public tests exercise invalid labels, first-iteration dispatch, weighted/multiple
target training and CPU-equivalent predictions. This does not port the earlier
generic DMatrix metadata validation performed while input is constructed.

Logistic/gamma intercept validation and logarithm/logit margin conversion now
run on Metal. Logistic keeps the CPU's inclusive [0,1] validation, `1e-6` clamp
and `-logf(1/x - 1)` operation order. Gamma requires a positive intercept; the
Poisson/Tweedie/Cox logarithm path retains the original unchecked-log behavior.
Cox gradients and its prediction transform are host-side in the vendored source;
its CUDA-capable base-score conversion is included here.

The shared CPU-compatible FP32 logarithm now handles positive subnormals and
exceptional values as well as the column sampler's original uniform domain.
Native base-score checks cover 4,570,936 comparisons with exact CPU bits except
NaN classification. This is observed agreement with the tested Apple CPU libm,
not a proof for every FP32 input or other libm versions. Source/output guards,
actual objective APIs and rejected vectors remaining wholly unchanged are
checked. Public tests cover raw margins, serialization and validation errors.

`label_validation_*` and `base_score_*` expose independent dispatch counts and
GPU times, including completed error paths. Identity conversions require no
dispatch. Model storage, input staging and error reporting remain host-side.


## SHAP contributions and interactions

The vendored CPU and CUDA implementations use a fixed eight-point
QuadratureTreeSHAP recurrence. Metal now performs that traversal, edge-return
calculation, repeated-feature bookkeeping, contribution accumulation and
interaction finalization on the GPU. It preserves CPU tree/edge accumulation
order, including dropout weights. Missing and categorical decisions, vector-leaf
targets, base margins, bias addition, interaction symmetry and diagonal subtraction
are included. This ports the upstream quadrature semantics; the API's “exact”
setting does not imply mathematically exact integration for every deep tree.

Traversal uses a runtime-sized stack, tested through depth 130, rather than
CUDA's current depth-127 limit. Rows are batched against a 64 MiB workspace target
and the device's per-buffer limit; at least one row is processed, so an unusually
wide interaction row may exceed the target. The output still occupies its full
host array. Uploaded model buffers are reused across these batches. Commands are
synchronous, and no performance claim is made for this implementation.

Model compression, branch-cover ratios, root-mean calculation and quadrature-rule
construction stay on the host, matching CUDA's model-preparation approach.
Sparse/quantized page decoding and categorical recoding also remain host-side;
those are separate unfinished input/metadata operations. Approximate-contribution
requests retain the existing CPU implementation; CUDA rejects those requests, so
they are not missing CUDA-accelerated SHAP coverage. Conditional exact SHAP is
rejected as it is by the upstream CPU and CUDA implementations.

Native validation covers 576 cases and 424,350,144 output entries, including
unused zero entries, against independent one-feature Shapley values. Maximum
absolute error is `2.38418579e-7`; guards, input preservation, overlapping storage,
empty inputs, shape rejection and batch sizing are checked. Public validation
covers 87 cases: identical-model CPU comparisons at `rtol=atol=2e-5`, independent
feature-subset enumeration on small trees at `2e-6`, scalar/vector/multiclass/forest/
dropout models, missing and categorical data, recoding, external pages, quantized
input, base margins, tree limits, zero/negative/tiny covers, depth 130 and wide
interaction batches. Interaction matrices must be symmetric and sum by row to
CPU additive contributions within the stated tolerance.

`shap_dispatches` and `shap_interaction_dispatches` expose completed GPU batches;
`shap_gpu_time_ns` reports their combined GPU duration separately from ordinary
prediction timing. These measurements exclude host model/input preparation,
staging and result copying.


## Reduced-gradient final leaf fitting

Custom tree objectives can supply separate gradients for split selection and
final leaf values. Metal now computes the final leaf-gradient sums and weights
in `MultiTargetHistBuilder::ExpandTreeLeaf`. It uses the row indices already
resident in each Metal partitioner; it no longer copies those indices back for
CPU summation. Host code supplies segment metadata and allocates buffers. The GPU
clears the sums, reduces indexed gradients in blocks, accumulates completed leaf
sums across external-memory pages, and computes the final weights. Gradients use
guarded shared host views when eligible and staged input otherwise.

Sums use software binary64, following the CPU precision contract. CUDA uses its
quantized-gradient representation for this operation; this implementation does
not claim identical intermediate CUDA sums. The weight calculation preserves
CPU L1/L2 regularization, maximum-step clipping, FP32 conversion, bound application
and subsequent learning-rate multiplication. Nonpositive Hessians produce zero
before bounds are applied. `min_child_weight` controls split validity and is not
an additional cutoff in the upstream final-weight formula. The internal weight
API handles per-leaf/per-target bounds, although public reduced-gradient training
rejects monotonic constraints in both upstream CPU and CUDA implementations.

The final leaf values return to the host tree, matching CUDA's model update.
Generic gradient allocation/layout copies earlier in the training path remain
host work. This closes final leaf refitting, not every training-state operation.
The API currently checks a maximum of 65,535 targets, 32-bit leaf/task indices and
the Metal device's buffer-size limits.

Native validation covers 304 cases and 24,327,036 source gradient pairs across
row-major, column-major, padded and broadcast views. Independently accumulated
sums match 46,852 checked fields exactly; 156,240 weight results match the CPU
formula by bits, with NaNs compared by classification. Tests cover million-row
inputs, page offsets, permuted output slots, empty leaves, tiny/large/nonfinite
inputs, negative Hessians, regularization, bounds, guards and malformed metadata.
Public validation adds 66 cases across four input layouts, both sampling methods,
zero/full sampling, regularization/clipping, root-only trees, saved-model
continuation and the upstream monotonic-constraint rejection. Training outputs
use the existing `rtol=atol=2e-5` CPU comparison.

`leaf_sum_init_dispatches`, `leaf_sum_dispatches` and `leaf_weight_dispatches`
count completed clearing, per-page reduction and weight commands.
`leaf_fitting_gpu_time_ns` measures their combined GPU duration separately from
root-gradient sums and row partitioning. Empty reductions do not dispatch. No
performance gain is claimed in this coverage-focused update.


## Linear coordinate descent

`booster="gblinear", updater="coord_descent", device="mps"` now dispatches Metal
gradient sums and residual updates for the bias and each selected feature. These
are the four GPU operation families in `linear/updater_gpu_coordinate.cu`.
Host code prepares CSC columns, selects features, calculates scalar coordinate
steps and updates model weights, matching CUDA's placement. Linear prediction,
contributions/interactions, convergence checks and training-weight totals are
also host operations upstream. The default `shotgun` updater remains host; it
does not silently change algorithms when selecting MPS. Upstream's deprecated
`gpu_coord_descent` user parameter still raises its original error, and the
vendored version's `gblinear` deprecation warning remains visible.

The numerical reference is the CPU coordinate updater. Metal reduces FP32
feature products in software binary64, preserves the CPU residual multiplication
order, excludes negative Hessians as the CPU does, and includes the Hessian's
addition of positive zero when updating a pair. This differs from CUDA's FP32
reductions and some of its arithmetic/filtering choices; identical CUDA
intermediate values are not claimed. Sorted duplicate CSC row entries are
processed sequentially by one thread per run, avoiding residual write races.

All five selectors are supported. Greedy/thrifty selectors see completed residual
updates, following the CPU contract; the CUDA updater instead supplies its
unchanged host gradient snapshot to selectors. CSC entries and mutable gradients
are copied into Metal-owned storage for each boosting update, and residuals are
read back before host selection needs them and at completion. Keeping data
resident between boosting rounds and reducing synchronization are deferred
performance work. As in CUDA, training requires a single CSC block; external
multi-page input is rejected explicitly. Device buffer limits, signed 32-bit
feature identifiers and 32-bit residual dispatch sizes are checked.

Native validation covers 49 cases and 4,118,809 source gradient pairs, with
3,159 exact sum-field comparisons and 148,277,124 residual-field comparisons.
Finite residuals match CPU bits, including signed zeros; NaNs use classification.
The dyadic finite sum inputs allow exact independent CPU accumulation. Cases
include million-row input, multiple groups, empty columns/rows, duplicate rows,
subnormals, large/nonfinite values, negative Hessians, input preservation,
guards and invalid metadata. The 136 public tests exercise dense/missing/CSR data,
regression/binary/multiclass/multitarget objectives, all selectors, L1/L2 penalties,
`top_k`, zero weights/steps, convergence and saved-model continuation. Predictions,
contributions and serialized weights use `rtol=atol=2e-5` against CPU.

`linear_gradient_dispatches` and `linear_residual_dispatches` expose completed
commands; `linear_gpu_time_ns` covers both, excluding host selection, scalar
updates and staging. Empty operations and zero residual steps require no
dispatch. No performance measurement accompanies this coverage update.
