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

## Implementation authority and validation limits

The vendored CUDA implementation defines the intended algorithms, precision,
random streams and update order. CPU XGBoost is a useful comparison where it
implements the same behavior; it is not the specification when the two differ.
Independent scalar implementations of CUDA formulas/control flow can run on this
Mac and provide test oracles. They cannot establish measured CUDA equivalence.
**No tests in this repository have yet compared this port with execution on an
NVIDIA GPU.** Such a run remains a separate validation requirement.

Several earlier kernels deliberately followed CPU behavior. Their tests remain
useful regression evidence, but the following differences are unfinished porting
work, not accepted substitutes for CUDA behavior:

- Quantized training state and tree reductions: original per-target quantization,
  both integer sampling methods, histogram caches and all scalar/vector/hist/approx
  root reductions now share GPU-owned integer state. Gradient-based sampling follows
  the CUDA pipeline described below. Leaf sums and split scans/child statistics
  still need integer-state alignment.
- Row/column/ranking sampling: uniform row sampling targets CUDA's modern
  `<cuda/std/random>` Philox4x64 branch, including seed consumption and replay.
  The older Thrust branch is not an equivalent stream. Gradient-based sampling now
  uses that RNG and CUDA-derived norms/thresholds/integer rescaling. Compiler
  contraction and hardware-dependent reduction details remain unverified on NVIDIA;
  column/ranking random streams still need alignment.
- Quantile sketches: CPU row queues, `SetPruneSorted`, ordered summary merging
  and pruning differ from CUDA's sorted weighted scan and GPU sketch pipeline.
  Existing exact CPU cut tests do not validate that CUDA pipeline.
- Linear coordinate descent: binary64 reductions, filtering, residual arithmetic
  and fresh host gradients for greedy/thrifty selectors differ from CUDA's FP32
  reductions and unchanged host selector snapshot.
- Ranking metric edge cases: some degenerate-query fallbacks currently preserve
  CPU semantics where CUDA differs.
- Split tie-breaking, reduction order, quantile statistics and objective edge
  cases still need a complete source audit; GPU dispatch coverage alone is not
  an algorithm-parity claim.

Architecture-specific mechanics may differ where Metal requires them. Algorithm
changes must be explicit and justified, and any observed numerical differences
must be reported rather than hidden by widening test tolerances.

| CUDA subsystem | Principal upstream source | Metal status |
| --- | --- | --- |
| Gradient histogram accumulation | `tree/gpu_hist/histogram.cu`, `histogram.cuh` | Implemented: dense/sparse, quantized 64-bit sums, shared-memory dense path |
| Approximate tree method | `tree/updater_gpu_hist.cu` (`grow_gpu_approx`) | `tree_method=approx` now runs on MPS; original Hessians are extracted and cuts built before sampling, once per updater call; row/column sampling reset per tree. GPU copy/root/partition operations are integrated. Both sampling methods and all roots share quantized state; weighted sketches still differ from CUDA; preprocessing remains hybrid |
| Gradient quantization | `tree/gpu_hist/quantiser.cuh` | GPU positive/negative binary64 bounds, CUDA CreateRoundingFactor scaling, truncation toward zero and minimum-one positive Hessians; scales remain in GPU buffers and guarded shared host views avoid eligible copies. Original per-tree scales feed both integer sampling methods, all roots and histograms; leaf/split statistics need integer alignment |
| Histogram subtraction, cache management | `tree/gpu_hist/histogram.cuh` | Metal-owned integer cache; external-page accumulation and subtraction retain exact quantized sums, with a binary64 view materialized for existing split evaluators. GPU growth copies both buffers, empty first pages overwrite reused ranges, and feature gathering remains GPU. Host cache policy/metadata and binary64 split scans remain |
| Split evaluation, categorical partitions, constraints | `tree/gpu_hist/evaluate_splits.cu`, `evaluate_splits.cuh`, `multi_evaluate_splits.cu` | Metal scalar/vector candidate scoring, category ordering and missing directions; software binary64, exact bounded scans, conservative numerical-candidate screening, monotonic bounds and pooled vector weights; host interaction masks and tree updates |
| Row partition and final leaf assignment | `tree/gpu_hist/row_partitioner.cu` | Metal stable classify/prefix/scatter with cooperative scans for large nodes, resident row IDs, GPU leaf positions with guarded shared gradient inputs/output slices; supports numerical/categorical splits and missing values |
| Sampling and gradient sums | `tree/gpu_hist/sampler.cu`, `tree/updater_gpu_hist.cu`, `common/random.cu` | CUDA Philox uniform/gradient-based integer row sampling and shared integer root sums for every layout; gradient sampling uses original quantizers, binary64 norms, quantized prefixes, integer rescaling and saved-state replay; column initialization/selection/order run on Metal but still need CUDA RNG alignment |
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
| Quantile sketching, cuts, weighted/category handling | `common/quantile.cu`, `data/gradient_index.cu` | Group/Hessian weighting, sorted-column weighted summaries, streaming level compaction, final numeric merging, rank repair and pruning use Metal. Row ingestion/queue folding, value sorting and category collection remain CPU. Final cut materialization is host-side in both CUDA and Metal. SparsePage bin assignment and hit counts use Metal; adapter-built QuantileDMatrix bins remain CPU |
| Compressed GPU matrix, external memory, iterators | `data/ellpack_page.cu`, `iterative_dmatrix.cu`, `quantile_dmatrix.cu` | Host GHistIndexMatrix remains authoritative; page-owned Metal copies and dense GPU transpose cache persist for each source page lifetime |
| Categorical recoding, metadata manipulation, validation | `data/cat_container.cu`, `metainfo.cu`, `array_interface.cu` | CPU |
| Linear coordinate-descent booster | `linear/updater_gpu_coordinate.cu` | GPU bias/feature gradient reductions and residual updates with `updater=coord_descent`; host selectors, scalar weight/model updates, CSC preparation and linear predictions match CUDA's placement; default `shotgun` remains host |
| Gradient layout copies and Hessian extraction | `gbm/gbtree.cu`, `tree/updater_gpu_hist.cu`, `tree/updater_quantile_hist.cc` | Metal bit-preserving target-column copies, sampling-buffer layout conversion, reduced-value-gradient copies and unsampled Hessian extraction; guarded host sharing or staged snapshots for aliases. Allocation and other state manipulation remain host |
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

1. Identify the corresponding CUDA algorithm and any CPU/CUDA differences. Test
   actual Metal output against independent scalar references derived from CUDA,
   including edge cases and documented tolerances. Compare CPU XGBoost where its
   algorithm agrees; do not force CPU-specific behavior to pass that comparison.
2. Check a completed-operation counter so a CPU fallback cannot pass the test.
3. Validate the integrated API, model persistence, and CPU-only build.
4. Keep dispatch instrumentation for coverage checks. When performance work
   resumes, measure cold/warm time and staging costs on identical inputs and
   parameters, including preprocessing.
5. Keep limitations visible. A feature's training output matching CPU does not mean
   its gradients, prediction, metrics, or preprocessing have been ported. Track
   CUDA-source alignment separately from dispatch coverage and CPU agreement.
6. Run a shared fixture suite on NVIDIA hardware before claiming measured CUDA
   equivalence. Local scalar-oracle tests do not replace that step.

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
construction, CUDA algorithm alignment and collectives are separate coverage requirements.
The observed prediction speedups do not establish any of those as complete.

## Legacy CPU-formula gradient sampler

The earlier low-level `metal::GradientSample` API follows the CPU sampler, including
sequential FP32 target/prefix accumulation, Apple libc++ draws and CPU threshold
fallbacks. Its exact CPU-reference tests are retained as regression coverage.
**Production MPS training no longer calls this API.** It uses the
[CUDA-derived shared-state sampler](#cuda-derived-gradient-based-sampling).

Historical performance measurements of the ordered FP32 prefix and its checkpoint
optimization describe that earlier implementation. They have not been rerun for
the integer sampler and are not performance claims for the current path.

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
from upstream CUDA degeneracy handling. The current tests use upstream CPU;
those differences remain to be corrected and validated against CUDA formulas.

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

This is partial sketch coverage. CPU row ingestion still builds and sorts queues
and forms their weighted summaries. Streaming level compaction now runs on Metal,
as described below. Finalization copies retained levels into Metal buffers and
reads back completed summaries. Final cut
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

## Sketch weights, sorted summaries and streaming compaction

MPS sketch construction now expands group weights and multiplies row/group weights
by Hessians on Metal. It preserves the CPU's FP32 product, subnormal results and
signed zeros. Repeated group boundaries are supported through GPU upper-bound
lookup. Plain row weights need no transformation; implicit unit weights need no
array in the row-queue path. The sorted-column path creates its full weight array
on the GPU because the CPU reference requires explicit per-row weights.

Sorted-column ingestion computes nonzero source counts, unique-value counts,
weighted ranks and retained entries on Metal. Ordered software binary64 sums
and the CPU's FP32 rounding of interior selection goals preserve the original
`SetPruneSorted` behavior. Empty columns, sufficient budgets, zero weights and
the final endpoint have distinct reference behavior that is tested explicitly.
The kernel receives already sorted columns; sorting the source values still
requires porting. This path is integrated into `HostSketchContainer::PushColPage`
and public `tree_method=approx` training. It still follows the CPU sketch
algorithm; CUDA sketch alignment remains required.

Every MPS numeric sketch queue flush now sends its incoming summary and retained
levels to Metal. The GPU prunes to the level limit, merges/repairs ranks, clears
consumed levels and carries oversized results upward until they fit. Untouched
higher levels retain their contents. The completed levels return to existing
host storage, and finalization continues through the existing GPU finalizer.
The implementation handles streaming compaction for both row-queue summaries and
sorted-column summaries. Row collection, consecutive-value folding, queue sorting
and queue-to-summary formation remain CPU work, as does category collection.
Host metadata, allocation and staging remain; this is not a claim of complete
device-resident sketch state or complete preprocessing coverage.

Native checks compare actual GPU weights, sorted summaries and every retained
streaming level with upstream CPU arithmetic, including million-row inputs.
Finite fields compare by bits and NaNs by classification. Tests also cover
independent budgets and carry propagation in a batched command, source
preservation, invalid metadata, and production cut creation across three pages
with grouped weights, Hessians, empty columns and categories. The public sketch
tests retain exact CPU cut-bit comparisons and require streaming GPU dispatches;
grouped training also requires a weight-expansion dispatch. No tolerance was
loosened and no performance experiment accompanies this update.

`sketch_weight_dispatches`, `sketch_sorted_dispatches` and
`sketch_update_dispatches` count completed commands. `sketch_ingest_gpu_time_ns`
measures their GPU time, separately from finalization and excluding host queue
work/staging. Empty command batches do not dispatch; an empty incoming summary
can still dispatch a level update against retained state. The runtime checks
32-bit row/task/count limits, level capacities and Metal buffer-size limits.

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

Sums currently use software binary64, following CPU precision. CUDA uses its
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

The current numerical reference is the CPU coordinate updater; the differences
below remain CUDA-alignment work. Metal reduces FP32
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


## Approximate training and gradient state integration

`tree_method="approx", device="mps"` now uses the shared builder with Metal
Hessian extraction, buffer copies, root statistics, histogram/split evaluation,
row partitioning and final leaf assignments. The MPS updater follows the ordering
in `tree/updater_gpu_hist.cu::GPUGlobalApproxMaker::Update` and
`GPUHistMakerDevice::Reset`: extract **original, unsampled** Hessians, build cuts
once per updater call, then sample rows and reset column sampling separately for
each tree. Constant-Hessian tasks cache cuts; other tasks regenerate at the next
updater invocation. CPU approx intentionally retains its different shared-forest
sampling and sampled-Hessian sketching order.

This is partial CUDA algorithm alignment. The underlying sketch, row/column RNG and tree-statistic storage still require
CUDA alignment. Histogram quantization now follows the CUDA formulas, but its
placement in training does not yet match CUDA. In particular, Metal still samples float gradients before histogram
quantization; CUDA samples already quantized gradients. Source value sorting,
row queue construction and category collection remain host work. The updater is
usable but not a complete CUDA-equivalent approximate backend. Vector-leaf approx
remains unsupported, as upstream does not implement it.

The new gradient-state kernel copies both raw words of each gradient pair or
extracts just the Hessian word. It handles C/F layouts, slices, padded strides and
broadcast sources; overlapping destinations are rejected. Aliased input/output
uses a snapshot, and guarded shared views fall back to copied staging when needed.
Copies are also used by target selection, histogram sampling buffers and
reduced-gradient leaf refitting. Final leaf positions use the resident Metal row
IDs instead of the CPU-only Hessian overload.

Fresh external SparsePage-derived gradient-index pages now preserve their MPS
context, so bin assignment executes on Metal. Previously those pages created a
CPU context; the earlier blanket GPU-binning claim was too broad. Adapter-built
QuantileDMatrix bins remain host work.

`tests/test_metal_approx.py` contains 141 integration cases. Thirty fixed-gradient
cases independently check CUDA updater ordering: cuts match the unsampled sketch
reference at sampling rates 0, .3 and .8; two rounds of three trees make two cut
builds and six row/column resets. They cover dense, missing, CSR, categorical and
external-page inputs and both row samplers. The cut reference uses CPU approx
without sampling to isolate ordering; it does not validate CUDA's GPU sketch.
Two scalar tests independently check the no-sampling leaf formula. Other tests
cover regression, classification, multi-output scalar trees, both growth policies,
constraints, forests, empty inputs and saved-model continuation. CPU model
comparisons are limited to common update flow, while sampled models check source
ordering, reproducibility, finite results and actual GPU dispatch.

The native gradient-state test covers 276 cases, with 25,026,384 copied pairs and
25,026,384 extracted Hessians in its layout loops, plus alias/read-only checks.
It compares raw bits, including signed zeros, subnormals and NaN payloads, and
checks guards, output padding, staging counters and rejected invalid views.
`gradient_copy_dispatches`, `hessian_extract_dispatches` and
`gradient_state_gpu_time_ns` expose completed work; the corresponding
`gradient_state_shared_input_bytes`, `gradient_state_shared_output_bytes` and
`gradient_state_staging_bytes` report view extents, including padding. GPU time
excludes staging and host allocation. See [validation](../validation/approx-gradient-state-m3.txt).


## CUDA gradient quantizer formulas

Histogram quantization now follows `tree/gpu_hist/quantiser.cu`,
`tree/gpu_hist/quantiser.cuh::GradientQuantiser::ToFixedPointImpl` and
`common/deterministic.cuh::CreateRoundingFactor<double>`:

1. Accumulate positive and negative gradient/Hessian magnitudes separately in
   binary64, and take the larger signed magnitude sum for each channel.
2. Compute `delta = bound / (1 - 2 * n * epsilon<double>)`. Use the exponent
   returned by `frexp(delta)` to form the power-of-two rounding factor, divide by
   `2^62` for the dequantization factor, and invert for quantization. This follows
   the actual CUDA code even at zero and exact powers of two; a mathematical
   ceiling of a logarithm is not equivalent at those boundaries.
3. Truncate scaled gradients/Hessians toward zero. A strictly positive Hessian
   that truncates to zero becomes integer one, preserving positive curvature.

The previous Metal maximum-times-row-count bound and ties-away rounding are
removed. Metal reduces bounds using software binary64, evaluates the scale on GPU,
and performs exact power-of-two truncation through integer decoding of FP32 bits.
The resulting quantized gradients and scales remain in GPU buffers. Eligible raw
host gradients use guarded shared views. The existing finite-input rejection and
32-bit row limit remain explicit; CUDA float-to-integer conversion on NaNs or
infinities does not supply a meaningful finite reference contract.

Native validation compares **99 bound reductions and 19,305,306 quantized values**
against independent scalar translations of CUDA's formulas, including **1,507,902
minimum-one Hessians**, and **1,210 scale boundary cases**. Exact representable
bound sums match bit for bit; wide-exponent sums are checked against an FP64
reduction error bound, and the chosen scales and integer outputs compare exactly.
Tests cover empty/zero inputs, sign-separated cancellation, subnormals, outliers,
nonfinite rejection, maximum row counts, output guards and million-row arrays.
The separate histogram suite checks **713,814 conversions across six exponents**
with the new truncation/curvature rule and all 576 integer histogram cases.
Four additional end-to-end tests in `tests/test_metal_cuda_quantizer.py` check
both `hist` and `approx`, verifying child Hessian statistics and predictions
against the CUDA minimum-one formula with tiny/subnormal inputs. They explicitly
verify that CPU predictions differ; forcing CPU equality would reject the
intended CUDA behavior in these cases.

This validates the quantizer formulas on Metal, not actual NVIDIA execution or
the whole CUDA training pipeline. Backend reduction topology can change FP64
bound sums near rounding boundaries. Shared integer state now feeds both sampling
methods and every root layout. Remaining work includes integer leaf sums and
integer split scans/child statistics. Histogram cache page
addition/subtraction now retain integer statistics, as described below. Full Python regression validation passes without tolerance
changes. See [validation](../validation/cuda-quantizer-m3.txt).


## Integer histogram cache and cancellation

Metal histogram caches now preserve quantized 64-bit gradient/Hessian pairs
through page accumulation, growth, slot reuse and subtraction, matching the
integer arithmetic in `tree/gpu_hist/histogram.cuh::SubtractionTrick`. A separate
binary64 view is materialized for the existing split evaluator and host tree
orchestration. Rounded values are never fed back into integer cache arithmetic.
The buffers share a retained quantizer scale. This doubles persistent histogram
storage from 16 to 32 bytes per bin while the binary64 view remains necessary.

Page zero replaces both views, even when a node has no rows on that page. Later
pages add integer pairs. This removes the earlier host clearing loop and prevents
stale cache contents from surviving an empty first page. Growth copies both
buffers on GPU; cache eviction/tree reset still use the existing host node maps,
and the next page-zero build overwrites reused ranges. Subtraction preserves its
source ranges and converts only the resulting integer difference. Dense scalar
hist root sums also reduce one feature's integer bins before conversion.

This distinction matters with cancellation: `(2^61+3) - (2^61+1)` must give 2;
subtracting the binary64 views gives 0. Similarly, accumulating page sums before
conversion retains small terms that would disappear beside a large page sum.
The current split scans and child statistics still consume binary64 views, and
leaf fitting still needs quantized state. All roots now share integer state as
described below; both sampling methods now follow original-gradient quantization.
The remaining leaf/split gaps are not resolved by the cache arithmetic.

Native shader tests check **75 page/subtraction cases, 5,013,975 stored/subtracted
pairs and 225 integer root reductions**, including cancellation, signed carries,
noncontiguous offsets, five scales, empty pages and guards. **48 production API
cases** additionally exercise all index widths, dense/sparse layouts, buffer
growth, tree/scale reuse, empty first pages, dispatch counters and exact root
sums. Existing nonquantized binary64 storage operations retain their separate
native regression tests.

Eight new public hist-training cases check exact dyadic cancellation in memory
and across two external pages, including subnormal leaf values. They verify
cached, in-place and reloaded-model predictions. This exposed Metal FP32 additions
flushing subnormal leaves to zero after the model had stored the correct value;
ordinary traversal and training-cache updates now use the existing subnormal-
preserving FP32 helpers. No prediction tolerance was widened.

`histogram_integer_store_dispatches` counts completed integer cache writes and is
a subset of histogram construction/conversion counts.
`histogram_integer_accumulation_dispatches` counts its later-page additions;
`histogram_integer_subtraction_dispatches` is a subset of histogram subtraction,
and `histogram_integer_sum_dispatches` a subset of gradient sum operations. These
are coverage counters, not independent timing categories. Generic nonquantized
storage APIs continue using binary64. See [validation](../validation/integer-cache-m3.txt).


## Shared integer training state and CUDA uniform sampling

`metal::QuantizedGradients` owns each tree's per-target integer buffers and scales.
The hist/approx sampler passes that state explicitly to histogram builders, which
retain ownership through tree growth and external pages. Root reductions sum the
same sampled integers before conversion to binary64, including sparse, missing,
vector, external-page and zero-bin roots. There is no integer-to-FP32-to-integer
bridge or host-pointer registry. C-order/strided inputs use a GPU layout copy;
temporary raw/quantizer workspace is released after quantization.

Uniform sampling now follows `GPUHistMakerDevice::Reset`: quantize the original
gradients, choose the per-tree seed, then mask the integer pairs. Rejected outliers
do not change the original quantizer scales. Float replay uses the identical row
mask for existing position bookkeeping and reduced value gradients. It does not
feed histogram/root arithmetic. Rates below one consume a host seed even for an
empty input; rate one consumes none. Forest trees receive separate states/seeds.

The targeted CUDA configuration is the `<cuda/std/random>` branch of
`common/random.cuh`: Philox4x64-10, `uint32(seed)`, `discard(row)`, and
`uniform_real_distribution<float>`. Reference headers are pinned to
[NVIDIA CCCL commit 3d0d040ac0dc5907daf2a0731edb378e5562099d](https://github.com/NVIDIA/cccl/blob/3d0d040ac0dc5907daf2a0731edb378e5562099d/libcudacxx/include/cuda/std/__random/philox_engine.h)
and its
[`generate_canonical` conversion](https://github.com/NVIDIA/cccl/blob/3d0d040ac0dc5907daf2a0731edb378e5562099d/libcudacxx/include/cuda/std/__random/generate_canonical.h).
The conversion rounds a 64-bit word to FP32 before division by 2^64; a result of
exactly one is retained, as is CUDA's `draw <= probability` predicate. The legacy
Thrust fallback is a different stream and is not claimed equivalent. Thus uniform
sampling intentionally changes same-seed models from the previous CPU-oriented
Metal implementation and from CPU XGBoost.

Native checks use independent host 128-bit products plus CCCL's published 10000th
output, and validate **1,060,276 exact words/draws**, including large skip positions.
The production-state suite checks **228 cases / 4,319,326 pairs**, per-target scales,
integer root sums, original-data lifetime, histogram ownership, padding/broadcast
inputs, replay with a different target count, boundary probabilities, and RNG
consumption across empty/no-op/forest cases. This exposed a real column-extent bug:
`TensorView::Values()` includes backing storage beyond the logical slice. Quantizer
and histogram inputs now use each target's logical row count.

There are **60 new public cases** for root cancellation (including FP32 subnormals,
zero-bin and external-page roots) and retention of the original scale when large
Hessians are sampled out, plus exact per-tree/round/target forest references.
Uniform integration tests supply independent CUDA masks
to CPU custom objectives and then compare predictions at the unchanged tolerances.
This validates sampled tree construction, prediction-cache updates, and reduced
leaf replay, including resumption. Split/column tests with CPU's own sampling are
isolated from the now intentionally different row RNG. Legacy low-level LCG tests
remain regression checks; production uniform training no longer uses that engine.

`quantized_gradient_sum_dispatches` is a subset of gradient sum calls;
`cuda_uniform_sampling_dispatches` counts completed integer sampling calls and is
a subset of sampling calls. Float replay does not increment that integer-sampling
counter. CPU-only builds expose both as zero. See the
[validation record](../validation/shared-quantized-state-m3.txt).

**Still unfinished:** leaf sums, split scans/child statistics, column/ranking RNG,
sketches and other ledger gaps remain. The subsequent gradient sampler port is
described below. These tests did not execute on NVIDIA hardware. No new performance
benchmark accompanies this change.

## CUDA-derived gradient-based sampling

Production gradient-based sampling now operates on the same original quantized
state as uniform sampling. `tree/gpu_hist/sampler.cu` and `quantiser.cuh` define the
pipeline: convert original integer pairs to binary64, form FP32 regularized norm
terms, reduce targets in FP32, take the FP32 square root, sort norms, quantize the
norms with their own CUDA rounding factor, and scan their integer values. Threshold
candidates use the dequantized FP32 prefixes, with the first valid candidate chosen.
The no-candidate case uses CUDA's index-one fallback, not CPU's total/budget fallback.
A zero sample budget yields an infinite threshold and rejects every row.

Selection uses the modern CUDA Philox stream already described above. Accepted
pairs are dequantized to binary64, multiplied by a reciprocal computed in FP32,
and truncated back into the original integer scales, preserving minimum-one
positive Hessians. Even probability-one pairs retain CUDA's integer→binary64→integer
round trip. No FP32 gradient array intervenes in histogram/root state. Raw value
gradients instead replay the same saved norms, threshold and seed using FP32
rescaling, including when reduced split gradients have a different target count.
Only the scalar seed and command orchestration remain on the host; norms, sorting,
prefixes, threshold selection, decisions and rescaling execute on Metal.

The norm reduction models the fixed-segment policy in
[CCCL's segmented reduction selector](https://github.com/NVIDIA/cccl/blob/3d0d040ac0dc5907daf2a0731edb378e5562099d/cub/cub/device/dispatch/tuning/tuning_segmented_reduce.cuh)
and [reduction agent](https://github.com/NVIDIA/cccl/blob/3d0d040ac0dc5907daf2a0731edb378e5562099d/cub/cub/agent/agent_reduce.cuh),
pinned to commit `3d0d040ac0dc5907daf2a0731edb378e5562099d`: SM60–SM90 uses one
thread through 16 targets, a warp through 512, and a 256-thread block above that.
Striped per-thread additions, ascending warp shuffle offsets and final warp-sum
addition are modeled explicitly. Older Thrust reductions and newer architecture
policies are not claimed bit-equivalent. The norm-bound total uses a Metal binary64
reduction; CUDA's total reduction is itself order-dependent near rounding boundaries.

The binary64 norm implementation rounds the squares separately and fuses
`double(0.1f) * h² + g²` before FP32 conversion. This is an explicitly tested
contraction model, **not a measured claim about NVCC's emitted instructions**.
NVIDIA allows contractions and reassociation; the actual compiler configuration
and GPU execution still need comparison. Nonfinite FP32 norms (when the budget is
nonzero) and rescaled coordinates outside signed int64 range raise explicit errors:
the corresponding CUDA C++ floating-to-integer conversion has undefined behavior,
so no portable output equivalence is claimed for those inputs. See NVIDIA's
[mathematical function/compiler appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).

Independent native host references check exact norms, thresholds, roots, replay,
per-coordinate integer rescaling, seed lifetime and recursive scan boundaries.
They include subnormals, minimum-one Hessians, wide exponent ranges, arbitrary
strides/broadcasts, zero budgets and empty inputs. The public CPU comparisons
supply CUDA-derived sampling through custom objectives and retain their original
prediction tolerances. Because that objective interface is FP32, the native
integer checks and exact forest-root tests supply the stronger state-level evidence.
Legacy CPU-formula low-level sampler tests remain regression coverage, but production
training no longer calls that sampler.

`cuda_gradient_sampling_dispatches` counts completed nonempty integer sampling
calls, once per tree, within `gradient_sampling_dispatches`/`sampling_dispatches`.
Float replay does not count again. CPU-only builds expose the counter as zero.
The [validation record](../validation/cuda-gradient-sampling-m3.txt) records results.
Full CUDA coverage and NVIDIA execution validation remain unfinished; performance
benchmarking remains deferred.
