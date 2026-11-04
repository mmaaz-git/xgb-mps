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
  the CUDA pipeline described below. Reduced value-leaf sums now quantize sampled
  floats with original value scales and accumulate integers across pages. Scalar
  and vector split scans and persistent child statistics now retain integers as well.
- Row/column/ranking sampling: uniform row sampling targets CUDA's modern
  `<cuda/std/random>` Philox4x64 branch, including seed consumption and replay.
  The older Thrust branch is not an equivalent stream. Gradient-based sampling now
  uses that RNG and CUDA-derived norms/thresholds/integer rescaling. Compiler
  contraction and hardware-dependent reduction details remain unverified on NVIDIA;
  weighted columns now use the same modern Philox stream, and unweighted columns
  use Thrust's distinct shuffle stream. Ranking random streams still need alignment.
- Quantile sketches: ordinary MPS matrices now use CUDA-derived raw sorting,
  FP32 scans and resident summaries, as detailed below. Adapter/device-array
  ingestion and distributed sketch collectives remain open. Scan association,
  sorting ties and rank-repair scheduling are explicitly bounded validation limits.
- Interaction constraints now follow CUDA's cumulative group unions and ascending
  nonroot queries, with explicit overlapping-group tests; the former CPU history
  filter has been removed from production MPS training and the source oracles.
- Linear reductions now use FP32 pairs, CUDA Hessian/residual rules and unchanged
  selector snapshots. CUB tile ordering, NVCC contraction and actual NVIDIA
  execution remain unverified; they are not cross-device bitwise guarantees.
- Precision/NDCG/MAP now follow CUDA-source arithmetic and degenerate-query
  weighting. Parallel scan/reduction association remains architecture-dependent;
  classification AUC follows CUDA term/output rules and GPU normalization.
  Grouped ROC's source pair terms and weight edge behavior are checked below;
  its count-based accumulation has a different association from CUDA's pair scan.
- Scalar split enumeration now follows CUDA's exclusive numeric scan and strict
  missing-direction comparison; vector scans now preserve CUDA inclusive scans,
  double chunk argmax and FP32 updates. Other reduction order, quantile statistics
  and objective edge cases still need a complete source audit.
  GPU dispatch coverage alone is not an algorithm-parity claim.

Architecture-specific mechanics may differ where Metal requires them. Algorithm
changes must be explicit and justified, and any observed numerical differences
must be reported rather than hidden by widening test tolerances.

| CUDA subsystem | Principal upstream source | Metal status |
| --- | --- | --- |
| Gradient histogram accumulation | `tree/gpu_hist/histogram.cu`, `histogram.cuh` | Implemented: dense/sparse, quantized 64-bit sums, shared-memory dense path |
| Approximate tree method | `tree/updater_gpu_hist.cu` (`grow_gpu_approx`) | `tree_method=approx` now runs on MPS; original Hessians are extracted and cuts built before sampling, once per updater call; row/column sampling reset per tree. GPU copy/root/partition operations are integrated. Both sampling methods and all roots share quantized state; ordinary raw CSR sketches now follow CUDA; adapter preprocessing remains hybrid |
| Gradient quantization | `tree/gpu_hist/quantiser.cuh` | GPU positive/negative binary64 bounds, CUDA CreateRoundingFactor scaling, truncation toward zero and minimum-one positive Hessians; scales remain in GPU buffers and guarded shared host views avoid eligible copies. Original per-tree scales feed both integer sampling methods, all roots and histograms; reduced value leaves now use original scales and integer sums; scalar/vector split statistics retain integers |
| Histogram subtraction, cache management | `tree/gpu_hist/histogram.cuh` | Metal-owned integer cache; external-page accumulation and subtraction retain exact quantized sums, with a binary64 view for host orchestration. GPU growth copies both buffers, empty first pages overwrite reused ranges, and feature gathering remains GPU. Scalar/vector split gathering now reads integer buffers; host cache policy/metadata remain |
| Split evaluation, categorical partitions, constraints | `tree/gpu_hist/evaluate_splits.cu`, `evaluate_splits.cuh`, `multi_evaluate_splits.cu` | Scalar CUDA-order integer prefixes/children, exclusive numeric scans, strict missing-direction ties, stable categories and software-binary64 gains. Vector integer scans retain inclusive forward/backward order, double chunk argmax followed by FP32 updates, projected stable category ordering, bounded/pooled weights and persistent child sums. GPU feature winner reduction, parent gains, unscaled final weights, monotonic-bound updates and categorical bitset assembly from resident sorted indices; GPU-owned interaction masks and monotonic bounds, group unions, ascending feature filtering, bound initialization/growth; candidate staging and tree metadata remain host |
| Row partition and final leaf assignment | `tree/gpu_hist/row_partitioner.cu` | Metal stable classify/prefix/scatter with cooperative scans for large nodes, resident row IDs, GPU leaf positions with guarded shared gradient inputs/output slices; supports numerical/categorical splits and missing values |
| Sampling and gradient sums | `tree/gpu_hist/sampler.cu`, `tree/updater_gpu_hist.cu`, `common/random.cu` | CUDA Philox uniform/gradient-based integer row sampling and shared integer root sums for every layout; gradient sampling uses original quantizers, binary64 norms, quantized prefixes, integer rescaling and saved-state replay; columns use modern CUDA Philox weighted priorities and Thrust Feistel shuffle with GPU compaction/ordering; shuffle key setup and cache metadata remain host as in CUDA, while feature arrays still stage through host |
| Squared/logistic/squared-log/gamma/pseudo-Huber/Poisson/Tweedie/hinge/multiclass/expectile gradients | `objective/regression_obj.cu`, `hinge.cu`, `multiclass_obj.cu` | Metal kernels with guarded synchronous host-memory sharing where eligible; means/weighted means, expectile initial gradients/Hessians and Newton stump fitting use Metal; regression and multiclass objective label predicates now use Metal; generic metadata validation and other initialization work remain CPU |
| Sigmoid/exp/hinge/softmax/argmax/expectile prediction transforms | Same objective files | Metal kernels; identity transforms require no dispatch |
| Base-score validation and margin conversion | `objective/regression_obj.cu` | Metal checked logistic/gamma intercepts, clamped FP32 logit and logarithms for gamma/Poisson/Tweedie/Cox. Identity mappings require no dispatch; expectile's host margin loop matches upstream host execution |
| Absolute-error and quantile objectives, associated statistics | `objective/regression_obj.cu`, `quantile_obj.cu` | Metal per-iteration residual scales, gradients and quantile prediction sorting; absolute-error initial weight sums, means, target-mean broadcast and stump fitting, and initial quantile estimation use Metal; other initialization work remains CPU |
| Ranking losses, pair construction, normalization | `objective/lambdarank_obj.cu` | Prediction order and NDCG ideal-label order use stable grouped Metal sorting; query-weight sums use Metal. NDCG discounts, ideal-gain calculation and ranking label validation also use Metal; biased top-k/mean pair construction, gradients, MAP prefixes and query normalization execute on Metal. Unbiased loss corrections, per-row bias costs, cross-query aggregation and regularized bias updates also execute on GPU; small bias vectors return to host for serialization |
| AFT survival loss | `objective/aft_obj.cu`, `common/survival_util.h` | Metal normal/logistic/extreme gradients and prediction/base-score transforms; GPU identifies small censored probabilities for original CPU gradient recomputation, exposed through correction counters/time. Full GPU coverage remains incomplete |
| Base score estimation, leaf value fitting | `tree/fit_stump.cu`, `common/stats.cu`, `tree/gpu_hist/leaf_sum.cu` | Means/weighted means, quantiles/medians, Newton stump gradient aggregation and unregularized weights on Metal; final vector-leaf refitting quantizes sampled float value gradients with original scales, uses GPU segmented integer sums and cross-page integer accumulation, then regularized/clipped weights; final tree updates stay host as in CUDA. Other initialization/state work still requires porting/audit |
| Batch/inplace/leaf tree traversal | `predictor/gpu_predictor.cu` | Metal: scalar/vector leaves, category bitsets, missing values, tree ranges, dropout weights; CPU input views stage dense chunks and recode categories |
| Prediction cache from leaf IDs | `predictor/predictor.cu` | Metal; supports strided target slices and negated sampling positions |
| SHAP contributions and interactions | `predictor/gpu_predictor.cu`, `predictor/interpretability/shap.cu` | Metal fixed eight-point quadrature, repeated-feature paths, scalar/vector leaves, categorical/missing decisions, dropout weights, bias addition, symmetry and diagonal finalization; model compression/root means remain host as in CUDA; dense input staging/category recoding remain host |
| RMSE/RMSLE/MAE/MAPE/logloss/error/Poisson/gamma/Tweedie/pseudo-Huber metrics | `metric/elementwise_metric.cu` | Metal loss calculation and scaled block reductions; final small reduction in host FP64 |
| Multiclass error and logloss | `metric/multiclass_metric.cu` | Metal loss calculation and scaled block reductions; final small reduction in host FP64 |
| Quantile and expectile metrics | `metric/elementwise_metric.cu` | Metal loss calculation, subnormal-preserving products and binary64 block reductions; small final host sum |
| AFT likelihood and interval-regression accuracy | `metric/survival_metric.cu` | GPU probability calculations and binary64 block sums; small final host reduction |
| ROC-AUC | `metric/auc.cu` | Binary, multiclass and multilabel CUDA-source term casts, batched sorting, binary64 prefixes and tied-score trapezoids; raw curve outputs, per-output aggregation metadata and normalization on GPU; collective transport remains host |
| PR-AUC | `metric/auc.cu` | Binary, multiclass and multilabel CUDA-source term casts, separate FP32 validity/recall totals, binary64 source integral and distinct binary/multi-output invalid results; GPU metadata and normalization; collective transport remains host |
| Ranking metrics | `metric/rank_metric.cu`, `metric/auc.cu` | NDCG/MAP/precision use CUDA-source arithmetic/fallbacks, grouped sorting, actual-label MAP prefix scans and query-sum reduction in one Metal command. NDCG discounts/inverse ideal gains and ranking label validation also use Metal; cache values are read back to existing host-owned arrays. Grouped ROC uses GPU stable label order, merge-based pair counts, query normalization and final aggregation. Grouped PR uses CUDA-source FP32 cast boundaries, original-order positive totals, binary64 prefixes/integrals, two-row validity and GPU aggregation |
| Quantile sketching, cuts, weighted/category handling | `common/hist_util.cu/.cuh`, `common/quantile.cu`, `data/gradient_index.cu` | Ordinary CSR matrices use Metal sorting, FP32 weight scans, category deduplication, resident pruning/merging and rank repair. Final cut materialization is host-side in both CUDA and Metal. Adapter/device-array ingestion, adapter-built QuantileDMatrix bins and distributed sketch collectives remain open |
| Compressed GPU matrix, external memory, iterators | `data/ellpack_page.cu`, `iterative_dmatrix.cu`, `quantile_dmatrix.cu` | Host GHistIndexMatrix remains authoritative; page-owned Metal copies and dense GPU transpose cache persist for each source page lifetime |
| Categorical recoding, metadata manipulation, validation | `data/cat_container.cu`, `metainfo.cu`, `array_interface.cu` | CPU |
| Linear coordinate-descent booster | `linear/updater_gpu_coordinate.cu` | GPU FP32 bias/feature gradient reductions, CUDA residual order/filter rules and unchanged selector snapshots with `updater=coord_descent`; host selectors, scalar weight/model updates, CSC preparation and linear predictions match CUDA's placement; default `shotgun` remains host |
| Gradient layout copies and Hessian extraction | `gbm/gbtree.cu`, `tree/updater_gpu_hist.cu`, `tree/updater_quantile_hist.cc` | Metal bit-preserving target-column copies, sampling-buffer layout conversion, reduced-value-gradient copies and unsampled Hessian extraction; guarded host sharing or staged snapshots for aliases. Allocation and other state manipulation remain host |
| Vector model-buffer copies and learning-rate scaling | `tree/multi_target_tree_model.cu` | Metal blit copies for base/child weights and category storage, and GPU FP32 learning-rate products for new child weights. Host model allocation/growth and metadata remain; buffers use guarded synchronous host-memory sharing or staging |
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

Precision, NDCG and MAP now follow `metric/rank_metric.cu`:

- Precision forms each term as `double(label) / double(n) * double(weight)` before
  reduction. Empty queries and top-k zero contribute zero while retaining their
  weight in the denominator. This replaces the CPU-style FP32 product and final
  division, which lost small terms and produced NaN for empty queries.
- NDCG first reduces `discount * gain`, then computes `(DCG * inverse_IDCG) * weight`.
  Its nonpositive-inverse fallback remains unweighted, as in CUDA. The `<= 0`
  predicate allows NaN inverse gains to propagate instead of selecting the fallback.
- MAP scans actual sorted labels in binary64, reduces `prefix / rank * label`,
  then computes `(area * weight) / min(total_relevance, topk)`. Empty/nonpositive
  relevance uses the **unweighted** one/zero fallback; CPU weights that fallback.
  Positive relevance with top-k zero still yields NaN, matching the CUDA formula.

MAP now uses the existing recursive segmented binary64 GPU prefix scan, replacing
integer hit prefixes and the CPU-ordered serial soft-label path. Accepted labels
near zero/one, including tiny negatives, retain their actual values. Empty groups
reset segment keys correctly, and prefixes cover the entire query even for top-k
metrics. Sort, scan, query scoring and final sum share one command; only the final
score and weight return to host. The existing `ranking_metric_dispatches` counter
counts completed commands, including the new scan. No statistics ABI changed.

These changes target source algorithms, **not a particular NVIDIA reduction tree**.
Severe cancellation can change MAP's zero-hit branch under different legal FP64
scan associations. The three-label cancellation tests explicitly document the
current Metal result and its difference from serial CPU; they do not establish
NVIDIA execution equivalence. The public finalizer still rejects NaN or excessive
scores according to upstream rules. This vendored version defaults bare `map`
(and other unqualified ranking metrics) to 32 ranks; explicit top-k tests also
exercise whole queries.

Validation includes 15 exact edge/order cases that distinguish the old CPU-based
implementation, plus 896 source-formula cases over 9,511,680 rows with strides,
empty groups, ties, zero/extreme weights and multi-level prefixes. Maximum scaled
source-reference error is `1.135e-13`, within the unchanged `2e-12` tolerance.
All 101 public ranking tests pass, including 16 added cases. Compatible CPU
comparisons remain; zero-relevance MAP expectations and precision terms use CUDA
formulas. Full validation passes **1,619 Python tests (one skipped)**, every rebuilt
native suite and the CPU-only build/API check. NDCG cache checks remain separate
below.
[Validation record](../validation/cuda-ranking-metrics-m3.txt).

### Grouped ROC CUDA-source contract

Grouped ranking ROC sorts labels stably on GPU and counts concordant/tied
pairs during parallel merges of prediction keys. This preserves CUDA's rule that
equal-label pairs participate in input order. It avoids the CPU's quadratic pair
enumeration and CUDA's quadratic pair scan: counts require `O(N log² m)` work for `N` rows and maximum query size
`m`, followed by integer reductions and software-binary64 normalization. Query
weights are squared in FP32, and half-weight products retain FP32 subnormal
rounding. CUDA computes each pair's FP32 prediction difference, maps it to
`1`, `0.5` or `0`, multiplies by the squared weight in FP32, scans pair terms and
weights in binary64, then divides if the weight sum is positive. NaN differences,
including equal signed infinities, contribute zero. The merge counter preserves
those classifications, including finite subnormal differences and signed zeros.

Queries shorter than three rows are excluded. Zero squared weights give zero
scores but retain the valid-query count. Finite weights whose FP32 square overflows
give NaN scores, as CUDA's `Inf/Inf` or `NaN/Inf` arithmetic specifies. The former
Metal host check copied CPU's error behavior; it has been removed. The shared
metric finalizer still checks non-NaN scores after collective aggregation and
warns on NaN. Private NaN query weights give zero scores because the scanned weight
is not positive; public metadata continues to reject nonfinite/negative weights.

For finite squared weights, the numerator is `wins*w + ties*float(0.5f*w)` and the
denominator is `pairs*w`. Integer counts are exact; the final binary64 products,
sum and division introduce a different association from CUDA's repeated pair
additions. Native CUDA-source pair enumeration and compatible CPU comparisons
retain the `2e-12` tolerance. This is a justified algorithm adaptation, not a
bitwise CUB/NVIDIA reduction guarantee. No actual NVIDIA run has been performed.
Million-row analytic cases use unit weights and exact expected scores, exercising
pair counts beyond 32 bits without running a trillion-comparison CPU oracle. They
exceed CUDA's `n_threads < INT32_MAX` pair-launch check and establish an extended
Metal domain. Empty/one-row groups are safely zeroed. CUDA's `TemporaryArray<double>`
allocation is uninitialized and those groups generate no pair writes, so a mixed
valid/empty batch can reduce unwritten slots. That undefined behavior is not copied.
Input arrays remain staged and host-owned. Only the aggregate score/valid count
normally returns to the host; optional native-test diagnostics expose per-query
results. The public metric retains distributed/final ratio handling on the host.

The existing **272 native cases / 234,208,157 independently enumerated pairs** and
three million-row analytic cases remain. **390 added query cases / 19,110 rows**,
plus an explicit overflow fixture (**2,168,663 enumerated pairs** combined), cover the source weight branches, half-weight rounding,
exceptional predictions, stable ties, short/empty queries, guards, unchanged inputs
and completed GPU counters. All **73 public grouped AUC tests** pass, including
nine overflow profiles, 12 source pair-term cases, model reloads and existing
training comparisons. Full regression passes **1,686 Python tests, one skip**,
every native suite and the CPU-only build/device check (2026-11-04,
11:43:43–11:47:54 UTC). [Validation record](../validation/cuda-grouped-roc-m3.txt).

### Grouped PR CUDA-source contract

Grouped PR-AUC now follows `metric/auc.cu::RankingPRAUC` and
`RankingPRAUCImpl`. Every false-positive term is computed in binary64 as
`(1.0 - double(label)) * double(weight)` and then rounded **once to FP32** before
binary64 prefix accumulation. Positive terms retain FP32 `label * weight`.
A separate GPU reduction obtains recall denominators from the original row order;
sorted prefixes no longer supply those denominators. This replaces the earlier
CPU first-term special case and FP32 subtraction on later rows.

The GPU integral now uses `metric/auc.h::CalcDeltaPRAUC`'s divisions and
difference-of-logarithms sequence, replacing the stabilized integral formerly
shared with classification PR. Validity follows CUDA's final `FP * TP > 0` and
**at least two documents** rule. A one-row query with a fractional label is excluded,
although CPU accepts it. Zero-weight queries remain invalid. GPU range checks
reject invalid relevance labels, including negative subnormals; NaN label queries
are excluded by the arithmetic validity test. Public metadata already rejects
nonfinite/negative weights; direct native calls also test their CUDA-formula
behavior without adding CPU-specific validity rules.

Stable sorting, actual FP/TP scans, original-order positive totals, integration,
query validation and aggregation execute within one Metal command. The existing
`ranking_pr_auc_dispatches` counter counts completed commands; no statistics ABI
changed. Arrays still stage through host-owned storage, and the public metric
retains the final host/distributed ratio. Empty queries safely contribute zero
score and valid count. That is an explicit extension: vendored CUDA code uses
`LastOf` on groups and compact `reduce_by_key` totals without guarding empty groups;
this port does not reproduce invalid indexing or shifted total slots.

Native validation passes **16 edge/cast checks** plus **297 production cases /
14,198,354 rows / 4,660 query results**, with a CUDA-source host oracle, retained
compatible CPU checks and an independent `log1p` integral. Maximum observed
absolute reference difference is `7.883e-15`; the original `2e-12` absolute/relative
tolerance is unchanged. Cases include million-row queries, three scan levels,
strides, empty groups, ties, fractional/subnormal labels, extreme weights,
nonfinite native inputs and six GPU label-validation errors. All **53 public
grouped-AUC cases** pass, including 21 new PR cases and model reload checks. Full
validation passes **1,640 Python tests (one skipped)**, every rebuilt native suite
and the CPU-only build/API check.
[Validation record](../validation/cuda-ranking-pr-m3.txt).

These tests establish source-formula agreement for the tested inputs, not NVIDIA
execution equivalence. CUB associations, NVCC contraction, libdevice logarithms
and severe cancellation can differ. Classification AUC now uses the source-aligned
implementation described below; grouped ROC still needs its source/rounding evidence
completed.

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


## Classification AUC arithmetic and normalization

Binary, multiclass and multilabel ROC/PR curves now follow `metric/auc.cu`'s
`BinaryAUC`, `MultiAUC` and PR-total rules. Each curve's negative term is
`float((1.0-double(label))*double(weight))`, then promoted to binary64; positive
terms round `label*weight` in FP32. PR independently reduces sorted FP32
`(1.f-label)*weight` and `label*weight` terms for validity/recall totals. The
negative total deliberately has a different rounding boundary from curve terms.
ROC retains raw FP/TP/area even for degenerate curves. Invalid binary PR returns
`{0,0,0}`; nonempty invalid multi-output PR returns `{1,1,NaN}`. Empty outputs are
zero triples. Those distinctions matter when workers aggregate results.

`xgb_auc_result` prepares local area, aggregation weight and integral on GPU.
`xgb_auc_normalize` implements CUDA `ScaleOutputs`: per-output division, prevalence
weights for multiclass ROC, macro averaging for multilabel ROC and PR, parallel
weighted sums, and the final ratio. Production multiclass/multilabel metrics use
both kernels. Host code retains the collective transport and stages the small
triples between commands, including local execution; fully resident aggregation
and GPU collectives remain open. Binary final host ratios match CUDA's placement.
Normalization commands increment the existing generic `metric_dispatches` and
GPU-time counters. Curve-family and sort counters retain their existing scope;
empty output preparation is recorded as a generic metric command. No statistics
ABI changed. Optional internal `AUCTrace` readbacks are used only by native tests.

**Numerical compatibility intentionally replaces the earlier stabilized PR
integral.** Both classification and grouped PR now use
`metric/auc.h::CalcDeltaPRAUC`'s difference-of-logarithms sequence. For a negative
row weighted `1e30` preceding a positive row weighted `1`, this source formula
returns `1`, while the mathematical integral is approximately `5e-31`. The old
Metal formula retained the small area; the new exact fixture requires `1`.
This is source alignment, not an accuracy improvement or proof of NVIDIA results.
With extreme weights, different legal prefix associations can also produce NaNs
or substantially different areas. These limitations are not hidden by comparing
only ordinary datasets.

Validation retains all prior input families and adds non-dyadic fractional terms:
**486 native cases / 7,607,088 score-label pairs for each of ROC and PR**, 420
independent weighted-pair checks and 519 independent integral checks. Ordinary
source/compatible CPU thresholds remain unchanged; maximum ordinary PR source
error is `1.660e-14`. Four exact output/cast fixtures and **25 GPU normalization /
metadata cases**, including 65,535 outputs, empty metadata, NaNs, zeros, guards
and rejected shapes, test the new operations. All **97 public AUC tests** pass,
including 26 added cases, model reloads and explicit normalization-dispatch checks.

The extreme-weight validation contract is explicitly different from the former
stabilized-integral contract. It verifies **1,690,464 FP/TP terms exactly** and
**1,690,464 prefixes against independent 320-bit integer sums**. Prefix error uses
`(gamma_d + 2u) * exact_sum`, with `u=2^-53`, `gamma_d=d*u/(1-d*u)` and a bound on
the depth of the GPU additions. Recall totals receive their own reduction bound.
The source integral is then evaluated on the captured GPU endpoints: four ULPs
per logarithm are propagated through the source expression, plus reduction error.
This is a tested math budget, not a libdevice accuracy guarantee. All 78 checked
extreme curves pass, including **15 source-formula NaNs**; maximum prefix/bound
and integral/bound ratios are `0.1690` and `0.01976`. CUDA's CUB association, compiler
contraction, actual NVIDIA execution and cross-Apple behavior remain unverified.
Full regression passes **1,666 Python tests, one skip**, all rebuilt native suites
and the CPU-only build/device check (2026-11-04, 11:35:37–11:39:51 UTC).
[Validation record](../validation/cuda-classification-auc-m3.txt).

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

Column selection follows `common/random.cu` and the modern branch of
`common/random.cuh`. The parent consumes one seed per actual sample, with unchanged
per-tree/per-level caching and per-node resampling. Rates of one reuse the input
without a seed or sampler dispatch. Feature initialization and final integer-ID
sorting remain GPU operations.

Weighted selection now uses the existing Philox4x64 implementation: seed, discard
by candidate position, then `uniform_real_distribution<float>`. Every candidate
independently generates its draw, gathers its feature weight, applies the `1e-6`
floor and computes FP32 `log(u)/weight` on GPU. Priorities sort stably descending,
the requested prefix is selected, and feature IDs sort ascending. The modern
Philox and older Thrust weighted engines differ; the older stream is not claimed.

Unweighted CUDA sampling uses **Thrust's default minstd_rand engine and Feistel
shuffle**, even with modern Philox available. Upstream constructs 24 round keys
on host using a binary64 uniform distribution and integer conversion. Metal keeps
that placement and those rounding boundaries. GPU threads apply the 24-round
bijection across a padded power-of-two domain, then a 64-bit inclusive prefix and
stable compaction retain indices within the actual column count. This follows
Thrust's shuffle, not its separate cycle-walking adapter. The former CPU MT19937,
Fisher-Yates and fused small-column kernels have been removed. The pinned source
is [NVIDIA CCCL](https://github.com/NVIDIA/cccl/tree/3d0d040ac0dc5907daf2a0731edb378e5562099d),
particularly its [shuffle](https://github.com/NVIDIA/cccl/blob/3d0d040ac0dc5907daf2a0731edb378e5562099d/thrust/thrust/system/detail/generic/shuffle.inl)
and [bijection](https://github.com/NVIDIA/cccl/blob/3d0d040ac0dc5907daf2a0731edb378e5562099d/thrust/thrust/detail/random_bijection.h).

**Same-seed sampled models intentionally change from the former CPU stream.**
Native source tests preserve all old input families and add seed-modulus boundaries:
**550 cases / 3,281,095 selected IDs / 4,610,295 exact Philox draws / 1,320,896 exact
permutation entries**. They cover million-column inputs, padded/non-power-of-two
sizes, signed/high feature IDs, weight floors, deliberate priority ties, guards,
unchanged inputs, cached levels, parent RNG state and completed GPU counters.
Optional diagnostic permutation readback is used only by tests. The internal
`SampleFeatures` signature changed; final validation rebuilds all native callers.
No public API or statistics layout changed.

The independent reference was also compared with **actual, unmodified pinned
Thrust/CCCL headers compiled for the host**: 66 permutations / 434,730 indices,
144 round keys and 60,000 Philox uniforms agree exactly. This is stronger than two
local rewrites agreeing, but is still not NVIDIA execution. The optional probe
is `tests/cuda_column_upstream.cc`; build details are in the validation record.

Weighted log priorities retain the tested Apple-compatible logarithm. All native
keys in this corpus match host `std::log(float)` exactly, including deliberate
ties. The existing 8,086,019-input log check remains separate local math evidence.
**This does not guarantee NVIDIA libdevice rounding**: tightly spaced priorities
can change order across math libraries. New public weighted-selection fixtures
require a boundary separation exceeding eight FP32 ULPs of their independent
logarithm reference. No tolerance was widened to conceal a failed selection.

Public coverage adds **48 scalar/vector training cases**, checking independently
predicted single-feature roots over three rounds, hierarchical rates, zero/varied
weights, seeds and CPU-loaded model predictions. All 88 column tests and a broader
307-test focused run pass. Unsampled vector fixtures retain original CPU-training
comparisons; sampled fixtures now use source split arithmetic and same-model CPU
prediction checks at unchanged tolerances. A custom-objective intercept mismatch
in the first test revision was corrected by retaining the original built-in
objective for the unsampled baseline.

Host feature vectors and cache metadata remain authoritative. One command contains
sampling, compaction/priority ordering and final ID sorting, then selected IDs
return to host. `column_init_dispatches`, `column_sampling_dispatches` and
`column_sampling_gpu_time_ns` retain their scope; GPU time excludes host key setup,
staging and waiting. Resident ownership, actual NVIDIA/cross-Apple checks and
broader training-schedule correspondence remain open. No new benchmark was run;
historical CPU-stream/fused-kernel timings are not measurements of this version.
Full validation passes **1,734 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/device check (2026-11-04, 11:59:26–12:03:47 UTC).
[Validation record](../validation/cuda-column-sampling-m3.txt).

## CUDA-derived resident sketches

`metal_cuda_sketch.cc`, `metal_cuda_sketch.h` and the runtime/shader now implement
ordinary CSR ingestion from `DeviceSketchWithHessian` and the single-worker
`SketchContainer` lifecycle. `SketchOnDMatrix` dispatches to this path for MPS
regardless of the CPU `use_sorted` setting. It covers in-memory and external-page
ordinary matrices, including group weights and Hessian-weighted `approx`.

- Raw windows retain CUDA's 64M-entry partitioning and CSR-based covered-row counts.
  Stable radix sorting orders values then feature IDs. GPU binary search gathers
  row weights with page base-row offsets; group/Hessian unification uses Metal.
- A hierarchical FP32 inclusive scan computes per-feature weights. Categorical
  duplicates and their attached scanned weights are compacted on GPU before
  sketch ranks are formed. Unweighted ranks use sample positions, not a scan of ones.
- Per-window budgets use CUDA's row count, epsilon and `LimitSizeLevel`. Pruning
  retains endpoints and answers FP32 rank queries. Reverse input scanning merges
  duplicate raw weights while avoiding double-counted selected summaries.
- GPU summaries/column pointers persist across pushes. Parallel merge partitions
  implement opposite-summary rank contributions, equal-value combination and
  first-entry uniqueness. Intermediate pruning follows cumulative rows seen.
- Final summaries are copied once for host `QueryCutValues` and category validation
  and expansion, matching CUDA `MakeCuts` placement. No host level queues or sorted
  CSC pages participate in ordinary MPS sketching.

Numerical and placement limits are explicit. Thrust does not promise the same
ordering of equal feature/value keys as our stable sort. CUB may associate FP32
scans differently; the public oracle states Metal's 256-entry tree association,
while native dyadic-weight fixtures have exact sums independent of association.
Input-summary reverse scans use one GPU lane per column in a deterministic order.
CUDA's in-place `FixError` reads neighboring ranks while other threads may write
those ranks; Metal applies its equations left-to-right per column, avoiding the
race and ensuring monotonic rank bounds. These are not NVIDIA bitwise-equivalence
claims. Raw CSR/metadata staging, allocation and size-readback synchronization
remain. Group/Hessian weights currently stage through host memory before the
resident sketch uploads them. Performance has not been measured for this path.

Ordinary CPU and Metal cut caches are tagged by sketch backend. An initialized
training request switching backends rebuilds the index; cut inspection/prediction
and `forbid_regen` reference requests preserve it. QuantileDMatrix fixed cuts stay
fixed. The adapter/device-array path and distributed GPU sketch exchange are not
implemented by this change.

Native tests verify 64 cases, 276 pushes, 1,867,178 raw entries and 229,976 exact
summary fields against an independent scalar CUDA-equation reference. Cases cover
window splits inside rows, empty rows/columns, base-row offsets, duplicates,
categories, zero/tiny/large weights, signed-zero weights, infinite feature values
and the one-row pruning budget. Intermediate pruning retains both index and
value uniqueness predicates: equal infinities do not compare equal by subtraction. Public tests
verify exact source-oracle cuts, weighted/approx ordering, external pages, backend
switches and reference preservation. Thirty `approx` cases independently check cuts and all scalar integer splits/
leaves across two one-round calls with changing Hessians, both samplers and growth
policies. CPU tree-equivalence tests now explicitly fix their cuts; ordinary
sketch tests no longer require the distinct CPU algorithm's thresholds.

A subnormal-cut integration fixture also exposed a prediction bug: Metal float
comparison flushed tiny thresholds while CPU traversal distinguished them.
Numeric and category traversal now use the existing bit-preserving comparison;
five threshold fixtures and a negative-subnormal category fixture cover the fix.
Final full validation passes **1,773 Python tests, one skip**, every rebuilt
native suite/reference updater and CPU-only build/device check (2026-11-04,
12:32:43–12:36:45 UTC). The Python suite took 93.03 seconds.
[Validation record](../validation/cuda-resident-sketch-m3.txt).

## Numeric sketch finalization (earlier CPU-compatible helpers)

The following records the earlier implementation and its original validation.
These helpers remain tested; ordinary MPS matrices now bypass this level-queue
pipeline in favor of the resident CUDA-derived path above.


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

## Sketch weights, sorted summaries and streaming compaction (earlier implementation)

The historical sorted/level-queue path below is superseded for ordinary MPS
matrices by the resident sketch above; the weighting helper remains in use.

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
final leaf values. Production MPS now follows `updater_gpu_hist.cuh::Reset` and
`gpu_hist/leaf_sum.cu`: construct the value quantizer from original gradients,
replay sampling in FP32 before tree growth, convert each sampled float pair with
the original scales, and sum integer pairs per leaf and across pages. This is a
different path from rescaling the already-quantized split gradients; substituting
that path would change value-gradient rounding. Minimum-one positive Hessians are
preserved at the sampled-float conversion, as in CUDA.

The new `metal::GradientQuantizer` owns original per-target scales without
allocating integer rows. It shares the GPU sign-separated bound/scale builder
with `QuantizedGradients`. Noncontiguous target columns use a GPU layout copy;
raw staging and bound workspace are released after completion. The leaf state
retains quantizer ownership and copies the small scale array on the GPU. Neither
sampling rejection nor individual external pages recompute the scales.

Resident row partitions provide leaf segments and global row indices. Metal
converts indexed sampled floats using their original scales, reduces integer
blocks, and adds integer leaf totals across external pages. A separate binary64
view is refreshed for inspection and final weight calculation; rounded views
never feed later integer accumulation. Persistent sums use 32 bytes per leaf/target
for the integer state and its binary64 view. Host code still owns segment metadata,
allocation and the final tree update, as previously documented.

The weight calculation preserves the upstream L1/L2 formula, maximum-step clipping,
FP32 conversion, bounds and subsequent learning-rate multiplication. Nonpositive
Hessians produce zero before bounds. `min_child_weight` governs split validity,
not an extra final-weight cutoff. Public reduced-gradient training still rejects
monotonic constraints, as upstream CPU and CUDA do. The API checks 65,535 targets,
32-bit leaf/task indices and Metal buffer-size limits. Nonfinite or out-of-int64
sampled-value conversions raise an explicit error; CUDA C++ conversion is undefined
for those inputs. No portable CUDA result is claimed for that domain.

Independent host CUDA-formula tests process **11,594,018 pairs in 250 cases** and
check **20,458 materialized sum fields** exactly, plus original positive-curvature
scales and conversion-boundary cases. They cover C/F/padded/broadcast views, global
page offsets, output-slot permutations, empty leaves, million-row input, source
lifetime, float replay, cancellation across pages and final weight formulas.
**24 additional public cases** check exact cancellation and retention of original
value scales after uniform/gradient-based sampling, including zero-bin and external
pages, cached/in-place predictions and reloads. Existing CPU comparisons keep their
original tolerances. The two-argument low-level `MakeLeafSums` retains its legacy
binary64 tests; production always supplies the original value quantizer.

`leaf_integer_sum_dispatches` counts successful integer page reductions within
`leaf_sum_dispatches`; initialization and final weight counters remain separate.
`leaf_fitting_gpu_time_ns` includes value-bound calculation, clearing, leaf sums
and weights, excluding staging/layout copies and host tree updates. CPU-only builds
expose the new counter as zero. See [validation](../validation/integer-leaf-sums-m3.txt).
NVIDIA execution validation and other training-state gaps remain; no new performance
benchmark was run.


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

The numerical rules now follow `updater_gpu_coordinate.cu`: bias and feature
sums return FP32 `GradientPair`, without negative-Hessian filtering. Feature terms
round `g*x` and `(h*x)*x` in FP32; residual terms use `(h*step)*x`, then add the term
to the gradient and positive zero to the Hessian. Bias updates skip a zero step;
feature updates still execute, preserving exceptional-value and signed-zero
behavior. Host `CoordinateDelta` formulas and weight updates remain unchanged.

The previous implementation reduced in binary64, filtered Hessians and used
CPU residual order. It also copied residuals back for greedy/thrifty selectors.
CUDA instead supplies the original input gradient snapshot to both `Setup` and
`NextFeature`, and leaves that input unchanged after the update. MPS now does the
same. Explicit fixtures show CUDA-style selectors choosing a different feature
from CPU after a bias update; CPU topology/weight identity is not asserted there.

Metal uses recursive balanced FP32 reductions with 256-thread groups. It does
not emulate the tile policy of a particular CUB build. The legacy
`cub::DeviceReduce::Sum` called by this XGBoost source can use different reduction
orders across GPU architectures; see [NVIDIA's CUB documentation](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html).
NVCC contraction and exact NVIDIA execution remain unverified. Subnormal-preserving
FP32 helpers avoid Metal's native FTZ behavior. Counts are checked against CUDA's
signed 32-bit reduction interface rather than silently narrowed.

CSC data and mutable gradients are owned on GPU for each boosting update, as
before. Feature selectors receive the untouched original host gradients. Production
training no longer reads residuals back; the low-level readback API remains for
validation. Sorted duplicate row entries are processed serially on Metal to avoid
write races. CUDA's one-thread-per-entry residual update races on such duplicate
rows, so equivalence is claimed only for its defined distinct-row case; the
serial duplicate extension remains explicitly tested. Multi-page input remains
rejected as in CUDA. Cross-round storage reuse is not implemented.

Native validation retains **49 cases / 4,118,809 source pairs** and checks **3,564
sum fields**, of which **2,991** permit exact comparison or nonfinite classification.
For other FP32 sums, an independent high-precision sum of source terms is checked
against the balanced-tree forward-error bound `gamma_d * sum(abs(terms))`; the
maximum observed error is 0.098772 of that bound. This replaces the obsolete
binary64 contract explicitly, rather than relaxing a failed FP32 comparison.
**148,277,125 residual fields** retain exact finite-bit comparisons and NaN
classification against CUDA's multiplication/filter rules. Checks include empty/
million-row data, multiple targets, subnormals, negative Hessians, overflow,
nonfinite values, duplicates, input preservation, guards and malformed metadata.

Public validation uses a test-only host updater derived from the CUDA source,
with an independent adjacent-pair FP32 reduction. It shares the upstream host
selectors and coordinate formulas that CUDA itself calls; it calls no Metal code.
All 136 previous public scenarios remain, augmented by two selector-divergence
fixtures, ten negative-Hessian cases and ten direct production input-ownership
checks. Training spans all five selectors, dense/missing/CSR data, multiple
objectives/targets, penalties, `top_k`, zero steps, convergence and continuation.
The existing `rtol=atol=2e-5` is unchanged. CPU comparisons remain where selector
algorithms agree; actual NVIDIA execution is not claimed. Final validation passes
**1,603 Python tests (one skipped)**, every rebuilt native suite and the CPU-only
build/API check. [Validation record](../validation/cuda-linear-m3.txt).

`linear_gradient_dispatches` and `linear_residual_dispatches` count completed
commands; `linear_gpu_time_ns` covers GPU time, excluding host selection, scalar
updates and staging. Empty operations and zero bias steps do not dispatch; zero
feature steps do. No performance benchmark accompanies this alignment update.


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

This paragraph describes the earlier updater milestone; subsequent integer-state,
row/column sampling and resident-sketch sections supersede these gaps. At that
point, the underlying sketch, RNG and tree-statistic storage still required CUDA alignment. Histogram quantization now follows the CUDA formulas, but its
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
external-page inputs and both row samplers. The original cut reference used CPU approx without sampling. It now uses the
independent CUDA-equation sketch oracle described above, including the declared
Metal FP32 scan association. An additional thirty cases audit sketch-to-tree
behavior over changing Hessians.
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
methods and every root layout. Remaining work includes vector integer split scans/child statistics. Reduced value-leaf
sums now use original value quantizers and integer page accumulation. Histogram cache page
addition/subtraction now retain integer statistics, as described below. Full Python regression validation passes without tolerance
changes. See [validation](../validation/cuda-quantizer-m3.txt).


## Integer histogram cache and cancellation

Metal histogram caches now preserve quantized 64-bit gradient/Hessian pairs
through page accumulation, growth, slot reuse and subtraction, matching the
integer arithmetic in `tree/gpu_hist/histogram.cuh::SubtractionTrick`. A separate
binary64 view is materialized for the legacy vector split evaluator and host tree
orchestration. Scalar split evaluation reads the integer cache directly. Rounded values are never fed back into integer cache arithmetic.
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
Later scalar and vector split updates retain integers through winner selection
and child-node propagation, rather than consuming the rounded binary64 views.
All roots share integer state and both sampling methods follow original-gradient
quantization. Reduced value-leaf sums have their own original quantizers and
integer cache. Histogram/leaf cache arithmetic alone did not resolve split gaps;
the dedicated split updates below provide that alignment.

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

**Still unfinished:** split scans/child statistics, column/ranking RNG,
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


## Scalar CUDA-order integer splits

The scalar `hist` and `approx` builders now gather the histogram's original
integer buffer, using its retained quantizer. Root reduction returns both the
integer sum and its binary64 view in one GPU command. Each node retains its exact
integer parent and winning child sums; loss-guide entries cannot lose their
statistics when split scratch or histogram-cache slots are reused. Base weight
and parent cover use the materialized original parent, rather than adding two
rounded child views.

The source authority is `gpu_hist/evaluate_splits.cu`, `gpu_hist/evaluator.cu`,
`updater_gpu_common.cuh` and `split_evaluator.h`:

- Numeric candidates use exclusive integer prefixes, including the boundary at
  negative infinity. Missing sums and both children are formed in integers;
  only gain/weight inputs convert to binary64.
- Numeric and one-hot candidates compare missing-left and missing-right gains
  with strict `>`; ties go right. Candidate ties retain the first CUDA enumeration
  position. Partition candidates retain forward-before-backward order, the
  `max_cat_threshold - 1` candidate limit, and stable category-weight ordering.
- Candidates compare raw FP32 child gains. Host feature reduction preserves
  sampled-feature order on ties, then subtracts parent gain once. The old
  conservative FP32 screening path is not used by this scalar CUDA-order kernel.
- IEEE subnormal category weights and gains retain their ordering through integer
  comparison keys/software-binary64 comparisons. Native Metal FP32 comparisons
  flush these inputs; exact tests exposed and prevented that difference. Shared
  scalar weight comparisons now also preserve tiny bounds/max-delta parameters.

Local verification uses independent host CUDA-formula oracles, not CPU-booster
identity. The native shader test compares **792 cases / 123,984 bins**, including
**72 cancellation cases**, with exact integer children, materialized sums, gains,
selected bins/directions, category order, partitions and output guards. Native
shared-state tests also compare returned root integers directly with independent
quantization/sampling sums.

**85 new public training cases** audit complete scalar trees: the best feature
and candidate across all allowed features, integer row/child membership, gains,
cover, leaf weights and premature stopping below growth limits. They cover
numeric/one-hot/partition features, missing and
empty columns, regularization, monotonic and interaction constraints, depthwise/
loss-guide growth, cache capacities 1/64, external pages, uniform sampling and
`hist`/`approx`. Six cancellation cases force a deeper split to recover small
child sums from an integer parent whose binary64 view lost them. Predictions and
CPU-loaded model predictions agree exactly in these cancellation cases.

The existing column-sampled scalar split tests now audit the best CUDA candidate
within each selected feature; they do not claim to validate CUDA's column RNG.
At this scalar checkpoint, vector split tests retained their prior CPU comparisons;
the vector update below supersedes that assumption. Root-reduction tests now
prevent splitting so CPU comparisons test root arithmetic independently of
incompatible split tie rules. No numerical tolerance was widened.

At this scalar checkpoint, vector-leaf integer splits remained (now addressed
below). Remaining scope includes interaction metadata; GPU feature-winner
reduction, unscaled weight calculation and monotonic bounds are addressed below; CUDA sketch
and column-RNG differences; extreme nonfinite gain/host expansion semantics; and
actual NVIDIA execution. Gain arithmetic currently uses explicit separately
rounded binary64 operations with an FP32 leaf-weight square. NVCC contraction
and build-dependent floating-point behavior have not been compared on NVIDIA;
local exact results are not a hardware equivalence claim. No performance
benchmarks were run for this change.

Validation: [scalar integer splits on M3](../validation/cuda-integer-splits-m3.txt).

Regression migration is explicit: the first complete run passed every native
suite but reported 28 CPU-model-identity failures (1,367 Python tests passed).
Strict CPU identity fixtures now use positive weights where zero-weight missing
rows previously created different legitimate default directions. Zero-weight
coverage remains in the CUDA-source training audits, including weighted `approx`.
Scalar row/column sampling and resident-cache tests now audit the recorded raw
gradients against exact source-derived integer states/candidates. Gradient sampling
exposes its original host-oracle integers, avoiding a lossy FP32 bridge. Repeated
built-in-objective training still matches recorded-objective training exactly in
the sampling/cache tests. Changing-data workspace tests compare predictions for
the same serialized model on CPU/MPS and audit each new scalar tree. Vector tests
retain their CPU model comparisons. The focused migration run passes 393 tests
(one skipped). The complete final suite passes **1,395 Python tests (one skipped)**,
all native suites and the CPU-only build/API check. A final strengthening of the
oracle's leaf-stopping conditions passes all 222 affected tests. See the validation
file for separate logs and the initial failure record.


## Vector CUDA-order integer splits

The multi-output `hist` builder uses original integer histogram buffers and
per-target quantizers. Exact root integers and selected child sums persist by node
ID, including loss-guide queues and reused histogram/split workspaces. The separate
original-value quantizers and float sampling replay for reduced-gradient leaf
fitting are preserved.

Source authority: `gpu_hist/multi_evaluate_splits.cu`, `evaluate_splits.cuh`,
`updater_gpu_common.cuh` and `split_evaluator.h`.

- Numeric scans are inclusive; forward candidates send missing values right,
  backward candidates send them left. Both passes run, even without missing rows.
- One-hot candidates enumerate all missing-right choices before all missing-left
  choices. Partition candidates use a sorted right-child prefix/missing-left and
  left-child suffix/missing-right, limited to `max_cat_threshold - 1` candidates.
- Category scores accumulate binary64 sums of FP32 child/parent-weight products;
  ordering is stable. Each target uses its own original gradient/Hessian scales.
- A 32-candidate chunk chooses its binary64 argmax before narrowing to FP32.
  Strict FP32 comparisons select between chunk winners. Global-double argmax and
  early per-bin FP32 rounding would both differ; a dedicated fixture catches both.
- Feature reduction compares raw FP32 winners, breaking ties by lower feature ID.
  It recomputes parent gain with the current bounds and subtracts that binary64
  gain only after selecting the winning feature.
- Unconstrained gains use double weights/formulas. Constrained gains use bounded
  float weights and a float weight square; violated monotonic order pools weights
  using twice the FP32 penalties. A zero-Hessian target still receives bounds when
  the average Hessian permits a split. Shared max-delta/alpha predicates preserve
  subnormals through bit checks.

The production API exposes `cuda_integer_multi_split_dispatches`, also included
in the general/resident split counters; CPU-only builds expose zero. The statistics
ABI grew, so native API test executables must be rebuilt before use. Legacy
binary64 split APIs remain tested but production vector training no longer uses them.

Independent direct-kernel tests pass **1,528 tasks / 76,102 exact child pairs**,
with 1–129 targets, 0–131 bins, both directions and CUDA's numeric one-pass test
hook, regularization, zero Hessians, bounds, stable categories, heterogeneous
scales, cancellation, subnormal values and output guards. Comparisons include
exact gains, bins, missing directions, partitions, category ordering, integer
children and their binary64 views.

**94 public cases** check complete vector trees: 64 layout/constraint/growth/cache
combinations; 9 uniform/gradient-sampled cases; 6 zero-Hessian-target cases;
6 deep cancellation cases; 3 equal-feature/leaf-limit cases; and 6 reduced-gradient
cases with independently quantized original value scales and float replay. The
reference checks best allowed features/candidates, exact child row membership,
gains, covers, base/leaf weights and premature stopping below growth limits.
Column-sampled cases check the best candidate within the selected feature, not
CUDA column RNG equivalence. CPU-loaded models produce identical predictions.

The initial existing Python suite passed 1,394 cases and failed one strict CPU
vector topology assertion: a categorical missing-direction tie. That test now uses
recorded gradients and the complete CUDA-source reference, while retaining its
CPU numerical prediction comparison at the original tolerance. The focused run
passes all 118 cases; no tolerance was widened. Final validation passes **1,489
Python tests (one skipped)**, every native suite and the CPU-only build/API check.
A final parent-gain source correction was rebuilt and verified with the complete
Python suite; the native shader and API source were unchanged after their full run.
[Validation record](../validation/cuda-integer-vector-splits-m3.txt).

At this vector checkpoint, feature reduction, weight calculation and constraint
updates remained on CPU; the following update moves them onto Metal. Interaction
masks and generic model-buffer operations remained unported at that checkpoint;
model copies and learning-rate products are covered by the update below. Generic
state ownership, sketches, RNG differences and the other ledger entries remain.
The source-derived model uses separately rounded binary64 gain operations; actual
NVCC contraction/build behavior has not been compared. Extreme nonfinite gains
and host expansion rules still require alignment. Local tests do not establish
NVIDIA hardware equivalence. No performance benchmark accompanies this change.


## GPU split finalization and constraint bounds

`FinalizeSplitNodes` reads the resident output of the scalar/vector integer split
kernels. A 32-lane group reduces each node's raw feature candidates, retaining
sampled-feature order on scalar ties and lower feature ID on vector ties. It then
calculates bounded base/child weights, constrained pooled vector weights, current
parent gain and vector Hessian traces. Scalar parent gain/subtraction stay FP32;
vector gain accumulates in binary64 and subtracts before the final FP32 conversion.
The GPU returns a winner index, weights and statistics, which are retained by node
ID across loss-guide queues and histogram-cache reuse. The host no longer recomputes
these values when applying a split. Root-weight calls use the same finalizer with
empty feature ranges, including completely empty feature matrices.

The source authority remains `gpu_hist/evaluate_splits.cu`,
`multi_evaluate_splits.cu` and `split_evaluator.h`. Vector root cover now uses CUDA's
binary64 target trace before conversion to model FP32. The previous CPU-style
sequential FP32 sum could lose several small positive target contributions.
The source oracle was updated to match that CUDA rule; CPU behavior is unchanged.

`UpdateSplitBounds` computes/scatters monotonic child bounds on Metal. The scalar
midpoint is `(left + right) / 2`; the vector midpoint is
`left + 0.5 * (right - left)`, with distinct FP32 rounding boundaries. Software
operations preserve subnormals. Weight inputs are snapshotted; bounds use guarded
borrowed host memory or initialized copies. Overlapping bounds, invalid node IDs
and layouts are rejected; NaN midpoints report an error rather than silently
writing invalid constraints. The tree evaluator has an explicit Metal flag,
because this backend's `Context::Device()` still represents host-owned CPU data.

Counters `split_finalize_dispatches` and `split_bounds_dispatches` identify these
completed GPU operations, including root-only finalization. Their times are added
to `split_gpu_time_ns`, but their dispatches do not increase the existing candidate
split counters. CPU-only builds expose both counters as zero. Statistics ABI and
tree-evaluator layout changed; native test executables must be rebuilt.

Direct independent CUDA-source tests compare **448 finalized nodes / 11,904 target
weights**, with scalar/vector ties, 0–129 features, 1–129 targets, invalid candidates,
zero Hessians, regularization, bounded/pooled weights, subnormals and output guards.
**90 production bound updates** cover both midpoint formulas, three monotonic
orders, multiple target counts, tiny/large values, aliases and untouched nodes.
Additional API checks reject stale/invalid finalization layouts, overlapping or
out-of-range bounds, and NaN midpoints. Thirteen public cases audit complete trees,
assert dispatch/bound counts, exercise multi-chunk feature selection and verify
root traces that differ from CPU's FP32 accumulation. The first complete Python
suite passes 1,502 tests (one skipped). Final validation also passes **1,502
Python tests**, every native suite and the CPU-only build/API check.
[Validation record](../validation/split-finalization-m3.txt).

At this checkpoint, interaction masks and constraint allocation/initialization
remained; interaction masks are addressed below. Monotonic-bound initialization
and persistent GPU ownership remain. Categorical bitset materialization, missing at this
checkpoint, is covered by the later update below. Finalization
currently uses a separate synchronous command: candidate readback/staging has not
been eliminated. Host tree orchestration remains. Vector model-buffer copies and
learning-rate application were the next missing operations, addressed below.
Extreme nonfinite candidate/expansion rules,
NVCC contraction and actual NVIDIA execution remain unverified. Other coverage
ledger entries are unchanged. No performance benchmark accompanies this step.

## Vector model-buffer copies and learning-rate scaling

The source authority is `tree/multi_target_tree_model.cu`: `CopyBatch` calls
device-to-device batch memcpy, and `ApplyLearningRate` multiplies each new child
weight by `eta` with a single FP32 operation. The Metal branch in
`multi_target_tree_model.cc` now routes float weights and category words through
`CopyModelBuffer`, then scales only the new child-weight slice through
`ScaleModelWeights`. Parent weights are copied unchanged, and old untouched nodes
and category words survive model growth. Scalar tree learning-rate multiplication
remains on the host, as it does in the upstream CUDA updater.

Copies use a Metal blit encoder and preserve all 32-bit payloads, including NaNs,
signed zero and category bit patterns. Disjoint destination ranges can arrive in
any order; overlapping destinations and invalid extents are rejected before
dispatch. Sources overlapping the destination are snapshotted before any writes.
The frontend also snapshots sources referring to the old model before resizing
its host vector. Eligible host memory is borrowed through guarded shared buffers;
unaligned or otherwise ineligible memory uses initialized staging buffers, so
holes and untouched prefixes remain intact.

The scaling kernel loads/stores raw bits. Its software-binary64 product of two
FP32 operands is exact before one FP32 rounding, preserving gradual underflow,
signed zero and overflow. This avoids Metal's native subnormal flushing. NaN
arithmetic is checked by classification; copy operations retain exact NaN bits.
The local reference is an independent host FP32 multiply with contraction disabled.
Actual NVIDIA execution and alternate compiler/FTZ settings remain unverified.

Completed-operation counters are `model_weight_copy_dispatches`,
`model_category_copy_dispatches` and `model_scale_dispatches`; `model_copy_words`
counts copied 32-bit words, and `model_state_gpu_time_ns` sums these command times.
Empty copies/scales do not dispatch. CPU-only builds expose all fields as zero.
The statistics ABI changed, so native API test executables must be rebuilt.

Native checks pass **81 copy cases, 80 scale cases / 1,053,248 products, and 16
growing trees**. Cases cover unsorted destinations, aliases, read-only sources,
unaligned buffers, holes/guards, exceptional FP32 values and invalid inputs.
Integrated trees use 2–129 targets, multiple expansions, unscaled parent weights,
preserved older nodes, aliased category sources across resize and serialization.
Nineteen public cases cover numeric/one-hot/partition trees, depthwise/loss-guide
growth, zero/ordinary/unit learning rates, continuation from serialized models,
exact dispatch/word accounting, complete CUDA-source tree oracles and CPU-loaded
prediction equality. Root-only trees perform no model copy/scale dispatch.
Final validation passes **1,521 Python tests (one skipped)**, every rebuilt native
suite and the CPU-only build/API check. [Validation record](../validation/model-state-m3.txt).

This completes the two operations in that CUDA translation unit, not all model
state handling. Model allocation/growth remains host-owned and both operations
are synchronous. Category bitset assembly was the next gap, addressed below.
Interaction masks were a subsequent gap, addressed below. Generic state ownership and other ledger gaps remain
unfinished. No benchmark was run.

## Categorical split bitset assembly

`BuildSplitCategories` now builds the winning one-hot/partition bitsets on Metal
for scalar and vector training. Its source authority is scalar
`gpu_hist/evaluate_splits.cu::SetCategoricalSplit`, vector
`gpu_hist/multi_evaluate_splits.cu` and `common/bitfield.h`. Category zero occupies
the most significant bit of the first 32-bit word. One-hot uses the selected cut;
partition mode consumes the sorted prefix length already calculated at CUDA's
direction-dependent threshold boundary by the split kernel.

The operation reads the resident quantized candidate records and sorted indices.
Each workgroup first clears its entire bitset, synchronizes device memory, then
sets selected bits with atomic OR. Numeric and invalid candidates leave zeroed
storage, including on reuse. Cuts and request metadata are staged; completed
bitsets return to host-owned tree metadata. The two production quantized split
evaluators now accept a null sorted-index output, avoiding their previous order
readback in scalar/vector training. Other candidate readbacks remain.

Host checks validate candidate IDs and cut extents against logical input sizes,
including stale-workspace rejection after empty or nonquantized evaluation.
GPU checks catch malformed bin/partition/order indices, invalid categories and
insufficient output capacity before any out-of-bounds access. Category checks
use raw float bits so negative subnormals cannot disappear through Metal FTZ;
valid fractional values retain upstream `AsCat` truncation. Signed zero maps to
category zero. A reported GPU error leaves the caller's output vector unchanged.

`split_category_dispatches` counts completed GPU commands and
`split_category_nodes` counts requested bitset slots, including commands that
complete with a reported validation error. Timing joins `split_gpu_time_ns`.
Empty requests dispatch nothing. CPU-only builds expose zero counters, and the
statistics ABI change requires rebuilding native API executables.

Native checks pass **864 category slots / 33,738,112 exact words / 16 successful
production API calls**, including deliberate malformed-input cases. They cover
scalar/vector layouts, 1/3/33 targets, 1–513 feature bins, shuffled requests and
category orders, both partition directions, empty/invalid candidates, reused
dirty buffers, output guards, unchanged input arrays, word boundaries and
category `2^24 - 1`.
Production API tests mutate returned host orders/results and still obtain the
original GPU bitsets, check the no-readback path, validate stale/range errors and
recover after GPU-reported errors. Twenty-four public tests audit complete trees,
check dispatch coverage, and compare predictions from a CPU-loaded model. They
cover scalar/vector trees, both growth policies, both categorical split modes,
missing/zero-weight rows and sparse category IDs through 4,095. The maximum valid
ID is tested directly at the kernel layer: current public cut construction emits
every ID through the maximum, so that ID would require `2^24` training bins.
Final validation passes **1,545 Python tests (one skipped)**, every rebuilt native
suite and the CPU-only build/API check.
[Validation record](../validation/split-categories-m3.txt).

Interaction masks and their CUDA algorithm alignment were the next gap, addressed
below. Monotonic-bound initialization, generic metadata/category recoding, sketch
ingestion, persistent ownership and other ledger gaps remain. This operation is a separate synchronous command; no
performance benchmark or NVIDIA hardware comparison was run.


## CUDA feature-interaction constraints

Source authority: `tree/constraints.cu`, used by both scalar and vector CUDA
updaters. `Configure` constructs group-to-feature and feature-to-group CSR metadata
on the host; `Reset`, `RestoreFeatureListFromSetsKernel`,
`InteractionConstraintSplitKernel`, query-buffer clearing and `Query`/`QueryNode`
execute on GPU. The Metal implementation now covers these operations:

- GPU-owned node bitsets, with GPU zeroing, copying on growth and reset per tree.
- GPU union of every group containing the selected split feature. The parent
  retains this union; both children OR in the parent and the split feature itself.
- GPU input-feature bitsets, mask intersection, exclusive prefix scan and scatter
  to return ascending unique IDs for nonroot queries. Root or disabled queries
  preserve input ordering and duplicates; `QueryNode` has no root bypass.
- Empty enabled groups permit split-history features only. Disabled constraints
  bypass operations. Duplicate members do not duplicate CSR group memberships.

The private representation uses 32-bit words for native Metal atomics rather than
CUDA's 64-bit words, and grows lazily instead of preallocating `MaxNodes`. Both
adaptations preserve logical sets and query ordering. There is no CPU shadow of
allowed-feature masks. Host metadata parsing, sampled-list upload and completed
feature-list readback remain. Commands currently synchronize individually.

**Algorithm correction:** CUDA accumulates group unions over split history; the
CPU class instead selects groups containing the entire history. For
`[[0,1],[1,2]]` after splits 0 then 1, CUDA allows 2 and CPU does not. The previous
Metal path and split-reference filtering used the CPU rule. Both now follow CUDA;
explicit overlap tests prevent regressions, while CPU-loaded prediction checks
remain applicable. Existing disjoint-group tests alone could not expose this.

`interaction_reset_dispatches`, `interaction_split_dispatches` and
`interaction_query_dispatches` count completed commands, not individual kernels.
`interaction_buffer_growths` counts completed commands that enlarge mask storage;
`interaction_gpu_time_ns` records command GPU time. GPU validation-error commands
are included; invalid queries leave persistent masks unchanged. Disabled, root
bypass and zero-feature queries do not dispatch. CPU-only counters are zero.

Focused validation passes **834 production API checks** with Metal API Validation
on: independent host set unions; comparison to the distinct CPU constraint class;
root/disabled behavior; unlisted, empty, duplicate and overlapping groups; word
boundaries, groups longer than 256, three-level prefix scanning at 65,537 features;
shuffled/duplicate/empty sampled lists; high node IDs, child OR/reuse, sibling
isolation, resets, input ownership and malformed input recovery.
**30 public cases** cover scalar `hist`/`approx` and vector `hist`, numeric/one-hot/
partition splits, both growth policies, repeated trees and model continuation with
changed groups. Complete CUDA-source split oracles, exact operation counters and
CPU-loaded predictions are checked. Final validation passes **1,575 Python tests
(one skipped)**, every rebuilt native suite and the CPU-only build/API check.
[Validation record](../validation/interaction-constraints-m3.txt).

Monotonic-bound allocation/initialization was the next gap, covered below. Generic
metadata, sketch and other algorithm gaps, persistent GPU ownership and external
NVIDIA validation remain.
No benchmark or NVIDIA hardware comparison was run.


## GPU-owned monotonic bounds

CUDA `TreeEvaluator` in `tree/split_evaluator.h` initializes 256 nodes per target
with finite bounds, then grows to `(2 * max_child + 1) * targets`. Its device
`HostDeviceVector::Resize` uses `DeviceUVector::resize`: allocate, device-copy the
retained prefix and `thrust::fill` only the new tail. This was host vector work
in the prior MPS branch, even though subsequent bound updates already used Metal.

`SplitBoundsStorage` now owns persistent Metal buffers. A blit copies the retained
values on growth, and `xgb_init_split_bounds` writes the exact finite FP32 bound
bits into the new tail. Existing scalar/vector `xgb_update_split_bounds` executes
directly on those buffers. Its arithmetic and validation are shared with the
existing host-span API through a common encoder; production bound updates no
longer borrow/stage host bounds or copy them back. Weight inputs are still staged
and snapshotted before updates, including aliases.

Shared read views become visible after synchronous completion and remain valid
until growth or destruction. The histogram evaluator reacquires views after
`AddSplit`, as it did after host-vector growth. CPU/CUDA branches keep their
existing storage. Unconstrained/all-zero constraints allocate no bound buffers.
Feature-constraint parsing and padding stay host operations, matching CUDA setup.
The API grows without shrinking, validates extents before GPU work and limits
logical counts to 32-bit shader dispatch indices and the device allocation limit.

`split_bounds_init_dispatches` counts completed initialization/growth commands,
one for the pair of buffers. Timing joins `split_gpu_time_ns`; bound-update
commands retain `split_bounds_dispatches`. Empty/no-growth operations do not
dispatch. CPU-only builds expose zero. The statistics ABI and evaluator storage
changed, so every native executable is rebuilt for the full validation.

Focused native checks pass **24 cases / 76 updates / 3,847,176 exact values**.
Production storage checks include zero/unchanged growth, retained shared views,
size boundaries through 65,537 values, finite initial/tail bits, aliasing and
invalid-input recovery. Production `TreeEvaluator` comparisons use the shared
CPU/CUDA bound formulas, with 1/2/33/129 targets, scalar/vector midpoint rules,
subnormals, signs and no-op constraints. Node IDs 127/128, 255/256 and 1023/1024
force repeated growth and verify every retained and new bound.
Nineteen public cases include six new deep scalar `hist`/`approx` and vector
`hist` cases under both growth policies. Trees exceed 256 nodes, reuse histogram
cache capacity one and undergo complete CUDA-source tree audits, exact dispatch
checks, monotonicity checks and CPU-loaded prediction comparisons.
Final validation passes **1,581 Python tests (one skipped)**, every rebuilt native
suite and the CPU-only build/API check. [Validation record](../validation/bound-state-m3.txt).

Host candidate preparation still reads/stages bounds; full device ownership of
the training pipeline, generic state/model operations, sketch/CUDA algorithm
alignment, AFT corrections, interoperability, collectives and actual NVIDIA
validation remain. No benchmark or NVIDIA execution comparison was run.
