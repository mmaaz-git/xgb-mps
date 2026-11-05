# CUDA → Metal coverage

The project goal is behavioral coverage of **all XGBoost operations accelerated by
CUDA**, with measured performance. Passing training tests alone does not establish
that those operations execute on the GPU. Metal compute is the implementation
technology; `mps` is the public device selector. Apple's Metal Performance Shaders
library is not a replacement for XGBoost's custom CUDA kernels.

Current architecture: CPU orchestrates training and still owns much of the generic
state; selected stages and native inputs/metadata own Metal buffers. Remaining
CPU execution is explicitly listed below. This is not yet a complete GPU backend.

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
  use Thrust's distinct shuffle stream. Mean ranking now uses independent Philox
  integer draws and CUDA pair order; gradient/cost rounding and combined weight
  arithmetic now follow the CUDA formulas with explicit deterministic edge policies below.
- Quantile sketches: ordinary MPS matrices now use CUDA-derived raw sorting,
  FP32 scans and resident summaries, as detailed below. Explicit MPS
  QuantileDMatrix and both ExtMemQuantileDMatrix cache modes use the GPU adapter
  frontend. Optional Snappy compute cache compression is now implemented under
  the existing native opt-in. Native Metal DLPack feature tensors and columnar tables now feed
  in-memory/external quantile construction, ordinary matrices and inplace
  prediction. Native metadata and custom gradients are implemented below;
  distributed sketches now use ordered GPU merge/prune reductions over shared
  Metal payloads, including empty workers and quantile constructors. Scan association,
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
| Squared/logistic/squared-log/gamma/pseudo-Huber/Poisson/Tweedie/hinge/multiclass/expectile gradients | `objective/regression_obj.cu`, `hinge.cu`, `multiclass_obj.cu` | Metal kernels with guarded synchronous host-memory sharing where eligible; means/weighted means, multiclass packed class priors and normalization/log/centering, expectile initial gradients/Hessians and Newton stump fitting use Metal; regression and multiclass objective label predicates now use Metal; native metadata import validation also runs on Metal. These gradient outputs now retain Metal vector storage; validation and objective input/output spans avoid implicit host mirrors. Initialization/statistics/stump callers now also retain Metal inputs/outputs through GPU stages; other consumers still need migration |
| Sigmoid/exp/hinge/softmax/argmax/expectile prediction transforms | Same objective files | Metal kernels; multiclass softmax follows source sequential binary64 accumulation of FP32 terms and argmax handles source NaN/subnormal comparisons; identity transforms require no dispatch. Nonranking objective transforms now retain explicit Metal vector storage |
| Base-score validation and margin conversion | `objective/regression_obj.cu` | Metal checked logistic/gamma intercepts, clamped FP32 logit and logarithms for gamma/Poisson/Tweedie/Cox. Identity mappings require no dispatch; checked results publish by GPU blit only on success and retain Metal vector storage; expectile's host margin loop matches upstream host execution |
| Absolute-error and quantile objectives, associated statistics | `objective/regression_obj.cu`, `quantile_obj.cu` | Metal source-expression gradients, FP32 residual-root sums, binary64 optional-weight sums and quantile prediction sorting; upstream host scale-statistic collective restored for empty/nonempty workers; absolute-error initial weight sums, means, target-mean broadcast and stump fitting, and initial quantile estimation/vector normalization use Metal; gradient payloads now retain Metal vector storage. Initialization means/broadcasts/stumps/quantiles now retain Metal storage; source-defined expectile/absolute host finalizers remain. Other generic state work still needs migration |
| Ranking losses, pair construction, normalization | `objective/lambdarank_obj.cu` | Prediction order and NDCG ideal-label order use stable grouped Metal sorting; query-weight sums use Metal. Predictions, labels, query weights, gradients, NDCG caches and unbiased-learning arrays now retain explicit Metal storage through objective calls. Top-k and CUDA-source Philox mean pairs, raw gradient/cost bounds, independent sign truncation, actual-label segmented MAP prefixes, query normalization and combined binary64 weight multipliers execute on Metal; the global cost-factor race and nonfinite frexp inputs have explicit deterministic policies. Unbiased loss corrections, per-row bias costs, cross-query aggregation and regularized bias updates execute on GPU. Source bias initialization/reload rules are preserved; serialization explicitly reads the small bias arrays |
| AFT survival loss | `objective/aft_obj.cu`, `common/survival_util.h` | Metal normal/logistic/extreme gradients for every row and prediction/base-score transforms, now with explicit Metal input/output storage; CPU corrections removed. Actual upstream expression replay covers sensitive tails; cross-library output differences remain, as detailed below |
| Base score estimation, leaf value fitting | `tree/fit_stump.cu`, `common/stats.cu`, `tree/gpu_hist/leaf_sum.cu` | Means/weighted means, quantiles/medians, Newton stump gradient aggregation and unregularized weights on Metal; final vector-leaf refitting quantizes sampled float value gradients with original scales, uses GPU segmented integer sums and cross-page integer accumulation, then regularized/clipped weights; final tree updates stay host as in CUDA. Means, medians/quantiles and stump wrappers now consume Metal spans and publish strided outputs/empty fills on GPU. Other state work still requires porting/audit |
| Batch/inplace/leaf tree traversal | `predictor/gpu_predictor.cu`, `predictor/predictor.cu` | Metal: scalar/vector leaves, category bitsets, missing values, tree ranges, dropout weights; GPU input decoding, missing checks, bin lower bounds and category-map application. Initialization, base margins, prediction caches and outputs now retain Metal storage, including learner copies; scalar fills/vector broadcasts reuse existing kernels. Tree node/category/leaf arrays now use persistent device mirrors; per-call traversal packing runs on Metal from those arrays, while per-tree metadata stays on host. Host-input packing remains on CPU; native dense/columnar snapshots use GPU copies; public result APIs retain existing Metal arrays without an extra host mirror; native outputs snapshot them on GPU and NumPy explicitly copies |
| Prediction cache from leaf IDs | `predictor/gpu_predictor.cu` | Metal IDs and output arrays, guarded input bindings and alias snapshots; supports strided target slices and negated sampling positions. GBTree retains Metal prediction storage through cache updates; hist/approx gradient and leaf-position producers now retain Metal arrays |
| SHAP contributions and interactions | `predictor/gpu_predictor.cu`, `predictor/interpretability/shap.cu` | Metal CUDA-source leaf weighting, branch-weighted basis, FP32 edge/pair extraction and 4/2/1 eight-point sums; repeated-feature paths, scalar/vector leaves, categorical/missing decisions, dropout weights, bias addition, symmetry and diagonal finalization; model compression/root means remain host as in CUDA. Outputs/base margins now retain Metal storage; raw feature inputs bind guarded shared memory with GPU alias snapshots. Input decoding/category mapping runs on GPU; host-input packing remains on CPU; native dense/columnar snapshots use GPU copies |
| RMSE/RMSLE/MAE/MAPE/logloss/error/Poisson/gamma/Tweedie/pseudo-Huber metrics | `metric/elementwise_metric.cu` | CUDA-source FP32 weighted terms and complete binary64 GPU reduction; explicit Metal input storage and guarded payload bindings; final ratio/collective transport host |
| Multiclass error and logloss | `metric/multiclass_metric.cu` | CUDA-source FP32 weighted terms and complete binary64 GPU reduction; explicit Metal input storage and guarded payload bindings; final ratio/collective transport host |
| Quantile and expectile metrics | `metric/elementwise_metric.cu` | CUDA-source FP32 terms, subnormal-preserving products and complete binary64 GPU reductions; explicit Metal input storage and guarded payload bindings; final ratio/collective transport host |
| AFT likelihood and interval-regression accuracy | `metric/survival_metric.cu` | GPU probability calculations and complete binary64 sums; explicit Metal input storage and guarded payload bindings; final ratio/collective transport host |
| ROC-AUC | `metric/auc.cu` | Binary, multiclass and multilabel CUDA-source term casts, batched sorting, binary64 prefixes and tied-score trapezoids; raw curve outputs, per-output aggregation metadata and normalization on GPU; resident inputs/summaries; host-summary distributed collective fallback |
| PR-AUC | `metric/auc.cu` | Binary, multiclass and multilabel CUDA-source term casts, separate FP32 validity/recall totals, binary64 source integral and distinct binary/multi-output invalid results; GPU metadata and normalization; resident inputs/summaries; host-summary distributed collective fallback |
| Ranking metrics | `metric/rank_metric.cu`, `metric/auc.cu` | NDCG/MAP/precision use CUDA-source arithmetic/fallbacks, grouped sorting, actual-label MAP prefix scans and query-sum reduction in one Metal command. NDCG discounts/inverse ideal gains, cached sort indices and ranking label validation retain Metal storage; metric inputs bind shared arrays directly. Grouped ROC uses GPU stable label order, merge-based pair counts, query normalization and final aggregation. Grouped PR uses CUDA-source FP32 cast boundaries, original-order positive totals, binary64 prefixes/integrals, two-row validity and GPU aggregation |
| Quantile sketching, cuts, weighted/category handling | `common/hist_util.cu/.cuh`, `common/quantile.cu`, `data/gradient_index.cu` | Ordinary CSR matrices use Metal sorting, FP32 weight scans, category deduplication, resident pruning/merging and rank repair. Final cut materialization is host-side in both CUDA and Metal. Explicit MPS in-memory and external quantile matrices add host dense/columnar and native Metal tensor/columnar ingestion, resident sketches and bins; native metadata conversion/validation is supported; distributed sketches use CUDA-order variable reductions with GPU payloads, including empty workers and all these constructors |
| Compressed GPU matrix, external memory, iterators | `data/ellpack_page.cu/.cuh`, `ellpack_page_source.cu`, `iterative_dmatrix.cu`, `quantile_dmatrix.cu` | External host cache adds direct adapter-to-ELLPACK GPU construction, concatenation, adaptive shared/private storage and two-page prefetch. Histograms, partitioning and prediction/SHAP preparation read compressed views directly. Host gradient-index/column mirrors are created only for CPU access; prefetch stays compressed. In-memory quantile matrices now use direct compressed construction or one-time compression of CPU bins, with separate lazy CPU views. Ordinary in-memory DMatrix now compresses raw CSR directly with separate lazy CPU views and cache-generation tracking. MPS-origin external disk caches now store/read compressed pages; internal quantile cuts+ELLPACK save/load is implemented. Ordinary external on_host selection now uses CUDA single-/multiple-page defaults. Optional Snappy compute compression/decompression now follows the native fallback opt-in, sparse eligibility and forced-prefetch rules. Mapped-file zero-copy remains open; CPU-origin external matrices retain their existing compatibility path |
| Categorical recoding, metadata manipulation, validation | `data/cat_container.cu`, `encoder/ordinal.cuh`, `data/data.cu`, `metainfo.cu`, `array_interface.cu` | Category sorting/schema checks/lower-bound mapping/first-unknown search, row-code mapping, native dictionary gathers and metadata conversion/validation/query-ID grouping run on GPU. Native float metadata retains Metal vector storage; internal row gathers, device copies/stacking and feature-type equality run on Metal. Group boundaries and public ordinary row slicing remain host as upstream. Typed dictionaries now own reusable Metal snapshots, copies and sorted indices; native names are gathered on GPU; recoding mappings stay Metal-owned through prediction, SHAP and ingestion. Explicit CPU access/serialization lazily publishes host mirrors; host shape/type descriptors remain |
| Linear coordinate-descent booster | `linear/updater_gpu_coordinate.cu` | GPU FP32 bias/feature gradient reductions, CUDA residual order/filter rules and unchanged selector snapshots with `updater=coord_descent`; host selectors, scalar weight/model updates, CSC preparation and linear predictions match CUDA's placement; default `shotgun` remains host |
| Gradient layout copies and Hessian extraction | `gbm/gbtree.cu`, `tree/updater_gpu_hist.cu`, `tree/updater_quantile_hist.cc` | Metal bit-preserving target-column copies, sampling-buffer layout conversion, reduced-value-gradient copies and unsampled Hessian extraction; guarded host sharing or staged snapshots for aliases. Allocation and other state manipulation remain host |
| Vector model-buffer copies and learning-rate scaling | `tree/multi_target_tree_model.cu` | Metal blit copies for base/child weights and category storage, and GPU FP32 learning-rate products for new child weights. Host model allocation/growth and metadata remain; buffers use guarded synchronous host-memory sharing or staging |
| Device arrays, scans, sorts, reductions, random generation | `common/*.cu`, `common/*.cuh` | Histogram/metric/gradient reductions, generic float and optional-weight sums, checked exact scans, category/quantile-row sorting, stable column radix sorting, stable grouped ranking sorting, recursive integer/binary64 prefix scans and sampling-specific random generation implemented; other general Metal primitives missing |
| Distributed GPU collectives | `collective/comm.cu`, `collective/coll.cu`, `collective/allreduce_v.cuh`, `collective/aggregator.cuh` | Metal numeric/bytewise fixed reductions, broadcast and fixed/variable allgather over shared buffers and host TCP; 1/2/3/4-worker tests. Ordered variable-size reduction and sketch consumers are implemented (1/2/3/5-worker tests); global quantization and other training consumers remain open. Distributed `mps` training still rejected |
| CUDA array interface / CuPy / cuDF / CUDA streams | `c_api/c_api.cu`, `data/device_adapter.cu` | Metal DLPack tensors and native columnar tables feed ordinary/quantile constructors, iterators and inplace prediction/CPU fallback through owned GPU snapshots. Metadata imports use GPU conversion/validation/grouping; custom gradients use GPU typed pair packing. Explicit `output_device="mps"` returns owned GPU predictions, including leaves and SHAP, with legacy/versioned DLPack exports to PyTorch/MLX. Public prediction result pointers retain synchronized shared Metal storage; native output snapshots avoid intermediate host mirrors, while NumPy copies explicitly. Known runtimes synchronize; generic producers/consumers must finish writes. Portable stream handshakes and end-to-end device ownership remain open. NVIDIA-specific interfaces cannot consume Metal memory |
| CUDA contexts, allocators, stream lifetimes | `common/cuda_*`, `common/device_*`, `common/host_device_vector.cu` | Owned shared Metal vector allocations with CUDA-derived access states, GPU fills/copies/growth and lazy host mirrors; native float metadata uses them. Reused queues/histograms/workspaces and guarded host views remain. The execution context still defaults generic state to host storage; remaining consumers, persistent end-to-end ownership and asynchronous scheduling are unfinished |

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

## Remaining porting work

Training gradients, sampling buffers, histogram state, leaf positions, prediction
caches/results, category dictionaries/mappings and prediction tree mirrors now
have explicit Metal storage paths. Source-defined host model construction,
serialization and SHAP compression remain on CPU. The detailed records below
show which consumers and edge cases have actually been exercised.

The remaining work includes a complete CUDA operation/arithmetic audit and GPU
collective integration. SHAP leaf weighting, basis recurrence, edge/pair expressions and eight-point
reduction association now follow the CUDA source, as recorded below; NVIDIA
compiler behavior and cross-tree atomic scheduling remain unverified. Fixed Metal
collectives now exist, but variable-size sketch reduction, global quantization and
histogram/root/leaf consumers still need integration. Federated MPS collectives
and asynchronous streams remain open. CUDA/NVIDIA execution comparison and testing on other Apple hardware remain
separate validation requirements; local passing tests cannot establish them.

Further performance work includes reducing synchronization and improving external
memory staging. It follows coverage/correctness work. Historical benchmark gains
do not establish full CUDA parity, and this publication change has no new timing
claim.

## Legacy CPU-formula gradient sampler

The earlier low-level `metal::GradientSample` API follows the CPU sampler, including
sequential FP32 target/prefix accumulation, Apple libc++ draws and CPU threshold
fallbacks. Its exact CPU-reference tests are retained as regression coverage.
**Production MPS training no longer calls this API.** It uses the
[CUDA-derived shared-state sampler](#cuda-derived-gradient-based-sampling).

Historical performance measurements of the ordered FP32 prefix and its checkpoint
optimization describe that earlier implementation. They have not been rerun for
the integer sampler and are not performance claims for the current path.

## Regression, hinge and expectile arithmetic

The generic objective kernel now evaluates the source FP32 expressions through
the shared preserving arithmetic helpers. This covers squared error,
logistic/logitraw, squared log, gamma, pseudo-Huber, Poisson, Tweedie and expectile.
Source multiplication and accumulation boundaries are explicit. Logistic keeps
`std::min` semantics for the exponent cap; loss floors use CUDA `fmaxf` semantics.
Expectile retains the source cumulative softplus margins, per-target gradient
and Hessian sums, ordered residual comparison and final sigmoid factors.

Squared-log Hessians use an FP32 numerator, a binary64 denominator and binary64
division before narrowing to FP32 and applying the floor. The source's
`std::pow(float, int)` promotes to double under the configured C++17 standard.
This was confirmed in NVIDIA 12.8.93 headers' modern GCC/libc++ overloads and GCC's
promotion traits. The square of an FP32 value is exactly representable in
binary64; the port uses that exact product. Actual NVIDIA compiler lowering and
libdevice results are not asserted. The numerator now spells the source order;
its isolated rewrite produces zero corpus differences, whereas retaining the
wider denominator changes 25,205 checked Hessians.

Hinge follows the source's double literals: label mapping, margin comparison and
weighted gradient are binary64 before the gradient narrows to FP32. The satisfied
margin branch retains the unweighted minimum-normal FP32 Hessian, including
unordered NaN comparisons. There are 6,089 witnesses against the previous FP32
calculation. Other objectives retain their respective FP32 formulas.

Sigmoid, exponential, hinge, cumulative expectile and quantile prediction
transforms now also preserve subnormals. Quantile retains the source insertion
sort, including strict comparisons, equal-value order and NaN barriers; it does
not replace that loop with a differently ordered sort. Identity and empty
operations remain dispatch-free. Existing guarded input sharing, alias-safe
output staging and counters are unchanged.

`tests/build_objective_reference.py` extracts actual scalar methods and device
lambda bodies from five vendored source files. The generated reference changes
view bindings and math callbacks, retaining the expressions, casts and branches.
Math callbacks round host binary64 exp/log1p/sqrt to FP32, evaluate pow in
binary64 and apply documented CUDA fmaxf NaN rules. A separate reference keeps
native host libm, so those differences are visible. Apple `std::fmax` propagates
a signaling NaN in a reproduced case, while CUDA documents choosing the numeric
argument. This is why native host fmax is not the CUDA edge-case oracle.
[NVIDIA fmaxf contract](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html).

The separate host-libm replay differs in 6,893 gradient/Hessian fields. Twelve
classification differences and eight finite tolerance failures come from
signaling-NaN clamps; another 27 finite tolerance failures come from cancellation
with weights of `1e20` or `1e30`. For example, a Poisson gradient can be zero with
native host expf and approximately `-5.96e22` with the explicitly rounded math
oracle after a `1e30` weight. These real cross-library differences remain
explicit; the port does not force CPU agreement by correcting rows on the host.
The detailed validation record reports the full diagnostics.

Native checks pass 624 cases / 14,305,782 result fields, including 545,836
subnormal and 236,211 nonfinite fields. They cover weighted/unweighted and
multi-target rows, all-bit random inputs, singularities, cancellation, slope/rho
boundaries, aliasing, guards and empty calls. NaNs compare by classification,
signed zero by equality; identity/quantile copies additionally preserve every
bit. Fifteen public transform cases cover CPU comparisons, raw margins,
dispatches and model reload. Existing mathematical and training comparison
tolerances are unchanged. NVIDIA/libdevice/NVCC contraction and cross-Apple
execution remain unverified; performance has not been benchmarked. Full
regression passes **1,856 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/device check (2026-11-04, 14:59:14–15:03:58 UTC).
[Validation record](../validation/cuda-objective-arithmetic-m3.txt).

## Explicit Metal QuantileDMatrix construction

`QuantileDMatrix(..., device="mps")` enables GPU construction from host dense and
columnar arrays, including DataIter batches and references. C callers use the
optional JSON field `device`. Omitting it retains upstream input-device inference;
the sklearn wrapper also retains that behavior. CPU-only builds accept explicit
CPU construction and reject MPS with the build requirement.

CUDA authority: AdapterDeviceSketch/MakeEntriesFromAdapter, MakeSketches,
InitFromCUDA, GetRowCounts, the encoded columnar adapter and SketchContainer::Push.
Windows count 2^26 raw slots, including missing entries, and can split a row.
Pruning uses ceil(window slots / features). All-missing windows advance rows_seen
before returning; the independent native oracle was corrected to retain this
source order. Training recodes category values before missing filtering.

GPU count/scan/compaction produces raw records and row pointers consumed directly
by resident sketches and binning. Group weights use the existing GPU expansion;
weights change per callback batch. Cut materialization remains host-side, as in
CUDA. Infinity validation occurs at bin/page construction after sketching, matching
CUDA's phase; an infinity missing sentinel is valid. The second iterator pass
assigns bins and global row offsets on GPU, including partial boundary rows.
There is no raw-value readback between decoding and sketching/bin assignment.

Final quantized bins and row pointers return to existing host gradient-index
storage. A host column mirror supports CPU interoperability, as CUDA's conversion
to a host gradient-index page does. Raw packing/uploads, metadata aggregation,
dictionaries and ownership remain host work, with synchronous commands. Native
Metal tensor handles and raw float16 are implemented in later sections below;
GPU collectives remain open. External disk and compressed host caches are covered below.
No speedup is claimed.

65 native cases / 2,184 windows / 1,126,724 exact fields check resident summaries
against independent CUDA-source formulas after every push, plus raw values, row
maxima, compressed bins and global offsets. 58 public tests cover all ten numeric
types, masked/categorical/strided columns, empty/streamed batches, row/group/zero
weights, reference cuts, code remapping, prediction, infinity and changing-input
errors. A broadcast fixture crosses the actual 67-million-slot boundary. Focused
regression passes 140 tests. Full regression passed **2,004 Python tests, one
skip**, all rebuilt native suites and the CPU-only build/check
(2026-11-04, 16:27:53–16:34:12 UTC).

Counters: adapter_prepare_dispatches (count/scan plus nonempty compaction),
adapter_index_dispatches (global row-pointer output), adapter_gpu_time_ns and
adapter_debug_readback_bytes (raw input diagnostics only). Binning retains its
separate counters. Production leaves raw diagnostic readback unchanged. CPU-only
values are zero; device time excludes packing, transfers and host storage.
NVIDIA execution and cross-Apple validation remain unperformed.
[Validation](../validation/cuda-adapter-construction-m3.txt).

## Metal disk-backed external quantile construction

`ExtMemQuantileDMatrix(iterator, device="mps")`, with `DataIter(on_host=False)`,
constructs quantiles from dense/columnar host batches using Metal. Default input
placement is unchanged. CUDA authority is `quantile_dmatrix.cu::MakeSketches`,
`extmem_quantile_dmatrix.cu::InitFromCUDA` and
`ellpack_page_source.cu::ExtEllpackPageSourceImpl::Fetch`, especially the disk-cache
branch that retains one page per input batch.

In-memory and external construction share the same adapter windows, GPU shape
counts, weighted resident sketches and reference cuts. The second external pass
allocates one gradient-index page, assigns its bins and local row pointers on GPU,
builds the host column mirror from those bins, and writes the existing disk format.
No full-dataset gradient-index allocation is made. Metadata remains aggregated on
the host. Completed cache reads use the existing prefetch/mmap machinery and do
not call the raw iterator or resketch. Row counts, non-missing counts and number of
batches are checked across passes. Destruction removes the cache; failed
construction also releases readers and removes partial cache files.

Multi-page tests exposed two host conversion helpers that applied global row bases
to page-local storage. Dense column conversion and sparse feature recovery now use
local offsets. Native checks verify both helpers on pages with nonzero global
bases, including dense, sparse, empty and all-missing pages.

48 native cases make 864 cached-page visits and 2,720,419 exact field checks.
CPU quantization with the same reference cuts independently supplies expected bins,
row pointers, columns/missing flags and hit counts. Fixtures cover 8/256/257 bins,
weighted/unweighted rows, numerical/categorical data, interrupted cache traversal,
input independence and file cleanup. The 43 public cases verify CUDA-source cut
formulas, all ten numeric types, masked/reversed arrays, row/group/zero weights,
CPU/MPS/external references, category recoding, metadata, vector trees, CPU training
from cached columns, same-model CPU prediction and failed-iterator cleanup.
Upstream does not implement `get_data()` for ExtMemQuantileDMatrix; exact page
inspection is performed by the native tests, without adding a new public API.

This checkpoint completed the disk-backed adapter construction path. The subsequent
host-cache step below adds `on_host=True`, adaptive shared/private ratios,
concatenation and GPU compressed pages. Native Metal arrays and GPU collectives
remain open. Raw packing, dictionaries, metadata and disk I/O remain host work.
Existing adapter/sketch/binning counters apply; this disk checkpoint made no statistics ABI change.
No new speedup, NVIDIA-equivalence or cross-Apple claim is made.
Full regression passed **2,047 Python tests, one skip**, every rebuilt native
suite and the CPU-only build/API checks (2026-11-04, 16:48:37–16:54:33 UTC).
[Validation](../validation/cuda-extmem-construction-m3.txt).

## Metal compressed external-memory host cache

`ExtMemQuantileDMatrix(iterator, device="mps")` now supports the default
`DataIter(on_host=True)`. `min_cache_page_bytes` controls concatenation and
`cache_host_ratio` divides each compressed page between Metal shared and private
buffers. No disk file is created. `on_host=False` retains the disk implementation
above and rejects an explicit ratio. Default construction-device inference is unchanged.

CUDA authority is `ellpack_page.cu::CalcNumSymbols/Copy`,
`compressed_iterator.h::CompressedBufferWriter`,
`ellpack_page_source.cu::CalcCacheMapping/EllpackHostCacheStreamImpl` and
`batch_utils.cc::DftPageSizeHostRatio`. Metal preserves the eight-byte prefix,
MSB-first variable-width symbols, dense/relatively-dense/sparse symbol rules,
missing padding and concatenation across non-byte-aligned row boundaries. GPU
kernels pack and copy symbols; restored row counts, exact prefixes, bins and hit
counts also run on GPU. Reading split buffers handles symbols crossing the split.

The cache plan retains CUDA's padded input-byte sums, page grouping, ratio
endpoints and one-page override. By default the minimum page size is one sixteenth
of the device budget; half the budget is available to the cache. Auto train ratios
are zero when the cache fits, otherwise `(total - available * 0.85) / total`;
validation defaults to one. A single resulting cache page forces ratio zero,
including with an explicit ratio. Metal uses
[`recommendedMaxWorkingSetSize`](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
as its budget. Apple's shared and private storage both use unified memory;
private means GPU-only access, not separate VRAM. The NVIDIA C2C link-count branch
has no Apple equivalent. [Apple storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus).

The existing asynchronous page-source machinery retains one active and one
prefetched decoded page. `prefetch_copy` joins split storage on GPU before
restoration; a fully private page needs no join. Input callbacks are released
after construction. The later consumer step below retains compressed GPU views
in training pages, avoiding restored GPU indices and their dense transpose.
Constructor and interrupted-iteration checks cover
both cache modes. CPU-only builds reject MPS construction explicitly.

This does not complete compressed matrix ownership. The subsequent direct
construction step below removes temporary host quantized bins from this path.
At that checkpoint restoration materialized host gradient-index/column mirrors.
The subsequent consumer/ownership steps below read compressed symbols and create
host mirrors only for CPU access. Open areas include disk-backed compressed storage,
CUDA's optional Snappy/nvcomp compression/decompression, native
Metal array/half interfaces, metadata ownership and GPU collectives. Shared/private
cache bytes do not measure whole-process memory savings. No speedup is claimed.

Native format checks pass 257 cases / 82,394 exact bytes / 55,438,557 exact fields,
comparing actual bytes with the upstream scalar bit writer for all widths 1–32,
split offsets, empty pages and unaligned concatenation. Restored
indices, hits, rows and columns are checked exactly, along with retained GPU
buffers and 80 deterministic cache-plan combinations. The external-source suite
now covers 192 cases / 2,322 cached-page visits / 10,815,835 exact fields across
disk and host caches. Ninety-six new public cases check ratios, grouping, CUDA
source cuts, CPU shared-cut predictions, references/category recoding, metadata,
iterator changes, cleanup and absence of repeat input work. The focused public
regression passes 197 cases. Full regression passes **2,143 Python tests, one
skip**, every rebuilt native suite and the CPU-only build/API checks
(2026-11-04, 17:18:34–17:24:16 UTC).
[Validation](../validation/cuda-ellpack-cache-m3.txt).

Eight statistics fields were appended: `ellpack_pack_dispatches` (also symbol
maxima), `ellpack_copy_dispatches`, `ellpack_restore_dispatches`,
`ellpack_cache_dispatches` (buffer blits), `ellpack_gpu_time_ns`,
`ellpack_debug_readback_bytes`, `ellpack_cached_host_bytes` and
`ellpack_cached_device_bytes`. Cache-byte counters are cumulative committed
logical bytes, not current allocation gauges. Production performs no diagnostic
compressed-byte readback. GPU time excludes host staging, metadata and column
mirrors; dense transpose retains its existing counter. Native callers must rebuild
for the extended Statistics layout; the full validation does so. CPU-only values
remain zero. NVIDIA execution and cross-Apple validation remain unperformed.

## Direct adapter-to-compressed construction

External host-cache construction now passes each resident `AdapterInput` directly
to its compressed page. It no longer creates a temporary host gradient-index or
column mirror and does not upload host bins for packing. Disk construction and
ordinary/in-memory matrices retained their existing representations at that
checkpoint; the later in-memory step below extends compressed ownership.

CUDA authority is `ellpack_page.cu::WriteCompressedEllpackFunctor`,
`CopyDataToEllpack`, `WriteNullValues` and the adapter page constructor. Numeric
upper bounds, categorical integer conversion/lower bounds and final-bin clamps
use the same GPU search helper as existing bin assignment. Dense and relatively
dense pages write feature-local symbols at their original feature positions,
including explicit missing symbols for the latter. Sparse pages write global
symbols in compacted feature order. Per-row GPU-private cursors accumulate across
partial-row windows; a final GPU pass fills trailing nulls, as CUDA does.

The existing raw window size remains 2^26 slots, including missing values. Windows
must be contiguous and ordered; completion checks their full range. Row width,
feature count, missing counts, row counts and batch counts are validated; infinity
and features without cut points fail clearly. Incomplete construction cannot be
committed, restored or copied. GPU metadata/error readback remains small; raw
values, intermediate quantized bins and row cursors do not return to host during
this construction. Cut arrays and feature-type descriptors still upload once per
input page. Host input packing, metadata aggregation and synchronous dispatch remain.

The native test independently searches cuts from scalar raw values and feeds the
upstream scalar bit writer. It passes 73 cases / 2,933 windows / 81,666 exact bytes /
7,455,890 fields and 12 expected errors. It covers one-slot windows, sparse rows
split repeatedly, all-missing and empty inputs, dense/relative/sparse layouts,
float/double input, categorical fractional codes, subnormals, signed zero, 8/16/32-bit
restored indices, recursive row-prefix sizes and the empty-cut domain. Restored
global bins, row pointers and hit counts are exact. Public tests retain all prior
cache/reference/category cases and add ten numeric input types with reversed
strides, a real sparse 2^26-slot seam, infinity sentinels and changed maximum row
width with unchanged total nnz. Focused construction regression passes 210 tests.

Three appended counters distinguish direct work from staging:
`ellpack_adapter_dispatches` counts direct writes and sparse null completion;
`ellpack_adapter_values` counts successfully binned non-missing values;
`ellpack_pack_input_bytes` counts host index/row-pointer bytes uploaded by the older
`Pack(GHistIndexMatrix)` conversion. The host cache requires the last counter,
`adapter_index_dispatches`, `binning_dispatches` and diagnostic readback counters
to remain unchanged during construction. The former GPU binning counter applies
to the host-index path; fused compressed binning uses the new counter. All native
callers rebuild for the appended Statistics fields; CPU-only values remain zero.

At that checkpoint restoration still created host mirrors using temporary
uncompressed GPU indices. The subsequent steps below retain compressed GPU views
and create host mirrors only on CPU access; broader ownership remains unfinished.
No new benchmark, NVIDIA execution comparison or cross-Apple validation was run.
Full regression passes **2,156 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/API checks (2026-11-04, 17:34:35–17:40:21 UTC).
[Validation record](../validation/cuda-direct-ellpack-m3.txt).

## Compressed GPU consumers

External host-cache pages now retain a compressed GPU view: its head/tail buffers,
split byte offset, row stride, symbol width/count, density, feature count and cuts.
The view owns its buffers independently of the cache object. Prefetch may join
split storage, while direct reads handle symbols crossing the shared/private split.
Restored GPU index/row-pointer arrays and the dense transpose are no longer retained.

CUDA authority is `ellpack_page.cuh::EllpackAccessorImpl::GetBinIndex/NullValue`,
the histogram and partition access in `tree/gpu_hist`, and
`predictor/gpu_data_accessor.cuh::EllpackLoader::GetElement`. Dense/relatively dense
lookup reads a local bin at the feature position; sparse lookup searches global
bins within the padded row. Missing symbols return the missing-bin sentinel.
Prediction returns category cut values, negative infinity for a numeric first
bin, and the preceding cut otherwise. Existing category mapping runs on GPU.

Global and shared-memory histograms, including the one/two-bin SIMD path, read
compressed symbols directly. Modular integer sums, feature offsets and row/node
selection retain the existing algorithm. Row partition classification uses the
compressed accessor, followed by the unchanged stable prefix/scatter pipeline.
Both paths share arithmetic bodies with legacy typed-index entry points through
template specialization; legacy kernel interfaces remain unchanged. Prediction
and SHAP prepare their GPU-owned input chunks directly from the compressed view,
without packing the host index mirror or re-uploading bin bytes.

Native checks pass 139 cases / 28,485,537 exact fields across 278 histograms,
1,946 partitions and 417 input chunks. After creating a page, they overwrite its
host bins, row pointers and cut values and destroy the source cache object.
Independent CPU bins/sums/stable partitions still agree exactly. Cases cover
dense/relative/sparse/all-missing/empty layouts, 2/17/257/65537 bins, four cache
ratios, both prefetch modes, nonzero page bases, categorical/missing decisions,
mapped prediction values and partial input chunks. Large cases exercise both
shared histogram branches and the non-serial partition prefix. Twelve public
cases check actual tree training, model reload, margins, leaf IDs, SHAP
contributions and interactions against the same CPU model. Existing SHAP tolerances
remain 2e-5; margins and leaves compare exactly.

Three appended counters record completed compressed consumer commands:
`ellpack_histogram_dispatches`, `ellpack_partition_dispatches` and
`ellpack_input_dispatches`. Empty inputs do not increment them. Existing histogram,
partition, shared-memory, input-preparation and category-map counters remain active.
Native tests require no additional bin upload, restoration or compressed-byte
readback during consumer calls. CPU-only values are zero; all native callers rebuild
for the extended Statistics layout.

At that checkpoint host mirrors were still built eagerly for CPU interoperability
and DMatrix contracts. Temporary GPU unpacking/readback remained; that milestone does
not claim fully device-owned matrices or less whole-process memory. Ordinary and
in-memory quantile matrices retained typed-index storage at that checkpoint. Native arrays, optional
Snappy cache compression and GPU collectives remain separate gaps. No performance,
NVIDIA execution or cross-Apple claim is made.
Full regression passes **2,168 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/API checks (2026-11-04, 17:52:21–17:58:13 UTC).
[Validation record](../validation/cuda-compressed-consumers-m3.txt).

## Host mirrors on demand for compressed caches

External host-cache prefetch now acquires a compressed GPU view without allocating
host row pointers, index bytes, hit counts or column data. `GHistIndexMatrix`
records its device-only row count separately. CPU-only methods guard against
absent mirrors; raw host arrays are documented as unavailable on these MPS batches.
The prediction view requests host columns only inside CPU `DoFill`, so GPU input
preparation never asks for them.

`MetalPageSource` always prefetches device-only views. On a CPU batch request, its
iterator thread explicitly materializes the current page before returning it.
The device choice and reset share the existing iterator lock; workers never read
a mutable CPU/MPS flag. Completed traversals can reuse prefetched futures across
device changes; interrupted traversals clear the ring as before. Materialization retains the exact
compressed view, is idempotent, and requires no host-bin re-upload. Retained views
own their buffers after the cache and DMatrix are destroyed.

Construction still validates actual compressed non-missing counts using the GPU
count/prefix kernels, reading only one scalar per completed page. Host bin/column
restoration is not needed for this check. Quantile-cut export and reference-cut
lookup request metadata without forcing host mirrors, even for a CPU child matrix.
This follows CUDA's separation of compressed consumers from CPU representations;
the fork's existing CPU interoperability for external MPS matrices is preserved.

Approximate SHAP remains an explicit CPU fallback, with a CPU context used for
batch access. CUDA rejects approximate contributions and interactions in
`predictor/interpretability/shap.cu`; they are not missing CUDA GPU kernels.
Exact SHAP, interactions, margins and leaf IDs continue through compressed GPU
input preparation. DART, multiclass and vector-leaf paths are covered.

Four appended counters expose view acquisition, count dispatches, completed host
materializations and host mirror readback bytes: `ellpack_view_pages`,
`ellpack_count_dispatches`, `ellpack_host_materializations` and
`ellpack_host_readback_bytes`. Mirror bytes count row pointers, typed index bytes
and 32-bit GPU hit counts copied to host; the construction scalar is separate.
The existing `ellpack_restore_dispatches` now increments only for explicit host
restoration. CPU-only builds report zero and all native callers rebuild.

Native validation passes 278 consumer cases / 72,017,790 exact fields, with both
poisoned eager mirrors and completely absent mirrors, followed by exact explicit
restoration and idempotence checks. The external-cache suite passes 192 cases /
2,322 CPU page visits / 10,845,265 exact fields, adding CPU/MPS switches,
interrupted traversals and GPU reads after DMatrix destruction. Focused public
regression passes 148 cases, including 27 new cases with GPU-only no-readback
assertions, CPU training/prediction, metadata-only access, approximate fallback,
model reload, multiclass, dropout and vector leaves. Existing numerical tolerances
are unchanged; unsupported vector-leaf `strict_shape` is not requested.

Ordinary/in-memory matrices retained typed GPU indices and eager host mirrors at
that checkpoint; in-memory quantile ownership is extended below.
Compressed disk-cache format and internal ELLPACK save/load, optional Snappy
compression, native arrays, general metadata/device state, and collectives remain
separate gaps. No new performance, whole-process memory, NVIDIA execution or
cross-Apple claim is made.
Full regression passes **2,195 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/API checks (2026-11-04, 18:11:39–18:17:33 UTC).
[Validation record](../validation/cuda-lazy-host-m3.txt).

## Compressed in-memory quantile matrices

Explicit `QuantileDMatrix(..., device="mps")` construction now writes each adapter
batch directly into ELLPACK, then copies it into the final GPU page. It preserves
batch/row/column/non-missing checks, single-batch metadata moves, weights, categories
and iterator reset. Actual compressed non-missing counts are validated on GPU.
No host row pointers, typed bins, hit counts or columns are allocated for its
initial GPU view. Cuts and shape metadata remain host-owned.

This follows `iterative_dmatrix.cu::InitFromCUDA`: construct batch ELLPACK pages
and concatenate them. A CPU-created quantile matrix uses its existing quantized
bins on first MPS access, uploads them once and packs on GPU. Its maximum row
width is computed from host row pointers, matching
`ellpack_page.cu::EllpackPageImpl(GHistIndexMatrix)`; this host loop is intentional.
No raw-input callback is needed after construction in either direction.

`IterativeDMatrix` retains separate CPU and compressed GPU representations. A mutex
serializes first conversion and publication; existing batch readers keep their
original view. CPU materialization fills a fresh host matrix and publishes it only
after completion. GPU views remain without host arrays even after CPU access, and
retained views own buffers after matrix destruction. This extends the existing
compressed histogram, partition and prediction/SHAP consumers without new kernels
or arithmetic changes. Repeated CPU/MPS switching does not repeat bin uploads or
host materialization.

Cut-only reference lookup uses the owner's existing representation, including a
CPU reference for an MPS child. `get_data()` explicitly requests CPU representation
for sparse export even if the matrix context is MPS. The existing internal
ELLPACK save/load contract is separate and still unported; no new public
quantile-matrix serialization API is introduced.

The new native suite passes **260 cases / 3,266,138 exact fields** under Metal API
validation. It independently quantizes raw input on CPU using the same cuts, then
compares every row pointer, bin, hit count and dense/sparse column. The grid covers
max_bin 2/17/257/65537, dense/relative/sparse/all-missing layouts, weights,
categories, single/multiple batches, empty batches and zero rows. **10 cases**
start six simultaneous CPU/GPU requests and check one published view per device
and exact conversion counts. GPU reads remain valid after DMatrix destruction.

**27 new Python cases** cover both construction devices, training, margins, leaves,
exact SHAP/interactions, model reload and CPU sparse export. A sparse input crosses
the actual 2^26-slot adapter window boundary. Existing adapter checks now require
direct compressed construction and no host staging; two added iterator cases
reject changed columns and row width with unchanged total nnz. Same-model CPU
margins/leaves and CPU exports remain exact; SHAP retains the existing 2e-5 tolerance.
The focused existing regression passes 249 cases; the new public/changed-iterator
run passes 32 cases (27 new plus all five changed-iterator checks).

Ordinary DMatrix ownership was still open at that checkpoint; the next step below
extends it to ordinary in-memory matrices. Compressed disk caches/internal ELLPACK
save/load, optional Snappy, native device arrays, state and collectives remain open.
This is ownership/coverage work, with no new performance, whole-process memory,
NVIDIA execution or cross-Apple claim.
Full regression passes **2,224 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/API checks (2026-11-04, 18:29:39–18:35:28 UTC).
[Validation record](../validation/cuda-inmemory-ellpack-m3.txt).

## Ordinary in-memory matrices: direct CSR compression

`SimpleDMatrix` now constructs MPS quantized pages directly from raw CSR through
`CompressedPage::PushCSR`, without an eager host gradient-index/column mirror or
an intermediate typed GPU index. It retains the original raw CSR, as CUDA does.
The source contract is `ellpack_page.cu::EllpackPageImpl(DMatrix*)`,
`CreateHistIndices`, and `CompressBinEllpackKernel`.

The kernel implements dense, feature-relative sparse and global-bin sparse
layouts. Relative sparse rows locate feature slots with lower-bound search;
numeric bins use clamped upper-bound search, categorical bins use lower-bound
search with `AsCat` conversion, and null slots are written explicitly. Output
row offsets cover appended pages and subwindows; input row offsets can be nonzero.
Raw row pointers and entries upload in bounded row windows. CUDA's
`TotalMemory/(16*stride*sizeof(Entry))` policy is mapped to Apple's recommended
working-set budget and maximum buffer length. An internal diagnostic argument can
reduce the row-window limit. Empty/zero-stride pages avoid division by zero and
spurious work. Row-width calculation remains on CPU, matching CUDA `GetRowStride`.

The existing CUDA-derived sketch handles optional Hessians and weights. Cache
regeneration still respects max-bin/sparse-threshold changes, explicit regeneration
and mutable CPU/MPS sketch changes. Fixed references and empty-parameter readers
retain cuts. Constant-Hessian `approx` reuses its page; variable-Hessian `approx`
regenerates per updater invocation. Old batch readers retain their original
buffers after regeneration. A mutex serializes publication and first conversion.

CPU-only batch access materializes a separate host view once. Approximate pages
with a NaN sparse threshold omit columns, matching ordinary GHist construction;
normal hist pages create columns. A CPU-origin fixed-cut request can instead pack
existing host bins once. Neither direction rereads raw input after conversion.
Mutable MPS training after CPU training explicitly regenerates CUDA-derived cuts
and compresses raw CSR, rather than retaining the CPU sketch's bins.

`BatchParam::for_cuts` marks metadata-only requests from cut export/reference
lookup; it is cleared in cached generation parameters. Such a request returns an
existing representation without uploading bins or materializing CPU arrays. This
is needed because ordinary matrices keep their initialization context even when
training generated MPS cuts. Raw `get_data()` and ordinary CPU prediction use the
original CSR and therefore need no quantized host mirror either.

Three counters record completed raw compression dispatches, values and uploaded
raw CSR bytes: `ellpack_csr_dispatches`, `ellpack_csr_values`, and
`ellpack_csr_input_bytes`. Bytes include each window's rows/entries, excluding cut
metadata. Existing host-bin upload, materialization and typed-index counters must
remain unchanged on direct construction. CPU-only builds report zeros.

Native raw-format validation passes **70 cases / 568 windows / 120,537 exact bytes /
7,573,364 fields**, plus 12 expected errors. An independent scalar search and the
upstream compressed writer verify every byte, row pointer, global bin and hit count.
Tests include windows 1/7/31/default, appended empty pages, nonzero CSR offsets,
zero rows/cuts, dense/relative/sparse/all-missing layouts, numerical subnormals,
categorical fractional values and clamping, max-bin 3/17/257/65537, and a 65,537-row
sparse case. Malformed row widths, features, cut points and construction order fail.

Native matrix validation passes **130 cases / 1,173,186 exact fields**, including
**10 concurrent conversions and 35 regenerated views**. Independent CPU searches
use the same cuts across hist/approx, weights, categories, layouts and bin widths.
It checks metadata-only access, exact CPU rows/bins/hits/columns, fixed-reference
errors, CPU-origin conversion, backend switching and GPU reads after regeneration
and matrix destruction. The focused existing regression passes 303 tests.

**40 new public tests** cover scalar hist/approx, constant/variable Hessians,
vector hist, DART, CPU predictions, exact SHAP/interactions, supported approximate
SHAP, model reload, raw export, references, max-bin changes and cross-booster reuse.
Unsupported vector approx and approximate vector SHAP are upstream limitations;
the latter's error is checked. Approx-to-hist reference construction retains the
existing sparse-threshold mismatch error. Numeric tolerances remain unchanged.

At this checkpoint disk-backed `SparsePageDMatrix` and
`ExtMemQuantileDMatrix(on_host=False)` still used host gradient-index pages; the
following step replaces MPS-origin disk storage. Optional Snappy, native Apple
arrays, general state/metadata and collectives remain open. No performance, whole-process memory, NVIDIA execution or cross-Apple claim.
Full regression passes **2,264 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/API checks (2026-11-04, 18:46:04–18:52:37 UTC).
[Validation record](../validation/cuda-ordinary-ellpack-m3.txt).

## Compressed disk caches and internal quantile page IO

The Metal implementation now uses CUDA's raw ELLPACK disk fields and the existing
8-byte aligned streams: row count, dense flag, stride, compressed-byte count and
payload, base row, symbol count. The compressed payload retains CUDA's eight-byte
leading padding, bit order and actual allocation length, including excess padding
from concatenated pages. Cuts are supplied separately, as in CUDA's page format.
Authority: `ellpack_page_raw_format.cu`, `common/compressed_iterator.h`,
`SparsePageDMatrix::GetEllpackBatches`, and `IterativeDMatrix::Save/Load` in CUDA.

Production writing gathers compressed head/tail bytes into an owned shared buffer,
then writes them without decoding bins. Reading checks short fields/payloads,
boolean representation, supported rows/stride, symbol geometry, buffer length and
base-row overflow before publishing a view. Loaded payloads are copied into owned
Metal storage: `prefetch_copy=True` adds a private-buffer blit; false retains the
owned shared copy. Both survive reader unmapping and file deletion. This is an
explicit copy fallback, not CUDA HMM/ATS-style mapped-file zero-copy. Compressed
payload symbols are not exhaustively validated; these are internal cache files.

`MetalDiskPageSource` uses existing aligned disk streams, cache offsets and the
bounded prefetch mechanism with two pages. Workers always return compressed GPU
views. The iterator thread materializes CPU rows/bins/hits/columns only for CPU
access; `for_cuts` metadata requests avoid it. Interrupting/resetting traversal,
switching CPU/MPS, releasing callbacks and destroying caches preserve ownership.
Existing host-cache readers now honor `for_cuts` explicitly too.

`ExtMemQuantileDMatrix(device="mps", on_host=False)` writes one compressed page per
input batch directly from adapter windows. Per-pass batch geometry/non-missing
checks, metadata accumulation, construction-error cleanup and fixed-cut rules
remain. Its cache suffix is `.metal-ellpack.page`. Ordinary external MPS training
uses CUDA-derived sketches and raw CSR compression into `.gradient_index.page`;
that suffix is shared with the CPU implementation, but the generation determines
its format. Old prefetch ownership is released before replacing the cache source.
Mutable CPU/MPS, max-bin and Hessian-driven regeneration remain intact. Retained
GPU pages own their buffers across file overwrite and matrix destruction.

Internal `IterativeDMatrix::Save/Load` writes the existing cuts format followed by
raw ELLPACK. CUDA's cuts serialization omits minimum values; its matrix serialization
omits labels, weights, general MetaInfo and training parameters. Metal preserves
that internal contract, supports GPU page inspection/re-save, and rejects CPU
materialization without the omitted training parameters. This adds no public
QuantileDMatrix serialization API. CPU-only builds retain unsupported-operation
errors. A CPU-created QDM must first acquire a compressed Metal view before saving.

Five counters report completed IO: `ellpack_io_read_pages`, `ellpack_io_write_pages`,
`ellpack_io_read_bytes`, `ellpack_io_write_bytes`, and `ellpack_io_dispatches`.
Byte counters measure compressed payload only, excluding aligned headers/cuts.
Blit time contributes to existing ELLPACK GPU time. CPU-only builds report zeros;
production IO does not use diagnostic compressed readback counters.

Focused native checks pass **224 external-cache cases / 2,898 page visits /
12,662,693 exact fields** across host/disk quantile caches and ordinary disk caches.
They compare CPU rows/bins/hits/columns, both prefetch policies, repeated interrupted
CPU/MPS traversal, released callbacks, file cleanup, mutable CPU/MPS/max-bin
regeneration and retained compressed views after overwrite/destruction.

The independent IO suite passes **124 cases / 18,400 exact bytes / 11,875,608
fields / 111 invalid streams / three internal QDM save/load cases**. It covers
1/2/17/257/65537 bins, dense/relative/global sparse/empty layouts, zero rows,
split ratios 0/.23/1, oversized padding, exact raw fields, real mmap close/unlink,
CPU reconstruction, explicit IO counters, every truncated prefix of a fixture and
invalid geometry. Upstream's CPU bit writer supplies expected payload bytes;
a separate field encoder supplies expected aligned streams.

**254 existing focused public tests and 50 new public cases pass.** New cases
cover independent disk-field parsing, scalar/vector/multiclass/DART, hist and
constant/variable-Hessian approx, missing values/categories/weights, exact margins
and leaves, SHAP comparisons, model reload, CPU mirror boundaries, max-bin/device
regeneration and cleanup. Existing numerical tolerances are unchanged. A new
Python fixture initially assumed four padding bytes; source inspection corrected
it to CUDA's eight. Native exact-byte tests already used the upstream definition.

At this checkpoint ordinary external host-cache selection was still open; the
following step connects it. Remaining gaps include zero-copy mapped-file lifetime
integration, optional Snappy, native Apple arrays, general metadata/state, remaining algorithm audits and
collectives. CPU-origin fixed external caches keep their existing compatibility
path; CUDA rejects CPU-created external QDM GPU access, so adding compressed
conversion there is not required for CUDA parity. No NVIDIA,
cross-Apple, performance or whole-process memory claim. Full validation passes
**2,314 Python tests, one skip**, every rebuilt native suite and CPU-only build/API
checks (2026-11-04, 19:06:50–19:12:57 UTC). A final local autorelease scope around
the IO suite's compressed-count diagnostic was rebuilt and passed the complete
IO suite at 19:13:07 UTC; no API/ABI or arithmetic changed.
[Validation record](../validation/cuda-disk-ellpack-m3.txt).

## Ordinary external-memory host caches

`SparsePageDMatrix` now honors `on_host=True` for MPS quantized pages, following
`sparse_page_dmatrix.cu::GetEllpackBatches` and `ellpack_page_source.cu::CalcCacheMapping`.
The raw `.row.page` file remains, matching CUDA's ordinary external matrix design.
With host caching, no quantized disk file is created by MPS construction. Disk mode
continues to use the compressed format described above.

The existing `MetalPageSource` gains construction from ordered compressed CSR
pages. Each page is counted on GPU, committed to the selected shared/private
storage, and retained by the cache; metadata offsets are committed before
prefetch starts. Existing compressed view, two-page prefetch and explicit CPU
materialization paths are reused. No new shader or arithmetic is needed.

CUDA's ordinary-matrix cache plan uses its validation default: multiple pages have
a host ratio of one; a single page overrides it to zero. Metal uses shared buffers
for the host portion and private buffers for device storage. Both consume the
same physical unified memory on Apple hardware. Ordinary page concatenation and
explicit cache-host ratios remain rejected as upstream requires. The C API still
defaults to `MatchingPageBytes()` (zero) and rejects other page-size values.

Cache reuse now checks the active source's storage, not just a disk filename.
Metadata-only and empty-parameter readers preserve it even with a CPU context.
New mutable CPU/MPS/max-bin/approx generations release old source ownership and
choose the correct cache key/format. Host generations use a non-file
`.metal-ellpack.page` cache entry; CPU and Metal disk generations use
`.gradient_index.page`. A previously generated CPU disk cache can remain until
matrix destruction; a subsequent host generation does not write it. Retained GPU
views own their buffers after generation replacement or matrix/cache destruction.

Native validation passes **288 cases / 3,570 page visits / 16,273,773 exact fields**.
The 64 new cases cross bins 8/257, dense/relative/global sparse/all-missing layouts,
categories, weights and single/multiple input batches. Tests independently compute
cache bytes with CUDA's CPU bit writer, check private/shared default accounting
and actual file inventory, and compare all CPU rows/bins/hits/columns. They retain
GPU pages across CPU/MPS/max-bin regeneration, interrupted traversal, both prefetch
policies and matrix destruction. Existing disk and external quantile cases still pass.

Focused public regression passes **390 tests**, including 30 new host variants
covering hist, constant/variable-Hessian approx, scalar/vector/DART, categories,
missing values, weights, CPU predictions/SHAP, fixed references, model behavior,
regeneration and cleanup. **22 additional edge cases pass** for host/disk,
single/multiple batches, empty/nonempty rows, hist/approx, cache reuse and rejected
concatenation. Empty predictions retain the upstream `(0, 0)` shape and are compared
against CPU output; the initial fixture's `(0,)` assumption was corrected without
changing implementation or tolerances.

The CPU-only API check now also trains ordinary external caches with both host
options, regenerates max-bin 8/17/8, checks disk-file selection and verifies cleanup.
No statistics ABI, CUDA algorithm, numerical tolerance or shader change. Full
regression passes **2,366 Python tests, one skip**, every rebuilt native suite and
the expanded CPU-only build/API checks (2026-11-04, 19:21:04–19:27:22 UTC).
No performance or cross-hardware equivalence claim.
[Validation record](../validation/cuda-external-host-m3.txt).

## Prediction input preparation and row-level recoding

Prediction and SHAP use a GPU-owned input chunk, replacing CPU `DoFill` loops.
Raw CSR pages, dense float arrays, typed/strided arrays, masked columnar arrays,
CSR array interfaces and dense/sparse gradient-index pages are decoded by Metal.
The prepared buffer feeds traversal or SHAP directly; it is not read back to
fill another host array. SHAP base margins use a contiguous source span instead
of a separate CPU row-copy loop.

CUDA authority is `gpu_predictor.cu::DeviceAdapterLoader`,
`gpu_data_accessor.cuh::SparsePageLoaderNoShared/EllpackLoader`,
`device_adapter.cuh`, `CatAccessor` and the typed array-interface conversions.
Missing values are filtered before mapping for adapter input. Quantized numeric
bins use negative infinity for the first bin and the preceding cut otherwise;
categorical bins use their stored category value. The Metal backend reads its
existing row-wise gradient-index representation rather than adding ELLPACK just
for prediction. No column transpose is needed by the decoder.

All ten numeric types accepted by the current host adapters are covered: float32,
float64 and signed/unsigned 8/16/32/64-bit integers. Binary64 input narrows on GPU
through the existing software conversion. Raw float bits and missing comparisons
preserve subnormal values and signed zero. Byte-window packing supports negative,
zero and positive strides and keeps mask bit offsets intact across chunks.
CUDA float16 device-array entry points and native Apple device-array interop
remain open; Apple host adapters already reject float16 interfaces and float128.

Finite category codes in the int32 conversion domain follow truncation toward
zero and mapping bounds, including negative fractions mapping as zero. Missing,
infinite and out-of-range codes are preserved without invoking an undefined C++
cast. Actual NVIDIA behavior for casts outside that domain is unverified. The
raw CSR decoder retains the host view's last-entry behavior for duplicate feature
indices; CUDA's lookup assumes canonical rows, so duplicate/unsorted rows are not
a CUDA-equivalence claim.

Host work remains: raw byte/descriptor packing, allocation, uploading input/maps,
two CSR chunk-boundary offset reads and a four-byte GPU status read. Preparation
and traversal are separate synchronous commands. Physical strided windows must
fit the Metal buffer limit. The later explicit MPS QuantileDMatrix path above
adds local dense/columnar training ingestion and bin construction. Persistent
dictionary/raw-input ownership, native device arrays, optional cache compression,
metadata operations and collectives remain open. No performance improvement is claimed for this change.

Direct native checks pass 144 cases / 6,677,824 values (exact bits except NaN
classification), plus two invalid sparse-input errors. All integer extrema,
random bit patterns, double rounding boundaries, strides, masks, custom missing
sentinels, category bounds, empty inputs, compressed bin widths and nonzero page
base IDs are exercised. Public tests add 71 cases and strengthen the existing
category round trips with GPU-preparation/recode and zero-readback assertions.
The focused suite passes 210 tests without relaxed prediction tolerances.

Counters: `input_prepare_dispatches`, `input_recode_dispatches`,
`input_prepare_gpu_time_ns`, `input_debug_readback_bytes`. They report completed
GPU commands and explicit diagnostic reads; production consumers leave dense
input readback at zero. Raw packing, transfers and waits are outside GPU time.
CPU-only builds report zeros. Full regression passed **1,946 Python tests, one
skip**, every rebuilt native suite and the CPU-only build/check
(2026-11-04, 16:05:29–16:10:51 UTC).
[Validation](../validation/cuda-input-preparation-m3.txt).

## Typed categorical name sorting and recoding

`CatContainer::Sort` now routes MPS dictionaries to GPU segmented index
initialization, sorting by feature and typed name, and extraction of the sorted
local indices. `cpu_impl::MakeCatAccessor` dispatches MPS recoding to GPU schema
validation, segmented lower-bound searches, candidate equality checks and the
first missing-name reduction. The existing API name reflects its host views;
with an MPS context, the name lookup calculations now run on Metal. These calls
are used by model training, batch/in-place/leaf prediction and SHAP.

The source type list contains strings and unsigned/signed 8/16/32/64-bit integers.
It does not support floating category names. Integer widths remain intact; in
particular, signedness and values above 2^53 never pass through float/double.
Strings compare lexicographically as signed int8 spans, including embedded NULs
and non-ASCII bytes. The test corpus includes long shared prefixes, empty
strings, and empty columns between populated feature segments. The Python
adapter's pre-existing rejection of uint64 names above INT64_MAX remains intact;
full unsigned range behavior is checked through the native API.

CUDA validates types and empty status across the entire dictionary before any
name searches. Metal preserves that order: a later schema mismatch takes
precedence over an earlier unknown name. Source BasicChecks remain on the host.
Unknown-name selection is a GPU minimum over missing indices; only formatting
the selected error is host-side. Output maps are copied to the host after
successful validation to serve the existing accessors. Sorting uses stable
parallel merge ranks instead of Thrust's sort; unique names have the same order.
Duplicate-name tie order is not an NVIDIA-equivalence claim.

Native tests compare against the vendored host encoder and independently known
maps: 33 cases / 2,730,917 exact indices and 80 expected errors. They cover every
source type, 0/1/31/32/33/255/256/257/4097/65537 sizes, permuted/subset/single-name
incoming dictionaries, repeated feature boundaries, output guards, container
copy/sort/accessor integration and completed GPU counters. Nineteen public tests
check prediction, leaf indices, contributions, interactions, in-place calls,
serialization, category subsets, unknown names and CUDA schema-error precedence.
They require exact same-model outputs after code reordering and use a 3e-6
CPU comparison for ordinary numerical output.

The new counters `category_sort_dispatches`, `category_recode_dispatches` and
`category_gpu_time_ns` describe completed commands. Empty sorting and zero-feature
recoding require no dispatch; nonzero-feature recoding validates schemas even
with no category entries. Invalid schemas/unknown names still count GPU work.
All counters report zero in CPU-only builds.

The later input-preparation step below removes host row-level recoding from
prediction and SHAP. Dictionary copy/storage ownership, cached GPU dictionaries
and maps, general metadata operations and native device arrays remain unfinished.
Compressed external caches are covered above. The name milestone alone did not port the
entire data preparation pipeline. No performance benchmark or NVIDIA/cross-Apple comparison was run.
Full regression passed **1,875 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/check (2026-11-04, 15:41:19–15:46:40 UTC).
[Validation](../validation/cuda-category-names-m3.txt).

## Adaptive objectives and generic reduction precision

The absolute-error and quantile objective branches now preserve the vendored
CUDA expressions: FP32 residual subtraction, sqrt(abs(residual)), row weighting,
hypot/tanh, ordered comparisons and left-associated products. Software-backed
math preserves subnormals. Hypot returns infinity if either argument is infinite,
including when the other is NaN. Quantile keeps its zero-scale/zero-weight early
return, continuous ratio at zero and source fmaxf curvature floor.

A return type of double does not imply binary64 accumulation here.
`numeric.cu::Reduce` passes `thrust::plus<float>` to `dh::Reduce`: it adds in FP32
and widens the final result. `optional_weight.cu::SumOptionalWeights` instead
passes transparent `cuda::std::plus` with a double initial value and accumulates
in binary64. The new Metal generic reducer and adaptive root sums preserve this
distinction. Default optional weights retain the source FP32 count product.
Two-term fixtures distinguish both precision loss (`1e8 + 1`) and FP32 overflow
from a binary64 weight sum. This also corrects absolute-error initialization's
use of the generic reducer; other optional-weight callers keep binary64 sums.

Each adaptive call reduces its statistics completely on GPU, copies only the
`targets + 1` double statistics to the host collective, then computes scales and
gradients on GPU. This matches CUDA's explicit CPU-context GlobalSum. Empty
workers participate before returning and dispatch no kernels. The independent
Learner guard still rejects distributed MPS training: this change restores
objective behavior and does not implement GPU collectives or distributed trees.

The extracted-source reference rebinds the actual objective lambda bodies and
substitutes only math primitives. Tests check 2,359,332 source-composition fields
exactly (NaNs by class, signed zeros by equality). A separate 1,000,800-field
hypot/tanh corpus compares against host binary64 libm rounded to FP32, with a
one-ULP ceiling and a maximum observed error of zero. Actual objective calls use
exactly representable root/weight sums, an independent global-data reference and
the original CPU objectives at an unchanged 3e-6 scaled tolerance. One/two/three
local worker runs cover 308 calls / 8,815,888 checks, including empty workers,
zero/tiny weights, multiple targets and 65,537-row recursive sums. There are
38,792 output pairs that distinguish global from worker-local scales.

Metal uses a recursive 256-input FP32 reduction tree. Its association is not CUB's
architecture-dependent association. Wide-exponent generic-reduction tests now
use the FP32 forward-error bound instead of the old, incorrect binary64 bound;
exact-integer/subnormal/nonfinite cases and binary64 optional-weight checks stay
strict. The generic reducer preserves CUDA's outer 1,073,741,823-element batch
boundary, but that capacity has not been exercised on this machine. The adaptive
fused reduction uses its own tree over the complete local input; huge inputs
crossing CUDA's batch boundary remain an association-validation limit.
NVIDIA libdevice results, compiler contraction, cross-Apple execution and current
performance remain unmeasured. No CPU correction or relaxed ordinary-training
tolerance was introduced. Full regression passed **1,856 Python tests, one skip**,
every rebuilt native suite and the CPU-only build/check (2026-11-04,
15:24:25–15:29:25 UTC). [Validation](../validation/cuda-adaptive-source-m3.txt).

## Multiclass gradient and transform arithmetic

`multiclass_obj.cu::GetGradient` now has a dedicated Metal kernel with one thread
per row. It retains CUDA's positive `FLT_MIN` initial maximum, `fmaxf` NaN rules,
FP32 margin differences/exponentials and a sequential binary64 sum. The final
sum is cast to FP32 once before probability division. Hessians retain the
left-associated FP32 products and the unweighted `1e-16f` floor; the label class
alone subtracts one before multiplying by the row weight. The previous kernel
repeated a full row scan for each class and used compensated FP32 summation.

Prediction softmax follows `common::Softmax`, whose initial maximum is the first
margin instead of positive `FLT_MIN`. Exponential terms are stored and reused
before division, as in the source. This distinction matters for sufficiently
negative margins: gradient denominators can underflow to zero while transformed
probabilities remain defined. The source behavior is preserved explicitly.
`common::FindMaxIndex` also begins with the first element. Argmax therefore keeps
index zero for a leading NaN, retains the first equal maximum and compares
subnormals without flushing them to zero.

Existing software FP32 helpers preserve operation boundaries and subnormals;
software binary64 implements the sum. The exponential oracle rounds host
binary64 `exp` to FP32. These are source-expression checks, not a claim of NVIDIA
libdevice or NVCC contraction equivalence. All 12,181,658 checked result fields
match the oracle exactly (NaNs by classification and signed zeros by equality).
The unmodified host `common::Softmax` helper is checked separately with a four-ULP
budget; its observed maximum difference is two ULP. No GPU performance benchmark
was run, so removing repeated scans is not a measured speedup claim.

Production dispatch keeps the existing synchronous shared-memory eligibility,
overlap-safe output staging and objective/transform counters. Multiclass requires
one label per row. Empty operations do not dispatch; invalid label shape does
not dispatch or change output. All other objective/transform arithmetic remains
subject to its separate source audit.

Native validation covers 192 cases, 1–4,097 classes, weighted/unweighted rows,
zero/subnormal/large weights, ordinary and extreme margins, NaNs/infinities,
leading-NaN argmax, aliases and guards. It includes 7,199 ordinary-FP32 and 1,236
compensated-FP32 denominator witnesses, 299,290 initial-maximum witnesses and
2,638 argmax witnesses. Thirty public Python cases cover softprob/softmax,
2/17/257 classes, extreme/tiny/tied margins, unchanged raw margins, GPU dispatch
and serialization. Existing training comparisons retain their tolerances. Full
regression passes **1,841 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/device check (2026-11-04, 14:41:03–14:45:50 UTC).
[Validation record](../validation/cuda-multiclass-softmax-m3.txt).

## Multiclass priors and shared vector arithmetic

`multiclass_obj.cu::InitEstimation` now uses Metal for its class-weight histogram,
normalization, logarithms and subtraction of the log mean. Weight sums and the
mean already dispatched Metal; scalar mean accumulation is now FP32 to match
`stats.cu::Mean`, replacing its former binary64 accumulation of FP32 terms.
Sample/weighted-sample means retain their distinct source binary64 expressions.
The shared `VecScaDiv` path also covers quantile initialization.

The histogram follows `linalg_op.cu::SmallHistogram`: stable ascending label sort,
run-length segment boundaries and segmented FP32 weight sums. Run-start flags
and an exclusive integer prefix give the same boundaries as CUDA's run counts
followed by inclusive scan. Tail bins are zeroed on GPU, including empty input.
The CUDA ArgSort dispatch uses `is_overwrite_okay=false`; labels and weights
remain unchanged. Outputs retain strided guards.

**Missing-class behavior follows CUDA, not the CPU histogram.** CUDA allocates
unique labels but does not scatter sums to those class IDs. It packs distinct
runs into the leading output bins. Labels `[1,1,3]`, four classes, therefore
produce `[2,1,0,0]` instead of CPU's `[0,2,0,1]`. This can alter initial predictions
when classes are absent. Source witnesses and public tests cover that difference.
The objective's existing predicate requires integral in-range labels, so
fractional labels are rejected. The low-level helper additionally rejects
nonfinite labels or more distinct runs than bins without modifying host output;
CUDA's fixed-size run buffers have no safe result for the latter case.

`VecScaMul` performs binary64 `float_value * double_scalar` then narrows to FP32;
`VecScaDiv` retains the host-computed binary64 reciprocal. `LogE` first adds its
FP32 epsilon and then evaluates logarithm; centering subtracts in FP32. The
existing software math helpers preserve subnormals. Vector operations update
only selected elements in strided views, using staged snapshots.

Histogram sums and scalar means use FP32 local additions and a 256-lane tree.
Mean terms divide by `float(n)` before reduction. CUB association is not reproduced;
tests compare exact integer/power-of-two cases and independent high-precision
sums. Finite error budgets use `k*epsilon/(1-k*epsilon)*sum(abs(terms))`, with
`k=ceil(n/256)+8`; overflow and nonfinite outputs have separate classification
checks. A cancellation witness distinguishes the FP32 mean from its former wider
accumulation. Host allocation, scalar staging and shared collective call sites
remain; this does not establish full device residency or distributed transport.

New counters: `small_histogram_dispatches`, `vector_arithmetic_dispatches`,
`float_mean_dispatches`. The first two use `initialization_gpu_time_ns`; float
means are also included in `mean_dispatches`/`mean_gpu_time_ns`. CPU-only builds
report zero. Actual NVIDIA/libdevice execution and other Apple GPUs remain
unverified; no performance benchmark was run. Native checks pass **258 cases /
10,499,275 fields**, including 660,150 packed-class and 108,340 binary64-scalar
witnesses. Observed vector error is 0 FP32 ULP; the maximum finite reduction error
is 0.1068 of its association bound. Full regression passes **1,811 Python tests,
one skip**, every rebuilt native suite and the CPU-only build/device check
(2026-11-04, 14:27:19–14:32:28 UTC).
[Validation record](../validation/cuda-multiclass-init-m3.txt).

## AFT device execution and numerical behavior

AFT gradients now evaluate every row on GPU. The correction-index buffers, GPU
compaction, host readback of those indices and `CorrectSurvival` CPU loop are
removed. Only completed FP32 gradient pairs return to host. The legacy
`survival_cpu_corrections` and `survival_cpu_time_ns` fields remain zero for API
compatibility. Input staging and synchronization remain; this change does not
provide persistent end-to-end device ownership.

The code retains `common/survival_util.h`'s binary64 PDF/CDF expressions,
censor-specific limits, clipping, FP32 narrowing and weighted FP32 products.
NaN lower bounds now take the source's ordered `lower <= 0` branch correctly;
identical NaN bound bits no longer compare equal. Metrics already received those
guard fixes in the preceding metric port.

**The old all-rows CPU-equivalence contract has changed explicitly.** Nearby CDF
values can cancel, amplifying a few ULP of transcendental rounding into large
changes in gradient/Hessian clipping. Preserving CUDA's formula does not imply
matching Apple CPU libm in that region. The existing 307,275-pair stress corpus
has 322 fields outside the old CPU tolerance, with a maximum weighted difference
of 44.7845306. None is hidden by CPU recomputation or a larger tolerance.
Uncensored and well-conditioned rows retain the original `1e-6 + 2e-5*abs(ref)`
CPU check. The former `1e-5` likelihood threshold now classifies test conditioning
only; it does not select a production computation path.

Validation covers every row in that corpus, plus 9,408 domain-edge rows:

- `tests/build_aft_reference.py` extracts the actual upstream probability and
  AFT class bodies. Only log/exp/erf callbacks and distribution/limit type bindings
  change; arithmetic, predicates and clipping expressions are retained.
- Diagnostic kernels expose production log/z/PDF/CDF/derivative values and raw
  gradients. Each transcendental is independently compared with host libm at the
  actual argument, retaining the existing four-binary64-ULP test budget. Observed
  log/exp/erf maxima are 2/1/3 across 1,900,098 fields; this is sampled evidence,
  not a universal bound or NVIDIA comparison.
- Replaying those checked values through upstream expressions matches 3,166,830
  binary64 source fields exactly (NaNs by classification). Production FP32
  gradients/weights match the resulting raw GPU outputs on all 316,683 rows.
- Public training, metrics, reload and transform tests retain their existing CPU
  tolerances and now require zero CPU correction count and time.

This establishes source-expression coverage and checked math-library accuracy,
not cross-device output equivalence. NVCC contraction, NVIDIA libdevice results
and execution on other Apple generations remain unverified. No performance
benchmark was run for this change. Full regression passes **1,787 Python tests,
one skip**, every native suite and the CPU-only build/device check
(2026-11-04, 14:12:43–14:17:04 UTC).
[Validation record](../validation/cuda-aft-device-m3.txt).

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

## Elementwise and survival metric reductions

`elementwise_metric.cu::Reduce` and `multiclass_metric.cu` form each weighted
row result in FP32, then convert it and the weight to binary64 for reduction.
Metal now follows those boundaries for RMSE/RMSLE/MAE/MAPE, logloss, threshold
error, Poisson, both gamma metrics, Tweedie, pseudo-Huber and multiclass metrics.
The former scaled FP32 block sums and host scale-restoration loop are removed.
Quantile/expectile keep their source FP32 operation order, including both terms
of the quantile expression and left-associated expectile products. Survival
metrics retain source binary64 likelihood/accuracy and weight products. Ordered
NaN-bound comparisons now match the source before likelihood regularization.

All these paths finish with recursive 256-input GPU binary64 pair reductions.
The existing integer reduction fast path is used only when its exponent-span
check proves every subset sum exactly representable in binary64. Other blocks
use software binary64 additions. Only the final 16-byte residue/weight pair
returns to the host, alongside a separate 4-byte validation flag in the generic
metric path. CUB association is not reproduced; tests separately compare
row expressions and the reduction against compensated sums, with a `32*epsilon*
sum(abs(terms))` budget for the checked finite cases. Final ratios remain host,
as in CUDA. The previously omitted `GlobalSum` calls in scalar/pseudo-Huber/
quantile/expectile MPS branches are restored. This restores the shared collective
call sites; it does not implement distributed Apple GPU transport.

The row helpers preserve FP32 subnormals and rounding between expressions.
Logs, log1p and exponentials use the existing software binary64 functions before
FP32 conversion. Log1p recovers the low term of `1+x`; sqrt refines a native seed
in binary64 and handles negative subnormals explicitly. Gamma uses a Taylor
series near 1 and 2, recurrence plus Stirling elsewhere, and a binary64 sine/reflection
formula for negative nonintegers. Gamma's zero/pole/nonfinite behavior follows
the [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html).
These are independently tested approximations, not a reconstruction of libdevice.
The [CUDA accuracy table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
also allows function-specific errors; no cross-device bitwise promise is made.

Multiclass metric validation now runs on GPU and casts labels to signed integers
before range checking, as CUDA does. Thus `-0.75` maps to class zero and `1.75` to
class one; the objective's different label predicate is unchanged. Nonfinite or
out-of-int32 input casts have no defined CUDA-source contract and are explicitly
rejected. Atomic error flags replace CUDA's competing writes of an offending
label. Multiclass logloss applies the source `prediction > epsilon` branch,
including NaNs; merror preserves first-index ties and ordered comparisons.

Validation checks **15,628,618 exact row-result fields**, **349 production
cases**, and **500,285 math fields** matching host binary64 libm rounded to FP32.
The largest finite reduction error is 0.03734 of the stated bound. Existing
public metric/survival tolerances remain unchanged. Actual NVIDIA libdevice,
NVCC contraction, CUB association and cross-Apple execution remain unverified.
Full regression passes **1,787 Python tests, one skip**, all rebuilt native
suites and the CPU-only build/device check (2026-11-04, 13:59:56–14:04:22 UTC).
[Validation record](../validation/cuda-metric-reductions-m3.txt).

## Ranking gradient numerical behavior

Mean pair construction now follows `objective/lambdarank_obj.cuh`'s
`MakePairsOp::WithSampling` and the modern CCCL branch of `common/random.cuh`.
Prediction ranks and then labels sort stably on GPU. Each pair independently
seeds Philox4x64 with the wrapped 32-bit query seed, discards its query-local
pair index, and applies `uniform_int_distribution<size_t>` to the complement of
its equal-label bucket. Pair order is sample-major, repeat-minor. Singleton
ranges consume no engine output; homogeneous queries produce `(0,0)` pairs.
The older Thrust CUDA stream is not claimed equivalent.

One Metal lane samples each pair. A separate query kernel builds incidence
lists in pair-index order for the existing gradient gather; that list-building
step remains sequential within a query. Production pair buffers retain the
64 MiB batching budget (a single larger query may exceed it within the device
limit). An optional internal trace copies raw ordered pairs for native tests;
normal training does not allocate that trace or copy pair indices to the host.

Validation checks **9,068,696 exact production pairs in 74 cases**, including
prediction/label ties, signed zeros, subnormal labels, empty queries, padded
label strides, seed wraparound, noncanonical rank order and reused batch buffers.
The distribution test checks **659,200 exact draws and consumed positions**,
including 64-bit discard offsets. A separate host program checks **83,328 draws
and consumed engine positions against unmodified NVIDIA CCCL headers**, pinned
to commit `3d0d040ac0dc5907daf2a0731edb378e5562099d`.
[Validation record](../validation/cuda-ranking-pairs-m3.txt).

Both ranking methods now follow CUDA's gradient arithmetic: compute raw
LambdaGrad values, reduce per-query maxima, create FP32 rounding factors, then
truncate each endpoint's gradient and Hessian before accumulation. The two
signs are truncated independently; they can round to different magnitudes.
Normalization uses the untruncated gradients. The final query multiplier is
`float(double(weight) * weight_norm)`, followed by one FP32 multiplication.
MAP now uses the generic recursive segmented binary64 prefix for actual labels
and label/rank. The previous soft-label kernel already used actual labels in a
serial GPU branch; the earlier ledger overstated that gap. This change removes
that serial branch and corrects the old transitional test oracle's binary-hit
approximation for soft labels.

Unbiased training computes raw costs for **every** unequal-label pair, including
untracked positions, and obtains the global minimum absolute bias. It calculates
per-query cost rounding factors, then recomputes pairs in a second pass to apply
binary64-truncated row costs. Bias updates retain the source's conditional ratio
and power formula. The two passes reuse bounded pair storage; gradients and costs
are never recomputed on CPU. Ordinary ranking can retain its raw pair values in
one batch while obtaining factors, avoiding a redundant math pass.

Two source edge cases require explicit policies:

- CUDA's query kernel writes every query's factor into the same
  `d_cost_rounding[0]` slot concurrently. Metal takes the maximum of those factors
  on GPU. This is a deterministic conservative choice, **not a claim to reproduce
  the winner of an NVIDIA data race**.
- For nonfinite `frexp` inputs, the exponent is unspecified in both NVIDIA's
  [single-precision](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)
  and [double-precision](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html)
  functions. Metal explicitly uses exponent zero (factor one), including zero
  bias minima that cause nonfinite bounds. NaNs are ignored in maximum bounds
  but still propagate through gradient/cost arithmetic. Finite source formulas,
  even beyond the positive-denominator range, are evaluated without introducing
  a new clamp. Reproducible-sum guarantees do not extend to invalid bounds.

Validation covers **162,851 exact rounding cases / 977,106 FP32/binary64 fields**
(NaNs by classification), including thousands of asymmetric sign truncations,
subnormals, exact powers, 32-bit counts and nonfinite edge policies. Production
trace tests cover **64 cases**, 1,600 bound/multiplier fields and **2,113,664 MAP
prefix fields**, including untracked cost maxima, global factors differing from
the last query's factor, soft labels, zero bias, empty queries and recursive scans.
These optional traces add no readback to normal training.

The host CUDA-formula oracle replaces the transitional mean-only reference for
both methods. All **2,444,200 ordinary gradient pairs in 1,442 cases** match it
bit for bit. **2,592 unbiased calls** pass unchanged numerical tolerances; all
118,368 serialized bias values match, with maximum scaled gradient error
`1.69500709e-7`. Four additional 524,288-row calls exercise repeated buffer
batches. Python tests check same-model CPU predictions, independent per-round
metrics and deterministic training/bias replay. CPU-trained model identity is
not the contract: CPU does not apply CUDA's truncation.
[Arithmetic validation](../validation/cuda-ranking-arithmetic-m3.txt).

The oracle uses binary64 host transcendental functions and explicitly rounds
sigmoid exponentials to FP32. The Metal sigmoid matches the local binary64-exp
oracle on 2,996,040 inputs; Apple's dynamic `expf` differs on 4,962 of those
(up to four sigmoid ULPs). CUDA libdevice rounding and CUB association have not
been measured. Ordered GPU gathers replace atomic additions of truncated terms;
mean incidence-list construction is still serial within a query. Cross-query
bias sums use ordered 128-query blocks and a final reduction. Small bias vectors
and output gradients return to host storage. Persistent ownership, less
synchronization, performance tuning and NVIDIA/cross-Apple validation remain open.
Query metadata now follows `RankingCache::InitOnCUDA` and
`SegmentedTrapezoidThreads`: GPU adjacent-offset differences and maximum reduction,
mean/top-k pair counts, and exact uint64 inclusive scans of `[0, counts...]`.
Metal's 256-pair tile offsets are scanned on GPU too. The existing recursive
integer scan is reused; no floating-point conversion is involved. Prefixes can
approach the uint64 limit for valid uint32 row/pair parameters.

Each `RankingCache` owns immutable shared Metal offset/task buffers. The objective,
cached sort, NDCG initialization and precision/NDCG/MAP scoring reuse them; parameter
or MetaInfo-identity changes renew the existing owning cache. Standalone grouped sort and
ROC/PR-AUC calls build their query maxima on GPU. Host code still validates shapes
and cached offset identity, reads completed size metadata, allocates storage, and
partitions queries into bounded pair batches. These are explicit synchronization
and scheduling costs, not host pair-count/prefix generation. Row data, gradients,
discounts and bias state still stage through host storage; this change does not
claim complete ranking residency or asynchronous execution.

Native tests check 180 metadata cases and 12,611,280 task fields exactly against
wide-integer CUDA-formula references, with explicit pair enumeration for small
queries. 131,415 checked prefixes cannot be represented exactly by binary64.
They include empty/singleton queries, scan boundaries, 262,145-query recursive
scans and near-uint64 totals without allocating logical rows/pairs. Cache tests
check exact sorting and cached/fresh gradient replay, changed seeds, retained
ownership, invalid offsets and stale-group/parameter rejection. Public tests also
check objective-cache renewal and metric reuse.
[Validation record](../validation/cuda-ranking-metadata-m3.txt).

Earlier ranking performance and CPU-equivalence reports describe the old
unrounded arithmetic (and, for mean, the CPU random stream). They remain
historical records, including failed strict prediction checks, and are not
measurements of the current implementation. No new benchmark was run here.

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
At that checkpoint sparse/quantized page decoding and categorical recoding were
host-side; the later input-preparation and ownership steps above move these
consumers onto GPU. Approximate-contribution
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
alignment, interoperability, collectives and actual NVIDIA validation remain.
The AFT corrections listed at that checkpoint have since been removed. No
benchmark or NVIDIA execution comparison was run for the bound-state change.


## Optional Snappy cache codec (2026-11-04)

CUDA authority: `common/device_compression.cu/.cuh/.h` and
`data/ellpack_page_source.cu`. The raw byte-format authority is
[Google Snappy's format](https://github.com/google/snappy/blob/main/format_description.txt);
[nvCOMP's Snappy API](https://docs.nvidia.com/cuda/nvcomp/c_api.html#snappy)
describes batched chunk buffers, actual output sizes and completion/status handling.
The Metal port does not link nvCOMP or emulate NVIDIA's hardware decompression engine.

Two Metal compute kernels now encode and decode raw Snappy blocks. Encoding uses
independent 32 KiB matching windows and real LZ77 backreferences; a block contains
up to 2 MiB. The compressor's choices can differ from nvCOMP because the byte
format admits many valid encodings. Decompression handles literals, all four
extended-length forms, COPY1/COPY2/COPY4, uint32 varints and overlapping copies.
It checks truncation, impossible offsets, declared-size mismatches, output bounds
and trailing bytes before a reconstructed page can be published. Cooperation
within a copy reads only the already-produced prefix, so overlap cannot race.

Hash/output workspace is bounded to one block at a time, with an autorelease
scope per block. Compressed blocks are coalesced into owned shared Metal storage;
CPU work is metadata, raw length headers and completion/status inspection. Payload
matching, encoding, copying and decoding execute on GPU. The host exchange helpers
in `metal_snappy.h` are explicit diagnostic boundaries; cache production uses the
same kernels directly on Metal buffers and performs no diagnostic byte readback.

`CompressedPage::Commit` accepts the existing native compression ratio and
`allow_decomp_fallback`. Apple has no CUDA DE capability, so the fallback opt-in is
required, just as it is for CUDA's compute branch. The option is wired from
`ExtMemConfig` for external quantile host caches, with default ratio 0.4. No new
Python option or automatic enablement is introduced. Ordinary external matrices
and disk caches keep their existing selection policy; public selection, throughput
and profitability tuning are deferred.

Eligibility follows `!IsDenseCompressed()`: `row_stride != features`, excluding
both fully dense and feature-relative sparse pages. The host cache is split into
a plain prefix and Snappy middle; its private device suffix is unchanged. Logical
page offsets and `HostBytes` still describe original ELLPACK bytes.
`HostStorageBytes` exposes the actual plain-plus-Snappy payload, which can also grow
for incompressible inputs. These counts exclude allocation padding and metadata.
The old full uncompressed page is released when committing.

Every compressed read forces reconstruction into an owned private buffer even if
`prefetch_copy=false`. `View`, `Restore`, GPU non-missing counts, diagnostic readback
and copying a committed source all use that path. Views survive cache destruction;
explicit CPU access remains the only host-bin materialization boundary. Disk IO
continues to consume reconstructed raw ELLPACK views, without changing its format.

Counters expose compression, decompression and coalescing dispatches, input,
compressed and decoded bytes, explicit diagnostic readbacks and GPU time. CPU-only
builds expose zeros. The statistics ABI changed, requiring all native executables
to be rebuilt.

Validation uses Google Snappy 1.2.1 as a test-only independent oracle in both
directions, including COPY4 offsets above 65,535, non-minimal valid headers and
literal lengths, overlap, random/incompressible/repeated data, partial chunks and
2 MiB boundaries. Malformed streams and deterministic mutations are checked
against Google's decoder; guard bytes are checked on success and failure.
Cache tests independently calculate logical/physical byte counts and verify exact
CPU bins/rows/hits/columns, sparse/dense/empty eligibility, native opt-in/defaults,
both prefetch settings, interrupted CPU/MPS traversal, source copying and retained
views. Six compressed-cache fits compare predictions, leaf IDs and exact SHAP with
plain Metal training, plus CPU evaluation of the fitted model (absolute tolerance
1e-5 for prediction/SHAP). Full regression passes 2,366 Python tests (one skip),
every rebuilt native suite and CPU-only build/API checks. The final ELLPACK suite
passes 355 cases, including multi-block cache reconstruction at nonzero offsets.
Results are recorded in [the validation record](../validation/cuda-snappy-m3.txt).

This closes the optional Snappy compute operation gap, not full CUDA parity.
Native Apple feature arrays/half are covered in the following step. Mapped-file
zero-copy, broader state/metadata ownership,
remaining CUDA algorithm audits and GPU collectives still need work. Actual
NVIDIA and other Apple GPU comparisons remain unperformed. No performance claim
is made for the initial codec.

## Native Metal feature tensors and float16

CUDA's `c_api/c_api.cu`, `data/device_adapter.cu/.cuh`, array-interface handling and
quantile/prediction frontends define the corresponding operation coverage. Metal
uses DLPack's `kDLMetal` representation: `DLTensor.data` is an `id<MTLBuffer>`
handle, with byte offset and element strides; it is never treated as a CPU float
pointer. The descriptor frontend validates device, dimensions, dtype, lanes,
stride arithmetic, alignment and addressed buffer extent before GPU access.
Valid buffer handles and producer completion are caller preconditions, like valid
CUDA pointers and stream ordering upstream.

`XGProxyDMatrixSetDataMetal` snapshots the addressed byte span into private Metal
storage with a completed GPU blit. Dense, sliced, transposed, reversed and
broadcast views retain their original logical layout. The snapshot remains valid
after producer mutation or destruction. Empty inputs need no source buffer or
copy. Features feed the existing bounded raw-sketch/ELLPACK adapter pipeline;
flattened 2^26-slot windows preserve partial rows. `XGBoosterPredictFromMetal`
feeds the existing GPU raw prediction pipeline. No feature-payload CPU staging
or diagnostic readback is used by these native paths. This is an owned GPU copy,
not zero-copy producer aliasing.

The Python bridge accepts real PyTorch/MLX Metal tensors and legacy/versioned
DLPack capsules. It consumes each capsule once, retains its owner, invokes the
managed deleter once with the GIL held and rejects unsupported major versions
before reading their tensor fields. Known producer work completes before every
import, including reused exports. Generic producers must supply ready buffers;
DLPack does not provide a portable Metal queue handshake here. MLX assignment
can replace an allocation; retained exports describe the allocation exported,
not a future replacement of the Python array.

Native feature tensors select MPS by default for `QuantileDMatrix` and both
`ExtMemQuantileDMatrix` cache modes; host metadata remains supported. Explicit
CPU construction rejects them. Inplace prediction requires an MPS model and
returns the existing host NumPy result. Float16/32/64, signed/unsigned 8/16/32/64
and bool are decoded on GPU. Binary16 conversion explicitly preserves subnormals,
signed zero and infinities; NaNs retain their classification. Producer dtype
restrictions still apply. BF16, complex and float128 reject explicitly.

Validation uses PyTorch 2.15.0 and MLX/MLX-Metal 0.32.4. Their feature values,
quantile cuts and trained-model predictions are checked against the existing
host-input Metal path; CPU evaluation of the same fitted model is checked too.
It does not compare Metal CUDA-style sketch construction against CPU's different
sketch algorithm. Native tests independently decode private/shared buffers,
mutate and release sources before use, check all 65,536 half patterns, signed
strides, offsets, broadcast windows and invalid extents. Focused results are
93 Python tests and 105 native cases / 145,567 field checks / nine rejected
descriptors. The test-only decoder readbacks are explicitly counted.

Seven counters record native imports, copies/bytes, adapter/prediction windows,
GPU copy time and feature-input CPU staging bytes. CPU builds expose zeros and
the new C entry points reject clearly. All native callers rebuild for the
statistics ABI change. [Validation record](../validation/cuda-metal-tensor-m3.txt).
Full regression passes **2,459 Python tests, one skip**, every rebuilt native
suite/reference updater and the CPU-only build/API checks (2026-11-04,
20:06:27Z–20:13:22Z). The final producer-queue tests are included.

The tested macOS setup uses one Homebrew OpenMP runtime for XGBoost and PyTorch,
selected with `DYLD_LIBRARY_PATH` before Python starts. Loading the wheel's bundled
runtime alongside Homebrew reproduced a worker-thread crash; this is documented
and the test launcher configures the shared runtime. Arbitrary mixed-runtime
packaging is not fixed by the Metal import code.

Metadata imports are implemented in the next section. Ordinary/columnar tensor
inputs and owned GPU outputs are implemented in later sections below. Broader
device ownership and a portable producer queue protocol remain open.
No new performance benchmark, NVIDIA
comparison or other Apple-device validation was performed.

## Native Metal metadata conversion, validation and query groups

`XGDMatrixSetInfoFromMetal` and Python metadata dispatch now accept Metal DLPack
labels, weights, base margins, survival bounds, feature weights, group sizes and
query IDs. The implementation authority is `data/data.cu`: `CopyTensorInfoImpl`,
`CopyGroupInfoImpl`, `CopyQidImpl` and `MetaInfo::SetInfoFromCUDA`, plus the shared
`LabelsCheck`/`WeightsCheck` predicates. This ports native metadata import
operations; generic metadata storage, append/gather and other state manipulation
remain separate unfinished work.

Imports retain the previous native-buffer bounds, ownership and producer
completion rules. Metadata also accepts scalars and empty matrix dimensions.
Labels/base margins preserve CUDA's supplied shape, filling missing dimensions
with ones. They do not apply the CPU setter's sample-count-based reshape.
Vector fields accept vector-shaped data; Python retains its existing public
shape checks. Single-batch quantile construction retains a managed export so a
caller-supplied metadata capsule is consumed once across iterator passes.

Contiguous FP32 input uses a GPU blit; other supported dtypes/strides use typed
GPU conversion into FP32. Label validation rejects nonfinite converted values.
Weight and feature-weight validation also reject negative values, including
negative subnormals; both signed zeros are legal. Bounds and base margins have
no such import-time finite-value predicate, matching CUDA. Validation occurs
after the copy, so invalid float metadata may already be stored when an error
is raised, as in the source CUDA setter.

At this checkpoint, the destination kept the existing `MetaInfo` allocation. Safe synchronous
shared-buffer views allow GPU writes directly into it; otherwise a GPU buffer
and counted host copy provide the fallback. No input payload is converted through
NumPy or a CPU element loop. This does not claim private end-to-end GPU metadata
ownership. Eight new counters expose GPU copies, conversions, validations, QID
commands, element counts, shared output bytes, host payload copies and GPU time.
CPU builds report zero and reject the new entry point explicitly. The later Metal
vector step below replaces float destinations with owned GPU storage and lazy
host mirrors; the conversion/validation kernels remain connected.

Group sizes follow CUDA's integer conversion and host prefix sum, retaining zero
sizes and overflow validation. FP32/FP64 group-size inputs reject; binary16 follows
the source's accepted conversion branch. Query IDs convert to uint32 before
sortedness checks, so wide integer IDs follow CUDA's narrowing behavior. Metal
then marks runs, scans flags, emits run lengths and scans their counts on GPU,
publishing only final group boundaries to the host. Binary64-to-integer
conversion avoids intermediate FP32 narrowing. Unsorted IDs leave existing
groups intact. Empty IDs produce `[0]` without reproducing an unsigned launch-size
underflow. Nonfinite/out-of-range floating-to-integer inputs reject explicitly;
the corresponding out-of-domain C++ casts are not a verified NVIDIA behavior.

Focused validation passes **332 metadata Python cases**, plus all **93 native
feature-input regressions**. Tests use real PyTorch 2.15.0 and MLX 0.32.4 inputs
for every metadata field, six producer dtypes, vectors/columns/slices/broadcasts,
scalars, versioned capsules, empty data and constructor/iterator reuse. Native
tests pass **322 cases / 1,011,267 field checks / 24 rejected inputs**, covering
private/shared buffers, every integer width, negative strides, all half bit
patterns, signed zeros/subnormals/nonfinites and GPU scan tile boundaries through
65,537 IDs. CPU scalar casts and independently enumerated groups are the oracles.

Regression, multiclass, multi-target, ranking and AFT fits match host-metadata
Metal fits exactly. CPU fits using native versus host metadata serialize
identically. CPU evaluation margins agree exactly with the fitted Metal model;
multiclass probabilities use `rtol=2e-7, atol=2e-8` after observing four one-ULP
differences among 771 probabilities (maximum absolute difference 1.49e-8).
The tolerance is confined to that CPU/GPU softmax comparison. External host/disk
iterators preserve imported labels/weights after source release and clean up
their caches. [Validation record](../validation/cuda-metal-metadata-m3.txt).
Final regression passes **2,791 Python tests, one skip**, every rebuilt native
suite/reference updater and the CPU-only build/API checks (2026-11-04,
20:26:46Z–20:33:48Z). No production or test changes followed that final build/run.

No new benchmark, NVIDIA execution or other Apple-device comparison was run.
Custom gradients and ordinary feature tensor inputs are implemented below.
Owned GPU outputs are implemented below. General
metadata/state manipulation and ownership, remaining CUDA arithmetic audits and
GPU collectives remain open.

## Native Metal custom gradients

CUDA authority: `c_api/c_api_utils.h::CustomGradHessOp` and
`c_api/c_api.cu::CopyGradientFromCudaArrays`. For each row/target, the source
independently casts the supplied gradient and Hessian to FP32 and packs them into
one pair. There is no Hessian clamping or finite-value validation at this stage.
The Metal kernel preserves those rules, including signed zeros, subnormals,
negative Hessians and nonfinite values. Later training stages retain their own
validation and numeric policies.

The existing `XGBoosterTrainOneIter` and `XGBoosterTrainOneIterWithSplitGrad` entry
points recognize explicitly tagged Metal descriptors (`device_type=8`). Host and
CUDA array interfaces keep their existing dispatch. Each gradient/Hessian pair
must have matching devices and normalized shapes, and its rows must match the
training matrix. Split and value pairs are independent: either may use host or
Metal inputs, with different dtypes and target widths. Invalid descriptor/shape
inputs leave the model unchanged. CPU-only builds reject Metal input clearly
before touching the opaque buffer handle.

Owned GPU snapshots validate dtypes, device, offsets and signed/broadcast strides
before destination resizing. The GPU decodes float16/32/64, signed/unsigned
integers and bool and writes packed FP32 pairs into an eligible shared host
allocation, otherwise into a GPU buffer followed by a counted copy. This closes
the native import operation; it does not implement generic Metal-owned
`HostDeviceVector`/`linalg` allocation or end-to-end private GPU training state.

Python captures managed exports through each C call. Aliased capsules across
gradient/Hessian and split/value arguments are consumed and deleted exactly once.
Real PyTorch/MLX tensors, plain custom functions, `Objective`, `TreeObjective`,
and direct `Booster.boost` calls use this route. Known producer work is completed
before import; generic providers must supply ready buffers. Legacy flattened
vectors/single columns retain upstream's FutureWarning and reshape by changing
the descriptor, without reading tensor payloads on the CPU. Row vectors are not
silently reshaped. Scalars are additionally accepted for one training row, while
the original NumPy Python helper requires at least one dimension. Empty inputs
preserve their normalized shapes. Callback predictions still arrive as NumPy.
Explicit GPU prediction outputs are implemented below; a portable queue handshake
remains open.

Counters: `gradient_import_dispatches`, `gradient_import_pairs`,
`gradient_import_shared_output_bytes`, `gradient_import_host_copy_bytes`, and
`gradient_import_gpu_time_ns`. Input snapshots use the existing `tensor_*`
counters. These count import work separately from later gradient-state kernels.
CPU-only builds report zero. Native callers of the statistics structure must be
rebuilt when its fields grow.

Native validation passes **333 cases / 328,515 field checks / 17 invalid inputs**.
Independent scalar casts check every integer width, float32/64, differing
G/H dtypes and layouts, both source storage modes, signed strides, transposes,
broadcasts and offsets. All 65,536 half patterns are checked for both G and H,
with exact non-NaN bits and NaN classification. FP32 edge patterns retain their
bits, and negative Hessians remain negative. GPU source overwrite/release cannot
alter the result. Invalid shape/device/bounds/dtype inputs preserve the existing
destination. **163,084 pairs / 1,304,672 shared output bytes / zero host payload
copy bytes** were observed; the copy fallback remains available.

Python tests compare native and host inputs with **byte-identical saved models**
on both CPU and MPS. They cover two producers, six gradient dtypes with separately
typed Hessians, five layouts, legacy flattened shapes, GPU arithmetic inside
callbacks, split/value device placement, exact-once capsule deletion, failed-call
model preservation and empty/scalar inputs. This is input-path equivalence, not
an NVIDIA execution comparison. No new benchmark or cross-hardware run.

Final focused regression: **587 passed** (162 new gradient tests, 93 feature
regressions, 332 metadata regressions). Full regression: **2,953 Python tests
passed, one skipped**, every rebuilt native suite/reference updater and CPU-only
build/API checks passed on 2026-11-04, 20:44:51Z–20:52:00Z. No production or test
changes followed that final build/run.
[Validation record](../validation/cuda-metal-gradient-import-m3.txt).

## Native ordinary matrices and sparse-page import

CUDA authority: `data/simple_dmatrix.cu` and `simple_dmatrix.cuh`, especially
`CopyToSparsePage`, `CountRowOffsets` and `CopyDataToDMatrix`; external iterator
input uses `data/sparse_page_source.cu::DevicePush`. Metal reuses its existing
adapter kernels to decode typed values, filter NaNs/the configured missing value,
count entries, scan row offsets and pack ascending `(feature, value)` entries.
Nonmissing infinities reject before the output page is resized/published;
infinities configured as missing are accepted. Both signed zeros compare as
missing when zero is configured, while stored finite FP32 bits remain exact.

`XGDMatrixCreateFromMetal` accepts the same native tensor descriptor as the
proxy API, with the dense constructor's configuration. Python `DMatrix`, ordinary
`DataIter`, and CPU tree inplace-prediction fallback now use the native import.
Two-dimensional zero-row/zero-column tensors preserve their shape; this also
removes the older native feature bridge's artificial positive-column restriction.
Proxy fallback is selected using the MPS marker as well as `DeviceOrd`, because
this fork's general storage still uses the CPU device ordinal for Metal.

Source tensors enter owned GPU snapshots. Sparse entries/offsets then copy by
GPU blit into safely shareable `SparsePage` allocations; a counted host payload
copy handles allocations that cannot be borrowed. Subsequent raw-CSR sketch and
ELLPACK kernels bind eligible shared spans at their actual byte offsets. Read-only
or otherwise ineligible VM regions use an explicit upload fallback. Borrowed
views never enter persistent caches and are released within the source lifetime
after synchronous commands complete. This avoids an extra CPU payload staging
step for eligible ordinary inputs while retaining host `get_data`, save/load,
slicing, CPU training and prediction. Generic allocation/ownership, CPU slicing
and external serialization retain the existing architecture.

External ordinary inputs preserve both existing cache modes and do not reread
producers after cache construction. Raw ordinary row pages still use their
existing disk serialization; compressed host/disk cache behavior is unchanged.
Categorical values and explicit feature types are supported as before. Native
columnar schemas/dictionaries are covered by the subsequent step below. Direct native
CSR/CSC device-array bindings would be an extension: the vendored upstream
`python-package/xgboost/data.py::dispatch_data_backend` explicitly rejects
`cupyx` CSR/CSC, and `data/proxy_dmatrix.cc::SetCsr` requires host input. They are
not required for CUDA parity. Host sparse inputs and internal GPU sparse kernels
remain fully within the porting goal.

Five output counters (`tensor_csr_dispatches`, `tensor_csr_values`,
`tensor_csr_shared_output_bytes`, `tensor_csr_host_copy_bytes`,
`tensor_csr_gpu_time_ns`) report publication commands and packed entry counts.
Filtering/counting/scanning/scattering retain the `adapter_*` counters, and input
snapshots retain `tensor_*`. `csr_shared_input_bytes` / `csr_input_copy_bytes`
measure raw CSR spans bound to sketch/compression consumers, including row
pointers; totals can include the same span more than once. Existing
`ellpack_csr_input_bytes` remains the logical input extent, not a physical-copy
count. Native statistics callers must rebuild for the extended structure.

Independent native validation passes **199 cases / 550,595 field checks / six
rejected inputs**, covering private/shared sources, all integer widths,
float32/64, every half pattern (infinities tested separately), offsets,
negative/broadcast/transposed strides, signed zero/subnormal/missing rules,
source overwrite/release and empty dimensions. CPU scalar casts and row loops
construct the expected CSR. Observed **272,422 entries / 2,213,760 shared output
bytes / zero host output-copy bytes**. A separate writable/read-only mapped
input fixture verifies exactly **264 shared and 264 copied bytes**, then unmaps
the source before reading the resident sketch summary.

Focused Python regression passes **773 tests**: 186 new ordinary-input checks,
93 feature tests, 332 metadata tests and 162 custom-gradient tests. Saved models
are byte-identical for native/host inputs separately on CPU and MPS; the tests
do not require different CPU/CUDA sketch algorithms to produce identical trees.
Regeneration of cuts at different bin counts, categorical/vector fits, DART,
linear coordinate descent, serialization, duplicate row slices, iterator source
release/cache cleanup and CPU fallback margins/iteration ranges are covered.
The first focused run exposed a real missing CPU fallback route, which was fixed.
Tests also now assert two existing upstream restrictions explicitly: ordinary
external matrices cannot export a single CSR page, and vector-leaf training
requires `hist`. No numeric tolerances were widened. No new benchmark, NVIDIA
execution or other Apple-device comparison.
[Validation record](../validation/cuda-metal-ordinary-tensor-m3.txt).

Final full regression passed on 2026-11-04, **21:02:23Z–21:09:34Z**:
**3,139 Python tests passed, one skipped**, all 82 rebuilt native check groups
and the CPU-only build/API/device/cache checks. The non-Metal constructor rejects
explicitly and leaves its output handle unset. No production/test changes followed
the final build/run; documentation was finalized afterward.


## Native columnar tensors, masks and dictionaries

CUDA authority: `data/device_adapter.cu::CudfAdapter`,
`device_adapter.cuh::EncCudfAdapterBatchImpl`, `data/columnar.h`,
`data/cat_container.cu`, and `predictor/gpu_predictor.cu::DispatchedInplacePredict`.
The table has independently typed one-dimensional columns, equal row counts,
packed validity masks, optional numeric/string dictionaries and optional reference
categories. Missing values stay missing before category-map application. Numeric
category names support signed/unsigned 8/16/32/64-bit integers; floating names
reject as upstream does. String names use int32 offsets and signed int8 bytes.

The existing native C entry points now also accept a `columns` descriptor.
Python exposes `MetalColumnar` and `MetalColumn`, retaining managed DLPack exports
and reference-category handles across iterator resets without re-consuming
aliased capsules. PyTorch/MLX synchronize before each snapshot; generic producers
must supply ready buffers. Feature names/types are inferred, with explicit caller
overrides and categorical enablement checks. Metadata and gradient entry points
reject tables, retaining their tensor-only contract.

Metal snapshots the addressed strided column/mask spans and consolidates them
with GPU blits. The existing heterogeneous decoder, missing filters, sparse
packing, quantile windows and prediction consumers reuse this buffer directly.
Ordinary and quantile construction, both external cache modes, reference-category
recoding and CPU tree inplace fallback are connected. Dictionary gathers preserve
exact integer/string bytes on GPU, using guarded shared output views or a counted
copy fallback. Dictionaries then use the existing owned host container and GPU
sort/recode kernels. They still incur host container copies and uploads in the
existing dictionary packer; persistent GPU dictionary ownership is **not** complete.
Consolidation currently adds a GPU copy; performance tuning remains deferred.

Tests found a preexisting shared CPU `CatAccessor` defect on this ARM machine:
NaN-to-int conversion could yield zero, incorrectly recoding a missing category.
An explicit NaN guard now preserves missing values before casting. This removes
undefined C++ behavior rather than requiring Metal to reproduce it. Tests compare
missing rows with an independent validity mask and exercise both CPU and MPS
predictions. Actual NVIDIA NaN conversion/execution has not been measured.

New counters are `columnar_imports`, `columnar_pack_bytes`,
`columnar_gpu_time_ns`, `category_import_dispatches`,
`category_import_shared_bytes`, and `category_import_copy_bytes`. The first
three cover table consolidation and GPU time (including name gathers); dictionary
byte counters distinguish shared output from CPU payload copy. Existing tensor
counters track source snapshots. CPU-only builds report zero. Every native
statistics consumer must rebuild after the structure grows.

Native tests cover private sources, overwritten/released producers, all eight
mask bit offsets, negative/zero strides, partial flattened windows, padded
prediction features, exact names of every integer width, malformed masks/string
offsets and CPU NaN preservation. Python tests cover real PyTorch and MLX,
heterogeneous float16/float32/integer/bool columns, integer names beyond FP64
precision, UTF-8/embedded-NUL names, reordered/unknown dictionaries, references
across empty/nonempty iterator batches, CPU fallback, saved models and exact-once
aliased legacy/versioned capsule deletion. Native/host inputs produce identical
model bytes for the tested CPU and MPS fits separately. Schema names are matched
before byte comparison (`q` and `float` serialize differently).

Final validation passed on 2026-11-04, **21:22:27Z–21:30:07Z** (7m40s).
The new focused suite passes **62 Python tests**; native validation passes
**48 cases / 9,263 field checks / 14 rejected descriptors**. Full regression passes
**3,201 Python tests, one skip** (617 warnings, 174.10s), all **83 rebuilt native
check groups** and the CPU-only build/API/device/cache checks. No production or
test changes followed the final build/run; documentation was finalized afterward.
[Validation record](../validation/cuda-metal-columnar-m3.txt).
No new performance benchmark, NVIDIA execution or other Apple-device run. GPU
outputs, generic metadata Slice/Extend and state ownership, remaining
source-arithmetic audits and GPU collectives remain open.


## Owned Metal prediction outputs and DLPack consumers

CUDA authority: `c_api/c_api.cu::InplacePreidctCUDA` selects the proxy's device
and returns `ConstDevicePointer()`. Python `_prediction_output` and
`_data_utils.from_array_interface` then make an independent CuPy copy by default.
The ordinary batch C API returns a host pointer upstream. This fork adds an
explicit `output_device="mps"` option to both batch and inplace prediction;
existing defaults remain NumPy for host/Metal inputs and CuPy for CUDA inputs.
`output_device="cpu"` explicitly selects NumPy. Output storage selection does not
change the booster device or prediction algorithm.

`xgb.MetalOutput` owns a private Metal buffer and exposes shape, size, ndim, float32
dtype and the DLPack producer protocol. PyTorch and MLX consume it on GPU without
host conversion. Values, margins, leaves, SHAP contributions and interactions
retain their public shapes, including empty dimensions. Upstream's strict leaf
shape restriction for vector trees still rejects. `.numpy()` is an explicit
host readback; implicit NumPy conversion rejects. The allocation survives model
and matrix deletion, cache updates and the original result object's destruction
while consumers still reference it. GPU output can feed native feature, metadata
and custom-gradient inputs again.

Every export has an independent managed descriptor retaining shared ownership.
Legacy and v1.0 versioned capsules are supported. Default/`copy=False` exports
share writable storage; writes are visible to other consumers. `copy=True` makes
an independent GPU snapshot and sets the versioned IS_COPIED flag. An unconsumed
capsule frees its descriptor; a consumed one is renamed and its consumer invokes
the deleter. Native capsule names and a native destructor avoid Python callback
and temporary-name lifetimes during interpreter shutdown. The optional bridge
resolves CPython's capsule API at runtime, without linking a Python version.
Native clients use the standard DLPack managed-tensor deleter directly. The
unmodified DLPack v1.0 ABI header and Apache-2.0 license are vendored under
`src/common/dlpack`.

Production snapshots complete synchronously. Loaded PyTorch/MLX runtimes also
synchronize before re-export/readback, so known consumer writes are visible.
Other consumers must complete writes themselves. Only `stream=None` and Metal
device 0 exports are accepted. There is no portable Metal queue handshake here,
no CUDA-to-Metal transfer and no asynchronous lifetime guarantee for arbitrary
foreign work submitted after export. These are explicit platform boundaries,
not evidence of CUDA stream parity.

Tree traversal, SHAP, scalar/AFT transforms and row transforms now bind eligible
writable host spans directly through guarded unified-memory views. Ineligible
spans use counted upload/readback fallbacks. Argmax retains a GPU snapshot before
compacting row outputs; overlapping SHAP margins are also snapshotted on GPU to
avoid read/write races. Existing numerical kernels and tolerances are unchanged.
The final result is copied to private GPU storage before the next C API call can
reuse the prediction pointer. Shape metadata alone is read by Python during
construction; output values do not pass through a NumPy array.

General `HostDeviceVector` state, cache copies, margin initialization and host
input/model preparation remain host-owned. This step removes payload copies at
the changed kernel bindings and output publication sites; it does not prove
that every operation in prediction or training has stopped accessing host data.
Linear prediction still runs on host as it does upstream; selecting Metal output
only controls its result allocation.

New counters: `prediction_shared_output_bytes`, `prediction_host_input_bytes`,
`prediction_host_output_bytes`, `prediction_device_copy_bytes`,
`output_snapshot_dispatches`, `output_snapshot_shared_input_bytes`,
`output_snapshot_host_copy_bytes`, `output_snapshot_bytes`, `output_gpu_time_ns`,
`output_host_readback_bytes`, `output_dlpack_exports`, `output_live_buffers` and
`output_live_exports`. Snapshot totals include requested GPU clones; source
shared/upload bytes refer to initial host-span import. Output GPU time covers
snapshot/clone commands, not explicit readback. Live counters check ownership;
other counters are cumulative and can count repeated bindings. CPU-only builds
report zero and reject output APIs clearly, leaving result handles unset. All
native statistics consumers must rebuild for the extended structure.

Focused validation passes 100 Python tests, 12 native output cases / 3,389 field
checks / 13 rejected calls, and the expanded 592-case SHAP suite with 425,539,584
analytic values. Native checks preserve exact float32 bit patterns, including
NaN payloads, signed zero and subnormals, and cover scalar through four-dimensional
shapes, empty outputs, source overwrite/release, real private-buffer GPU consumers,
copy independence, shared writes, guarded offsets, read-only source fallback,
unaligned readback, overflow and malformed C API calls. Python checks cover real
PyTorch/MLX consumers, all prediction modes, transforms, CPU/MPS storage choices,
host/native/columnar/CSR inputs, chunk boundaries, output-to-input chaining,
legacy/versioned deletion, threaded exports and isolated interpreter shutdown.

Final full regression passed on 2026-11-04, **21:45:27Z–21:53:51Z** (8m24s):
**3,301 Python tests passed, one skipped** (620 warnings, 181.46s), all **84
rebuilt native check groups**, the source reference updater, and the CPU-only
build/API/device/cache checks. No production or test changes followed the final
build/run; documentation was finalized afterward.
[Validation record](../validation/cuda-metal-output-m3.txt).
No new performance benchmark, NVIDIA execution or other Apple-device comparison.
Generic metadata Slice/Extend was the next gap, addressed below. Persistent
state/dictionary ownership, remaining source-arithmetic audits and GPU collectives
remain open.


## Metal vector storage and internal metadata operations

CUDA authority: `common/host_device_vector.cu`,
`common/device_vector.cuh::DeviceUVectorImpl`, `data/data.cu::SliceMetaInfo`,
`data/metainfo.cu::CheckFeatureTypes`, and shared `MetaInfo::Extend` /
`linalg::Stack` control flow. CUDA performs host copies when the relevant access
states already permit them; a CUDA ordinal alone does not imply device work.

`DeviceOrd::MPS()` now identifies Metal storage independently of the existing
execution-context selector. `host_device_vector_metal.cc` implements the same
three access states: host writable, both sides readable, or device writable.
Explicit device access creates/synchronizes an owned shared Metal allocation;
mutable host access invalidates the device copy, and device writes invalidate
the host mirror. Host readback is lazy and repeated reads do not copy again.
Metal device pointers are addresses in owned shared allocations, **not** the
MTLBuffer handles used by Metal DLPack descriptors.

GPU byte kernels fill POD payloads without floating-point conversion; GPU blits
copy, preserve growth prefixes and append arrays. Capacity survives shrinkage.
Like CUDA's DeviceUVector, device `Resize(n)` does not promise initialization of
new elements; `Resize(n, value)` fills them explicitly. Host resize retains
std::vector initialization. Constructors, initializer-list/host/device copies,
moves, device switches, CPU access and all existing explicitly instantiated
payload types are supported, including FP64 bytes, gradient pairs and tree
structures. CUDA-style source-device adoption and access-state branch choices
are retained. Shared buffers and synchronous commands are Metal-specific;
asynchronous CUDA stream behavior is not claimed.

Mixed host/native iterator metadata remains supported. A host source can append
directly into a device destination, and a device-only source can supply the
storage device for a CPU destination. Self-extension snapshots the original
extent before resizing; it avoids the upstream implementation's unsafe
post-resize self-size calculation. Overlapping native buffer copies use a GPU
temporary. These defined compatibility behaviors do not reproduce invalid
memory access. Bounds, overflow and unsupported storage ordinals reject.

Native float metadata imports now write directly into these allocations. Labels,
weights, base margins, survival bounds and feature weights retain device storage;
conversion/validation and existing metadata counters remain connected. Public
getters explicitly obtain host views, as upstream's GetInfo does. Replacing a
same-shaped GPU field reuses its allocation and invalidates its old host mirror.
Group boundaries remain host vectors as in CUDA.

Internal `MetaInfo::Slice` dispatches to Metal for an MPS execution context.
Labels and margins gather complete rows with all targets, and bounds/weights use
one-column gathers; empty fields remain empty. Row IDs may repeat or reorder.
Out-of-range IDs report an error before an invalid device read. The CUDA code
uses unchecked gathers for those invalid inputs; its undefined behavior is not
reproduced. `MetaInfo::Copy`/`Extend` and `linalg::Stack` retain Metal vector
storage where selected. Feature-type equality uses a GPU byte comparison when
either input has a valid device copy.

Public ordinary `SimpleDMatrix::Slice` explicitly makes a CPU context upstream,
even for CUDA-origin matrices. It remains host-side here and still preserves
native-origin metadata correctly. External-memory matrices continue to reject
row slicing. Those choices are source-defined behavior, not unported GPU work.
Ranking group-aware row slicing is not added; group orchestration remains the
higher-level upstream responsibility.

This step **does not** change the default storage returned by
`Context(device="mps").Device()` from CPU to Metal. That change requires migrating
the remaining generic consumers and dispatchers. Existing consumers can still
request HostSpan/HostView and cause an explicit mirror readback; training
predictions, gradients, model/dictionary storage and other state are not all
resident. The new storage path makes that migration possible and already serves
native metadata, but it is not a claim of end-to-end residency or speedup.

Fourteen counters report vector allocations/live buffers/capacity, copy and fill
commands/bytes, host-to-device import bytes, explicit readback bytes, CPU payload
fallback bytes, GPU time, metadata gathers/values and feature-type comparisons.
Capacity is the sum of owned vector buffers, not whole-process or peak GPU memory.
Copy totals include synchronization and growth; commands are synchronous. Existing
metadata shared/copy counters still describe the conversion destination binding.
CPU-only builds report zero. Native statistics consumers must rebuild again.

Focused validation passes **83 native cases / 1,432,716 exact fields / 33 rejected
calls**, covering access transitions, repeated access without transfers, source
and destination copy states, float bit patterns, signed/unsigned widths, all
struct payload families, retained capacity, ownership/moves, mixed and self
extension, overlapping copies, metadata row oracles, shape preservation,
copy/stack residency, malformed requests and feature-type equality. The existing
**935-test** interoperability regression passes. **20 new Python tests** check
lazy metadata mirrors and allocation reuse, empty/host/native replacement,
external iterator concatenation with empty and mixed batches, model bytes,
public slicing restrictions, and binary metadata round trips with real Torch/MLX.

For model-byte comparisons, both inputs use identical feature-page boundaries.
The first fixture compared external pages with one in-memory matrix; weighted
CUDA-derived sketch merging can select different cuts across those layouts.
The corrected experiment isolates metadata storage without changing values or
numeric tolerances. Other fixture corrections preserve matrix metadata shapes
and explicitly assert external-memory slicing rejection. A first shader build
also exposed mismatched scalar/vector thread-grid attribute types, which were
corrected before native execution passed.

Final full regression completed **2026-11-04 22:10:53–22:19:44 UTC** (8m51s):
**3,321 Python tests passed, one expected skip** (620 warnings, 185.45s), all
**85 rebuilt native check groups**, the source reference updater, and the
CPU-only build/API/device/cache checks. No production or test changes followed
the final build/run; documentation was finalized afterward.
[Validation record](../validation/cuda-metal-vector-state-m3.txt).
No new benchmark, NVIDIA execution or other Apple-device comparison.
Remaining work includes migrating generic allocations
and their consumers, persistent dictionary ownership, remaining CUDA arithmetic
and operation audits, GPU collectives and cross-hardware validation.

## Objective inputs, gradients and transforms retain Metal storage

CUDA authority is the existing objective `.cu` implementations and
`HostDeviceVector`/`linalg::Tensor::View` access rules. The previous storage step
made a Metal ordinal available; this step connects it to regression,
classification, absolute/quantile and AFT objective calls. It does not alter
shader arithmetic or replace CUDA formulas with CPU formulas.

The Metal branches run before generic `SetDevice(CPU)` or label/gradient host-view
access. Predictions, labels, weights, survival bounds and expectile/quantile
levels use explicit const Metal spans; gradient matrices select Metal storage
before reshaping and writable access. Empty multiclass inputs still return early,
as upstream does. Repeated calls reuse valid input allocations and the output
buffer. CPU callers can subsequently access these objects and Metal callers can
reuse them afterward. Mutable device access invalidates the host mirror.

Regression and multiclass label validation consumes a device tensor view.
Sigmoid/exponential/hinge/softmax/argmax/expectile/quantile and AFT transforms use
writable device spans, preserving shape and argmax compaction. Identity
transformations remain no-ops arithmetically. Checked logistic/gamma and logarithmic
base-margin conversion uses guarded input binding, GPU scratch and a GPU blit
into the destination after successful validation. An invalid checked intercept
leaves every payload byte unchanged. AFT margin transforms already support
shared output binding and now receive device storage. Cox gradients remain on
the host as in upstream CUDA; its shared logarithmic margin helper is migrated.
Expectile's host margin conversion remains upstream-defined host work.

Adaptive and AFT gradient wrappers bind eligible inputs and outputs directly.
Host native-API callers remain supported. Read-only/unshareable inputs use a
counted upload; overlapping or insufficiently aligned outputs use a GPU scratch
result followed by the counted host compatibility copy. Metal float2/uint2 output
binding requires eight-byte alignment even though host GradientPair alignment is
four. All commands complete synchronously. Adaptive residual statistics still
cross the small host collective that the CUDA implementation explicitly uses;
that is separate from a full prediction/label/gradient payload readback.

Existing objective shared-input/shared-output/staging counters now also cover
these adaptive/AFT bindings, label validation and base-margin conversion. They
count bindings/fallback payload bytes, not whole-program transfers, GPU temporary
capacity or the small status/collective messages. Base-score dispatches still
count transformation requests; its GPU time includes the successful publication
blit. The statistics ABI is unchanged.

Validation adds **384 native cases / 1,003,052 field checks / eight rejected calls**.
Actual ObjFunction calls cover all named nonranking gradient families and all
three AFT distributions, zero/single/multiple rows, scalar/multiple targets,
weights, repeated calls, output ownership and CPU/MPS handoffs. CPU comparisons
use the existing 3e-6 scaled tolerance for well-conditioned fixtures. Independent
existing CUDA-expression and sensitive-tail oracles remain the arithmetic
checks. Counters and access flags require no implicit vector readback or payload
staging during resident calls, no repeated uploads/allocations, and explicit host
access afterward. Invalid labels are written on GPU over a stale host mirror;
invalid checked margins preserve their bytes. Direct binding cases compare exact
result bits across disjoint, overlapping, read-only and four-byte-aligned output
storage with output guards. Existing native objective/adaptive/AFT/base-score
and host-memory checks pass, as do **619 focused Python regressions**.

A first new test failed to compile because a templated Context update needed an
explicit Args container; the fixture was corrected. The focused shell driver
also omitted the existing adaptive test's communicator JSON argument; it resumed
with that argument without rerunning the already-passed objective oracle. Review
caught the Metal eight-byte output requirement before the final run; it is now
covered by a dedicated four-byte-aligned host-output fixture. No shader formulas
or tolerances were changed to obtain passing results.

Final full regression completed **2026-11-04 22:27:29–22:35:02 UTC** (7m33s):
**3,321 Python tests passed, one expected skip** (621 warnings, 191.53s), all
**86 rebuilt native check groups**, the source reference updater, and the CPU-only
build/API/device/cache checks. The skip is the unavailable-device test on a host
with an accessible GPU. No production or test changes followed the final build/run;
documentation was finalized afterward.
[Validation record](../validation/cuda-metal-objective-state-m3.txt).

This is not end-to-end training residency:
base-score estimation/stump and other initialization consumers, ranking, metrics,
tree/prediction state and public output plumbing still have host accesses. In
particular, the existing C prediction result API obtains a host vector before
an explicit MetalOutput snapshot. Those paths require their own migration;
Device() still defaults to CPU and must not be flipped without the remaining
dispatcher audit. Persistent dictionary ownership, other CUDA operation/arithmetic
audits, GPU collectives and cross-hardware validation remain open.
No new benchmark, NVIDIA execution or other Apple-device comparison.


## Initialization and stump payloads retain Metal storage

CUDA authority is `common/stats.cu`, `common/numeric.cu`, `tree/fit_stump.cu`
and the objective initialization methods in `init_estimation.cc`,
`multiclass_obj.cu`, `regression_obj.cu` and `quantile_obj.cu`. This step connects
the previously implemented statistics/initialization kernels to Metal vector
storage. It does not change their reduction, sorting, quantile, gradient or
stump arithmetic.

`Mean`, `SampleMean`, `WeightedSampleMean`, `Median`, generic `Reduce` and
`tree::FitStump` select explicit MPS input/output storage before generic host
accesses. Scalar FP32 mean input strides are retained. Generic sums still return
host scalars, including the source-defined FP32 batching/aggregate boundary;
optional-weight sums still use binary64. Mean denominator/error scalars and
class-histogram validation flags remain host checks. These are distinct from
full-array readbacks. GPU collective support remains a separate open requirement;
this step validates single-worker initialization.

The lower bindings guard eligible unified-memory views. Unshareable/read-only
inputs use counted uploads; gradient reduction inputs require eight-byte Metal
float2 alignment even though C++ GradientPair permits four. Offset input views
are bound directly, or copied on GPU into offset-zero buffers for sort helpers.
Results are computed into GPU scratch and published with a byte-copy kernel that
preserves FP32/binary64/POD representations, output strides and padding. Empty
means/stumps use GPU zero fills; empty quantiles use GPU NaN fills. Publication
after computation preserves aliased input/output behavior. Class histograms
snapshot their existing backing storage on GPU; validation failures leave the
original output unchanged. Host output compatibility uses a counted copy only
when direct sharing is unavailable. All commands complete synchronously.

GLM-like and Newton-based base-score estimation now constructs/consumes device
storage. Multiclass priors retain device labels, weights, bins, normalization,
logs and the mean scalar; a new GPU centering kernel performs the existing FP32
subtraction without reading that scalar on CPU. It snapshots an overlapping mean
on GPU before updating the vector. Quantile base-score normalization keeps its
device vector. Expectile initial gradient/Hessian construction and absolute-error
target broadcasts feed resident gradient/stump stages. CUDA's explicit host
expectile mean/final monotonic projection and absolute-error target-mean addition
remain on host. The tests account for exactly those small readbacks.

Seven initialization payload/copy counters are appended to the native statistics
structure and public JSON (zero in a CPU-only build). Shared input/output and
host input/output fields count binding/fallback bytes, not unique allocations or
whole-program traffic. Copy dispatches/bytes include result publication and
packed/alias snapshots; fill dispatches cover empty-result initialization. Their
GPU duration contributes to `initialization_gpu_time_ns`, while existing reduction
and mean durations retain their separate scopes. Control parameters, small
status checks and source-defined scalar returns are excluded. Native clients
must be rebuilt for the enlarged statistics structure.

Validation adds **250 native cases / 3,625 field checks / three rejected calls**
with Metal API validation. Actual initialization covers 14 objective families,
weighted/unweighted data, scalar/multiple targets and multiple row counts. Public
statistics/stump calls cover empty/single/33/257/4097 rows, repeated calls, explicit
CPU/MPS handoffs, output ownership and no unexpected readback/staging. Strided
FP32/precise-pair output guards, NaN/zero fills, offset sorting, aliased mean and
quantile outputs, overlapping centering scalars, read-only inputs and four-byte
aligned gradient input fallbacks are checked. Center subtraction is checked
bitwise for signed zeros, subnormals and infinities (NaNs by classification).
Invalid weighted means and invalid/overflowing class histograms preserve output.
Final live vector gauges return to zero.

The first new fixture needed OptionalWeights' explicit scalar constructor. Its
initial direct CPU comparison for a 4,097-row FP32 mean also exposed the expected
accumulation-order difference: the CPU's longer serial path rounded further from
the independent sum of FP32 terms. That fixture now bounds each backend's error
independently by its accumulation depth. Existing numerical kernels, CPU/source
oracles and test tolerances were not changed. Other well-conditioned CPU fixtures
retain the 3e-6 scaled tolerance; exact dyadic and copy tests remain exact.

Final full regression completed **2026-11-04 22:44:49–22:52:54 UTC** (8m05s):
**3,321 Python tests passed, one expected skip** (622 warnings, 191.79s), all
**87 rebuilt native check groups**, the source reference updater, and the CPU-only
build/API/device/cache checks. The skip is the unavailable-device test on a host
with an accessible GPU. No production or test changes followed the final build/run;
documentation was finalized afterward.
[Validation record](../validation/cuda-metal-initialization-state-m3.txt).
Remaining ownership work includes ranking, metrics, predictor/updater consumers,
public prediction result plumbing and persistent dictionaries. The execution
context still defaults generic state to CPU and must not be switched globally
before those consumers are audited. Other CUDA operation/arithmetic audits, GPU
collectives and cross-hardware validation remain open. No benchmark, NVIDIA
execution or other Apple-device comparison was run.


## Scalar metric inputs retain Metal storage

CUDA authority is `metric/elementwise_metric.cu`, `multiclass_metric.cu` and
`survival_metric.cu`: row expressions run on the GPU, the reduction returns a
pair of binary64 scalars, and the caller performs the host collective and final
ratio. This change preserves those formulas, cast boundaries, reduction trees,
scalar returns and collective placement. No numerical shader changed.

Scalar/pseudo-Huber Metal branches now run before eager `SetDevice(CPU)` and
host label/prediction/weight views. Scalar/multiclass, quantile/expectile and AFT
likelihood/interval-accuracy callers use explicit Metal spans for predictions,
labels or survival bounds, weights and quantile/expectile levels. The associated
`MetricSums`, `QuantileMetricSums` and survival metric encoder bind eligible
shared storage directly with its byte offset. Read-only or unshareable host
inputs retain a counted upload fallback. Commands remain synchronous and do not
modify inputs. Only the existing 16-byte reduced result returns to the host;
validation flags remain small scalar checks. The AFT objective binding path and
its accounting are unchanged.

The audit also found that the quantile/expectile Metal branches bypassed
`CheckRowWeights`, which CPU and CUDA invoke through their shared reducer. Those
branches now apply the same check before dispatching. In particular, one-row
queries with a query weight for every row must be rejected even though their
weight-array length matches the row count. Empty-worker behavior and nonpositive
aggregate-weight errors retain their existing upstream placement.

`metric_shared_input_bytes` / `metric_host_input_bytes` are appended to the native
statistics structure and exported in the public JSON, with zeros in CPU builds.
They count eligible payload binding bytes and fallback upload bytes for these
scalar metric families; AUC/ranking input paths are not yet included. They exclude
control arguments, final scalar results and validation flags. Existing
`metric_final_readback_bytes`, dispatch/reduction counts and timing scopes remain
unchanged. All native clients must rebuild for the enlarged statistics structure.

New validation passes **587 native cases / 44,478 field checks / 110 rejected
calls** with Metal API validation. Actual Metric evaluators cover 15 scalar,
multiclass and ordered metric names, both survival metrics with all three AFT
distributions, empty/single/33/257/1025 rows, scalar/multiple targets and absent,
positive or zero weights. Well-conditioned CPU comparisons retain the 3e-6
scaled tolerance; existing independent CUDA-source and sensitive-tail oracles
remain the arithmetic authority. Tests require no input mirror or fallback
staging for resident arrays, exactly 16 bytes of final result for nonempty
calls, unchanged input access flags and no repeated vector uploads/allocations.
GPU fills followed by CPU/MPS handoffs check updated inputs. Empty/zero-weight
ordered metrics and malformed weights are rejected; invalid multiclass labels
are written on GPU and rejected by GPU validation without host mirrors.

Thirty-three direct binding cases compare exact result bits across offset shared
views and independent read-only fallbacks for predictions, labels/bounds, weights
and levels. Input payloads and guards remain unchanged, byte counters match the
specific inputs, and final live vector gauges return to zero. **139 focused
Python tests pass**, including four new CPU/MPS query-weight rejection cases and
public binding-counter assertions for scalar/multiclass/ordered metrics.
No test tolerances or arithmetic kernels were changed to obtain these results.

Final full regression completed **2026-11-04 22:57:39–23:05:21 UTC** (7m42s):
**3,325 Python tests passed, one expected skip** (623 warnings, 191.15s), all
**88 rebuilt native check groups**, the source reference updater, and the CPU-only
build/API/device/cache checks. Existing metric oracles passed **349 cases /
15,628,618 exact fields**, with maximum reduction-bound ratio 0.0373374008;
288 quantile/expectile cases covered 1,727,040 loss/weight pairs. The skip is the
unavailable-device test on a host with an accessible GPU. No production or test
changes followed the final build/run; documentation was finalized afterward.
[Validation record](../validation/cuda-metal-metric-state-m3.txt).
Remaining ownership work includes AUC/ranking metrics and ranking objectives,
predictor/updater consumers, public result plumbing and persistent dictionaries.
Context.Device() still defaults generic state to CPU until those dispatchers and
consumers are migrated. Remaining CUDA operation/arithmetic audits, GPU
collectives and cross-hardware validation are unfinished. No benchmark, NVIDIA
execution or other Apple-device comparison was run.

## AUC inputs and summaries retain Metal storage

Authority: `metric/auc.cu`, especially `MultiAUC` and `ScaleOutputs`. CUDA retains
multi-output summaries through its GPU collective and normalization, whereas
binary and grouped curves return small scalar results. The previous MPS path
requested host inputs, copied summaries to host and uploaded them again. The
single-worker path now uses explicit Metal prediction/label/weight views and an
owned double vector for summaries. The CPU path and numerical kernels are unchanged.

Column sorting and all three grouped-sort paths accept byte offsets, so strided
shared input views need no packing copy. Labels, weights and normalization
summaries bind with their own offsets; read-only/unshareable inputs use counted
uploads. AUC computes into scratch and publishes to eligible output spans by GPU
blit after all input reads, preserving aliased output semantics. An unshareable
output uses counted host publication. Optional native trace/individual-query
readbacks remain diagnostics, not production transfers.

Empty local multi-output workers now skip the irrelevant label-column check and
produce zero summaries before aggregation, matching CUDA. Global branch/shape
metadata remains host collective work. Distributed multi-output evaluation still
materializes **only the summary vector** for the existing CPU collective, then
uploads it for GPU normalization. This explicitly remains a transport gap versus
CUDA; no distributed MPS learner support is enabled by these metric tests.

`metric_shared_input_bytes` / `metric_host_input_bytes` now include these AUC
families and summaries. New shared/host output and GPU-copy counters expose summary
publication; `auc_final_readback_bytes` counts 8-byte normalization and 16-byte
grouped returns. Raw binary triples are counted as output publication. Diagnostics
and scalar validation flags are excluded. All native clients must rebuild for
the five appended statistics fields; the public CPU build exports zeros.

With Metal API validation, the new actual-evaluator suite passes **172 cases /
392,552 field checks / five rejected calls** on one worker. It checks input access
states, repeats, GPU mutation and CPU handoffs, zero/absent/positive weights,
empty/degenerate data and GPU-written invalid ranking labels. Exact binding tests
exercise nonzero offsets, row/column strides, independent read-only fallbacks,
aliases in every input slot, unaligned output fallback and unchanged guards.
Grouped tests cover small/medium/radix sorting. Final vector allocation gauges are
zero. **96 cases per worker** pass with two and three local workers, including
empty workers with and without target metadata and all-empty datasets; assertions
allow exactly the summary-only collective transfer. The existing **170 public
AUC tests** pass with new binding/copy counters. No tolerance was widened.

Full regression completed **2026-11-04 23:12:24–23:20:15 UTC** (7m51s): **3,325
Python tests passed, one expected skip** (621 warnings, 191.32s), all **89 rebuilt
native check groups**, the source reference updater and CPU-only build/API/device/
cache checks. Existing source arithmetic, cancellation, cast-boundary, prefix and
independent pair-counting oracles all passed unchanged. No production or test
changes followed this final build/run; documentation was finalized afterward.
[Validation record](../validation/cuda-metal-auc-state-m3.txt).
Next ownership work is precision/NDCG/MAP caches and metrics, ranking objectives,
predictor/updater consumers, public result plumbing and persistent dictionaries.
CUDA operation/arithmetic audits, GPU collectives and NVIDIA/other-Apple hardware
validation remain unfinished. No benchmark was run for this step.

## Ranking metrics and caches retain Metal storage

CUDA authority: `common/ranking_utils.cu` and `metric/rank_metric.cu`. Precision,
NDCG and MAP already had GPU arithmetic; their consumers and cache accessors still
requested host inputs and cached arrays. This change retains explicit Metal
prediction/label/query-weight spans, NDCG discounts/inverse ideal gains and sorted
indices. Cache construction selects Metal before allocating index arrays; otherwise
its first use uploads a CPU-allocated array even when the sorter itself is on GPU.
The new test detected that allocation gap, which is now fixed.

Label validation, grouped sorting, ideal-gain initialization and ranking score
wrappers bind guarded shared spans with their offsets. Sort/cache results compute
into scratch, then publish by GPU blit after all input reads. Read-only inputs and
unaligned/unshareable outputs retain counted compatibility paths. Overlapping
NDCG discount/inverse output spans use ordered host publication so both shared
and mixed-eligibility destinations preserve the previous discount-then-inverse
write order. Production caches have disjoint owned allocations. Optional-weight
normalization retains its source scalar return; query metadata remains host
input to the existing owned GPU plan. Final metric score/weight scalars and host
collectives retain upstream placement. No arithmetic kernel or tolerance changed.

Existing input/publication counters now include ranking metrics, validation and
cache/index arrays. `metric_final_readback_bytes` includes their 16-byte final
score/weight return for calls with at least one query, even when its rows are
empty. No statistics fields or ABI changes were needed for this extension.

New native validation passes **166 cases / 47,135 checks / three rejected calls**
with Metal API validation. Actual metric `Evaluate` calls use real matrix caches
and test first use/reuse, NDCG gain-parameter reset, GPU prediction writes,
CPU/MPS handoffs and GPU-written invalid labels. Tests require no input mirrors,
no vector reuploads, no metric/cache staging, exact final-result/publication byte
counts and no repeated allocations. Direct cache tests verify accessor devices,
CPU reference values and sort indices. Binding tests cover small/medium/radix
sorts, row strides, every read-only score-input slot, cache-output aliases,
unaligned fallbacks, overlapping outputs and unchanged guards. Final vector live
and capacity gauges return to zero. Empty Precision uses CUDA's zero contribution;
NDCG/MAP retain their source fallback rules rather than the CPU's divergent edges.

All **277 focused Python ranking/AUC tests** pass (27 warnings, 7.28s), including
existing LambdaRank training and cache lifetime checks and new public binding
assertions. Full regression completed **2026-11-04 23:27:55–23:35:45 UTC** (7m50s):
**3,325 Python tests passed, one expected skip** (622 warnings, 190.60s), all **90
rebuilt native check groups**, the source reference updater and CPU-only build/
API/device/cache checks. The existing ranking tests passed 896 score cases,
336 NDCG cache cases, 2,444,200 exact gradient pairs and 2,592 unbiased calls.
No production/test edits followed the final build/run; docs were finalized afterward.
[Validation record](../validation/cuda-metal-ranking-metric-state-m3.txt).
LambdaRank gradients/bias arrays still stage through host-owned consumers;
GPU bias calculations already exist, but their payload migration is next. Other
remaining work includes predictor/updater state, result plumbing, dictionaries,
source coverage audits, GPU collectives and cross-hardware validation. No benchmark
or NVIDIA execution comparison was run.

## LambdaRank gradients and bias retain Metal storage

CUDA authority: the shared `objective/lambdarank_obj.cc` lifecycle and
`objective/lambdarank_obj.cu` gradient/bias state. Pairwise, NDCG and MAP now pass
explicit Metal prediction, label, query-weight, gradient and bias spans through
the existing GPU ranking implementation. Bias vectors are initialized with GPU
fills. The NDCG caches migrated in the preceding step remain resident here.
`SaveConfig` is the explicit host boundary and retains upstream F32 serialization
of the double bias arrays. Reading that mirror does not force a reupload on the
next gradient call.

The lower wrapper reuses guarded objective bindings, including byte offsets for
all immutable inputs and eight-byte alignment for doubles. Bias kernels mutate
their input, so GPU snapshots preserve aliased input reads until computation
finishes. Gradients and both updated bias arrays publish from scratch by GPU blit.
Overlapping output/diagnostic spans use ordered host publication, preserving the
previous gradient, diagnostics, positive-bias, negative-bias write order even
with mixed sharing eligibility. Read-only inputs and unshareable destinations
retain counted fallback paths. Production objective arrays are disjoint owned
buffers. Existing `objective_shared_input_bytes`, `objective_shared_output_bytes`
and `objective_staging_bytes` now include ranking; no statistics ABI growth.

The audit also found an initialization difference. CPU/CUDA use persistent row
costs to determine whether bias has been initialized. Metal fuses that scratch
inside its gradient command and previously checked only whether bias was empty.
A fresh objective loading saved bias consequently preserved it at iteration zero
and accepted starting later; CPU/CUDA instead reset it at zero and reject starting
that state later. Metal now records the logical scratch shape and follows the
source rule. An already initialized objective can still load bias and continue;
its first call imports the serialized F32 values once. CPU handoff allocates row
scratch lazily without resetting learned bias. Reinitialization replaces the old
shape, including empty-row state after device handoffs.

New native tests pass **394 cases / 85,031 checks / 44 expected rejections** with
Metal API validation. Actual objective calls cover all three losses, top-k and
mean sampling, biased/unbiased training, absent/positive/zero weights, empty to
257-row data, repeated calls, GPU prediction writes, exact binding byte counts,
serialization/reloads and CPU/MPS handoffs. Resident calls require no host mirrors,
vector imports or objective/cache payload staging, with no repeated owned-vector
allocation. An independent scalar CUDA-source reference checks gradient and bias
values; existing tolerances are unchanged. CPU comparisons are restricted to
compatible top-k behavior. CPU MAP's empty-query implementation indexes the first
row unconditionally, so it is not used as an empty-query reference; Metal's empty
state and valid nonempty CPU handoffs remain covered.

Direct bindings cover small/medium/radix sorting through 8,197 rows, strided labels,
offsets, every immutable input slot with read-only memory, unaligned outputs,
input/output aliases, overlapping gradient/bias destinations and diagnostic
publication order. Guards and vector lifetime gauges are checked. The existing
524,288-row batched unbiased tests now also assert resident input/output state
and no host transfers. Their host prediction mutations reacquire writable access
instead of retaining an old host reference across device calls.

Full regression completed **2026-11-04 23:48:34–23:56:22 UTC** (7m48s): **3,325
Python tests passed, one expected skip** (620 warnings, 190.13s), all **91 rebuilt
native check groups**, the source reference updater and CPU-only build/API/device/
cache checks. Existing ranking oracles retained 2,444,200 exact gradient pairs,
2,592 unbiased calls with zero serialized-bias error, and four exact 524,288-row
batched runs with no measured host transfers. No production/test edits followed
the final build/run; documentation was finalized afterward.
[Validation record](../validation/cuda-metal-ranking-gradient-state-m3.txt).
No arithmetic kernels, pair construction, RNG consumption, reduction order or
tolerances changed. This does
not establish end-to-end training residency: predictor/updater consumers, result
plumbing, persistent dictionaries, other source audits and GPU collective
transport remain open. No benchmark or NVIDIA execution comparison was run.

## Prediction caches and outputs retain Metal storage

CUDA authority: `predictor/gpu_predictor.cu`, `predictor/predictor.cu`, and the
shared predictor/GBTree/learner lifecycle. The existing Metal traversal and SHAP
kernels now receive explicit Metal output arrays from their callers. Prediction
initialization selects Metal before allocation, copies resident base margins on
GPU, fills scalar bases on GPU and reuses `BroadcastTargets` for vector bases.
Published model base scores retain both CPU and Metal read mirrors, preserving
the source's concurrent prediction contract.

Batch, inplace and leaf prediction no longer request host vectors before entering
the Metal path. GBTree retains prediction storage through training cache updates;
leaf IDs bind directly on Metal and the existing leaf-addition kernel updates
strided target slices. Learner raw/transformed prediction copies and evaluation
copies preserve that storage. SHAP contributions/interactions use resident output
and base-margin views while retaining source host model/quadrature preparation.
CPU consumers copy the host values of shared Metal base margins without inheriting
their device placement. Complete prediction caches also follow CPU/MPS device
changes when neither initialization nor traversal runs. This preserves CPU
expectile transforms and raw Metal outputs across repeated device switches.
Cache version/range/DART rules and prediction arithmetic are unchanged.

Guarded prediction input bindings now include raw traversal/SHAP features and
leaf IDs/weights. Read-only/unshareable arrays retain counted staging. If these
inputs overlap outputs, GPU snapshots preserve original reads before direct
publication. Existing SHAP margin snapshots use the same helper. Output fallbacks
retain initial values and strided gaps. `prediction_host_input_bytes` accounts for
these fallback inputs, `prediction_shared_output_bytes` includes leaf updates,
and `prediction_device_copy_bytes` includes their alias snapshots. No statistics
fields, ABI or arithmetic kernels changed.

New native validation passes **123 cases / 19,058,906 checks / four expected
rejections** with Metal API validation. Actual predictor calls cover scalar,
multiclass and vector leaves, tree weights, base margins, incremental batches,
native private-buffer inplace inputs, leaf outputs, SHAP/interactions and CPU
handoffs. Empty/one/33/65,537-row datasets exercise initialization and chunk edges.
GPU-written margins, zero-tree output resets and invalid margin shapes are checked.
Leaf-cache tests cover sampled-out negated positions and padded target slices.
Actual learners loaded from CPU-trained models verify cache increment/repeat/reset,
nonzero iteration starts, raw/transformed output and repeated evaluation. Resident
calls require device-authoritative outputs, no vector readback/import and no
prediction payload staging. Direct bindings check offsets, read-only inputs,
unaligned outputs, aliases, byte-exact publication and unchanged guards. Existing
SHAP reference tests now count feature-input snapshots as well as margin snapshots;
their numerical expectations and tolerances are unchanged.

Focused Python validation passes **75 tests**, including 12 new repeated cache
device-switch cases with expectile/logistic/softmax, zero/three rounds and
absent/present margins. These exercise prediction, raw output, evaluation and
inplace prediction; CPU calls must not dispatch Metal arithmetic. The initial
full regression exposed the shared-margin CPU handoff bug above and three stale
dispatch-count expectations. The latter now count vector base broadcasts into
fresh training/prediction caches separately from objective initialization. No
numerical expectations or tolerances were relaxed.

Final regression completed **2026-11-05 00:18:46–00:22:07 UTC** (3m21s): **3,337
Python tests passed, one expected skip** (622 warnings, 194.16s), plus the CPU-only
build and API/device/cache checks. All **92 native groups** and the source linear
reference updater were rebuilt successfully in the initial full run. After the
handoff correction, the changed prediction/learner checks were rebuilt and passed
again with the final counts above; unchanged native groups were not redundantly
rebuilt. No production/test edits followed the final build and focused checks.
[Validation record](../validation/cuda-metal-prediction-state-m3.txt).
At that checkpoint, the public CAPI result path still materialized a host mirror
before Python created an owned `MetalOutput`; the next section addresses it. Training gradient/updater ownership, persistent model and
dictionary storage, remaining CUDA audits, GPU collectives and cross-hardware
validation remain open. No benchmark or NVIDIA execution comparison was run.


## Public prediction results retain Metal storage

The legacy prediction, shaped DMatrix prediction and shared inplace result APIs
now preserve already-valid Metal output storage. `MetalVectorBuffer` allocates
CPU-addressable shared buffers and all its producers complete synchronously.
Returning that buffer's float address satisfies the existing C API contract;
it does not expose an opaque Metal handle or change the ABI. Empty arrays still
return null, including when a previous result left reusable allocation capacity.
The result remains owned by learner thread-local storage and must be copied before
reuse as before.

Publication checks both Metal placement and a valid device mirror. It does not
upload an existing host fallback just to return it. CPU/CUDA fallback access modes
are preserved, as are host-producing linear predictions and approximate
attributions. Ordinary NumPy output still makes an independent CPU copy. Native
`MetalOutput` uses its existing GPU snapshot to retain the result independently of
later predictions, learner destruction and foreign consumer writes.

Native validation passes **780 API cases / 187,757 checks** with Metal API
validation: eight model kinds, margins/no margins, legacy/shaped APIs, dense/native
inplace prediction, and 33/zero/one rows to exercise empty-after-reuse behavior.
Tests compare immediate float values and shapes against the same CPU-trained
models, inspect actual pointer identity and TLS access state, require no vector
readback for resident tree results and preserve host fallback authority. Every
result is GPU-snapshotted with exact binding byte counts and checked byte-for-byte
after model destruction. All output/export/vector lifetime gauges return to zero.
The upstream host linear interaction producer has a size inconsistent with the
common shaped API; it is excluded from these tests without reading beyond storage.
Linear raw and contribution fallbacks are covered. No linear arithmetic changed.

**153 focused Python tests pass** (151 output/prediction tests plus two concurrent
prediction cases). Existing PyTorch/MLX consumers now check implicit vector
readbacks as well as explicit output readbacks. Eight added cases verify NumPy
copy independence and CPU/Metal results surviving worker thread exit and model
release. Existing export alias/copy, external writes, GC, shutdown, empty and
65,537-row chunk tests remain intact. One-time model publication is warmed before
transfer measurements; source-defined base-score mirrors and explicit CPU output
copies are not claimed to disappear. Numerical tolerances are unchanged.

Full regression completed **2026-11-05 00:28:56–00:32:17 UTC** (3m21s): **3,345
Python tests passed, one expected skip** (623 warnings, 192.55s), plus the CPU-only
build and API/device/cache checks. The native runner now contains 93 groups;
this C API change rebuilds/tests its new group and runs the full Python regression
without redundantly rebuilding unchanged kernel groups. No production/test edits
followed the final build and focused checks.
[Validation record](../validation/cuda-metal-prediction-result-m3.txt).
Training gradient/updater arrays, persistent model/dictionary storage, remaining
CUDA arithmetic audits, GPU collective transport and cross-hardware validation
remain open. No benchmark or NVIDIA execution comparison was run.


### Tree-training gradient, sampling and leaf-position storage

CUDA authority: `gbm/gbtree.cu` group extraction, the per-tree gradient and sampling
lifecycle in `tree/updater_gpu_hist.cu`, and `GPUGlobalApproxMaker::Update` for
original-Hessian sketching before per-tree sampling. This change preserves those
already-ported algorithms and moves their payloads into owned Metal storage.

`GBTree::CopyGradient` and its scalar-group scratch now select MPS explicitly.
The hist updater consumes device views, allocates its F-order forest/multi-target
sampling buffer on Metal, and preserves the original value-gradient quantizer
before replay for reduced-gradient leaf fitting. Quantizer layout conversion also
uses a Metal tensor. `Sampler::ApplySampling` no longer requests a host mirror.
Float replay binds disjoint resident output elements directly; unshareable views
retain staged read/modify/write semantics, including padding and guards.

Both scalar and vector hist leaf-position arrays are Metal-owned. Approximate
training likewise extracts original Hessians into a Metal tensor, samples an
independent Metal gradient buffer per tree and publishes leaf positions on GPU.
Its builder consumes a matrix view instead of a host vector. The CPU path retains
its original gradient/Hessian handling and explicitly restores CPU positions on
a device handoff; an unreachable Metal branch in the CPU-only initialization
helper was removed.

Approximate sketching combines Hessians and row/query weights into a resident
output span. Overlapping input/output ranges compute into scratch and publish by
GPU blit only after reads complete. The sketch retains an independent GPU weight
snapshot, including `SetWeights`, so reuse/destruction of the caller's storage
cannot corrupt later input windows. The existing host-returning weight overload
remains for other callers. Empty, unweighted input stays unweighted.

The existing gradient-state byte counters now include sampling replay, unified
sketch weights and sketch-weight input snapshots; there is no statistics ABI
change. Source-defined scalar root statistics, host model orchestration, sketch
finalization and input metadata remain. This is not a claim that tree training
has no CPU work or that every host call should be removed. `Context::Device()`
still defaults to CPU for generic MPS consumers until their audit is complete.

`tests/metal_training_state_api.cc` passes **196 cases / 391,152 checks** with the
Metal debug layer. It checks actual hist/approx updater gradient/value-gradient
and leaf-position ownership, scalar/vector/reduced gradients, one/three-tree
forests, both samplers, missing values and repeated CPU/MPS handoffs. Leaf IDs
are checked by independently walking the finished public model on CPU. Group
extraction, quantizer layout scratch, replay and weight bindings require no
payload staging or vector transfers for resident arrays. C/F/padded replay
matches staged replay bit-for-bit, including guards; existing independent
CUDA-source oracles separately validate its arithmetic. Weight tests check
input/output aliases, read-only inputs, unaligned outputs, exact fallback byte
counts and snapshots surviving input GPU writes and allocation destruction.

Full regression completed **2026-11-05 00:42:43–00:50:37 UTC** (7m54s): all
93 existing native groups were rebuilt and passed, in addition to the new group;
**3,345 Python tests passed, one expected skip** (623 warnings, 189.59s), and the
CPU-only incremental build/API/device/cache checks passed. The runner now has
94 native groups. Review added a final autorelease scope to destroy command-held
borrowed buffers before the owning quantizer scratch tensor. After rebuilding,
the five affected native groups (training state, quantized state, CUDA gradient
sampling, gradient copies/Hessian extraction and integer leaf fitting) passed,
as did **223 approx/sampling Python tests** (one warning, 23.95s). This final
validation ran **00:50:41–00:51:19 UTC**, exit 0. No production/test edits followed.
[Validation record](../validation/cuda-metal-training-state-m3.txt).

Persistent model/dictionaries, remaining CUDA operation/arithmetic audits, GPU
collective transport and cross-hardware validation remain open. No benchmark or
NVIDIA execution comparison was run; full CUDA parity is incomplete.


### Categorical dictionary and recoding storage

CUDA authority is `data/cat_container.cu`: independent device copies, lazy host
publication, typed category views, sorted indices and owned recoding mappings.
The existing Metal comparison, merge-sort, schema validation and name-search
kernels are unchanged. This change ports the surrounding data lifecycle.

`CategoryDictionary` owns packed name bytes and kernel metadata in Metal vector
buffers. Packing computes offsets/descriptors on the host and copies payloads by
GPU commands; it no longer assembles a full CPU byte array. Integer widths,
signed-byte string comparisons, empty columns/strings, embedded nulls and unused
trailing string bytes are preserved. Cloning copies the owned GPU buffers and
rebuilds views referencing the clone. Explicit host publication performs a
tracked readback before reconstructing CPU category arrays.

`CatContainer` caches its immutable Metal dictionary, keeps feature segments and
sorted indices in Metal storage, and lazily materializes names for `HostView` or
serialization. MPS copies remain independent; CPU copies preserve CPU placement.
Reload invalidates the GPU snapshot and now clears prior columns before loading;
previously the shared loader appended columns when reusing an existing object.
The test-only `Push` helper also invalidates the snapshot. The generic execution
context still defaults to CPU; these consumers select MPS explicitly.

Recoding returns an owned `HostDeviceVector` mapping. Prediction, SHAP, native
DMatrix conversion and quantile adapter consumers keep that mapping alive through
synchronous GPU use. Sort/recode results publish by GPU copy into shareable output,
with guarded host fallbacks. Recoding computes into scratch and publishes only
after schema/missing-name checks, preserving aliased input and leaving output
unchanged on failure. Existing source-defined scalar error/status returns remain.

Native columnar imports gather typed names into GPU-owned buffers and snapshot
those buffers into their dictionary. Reference categories are independently
snapshotted on GPU. Production DMatrix and Metal quantile construction select
Metal containers. Feature-prefix/shape/type descriptors and error formatting
remain host control metadata; this is not a claim that all metadata handling or
CPU code has disappeared. Snapshotting a transient adapter may still require an
additional GPU copy, which is separate from later performance work.

Three diagnostics expose dictionary/mapping shared input, shared output and
staged bytes through `metal_info`. Existing vector counters track dictionary
allocation, clone/import and explicit readback. The internal C++ statistics
structure grew, so all native clients are rebuilt; the public C API stays intact
and CPU-only builds report the new optional JSON fields as zero.

`tests/metal_category_state_api.cc` passes **37 cases / 530,726 checks** with
Metal API validation. It exercises all eight integer widths, strings, empty
columns, exact snapshot bytes, source replacement, CPU/MPS handoffs, repeated
sorting, serialization, reload into existing containers, concurrent immutable
readers, guarded mappings, input/output aliasing, read-only input, unaligned
output and failure publication. Resident repeated calls require no vector
upload/readback or category staging; all owned vector allocations are released.
Existing native checks pass **33 typed-dictionary cases / 2,730,917 exact indices /
80 expected errors** and **48 native-columnar cases / 9,263 fields / 14 rejected
descriptors**. Their arithmetic/serialization expectations remain unchanged.
**81 focused Python tests pass** (242 warnings, 8.33s). Fifteen new prediction
checks cover reordered string/int64/uint64 names across batch, leaf, contribution,
interaction and inplace calls, using changing tree ranges to exercise actual
recoding rather than merely cached results.

Full validation completed **2026-11-05 01:05:07–01:13:24 UTC** (8m17s), exit 0:
all **95 native groups** rebuilt and passed; **3,360 Python tests passed, one
expected skip** (640 warnings, 192.20s); and the CPU-only incremental build
(25 steps) plus API/device/native input/output/quantile/cache checks passed.
No production/test changes followed that run.
[Validation record](../validation/cuda-metal-category-state-m3.txt).
Persistent prediction-model storage, remaining CUDA arithmetic/operation audits,
GPU collective transport and cross-hardware validation remain open. No benchmark
or NVIDIA execution comparison was run. Full CUDA parity is incomplete.


### Prediction model publication

CUDA authority is `predictor/gpu_predictor.cu` and `predictor/gbtree_view.h`:
each prediction constructs transient view descriptors under `GBTreeModel::Mutex`,
while `RegTree` retains its node, category and vector-leaf device mirrors. Metal
now selects those same device-aware views with `DeviceOrd::MPS()` for batch,
inplace and leaf prediction. CPU prediction still selects CPU explicitly,
including when it is dispatched from a CUDA context. Generic context semantics
and the shared CUDA view/mutation/thread-safety rules are unchanged.

Previously `MakeMetalPrediction` walked every node on the CPU, assembled full
node/category/leaf vectors, then uploaded them. Now a Metal kernel packs its
traversal representation directly from the device arrays. Host preparation reads
only view sizes, strides, per-tree weights and descriptors. Scalar raw node flags,
missing directions, deleted nodes and floating-point payload bits are preserved;
vector leaves use the right-array leaf-index mapping and leaf matrix strides.
Category words are copied on GPU and segment offsets adjusted to the packed
buffer. Existing traversal kernels and their arithmetic are unchanged.

Packed buffers are owned snapshots, valid after the input model is changed or
destroyed. They are constructed per call, rather than cached using an insufficient
pointer/tree-count key. The lasting cache is the model's existing device mirrors:
leaf edits, split changes and reloads invalidate them using HostDeviceVector's
normal access rules. Immutable concurrent readers reuse those arrays. First
publication imports model arrays; subsequent publication needs no vector upload,
readback or model payload staging. Explicit CPU model views still switch the
small tree-group vector to CPU, as upstream does, so the next GPU call may
reimport that vector. This is separate from changes to node/leaf/category arrays.

Three optional diagnostics expose model pack dispatches, shared input bytes and
staging bytes. Per-tree control descriptors are excluded. The internal Statistics
structure grows, requiring all native clients to be rebuilt; the public C API is
unchanged and CPU-only builds report zero for the new JSON fields.

**Scope correction from source audit:** SHAP model compression is not a missing
GPU operation. CUDA `interpretability/shap.cu::CompressTreeBucket` explicitly
builds host node/category/leaf arrays and uploads them, and
`PrepareGpuQuadratureModel` computes root means and depth buckets on the host.
Metal's corresponding host preparation remains intentional. At this checkpoint,
SHAP arithmetic still needed a separate source audit: CUDA applied tree weights
to leaf payloads during compression, while Metal weighted accumulated
contributions. The following SHAP step addresses that operation-order difference.

Focused validation passes the original 98 new native cases / 719,746 checks and
123 existing prediction/storage cases / 19,058,906 checks / four expected
rejections, plus 199 Python tests (41 warnings, 13.49s). Final native coverage adds
threadgroup-boundary cases: **103 cases / 1,124,304 checks** pass with Metal API
validation. Tests cover categorical/numeric and scalar/vector trees, deleted
slots, reordered leaf indices, mutations without shape changes, reload into the
same objects, CPU/MPS read handoffs, slices, weights, snapshots after destruction,
empty models and concurrent readers. Twelve new Python cases assert resident
model publication across forced cache misses and CPU comparisons; two additional
cases exercise concurrent first publication. Existing cache tests now explicitly
warm the model mirrors before requiring zero repeated imports; numerical
expectations and tolerances are unchanged.

Full validation completed **2026-11-05 01:24:19–01:32:40 UTC** (8m21s), exit 0:
all **96 native groups** rebuilt and passed; **3,374 Python tests passed, one
expected skip** (638 warnings, 200.79s); and the CPU-only incremental build (four
steps) plus device/native input/output/quantile/cache API checks passed, including
the new zero-valued diagnostics. No production or test edits followed that run.
[Validation record](../validation/cuda-metal-prediction-model-m3.txt).
Remaining CUDA operation/arithmetic audits, GPU collectives and cross-hardware
validation remain open. No benchmark or NVIDIA execution comparison was run.


### CUDA-source SHAP recurrence and arithmetic

Authority is `predictor/interpretability/shap.cu`: `CompressTree`,
`QuadratureShapTaskRunner` and `QuadratureShapInteractionTaskRunner`,
`ExtractQuadratureEdgeDeltaLocal`, `ExtractQuadratureInteractionDeltaLocal`,
`SubgroupOps::Sum` and the interaction finalizer. This corrects specific earlier
CPU-formula choices, rather than moving CUDA's intentional host model preparation.

- Host compression applies each tree's FP32 weight to scalar/vector leaf payloads
  before traversal. The internal `ShapTree` weight slot is now padding; leaf
  payloads include the weight. Root means retain the shared source-defined double
  host computation and weighting.
- Each descent multiplies the parent quadrature basis by the branch probability,
  then the new basis factor, and divides out the previous factor for repeated
  features. The old separate path-probability product has been removed. Leaves
  return basis times the weighted leaf; quadrature weights apply at edge
  extraction, after returned subtree values have been combined.
- Edge extraction forms `ret * quadrature_weight`, subtracts the exit ratio from
  the enter ratio, then multiplies. Pair extraction forms the partner factor
  times that per-point edge result, then divides by the partner basis.
- Eight values reduce in shuffle-down association at offsets 4, 2, 1. Metal
  evaluates those dependencies within each row thread. Contributions accumulate
  at each returned edge, without a late tree-weight multiply or per-tree feature
  subtotal. Right/left subtree return order, bias addition, interaction symmetry
  and diagonal subtraction follow the source expressions.

Metal uses explicit FP32-preserving arithmetic, including subnormal handling.
Its existing per-row traversal and dynamic stack remain: this is not a claim to
reproduce NVIDIA warp placement, compiler FMA contraction or nondeterministic
cross-tree atomic-add scheduling. The reference fixes a serial tree order and
uncontracted FP32 expressions. Actual NVIDIA comparison is still required before
claiming measured CUDA equivalence. CUDA supports depth through 127 edges; the
existing Metal depth130 extension is checked against the same recurrence, not
claimed as a supported CUDA execution case.

`tests/cuda_shap_reference.py` is an independent recursive replay of the CUDA
expressions using saved model data, host root means and the endpoint quadrature
rule. It does not use Metal's iterative frame algorithm. The production API
passes **52 new exact FP32 cases**: scalar regression, multiclass, vector leaves,
weighted scalar/vector trees, numeric/categorical splits, unknown/missing values,
repeated features, depths 2/17/65/127/130, zero covers, subnormal and large leaves,
clamped tiny branch probabilities and large cover ratios. Every ordinary model
case includes a witness that serial eight-point summation gives different bits.
NaN location/classification is checked; NaN payload bits are unspecified. Finite
fields and infinities are compared as raw FP32 bits. Native analytic checks pass
**592 cases / 425,748,672 output values**, with maximum absolute error zero.

The first focused run exposed four deep zero-cover differences: the CPU path's
separate probability product underflows while its unscaled basis overflows,
producing NaNs. CUDA's branch-weighted recurrence remains finite on these cases.
They now require exact agreement with the independent CUDA replay and keep the
unchanged CPU tolerance on all finite CPU fields. This is an explicit reference
correction, not relaxed accuracy or a fallback to the old algorithm. The initial
new oracle also needed to read current-format dropout weights from
`gradient_booster.model.weight_drop`; fixture weights are now checked after
reload so an ignored field cannot pass unnoticed.

Focused validation passes **127 Python tests** (2 warnings, 4.41s), including
40 initial source-reference cases and the existing SHAP suite. Twelve additional
extreme-payload cases pass (0.26s). Existing mathematical coalition, symmetry,
CPU-comparison, guards/aliasing, batching and malformed-input checks retain their
expectations and tolerances. Native fixtures now provide preweighted leaves to
match the revised internal packed-model contract. The public API and Statistics
layout are unchanged; native callers must rebuild for the internal ShapTree
semantic change. Stack-frame storage decreases from 88 to 84 bytes.

Full validation completed **2026-11-05 01:42:51–01:51:15 UTC** (8m24s), exit 0:
all **96 native groups** rebuilt and passed; **3,426 Python tests passed, one
expected skip** (638 warnings, 205.30s); and the CPU-only incremental build (three
steps) plus device/native input/output/quantile/cache API checks passed.
No production or test edits followed the full run.
[Validation record](../validation/cuda-metal-shap-source-m3.txt).
GPU collective integration, remaining CUDA operation/arithmetic audits and
cross-hardware validation remain open. No benchmark was run for this step.


### Metal collective backend and local-worker transport

`CommGroup` now selects a distinct Metal backend for explicit MPS payload views.
Generic MPS contexts still expose CPU to existing generic consumers, so CPU
control scalars continue using the CPU backend. This does not remove distributed
training guards or silently reinterpret CPU payloads as Metal.

Implemented operations:

- Numeric sum/min/max for the types accepted by `coll.cu::GetNCCLType`: FP16,
  FP32, FP64, signed/unsigned 8-, 32- and 64-bit integers. Numeric 16-bit integers
  and extended floats are rejected, as in CUDA. FP64 uses existing GPU software
  arithmetic; FP16 conversion preserves subnormals and round-to-nearest-even.
- Bytewise AND/OR/XOR across all payload types, including floating-point bytes,
  before numeric type dispatch. The allgather plus rank-ordered GPU fold follows
  `coll.cu::BitwiseAllReduce`.
- Numeric ring reduce-scatter/allgather, with a rank-ordered small-array path,
  owned Metal receive scratch and GPU reductions between communication steps.
  NCCL selects its own topology/order; this is not a bitwise NVIDIA-run claim.
- Broadcast, fixed allgather, and variable allgather using ring or broadcast.
  The local variable-gather copy runs on GPU, and the typed wrapper preserves
  device output ownership. Empty segments bypass zero-byte socket submissions;
  that fix is shared with CPU ring allgather and has a CPU regression check.

Apple shared allocations are CPU-addressable. GPU producers complete before TCP
sends, and receives finish before GPU consumers. There are no CPU arithmetic
reductions or payload-vector mirrors in these Metal paths. Networking itself is
host TCP; this is not an implementation of NCCL or GPU-direct transport. The
current backend is synchronous. Guarded host views can stage when they cannot be
borrowed; resident-payload tests require zero staging bytes.

Arithmetic authority is the vendored CUDA dispatch plus NVIDIA's
[NCCL scalar reduction source](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/device/reduce_kernel.h)
and [half min/max contract](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html).
Tests compare exact finite/infinite bits and quiet-NaN classification, with
explicit signed-zero and number-preferred NaN rules. Random full-exponent inputs
exposed a signaling-NaN difference in the host reference's native minimum; the
reference now specifies CUDA's documented policy rather than relying on that
host instruction. GPU arithmetic was not replaced with a CPU calculation.

The new native group exercises 1/2/3/4 localhost workers, all numeric and bytewise
types, empty/short/uneven/large buffers, arbitrary byte offsets, sentinels,
CPU/MPS handoffs, overlapping local GPU copies, both allgather-v algorithms,
99 invalid calls and a deliberately disconnected peer. It includes exhaustive
half encodings and random FP32/FP64/integer operands with independent native
scalar arithmetic. GPU counters assert actual reduction/copy dispatches and no
resident-payload staging. The ordinary workers pass 6,026 cases and 12,230,282
field checks. Full regression passes all 97 native groups, 3,426 Python tests
(one expected skip), and the CPU-only build/API checks.

Variable-size `AllreduceV` and resident sketch pack/merge/prune/load are next.
Global quantizer scales and integer histogram/root/leaf reductions must then be
connected before distributed training is enabled. CUDA intentionally aggregates
quantizer bounds and row counts on the host (`gpu_hist/quantiser.cu`); that control
work does not require a GPU collective to match the source. AUC's earlier host-summary
fallback, other consumers, federated transport, cross-machine testing and NVIDIA
execution validation remain open. No benchmark or performance claim accompanies
this change. See the [validation record](../validation/cuda-metal-collectives-m3.txt).
