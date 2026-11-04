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

Latest step: **resident AUC inputs and summaries**. Binary, multiclass, multilabel
and grouped ROC/PR evaluators now consume Metal predictions, labels and weights.
Multi-output summaries stay in owned Metal storage through normalization. Sorting
accepts offset views directly; result publication supports aliased inputs and a
counted fallback for unshareable outputs. Arithmetic kernels are unchanged.

Focused validation passes **172 single-worker native cases / 392,552 field checks /
five rejected calls**, **96 cases per worker** with two and three local workers,
and all **170 public AUC tests**. Empty workers without local target metadata now
enter the same collective path as populated workers, as CUDA requires.
Full regression passes **3,325 Python tests, one expected skip**, all **89 rebuilt
native check groups**, and the CPU-only build/API checks.
[AUC storage validation](validation/cuda-metal-auc-state-m3.txt).

Distributed AUC retains a documented host-summary collective fallback; these
metric tests do not enable distributed MPS training. Precision/NDCG/MAP caches
and metrics, ranking training, predictor/updater state, public result plumbing
and persistent dictionaries still need ownership work. Generic context storage
still defaults to CPU. No new benchmark or NVIDIA execution comparison.

The preceding step added **resident scalar metric inputs**. Scalar/multiclass,
quantile/expectile and AFT likelihood/interval-accuracy evaluators now consume
Metal prediction, label, weight and level arrays without first requesting host
mirrors or uploading those payloads again. CUDA's two-number reduction result,
final ratio and host collective remain intact. Quantile/expectile metrics also
now reject query-group weights through the same check used by CPU and CUDA.

Focused validation passes **587 native cases / 44,478 field checks / 110 rejected
calls** and **139 Python tests**, including exact binding comparisons, repeated
calls, GPU writes, CPU/MPS handoffs and read-only input fallbacks. Full regression passes **3,325
Python tests, one expected skip**, all **88 rebuilt native check groups**, and the
CPU-only build/API checks.
[Metric storage validation](validation/cuda-metal-metric-state-m3.txt).

At that checkpoint, AUC/ranking metrics, ranking training, predictor/updater state,
public result plumbing and persistent dictionaries still needed ownership work.
The AUC portion is addressed by the latest step above. No benchmark or NVIDIA
execution comparison was run for the scalar metric step.

The preceding step added **GPU storage through initialization and stump fitting**. Means,
weighted means, medians/quantiles, class priors, initial gradients and prediction
broadcasts now consume Metal inputs and publish results with GPU copies or fills.
Their objective callers retain Metal storage through the GPU stages. Multiclass
centering reads its mean on GPU. CUDA's scalar returns and expectile/absolute-error
host finalizers remain intact.

New native validation passes **250 cases / 3,625 field checks / three rejected
calls**, checking actual objective initialization, CPU comparisons, independently
bounded FP32 reductions, access states, strides, empty inputs, aliasing and guarded
fallbacks. Full regression passes **3,321 Python tests, one expected skip**, all
**87 rebuilt native check groups**, and the CPU-only build/API checks.
[Initialization storage validation](validation/cuda-metal-initialization-state-m3.txt).

This is another consumer migration, not complete training residency. Ranking,
AUC/ranking metrics, tree/prediction callers and public result plumbing still need work. The
execution context still defaults generic state to CPU storage. No benchmark or
NVIDIA execution comparison was run for this step.

The preceding step added **resident objective inputs and gradient outputs**. Regression,
classification, absolute/quantile and AFT objective calls now consume explicit
Metal spans and write gradients into owned Metal storage. Their prediction
transforms, plus scalar regression and AFT base-margin conversions, retain it.
Regression/classification label checks no longer request a host mirror first.
Checked margin conversion publishes its result only after validation succeeds.

The arithmetic kernels are unchanged. Native checks verify CPU comparisons,
GPU writes, repeated calls without readback/reupload, CPU/MPS handoffs, invalid
inputs and guarded buffer fallbacks. Focused validation passes **384 cases /
1,003,052 field checks / eight rejected calls** and **619 Python tests**.
Full regression passes **3,321 Python tests, one expected skip**, all **86
rebuilt native check groups**, and the CPU-only build/API checks.
[Objective storage validation](validation/cuda-metal-objective-state-m3.txt).

This closed objective-level storage gaps. Ranking, AUC/ranking metrics, tree/prediction
callers and public result plumbing still need consumer
migration and can request host mirrors. The execution context still defaults
generic state to CPU storage. No new speedup or NVIDIA execution claim.

The preceding step added **Metal vector storage and metadata operations**. The generic vector
layer now supports owned Metal allocations, GPU fills/copies/growth, and lazy
host mirrors with CUDA-derived access rules. Native labels, weights, margins,
survival bounds and feature weights retain this storage after import. Internal
metadata slicing gathers rows on GPU; metadata copies/stacking and feature-type
equality use Metal when their inputs have device storage.

This is a storage foundation, not complete training residency. The execution
context still selects host storage by default for generic state, and existing
consumers can request host views. Public ordinary matrix slicing stays on CPU,
as upstream does; external-memory matrix slicing remains unsupported.

Focused validation passes **83 native cases / 1,432,716 exact fields / 33 rejected
calls**, **20 new public tests**, and the **935-test interoperability regression**.
Full regression passes **3,321 Python tests, one expected skip**, all **85
rebuilt native check groups**, and the CPU-only build/API checks.
[Storage validation](validation/cuda-metal-vector-state-m3.txt).
No new benchmark or NVIDIA execution comparison.

The preceding step added **owned GPU prediction outputs**. Request
`predict(..., output_device="mps")` or `inplace_predict(..., output_device="mps")`
to get an `xgb.MetalOutput`. PyTorch and MLX consume it through DLPack; `.numpy()`
explicitly copies it to a host array. Existing default return types are unchanged.
Outputs include values, margins, leaves, SHAP contributions and interactions.

Each result owns a private Metal buffer independently of the model and prediction
cache. Multiple consumers can share it; an explicit DLPack `copy=True` export
creates another GPU allocation. Eligible prediction and transform buffers now
use guarded unified-memory views before the GPU snapshot. Generic prediction
state remains host-owned, so this does **not** complete device ownership across
the training/prediction pipeline. Known PyTorch/MLX queues synchronize; arbitrary
consumer writes must be completed before sharing or reading back.

Focused validation passes **100 new Python tests**, **12 native output cases /
3,389 field checks / 13 rejected calls**, and **592 native SHAP cases**, including
overlapping margins. Full regression passes **3,301 Python tests, one skip**,
all **84 rebuilt native check groups**, and the CPU-only build/API checks.
[Output validation](validation/cuda-metal-output-m3.txt).
No performance benchmark or NVIDIA execution comparison was run for this step.

The preceding step added **native Metal columnar input**. `xgb.MetalColumnar` accepts named or
unnamed PyTorch/MLX columns with independent datatypes. `xgb.MetalColumn` adds
packed validity masks and integer or UTF-8 categorical dictionaries. Ordinary
and quantile matrices, external iterators, reference-category recoding, inplace
prediction and CPU tree fallback now accept these tables. Feature values remain
in owned GPU snapshots; mask decoding and row-code mapping run on Metal.

Integer dictionary names are copied exactly, including int64 values beyond
float precision. Name gathering runs on GPU into owned dictionary storage;
existing dictionary sorting and mapping kernels are reused. General persistent
dictionary storage/copy ownership remains host-based. A shared CPU accessor fix
also preserves missing categories explicitly instead of casting NaN to an integer.

Validation passes **62 new Python tests** and **48 native cases / 9,263 field
checks / 14 rejected descriptors**. Full regression passes **3,201 Python tests,
one skip**, all **83 rebuilt native check groups**, and CPU-only build/API checks.
[Columnar validation](validation/cuda-metal-columnar-m3.txt).
No new benchmark or NVIDIA execution comparison.

The preceding step added **native Metal input for ordinary `DMatrix`**. PyTorch/MLX feature
tensors and DLPack capsules now work in ordinary constructors and external-data
iterators. Missing-value filtering, per-row counts, prefix scans and sparse entry
packing run on GPU. Native inputs also work with CPU tree prediction fallback.

Focused validation passes **773 Python tests**, including 186 new ordinary-input
checks, and **199 native cases / 550,595 field checks / six rejected inputs**.
Native versus host inputs produce byte-identical models separately on CPU and
MPS for supported hist/approx and categorical/vector fixtures. MPS DART and
linear fixtures also match.
Save/load, slicing, changed bin counts, empty dimensions and external-cache reuse
are covered. Full regression passes **3,139 Python tests, one skip**, and all
82 rebuilt native check groups, with CPU-only build/API checks passing.
[Ordinary tensor validation](validation/cuda-metal-ordinary-tensor-m3.txt).

Sparse outputs use guarded shared-memory views with a counted copy fallback.
Downstream raw-CSR sketches and compressed-page builders consume eligible shared
allocations directly. This preserves the existing host APIs; generic allocation
and state ownership still need work. No new benchmark.

The preceding step added **native Metal custom-gradient imports**. `Booster.boost` and custom
objective callbacks accept PyTorch/MLX GPU gradients and Hessians, including
separate split-selection and leaf-value pairs. Metal follows CUDA's independent
typed casts and pair packing, with no CPU conversion of input tensor values.
Split/value pairs can independently come from CPU or Metal; each gradient and
its Hessian must share a device. Legacy flattened inputs retain their warning.

Native validation passes **333 cases / 328,515 field checks / 17 rejected inputs**,
including all float16 bit patterns and private buffers. The **162 new Python tests**
compare native-input models with host-input models byte for byte on both CPU and
MPS. Focused regression passes **587 tests**; full regression passes **2,953 Python
tests, one skip**, with every rebuilt native suite and CPU-only build/API checks
passing.
[Gradient import validation](validation/cuda-metal-gradient-import-m3.txt).

Packing writes through safe shared destination views, with a counted copy
fallback. Generic gradient allocation remains host-owned; callback predictions
still arrive as NumPy arrays. Explicit GPU prediction outputs are implemented
above; end-to-end device ownership remains open. No new benchmark.

The preceding step added **native Metal metadata imports**. PyTorch/MLX labels, weights,
base margins, survival bounds, feature weights and ranking groups/query IDs now
enter through GPU conversion and validation. Query IDs use GPU run-length
encoding and prefix scans; final group boundaries remain host-side as in CUDA.
Scalar, vector and matrix inputs preserve CUDA's shape and dtype rules.

Focused validation passes **425 Python tests** (332 metadata and 93 feature-input
regressions) and **322 native cases / 1,011,267 field checks / 24 rejected inputs**.
CPU models trained with imported GPU metadata serialize identically to models
trained with host metadata. Native/host-input Metal fits agree exactly; CPU
multiclass output comparison allows measured FP32 softmax rounding while margins
remain exact. Full regression passes **2,791 Python tests, one skip**, every
rebuilt native suite and the CPU-only build/API checks.
[Metadata validation](validation/cuda-metal-metadata-m3.txt).

At that checkpoint, float metadata used shared views of host destinations with a
counted copy fallback. The latest storage step above adds owned Metal allocations
and internal concatenation/slicing. Broader consumer and state migration remains
unfinished. No new benchmark.

The preceding step added **native PyTorch/MLX Metal feature tensors and float16 input**.
`QuantileDMatrix`, both external quantile cache modes and `inplace_predict` now
accept Metal DLPack inputs. Strided views are copied into owned GPU storage and
decoded on the GPU, without staging feature values through NumPy or CPU arrays.
Tests cover real PyTorch/MLX producers, private buffers, source mutation/release,
legacy/versioned capsule ownership and all 65,536 float16 bit patterns.

Focused validation passes **93 Python tests** and **105 native cases / 145,567
field checks / nine rejected descriptors**. Full regression passes **2,459 Python
tests, one skip**, every rebuilt native suite and the CPU-only build/API checks.
Native metadata, ordinary `DMatrix` inputs and explicit GPU prediction outputs
are covered above. Default Metal prediction results still return NumPy.
PyTorch coexistence requires a single OpenMP runtime, configured below.
No new benchmark. [Tensor validation](validation/cuda-metal-tensor-m3.txt).

The preceding step added **GPU Snappy compression and decompression for sparse host caches**.
Metal now implements raw Snappy literals and all copy forms, including overlapping
backreferences. The independent Google CPU codec validates both directions;
repeated inputs actually compress. Cache reads reconstruct the original ELLPACK
bytes on the GPU before training, prediction or CPU materialization.

This follows CUDA's optional compute-fallback path: the existing native
`ExtMemConfig::SetParamsForTest(ratio, true)` enables it for sparse external
quantile host caches. Dense and feature-relative sparse pages bypass it. The
unspecified compression ratio is 0.4 and blocks are 2 MiB. **It is disabled by
default, has no new Python flag, and is not an Apple hardware decompression-engine
claim.** Public selection and performance tuning remain deferred.

Focused validation passes **248 valid codec cases / 346 rejected inputs /
132,381,418 exact decoded bytes**, plus **408 external-cache cases**, **310 GPU
consumer cases** and **237 public regressions**. Compressed and plain Metal fits
agree exactly on predictions, leaf IDs and SHAP; CPU predictions are checked too.
Full regression passes **2,366 Python tests, one skip**, every rebuilt native
suite and the CPU-only checks. The expanded ELLPACK suite passes **355 cases**,
including cache reconstruction across 2 MiB chunk boundaries. No new benchmark.
[Snappy validation](validation/cuda-snappy-m3.txt).

The preceding step added **ordinary external-memory host caches**.
`DMatrix(DataIter(on_host=True))` now keeps its MPS quantized pages in Metal storage
instead of writing a quantized disk cache. It follows CUDA's defaults: one input
page stays private to the GPU, multiple pages use shared host storage, and ordinary
matrices retain one cache page per input batch. The original raw CSR file remains.

Host/disk selection survives CPU/MPS and max-bin changes. Constant-Hessian `approx`
reuses its cache; variable-Hessian `approx` rebuilds it. Metadata reads avoid CPU
bin arrays, and retained GPU pages survive cache regeneration and destruction.

Focused validation passes **288 cache cases / 16,273,773 exact fields**,
**390 public regression tests**, and **22 empty/default/error cases**. This adds
64 native host-cache cases and 52 public cases. Full validation passes **2,366 Python
tests, one skip**, every rebuilt native suite and the expanded CPU-only checks.
No new benchmark. [External-host validation](validation/cuda-external-host-m3.txt).

The preceding step added **compressed disk caches and internal ELLPACK save/load**.
MPS external quantile matrices now bin directly into compressed disk pages;
ordinary external matrices compress raw CSR into the same format. GPU traversal
loads compressed views, and CPU bin/column arrays are created only on demand.
The disk field order, alignment and payload follow CUDA's raw ELLPACK format.
Loaded Metal buffers remain valid after their mapping or cache is destroyed.

Internal quantile save/load now preserves cuts and compressed pages, matching
CUDA's internal contract. This is not a new public `QuantileDMatrix.save_binary`
API and does not serialize labels or other training metadata.

Focused validation passes **224 cache cases / 12,662,693 exact fields** and
**124 IO cases / 18,400 exact bytes / 111 invalid streams / three internal
save/load cases**. **254 existing public tests and 50 new public cases** pass.
Full regression passes **2,314 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/API checks. A final resource-scope follow-up also passes its
targeted build and IO suite. No new benchmark was run. [Disk-format validation](validation/cuda-disk-ellpack-m3.txt).

Ordinary external host-cache selection is covered above. Mapped-file zero-copy
remains open; optional Snappy compute compression is covered above.
CPU-origin external matrices retain the existing compatibility path; upstream CUDA
rejects CPU-created external quantile matrices. Broader device/state and collective gaps remain.

The preceding step added **direct raw CSR compression for ordinary in-memory matrices**.
MPS training now builds ELLPACK directly from bounded windows of raw rows and
values, without a host quantized matrix or a typed GPU-bin upload. CPU quantized
views are materialized only on explicit demand. Metadata-only cut lookup, raw
`get_data()` export and ordinary CPU prediction leave those mirrors absent.

CUDA sketch selection and cache regeneration are preserved: constant-Hessian
`approx` reuses cuts, variable-Hessian `approx` regenerates them, and mutable
CPU/MPS training switches choose the corresponding sketch. Retained readers keep
their old generation when a new one is published.

Native validation passes **70 raw-format cases / 568 windows / 120,537 exact bytes**
and **130 matrix cases / 1,173,186 exact fields**, including concurrent conversion
and retained views after regeneration/destruction. **303 existing focused tests**
and **40 new public API cases** pass. Full regression at that checkpoint passed **2,264 Python tests,
one skip**, every rebuilt native suite and the CPU-only build/API checks.
No new benchmark was run.
[Ordinary-matrix validation](validation/cuda-ordinary-ellpack-m3.txt).

The preceding step added **compressed in-memory QuantileDMatrix ownership**. Explicit MPS
construction writes directly to GPU ELLPACK pages and joins input batches there.
CPU-created quantile matrices compress their existing bins once on first MPS use.
Training and prediction retain a GPU view with no host bin/row/column arrays;
CPU access creates a separate representation once. Concurrent requests publish
one representation per device, and retained GPU views outlive their matrix.

New native validation passes **260 cases / 3,266,138 exact fields**, including
**10 concurrent conversion cases**. **27 new public cases** cover both construction
devices, CPU export, model reload and a sparse input crossing the actual
67-million-slot window boundary. Two new iterator checks reject changed column
counts and increased row width. Existing CPU comparisons and tolerances are retained.
Full regression at that checkpoint passed **2,224 Python tests, one skip**, every rebuilt native suite
and the CPU-only build/API checks. No new speedup is claimed.
[In-memory validation](validation/cuda-inmemory-ellpack-m3.txt).

The preceding step added **host mirrors on demand for external compressed caches**. MPS
training, prediction and exact SHAP now use pages with no host row pointers,
bin arrays, hit counts or column data. CPU callers explicitly materialize a
mirror; metadata-only cut access does not. Prefetch always produces GPU views,
so repeated CPU/MPS switches do not spill host restoration into GPU traversal.

Native validation passes **278 consumer cases / 72,017,790 exact fields** and
**192 cache cases / 10,845,265 exact fields**, including retained GPU views after
cache destruction. Focused public regression passes **148 tests**, with **27 new
cases** covering device switches, CPU training, dropout, multiclass and vector
leaves. Approximate SHAP retains its explicit CPU fallback (CUDA does not implement
it). The latest step extends this ownership to in-memory quantile matrices.
Full regression at that checkpoint passed **2,195 Python tests**, all rebuilt native suites and the
CPU-only build/API checks.
[Lazy-host validation](validation/cuda-lazy-host-m3.txt).

The preceding step added **compressed GPU consumers for external host caches**. Histogram
building (global and shared memory), row partitioning, and prediction/SHAP input
preparation now read ELLPACK symbols directly. Their pages retain compressed
buffers instead of restored GPU indices and dense transposes.

Native checks pass **139 cases / 28,485,537 exact fields**, including **278
histograms, 1,946 partitions and 417 input chunks** with the host bins, row pointers
and cut values overwritten. Twelve new public cases check margins, leaf IDs,
SHAP contributions/interactions and model reload. Host mirrors were still eager at
that checkpoint; the latest step above makes them optional.
Full regression at that checkpoint passed **2,168 Python tests**, all rebuilt native suites and the
CPU-only build/API checks.
[Compressed-consumer validation](validation/cuda-compressed-consumers-m3.txt).

The preceding step added **direct GPU adapter-to-ELLPACK construction for external host caches**.
Decoded values now feed compressed bin writes directly, without an intermediate
host gradient-index matrix or host-bin upload. Sparse row positions stay on GPU
across input windows; missing padding is written after the values, following CUDA.
At that checkpoint the disk path retained its host format; the latest step above replaces it for MPS-origin caches.

New native checks pass **73 cases / 2,933 windows / 81,666 exact bytes / 7,455,890
fields**, plus **12 expected errors**. Focused public regression passes **210
tests**, including a sparse row split across the actual 67-million-slot window
boundary. Counters verify that the direct path does not stage bins on the host.
Full regression at that checkpoint passed **2,156 Python tests**, all rebuilt native suites and the
CPU-only build/API checks.
[Direct-construction validation](validation/cuda-direct-ellpack-m3.txt).

The preceding step added **GPU-compressed ELLPACK pages, adaptive shared/private caching and
page concatenation**. `ExtMemQuantileDMatrix(iterator, device="mps")` now supports
`DataIter(on_host=True)`, the iterator default, with `cache_host_ratio` and
`min_cache_page_bytes`. Metal packs, concatenates and reads CUDA's bitstream.
Completed caches do not need input callbacks.

The format tests pass **257 cases / 82,394 exact bytes / 55,438,557 exact fields**,
covering bit widths 1–32, split symbols and non-byte-aligned concatenation.
Expanded external-cache checks cover **192 cases /
2,322 cached-page visits / 10,815,835 exact fields**. **96 new public tests** cover
cache ratios, concatenation, references, training, metadata and failure handling;
the focused construction regression passed **197 tests**. Full validation at that checkpoint passed
**2,143 Python tests**, every rebuilt native suite and the CPU-only build/API
checks. [Compressed-cache validation](validation/cuda-ellpack-cache-m3.txt).

Apple shared and private buffers both use unified memory; this is a storage/access
policy, not additional VRAM capacity. Direct construction above removes that
checkpoint's intermediate host bins. The later consumer/ownership steps above
retain compressed GPU views and create host mirrors only for CPU access. CUDA's
optional Snappy cache compression, ordinary external host-cache selection, native Apple GPU tensor handles and
general device ownership remain unfinished. No new performance measurement was made.

The preceding disk-backed construction step supports `DataIter(on_host=False)`:
GPU decoding, resident CUDA-derived sketching and bin assignment write one disk
page per input batch. Its checks passed **48 native cases / 864 cached-page visits /
2,720,419 exact fields** and **43 public tests**. Full regression at that checkpoint
passed **2,047 Python tests**, every rebuilt native suite and the CPU-only checks.
[External construction validation](validation/cuda-extmem-construction-m3.txt).

The preceding adapter step added
`xgb.QuantileDMatrix(X, y, device="mps", max_bin=256)` for host dense/columnar
inputs and streamed batches. Its checks passed **65 native cases / 2,184 windows /
1,126,724 exact fields** and **58 public tests**; full regression at that checkpoint
passed **2,004 Python tests**, every rebuilt native suite and the CPU-only build/check.
[Adapter construction validation](validation/cuda-adapter-construction-m3.txt).

The preceding prediction step moved input conversion and row-level category
mapping for prediction/SHAP to Metal: **144 native cases / 6,677,824 prepared
values**, with **1,946 Python tests** passing at that checkpoint.
[Input preparation validation](validation/cuda-input-preparation-m3.txt).

The preceding category-name step moved typed sorting, whole-dictionary schema
checks and name-to-code mapping to Metal: **33 native dictionary cases /
2,730,917 exact indices**, **80 expected errors** and **19 public tests**.
[Category-name validation](validation/cuda-category-names-m3.txt).

The preceding adaptive step: **absolute-error and quantile gradients now preserve CUDA's source
arithmetic**, including subnormal values, hypot/tanh edge cases and weight order.
Residual roots now sum in FP32, while optional weights sum in binary64, matching
the distinct CUDA reducers. Small scale statistics pass through the same host
collective used upstream, including on empty workers; scale application and
row gradients run on GPU. Full distributed MPS training remains disabled.

New checks pass **2,359,332 source-expression fields**, **1,000,800 math fields**
and **308 production objective calls / 8,815,888 result checks** across one, two
and three local workers. The worker fixtures verify global aggregation, empty
partitions and recursive reductions. These establish source and CPU-reference
agreement for the tested cases; they do not establish NVIDIA/libdevice identity.
Full regression passes **1,856 Python tests**, every rebuilt native suite and
the CPU-only build/check.
[Adaptive-objective validation](validation/cuda-adaptive-source-m3.txt).

The preceding regression, hinge and expectile step preserves CUDA's source
arithmetic and prediction transforms. Squared-log Hessians retain their binary64
denominator; hinge retains its binary64 label expression. Its extracted-source
checks cover **624 cases / 14,305,782 fields**. Extreme stress cases retain
explicit CPU-math differences; ordinary training comparisons keep their existing
tolerances. [Objective arithmetic validation](validation/cuda-objective-arithmetic-m3.txt).

The preceding multiclass step aligned softmax with CUDA's sequential binary64
sum of FP32 terms and removed repeated class scans. Argmax preserves source
behavior for leading NaNs and subnormal values. Its native checks cover **192
cases / 12,181,658 fields**.
[Multiclass validation](validation/cuda-multiclass-softmax-m3.txt).

The preceding step moved multiclass base-score histograms, normalization,
logarithms and centering to Metal. **CUDA packs observed class sums into consecutive
bins:** labels `[1, 1, 3]` with four classes produce `[2, 1, 0, 0]`, unlike CPU's
`[0, 2, 0, 1]`. The port preserves this behavior; objective labels still must be
integral. Shared scalar means now accumulate FP32 terms in FP32, and quantile
initialization uses GPU vector normalization.
[Initialization validation](validation/cuda-multiclass-init-m3.txt).

The preceding AFT port moved every survival-gradient row to GPU, with zero CPU
corrections. Its source checks cover **316,683 rows / 3,166,830 exact fields**.
Tiny CDF differences can amplify library rounding: 322 stress-corpus fields differ
from CPU beyond its former tolerance, with a maximum weighted difference of
44.78. These differences remain explicit; NVIDIA equivalence is unverified.
[AFT validation](validation/cuda-aft-device-m3.txt).

The preceding metric port moved scalar, multiclass, quantile/expectile and survival
accumulation entirely to GPU, with one final 16-byte result pair. Its tests pass
**349 cases / 15,628,618 exact row-result fields** and **500,285 math fields**.
[Metric validation](validation/cuda-metric-reductions-m3.txt).

The preceding metadata port added GPU query maxima/counts and exact 64-bit
prefixes, with immutable buffers reused by ranking caches. Its tests check
**12.6 million exact task fields**. [Metadata validation](validation/cuda-ranking-metadata-m3.txt).

The preceding arithmetic port checks **2.44 million ordinary gradient pairs**
bit for bit against the host CUDA-formula oracle, plus multi-round unbiased
updates and **9.07 million exact sampled pairs**. CUDA's shared cost-factor race
and nonfinite rounding inputs have explicit deterministic policies.
[Arithmetic validation](validation/cuda-ranking-arithmetic-m3.txt) ·
[Behavior and remaining limits](docs/CUDA_PARITY.md#ranking-gradient-numerical-behavior).

The preceding step added CUDA-derived ordinary-matrix quantile sketches for
`hist` and `approx`, including external-memory pages: GPU sorting, FP32 scans,
category deduplication and resident summary pruning/merging.
[Sketch validation](validation/cuda-resident-sketch-m3.txt).

The major remaining areas are general metadata/state ownership and manipulation,
persistent dictionary storage, remaining state operations,
CUDA algorithm alignment and distributed collectives. Compressed disk caches,
internal ELLPACK save/load and optional Snappy compute compression are implemented.
Sensitive AFT tails and some
long-run ranking and Covertype comparisons have numerical differences.
Persistent GPU ownership and less synchronization also remain unfinished.
Native CSR/CSC device-array bindings are not a CUDA parity requirement: the
vendored upstream Python interface explicitly rejects CuPy CSR and CSC inputs.
The existing host-sparse and internal GPU sparse operations remain in scope.
The [coverage ledger](docs/CUDA_PARITY.md) records the details. Performance
experiments are deferred while these coverage gaps are addressed.

The current backend runs histograms, gradient quantization, several objectives,
prediction transforms, tree traversal, SHAP contributions/interactions, prediction-cache
updates, common metrics,
row partitioning, scalar/vector split evaluation, root-gradient sums, stump
fitting, reduced-gradient leaf refitting, linear coordinate descent, means,
stable column/grouped sorting, ROC/PR-AUC, ranking metrics,
gradient-based row sampling, uniform/weighted column sampling and quantile/median
estimation on Metal. Ordinary matrices now sort entries and maintain CUDA-derived
quantile summaries on Metal. Adapter-based preprocessing and several specialized
features still run on the CPU. Histogram conversion,
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
(`brew install libomp` if needed). The native test suite also uses Google's
independent Snappy library (`brew install snappy`, or set `SNAPPY_ROOT`). It is a
test dependency only; the production Metal library does not link it.

```sh
uv venv .venv
uv pip install --python .venv/bin/python numpy scipy pytest scikit-learn pandas
bash scripts/build.sh
export PYTHONPATH="$PWD/xgboost/python-package"
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

```python
import xgboost as xgb

print(xgb.metal_info())
model = xgb.XGBRegressor(device="mps", tree_method="hist")
model.fit(X, y)
prediction = model.predict(X)
```

To construct quantiles on Metal from host dense arrays or data frames:

```python
train = xgb.QuantileDMatrix(X, y, device="mps", max_bin=256)
valid = xgb.QuantileDMatrix(X_valid, y_valid, device="mps", ref=train, max_bin=256)
model = xgb.train({"device": "mps", "tree_method": "hist", "max_bin": 256}, train)
```

Construction device is independent of booster device. Omitting it selects MPS
for native Metal feature tensors and CPU for host arrays. Dense/columnar host
and native Metal `DataIter` batches are supported, including native metadata.
CPU quantized views are materialized on demand. Native CSR/CSC bindings are not
a CUDA parity requirement; upstream rejects those device-array inputs.

Native feature inputs use the producer's Metal DLPack export:

```sh
# Optional producer/test dependencies; versions validated on this Mac.
uv pip install --python .venv/bin/python 'torch==2.15.0' 'mlx==0.32.4'
```

```python
import torch

X_gpu = torch.randn(1000, 32, device="mps", dtype=torch.float16)
y_gpu = torch.randn(1000, device="mps")
train = xgb.QuantileDMatrix(X_gpu, y_gpu, max_bin=256)
model = xgb.train({"device": "mps", "max_bin": 256}, train)
prediction = model.inplace_predict(X_gpu)  # NumPy output
```

To keep the result in GPU storage:

```python
result = model.inplace_predict(X_gpu, output_device="mps")
prediction_gpu = torch.from_dlpack(result)  # shares the owned Metal buffer
prediction_cpu = result.numpy()            # explicit host copy

# Also supported by batch prediction, including SHAP and leaf outputs:
contributions = model.predict(train, pred_contribs=True, output_device="mps")
# MLX consumer: mlx.core.from_dlpack(result)
```

Consumer writes are shared across exports. To get independent storage, consume
`result.__dlpack__(copy=True)`. Results remain valid after model deletion and
later predictions. Output selection does not change the model's execution device.

MLX GPU arrays work through the same interface. `xgb.DMatrix(X_gpu, y_gpu)` and
ordinary `DataIter` batches also accept native feature tensors. The Python bridge completes
known PyTorch/MLX producer work before each import; generic Metal providers must
supply ready buffers. Imports make owned GPU snapshots rather than retaining
mutable feature storage. A DLPack capsule is consumed once; an iterator should
return its producer tensor on each callback. Labels, weights, base margins,
survival bounds, feature weights and group/qid metadata may be native tensors
or host arrays. Float16/32/64, integer widths and bool are supported
by the native interface, subject to each producer's dtype support.

For independently typed columns, use the explicit native table wrapper:

```python
columns = xgb.MetalColumnar({
    "amount": torch.randn(1000, device="mps"),
    "kind": xgb.MetalColumn(
        torch.arange(1000, device="mps", dtype=torch.int32) % 3,
        categories=torch.tensor([10, 20, 30], device="mps", dtype=torch.int64),
    ),
})
train = xgb.QuantileDMatrix(columns, y_gpu, enable_categorical=True)
model = xgb.train({"device": "mps"}, train)
prediction = model.inplace_predict(columns)
```

A column's optional `mask` is a contiguous GPU uint8 tensor of packed validity
bits: least significant bit first, 1 means valid. `mask_bit=0..7` selects the
first row's bit. String dictionaries use `categories=(offsets, values)`, with
GPU int32 offsets and int8 UTF-8 bytes. Pass `feature_types=model.get_categories()`
when constructing a new matrix that should recode against the training dictionary.
Table columns must have equal row counts, and category dictionaries must be
nonempty. A table retains its DLPack exports across iterator resets; producer
writes are completed before each import. A producer that replaces its allocation
requires a new table. Numeric feature types default to `q`, dictionary columns to
`c`; named tables also participate in prediction feature-name checks.

Custom objectives can return Metal gradients and Hessians through the same
DLPack bridge. `Objective` and `TreeObjective` are supported, including distinct
split/value gradient pairs. Each pair must use one device; different pairs may
use different devices. Predictions passed into the callback still use NumPy,
so a callback computing gradients in PyTorch/MLX must transfer those predictions.
The returned GPU gradients are packed on Metal without a NumPy conversion.

Set `DYLD_LIBRARY_PATH` above **before starting Python** when using PyTorch with
this Homebrew OpenMP build. PyTorch's wheel bundles another OpenMP runtime;
loading both caused a reproduced worker-thread crash. The test launcher selects
the Homebrew runtime for both libraries. Unconfigured mixed-runtime packaging
is not resolved by this backend.

For compressed in-memory caching, define your iterator with
`super().__init__(on_host=True, min_cache_page_bytes=0)` and use:

```python
train = xgb.ExtMemQuantileDMatrix(
    iterator, device="mps", max_bin=256, cache_host_ratio=0.25
)
model = xgb.train({"device": "mps", "tree_method": "hist", "max_bin": 256}, train)
```

Here 25% of each compressed cache page is CPU/GPU-accessible shared storage and
the rest is GPU-private storage. Omitting the ratio and minimum page size selects
CUDA-derived defaults using Metal's recommended working-set budget. A positive
minimum concatenates small input pages; zero preserves separate input pages.
As in CUDA, a cache containing only one page forces a zero host ratio. Both
storage modes use Apple's unified memory, so the ratio does not create additional
memory capacity. [Apple storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus).

For disk caching, use
`super().__init__(cache_prefix="/existing/directory/cache", on_host=False)` and
omit `cache_host_ratio`. This writes one disk page per input batch and removes
the files when the matrix is freed. Host caching creates no disk files.
`ref=train` is supported in both modes. Defaults without an explicit construction
device retain upstream behavior.

Linear GPU training uses `booster="gblinear", updater="coord_descent", device="mps"`.
Its default `shotgun` updater remains on the host, as it does with CUDA. The
vendored upstream version deprecates `gblinear`, but this port supports it.

Approximate training is also available with `tree_method="approx"`. It uses
unsampled Hessians for cuts and resets sampling per tree, following CUDA's updater
order. Both sampling methods, histograms and roots share quantized state. Ordinary
matrices now use CUDA-derived weighted sketches with explicit FP32 scan-order limits.

Explicit tree updater selection, `exact` and distributed training are rejected
for `mps`. Inaccessible Metal devices produce an
error. Other features may use CPU implementations as recorded in the coverage ledger;
selecting `mps` does not imply that every stage runs on the GPU.

## Current Metal operations

| Operation | Implemented coverage |
| --- | --- |
| Approximate training | GPU Hessian extraction, copies and tree operations; cuts before sampling and fresh sampling per tree; both sampling methods/roots share integer state; CUDA-derived raw CSR sketches |
| Gradient state | Shared per-tree integer gradients/scales for both sampling methods, all roots and histograms; GPU target/layout copies and bit-preserving Hessian extraction |
| Histograms | Dense/sparse indices, external pages, scaled integer sums and shared-memory accumulation; resident integer cache retains page additions/subtractions; scalar/vector split scans read integers directly, with a binary64 view for host consumers |
| Split evaluation | Scalar/vector CUDA-order integer scans and persistent children; vector double-precision chunk winners feed FP32 updates, with CUDA numeric/categorical enumeration. GPU feature-winner reduction, parent gains, unscaled final weights, monotonic-bound updates and categorical bitset assembly; GPU-owned interaction masks and monotonic bounds, including initialization/growth; unions and feature filtering on GPU; candidate staging and tree metadata remain host |
| Vector model storage | Bit-preserving GPU copies for parent/child weights and category words, with GPU FP32 learning-rate scaling of new child weights; allocation/growth and metadata remain host |
| Row partitioning | Stable GPU partitions, numeric/categorical/missing values, GPU leaf assignment, resident row indices |
| Bin assignment | GPU numeric/categorical cut search and hit counts for SparsePage-backed matrices; dense 8/16/32-bit compression; explicit MPS quantile adapters also bin on GPU; ordinary in-memory and quantile matrices plus external compressed caches create quantized host mirrors only for explicit CPU batch access; disk-backed ordinary pages retain host bins |
| Feature sketches | Ordinary CSR matrices: GPU row-weight gathering, sorting, FP32 scans, category deduplication, resident summary pruning/merging and ordered rank repair; final cuts materialized on host as in CUDA. Host dense/columnar and native Metal quantile inputs feed resident sketches; distributed sketches remain open |
| Category names | GPU typed dictionary sorting, schema/empty checks, segmented lower-bound recoding maps and first-unknown search; exact integer widths and signed-byte strings. Row-code mapping and native dictionary gathers run on GPU. Model dictionary copies and persistent storage remain host |
| Quantized data cache | Ordinary in-memory DMatrix pages use raw CSR-to-ELLPACK compression. Quantile matrices use direct adapter-to-ELLPACK construction or one-time compression of CPU bins. External host cache adds shared/private ratios and two-page prefetch. Histogram, partition and prediction/SHAP consumers read compressed buffers directly; CPU access explicitly materializes host mirrors. Compressed disk storage, internal ELLPACK save/load and optional native Snappy compression are implemented |
| Native Apple feature tensors | Metal DLPack inputs for ordinary and quantile matrices, external iterators and inplace prediction/CPU fallback; GPU sparse entry packing, strided snapshots, half decoding and PyTorch/MLX synchronization. Native columnar masks and integer/string dictionaries use the same consumers; persistent dictionary ownership remains open |
| Native prediction outputs | Explicit `output_device="mps"` returns owned private Metal results, with legacy/versioned DLPack exports to PyTorch/MLX, independent copy support and explicit host readback. Values, margins, leaves and SHAP outputs are supported; generic prediction state remains host-owned |
| Native custom-gradient imports | CUDA typed casts and GPU pair packing for regular and split/value entry points; real PyTorch/MLX and legacy/versioned capsules. Safe shared destinations or counted copy fallback; generic allocation remains host-owned |
| Native metadata imports | GPU typed copies/conversion and CUDA label/weight predicates into owned Metal vectors; GPU query-ID RLE/prefix scans. Float metadata has lazy host mirrors. Internal row slicing, device copies/stacking and feature-type equality use Metal; group boundaries and group-size prefix sums remain host as in CUDA |
| Generic vector storage | Explicit Metal storage ordinal with CUDA-derived host/device access states; GPU POD fills, copies, growth and concatenation, lazy readback, retained capacity and move ownership. Default execution-context storage and many consumers remain host-side |
| Uniform sampling | CUDA Philox4x64 row masks applied after original-gradient quantization; exact replay for reduced value gradients, both matrix layouts; targets the modern CCCL branch, not the older Thrust stream |
| Gradient-based sampling | Original quantized gradients → binary64 norm terms → FP32 target reduction → integer prefixes/threshold → Philox decisions → binary64 rescaling/requantization; saved norms/threshold replay reduced value gradients; pinned CCCL reduction policy, with NVIDIA compiler/hardware validation still required |
| Gradient quantization | GPU sign-separated binary64 bounds and CUDA scale formula; truncation and minimum-one positive Hessians; scales stay on GPU and feed both integer sampling methods/all roots/histograms. Value-leaf quantizers retain original scales; scalar/vector split statistics retain integers |
| Root statistics / stump fitting | All scalar/vector/hist/approx roots reduce shared integer gradients, including empty-bin roots; final root weights run on GPU; vector cover sums targets in binary64 before FP32 conversion; objective stump fitting retains its separate binary64 GPU reduction and owned Metal input/output storage |
| Reduced-gradient leaf fitting | Original value scales → float sampling replay → per-row integer conversion → integer leaf/page sums; binary64 view for regularized/clipped final weights; host tree receives completed leaves |
| Linear coordinate descent | GPU FP32 sums, CUDA residual/Hessian rules and original selector snapshots; all five feature selectors supported; scalar weight updates, selectors, CSC preparation and linear prediction remain host, as in CUDA |
| Generic sums | CUDA-source FP32 generic reductions with final double return; optional-weight reductions use binary64; scalar default weights need no array dispatch |
| Means | GPU scalar FP32 terms and accumulation; per-target sample and weighted sample means use binary64 terms/reductions; parallel association can differ from CUB; resident inputs and GPU result publication |
| Objective initialization | GPU multiclass packed class-weight histogram, normalization/logarithm/centering, expectile initial gradients/Hessians and absolute-error target-mean broadcast; quantile vector normalization also runs on GPU; these callers retain Metal payload storage, with source-defined scalar returns and expectile/absolute-error host finalizers |
| Objective validation / base scores | GPU logistic, gamma, squared-log, Poisson/Tweedie and multiclass label checks; checked logistic/gamma intercepts and logarithm/logit conversion, including Cox base-score conversion; device label views and owned scalar base-margin storage, with results published only after validation succeeds |
| Quantiles / medians | GPU stable radix sorting, integer prefix scans, CPU-compatible interpolation and sequential FP32 weighted CDFs; multiple quantile levels share one sort; resident inputs and GPU result/empty-NaN publication |
| Gradients | CUDA-source FP32/FP64 expressions for squared error, squared log error, logistic/logitraw, gamma, pseudo-Huber, Poisson, Tweedie, hinge, multiclass and expectile; adaptive absolute-error and quantile gradients preserve source expressions, FP32 residual-root sums and binary64 weight sums; host scale-statistic collective matches upstream; explicit Metal input spans and owned gradient outputs avoid implicit objective-level host mirrors |
| AFT survival | Normal/logistic/extreme gradients for every row, prediction/base-score transforms, likelihood and interval-accuracy metrics on GPU; gradient and transform payloads retain owned Metal storage; source-expression tests cover sensitive tails, whose numerical outputs can differ from CPU. Scalar survival metric consumers now also use resident inputs; AUC/ranking remain separate |
| Prediction transforms | CUDA-source sigmoid, exponential, hinge, cumulative expectile and quantile insertion sorting with preserved subnormals; multiclass softmax uses sequential binary64 sums of FP32 terms, and argmax preserves source NaN/subnormal comparisons; nonranking objective transforms retain Metal vector storage |
| Tree traversal | Batch/inplace/leaf prediction; numeric/categorical splits, missing values, scalar/vector leaves, tree ranges, dropout weights; GPU raw-input decoding and category-map application |
| SHAP contributions / interactions | GPU eight-point quadrature traversal, repeated-feature paths, scalar/vector leaves, categorical/missing splits, dropout weights and bias/interaction finalization; input conversion and row-level category mapping on GPU; raw packing/transfers remain host; approximate SHAP uses an explicit CPU fallback, as CUDA does not implement it |
| Training prediction cache | Leaf-value accumulation, including strided multi-target output and sampled rows |
| Ranking training | GPU query maxima/counts/integer prefixes and immutable cache reuse; stable grouped sorting, NDCG cache/label checks, top-k and CUDA-source Philox mean pairs, raw bounds, independent sign/cost truncation, parallel MAP prefixes, normalization and combined weights on Metal; unbiased bias updates use a deterministic global cost-factor policy |
| ROC-AUC | Binary, multiclass and multilabel; CUDA-source term casts, GPU sort/prefixes/trapezoids, raw curve outputs, aggregation metadata and final normalization; resident inputs and summaries; distributed summary collective still stages through host |
| Grouped ranking ROC-AUC | GPU stable label ordering, merge-based concordant/tied-pair counts, query normalization and aggregation; preserves label ties, short-query exclusions and FP32 squared-weight behavior; resident inputs |
| Column sampling | CUDA-source Philox weighted draws and Thrust Feistel unweighted shuffle; GPU compaction, stable priority ordering and integer feature sorting; 24 shuffle keys and cache metadata remain host as in CUDA, while feature arrays still stage through host |
| Grouped ranking PR-AUC | CUDA-source FP32 cast boundaries, original-order positive totals, segmented binary64 prefixes, integral arithmetic and two-row validity; GPU sorting, validation and aggregation; resident inputs |
| PR-AUC | Binary, multiclass and multilabel; CUDA-source term casts, separate recall/validity totals, source integral and invalid-output rules; GPU aggregation metadata and normalization; resident inputs and summaries; distributed summary collective still stages through host |
| Metrics | Scalar/multiclass and quantile/expectile FP32 row terms, GPU label checks and complete binary64 GPU sums; survival likelihood/accuracy binary64 terms and sums. These scalar callers and wrappers retain Metal inputs; AUC callers now retain inputs and multi-output summaries; precision/NDCG/MAP ownership migration remains pending. Final ratios and collective transport remain host |

AFT gradient evaluation is entirely on GPU, including numerically sensitive
censored rows. `survival_cpu_corrections` and `survival_cpu_time_ns` remain in the
statistics API for compatibility and are now always zero. Source-formula checks
cover extreme tails; cross-library output differences are reported above.

Some generic consumers still request host views. Elementwise and survival
metric inputs now stay on Metal and sums finish on GPU; final ratios and collective transport remain host,
as in the corresponding CUDA call sites. Metal has no native FP64 arithmetic. Split selection uses software binary64 operations and exact
integer scans to preserve histogram precision and CUDA candidate ordering. Tolerances and
numerical edge cases are tested explicitly.

`xgb.metal_info()` reports completed dispatch counters for each operation family.
It also reports GPU durations and histogram staging/wait/readback wall times. Wait
time overlaps GPU duration; adding those numbers would double-count time. Counters
are process-wide, so concurrent models contribute to the same totals.
`metric_reduction_passes` counts completed final reduction levels;
`metric_final_readback_bytes` is 16 bytes per nonempty successful scalar,
multiclass, quantile/expectile or survival metric call. `metric_label_dispatches`
counts GPU multiclass validation, including calls rejected for invalid labels.
`metric_shared_input_bytes` and `metric_host_input_bytes` count eligible bindings
and fallback uploads for scalar/multiclass, quantile/expectile, survival and AUC
metric inputs (including grouped AUC and normalization summaries). They exclude
final scalar results, precision/NDCG/MAP inputs and whole-training transfers.
`metric_shared_output_bytes` / `metric_host_output_bytes` count AUC summary
publication, including raw binary triples; `metric_copy_dispatches` /
`metric_copy_bytes` count its GPU blits. `auc_final_readback_bytes` counts explicit
8-byte normalization returns and 16-byte grouped returns; it excludes raw output
publication and optional diagnostics. These counters count bindings, not ownership;
native vector access-state/readback checks establish residency separately.
`gradient_sampling_dispatches` and `gradient_sampling_gpu_time_ns` report the
gradient-based sampler separately; they are also included in the general sampling
counters and must not be added to those totals. `cuda_gradient_sampling_dispatches`
counts the new integer sampler once per nonempty sampled tree; float replay shares
its saved state and does not increment that counter.
`ranking_roc_auc_dispatches` and `ranking_roc_auc_gpu_time_ns` likewise describe a
subset of ROC-AUC and generic metric work; their timings overlap those totals.
The corresponding `ranking_pr_auc_*` counters are subsets of PR-AUC and metric totals.
`ranking_plan_dispatches` counts completed query-metadata builds; `ranking_plan_reuses`
counts validated uses of a supplied immutable cache. `ranking_plan_gpu_time_ns`
measures metadata commands, separately from subsequent sort/gradient commands.
`cuda_sketch_batches` counts nonempty raw-entry windows (64M entries maximum,
matching CUDA). `cuda_sketch_input_entries` counts their input entries, while
`cuda_sketch_dispatches` and `cuda_sketch_gpu_time_ns` count completed commands
and GPU duration. These timings exclude staging, allocation, waits and host cut
materialization. `sketch_weight_dispatches` also covers group/Hessian weighting.
The older `sketch_sorted_*`, `sketch_update_*` and `sketch_finalize_*` counters
belong to CPU-compatible summary helpers; ordinary MPS training bypasses them.
`expectile_init_dispatches` and `prediction_init_dispatches` expose the new
objective-specific initialization work. `small_histogram_dispatches` and
`vector_arithmetic_dispatches` count the new class-run histogram and vector paths;
all four contribute to `initialization_gpu_time_ns`, separately from ordinary
objective GPU time. `float_mean_dispatches` is a subset of `mean_dispatches`,
whose duration remains in `mean_gpu_time_ns`; do not add these counts together.
`initialization_shared_input_bytes` / `initialization_shared_output_bytes` count
eligible payload bindings. `initialization_host_input_bytes` /
`initialization_host_output_bytes` count compatibility staging for unshareable
views. GPU publication/snapshot copies and empty-result fills are reported by
`initialization_copy_dispatches`, `initialization_copy_bytes` and
`initialization_fill_dispatches`; their GPU duration also contributes to
`initialization_gpu_time_ns`. These are scoped binding/copy counters, excluding
control arguments and source-defined scalar returns/status checks; they do not
measure whole-training transfers.
`label_validation_dispatches` / `label_validation_gpu_time_ns` and
`base_score_dispatches` / `base_score_gpu_time_ns` report objective label checking
and intercept conversion separately. Invalid checked inputs still count completed
GPU work. Empty inputs, identity conversions and unconditional label predicates
require no dispatch.
`category_sort_dispatches` and `category_recode_dispatches` count completed
category-name sorting and dictionary validation/mapping commands;
`category_gpu_time_ns` measures their GPU time. Schema or unknown-category errors
still count completed work. Empty sorts and zero-feature dictionaries dispatch
nothing; nonzero-feature empty dictionaries still receive GPU schema validation.
These counters exclude host dictionary packing, row-level recoding and staging.
`input_prepare_dispatches` counts completed prediction/SHAP preparation commands;
`input_recode_dispatches` counts those with a nonempty row mapping.
`input_prepare_gpu_time_ns` excludes raw packing, uploads and synchronization.
`input_debug_readback_bytes` counts explicit diagnostic reads of prepared inputs;
production prediction/SHAP leave it unchanged. Sparse validation failures still
count completed GPU work; empty input dispatches nothing.
`adapter_prepare_dispatches` counts GPU count/scan and compaction commands;
`adapter_index_dispatches` counts global row-pointer output. Binning uses its
existing counters. `adapter_gpu_time_ns` excludes packing, transfers and host
storage. `adapter_debug_readback_bytes` tracks raw compacted-input diagnostic
reads; production leaves it unchanged. Quantized bins/row pointers still return
to existing host storage.
`shap_dispatches` and `shap_interaction_dispatches` count completed contribution
and interaction batches; `shap_gpu_time_ns` covers both. These are separate from
ordinary prediction and input-preparation counters and exclude packing/transfers.
`leaf_sum_init_dispatches`, `leaf_sum_dispatches` and `leaf_weight_dispatches`
report GPU clearing, per-page leaf reductions and final weight calculation.
`leaf_integer_sum_dispatches` counts successful quantized page reductions within
the leaf-sum total. `leaf_fitting_gpu_time_ns` includes original value-bound
calculation and those stages, separately from root sums, histogram
work and row partitioning; source sharing/staging and final host tree updates are
excluded.
`linear_gradient_dispatches` and `linear_residual_dispatches` count completed
coordinate-descent reductions and updates; `linear_gpu_time_ns` measures those
commands. Host feature selection, scalar model updates and staging are excluded.
`histogram_integer_store_dispatches` counts integer cache writes, with later-page
additions also counted by `histogram_integer_accumulation_dispatches`.
`histogram_integer_subtraction_dispatches` and `histogram_integer_sum_dispatches`
identify integer subtraction and root-bin reductions. These overlap their
histogram/conversion/subtraction/gradient-sum operation totals.
`cuda_integer_split_dispatches` and `cuda_integer_multi_split_dispatches` identify
the production scalar/vector CUDA-order split kernels. They are subsets of
`split_dispatches` and `resident_split_dispatches`, not additional work.
`split_finalize_dispatches` counts GPU feature reduction/weight calculation,
including root-only calls; `split_bounds_dispatches` counts completed monotonic
bound updates. These are additional commands whose GPU time is included in
`split_gpu_time_ns`; they do not increment candidate-evaluation counters.
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

- Metal adapter construction: **65 native cases / 2,184 windows / 1,126,724 exact
  fields** against independent CUDA sketch formulas, raw records, bins and row
  offsets. **58 public tests** cover streamed/typed inputs, weights, references,
  category recoding, errors and the real 67-million-slot boundary.

- Raw prediction input: **144 native cases / 6,677,824 values**, exact bits for
  non-NaNs and NaN classification; ten numeric types, signed/broadcast strides,
  all eight validity-mask bit offsets, missing sentinels, sparse bounds and
  8/16/32-bit bins. **71 new public tests**, plus existing category tests, require
  GPU preparation and no dense-input readback.

```sh
bash scripts/test.sh
```

On Apple M3, validated through 2026-11-04:

- Python: **1,278 passed, 1 skipped**. The skip requires a machine without an
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
- Legacy low-level LCG sampling (no longer used by production uniform training): **3,780 exact cases / 1,019,088 Bernoulli draws** against the
  CPU standard-library implementation, including zero seeds, rounding boundaries,
  strided matrices, guard regions and CPU thread partitions.
- CUDA Philox uniform sampling: **1,060,276 exact words/draws**, CCCL's published
  engine check value, and 64-bit skip boundaries. Shared state: **228 native cases /
  4,319,326 pairs**, root cancellation, target scales, replay and ownership. **60
  new Python cases** check all root layouts and quantization before uniform rejection.
- Legacy low-level gradient sampling: **498 exact cases / 6,748,668 rows**, comparing
  norms, thresholds, masks and rescaled gradients to production CPU routines and
  libc++ random draws. Includes subnormals, overflow, equal gradients, zero/full
  budgets, strided storage and reduced-gradient replay. Production training now uses
  the CUDA-derived path instead.
- CUDA-derived gradient sampling: **136 native cases / 4,499,447 pairs**, plus
  54 production lifecycle calls and seven explicit rescaling-domain rejections.
  Independent checks cover 200,081 norm FMAs, 100,144 integer conversions, 91 wide
  target reductions, recursive integer prefixes and 86,268 rescaled integer pairs.
  Eight added public forest cases check exact integer-derived root predictions.
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
  subnormals and wide exponent ranges. These retain low-level binary64 reduction
  coverage; production scalar means now use the CUDA-derived FP32 path tested
  separately by the class-prior/vector suite below.
- Regression/hinge/expectile gradients and scalar/row transforms: **624 cases /
  14,305,782 fields** against extracted source expressions, including **545,836
  subnormal fields**, 25,205 double-denominator witnesses and 6,089 double-hinge
  witnesses. Fifteen public transform cases also pass; native-libm differences
  are recorded separately. [Validation](validation/cuda-objective-arithmetic-m3.txt).
- Multiclass gradients/transforms: **192 cases / 12,181,658 exact fields**,
  including 2,828,230 nonfinite classification checks, 1,236 compensated-FP32
  denominator witnesses, alias/guard checks and empty/invalid-shape handling.
  Thirty public prediction cases also pass; original CPU-equivalence tests retain
  their tolerances. [Validation](validation/cuda-multiclass-softmax-m3.txt).
- Class priors/vector arithmetic: **258 cases / 10,499,275 fields**, with 660,150
  packed-class witnesses and 108,340 cases distinguishing double scalar products.
  Vector results match the sampled host source formulas at observed 0 FP32 ULP;
  FP32 reduction errors stay below 0.107 of the documented association bound.
  **24 new public tests** cover initial scores, missing classes, invalid labels,
  reload and weight errors. [Validation](validation/cuda-multiclass-init-m3.txt).
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
- Integer histogram cache: **75 page/subtraction cases**, **5,013,975 stored or
  subtracted pairs**, **225 integer root sums**, and **48 production API cases**
  covering growth, reuse, empty first pages, cancellation and guards. Eight new
  training cases require exact predictions across in-memory/external pages,
  including subnormal leaf values.
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
- AFT source replay: **96 cases / 316,683 gradient pairs / 3,166,830 exact source
  fields**, plus **1,900,098 log/exp/erf fields** within four binary64 ULP of host
  libm (observed maxima **2/1/3**). Every production result matches the raw GPU
  formula, with zero CPU corrections. Includes all previous stress rows and
  additional NaN/Inf, invalid-scale, signed-zero and domain-guard combinations.
  The **75-case / 307,275-pair** public-objective stress corpus reports 322 sensitive
  CPU differences; well-conditioned and ordinary integration tolerances remain
  unchanged. [Validation](validation/cuda-aft-device-m3.txt).
- Shared host-memory views: **36 cases / 4,313,492 values** plus six fallback
  checks under Metal API Validation. **51 production objective API cases /
  1,693,795 gradient pairs** cover offsets, guards, read-only inputs, split VM
  regions and overlapping input/output storage. **49 production leaf cases /
  1,118,257 row positions** add page offsets, strided/subnormal Hessians and empty leaves.
- Classification ROC/PR-AUC: **486 cases / 7,607,088 score-label pairs per curve**,
  420 independent weighted-pair checks and 519 independent integral checks.
  CUDA-source cast/output fixtures, three-level scans, fractional labels, strides,
  subnormal/extreme weights and degeneracy are covered. **25 GPU normalization /
  metadata cases** include 65,535 outputs and three shape errors; **97 public tests**
  cover CPU/source agreement, reloads and GPU normalization dispatch. Extreme PR
  uses exact-term and bounded-prefix/source-arithmetic checks: 1,690,464 terms and
  prefixes each, 78 curves including 15 NaNs. The old stabilized PR accuracy
  contract has explicitly changed to CUDA source behavior; see the
  [validation record](validation/cuda-classification-auc-m3.txt).
- Grouped ranking ROC-AUC: **272 production cases / 234,208,157 independently
  enumerated pairs**, compared with both upstream CPU and an independent pair
  reference. Three additional million-row analytic cases each represent
  **500,002,500,003 pairs**, testing 64-bit counts without a quadratic CPU run.
  Covers label/score ties, padded strides, zero/tiny/large weights, infinities/NaNs
  in scores, short queries and output guards. **390 additional source-contract
  queries**, plus an overflow fixture, cover **2,168,663 enumerated pairs**, NaN and overflowing squared weights,
  subnormal half-weight rounding and GPU counters. Finite squared-weight overflow
  now returns NaN, matching CUDA source. All **73 public grouped AUC tests** pass.
  [CUDA-source validation](validation/cuda-grouped-roc-m3.txt).
- Mean-ranking pair sampling: **74 production cases / 9,068,696 exact raw pairs**;
  **41,200 RNG cases / 659,200 exact draws and consumed positions**, including
  large discard offsets. **83,328 draws/positions** also match pinned upstream
  CCCL on the host. Existing gradient/bias datasets use CUDA-pair/shared-math
  oracles for both methods, with unchanged tolerances. The previous unrounded
  mean reference has been replaced by the full arithmetic oracle below.
  [Validation](validation/cuda-ranking-pairs-m3.txt).
- Elementwise/survival metric reductions: **349 production cases / 15,628,618
  exact FP32 row-result fields**, plus **100,057 math inputs / 500,285 fields**.
  Covers subnormals, NaNs/infinities, overflow, fractional multiclass labels,
  signed cancellation, multiple targets/levels and recursive reduction boundaries.
  Only the final binary64 pair returns to host. [Validation](validation/cuda-metric-reductions-m3.txt).
- Ranking metadata: **180 native cases / 12,611,280 exact task fields**, including
  **131,415 integer prefixes not representable in binary64**. Covers empty queries,
  256-entry scan boundaries, recursive scans through 262,145 queries, maximum
  uint32 row/pair parameters, exact cached/fresh gradients with three seeds, retained
  buffer ownership and stale-input guards. [Validation](validation/cuda-ranking-metadata-m3.txt).
- Ranking gradient/cost arithmetic: **162,851 direct rounding cases / 977,106
  exact fields**, plus **64 production diagnostics cases / 1,600 bounds and
  multipliers / 2,113,664 MAP prefix fields**. **1,442 gradient cases / 2,444,200
  pairs** match the host CUDA-formula reference bit for bit. **2,592 unbiased
  calls / 118,368 bias values** and four additional 524,288-row batched calls
  pass; the largest scaled gradient error is `1.70e-7`, serialized bias error zero.
  The global cost-factor race and nonfinite `frexp` inputs have explicit policies;
  these results do not prove NVIDIA execution equivalence.
  [Validation](validation/cuda-ranking-arithmetic-m3.txt).
- Column sampling: **550 native cases / 3,281,095 selected feature IDs /
  4,610,295 exact Philox draws / 1,320,896 exact shuffle indices**, checked against
  CUDA-source references. An optional actual Thrust/CCCL host check verifies
  66 permutations, 434,730 indices, 144 shuffle keys and 60,000 draws. Native
  weighted keys match the local Apple log reference exactly, including constructed
  ties; this is not a libdevice rounding guarantee. Hierarchy, seed consumption,
  guards and GPU counters are checked. **48 added public cases** verify selected
  tree-root features independently; existing sampled scalar/vector fixtures use
  source arithmetic and same-model CPU prediction checks with unchanged tolerances.
  [Validation](validation/cuda-column-sampling-m3.txt).
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
- Legacy binary64 leaf-sum API: **304 native cases / 24,327,036 source gradient
  pairs**, 46,852 exact sum fields and 156,240 bitwise CPU weight comparisons
  (NaNs checked by classification). Covers strided/broadcast inputs, paged sums,
  empty leaves, subnormals, large/nonfinite values, L1/L2 penalties, clipping,
  bounds and guards. **66 public tests** cover training and saved-model
  continuation across dense, missing, sparse and external-page inputs.
- CUDA-derived integer leaf sums: **250 native cases / 11,594,018 source pairs**,
  20,458 exact materialized sum fields, original-scale retention, float rescaling,
  positive-curvature minimums and conversion boundaries. **24 additional public
  cases** check exact cancellation, sampled value scales, zero-bin/external-page
  roots and prediction/cache/reload behavior.
- Linear coordinate descent: **49 native cases / 4,118,809 source pairs**, 3,564
  FP32 sum checks (2,991 exact/nonfinite checks; others use a derived rounding-error
  bound), and 148,277,125 residual fields checked against CUDA-source arithmetic.
  Finite residuals match bits; NaNs use classification. Covers empty/million-row
  inputs, duplicates, subnormals, nonfinite values, negative Hessians and guards.
  **158 public cases** cover selectors, objectives/layouts, model continuation,
  explicit CPU/CUDA selector differences and original-gradient preservation.
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
  results**, plus 16 edge/cast checks against CUDA-source formulas and an independent
  continuous-integral reference; compatible CPU checks retained.
  Includes irregular/empty queries, recursive scan carries, widely differing query
  weights, fractional/subnormal labels, degenerate queries, strided storage and
  six GPU label-validation errors. Maximum observed reference difference is
  `7.883e-15`; tiny tied-score cases also receive relative-accuracy checks.
- Native objectives: **20 cases / 131,968 gradient pairs** compared against
  independent CPU mathematical references.
- Category-name dictionaries: **33 native cases / 2,730,917 exact indices**,
  plus **80 expected errors**, across every supported integer/string type.
  **19 public tests** verify GPU routing, reordered/subset categories, unknown
  names, global schema-error precedence, predictions, SHAP and model reload.
- Native adaptive objectives: **180 cases / 1,260 residual scales / 452,592 gradient
  pairs**, covering zero/tiny weights, multiple targets and response scales.
  New extracted-source checks cover **2,359,332 fields**, with **1,000,800**
  independently checked hypot/tanh fields (maximum observed error: zero FP32 ULP).
  Production objective tests cover **308 worker calls / 8,815,888 result checks**,
  including empty/global partitions, 65,537-row recursive reductions and FP32
  overflow distinguished from binary64 optional-weight sums.
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
[quantile results](validation/benchmark-quantile-v3-m3.json). These measurements
predate the CUDA-source adaptive-objective correction: residual roots now
accumulate in FP32 and gradients use subnormal-preserving math. The old checked
integer/binary64 residual reductions do not describe the current implementation;
no new performance measurement has been made for it. In the preceding
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

Historical measurements of the earlier version with CPU corrections; these have
not been rerun after removing that path.

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
AFT training runs were slower than CPU. The current implementation removes CPU
corrections; these measurements do not establish its performance.

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
- [x] Implement CUDA-derived gradient-based integer sampling and replay; NVIDIA execution validation remains.
- [x] Port uniform/weighted column sampling with CUDA-source Philox and Thrust shuffle streams.
- [ ] Port remaining training-state updates.
- [x] Port final numeric sketch level merging and pruning.
- [x] Port sketch weighting, sorted-column summaries and streaming level compaction.
- [x] Port exact SHAP contributions and interactions.
- [x] Implement Metal coordinate-descent reductions and residual updates.
- [x] Align linear FP32 arithmetic and selector snapshots with CUDA source.
- [x] Align precision/NDCG/MAP arithmetic, prefixes and degenerate-query weighting with CUDA source.
- [x] Align grouped PR-AUC term rounding, recall totals, integral and query validity with CUDA source.
- [x] Align classification AUC arithmetic/output rules and port multi-output normalization to GPU.
- [x] Port CUDA raw CSR sketching and resident summaries for ordinary hist/approx matrices.
- [x] Add explicit MPS QuantileDMatrix decoding, resident sketching and bin assignment.
- [x] Add explicit MPS disk-backed ExtMemQuantileDMatrix construction and cache lifecycle checks.
- [x] Add GPU ELLPACK pages, adaptive shared/private external caching and page concatenation.
- [x] Construct external host-cache ELLPACK pages directly from resident GPU adapter windows.
- [x] Read compressed external-cache pages directly in histograms, partitions and prediction/SHAP input preparation.
- [x] Create external-cache host mirrors only for CPU access; preserve compressed views across prefetch/device switches.
- [x] Keep in-memory quantile matrices compressed, with one-time CPU conversion and stable concurrent CPU/GPU views.
- [x] Compress ordinary in-memory DMatrix raw CSR directly; retain separate CPU views and cache-generation rules.
- [x] Port compressed MPS-origin disk caches and internal ELLPACK save/load.
- [x] Complete ordinary external host-cache selection.
- [x] Port optional Snappy cache compression under the native opt-in.
- [x] Add native Metal feature/metadata/gradient tensors and columnar masks/dictionaries.
- [x] Add owned GPU prediction outputs and PyTorch/MLX DLPack consumers.
- [x] Add generic Metal vector storage and internal metadata slice/copy/stack/equality operations.
- [x] Keep nonranking objective inputs/gradients/transforms in Metal vector storage.
- [x] Keep initialization/statistics/stump payloads on Metal through their GPU stages.
- [x] Keep scalar/multiclass, quantile/expectile and survival metric inputs on Metal.
- [ ] Migrate remaining state allocations/consumers and persistent dictionary ownership.
- [x] Port CUDA ranking gradient/cost arithmetic with explicit edge policies and source-oracle tests.
- [x] Port query maxima/counts/prefixes and reuse GPU metadata through ranking caches.
- [x] Port remaining elementwise/survival metric sums and CUDA FP32 row arithmetic; validate multiclass labels on GPU.
- [x] Evaluate every AFT gradient row on GPU; validate upstream expressions and report sensitive-tail numerical differences.
- [x] Port multiclass class-prior initialization, shared vector arithmetic and scalar FP32 mean accumulation.
- [ ] Finish quantile construction, objective coverage and generic state operations.
- [ ] Complete asynchronous stream interoperability and distributed GPU collectives.
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
are amplified by cancellation in tiny interval probabilities. The initial version
marked those rows for CPU recomputation, with visible correction counts and timing.
That policy has since been replaced by [GPU evaluation with source-expression
validation](validation/cuda-aft-device-m3.txt), including reported CPU differences.
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

**Historical checkpoint:** the CPU random stream and sequential sampler described
below have been replaced by [CUDA-source mean pairs](docs/CUDA_PARITY.md#ranking-gradient-numerical-behavior).
These benchmarks retain their original results and are not measurements of the
current sampler. CPU-trained model identity is no longer the mean-ranking contract.

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

Historical CPU-formula implementation; production now uses the CUDA-derived
integer sampler described in the latest update below. These performance results
have not been rerun for the new sampler.

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

Later CUDA-source alignment changes the overflow case from an error to NaN;
see [the current contract](docs/CUDA_PARITY.md#grouped-roc-cuda-source-contract).

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

This historical update used CPU random streams. They have since been replaced by
[CUDA-source sampling](docs/CUDA_PARITY.md#column-sampling-behavior); the timings
below describe the earlier implementation and have not been rerun for the new one.

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
`shotgun` algorithm also remains host. At this checkpoint the numerical reference
was CPU, including refreshed residuals for greedy/thrifty selection. The later
CUDA-alignment update below replaces those rules.

[Validation](validation/linear-coordinate-m3.txt): **1,066 Python tests pass (one
skipped)**, all native suites pass, and the separate CPU-only build/check passes.
New coverage includes 136 public cases and 49 native cases using 4,118,809 source
gradient pairs, with Metal API Validation enabled for the native linear tests.
The tests cover selectors, multiple outputs, missing/sparse inputs, L1/L2
penalties, saved-model continuation, million-row reductions, duplicate CSC rows,
exceptional arithmetic and buffer guards.

Coverage and correctness remain the priority; no new performance measurements
were run. At that checkpoint, sketch ingestion/preprocessing, `approx` training,
sensitive AFT corrections, generic state/metadata operations, interoperability
and distributed execution remained. Later updates address `approx`, ordinary
matrix sketches and AFT corrections; full CUDA parity is still incomplete.


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
sampling/RNG, quantization, leaf sums and linear-selector choices needed alignment
at this checkpoint. Subsequent updates below address several of them; current gaps
are tracked in the [ledger](docs/CUDA_PARITY.md).
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


### Integer histogram cache alignment

Histogram page accumulation and subtraction now preserve CUDA's quantized integer
sums. A binary64 view remains for split evaluation; cache arithmetic uses the
integer source. Growth preserves both buffers, and empty first pages clear reused
slots on GPU. Dense scalar hist roots also sum integer bins before conversion.
Persistent histogram storage is temporarily 32 bytes per bin for the two views.

[Validation](validation/integer-cache-m3.txt) covers over five million stored or
subtracted pairs, 48 production-cache cases, and eight new training cases that
retain small terms through cancellation. The subnormal cases exposed and fixed
prediction/cache additions flushing valid leaf values to zero. Cached, in-place
and reloaded predictions now match the exact reference. The full validation run
passes **1,218 Python tests (one skipped)**, every native suite and the CPU-only
build/check. No tolerance was widened and no performance benchmark was run.

At this cache checkpoint, quantization order and shared root state remained
unfinished; later updates below implement shared state, both sampling methods,
integer reduced leaves and scalar/vector integer splits.


### Shared quantized state and CUDA uniform sampling

Uniform training now follows CUDA's order: quantize original gradients, sample
integer pairs, then share those exact integers with histograms and every root
reduction. Sparse, missing, vector, `approx`, external-page and zero-bin roots
use this state. A multi-target slice-extent bug found by the new tests is also
fixed; quantizer bounds no longer include later targets' backing storage.

Uniform masks use CUDA's modern CCCL Philox4x64 branch, including integer-to-float
rounding, the inclusive comparison, seed consumption and reduced-gradient replay.
Same-seed models intentionally differ from the old CPU-oriented sampler. The older
CUDA/Thrust random stream is not claimed equivalent. The pinned source and precise
scope are in the [coverage ledger](docs/CUDA_PARITY.md#shared-integer-training-state-and-cuda-uniform-sampling).

[Validation](validation/shared-quantized-state-m3.txt) adds over a million exact
Philox words/draws, 228 production-state cases over 4.3 million pairs, and 60 public
root/order/forest cases. Integration tests check sampled trees against CPU custom
objectives supplied with independent CUDA-derived masks, at unchanged tolerances.
The full run passes **1,278 Python tests (one skipped)**, all native suites, and
the independent CPU-only build/API check. No NVIDIA execution comparison or new
performance benchmark was performed.

At this uniform-sampling checkpoint, gradient-based sampling still needed integer
norms, prefix sums and rescaling; the following update implements that pipeline.
At that checkpoint, leaf sums, split scans, column/ranking RNG, sketches and other
ledger gaps remained. Integer reduced-leaf sums are implemented below.


### CUDA-derived gradient-based integer sampling

Gradient-based sampling now starts from original quantized gradients and retains
those scales through norm calculation, threshold selection, Philox decisions and
integer rescaling. Histogram/root calculations consume the resulting integers;
reduced value gradients replay saved decisions in FP32. This removes the previous
CPU-specific norms, ordered float prefix, threshold fallback and random stream
from production MPS sampling.

The implementation models a pinned CCCL reduction policy and an explicit binary64
contraction choice. Native tests check each stage against independent host formulas,
including undefined conversion boundaries, while training tests retain their
existing prediction tolerances. See the [scope and limitations](docs/CUDA_PARITY.md#cuda-derived-gradient-based-sampling)
and [validation](validation/cuda-gradient-sampling-m3.txt). Actual NVIDIA compiler/
hardware comparison remains required; this is not measured CUDA equivalence.
No new performance measurements were run.


### CUDA-ordered integer reduced-leaf sums

Value-gradient leaf fitting now retains quantizer scales from the original values,
replays sampling in FP32 before tree growth, and quantizes sampled float values with
those original scales. GPU leaf reductions and page accumulation preserve integers;
a separate binary64 view supplies final regularized weights. This follows CUDA's
value-gradient path, which differs from its integer split-gradient sampling path.
The scale-only factory avoids allocating an unused integer array for every value row.

[Validation](validation/integer-leaf-sums-m3.txt) processes over 11.5 million native
value-gradient pairs and checks exact leaf sums, original scales, source lifetime,
strides, guards and conversion boundaries. New public cases preserve cancellation
across pages and the original positive-Hessian minimum after sampling rejects an
outlier. The full run passes **1,310 Python tests (one skipped)**, all native suites
and the independent CPU-only build/API check, without widening tolerances.

The previous binary64 low-level API retains regression coverage; production reduced
leaf fitting uses the integer path. At this checkpoint, split scans and child
statistics still required integer alignment; the next update addresses scalar
splits. Other ledger gaps and actual NVIDIA execution validation remain;
no new performance measurements accompany this checkpoint.


### CUDA-order scalar integer splits

Scalar `hist` and `approx` training now preserve integers through histogram scans,
missing-value subtraction, candidate selection and persistent child-node statistics.
Numeric scans are exclusive, missing-direction gain ties go right, and categorical
partitions retain CUDA's stable ordering and enumeration. Feature selection uses
raw child gains before subtracting parent gain. Tiny category weights/gains use
IEEE-preserving comparisons after exact tests exposed Metal's native flush-to-zero
behavior. Host tree orchestration remains; the following update addresses
vector-leaf splits.

Independent native checks cover 792 cases, including 72 cancellation cases. New
public tests audit complete trees against CUDA-derived host formulas, including
cache eviction, external pages, constraints, sampling and deeper child cancellation.
Existing scalar column/sampling/cache tests use those references where CPU split
rules differ. Strict CPU comparisons remain for compatible positive-weight
fixtures; zero-weight coverage remains in CUDA-source audits. The initial full run
exposed 28 CPU-model-identity failures; no tolerance was widened to hide them.
Final validation passes **1,395 Python tests (one skipped)**, all native suites and
the CPU-only build/API check. A final oracle improvement also checks premature
leaf stops; all 222 affected tests pass.

[Scope and remaining work](docs/CUDA_PARITY.md#scalar-cuda-order-integer-splits).
This is local source-derived validation, not measured NVIDIA equivalence. No new
performance benchmark was run.


### CUDA-order vector integer splits

Multi-output tree splitting now reads original integer histograms and per-target
scales. Root and selected child integers persist by node ID across histogram-cache
reuse and loss-guide growth. Numeric scans are inclusive in both directions;
one-hot candidates evaluate all missing-right choices before missing-left choices;
partitions retain stable projected category weights and CUDA prefix/suffix ordering.

CUDA selects candidates in binary64 within each 32-bin chunk, rounds that winner
to FP32, and compares those rounded chunk winners. The new Metal kernel preserves
both stages, including a test where either a global-double or per-bin-float maximum
would choose the wrong split. Feature ties prefer the lower feature ID; the
binary64 parent gain is subtracted only after feature selection.

Direct tests pass **1,528 cases / 76,102 exact child pairs**, including up to 129
targets, cancellation, tiny values, constraints and buffer guards. **94 new public
tests** audit whole trees and final leaves, including separate reduced-value
quantizers and float sampling replay. One previous CPU topology comparison exposed
a legitimate categorical missing-direction tie difference: vector split tests now
check the CUDA-source reference, while retaining their numerical CPU prediction
comparison at the original tolerance. The focused run passes all 118 cases.

At this vector checkpoint, feature-winner reduction, final weights and constraint/
interaction metadata still ran on the host; the next update addresses finalization
and monotonic bounds. Extreme nonfinite behavior, NVCC contraction and actual NVIDIA
execution remain unverified. Final validation passes **1,489 Python tests (one
skipped)**, all native suites and the CPU-only build/API check.
[Validation record](validation/cuda-integer-vector-splits-m3.txt). No performance
benchmark was run.


### GPU split finalization and constraint bounds

Scalar and vector training now use a Metal finalizer to select the winning feature,
subtract parent gain at CUDA's precision boundary, calculate bounded/pooled node
weights, and sum vector Hessian traces. It reads the existing GPU candidate buffers.
Scalar feature ties preserve sampled order; vector ties choose the lower feature
ID. Weights and statistics persist by node ID until the queued split is applied.
Root weights also use the GPU when there are no candidate features.

Monotonic bounds now update on the GPU, preserving the distinct scalar/vector
midpoint formulas and IEEE subnormal arithmetic. Guarded synchronous views share
host-owned bounds when possible, with a copied-buffer fallback. Weight inputs are
snapshotted before writes so aliases are safe. Root vector covers now accumulate
in binary64 before narrowing; regression cases expose the small contributions
lost by the previous CPU-style FP32 target accumulation.

Direct checks pass 448 nodes, 11,904 target weights and 90 production bound updates.
Thirteen new public cases verify dispatch coverage, whole-tree formulas, bound-update
counts, empty-feature roots, large target counts and many-feature selection.
Final validation passes **1,502 Python tests (one skipped)**, every native suite
and the CPU-only build/API check. [Validation](validation/split-finalization-m3.txt).

This is still a synchronous hybrid backend. At this checkpoint, interaction masks,
staging and tree metadata remained on the host; interaction masks are addressed below. Vector model copies, learning-rate scaling and
category bitset assembly were subsequent gaps, addressed below. Candidate
readbacks have not yet been removed or fused with finalization. No benchmark was run.


### Vector model-buffer copies and learning-rate scaling

Vector tree expansion now uses Metal blit copies for parent/child weights and
category words, followed by a GPU multiply for each new child weight. This follows
`multi_target_tree_model.cu`. Copies preserve every bit; scaling rounds once to
FP32, including subnormal results, signed zero and overflow. Unscaled parent
weights and existing untouched nodes remain intact. Sources aliased to the old
model are snapshotted before storage growth can invalidate them.

The native suite passes **81 copy cases, 80 scale cases / 1,053,248 products, and
16 growing trees**. Nineteen new public tests check categorical/numeric trees,
both growth policies, zero/ordinary/unit learning rates, model reload and resumed
training, GPU operation counters and complete CUDA-source tree formulas.
Final validation passes **1,521 Python tests (one skipped)**, all native suites
and the separate CPU-only build/API check. [Validation](validation/model-state-m3.txt).

This moves the CUDA copy/scale operations to Metal. Model storage is still owned
by host vectors, with guarded shared views or staging; allocation and persistent
GPU ownership remain unfinished. Category bitset assembly was the next gap,
addressed below; interaction masks are covered by the subsequent update. No performance benchmark was run.


### Categorical split bitset assembly

Scalar and vector training now assemble categorical split bitsets on Metal,
using resident candidate results and sorted category indices. The kernel clears
reused storage and sets one-hot or partition bits with CUDA's word layout.
Production training no longer reads the sorted indices back to the host.

Native checks pass **864 category slots / 33,738,112 exact words / 16 successful
API calls**, covering word boundaries, sparse IDs through `2^24 - 1`, invalid
inputs and dirty-buffer reuse. Twenty-four public tests cover both split modes,
scalar/vector trees, both growth policies, missing values and GPU dispatch
coverage, with whole-tree source oracles and CPU-loaded prediction checks.
Final validation passes **1,545 Python tests (one skipped)**, every rebuilt native
suite and the CPU-only build/API check. [Validation](validation/split-categories-m3.txt).

Bitsets still return to host-owned tree metadata. At this checkpoint, interaction
masks and their CUDA algorithm alignment remained unfinished; the update below
addresses them. Category recoding and other state operations remain. No performance
benchmark was run.


### CUDA feature-interaction constraints

Scalar and vector trees now keep interaction masks in Metal-owned buffers. GPU
operations reset/grow those buffers, union the groups associated with each split
feature, propagate masks to children, and filter sampled features in ascending
order. Root queries retain the original feature list, matching CUDA.

This also fixes a CPU/CUDA algorithm difference. With groups `[[0,1],[1,2]]`,
CUDA allows feature 2 after splits on 0 and 1; the CPU implementation excludes
it. The Metal path now follows CUDA. Independent source references and new
training fixtures explicitly check that difference, without changing tolerances.

Focused validation passes **834 native checks and 30 public training tests**,
including scalar `hist`/`approx`, vector trees, numeric/categorical splits, both
growth policies, repeated trees, and resumed models with changed constraints.
Native tests cover empty/duplicate groups, disabled/root bypasses, word and scan
boundaries through 65,537 features, buffer growth, reset and invalid-input recovery.
Final validation passes **1,575 Python tests (one skipped)**, every rebuilt native
suite and the CPU-only build/API check. [Validation](validation/interaction-constraints-m3.txt).

Host parsing and CSR metadata construction match CUDA's setup. Feature lists still
return to host candidate preparation; this is not fully device-resident training.
Monotonic-bound allocation/initialization was the next gap, addressed below. Other
ledger gaps remain. No performance benchmark or NVIDIA execution comparison
accompanies this step.


### GPU-owned monotonic bounds

Monotonic constraints now retain lower/upper bounds in Metal-owned buffers. GPU
initialization fills the new entries with CUDA's finite `-FLT_MAX`/`FLT_MAX` values;
growth copies old bounds on the GPU and initializes only the new tail. The existing
scalar/vector bound-update kernel writes directly to those buffers, eliminating
bound staging and copy-back for this path. Host candidate preparation reads a
completed shared-memory view.

The source is CUDA `HostDeviceVector::Resize` and `DeviceUVector::resize`, as called
by `TreeEvaluator`. Feature-constraint parsing/padding stays on the host, as in
CUDA. This change adds GPU allocation/initialization coverage; it does not change
bound-update arithmetic or claim fully device-resident training.

Focused checks pass **24 native storage cases / 76 updates / 3,847,176 exact values**
and **19 public cases**, including six new deep-tree cases that exceed the initial
allocation. Tests cover retained values, finite tail defaults, subnormals, aliased
weights, unchanged views without growth, errors, scalar/vector layouts and both
growth policies. Final validation passes **1,581 Python tests (one skipped)**, every
rebuilt native suite and the CPU-only build/API check. [Validation](validation/bound-state-m3.txt).

At that checkpoint, candidate staging, generic metadata/model ownership,
sketch/CUDA algorithm gaps, AFT corrections, interoperability and distributed
collectives remained. The later AFT port removes CPU corrections. No benchmark
or NVIDIA execution comparison was run for the bound-state change.


### CUDA linear coordinate-descent alignment

Linear training now follows CUDA's FP32 gradient sums, includes negative Hessians,
and applies residual products in CUDA order. Bias updates skip zero steps; feature
updates execute them, including their signed-zero and nonfinite behavior. Greedy
and thrifty selectors now see the original gradient snapshot, as CUDA specifies.
Residuals stay in the GPU-owned copy, leaving the input gradients unchanged.

A new host CUDA-source updater checks complete training with the existing `2e-5`
tolerance. CPU comparisons remain for selectors whose algorithms agree. Explicit
fixtures catch selector divergence, negative Hessians, product order, FP32 overflow
and zero-step behavior. Native checks retain 49 cases and over 148 million residual
field comparisons; public coverage now contains 158 cases. Final validation passes
**1,603 Python tests (one skipped)**, every rebuilt native suite and the CPU-only
build/API check. [Validation](validation/cuda-linear-m3.txt).

The reduction uses FP32 throughout but does not reproduce a specific NVIDIA
CUB tile policy. CUB's floating-point reduction order can vary by GPU architecture;
see the [coverage ledger](docs/CUDA_PARITY.md#linear-coordinate-descent) for the
source contract and validation limits. Actual NVIDIA execution and compiler
contraction remain unverified. No performance benchmark was run.


### CUDA ranking-metric alignment

Precision now divides and weights each label in binary64 before reduction; empty
queries contribute zero. NDCG applies inverse ideal gain after reducing DCG, and
retains CUDA's unweighted nonpositive-inverse fallback. MAP now scans actual labels
in parallel, multiplies weight before normalization and uses CUDA's unweighted
zero-relevance fallback. These replace earlier CPU-specific behavior.

Validation includes **15 exact edge/order checks**, **896 native formula cases over
9.5 million rows**, and **101 public ranking cases**. Existing comparison tolerances
remain unchanged. Tests distinguish CPU/CUDA rules, exercise empty and strided
queries, and cover recursive prefixes and near-binary labels. Catastrophic signed
cancellation can select different branches under different parallel scan orders;
we document the current Metal behavior without claiming NVIDIA bitwise equivalence.
[Validation](validation/cuda-ranking-metrics-m3.txt) · [Scope](docs/CUDA_PARITY.md#ranking-metric-numerical-behavior).

Full validation passes **1,619 Python tests (one skipped)**, every rebuilt native
suite and the CPU-only build/API check. Grouped AUC, ranking gradients/RNG and the
other coverage-ledger gaps remain separate work. No performance benchmark or NVIDIA execution comparison was run.


### CUDA grouped PR-AUC alignment

Grouped PR-AUC now rounds every false-positive term once after binary64 subtraction
and multiplication, reduces recall totals separately in original row order, and
uses CUDA's integral arithmetic. Queries require at least two documents and a
positive final FP/TP product. These changes remove earlier CPU-specific rounding
and single-row behavior. Sorting, totals, prefixes, integration and aggregation
remain on the GPU in one command.

Tests pass **297 native cases over 14.2 million rows**, **16 exact edge/cast and
nonfinite checks**, and **53 public grouped-AUC cases**, including 21 new cases.
The independent integral and CUDA-source oracle agree within the unchanged
`2e-12` tolerance. Empty queries remain a safe extension of CUDA's unguarded
indexing; hardware reduction/logarithm equivalence is still unverified.
[Validation](validation/cuda-ranking-pr-m3.txt).

Final validation passes **1,640 Python tests (one skipped)**, every rebuilt native
suite and the CPU-only build/API check. Classification AUC arithmetic, grouped ROC
source evidence and other ledger gaps remain. No performance benchmark or NVIDIA execution comparison was run.


### CUDA classification AUC and GPU normalization

Binary, multiclass and multilabel ROC/PR now use CUDA's FP32 cast boundaries,
separate PR recall/validity totals and distinct invalid-output rules. GPU kernels
prepare local aggregation metadata and perform per-output normalization, weighted
sums and final averaging. Host collective transport and staging remain.

The PR integral now follows CUDA's difference of logarithms. This removes the
previous stabilized formula's accuracy advantage for extreme row weights:
the documented `1e30` negative-weight example now returns `1`, as the source
requires, instead of approximately `5e-31`. Extreme-prefix association can also
change NaN outcomes. Exact terms, independently bounded prefixes and source
arithmetic on captured endpoints test those cases without claiming identical
NVIDIA execution.

Focused validation passes **486 native cases per curve**, **25 normalization /
metadata cases**, four exact cast/output checks, and **97 public tests**. Ordinary
thresholds are unchanged; the extreme-case contract is explicitly documented.
Full validation passes **1,666 Python tests, one skip**, all rebuilt native suites
and the CPU-only build/device check. The Python regression took 96.29 seconds;
the complete build-and-test run finished successfully in 4 minutes 14 seconds.
[Validation](validation/cuda-classification-auc-m3.txt).

Grouped ROC source evidence, remaining CUDA algorithms, ownership, interoperability
and GPU collectives remain open. No performance benchmark was run.

### Grouped ROC CUDA-source alignment

Grouped ROC retains CUDA's stable label ordering, FP32 score differences and
squared query-weight terms. The existing GPU merge counter computes the same
concordant/tied-pair counts without enumerating every pair. This changes the
floating-point summation association, so exact NVIDIA rounding is not claimed.

Finite weights such as `1e30` overflow when squared. MPS now returns a NaN metric
with the shared upstream warning, matching CUDA source; CPU continues to throw.
Private native NaN-weight inputs now produce zero scores with a valid-query count,
as CUDA's positive-weight branch specifies. Public metadata validation is unchanged.

Validation adds **390 query cases / 2,168,663 independently enumerated pairs** and
20 public tests, retaining all 272 existing native cases, the three million-row
analytic cases, CPU checks where applicable and the original `2e-12` tolerance.
All **73 public grouped AUC tests** pass. Full regression passes **1,686 Python
tests, one skip**, every native suite and the CPU-only build/device check. The
Python suite took 96.70 seconds; the complete run finished in 4 minutes 11 seconds.
Empty/one-row query slots remain safely zeroed: CUDA's temporary result allocation
does not initialize slots with no pairs. That undefined behavior is not reproduced.
The million-row cases also exceed CUDA's current pair-launch limit; they verify
Metal's larger supported domain, not an executable CUDA comparison.
[Validation](validation/cuda-grouped-roc-m3.txt). No benchmark was run.

### CUDA column sampling and shuffle

Weighted sampling now generates CUDA's modern Philox uniform draws independently
on GPU and forms FP32 log/weight priorities. Unweighted sampling uses Thrust's
24-round Feistel permutation, a GPU prefix scan and stable compaction. The 24
shuffle keys are generated on host, matching upstream Thrust's placement. Final
feature ordering remains on GPU; host tree/level caches and seed consumption are
unchanged. The former MT19937/Fisher-Yates kernels have been removed.

This changes sampled models from the CPU-oriented backend. Native tests retain
all earlier edge families and verify **550 cases**, 3,281,095 selected feature IDs,
4,610,295 exact Philox uniforms and 1,320,896 exact permutation entries. A separate
host build of pinned, unmodified Thrust/CCCL validates the scalar reference.
Weighted log keys match Apple libm exactly in this corpus; NVIDIA libdevice and
cross-Apple behavior remain unverified.

The **307-test focused Python run** passes, including 48 added cases that check
selected root features over three rounds against source RNG/selection formulas.
Unsampled CPU training checks remain; sampled models use source tree checks and
unchanged same-model CPU prediction tolerances. Full regression passes **1,734
Python tests, one skip**, every rebuilt native suite and the CPU-only build/device
check. The Python suite took 104.65 seconds; the complete run took 4 minutes 21 seconds.
No benchmark was run, and the removed CPU-stream fused path's timings do not
describe this implementation. [Validation](validation/cuda-column-sampling-m3.txt).


### CUDA-derived raw sketching and resident summaries

Ordinary `hist` and `approx` now consume raw SparsePage windows on Metal. Stable
radix passes order feature/value pairs; row weights are gathered from CSR offsets,
FP32 weights are scanned by feature, and categorical duplicates are compacted.
Summaries stay on the GPU across windows/pages, with CUDA rank queries, endpoint
selection, reverse duplicate-weight accumulation, parallel merge partitions and
rank repair. Final cut materialization stays on host, as in CUDA. Weighted scan
association and equal-value sorting order can differ from CUB/Thrust; no bitwise
cross-device guarantee is implied. Ordered rank repair avoids CUDA's neighbor
read/write race. These choices are detailed in the coverage ledger.

New tests check source equations, raw window boundaries, caches, fixed references,
external pages and sketch-to-tree behavior. Existing CPU training-equivalence
fixtures now explicitly share QuantileDMatrix cuts where the two sketch algorithms
differ; their prediction tolerances and datasets are retained. This exposed a
separate Metal traversal bug at subnormal split thresholds. Numeric and category
comparisons now preserve tiny values, with dedicated prediction regressions.

This step does not complete adapter preprocessing, Apple GPU array interoperability
or distributed sketch collectives. Input staging, allocation and synchronization
remain performance work. Final validation passes **1,773 Python tests, one skip**,
every rebuilt native suite and the CPU-only build/device check (4 minutes 2 seconds;
Python tests 93.03 seconds). [Validation](validation/cuda-resident-sketch-m3.txt).
No benchmark was run; earlier timings predate this sketch
pipeline and should not be read as measurements of its performance.
