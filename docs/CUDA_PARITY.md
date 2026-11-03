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
| Split evaluation, categorical partitions, constraints | `tree/gpu_hist/evaluate_splits.cu`, `evaluate_splits.cuh`, `multi_evaluate_splits.cu` | Metal scalar/vector candidate scoring, category ordering and missing directions; software binary64, exact bounded scans, conservative numerical-candidate screening, monotonic bounds and pooled vector weights; host column sampling, interaction masks and tree updates |
| Row partition and final leaf assignment | `tree/gpu_hist/row_partitioner.cu` | Metal stable classify/prefix/scatter with cooperative scans for large nodes, resident row IDs, GPU leaf positions with guarded shared gradient inputs/output slices; supports numerical/categorical splits and missing values |
| Sampling and gradient sums | `tree/gpu_hist/sampler.cu`, `tree/updater_gpu_hist.cu` | Metal uniform row sampling/replay with CPU-compatible LCG/Bernoulli draws, scalar/vector root-gradient sums and dense root histogram sums; gradient-based/column sampling remain CPU |
| Squared/logistic/squared-log/gamma/pseudo-Huber/Poisson/Tweedie/hinge/multiclass/expectile gradients | `objective/regression_obj.cu`, `hinge.cu`, `multiclass_obj.cu` | Metal kernels with guarded synchronous host-memory sharing where eligible; means/weighted means and Newton stump fitting use Metal; label validation and other objective-specific initialization remain CPU |
| Sigmoid/exp/hinge/softmax/argmax/expectile prediction transforms | Same objective files | Metal kernels; identity transforms require no dispatch |
| Absolute-error and quantile objectives, associated statistics | `objective/regression_obj.cu`, `quantile_obj.cu` | Metal per-iteration residual scales, gradients and quantile prediction sorting; absolute-error initial weight sums, means/stump fitting and initial quantile estimation use Metal; other initialization work remains CPU |
| Ranking losses, pair construction, normalization | `objective/lambdarank_obj.cu` | Prediction order and NDCG ideal-label order use stable grouped Metal sorting; query-weight sums use Metal. Pair construction, gradients, discounts, ideal-gain calculation and other normalization remain CPU |
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
| Ranking metrics | `metric/rank_metric.cu`, `metric/auc.cu` | NDCG/MAP/precision sorting, weighted query scoring and query-sum reduction execute in one Metal command. NDCG discounts and inverse ideal gains are still calculated on CPU after GPU ideal-label sorting. Label validation remains CPU. Grouped ROC/PR-AUC remains CPU |
| Quantile sketching, cuts, weighted/category handling | `common/quantile.cu`, `data/gradient_index.cu` | Quantile sketching/cut construction remain CPU; numeric/categorical bin assignment and hit counts for SparsePage-backed matrices use Metal; adapter-built QuantileDMatrix bins remain CPU |
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
Split selection, histogram conversion/accumulation/subtraction and uniform sampling
and root-gradient/stump reductions also execute on Metal. Host orchestration still synchronizes at each tree level;
ordinary gradient calls can share host arrays directly, while other gradient and prediction-state paths still stage data between host arrays and GPU buffers.
Keeping those arrays resident and reducing synchronization are the next architectural
steps. Host model construction and serialization can remain on the CPU.

Split tests include categorical partitions, missing directions, monotonic and
interaction constraints, vector leaves and external-memory pages. Gradient-based
sampling, remaining training-state updates and more efficient external-memory
staging remain required.

Remaining ranking objectives, sensitive AFT gradient corrections, SHAP, specialized
metrics, quantile construction, linear training and collectives are separate coverage requirements.
The observed prediction speedups do not establish any of those as complete.

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

The fused metric command reads back only score and weight totals. NDCG cache creation,
ranking pair construction/gradients and grouped AUC remain separate unfinished work;
GPU metric dispatch does not imply those operations have been ported.
