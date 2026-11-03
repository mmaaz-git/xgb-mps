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
| Gradient quantization | `tree/gpu_hist/quantiser.cuh` | Implemented with a different bounded fixed-point scheme; GPU maximum/finite reduction, exact integer scale selection and bit conversion; scales remain in GPU buffers |
| Histogram subtraction, cache management | `tree/gpu_hist/histogram.cuh` | Metal-owned shared cache, GPU integer-to-binary64 conversion, external-page accumulation, subtraction and GPU feature gathering; host cache policy/metadata |
| Split evaluation, categorical partitions, constraints | `tree/gpu_hist/evaluate_splits.cu`, `evaluate_splits.cuh`, `multi_evaluate_splits.cu` | Metal scalar/vector candidate scoring, category ordering and missing directions; software binary64, exact bounded scans, monotonic bounds and pooled vector weights; host column sampling, interaction masks and tree updates |
| Row partition and final leaf assignment | `tree/gpu_hist/row_partitioner.cu` | Metal stable classify/prefix/scatter, resident row IDs, GPU leaf positions; supports numerical/categorical splits and missing values |
| Sampling and gradient sums | `tree/gpu_hist/sampler.cu`, `tree/updater_gpu_hist.cu` | Metal uniform row sampling and replay with CPU-compatible LCG/Bernoulli draws; gradient-based/column sampling and gradient sums remain CPU |
| Squared/logistic/squared-log/gamma/pseudo-Huber/Poisson/Tweedie/hinge/multiclass/expectile gradients | `objective/regression_obj.cu`, `hinge.cu`, `multiclass_obj.cu` | Metal kernels; label validation and base-score estimation still CPU |
| Sigmoid/exp/hinge/softmax/argmax/expectile prediction transforms | Same objective files | Metal kernels; identity transforms require no dispatch |
| Absolute-error and quantile objectives, associated statistics | `objective/regression_obj.cu`, `quantile_obj.cu` | Metal per-iteration residual scales, gradients and quantile prediction sorting; base-score estimation remains CPU |
| Ranking losses, pair construction, normalization | `objective/lambdarank_obj.cu` | CPU |
| AFT survival loss | `objective/aft_obj.cu`, `common/survival_util.h` | CPU |
| Base score estimation, leaf value fitting | `tree/fit_stump.cu`, `common/stats.cu` | CPU |
| Batch/inplace/leaf tree traversal | `predictor/gpu_predictor.cu` | Metal: scalar/vector leaves, category bitsets, missing values, tree ranges, dropout weights; CPU input views stage dense chunks and recode categories |
| Prediction cache from leaf IDs | `predictor/predictor.cu` | Metal; supports strided target slices and negated sampling positions |
| SHAP contributions and interactions | `predictor/gpu_predictor.cu`, `predictor/interpretability/shap.cu` | CPU |
| RMSE/RMSLE/MAE/MAPE/logloss/error/Poisson/gamma/Tweedie/pseudo-Huber metrics | `metric/elementwise_metric.cu` | Metal loss calculation and scaled block reductions; final small reduction in host FP64 |
| Multiclass error and logloss | `metric/multiclass_metric.cu` | Metal loss calculation and scaled block reductions; final small reduction in host FP64 |
| Quantile and expectile metrics | `metric/elementwise_metric.cu` | Metal loss calculation, subnormal-preserving products and binary64 block reductions; small final host sum |
| Ranking, AUC and survival metrics | `metric/*.cu` | CPU |
| Quantile sketching, cuts, weighted/category handling | `common/quantile.cu`, `data/gradient_index.cu` | CPU |
| Compressed GPU matrix, external memory, iterators | `data/ellpack_page.cu`, `iterative_dmatrix.cu`, `quantile_dmatrix.cu` | Host GHistIndexMatrix remains authoritative; page-owned Metal copies and dense GPU transpose cache persist for each source page lifetime |
| Categorical recoding, metadata manipulation, validation | `data/cat_container.cu`, `metainfo.cu`, `array_interface.cu` | CPU |
| Linear coordinate-descent booster | `linear/updater_gpu_coordinate.cu` | `gblinear` rejected for `mps` |
| Device arrays, scans, sorts, reductions, random generation | `common/*.cu`, `common/*.cuh` | Histogram/metric reductions, checked exact scans, category/quantile-row sorting and sampling-specific random generation implemented; general Metal primitives missing |
| Distributed GPU collectives | `collective/comm.cu`, `collective/coll.cu`, `collective/aggregator.cuh` | Distributed `mps` training rejected |
| CUDA array interface / CuPy / cuDF / CUDA streams | `c_api/c_api.cu`, `data/device_adapter.cu` | NVIDIA-specific interfaces cannot consume Metal memory; corresponding Apple-device interoperability is missing |
| CUDA contexts, allocators, stream lifetimes | `common/cuda_*`, `common/device_*`, `common/host_device_vector.cu` | Shared-memory Metal runtime, reused command queue and histogram buffers; persistent end-to-end device ownership and asynchronous scheduling missing |

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
also execute on Metal. Host orchestration still synchronizes at each tree level;
gradients and prediction state still cross between host arrays and GPU buffers.
Keeping those arrays resident and reducing synchronization are the next architectural
steps. Host model construction and serialization can remain on the CPU.

Split tests include categorical partitions, missing directions, monotonic and
interaction constraints, vector leaves and external-memory pages. Gradient-based
sampling, remaining training-state updates and more efficient external-memory
staging remain required.

Remaining ranking/survival objectives, SHAP, specialized metrics, quantile
construction, linear training and collectives are separate coverage requirements.
The observed prediction speedups do not establish any of those as complete.
