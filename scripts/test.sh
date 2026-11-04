#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
echo "Compiling standalone tests against the production Metal shader"
echo "Checking mean-ranking random draws against Apple libc++"
clang++ -O2 -std=c++17 -fobjc-arc tests/metal_ranking_rng.mm \
  -framework Metal -framework Foundation -o results/metal_ranking_rng_test
MTL_DEBUG_LAYER=1 results/metal_ranking_rng_test xgboost/src/common/metal_histogram.metal
clang++ -std=c++17 -fobjc-arc tests/metal_kernel.mm \
  -framework Metal -framework Foundation -o results/metal_kernel_test
echo "Checking host-memory sharing, guards and VM-region fallback rules"
clang++ -O2 -std=c++17 -fobjc-arc tests/metal_memory.mm \
  -framework Metal -framework Foundation -o results/metal_memory_test
MTL_DEBUG_LAYER=1 results/metal_memory_test
echo "Checking GPU integer histograms against the independent CPU reference"
results/metal_kernel_test xgboost/src/common/metal_histogram.metal
echo "Checking gradient bounds, finite validation and quantization scales"
clang++ -std=c++17 -fobjc-arc tests/metal_bounds.mm \
  -framework Metal -framework Foundation -o results/metal_bounds_test
results/metal_bounds_test xgboost/src/common/metal_histogram.metal
echo "Checking compressed bin assignment and hit counts"
clang++ -std=c++17 -fobjc-arc tests/metal_binning.mm \
  -framework Metal -framework Foundation -o results/metal_binning_test
results/metal_binning_test xgboost/src/common/metal_histogram.metal
echo "Checking objective kernels against independent CPU mathematics"
clang++ -std=c++17 -fobjc-arc tests/metal_objective.mm \
  -framework Metal -framework Foundation -o results/metal_objective_test
results/metal_objective_test xgboost/src/common/metal_histogram.metal
echo "Checking Metal partitioning and index transposition against CPU references"
clang++ -std=c++17 -fobjc-arc tests/metal_partition.mm \
  -framework Metal -framework Foundation -o results/metal_partition_test
results/metal_partition_test xgboost/src/common/metal_histogram.metal
echo "Checking exact FP32 division against CPU arithmetic"
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_fp32_div.mm \
  -framework Metal -framework Foundation -o results/metal_fp32_div_test
MTL_DEBUG_LAYER=1 results/metal_fp32_div_test xgboost/src/common/metal_histogram.metal
echo "Checking software binary64 against CPU arithmetic"
clang++ -std=c++17 -fobjc-arc tests/metal_fp64.mm \
  -framework Metal -framework Foundation -o results/metal_fp64_test
results/metal_fp64_test xgboost/src/common/metal_histogram.metal
echo "Checking binary64 exp/log/erf against CPU libm"
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_transcendental.mm \
  -framework Metal -framework Foundation -o results/metal_transcendental_test
results/metal_transcendental_test xgboost/src/common/metal_histogram.metal
echo "Checking conservative split bounds against one million CPU gains"
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_split_bounds.mm \
  -framework Metal -framework Foundation -o results/metal_split_bounds_test
MTL_DEBUG_LAYER=1 results/metal_split_bounds_test xgboost/src/common/metal_histogram.metal
echo "Checking scalar and vector split kernels against independent CPU references"
clang++ -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_split.mm \
  -framework Metal -framework Foundation -o results/metal_split_test
results/metal_split_test xgboost/src/common/metal_histogram.metal
echo "Checking GPU histogram conversion, accumulation, subtraction and gathers"
clang++ -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_storage.mm \
  -framework Metal -framework Foundation -o results/metal_storage_test
results/metal_storage_test xgboost/src/common/metal_histogram.metal
echo "Checking root-gradient sums and stump weights"
clang++ -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_reduction.mm \
  -framework Metal -framework Foundation -o results/metal_reduction_test
results/metal_reduction_test xgboost/src/common/metal_histogram.metal
echo "Checking GPU means, weighted means and FP32 mean terms"
clang++ -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_mean.mm \
  -framework Metal -framework Foundation -o results/metal_mean_test
results/metal_mean_test xgboost/src/common/metal_histogram.metal
echo "Checking integer prefixes, full stable sorts, weighted CDFs and quantiles"
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_quantile.mm \
  -framework Metal -framework Foundation -o results/metal_quantile_test
results/metal_quantile_test xgboost/src/common/metal_histogram.metal
echo "Checking production median and mean APIs, including strided and empty inputs"
libomp_root="${LIBOMP_ROOT:-$(brew --prefix libomp)}"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_stats_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_stats_api_test
results/metal_stats_api_test
echo "Checking objective initialization against exact CPU gradients and broadcasts"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_initialization_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_initialization_api_test
MTL_DEBUG_LAYER=1 results/metal_initialization_api_test
echo "Checking objective label predicates and strided production validation"
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_label_validation.mm \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp -framework Foundation -framework Metal \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_label_validation_test
MTL_DEBUG_LAYER=1 results/metal_label_validation_test xgboost/src/common/metal_histogram.metal
echo "Checking CPU-compatible base-score conversions and validation errors"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_base_score_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_base_score_api_test
MTL_DEBUG_LAYER=1 results/metal_base_score_api_test
echo "Checking production ROC/PR-AUC against CPU and independent references"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_auc_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_auc_api_test
MTL_DEBUG_LAYER=1 results/metal_auc_api_test
echo "Checking grouped ranking ROC against CPU pairs and million-row analytic cases"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_ranking_roc_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_ranking_roc_api_test
MTL_DEBUG_LAYER=1 results/metal_ranking_roc_api_test
echo "Checking grouped PR integrals, segmented prefixes and GPU label validation"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_ranking_pr_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_ranking_pr_api_test
MTL_DEBUG_LAYER=1 results/metal_ranking_pr_api_test
echo "Checking production grouped ranking sorts against stable CPU order"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_ranking_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_ranking_api_test
MTL_DEBUG_LAYER=1 results/metal_ranking_api_test
echo "Checking GPU ranking label validation, discounts and ideal gains"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_ranking_cache_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_ranking_cache_api_test
MTL_DEBUG_LAYER=1 results/metal_ranking_cache_api_test
echo "Checking ranking sigmoid rounding against CPU and binary64 oracles"
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_ranking_math.mm \
  -framework Metal -framework Foundation -o results/metal_ranking_math_test
MTL_DEBUG_LAYER=1 results/metal_ranking_math_test xgboost/src/common/metal_histogram.metal
echo "Checking top-k/mean ranking gradients and bounded pair batches"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_ranking_gradient_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_ranking_gradient_api_test
MTL_DEBUG_LAYER=1 results/metal_ranking_gradient_api_test
echo "Checking unbiased ranking gradients, repeated bias updates and storage batches"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_ranking_unbiased_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_ranking_unbiased_api_test
MTL_DEBUG_LAYER=1 results/metal_ranking_unbiased_api_test
echo "Checking objective memory sharing and alias-safe copied fallbacks"
clang++ -O2 -std=c++17 -fobjc-arc \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_memory_api.mm \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -framework Metal -framework Foundation -o results/metal_memory_api_test
MTL_DEBUG_LAYER=1 results/metal_memory_api_test
echo "Checking production AFT gradients, transforms, metrics and correction accounting"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_survival_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_survival_api_test
results/metal_survival_api_test
echo "Checking adaptive objective scales and gradients against CPU references"
clang++ -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_adaptive.mm \
  -framework Metal -framework Foundation -o results/metal_adaptive_test
results/metal_adaptive_test xgboost/src/common/metal_histogram.metal
echo "Checking GPU sampling against the CPU random engine and Bernoulli distribution"
clang++ -std=c++17 -fobjc-arc tests/metal_sampling.mm \
  -framework Metal -framework Foundation -o results/metal_sampling_test
results/metal_sampling_test xgboost/src/common/metal_histogram.metal
echo "Checking gradient sampling norms, thresholds, rescaling and replay against production CPU"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_gradient_sampling_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_gradient_sampling_api_test
MTL_DEBUG_LAYER=1 results/metal_gradient_sampling_api_test
echo "Checking column-sampling logarithm against CPU libm and tie boundaries"
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_column_log.mm \
  -framework Metal -framework Foundation -o results/metal_column_log_test
MTL_DEBUG_LAYER=1 results/metal_column_log_test xgboost/src/common/metal_histogram.metal
echo "Checking column selection, MT19937 draws and tree/level/node caches against CPU"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_column_sampling_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_column_sampling_api_test
MTL_DEBUG_LAYER=1 results/metal_column_sampling_api_test
echo "Checking sketch level merging and pruning against CPU summaries and independent rank queries"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_sketch_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_sketch_api_test
MTL_DEBUG_LAYER=1 results/metal_sketch_api_test
echo "Checking SHAP against independent one-split values, guards and aliases"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_shap_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_shap_api_test
MTL_DEBUG_LAYER=1 results/metal_shap_api_test
echo "Checking resident leaf sums and regularized weights against CPU formulas"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_leaf_fitting_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_leaf_fitting_api_test
MTL_DEBUG_LAYER=1 results/metal_leaf_fitting_api_test
echo "Checking GPU linear reductions and residual updates against CPU arithmetic"
clang++ -O2 -std=c++17 -ffp-contract=off \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/metal_linear_api.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/metal_linear_api_test
MTL_DEBUG_LAYER=1 results/metal_linear_api_test
echo "Running training equivalence tests; a real GPU is required"
.venv/bin/python -m pytest -q tests --require-metal
