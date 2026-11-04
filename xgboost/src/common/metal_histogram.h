/**
 * Copyright 2026, XGBoost Contributors
 * Experimental Metal compute backend with host orchestration.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "hist_util.h"
#include "metal_gradient.h"
#include "metal_storage.h"
#include "row_set.h"
#include "xgboost/base.h"
#include "xgboost/span.h"

namespace xgboost {
class GHistIndexMatrix;
namespace common::metal {

struct Statistics {
  std::uint64_t histogram_dispatches;
  std::uint64_t rows_processed;
  std::uint64_t shared_histogram_dispatches;
  std::uint64_t gpu_time_ns;
  std::uint64_t quantization_dispatches, staging_time_ns, wait_time_ns, readback_time_ns;
  std::uint64_t objective_dispatches, transform_dispatches, objective_gpu_time_ns;
  std::uint64_t prediction_dispatches, prediction_gpu_time_ns, leaf_update_dispatches;
  std::uint64_t metric_dispatches, metric_gpu_time_ns;
  std::uint64_t metric_reduction_passes, metric_final_readback_bytes, metric_label_dispatches;
  std::uint64_t index_uploads, index_uploaded_bytes, resident_histogram_dispatches,
      partition_dispatches, leaf_partition_dispatches, row_init_dispatches, partition_gpu_time_ns,
      index_gpu_time_ns;
  std::uint64_t split_dispatches, split_gpu_time_ns;
  std::uint64_t residual_scale_dispatches;
  std::uint64_t sampling_dispatches, sampling_gpu_time_ns;
  std::uint64_t histogram_conversion_dispatches, histogram_subtraction_dispatches,
      resident_split_dispatches;
  std::uint64_t gradient_bound_dispatches;
  std::uint64_t gradient_sum_dispatches, gradient_sum_gpu_time_ns, stump_dispatches;
  std::uint64_t binning_dispatches, binning_gpu_time_ns;
  std::uint64_t mean_dispatches, mean_gpu_time_ns;
  std::uint64_t radix_sort_dispatches, quantile_dispatches, quantile_gpu_time_ns;
  std::uint64_t survival_gradient_dispatches, survival_metric_dispatches;
  std::uint64_t survival_cpu_corrections, survival_cpu_time_ns;
  std::uint64_t row_workspace_allocations, row_workspace_reuses;
  std::uint64_t float_sum_dispatches, float_sum_gpu_time_ns;
  std::uint64_t objective_shared_input_bytes, objective_shared_output_bytes, objective_staging_bytes;
  std::uint64_t histogram_shared_gradient_bytes, leaf_shared_gradient_bytes, leaf_shared_output_bytes;
  std::uint64_t roc_auc_dispatches, roc_auc_gpu_time_ns;
  std::uint64_t pr_auc_dispatches, pr_auc_gpu_time_ns;
  std::uint64_t group_sort_dispatches, group_sort_gpu_time_ns;
  std::uint64_t ranking_metric_dispatches, ranking_metric_gpu_time_ns;
  std::uint64_t ranking_label_dispatches, ranking_label_gpu_time_ns,
      ranking_cache_dispatches, ranking_cache_gpu_time_ns;
  std::uint64_t ranking_plan_dispatches, ranking_plan_reuses, ranking_plan_gpu_time_ns;
  std::uint64_t ranking_gradient_dispatches, ranking_gradient_gpu_time_ns,
      ranking_pairs, ranking_pair_batches;
  std::uint64_t ranking_bias_dispatches;
  std::uint64_t gradient_sampling_dispatches, gradient_sampling_gpu_time_ns;
  std::uint64_t ranking_roc_auc_dispatches, ranking_roc_auc_gpu_time_ns;
  std::uint64_t ranking_pr_auc_dispatches, ranking_pr_auc_gpu_time_ns;
  std::uint64_t column_init_dispatches, column_sampling_dispatches, column_sampling_gpu_time_ns;
  std::uint64_t sketch_finalize_dispatches, sketch_finalize_columns, sketch_finalize_gpu_time_ns;
  std::uint64_t expectile_init_dispatches, prediction_init_dispatches, initialization_gpu_time_ns;
  std::uint64_t label_validation_dispatches, label_validation_gpu_time_ns;
  std::uint64_t base_score_dispatches, base_score_gpu_time_ns;
  std::uint64_t shap_dispatches, shap_interaction_dispatches, shap_gpu_time_ns;
  std::uint64_t leaf_sum_init_dispatches, leaf_sum_dispatches, leaf_weight_dispatches,
      leaf_fitting_gpu_time_ns;
  std::uint64_t linear_gradient_dispatches, linear_residual_dispatches, linear_gpu_time_ns;
  std::uint64_t sketch_weight_dispatches, sketch_sorted_dispatches, sketch_update_dispatches,
      sketch_ingest_gpu_time_ns;
  std::uint64_t gradient_copy_dispatches, hessian_extract_dispatches, gradient_state_gpu_time_ns,
      gradient_state_shared_input_bytes, gradient_state_shared_output_bytes,
      gradient_state_staging_bytes;
  std::uint64_t histogram_integer_store_dispatches, histogram_integer_accumulation_dispatches,
      histogram_integer_subtraction_dispatches, histogram_integer_sum_dispatches;
  std::uint64_t quantized_gradient_sum_dispatches, cuda_uniform_sampling_dispatches;
  std::uint64_t cuda_gradient_sampling_dispatches, leaf_integer_sum_dispatches;
  std::uint64_t cuda_integer_split_dispatches;
  std::uint64_t cuda_integer_multi_split_dispatches;
  std::uint64_t split_finalize_dispatches, split_bounds_dispatches;
  std::uint64_t model_weight_copy_dispatches, model_category_copy_dispatches, model_copy_words;
  std::uint64_t model_scale_dispatches, model_state_gpu_time_ns;
  std::uint64_t split_category_dispatches, split_category_nodes;
  std::uint64_t interaction_reset_dispatches, interaction_split_dispatches,
      interaction_query_dispatches, interaction_buffer_growths, interaction_gpu_time_ns;
  std::uint64_t split_bounds_init_dispatches;
  std::uint64_t cuda_sketch_dispatches, cuda_sketch_batches, cuda_sketch_input_entries,
      cuda_sketch_gpu_time_ns;
  std::uint64_t small_histogram_dispatches, vector_arithmetic_dispatches, float_mean_dispatches;
  std::uint64_t category_sort_dispatches, category_recode_dispatches, category_gpu_time_ns;
  std::uint64_t input_prepare_dispatches, input_recode_dispatches, input_prepare_gpu_time_ns,
      input_debug_readback_bytes;
  std::uint64_t adapter_prepare_dispatches, adapter_index_dispatches, adapter_gpu_time_ns,
      adapter_debug_readback_bytes;
  std::uint64_t ellpack_pack_dispatches, ellpack_copy_dispatches, ellpack_restore_dispatches,
      ellpack_cache_dispatches, ellpack_gpu_time_ns, ellpack_debug_readback_bytes,
      ellpack_cached_host_bytes, ellpack_cached_device_bytes;
  std::uint64_t ellpack_adapter_dispatches, ellpack_adapter_values, ellpack_pack_input_bytes;
  std::uint64_t ellpack_histogram_dispatches, ellpack_partition_dispatches,
      ellpack_input_dispatches;
};

class Histogram {
 public:
  virtual ~Histogram() = default;
  // Invalidate uploaded data and gradients at each tree boundary.
  virtual void Reset() = 0;
  virtual void BindQuantized(std::shared_ptr<QuantizedGradients const> gradients,
                             bst_target_t target) = 0;
  virtual void Build(std::size_t page_idx, GHistIndexMatrix const& gidx,
                     RowSetCollection const& rows, std::vector<bst_node_t> const& nodes,
                     Span<GradientPair const> gradients, std::vector<GHistRow> const& output,
                     HistogramBuffer* storage) = 0;
};

std::unique_ptr<Histogram> MakeHistogram();
bool IsAvailable();
Statistics GetStatistics();

}  // namespace common::metal
}  // namespace xgboost
