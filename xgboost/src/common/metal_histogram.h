/**
 * Copyright 2026, XGBoost Contributors
 * Experimental Metal compute backend with host orchestration.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "hist_util.h"
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
  std::uint64_t ranking_gradient_dispatches, ranking_gradient_gpu_time_ns,
      ranking_pairs, ranking_pair_batches;
  std::uint64_t ranking_bias_dispatches;
  std::uint64_t gradient_sampling_dispatches, gradient_sampling_gpu_time_ns;
  std::uint64_t ranking_roc_auc_dispatches, ranking_roc_auc_gpu_time_ns;
  std::uint64_t ranking_pr_auc_dispatches, ranking_pr_auc_gpu_time_ns;
  std::uint64_t column_init_dispatches, column_sampling_dispatches, column_sampling_gpu_time_ns;
  std::uint64_t sketch_finalize_dispatches, sketch_finalize_columns, sketch_finalize_gpu_time_ns;
  std::uint64_t expectile_init_dispatches, prediction_init_dispatches, initialization_gpu_time_ns;
};

class Histogram {
 public:
  virtual ~Histogram() = default;
  // Invalidate uploaded data and gradients at each tree boundary.
  virtual void Reset() = 0;
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
