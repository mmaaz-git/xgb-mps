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
  std::uint64_t histogram_conversion_dispatches, histogram_subtraction_dispatches,
      resident_split_dispatches;
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
