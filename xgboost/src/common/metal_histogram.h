/**
 * Copyright 2026, XGBoost Contributors
 * Experimental Metal histogram backend. Other training stages use host memory.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "hist_util.h"
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
};

class Histogram {
 public:
  virtual ~Histogram() = default;
  // Invalidate uploaded data and gradients at each tree boundary.
  virtual void Reset() = 0;
  virtual void Build(std::size_t page_idx, GHistIndexMatrix const& gidx,
                     RowSetCollection const& rows, std::vector<bst_node_t> const& nodes,
                     Span<GradientPair const> gradients, std::vector<GHistRow> const& output) = 0;
};

std::unique_ptr<Histogram> MakeHistogram();
bool IsAvailable();
Statistics GetStatistics();

}  // namespace common::metal
}  // namespace xgboost
