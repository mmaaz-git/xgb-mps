/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>
#include <vector>

#include "quantile.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Merge levels in their original order, pruning after each merge and once more
// to the final budget. Sources remain unchanged; ranks retain FP32 CPU rounding.
struct SketchFinalization {
  Span<WQSummary<> const> levels;
  std::size_t intermediate_budget, final_budget;
};
std::vector<WQSummaryContainer> FinalizeSketches(Span<SketchFinalization const> columns);
}  // namespace xgboost::common::metal
