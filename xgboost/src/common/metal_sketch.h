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

// CPU-compatible weighting and sorted-column ingestion. Group boundaries, when
// present, map row IDs to group weights; absent Hessians/weights mean one.
std::vector<float> MakeSketchWeights(std::size_t rows, Span<float const> hessians,
                                     Span<float const> weights, Span<bst_group_t const> groups);
struct SortedSketchColumn {
  Span<::xgboost::Entry const> entries;
  std::size_t budget;
};
struct SortedSketchResult {
  WQSummaryContainer summary;
  std::size_t num_elements;
};
std::vector<SortedSketchResult> MakeSortedSketches(Span<SortedSketchColumn const> columns,
                                                   Span<float const> weights);

// Prune the incoming summary, merge into successive levels and carry overflow
// until it fits. Unconsumed levels and all source buffers remain unchanged.
struct SketchUpdate {
  Span<WQSummary<> const> levels;
  Span<WQSummary<>::Entry const> incoming;
  std::size_t limit;
};
struct SketchUpdateResult {
  std::vector<WQSummaryContainer> levels;
  std::size_t carry_level;
};
std::vector<SketchUpdateResult> UpdateSketches(Span<SketchUpdate const> columns);
}  // namespace xgboost::common::metal
