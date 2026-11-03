/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstdint>
#include <vector>

#include "metal_storage.h"
#include "xgboost/base.h"
#include "xgboost/span.h"
namespace xgboost::common::metal {
class SplitWorkspace {
 public:
  virtual ~SplitWorkspace() = default;
};
std::unique_ptr<SplitWorkspace> MakeSplitWorkspace();

struct SplitTask {
  std::uint32_t begin, end, kind, limit;
  double parent_grad, parent_hess;
  float root_gain, lower, upper;
  std::int32_t monotone;
};
struct SplitParameters {
  float reg_lambda, reg_alpha, max_delta_step, min_child_weight;
};
struct SplitResult {
  double left_grad, left_hess, right_grad, right_hess;
  float gain;
  std::int32_t bin;
  std::uint32_t default_left, partition;
};
void EvaluateSplits(Span<HistogramInput const> histograms, Span<SplitTask const> tasks,
                    SplitParameters const& parameters, std::vector<SplitResult>* results,
                    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace);

// Histograms are packed bin-major, target-minor. Parent sums and bounds are
// packed by task, followed by target. Only the winning child sums are returned.
struct MultiSplitTask {
  std::uint32_t begin, end, kind, limit, parent_begin, targets;
  double root_gain;
  std::int32_t monotone;
  std::uint32_t constrained;
};
struct SplitBounds {
  float lower, upper;
};
void EvaluateMultiSplits(Span<HistogramInput const> histograms, Span<MultiSplitTask const> tasks,
                         Span<GradientPairPrecise const> parents, Span<SplitBounds const> bounds,
                         SplitParameters const& parameters, std::vector<SplitResult>* results,
                         std::vector<GradientPairPrecise>* children,
                         std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace);
}  // namespace xgboost::common::metal
