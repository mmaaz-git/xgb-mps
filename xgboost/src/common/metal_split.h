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

// CUDA scalar split semantics: integer prefixes/missing/children, exclusive
// numeric scans, strict missing-direction comparison (ties go right).
struct QuantizedSplitTask {
  std::uint32_t begin, end, kind, limit;
  GradientPairInt64 parent;
  float lower, upper;
  std::int32_t monotone;
  std::uint32_t padding{0};
};
struct QuantizedSplitResult {
  GradientPairInt64 left, right;
  // Raw child gain: choose the winning feature BEFORE subtracting parent gain.
  float gain;
  // Numeric: exclusive prefix position, including position zero (-infinity).
  // Categorical: category index or sorted partition boundary.
  std::int32_t bin;
  std::uint32_t default_left, partition;
  GradientPairPrecise left_sum, right_sum;
};

struct SplitTask {
  std::uint32_t begin, end, kind, limit;
  double parent_grad, parent_hess;
  float root_gain, lower, upper;
  std::int32_t monotone;
};
struct SplitParameters {
  float reg_lambda, reg_alpha, max_delta_step, min_child_weight;
};
void EvaluateQuantizedSplits(Span<HistogramInput const> histograms,
                             Span<QuantizedSplitTask const> tasks,
                             SplitParameters const& parameters,
                             std::vector<QuantizedSplitResult>* results,
                             std::vector<std::uint32_t>* sorted_categories,
                             SplitWorkspace* workspace);
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
