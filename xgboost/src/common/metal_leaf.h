/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>
#include <memory>

#include "metal_split.h"
#include "xgboost/base.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// GPU-owned binary64 sums, accumulated by resident row partitions across pages.
class LeafSums {
 public:
  virtual ~LeafSums() = default;
  virtual Span<GradientPairPrecise const> Data() const = 0;
  virtual void Weights(SplitParameters const& parameters, Span<SplitBounds const> bounds,
                       float learning_rate, Span<float> output) const = 0;
};
std::unique_ptr<LeafSums> MakeLeafSums(std::size_t leaves, std::size_t targets);
}  // namespace xgboost::common::metal
