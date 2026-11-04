/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>
#include <memory>

#include "metal_split.h"
#include "xgboost/base.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
class GradientQuantizer;
// GPU-owned leaf sums accumulated across resident row partitions/pages. With an
// original value quantizer, arithmetic remains integer until materialization;
// Data() exposes a binary64 view, never the input to further integer addition.
// Calls without a quantizer retain the legacy binary64 low-level API.
class LeafSums {
 public:
  virtual ~LeafSums() = default;
  virtual Span<GradientPairPrecise const> Data() const = 0;
  virtual void Weights(SplitParameters const& parameters, Span<SplitBounds const> bounds,
                       float learning_rate, Span<float> output) const = 0;
};
std::unique_ptr<LeafSums> MakeLeafSums(
    std::size_t leaves, std::size_t targets,
    std::shared_ptr<GradientQuantizer const> original_quantizer = nullptr);
}  // namespace xgboost::common::metal
