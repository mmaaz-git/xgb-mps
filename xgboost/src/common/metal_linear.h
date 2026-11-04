/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Owns a snapshot of sorted CSC data and mutable row-major gradients for one
// coordinate-descent update. Feature -1 denotes the bias coordinate. Matches
// updater_gpu_coordinate.cu: FP32 sums include every Hessian, residual products
// use (h * delta) * x, and only the bias coordinate skips zero delta. CUB's
// architecture-dependent reduction tree is not a cross-device bitwise contract.
class LinearState {
 public:
  virtual ~LinearState() = default;
  virtual GradientPair Gradient(std::int32_t feature, std::uint32_t group) = 0;
  virtual void UpdateResidual(std::int32_t feature, std::uint32_t group, float delta) = 0;
  virtual void CopyGradients(Span<GradientPair> output) const = 0;
};
std::unique_ptr<LinearState> MakeLinearState(std::size_t rows, std::uint32_t groups,
                                             Span<bst_idx_t const> column_offsets,
                                             Span<Entry const> entries,
                                             Span<GradientPair const> gradients);
}  // namespace xgboost::common::metal
