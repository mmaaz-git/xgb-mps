/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstdint>

#include "optional_weight.h"
#include "xgboost/linalg.h"

namespace xgboost::common::metal {
enum class VectorOp : std::uint32_t { kScale = 0, kLog = 1, kSubtract = 2 };
// Scale uses binary64 multiplication followed by FP32 narrowing; log first adds
// its FP32 epsilon; subtraction is FP32. Untouched strided elements are retained.
void VectorArithmetic(VectorOp op, linalg::VectorView<float> values, double parameter);
// CUDA packs runs of sorted, distinct labels into consecutive bins. It does not
// scatter back to class IDs when classes are absent. Inputs must be finite and
// distinct-run count <= output size; overflow is rejected before host output changes.
void SmallHistogram(linalg::MatrixView<float const> indices, OptionalWeights const& weights,
                    linalg::VectorView<float> bins);
}  // namespace xgboost::common::metal
