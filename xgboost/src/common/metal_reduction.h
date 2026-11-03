/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>
#include <cstdint>

#include "xgboost/base.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
class HistogramBuffer;
// kFloat32 preserves FP32 per-term division; reductions use binary64 and can
// differ from CPU FP32 accumulation order. Sample means use binary64 terms.
enum class MeanKind : std::uint32_t { kFloat32, kSample, kWeightedSample };
void Mean(MeanKind kind, linalg::MatrixView<float const> values, Span<float const> weights,
          linalg::VectorView<float> output);
void SumGradients(linalg::MatrixView<GradientPair const> gradients,
                  linalg::VectorView<GradientPairPrecise> output);
void FitStump(linalg::MatrixView<GradientPair const> gradients, linalg::VectorView<float> output);
GradientPairPrecise SumHistogram(HistogramBuffer const* storage, std::size_t offset,
                                 std::size_t bins);
}  // namespace xgboost::common::metal
