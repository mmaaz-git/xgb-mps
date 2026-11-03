/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>

#include "xgboost/base.h"
#include "xgboost/linalg.h"

namespace xgboost::common::metal {
class HistogramBuffer;
void SumGradients(linalg::MatrixView<GradientPair const> gradients,
                  linalg::VectorView<GradientPairPrecise> output);
void FitStump(linalg::MatrixView<GradientPair const> gradients, linalg::VectorView<float> output);
GradientPairPrecise SumHistogram(HistogramBuffer const* storage, std::size_t offset,
                                 std::size_t bins);
}  // namespace xgboost::common::metal
