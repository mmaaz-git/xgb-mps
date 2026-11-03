/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <array>
#include <cstdint>

#include "xgboost/base.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Distribution values match ProbabilityDistributionType (normal/logistic/extreme).
void SurvivalGradients(std::uint32_t distribution, float scale, Span<float const> predictions,
                       Span<float const> lower, Span<float const> upper, Span<float const> weights,
                       Span<GradientPair> output);
std::array<double, 2> SurvivalMetricSums(std::uint32_t distribution, float scale, bool accuracy,
                                         Span<float const> predictions, Span<float const> lower,
                                         Span<float const> upper, Span<float const> weights);
void SurvivalTransform(bool to_margin, Span<float> values);
}  // namespace xgboost::common::metal
