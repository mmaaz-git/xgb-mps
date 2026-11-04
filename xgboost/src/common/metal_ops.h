/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <array>
#include <cstdint>

#include "xgboost/base.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Values form the host/shader ABI. Keep in sync with xgb_objective.
enum class Objective : std::uint32_t {
  kSquare = 0,
  kLogistic = 1,
  kSquaredLog = 2,
  kGamma = 3,
  kPseudoHuber = 4,
  kPoisson = 5,
  kTweedie = 6,
  kHinge = 7,
  kSoftmax = 8,
  kExpectile = 9
};
enum class Transform : std::uint32_t {
  kIdentity = 0,
  kSigmoid = 1,
  kExp = 2,
  kHinge = 3,
  kSoftmax = 4,
  kArgmax = 5,
  kExpectile = 6,
  kQuantile = 7
};
enum class AdaptiveObjective : std::uint32_t { kAbsolute = 0, kQuantile = 1 };
void AdaptiveGradients(AdaptiveObjective objective, Span<float const> predictions,
                       Span<float const> labels, Span<float const> weights, std::uint32_t targets,
                       Span<float const> alpha, Span<GradientPair> output);
std::array<double, 2> QuantileMetricSums(bool expectile, Span<float const> predictions,
                                         Span<float const> labels, Span<float const> weights,
                                         Span<float const> alpha, std::uint32_t targets);
std::array<double, 2> MetricSums(char const* name, Span<float const> predictions,
                                 Span<float const> labels, Span<float const> weights,
                                 std::uint32_t targets, float parameter = 0.5f);
Objective ObjectiveFromName(char const* name);
Transform TransformFromName(char const* name);
void Gradients(Objective objective, Span<float const> predictions, Span<float const> labels,
               Span<float const> weights, std::uint32_t targets, float parameter,
               float scale_pos_weight, Span<float const> alpha, Span<GradientPair> output);
void TransformPredictions(Transform transform, Span<float> values, std::uint32_t targets = 1);
// Per-row objective initialization, separate from transformed-margin gradients.
void InitExpectileGradients(float mean, Span<float const> labels, Span<float const> weights,
                           Span<float const> alpha, Span<GradientPair> output);
// Repeat a target vector across contiguous rows; preserves the input float bits.
void BroadcastTargets(Span<float const> targets, Span<float> output);
}  // namespace xgboost::common::metal
