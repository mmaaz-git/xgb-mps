/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <array>
#include <cstdint>

#include "metal_vector.h"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
inline auto PrepareGradientOutput(linalg::Matrix<GradientPair>* output, bst_idx_t rows,
                                  bst_target_t targets) {
  output->SetDevice(DeviceOrd::MPS());
  output->Reshape(rows, targets);
  return output->Data()->DeviceSpan();
}
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
enum class LabelDomain : std::uint32_t {
  kAny = 0,
  kUnitInterval = 1,
  kGreaterThanMinusOne = 2,
  kPositive = 3,
  kNonnegative = 4,
  kClassIndex = 5
};
// Checks the selected tensor elements, including strided label views. Empty
// inputs and the unconditional kAny predicate require no device operation.
bool ValidateLabels(LabelDomain domain, linalg::MatrixView<float const> labels,
                    std::int64_t n_classes = 0);
enum class BaseScoreTransform : std::uint32_t {
  kIdentity = 0,
  kLogistic = 1,
  kLogarithm = 2,
  kPositiveLogarithm = 3
};
// Invalid checked intercepts return false without modifying any input value.
bool BaseScoreToMargin(BaseScoreTransform transform, Span<float> values);
void AdaptiveGradients(Context const* ctx, AdaptiveObjective objective,
                       Span<float const> predictions, Span<float const> labels,
                       Span<float const> weights, std::uint32_t targets, Span<float const> alpha,
                       Span<GradientPair> output);
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
