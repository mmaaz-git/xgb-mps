/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstdint>

#include "xgboost/base.h"
#include "xgboost/linalg.h"
namespace xgboost::common::metal {
void UniformSample(linalg::MatrixView<GradientPair> gradients, float probability,
                   std::uint64_t seed, std::uint32_t threads);
}  // namespace xgboost::common::metal
