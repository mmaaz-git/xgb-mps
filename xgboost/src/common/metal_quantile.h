/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstdint>

#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
struct QuantileRequest {
  std::uint32_t target, padding;
  double alpha;
};
void Quantiles(linalg::MatrixView<float const> values, Span<float const> weights,
               Span<QuantileRequest const> requests, Span<float> output);
}  // namespace xgboost::common::metal
