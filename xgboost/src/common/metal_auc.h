/** Copyright 2026, XGBoost Contributors */
#pragma once
#include "xgboost/linalg.h"
#include "xgboost/span.h"
namespace xgboost::common::metal {
struct ROCAUCSums {
  double fp, tp, area;
};
// Stable descending scores and binary64 GPU scans/reductions. Results are
// unnormalized; the caller retains XGBoost's prevalence/macro aggregation rules.
// Predictions must not contain NaNs (sorting has no CPU reference ordering).
void ROCAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
            Span<float const> weights, bool multiclass, Span<ROCAUCSums> output);
// PR returns (1, 1, normalized area), matching the CPU aggregation convention.
void PRAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
           Span<float const> weights, bool multiclass, Span<ROCAUCSums> output);
}  // namespace xgboost::common::metal
