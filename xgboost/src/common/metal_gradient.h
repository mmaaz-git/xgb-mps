/** Copyright 2026, XGBoost Contributors */
#pragma once

#include "xgboost/base.h"
#include "xgboost/linalg.h"

namespace xgboost::common::metal {
// Synchronous bit-preserving layout copies and Hessian extraction. Inputs can
// broadcast; output elements must be disjoint. Aliases use snapshot semantics.
// Guarded shared host views are used when eligible, with copied staging otherwise.
void CopyGradients(linalg::MatrixView<GradientPair const> input,
                   linalg::MatrixView<GradientPair> output);
void ExtractHessians(linalg::MatrixView<GradientPair const> input,
                     linalg::MatrixView<float> output);
}  // namespace xgboost::common::metal
