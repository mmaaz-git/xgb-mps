/** Copyright 2026, XGBoost Contributors */
#pragma once

#include "xgboost/data.h"
#include "xgboost/span.h"

namespace xgboost {
class GHistIndexMatrix;
namespace common::metal {
// Quantile cuts and page row pointers are prepared by the caller.
void BinEntries(Span<Entry const> entries, Span<FeatureType const> feature_types,
                GHistIndexMatrix* output);
}  // namespace common::metal
}  // namespace xgboost
