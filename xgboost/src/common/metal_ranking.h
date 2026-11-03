/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstddef>

#include "xgboost/base.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Stable descending sort within each query. Output indices are query-relative.
// Groups begin at zero and end at values.Size(); empty groups are permitted.
// NaN scores have no defined CPU ordering and are outside this contract.
void SortGroups(linalg::VectorView<float const> values, Span<bst_group_t const> groups,
                Span<std::size_t> relative_indices);
}  // namespace xgboost::common::metal
