/** Copyright 2026, XGBoost Contributors */
#pragma once

#include "../encoder/ordinal.h"

namespace xgboost::common::metal {
// Typed dictionaries retain integer widths and signed-byte string comparisons.
void SortCategories(enc::HostColumnsView columns, Span<std::int32_t> sorted);
void RecodeCategories(enc::HostColumnsView original, Span<std::int32_t const> sorted,
                      enc::HostColumnsView incoming, Span<std::int32_t> mapping);
}  // namespace xgboost::common::metal
