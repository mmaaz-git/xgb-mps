/** Copyright 2026, XGBoost Contributors */
#pragma once
#include "xgboost/data.h"
#include "xgboost/string_view.h"

namespace xgboost::common::metal {
// CUDA-source metadata conversion/validation and query-ID grouping. Accepts
// scalar/vector/matrix Metal tensor descriptors with synchronous producer rules
// identical to ImportTensor. Float metadata retains owned Metal vector storage.
void SetMetadata(MetaInfo* info, StringView field, StringView descriptor);
void SliceMetadata(MetaInfo const& info, Span<bst_idx_t const> indices, MetaInfo* output);
}  // namespace xgboost::common::metal
