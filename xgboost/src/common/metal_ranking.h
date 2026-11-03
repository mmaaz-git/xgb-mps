/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstddef>
#include <cstdint>

#include "xgboost/base.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Stable descending sort within each query. Output indices are query-relative.
// Groups begin at zero and end at values.Size(); empty groups are permitted.
// NaN scores have no defined CPU ordering and are outside this contract.
void SortGroups(linalg::VectorView<float const> values, Span<bst_group_t const> groups,
                Span<std::size_t> relative_indices);

enum class RankingMetricKind : std::uint32_t { kPrecision, kNDCG, kMAP };
struct RankingSums {
  double score, weight;
};
// GPU sort and weighted query scoring in one command. NDCG uses the existing
// cache's discounts and inverse ideal gains. Caller validates label semantics.
RankingSums RankingScore(RankingMetricKind kind, linalg::VectorView<float const> predictions,
                         linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                         Span<float const> weights, std::uint32_t topk, bool minus = false,
                         bool exponential = true, Span<double const> discounts = {},
                         Span<double const> inverse_idcg = {});
}  // namespace xgboost::common::metal
