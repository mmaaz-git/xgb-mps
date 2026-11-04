/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <array>
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

enum class RankingLabelKind : std::uint32_t { kBinary, kNDCG };
// Returns validation flags: bit 0 = invalid binary/integer label, bit 1 =
// relevance exceeds 31. Empty inputs pass without a dispatch.
std::uint32_t CheckRankingLabels(linalg::VectorView<float const> labels, RankingLabelKind kind);
// Sort labels and calculate discounts/inverse ideal gains on GPU. The caller
// validates labels first; linear gains may include finite negative values.
void InitNDCG(linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
              std::uint32_t topk, bool exponential, Span<double> discounts,
              Span<double> inverse_idcg);

enum class RankingMetricKind : std::uint32_t { kPrecision, kNDCG, kMAP };
struct RankingSums {
  double score, weight;
};
enum class RankingLossKind : std::uint32_t { kPairwise, kNDCG, kMAP };
struct RankingBias {
  Span<double> positive, negative;
  double regularizer{1};
};
// Optional internal diagnostics for source-oracle tests. Normal training does
// not copy these GPU statistics or prefixes to the host.
struct RankingGroupStatistics {
  float normalization, gradient_rounding, hessian_rounding, weight;
  double cost_rounding;
  std::uint64_t padding;
};
struct RankingGradientTrace {
  Span<RankingGroupStatistics> groups;
  Span<double> global_cost_rounding;
  Span<std::array<double, 2>> map_prefix;
};
// Top-k or mean-sampled LambdaRank. Nonempty bias vectors enable unbiased loss
// and are updated on GPU, then copied back for objective serialization. Pair
// construction, statistics, normalization and FP32 gradient accumulation execute
// on GPU; output is in original row order. Optional mean_pair_trace receives
// raw CUDA-ordered query-relative prediction ranks for validation only. It must
// contain exactly rows*pairs entries; production callers leave it empty.
void RankingGradient(RankingLossKind kind, Span<float const> predictions,
                     linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                     Span<float const> weights, std::uint32_t pairs, bool normalization,
                     bool score_normalization, bool exponential, double weight_norm,
                     Span<double const> discounts, Span<double const> inverse_idcg,
                     Span<GradientPair> output, bool mean = false, std::uint32_t seed = 0,
                     RankingBias bias = {}, Span<std::array<std::uint32_t, 2>> mean_pair_trace = {},
                     RankingGradientTrace diagnostics = {});
// GPU sort and weighted query scoring in one command. NDCG uses the existing
// cache's discounts and inverse ideal gains. Caller validates label semantics.
RankingSums RankingScore(RankingMetricKind kind, linalg::VectorView<float const> predictions,
                         linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                         Span<float const> weights, std::uint32_t topk, bool minus = false,
                         bool exponential = true, Span<double const> discounts = {},
                         Span<double const> inverse_idcg = {});
}  // namespace xgboost::common::metal
