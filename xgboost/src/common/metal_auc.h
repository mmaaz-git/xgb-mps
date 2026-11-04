/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <vector>

#include "xgboost/linalg.h"
#include "xgboost/span.h"
namespace xgboost::common::metal {
struct ROCAUCSums {
  double fp, tp, area;
};
struct RankingAUCSums {
  double area, valid_groups;
};
// Stable descending scores and binary64 GPU scans/reductions. Results are
// unnormalized; the caller retains XGBoost's prevalence/macro aggregation rules.
// Predictions must not contain NaNs (sorting has no CPU reference ordering).
// Calls are synchronous. Eligible shared input/output spans bind directly, with
// guarded upload/publication fallbacks. Outputs may alias inputs: publication
// follows all input reads. Optional diagnostics explicitly return host arrays.
void ROCAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
            Span<float const> weights, bool multiclass, Span<ROCAUCSums> output);
// Optional native validation diagnostics; unused by production metric callers.
// Interleaved (FP,TP) terms/prefixes are output-major; totals has one pair/output.
struct AUCTrace {
  std::vector<double> terms, prefixes, totals;
};
// CUDA PR: valid binary and nonempty multi-output curves return (1,1,area);
// invalid binary/empty curves return zeros, invalid nonempty multi-output area is NaN.
void PRAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
           Span<float const> weights, bool multiclass, Span<ROCAUCSums> output,
           AUCTrace* trace = nullptr);
// CUDA per-output metadata: flat (local_area, aggregation_weight, integral) triples.
// The caller performs its collective sum before passing these to NormalizeAUC.
void MultiAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
              Span<float const> weights, bool multiclass, bool is_roc, Span<double> output);
double NormalizeAUC(Span<double const> summaries);
// Ranking ROC counts every pair in stable descending-label order, including
// equal labels. CUDA pair terms use FP32 differences and squared query weights.
// Overflowing squares return NaN scores; zero/NaN squares return zero. Queries
// shorter than three rows give zero score/count. Labels must not contain NaNs.
// Optional diagnostics expose each group's sum/count.
RankingAUCSums RankingROCAUC(linalg::VectorView<float const> predictions,
                             linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                             Span<float const> weights, Span<RankingAUCSums> individual = {});
// Grouped PR uses query weights, segmented binary64 prefixes and the CUDA
// continuous integral, with at least two documents required for a valid query. Prediction NaNs have
// no CPU sorting contract.
RankingAUCSums RankingPRAUC(linalg::VectorView<float const> predictions,
                            linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                            Span<float const> weights, Span<RankingAUCSums> individual = {});
}  // namespace xgboost::common::metal
