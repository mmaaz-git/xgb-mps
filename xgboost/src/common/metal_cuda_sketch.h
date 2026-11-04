/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "metal_adapter.h"
#include "quantile.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// CUDA's SketchContainer algorithm, separate from HostSketchContainer's level queue.
// Input windows are CSR entries. Summaries and column pointers remain on Metal
// between pushes; Read is used only for final cuts and native diagnostics.
struct CudaSketchSummary {
  std::vector<std::uint64_t> ptr;
  std::vector<WQSummary<>::Entry> entries;
};
class CudaSketch {
 public:
  virtual ~CudaSketch() = default;
  virtual void Push(Span<::xgboost::Entry const> entries, Span<bst_idx_t const> row_ptr,
                    std::size_t begin, bst_idx_t base_rowid, bst_idx_t rows_in_window) = 0;
  virtual void SetWeights(Span<float const> weights) = 0;
  virtual void Push(AdapterInput const& input) = 0;
  virtual CudaSketchSummary Read() const = 0;
};
std::unique_ptr<CudaSketch> MakeCudaSketch(bst_feature_t columns, bst_bin_t max_bin,
                                           Span<FeatureType const> types,
                                           Span<float const> row_weights);
HistogramCuts MakeCudaCuts(CudaSketchSummary const& summary, bst_feature_t columns,
                           bst_bin_t max_bin, Span<FeatureType const> types);
HistogramCuts CudaSketchOnDMatrix(Context const* ctx, DMatrix* data, bst_bin_t max_bin,
                                  Span<float const> hessian);
}  // namespace xgboost::common::metal
