/** Copyright 2026, XGBoost Contributors */
#if defined(XGBOOST_USE_METAL)
#include "metal_cuda_sketch.h"

#include <algorithm>
#include <limits>

#include "../collective/communicator-inl.h"
#include "categorical.h"
#include "hist_util.h"
#include "metal_sketch.h"
#include "metal_vector.h"

namespace xgboost::common::metal {
HistogramCuts CudaSketchOnDMatrix(Context const* ctx, DMatrix* data, bst_bin_t max_bin,
                                  Span<float const> hessian) {
  auto const& info = data->Info();
  auto types = info.feature_types.ConstHostSpan();
  auto weights = ConstDeviceSpan(info.weights_);
  HostDeviceVector<float> unified;
  // Unweighted input must stay unweighted: CUDA uses integer sample positions,
  // not a scan of FP32 ones. The existing kernel uses the same UnifyWeight formula.
  if (!hessian.empty() || (info.IsRanking() && !weights.empty())) {
    unified.SetDevice(DeviceOrd::MPS());
    unified.Resize(info.num_row_);
    MakeSketchWeights(
        info.num_row_, hessian, weights,
        info.IsRanking() ? Span<bst_group_t const>{info.group_ptr_} : Span<bst_group_t const>{},
        unified.DeviceSpan());
    weights = unified.ConstDeviceSpan();
  }
  auto sketch = MakeCudaSketch(info.num_col_, max_bin, types, weights);
  constexpr std::size_t kWindow = 1 << 26;  // hist_util.cuh::kSketchBatchNumElements
  for (auto const& page : data->GetBatches<SparsePage>()) {
    auto entries = ConstDeviceSpan(page.data);
    auto ptr = page.offset.ConstHostSpan();
    for (std::size_t begin = 0; begin < entries.size(); begin += kWindow) {
      auto end = std::min(entries.size(), begin + kWindow);
      auto first = std::upper_bound(ptr.begin(), ptr.end(), begin) - ptr.begin() - 1;
      auto last = std::lower_bound(ptr.begin(), ptr.end(), end) - ptr.begin();
      auto rows = std::max<bst_idx_t>(1, last - first);
      sketch->Push(entries.subspan(begin, end - begin), ptr, begin, page.base_rowid, rows);
    }
  }
  // CUDA SketchContainer::MakeCuts also materializes the final cuts on the host.
  sketch->AllReduce(ctx);
  auto summary = sketch->Read();
  return MakeCudaCuts(summary, info.num_col_, max_bin, types);
}
HistogramCuts MakeCudaCuts(CudaSketchSummary const& summary, bst_feature_t columns,
                           bst_bin_t max_bin, Span<FeatureType const> types) {
  HistogramCuts cuts{static_cast<bst_feature_t>(columns)};
  auto& ptr = cuts.cut_ptrs_.HostVector();
  auto& values = cuts.cut_values_.HostVector();
  float max_cat = -1;
  bool has_cat = false;
  WQSummaryContainer column;
  for (bst_feature_t f = 0; f < columns; ++f) {
    auto begin = summary.ptr[f], end = summary.ptr[f + 1];
    if (IsCat(types, f)) {
      has_cat = true;
      auto maximum = begin == end ? 0.f : summary.entries[end - 1].value;
      for (auto i = begin; i < end; ++i) {
        if (InvalidCat(summary.entries[i].value)) InvalidCategory();
      }
      CheckMaxCat(maximum, std::max<std::size_t>(1, end - begin));
      max_cat = std::max(max_cat, maximum);
      for (std::size_t c = 0; c <= static_cast<std::size_t>(maximum); ++c) values.push_back(c);
    } else {
      column.Reserve(end - begin);
      std::copy(summary.entries.begin() + begin, summary.entries.begin() + end,
                column.space.begin());
      column.SetSize(end - begin);
      auto queried = column.QueryCutValues(max_bin);
      values.insert(values.end(), queried.begin(), queried.end());
    }
    CHECK_LE(values.size(), std::numeric_limits<bst_bin_t>::max());
    ptr[f + 1] = values.size();
  }
  cuts.SetCategorical(has_cat, max_cat);
  return cuts;
}
}  // namespace xgboost::common::metal
#endif  // defined(XGBOOST_USE_METAL)
