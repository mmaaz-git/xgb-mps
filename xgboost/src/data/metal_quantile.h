/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/metal_adapter.h"
#include "cat_container.h"
#include "proxy_dmatrix.h"
#include "quantile_dmatrix.h"

namespace xgboost::data::metal_impl {
// The CUDA adapter frontend uses raw flattened 2^26-slot windows, including
// missing values and partial boundary rows. Preserve those pruning boundaries.
template <class Fn>
void AdapterWindows(Context const* ctx, DMatrixProxy const* proxy, float missing, Fn&& fn) {
  cpu_impl::DispatchAny<false>(proxy, [&](auto const& adapter) {
    using A = typename std::decay_t<decltype(adapter)>::element_type;
    if constexpr (std::is_same_v<A, ArrayAdapter> || std::is_same_v<A, ColumnarAdapter>) {
      std::vector<std::int32_t> map;
      enc::MappingView mapping;
      if constexpr (std::is_same_v<A, ColumnarAdapter>) {
        if (adapter->HasRefCategorical()) {
          CatContainer cats{adapter->RefCats(), true};
          cats.Sort(ctx);
          auto encoded = ::xgboost::cpu_impl::MakeCatAccessor(ctx, adapter->Cats(), &cats);
          mapping = std::get<0>(encoded).enc;
          map = std::move(std::get<1>(encoded));
        }
      }
      auto cols = adapter->NumColumns(), rows = adapter->NumRows();
      CHECK_GT(cols, 0);
      CHECK_LE(rows, std::numeric_limits<bst_idx_t>::max() / cols);
      bst_idx_t total = rows * cols;
      constexpr bst_idx_t kWindow = 1ULL << 26;
      for (bst_idx_t begin = 0; begin < total; begin += kWindow) {
        auto input = common::metal::PrepareAdapter(*adapter, missing, begin,
                                                   std::min(kWindow, total - begin), mapping);
        fn(*input);
      }
    } else {
      LOG(FATAL) << "Metal QuantileDMatrix construction currently accepts dense or columnar input.";
    }
  });
}

// The first batch is already fetched, as in cuda_impl::MakeSketches. Leaves the
// iterator reset; only the summary, shape and per-batch counts survive the pass.
void MakeSketches(Context const* ctx,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>* iter,
                  DMatrixProxy* proxy, std::shared_ptr<DMatrix> ref, BatchParam const& p,
                  float missing, common::HistogramCuts* cuts, ExternalDataInfo* ext,
                  std::vector<FeatureType>* types);
}  // namespace xgboost::data::metal_impl
