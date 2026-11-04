/**
 * Copyright 2024-2025, XGBoost Contributors
 */
#include "quantile_dmatrix.h"

#include <numeric>  // for accumulate

#include "../collective/allreduce.h"         // for Allreduce
#include "../collective/communicator-inl.h"  // for IsDistributed
#include "../common/error_msg.h"             // for InconsistentCategories
#include "../common/threading_utils.h"       // for ParallelFor
#include "cat_container.h"                   // for CatContainer
#include "gradient_index.h"                  // for GHistIndexMatrix
#include "proxy_dmatrix.h"                   // for DispatchAny
#include "xgboost/collective/result.h"       // for SafeColl
#include "xgboost/linalg.h"                  // for Tensor

#if defined(XGBOOST_USE_METAL)
#include "../common/metal_cuda_sketch.h"
#include "../common/metal_sketch.h"
#include "metal_page_source.h"
#include "metal_quantile.h"
#endif

namespace xgboost::data {
void GetCutsFromRef(Context const* ctx, std::shared_ptr<DMatrix> ref, bst_feature_t n_features,
                    BatchParam p, common::HistogramCuts* p_cuts) {
  CHECK(ref);
  CHECK(p_cuts);
  p.forbid_regen = true;
  // Fetch cuts from GIDX
  auto csr = [&] {
    // A reference needs only cut metadata. Keep an existing MPS matrix on its
    // compressed path even when the new matrix is being constructed for CPU.
    auto cut_ctx = ref->Ctx()->IsMPS() && ref->PageExists<GHistIndexMatrix>() ? ref->Ctx() : ctx;
    for (auto const& page : ref->GetBatches<GHistIndexMatrix>(cut_ctx, p)) {
      *p_cuts = page.cut;
      break;
    }
  };
  // Fetch cuts from Ellpack.
  auto ellpack = [&] {
    for (auto const& page : ref->GetBatches<EllpackPage>(ctx, p)) {
      GetCutsFromEllpack(page, p_cuts);
      break;
    }
  };

  if (ref->PageExists<GHistIndexMatrix>() && ref->PageExists<EllpackPage>()) {
    // Both exists
    if (ctx->IsCUDA()) {
      ellpack();
    } else {
      csr();
    }
  } else if (ref->PageExists<GHistIndexMatrix>()) {
    csr();
  } else if (ref->PageExists<EllpackPage>()) {
    ellpack();
  } else {
    // None exist
    if (ctx->IsCUDA()) {
      ellpack();
    } else {
      csr();
    }
  }
  CHECK_EQ(ref->Info().num_col_, n_features)
      << "Invalid ref DMatrix, different number of features.";
}

#if !defined(XGBOOST_USE_CUDA)
void GetCutsFromEllpack(EllpackPage const&, common::HistogramCuts*) { common::AssertGPUSupport(); }
#endif

#if defined(XGBOOST_USE_METAL)
namespace metal_impl {
void MakeSketches(Context const* ctx,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>* iter,
                  DMatrixProxy* proxy, std::shared_ptr<DMatrix> ref, BatchParam const& p,
                  float missing, common::HistogramCuts* cuts, ExternalDataInfo* p_ext,
                  std::vector<FeatureType>* p_types) {
  CHECK_EQ(collective::GetWorldSize(), 1) << "Distributed Metal sketching is not implemented.";
  auto& ext = *p_ext;
  auto& types = *p_types;
  std::unique_ptr<common::metal::CudaSketch> sketch;
  do {
    auto cols = BatchColumns(proxy);
    auto rows = BatchSamples(proxy);
    CHECK_GE(cols, 1);
    auto cats = cpu_impl::BatchCats(proxy);
    if (!ext.n_batches) {
      ext.n_features = cols;
      ext.cats = std::make_shared<CatContainer>(cats, BatchCatsIsRef(proxy));
      types = ref ? ref->Info().feature_types.ConstHostVector()
                  : proxy->Info().feature_types.ConstHostVector();
      CHECK(types.empty() || types.size() == cols);
      if (!ref) sketch = common::metal::MakeCudaSketch(cols, p.max_bin, types, {});
    } else {
      CHECK_EQ(cols, ext.n_features) << "Inconsistent number of columns.";
      CHECK_EQ(cats.n_total_cats, ext.cats->NumCatsTotal()) << error::InconsistentCategories();
    }
    if (sketch) {
      auto weights = proxy->Info().weights_.ConstHostSpan();
      std::vector<float> expanded;
      if (!weights.empty() && proxy->Info().IsRanking()) {
        expanded = common::metal::MakeSketchWeights(rows, {}, weights, proxy->Info().group_ptr_);
        weights = expanded;
      }
      CHECK(weights.empty() || weights.size() == rows);
      sketch->SetWeights(weights);
    }
    bst_idx_t nnz = 0;
    AdapterWindows(ctx, proxy, missing, [&](auto const& input) {
      nnz += input.NonMissing();
      ext.row_stride = std::max(ext.row_stride, input.RowStride());
      if (sketch) sketch->Push(input);
    });
    ext.batch_nnz.push_back(nnz);
    ext.nnz += nnz;
    ext.accumulated_rows += rows;
    ext.base_rowids.push_back(ext.accumulated_rows);
    ++ext.n_batches;
  } while (iter->Next());
  iter->Reset();
  ext.Validate();
  if (ref)
    GetCutsFromRef(ctx, ref, ext.n_features, p, cuts);
  else
    *cuts = common::metal::MakeCudaCuts(sketch->Read(), ext.n_features, p.max_bin, types);
  sketch.reset();
}
std::shared_ptr<GHistIndexMatrix> MakeIndexBatch(Context const* ctx, DMatrixProxy* proxy,
                                                 float missing, BatchParam const& p,
                                                 common::HistogramCuts const& cuts,
                                                 MetaInfo const& info, bst_idx_t expected_rows,
                                                 bst_idx_t base_row, bst_idx_t expected_nnz) {
  auto rows = BatchSamples(proxy);
  CHECK_EQ(rows, expected_rows);
  CHECK_EQ(BatchColumns(proxy), info.num_col_);
  auto page = std::make_shared<GHistIndexMatrix>(rows, base_row, common::HistogramCuts{cuts},
                                                 p.max_bin, info.IsDense());
  page->ResizeIndex(ctx, expected_nnz, info.IsDense());
  if (info.IsDense()) page->index.SetBinOffset(page->cut.Ptrs());
  bst_idx_t offset = 0;
  AdapterWindows(ctx, proxy, missing, [&](auto const& input) {
    input.WriteIndex(info.feature_types.ConstHostSpan(), page.get(), 0, offset);
    offset += input.NonMissing();
  });
  CHECK_EQ(offset, expected_nnz) << "Non-missing counts changed between iterations";
  page->InitColumns(ctx, p.sparse_thresh);
  proxy->Info().num_row_ = rows;
  proxy->Info().num_col_ = info.num_col_;
  proxy->Info().num_nonzero_ = offset;
  return page;
}
}  // namespace metal_impl
MetalPageSource::MetalPageSource(
    Context const* ctx,
    std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
    DMatrixProxy* proxy, BatchParam p, ExtMemConfig const& config, std::shared_ptr<Cache> cache,
    common::HistogramCuts cuts, ExternalDataInfo const& ext, MetaInfo* info, bool validation)
    : Super{config.missing, ctx->Threads(), static_cast<bst_feature_t>(ext.n_features),
            std::move(cache)} {
  auto layout = common::metal::MakeEllpackInfo(cuts, ext.row_stride, info->IsDense());
  std::vector<bst_idx_t> batch_rows;
  for (std::size_t i = 0; i < ext.n_batches; ++i)
    batch_rows.push_back(ext.base_rowids[i + 1] - ext.base_rowids[i]);
  auto plan = common::metal::PlanEllpackCache(layout, batch_rows, validation,
                                              config.cache_host_ratio, config.min_cache_page_bytes,
                                              common::metal::CacheWorkingSetBytes());
  metal_cache_->ctx = *ctx;
  metal_cache_->cuts = cuts;
  metal_cache_->max_bin = p.max_bin;
  metal_cache_->sparse_thresh = p.sparse_thresh;
  p.n_prefetch_batches = 2;
  this->param_ = p;
  for (auto rows : plan.rows) batch_ptr_.push_back(batch_ptr_.back() + rows);
  LOG(INFO) << "Metal cache pages: " << plan.rows.size() << ", host ratio: " << plan.host_ratio
            << ", min page bytes: " << plan.min_page_bytes;
  iter->Reset();
  std::size_t batch = 0;
  bst_idx_t offset = 0;
  while (iter->Next()) {
    CHECK_LT(batch, ext.n_batches) << "Different number of batches returned between iterations";
    CHECK_EQ(BatchSamples(proxy), batch_rows[batch]);
    CHECK_EQ(BatchColumns(proxy), ext.n_features);
    auto packed =
        common::metal::MakeCompressedPage(layout, batch_rows[batch], ext.base_rowids[batch]);
    packed->BeginAdapter(cuts, info->feature_types.ConstHostSpan());
    bst_idx_t nnz = 0;
    metal_impl::AdapterWindows(ctx, proxy, config.missing, [&](auto const& input) {
      packed->PushAdapter(input);
      nnz += input.NonMissing();
    });
    CHECK_EQ(nnz, ext.batch_nnz[batch]) << "Non-missing counts changed between iterations";
    packed->EndAdapter();
    proxy->Info().num_row_ = batch_rows[batch];
    proxy->Info().num_col_ = ext.n_features;
    proxy->Info().num_nonzero_ = nnz;
    info->Extend(proxy->Info(), false, true);
    auto target = plan.mapping[batch];
    if (target == metal_cache_->pages.size()) {
      if (!metal_cache_->pages.empty()) metal_cache_->pages.back()->Commit(plan.host_ratio);
      metal_cache_->pages.push_back(common::metal::MakeCompressedPage(
          layout, plan.rows[target], batch_ptr_[target], plan.bytes[target]));
      offset = 0;
    }
    metal_cache_->pages.back()->Copy(*packed, offset);
    offset += batch_rows[batch];
    ++batch;
  }
  CHECK_EQ(batch, ext.n_batches) << "Different number of batches returned between iterations";
  metal_cache_->pages.back()->Commit(plan.host_ratio);
  // Validate the actual compressed symbols, without building host bin/column
  // mirrors. Only one non-missing total per completed cache page is read back.
  for (auto const& page : metal_cache_->pages) nnz_ += page->NonMissing();
  CHECK_EQ(nnz_, ext.nnz);
  for (auto bytes : plan.bytes) cache_info_->Push(bytes);
  cache_info_->Commit();
  metal_cache_->offsets = cache_info_->offset;
  // No callback or input-array references survive cache construction.
  iter.reset();
  this->Fetch();
}
#endif

namespace cpu_impl {
// Synchronize feature type in case of empty DMatrix
void SyncFeatureType(Context const* ctx, std::vector<FeatureType>* p_h_ft) {
  if (!collective::IsDistributed()) {
    return;
  }
  auto& h_ft = *p_h_ft;
  bst_idx_t n_ft = h_ft.size();
  collective::SafeColl(collective::Allreduce(ctx, &n_ft, collective::Op::kMax));
  if (!h_ft.empty()) {
    // Check correct size if this is not an empty DMatrix.
    CHECK_EQ(h_ft.size(), n_ft);
  }
  if (n_ft > 0) {
    h_ft.resize(n_ft);
    auto ptr = reinterpret_cast<std::underlying_type_t<FeatureType>*>(h_ft.data());
    collective::SafeColl(
        collective::Allreduce(ctx, linalg::MakeVec(ptr, h_ft.size()), collective::Op::kMax));
  }
}

void GetDataShape(Context const* ctx, DMatrixProxy* proxy,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>* iter, float missing,
                  ExternalDataInfo* p_info) {
  auto& info = *p_info;

  auto const is_valid = data::IsValidFunctor{missing};
  auto nnz_cnt = [&]() {
    return DispatchAny(proxy, [&](auto const& value) {
      bst_idx_t n_threads = ctx->Threads();
      bst_idx_t n_features = info.column_sizes.size();
      linalg::Tensor<bst_idx_t, 2> column_sizes_tloc({n_threads, n_features}, DeviceOrd::CPU());
      column_sizes_tloc.Data()->Fill(0ul);
      auto view = column_sizes_tloc.HostView();
      common::ParallelFor(value.Size(), n_threads, common::Sched::Static(256), [&](auto i) {
        auto const& line = value.GetLine(i);
        for (bst_idx_t j = 0; j < line.Size(); ++j) {
          data::COOTuple const& elem = line.GetElement(j);
          if (is_valid(elem)) {
            view(omp_get_thread_num(), elem.column_idx)++;
          }
        }
      });
      auto ptr = column_sizes_tloc.Data()->HostPointer();
      auto result = std::accumulate(ptr, ptr + column_sizes_tloc.Size(), static_cast<bst_idx_t>(0));
      for (bst_idx_t tidx = 0; tidx < n_threads; ++tidx) {
        for (bst_idx_t fidx = 0; fidx < n_features; ++fidx) {
          info.column_sizes[fidx] += view(tidx, fidx);
        }
      }
      return result;
    });
  };

  /**
   * CPU impl needs an additional loop for accumulating the column size.
   */
  do {
    // We use do while here as the first batch is fetched in ctor
    if (info.n_features == 0) {
      info.n_features = BatchColumns(proxy);
      collective::SafeColl(collective::Allreduce(ctx, &info.n_features, collective::Op::kMax));
      info.column_sizes.clear();
      info.column_sizes.resize(info.n_features, 0);
      p_info->cats =
          std::make_shared<CatContainer>(cpu_impl::BatchCats(proxy), BatchCatsIsRef(proxy));
    } else {
      CHECK_EQ(info.n_features, BatchColumns(proxy)) << "Inconsistent number of columns.";
      auto cats = cpu_impl::BatchCats(proxy);
      CHECK_EQ(cats.n_total_cats, p_info->cats->NumCatsTotal()) << error::InconsistentCategories();
    }
    bst_idx_t batch_size = BatchSamples(proxy);
    info.batch_nnz.push_back(nnz_cnt());
    info.base_rowids.push_back(batch_size);
    info.nnz += info.batch_nnz.back();
    info.accumulated_rows += batch_size;
    info.n_batches++;
  } while (iter->Next());
  iter->Reset();

  std::partial_sum(info.base_rowids.cbegin(), info.base_rowids.cend(), info.base_rowids.begin());
}

void MakeSketches(Context const* ctx,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>* iter,
                  DMatrixProxy* proxy, std::shared_ptr<DMatrix> ref, float missing,
                  common::HistogramCuts* cuts, BatchParam const& p, MetaInfo const& info,
                  ExternalDataInfo const& ext_info, std::vector<FeatureType>* p_h_ft) {
  std::unique_ptr<common::HostSketchContainer> p_sketch;
  auto& h_ft = *p_h_ft;
  bst_idx_t accumulated_rows = 0;
  if (ref) {
    GetCutsFromRef(ctx, ref, info.num_col_, p, cuts);
    h_ft = ref->Info().feature_types.HostVector();
  } else {
    size_t i = 0;
    while (iter->Next()) {
      if (!p_sketch) {
        h_ft = proxy->Info().feature_types.ConstHostVector();
        cpu_impl::SyncFeatureType(ctx, &h_ft);
        p_sketch = std::make_unique<common::HostSketchContainer>(
            ctx, p.max_bin, h_ft, ext_info.column_sizes, !proxy->Info().group_ptr_.empty());
      }
      DispatchAny(proxy, [&](auto const& batch) {
        proxy->Info().num_nonzero_ = ext_info.batch_nnz[i];
        // We don't need base row idx here as Info is from proxy and the number of rows in
        // it is consistent with data batch.
        p_sketch->PushAdapterBatch(batch, 0, proxy->Info(), missing);
      });
      accumulated_rows += BatchSamples(proxy);
      ++i;
    }
    iter->Reset();
    CHECK_EQ(accumulated_rows, info.num_row_);

    CHECK(p_sketch);
    *cuts = p_sketch->MakeCuts(ctx, info);
  }

  if (!h_ft.empty()) {
    CHECK_EQ(h_ft.size(), ext_info.n_features);
  }
}
}  // namespace cpu_impl
}  // namespace xgboost::data
