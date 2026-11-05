/**
 * Copyright 2022-2025, XGBoost contributors
 */
#include "iterative_dmatrix.h"

#include <algorithm>  // for copy
#include <cstddef>    // for size_t
#include <memory>     // for shared_ptr
#include <utility>    // for move
#include <vector>     // for vector

#include "../common/categorical.h"  // for IsCat
#include "../common/error_msg.h"    // for Unreachable
#include "../common/hist_util.h"    // for HistogramCuts
#include "../tree/param.h"          // FIXME(jiamingy): Find a better way to share this parameter.
#include "batch_utils.h"            // for RegenGHist
#include "cat_container.h"          // for SyncCategories
#include "gradient_index.h"         // for GHistIndexMatrix
#include "proxy_dmatrix.h"          // for DataIterProxy, DispatchAny
#include "quantile_dmatrix.h"       // for GetCutsFromRef
#include "quantile_dmatrix.h"       // for GetDataShape, MakeSketches
#include "simple_batch_iterator.h"  // for SimpleBatchIteratorImpl
#include "xgboost/data.h"           // for FeatureType, DMatrix
#include "xgboost/logging.h"

#if defined(XGBOOST_USE_METAL)
#include "../common/metal_ellpack.h"
#include "metal_quantile.h"
#endif

namespace xgboost::data {
IterativeDMatrix::IterativeDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                                   std::shared_ptr<DMatrix> ref, DataIterResetCallback* reset,
                                   XGDMatrixCallbackNext* next, float missing, int nthread,
                                   bst_bin_t max_bin, std::string const& device)
    : proxy_{proxy} {
  // The external iterator, fetch the first batch
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_handle, reset, next};
  iter.Reset();
  bool valid = iter.Next();
  CHECK(valid) << "Iterative DMatrix must have at least 1 batch.";

  auto pctx = MakeProxy(proxy_)->Ctx();

  Context ctx;
  ctx.Init(Args{{"nthread", std::to_string(nthread)},
                {"device", device.empty() ? pctx->DeviceName() : device}});
  CHECK(ctx.IsMPS() ? !pctx->IsCUDA() : !pctx->IsMPS() && ctx.Device() == pctx->Device())
      << "Construction device must match the input device, except MPS accepts host arrays.";
  // hardcoded parameter.
  BatchParam p{max_bin, tree::TrainParam::DftSparseThreshold()};

  if (ctx.IsMPS()) {
    this->InitFromMPS(&ctx, p, std::move(iter), missing, ref);
  } else if (ctx.IsCUDA()) {
    this->InitFromCUDA(&ctx, p, std::move(iter), missing, ref);
  } else {
    this->InitFromCPU(&ctx, p, std::move(iter), missing, ref);
  }

  this->fmat_ctx_ = ctx;
  this->batch_ = p;

  SyncCategories(&ctx, info_.Cats(), info_.num_row_ == 0);

  LOG(INFO) << "Finished constructing the `IterativeDMatrix`: (" << this->Info().num_row_ << ", "
            << this->Info().num_col_ << ", " << this->info_.num_nonzero_ << ").";
}

#if defined(XGBOOST_USE_METAL)
void IterativeDMatrix::InitFromMPS(
    Context const* ctx, BatchParam const& p,
    DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>&& iter, float missing,
    std::shared_ptr<DMatrix> ref) {
  auto proxy = MakeProxy(proxy_);
  ExternalDataInfo ext;
  std::vector<FeatureType> types;
  common::HistogramCuts cuts{0};
  metal_impl::MakeSketches(ctx, &iter, proxy, ref, p, missing, &cuts, &ext, &types);
  ext.SetInfo(ctx, true, &info_);
  auto layout = common::metal::MakeEllpackInfo(cuts, ext.row_stride, info_.IsDense());
  auto compressed = common::metal::MakeCompressedPage(layout, ext.accumulated_rows, 0);
  bst_idx_t row_base = 0, entry_offset = 0, batch = 0;
  while (iter.Next()) {
    CHECK_LT(batch, ext.n_batches) << "Different number of batches returned between 2 iterations";
    auto rows = BatchSamples(proxy);
    CHECK_EQ(rows, ext.base_rowids[batch + 1] - ext.base_rowids[batch]);
    metal_impl::CheckBatchColumns(proxy, ext.n_features);
    auto input_page = common::metal::MakeCompressedPage(layout, rows, row_base);
    input_page->BeginAdapter(cuts, types);
    auto first = entry_offset;
    metal_impl::AdapterWindows(ctx, proxy, missing, [&](auto const& input) {
      input_page->PushAdapter(input);
      entry_offset += input.NonMissing();
    });
    CHECK_EQ(entry_offset - first, ext.batch_nnz[batch])
        << "Non-missing counts changed between iterations";
    input_page->EndAdapter();
    compressed->Copy(*input_page, row_base);
    proxy->Info().num_row_ = rows;
    proxy->Info().num_col_ = ext.n_features;
    proxy->Info().num_nonzero_ = ext.batch_nnz[batch];
    metal_impl::InheritEmptyFeatureTypes(proxy, types);
    if (ext.n_batches != 1) info_.Extend(std::move(proxy->Info()), false, true);
    row_base += rows;
    ++batch;
  }
  CHECK_EQ(batch, ext.n_batches) << "Different number of batches returned between 2 iterations";
  CHECK_EQ(row_base, ext.accumulated_rows);
  CHECK_EQ(entry_offset, ext.nnz);
  if (ext.n_batches == 1) {
    info_ = std::move(proxy->Info());
    ext.SetInfo(ctx, false, &info_);
  }
  info_.feature_types.HostVector() = types;
  compressed->Commit(0.0);
  CHECK_EQ(compressed->NonMissing(), ext.nnz);
  metal_ghist_ = std::make_shared<GHistIndexMatrix>();
  compressed->View(cuts, p.max_bin, false, metal_ghist_.get());
  iter.Reset();
}
#else
void IterativeDMatrix::InitFromMPS(Context const*, BatchParam const&,
                                   DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>&&,
                                   float, std::shared_ptr<DMatrix>) {
  LOG(FATAL) << "Metal QuantileDMatrix requires a build with USE_METAL=ON.";
}
#endif

void IterativeDMatrix::InitFromCPU(
    Context const* ctx, BatchParam const& p,
    DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>&& iter, float missing,
    std::shared_ptr<DMatrix> ref) {
  DMatrixProxy* proxy = MakeProxy(proxy_);
  CHECK(proxy);

  common::HistogramCuts cuts{0};
  ExternalDataInfo ext_info;
  cpu_impl::GetDataShape(ctx, proxy, &iter, missing, &ext_info);
  ext_info.SetInfo(ctx, true, &this->info_);

  /**
   * Generate quantiles
   */
  std::vector<FeatureType> h_ft;
  cpu_impl::MakeSketches(ctx, &iter, proxy, ref, missing, &cuts, p, this->info_, ext_info, &h_ft);

  /**
   * Generate gradient index.
   */
  this->ghist_ = std::make_unique<GHistIndexMatrix>(this->info_, std::move(cuts), p.max_bin);
  std::size_t rbegin = 0;
  std::size_t prev_sum = 0;
  std::size_t i = 0;
  while (iter.Next()) {
    cpu_impl::DispatchAny(proxy, [&](auto const& batch) {
      proxy->Info().num_nonzero_ = ext_info.batch_nnz[i];
      this->ghist_->PushAdapterBatch(ctx, rbegin, prev_sum, batch, missing, h_ft, p.sparse_thresh,
                                     Info().num_row_);
    });
    if (ext_info.n_batches != 1) {
      this->info_.Extend(std::move(proxy->Info()), false, true);
    }
    auto batch_size = BatchSamples(proxy);
    prev_sum = this->ghist_->row_ptr[rbegin + batch_size];
    rbegin += batch_size;
    ++i;
  }
  iter.Reset();
  CHECK_EQ(rbegin, Info().num_row_);
  CHECK_EQ(this->ghist_->Features(), Info().num_col_);

  /**
   * Generate column matrix
   */
  bst_idx_t accumulated_rows = 0;
  while (iter.Next()) {
    cpu_impl::DispatchAny(proxy, [&](auto const& batch) {
      this->ghist_->PushAdapterBatchColumns(ctx, batch, missing, accumulated_rows);
    });
    accumulated_rows += BatchSamples(proxy);
  }
  iter.Reset();
  CHECK_EQ(accumulated_rows, this->info_.num_row_);

  if (ext_info.n_batches == 1) {
    this->info_ = std::move(proxy->Info());
    ext_info.SetInfo(ctx, false, &this->info_);
    CHECK_EQ(proxy->Info().labels.Size(), 0);
  }

  info_.feature_types.HostVector() = h_ft;
}

BatchSet<GHistIndexMatrix> IterativeDMatrix::GetGradientIndex(Context const* ctx,
                                                              BatchParam const& param) {
#if defined(XGBOOST_USE_METAL)
  std::lock_guard<std::mutex> lock{metal_matrix_mutex_};
#endif
  if (param.Initialized()) {
    detail::CheckParam(this->batch_, param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }
  if (!ellpack_ && !ghist_
#if defined(XGBOOST_USE_METAL)
      && !metal_ghist_
#endif
  ) {
    LOG(FATAL) << "`QuantileDMatrix` not initialized.";
  }

#if defined(XGBOOST_USE_METAL)
  if (ctx && ctx->IsMPS()) {
    if (!metal_ghist_) {
      CHECK(ghist_);
      ghist_->RequireHost();
      // CUDA's GHist-to-ELLPACK constructor also computes stride from host rows.
      bst_idx_t stride = 0;
      for (bst_idx_t r = 0; r < ghist_->Size(); ++r)
        stride = std::max<bst_idx_t>(stride, ghist_->row_ptr[r + 1] - ghist_->row_ptr[r]);
      auto layout = common::metal::MakeEllpackInfo(ghist_->cut, stride, ghist_->IsDense());
      auto compressed = common::metal::MakeCompressedPage(layout, ghist_->Size(), 0);
      compressed->Pack(*ghist_);
      compressed->Commit(0.0);
      CHECK_EQ(compressed->NonMissing(), Info().num_nonzero_);
      auto view = std::make_shared<GHistIndexMatrix>();
      compressed->View(ghist_->cut, batch_.max_bin, false, view.get());
      metal_ghist_ = std::move(view);
    }
  } else if (!ghist_ && metal_ghist_) {
    CHECK_GE(batch_.max_bin, 2)
        << "Internal ELLPACK load restores only cuts and compressed pages, not training metadata.";
    // Build a separate CPU representation, then publish only when complete.
    // Existing Metal batches keep their immutable shape and compressed buffers.
    auto host = std::make_shared<GHistIndexMatrix>();
    host->cut = metal_ghist_->cut;
    host->base_rowid = metal_ghist_->base_rowid;
    host->max_numeric_bins_per_feat = metal_ghist_->max_numeric_bins_per_feat;
    host->SetDense(metal_ghist_->IsDense());
    host->SetMetalRows(metal_ghist_->Size());
    host->metal_page = std::atomic_load(&metal_ghist_->metal_page);
    auto cpu_ctx = ctx ? ctx->MakeCPU() : fmat_ctx_.MakeCPU();
    common::metal::MaterializeHost(&cpu_ctx, batch_.sparse_thresh, host.get());
    ghist_ = std::move(host);
  }
#endif
  if (!ghist_
#if defined(XGBOOST_USE_METAL)
      && !metal_ghist_
#endif
  ) {
    if (!ctx->IsCUDA()) {
      ghist_ = std::make_shared<GHistIndexMatrix>(ctx, Info(), *ellpack_, param);
    } else if (!fmat_ctx_.IsCUDA()) {
      ghist_ = std::make_shared<GHistIndexMatrix>(&fmat_ctx_, Info(), *ellpack_, param);
    } else {
      // Can happen when QDM is initialized on GPU, but a CPU version is queried by a different QDM
      // for cut reference.
      auto cpu_ctx = ctx->MakeCPU();
      ghist_ = std::make_shared<GHistIndexMatrix>(&cpu_ctx, Info(), *ellpack_, param);
    }
  }

  if (!std::isnan(param.sparse_thresh) &&
      param.sparse_thresh != tree::TrainParam::DftSparseThreshold()) {
    LOG(WARNING) << "`sparse_threshold` can not be changed when `QuantileDMatrix` is used instead "
                    "of `DMatrix`.";
  }

  auto page = ghist_;
#if defined(XGBOOST_USE_METAL)
  if (ctx && ctx->IsMPS()) page = metal_ghist_;
#endif
  auto begin_iter = BatchIterator<GHistIndexMatrix>(
      new SimpleBatchIteratorImpl<GHistIndexMatrix>(std::move(page)));
  return BatchSet<GHistIndexMatrix>(begin_iter);
}

BatchSet<ExtSparsePage> IterativeDMatrix::GetExtBatches(Context const* ctx,
                                                        BatchParam const& param) {
  // Sparse export is a host operation even if the matrix was constructed on MPS.
  auto cpu_ctx = ctx ? ctx->MakeCPU() : fmat_ctx_.MakeCPU();
  auto access_ctx = ctx && !ctx->IsMPS() ? ctx : &cpu_ctx;
  for (auto const& page : this->GetGradientIndex(access_ctx, param)) {
    auto p_out = std::make_shared<SparsePage>();
    p_out->data.Resize(this->Info().num_nonzero_);
    p_out->offset.Resize(this->Info().num_row_ + 1);

    auto& h_offset = p_out->offset.HostVector();
    CHECK_EQ(page.row_ptr.size(), h_offset.size());
    std::copy(page.row_ptr.cbegin(), page.row_ptr.cend(), h_offset.begin());

    auto& h_data = p_out->data.HostVector();
    auto const& vals = page.cut.Values();
    auto const& ptrs = page.cut.Ptrs();
    auto ft = Info().feature_types.ConstHostSpan();

    AssignColumnBinIndex(page, [&](auto bin_idx, std::size_t idx, std::size_t, bst_feature_t fidx) {
      float v;
      if (common::IsCat(ft, fidx)) {
        v = vals[bin_idx];
      } else {
        v = common::HistogramCuts::NumericBinValue(ptrs, vals, fidx, bin_idx);
      }
      h_data[idx] = Entry{fidx, v};
    });

    auto p_ext_out = std::make_shared<ExtSparsePage>(p_out);
    auto begin_iter =
        BatchIterator<ExtSparsePage>(new SimpleBatchIteratorImpl<ExtSparsePage>(p_ext_out));
    return BatchSet<ExtSparsePage>(begin_iter);
  }
  error::Unreachable();
  auto begin_iter =
      BatchIterator<ExtSparsePage>(new SimpleBatchIteratorImpl<ExtSparsePage>(nullptr));
  return BatchSet<ExtSparsePage>(begin_iter);
}

#if !defined(XGBOOST_USE_CUDA)
void IterativeDMatrix::InitFromCUDA(Context const*, BatchParam const&,
                                    DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>&&,
                                    float, std::shared_ptr<DMatrix>) {
  // silent the warning about unused variables.
  (void)(proxy_);
  common::AssertGPUSupport();
}

BatchSet<EllpackPage> IterativeDMatrix::GetEllpackBatches(Context const*, BatchParam const&) {
  common::AssertGPUSupport();
  auto begin_iter = BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_));
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}

void IterativeDMatrix::Save(common::AlignedFileWriteStream* output) const {
#if defined(XGBOOST_USE_METAL)
  std::lock_guard<std::mutex> lock{metal_matrix_mutex_};
  CHECK(output);
  CHECK(metal_ghist_) << "Internal ELLPACK save requires a compressed Metal page.";
  metal_ghist_->cut.Save(output);
  common::metal::WriteCompressedPage(*metal_ghist_, output);
#else
  LOG(FATAL) << "Not implemented";
#endif
}

IterativeDMatrix* IterativeDMatrix::Load(Context const* ctx,
                                         common::AlignedResourceReadStream* input) {
#if defined(XGBOOST_USE_METAL)
  CHECK(ctx && ctx->IsMPS());
  CHECK(input);
  std::unique_ptr<common::HistogramCuts> cuts{common::HistogramCuts::Load(input)};
  auto page = std::make_shared<GHistIndexMatrix>();
  CHECK(common::metal::ReadCompressedPage(input, *cuts, 0, true, page.get()));
  // Match CUDA's internal page-only contract: labels, MetaInfo and training
  // parameters are deliberately not serialized by this format.
  return new IterativeDMatrix{ctx, std::move(page)};
#else
  LOG(FATAL) << "Not implemented";
  return nullptr;
#endif
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
