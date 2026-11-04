/**
 * Copyright 2024-2026, XGBoost Contributors
 */
#include "extmem_quantile_dmatrix.h"

#include <memory>  // for shared_ptr
#include <string>  // for string
#include <vector>  // for vector

#include "../common/error_msg.h"    // for CacheHostRatio, InconsistentMaxBin
#include "../tree/param.h"          // FIXME(jiamingy): Find a better way to share this parameter.
#include "batch_utils.h"            // for CheckParam, RegenGHist
#include "proxy_dmatrix.h"          // for DataIterProxy
#include "quantile_dmatrix.h"       // for GetDataShape, MakeSketches
#include "simple_batch_iterator.h"  // for SimpleBatchIteratorImpl
#include "sparse_page_source.h"     // for MakeCachePrefix

#if defined(XGBOOST_USE_METAL)
#include "metal_page_source.h"
#include "metal_quantile.h"
#endif

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#endif

namespace xgboost::data {
ExtMemQuantileDMatrix::ExtMemQuantileDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                                             std::shared_ptr<DMatrix> ref,
                                             DataIterResetCallback *reset,
                                             XGDMatrixCallbackNext *next, bst_bin_t max_bin,
                                             ExtMemConfig const &config)
    : cache_prefix_{config.cache}, on_host_{config.on_host} {
  cache_prefix_ = MakeCachePrefix(cache_prefix_);
  auto iter = std::make_shared<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>>(
      iter_handle, reset, next);
  iter->Reset();
  // Fetch the first iter
  bool valid = iter->Next();
  CHECK(valid) << "Qauntile DMatrix must have at least 1 batch.";

  auto pctx = MakeProxy(proxy)->Ctx();
  {
    Context ctx;
    ctx.Init(Args{{"nthread", std::to_string(config.n_threads)},
                  {"device", config.device.empty() ? pctx->DeviceName() : config.device}});
    CHECK(ctx.IsMPS() ? !pctx->IsCUDA() : ctx.Device() == pctx->Device())
        << "Construction device must match the input device, except MPS accepts host arrays.";
    this->fmat_ctx_ = ctx;
  }

  BatchParam p{max_bin, tree::TrainParam::DftSparseThreshold()};
  if (fmat_ctx_.IsMPS()) {
    if (!on_host_)
      CHECK(detail::HostRatioIsAuto(config.cache_host_ratio)) << error::CacheHostRatioNotImpl();
    try {
      this->InitFromMPS(&fmat_ctx_, iter, proxy, p, config, ref);
    } catch (...) {
      // The class destructor is not called when construction fails. Release any
      // active readers before removing the partially written page cache.
      ghist_index_source_.reset();
      metal_page_source_.reset();
      metal_disk_source_.reset();
      DeleteCacheFiles(cache_info_);
      throw;
    }
  } else if (fmat_ctx_.IsCPU()) {
    CHECK(detail::HostRatioIsAuto(config.cache_host_ratio)) << error::CacheHostRatioNotImpl();
    this->InitFromCPU(&fmat_ctx_, iter, proxy, p, config.missing, ref);
  } else {
    p.n_prefetch_batches = ::xgboost::cuda_impl::DftPrefetchBatches();
    this->InitFromCUDA(&fmat_ctx_, iter, proxy, p, ref, config);
  }
  this->batch_ = p;

  SyncCategories(&fmat_ctx_, info_.Cats(), info_.num_row_ == 0);
}

ExtMemQuantileDMatrix::~ExtMemQuantileDMatrix() {
  // Clear out all resources before deleting the cache file.
  ghist_index_source_.reset();
  metal_page_source_.reset();
  metal_disk_source_.reset();
  std::visit([](auto &&ptr) { ptr.reset(); }, ellpack_page_source_);

  DeleteCacheFiles(cache_info_);
}

BatchSet<ExtSparsePage> ExtMemQuantileDMatrix::GetExtBatches(Context const *, BatchParam const &) {
  LOG(FATAL) << "Not implemented for `ExtMemQuantileDMatrix`.";
  auto begin_iter =
      BatchIterator<ExtSparsePage>(new SimpleBatchIteratorImpl<ExtSparsePage>(nullptr));
  return BatchSet<ExtSparsePage>{begin_iter};
}

void ExtMemQuantileDMatrix::InitFromMPS(
    Context const* ctx,
    std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
    DMatrixHandle proxy_handle, BatchParam const& p, ExtMemConfig const& config,
    std::shared_ptr<DMatrix> ref) {
#if defined(XGBOOST_USE_METAL)
  auto missing = config.missing;
  auto proxy = MakeProxy(proxy_handle);
  ExternalDataInfo ext;
  common::HistogramCuts cuts{0};
  std::vector<FeatureType> types;
  metal_impl::MakeSketches(ctx, iter.get(), proxy, ref, p, missing, &cuts, &ext, &types);
  ext.SetInfo(ctx, true, &info_);
  info_.feature_types.HostVector() = types;
  n_batches_ = ext.n_batches;
  batch_ptr_ = ext.base_rowids;

  if (on_host_) {
    auto id = MakeCache(this, ".metal-ellpack.page", true, cache_prefix_, &cache_info_);
    metal_page_source_ = std::make_shared<MetalPageSource>(
        ctx, iter, proxy, p, config, cache_info_.at(id), cuts, ext, &info_, ref != nullptr);
    batch_ptr_ = metal_page_source_->BatchPtr();
    n_batches_ = batch_ptr_.size() - 1;
  } else {
    // CUDA's disk path retains one compressed page per input batch.
    auto id = MakeCache(this, ".metal-ellpack.page", false, cache_prefix_, &cache_info_);
    metal_disk_source_ =
        std::make_shared<MetalDiskPageSource>(ctx, missing, p, cache_info_.at(id), cuts);
    auto layout = common::metal::MakeEllpackInfo(cuts, ext.row_stride, info_.IsDense());
    bst_idx_t batch = 0;
    iter->Reset();
    while (iter->Next()) {
      CHECK_LT(batch, ext.n_batches) << "Different number of batches returned between iterations";
      auto rows = ext.base_rowids[batch + 1] - ext.base_rowids[batch];
      CHECK_EQ(BatchSamples(proxy), rows);
      CHECK_EQ(BatchColumns(proxy), ext.n_features);
      auto compressed = common::metal::MakeCompressedPage(layout, rows, ext.base_rowids[batch]);
      compressed->BeginAdapter(cuts, types);
      bst_idx_t nnz = 0;
      metal_impl::AdapterWindows(ctx, proxy, missing, [&](auto const &input) {
        compressed->PushAdapter(input);
        nnz += input.NonMissing();
      });
      CHECK_EQ(nnz, ext.batch_nnz[batch]) << "Non-missing counts changed between iterations";
      compressed->EndAdapter();
      proxy->Info().num_row_ = rows;
      proxy->Info().num_col_ = ext.n_features;
      proxy->Info().num_nonzero_ = nnz;
      info_.Extend(proxy->Info(), false, true);
      metal_disk_source_->Append(*compressed);
      ++batch;
    }
    CHECK_EQ(batch, ext.n_batches) << "Different number of batches returned between iterations";
    iter->Reset();
    metal_disk_source_->Finalize();
  }
  bst_idx_t batches = 0, rows = 0;
  bst_idx_t nnz =
      metal_page_source_ ? metal_page_source_->NonMissing() : metal_disk_source_->NonMissing();
  for (auto const& page : GetGradientIndexImpl()) {
    CHECK_LT(batches, n_batches_);
    CHECK_EQ(page.base_rowid, batch_ptr_[batches]);
    CHECK_EQ(page.Size(), batch_ptr_[batches + 1] - batch_ptr_[batches]);
    CHECK_EQ(page.Features(), ext.n_features);
    rows += page.Size();
    ++batches;
  }
  CHECK_EQ(batches, n_batches_);
  CHECK_EQ(rows, ext.accumulated_rows);
  CHECK_EQ(nnz, ext.nnz);
  CHECK_EQ(cuts.HasCategorical(), info_.HasCategorical());
#else
  LOG(FATAL) << "Metal ExtMemQuantileDMatrix requires a build with USE_METAL=ON.";
#endif
}

void ExtMemQuantileDMatrix::InitFromCPU(
    Context const *ctx,
    std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
    DMatrixHandle proxy_handle, BatchParam const &p, float missing, std::shared_ptr<DMatrix> ref) {
  xgboost_NVTX_FN_RANGE();

  auto proxy = MakeProxy(proxy_handle);
  CHECK(proxy);

  common::HistogramCuts cuts{0};
  ExternalDataInfo ext_info;
  cpu_impl::GetDataShape(ctx, proxy, iter.get(), missing, &ext_info);
  ext_info.SetInfo(ctx, true, &this->info_);

  this->n_batches_ = ext_info.n_batches;
  this->batch_ptr_ = ext_info.base_rowids;

  /**
   * Generate quantiles
   */
  std::vector<FeatureType> h_ft;
  cpu_impl::MakeSketches(ctx, iter.get(), proxy, ref, missing, &cuts, p, this->info_, ext_info,
                         &h_ft);

  /**
   * Generate gradient index
   */
  auto id = MakeCache(this, ".gradient_index.page", false, cache_prefix_, &cache_info_);
  this->ghist_index_source_ = std::make_unique<ExtGradientIndexPageSource>(
      ctx, missing, &this->info_, cache_info_.at(id), p, cuts, iter, proxy, ext_info.base_rowids);

  /**
   * Force initialize the cache and do some sanity checks along the way
   */
  bst_idx_t batch_cnt = 0, k = 0;
  bst_idx_t n_total_samples = 0;
  for (auto const &page : this->GetGradientIndexImpl()) {
    n_total_samples += page.Size();
    CHECK_EQ(page.base_rowid, ext_info.base_rowids[k]);
    CHECK_EQ(page.Features(), this->info_.num_col_);
    ++k, ++batch_cnt;
  }
  CHECK_EQ(batch_cnt, ext_info.n_batches);
  CHECK_EQ(n_total_samples, ext_info.accumulated_rows);
  if (cuts.HasCategorical()) {
    CHECK(!this->info_.feature_types.Empty());
  }
  CHECK_EQ(cuts.HasCategorical(), this->info_.HasCategorical());
}

[[nodiscard]] BatchSet<GHistIndexMatrix> ExtMemQuantileDMatrix::GetGradientIndexImpl() {
#if defined(XGBOOST_USE_METAL)
  if (metal_page_source_) return BatchSet{BatchIterator<GHistIndexMatrix>{metal_page_source_}};
  if (metal_disk_source_) return BatchSet{BatchIterator<GHistIndexMatrix>{metal_disk_source_}};
#endif
  return BatchSet{BatchIterator<GHistIndexMatrix>{this->ghist_index_source_}};
}

BatchSet<GHistIndexMatrix> ExtMemQuantileDMatrix::GetGradientIndex(Context const *ctx,
                                                                   BatchParam const &param) {
  if (param.Initialized()) {
    detail::CheckParam(this->batch_, param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }

  CHECK(this->GHistIndexExists())
      << "The `ExtMemQuantileDMatrix` is initialized using GPU data, cannot be used for CPU.";
#if defined(XGBOOST_USE_METAL)
  if (metal_page_source_)
    metal_page_source_->Reset(ctx, param);
  else if (metal_disk_source_)
    metal_disk_source_->Reset(ctx, param);
  else
#endif
    this->ghist_index_source_->Reset(param);

  if (!std::isnan(param.sparse_thresh) &&
      param.sparse_thresh != tree::TrainParam::DftSparseThreshold()) {
    LOG(WARNING) << "`sparse_threshold` can not be changed when `QuantileDMatrix` is used instead "
                    "of `DMatrix`.";
  }

  return this->GetGradientIndexImpl();
}

#if !defined(XGBOOST_USE_CUDA)
void ExtMemQuantileDMatrix::InitFromCUDA(
    Context const *, std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>>,
    DMatrixHandle, BatchParam const &, std::shared_ptr<DMatrix>, ExtMemConfig const &) {
  common::AssertGPUSupport();
}

BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackBatches(Context const *,
                                                               const BatchParam &) {
  common::AssertGPUSupport();
  auto batch_set = std::visit([](auto &&ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                              this->ellpack_page_source_);
  return batch_set;
}

BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackPageImpl() {
  common::AssertGPUSupport();
  auto batch_set = std::visit([](auto &&ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                              this->ellpack_page_source_);
  return batch_set;
}
#endif
}  // namespace xgboost::data
