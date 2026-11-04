/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <memory>
#include <vector>

#include "../common/metal_ellpack.h"
#include "gradient_index.h"
#include "sparse_page_source.h"

namespace xgboost::data {
struct MetalCacheData {
  Context ctx;
  common::HistogramCuts cuts{0};
  bst_bin_t max_bin;
  double sparse_thresh;
  std::vector<std::shared_ptr<common::metal::CompressedPage>> pages;
  std::vector<bst_idx_t> offsets;
};
struct MetalCacheReader {
  std::shared_ptr<MetalCacheData const> cache;
  std::size_t index;
};
struct MetalCacheFormat {
  bool prefetch_copy;
  bool Read(GHistIndexMatrix* output, MetalCacheReader* reader) const {
    auto& cache = *reader->cache;
    cache.pages.at(reader->index)->View(cache.cuts, cache.max_bin, prefetch_copy, output);
    return true;
  }
};
class MetalCacheStreamPolicy {
 protected:
  std::shared_ptr<MetalCacheData> metal_cache_ = std::make_shared<MetalCacheData>();

 public:
  using FormatT = MetalCacheFormat;
  using ReaderT = MetalCacheReader;
  using WriterT = MetalCacheReader;  // Construction commits pages directly on GPU.
  auto CreatePageFormat(BatchParam const& p) const {
    return std::make_unique<FormatT>(FormatT{p.prefetch_copy});
  }
  std::unique_ptr<ReaderT> CreateReader(StringView, std::uint64_t offset, std::uint64_t) const {
    auto& offsets = metal_cache_->offsets;
    auto it = std::lower_bound(offsets.cbegin(), offsets.cend(), offset);
    CHECK(it != offsets.cend() && *it == offset);
    auto index = static_cast<std::size_t>(it - offsets.cbegin());
    CHECK_LT(index, metal_cache_->pages.size());
    return std::make_unique<ReaderT>(ReaderT{metal_cache_, index});
  }
  static void DestroyPage(std::shared_ptr<GHistIndexMatrix>* page) { page->reset(); }
};
class MetalPageSource : public SparsePageSourceImpl<GHistIndexMatrix, MetalCacheStreamPolicy> {
  using Super = SparsePageSourceImpl<GHistIndexMatrix, MetalCacheStreamPolicy>;
  std::vector<bst_idx_t> batch_ptr_{0};
  bst_idx_t nnz_{0};
  // Only the iterator thread reads this flag. Prefetch workers always create
  // device-only views, including futures retained across CPU/MPS resets.
  bool need_host_{false};
  void Fetch() final {
    CHECK(this->ReadCache());
    if (need_host_)
      common::metal::MaterializeHost(&metal_cache_->ctx, metal_cache_->sparse_thresh, page_.get());
  }

 public:
  MetalPageSource(Context const* ctx,
                  std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
                  DMatrixProxy* proxy, BatchParam p, ExtMemConfig const& config,
                  std::shared_ptr<Cache> cache, common::HistogramCuts cuts,
                  ExternalDataInfo const& ext, MetaInfo* info, bool validation);
  MetalPageSource& operator++() final {
    TryLockGuard guard{this->single_threaded_};
    ++count_;
    at_end_ = count_ == cache_info_->Size();
    if (at_end_)
      this->EndIter();
    else
      this->Fetch();
    return *this;
  }
  void Reset(BatchParam const& param) final { this->Reset(nullptr, param); }
  void Reset(Context const* ctx, BatchParam const& param) {
    TryLockGuard guard{this->single_threaded_};
    need_host_ = !ctx || !ctx->IsMPS();
    auto p = param;
    p.n_prefetch_batches = 2;  // CUDA's one active + one prefetched page.
    this->ResetCache(p);
  }
  std::vector<bst_idx_t> const& BatchPtr() const { return batch_ptr_; }
  bst_idx_t NonMissing() const { return nnz_; }
};
}  // namespace xgboost::data
