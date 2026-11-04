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
    need_host_ = !param.for_cuts && (!ctx || !ctx->IsMPS());
    auto p = param;
    p.n_prefetch_batches = 2;  // CUDA's one active + one prefetched page.
    this->ResetCache(p);
  }
  std::vector<bst_idx_t> const& BatchPtr() const { return batch_ptr_; }
  bst_idx_t NonMissing() const { return nnz_; }
};
// CUDA raw ELLPACK fields in the existing aligned disk stream. Prefetch workers
// acquire only compressed views; CPU conversion stays on the iterator thread.
class MetalDiskFormat : public SparsePageFormat<GHistIndexMatrix> {
  common::HistogramCuts const& cuts_;
  bst_bin_t max_bin_;
  bool prefetch_copy_;

 public:
  MetalDiskFormat(common::HistogramCuts const& cuts, bst_bin_t max_bin, bool prefetch_copy)
      : cuts_{cuts}, max_bin_{max_bin}, prefetch_copy_{prefetch_copy} {}
  bool Read(GHistIndexMatrix* page, common::AlignedResourceReadStream* input) override {
    return common::metal::ReadCompressedPage(input, cuts_, max_bin_, prefetch_copy_, page);
  }
  std::size_t Write(GHistIndexMatrix const& page, common::AlignedFileWriteStream* output) override {
    return common::metal::WriteCompressedPage(page, output);
  }
};
template <class S>
class MetalDiskFormatPolicy {
 protected:
  common::HistogramCuts cuts_{0};
  bst_bin_t max_bin_{0};

 public:
  using FormatT = MetalDiskFormat;
  auto CreatePageFormat(BatchParam const& p) const {
    return std::make_unique<FormatT>(cuts_, max_bin_, p.prefetch_copy);
  }
  static void DestroyPage(std::shared_ptr<S>* page) { page->reset(); }
};
class MetalDiskPageSource
    : public SparsePageSourceImpl<
          GHistIndexMatrix, DefaultFormatStreamPolicy<GHistIndexMatrix, MetalDiskFormatPolicy>> {
  using Super =
      SparsePageSourceImpl<GHistIndexMatrix,
                           DefaultFormatStreamPolicy<GHistIndexMatrix, MetalDiskFormatPolicy>>;
  Context cpu_ctx_;
  double sparse_thresh_;
  bool need_host_{false};
  std::vector<bst_idx_t> batch_ptr_{0};
  bst_idx_t nnz_{0};
  void Fetch() final {
    CHECK(this->ReadCache());
    if (need_host_) common::metal::MaterializeHost(&cpu_ctx_, sparse_thresh_, page_.get());
  }

 public:
  MetalDiskPageSource(Context const* ctx, float missing, BatchParam p, std::shared_ptr<Cache> cache,
                      common::HistogramCuts cuts)
      : Super{missing, ctx->Threads(), cuts.NumFeatures(), std::move(cache)},
        cpu_ctx_{ctx->MakeCPU()},
        sparse_thresh_{p.sparse_thresh} {
    cuts_ = std::move(cuts);
    max_bin_ = p.max_bin;
    p.n_prefetch_batches = 2;
    this->param_ = p;
  }
  void Append(common::metal::CompressedPage const& input) {
    CHECK(!cache_info_->written);
    CHECK_EQ(input.BaseRow(), batch_ptr_.back());
    page_ = std::make_shared<GHistIndexMatrix>();
    input.View(cuts_, max_bin_, false, page_.get());
    nnz_ += input.NonMissing();
    this->WriteCache();
    batch_ptr_.push_back(batch_ptr_.back() + input.Rows());
    ++count_;
    page_.reset();
  }
  void Finalize() {
    CHECK_GT(count_, 0);
    this->EndIter();
    // Construction has released its raw callbacks before later cache traversals.
    this->ResetCache(param_);
  }
  MetalDiskPageSource& operator++() final {
    TryLockGuard guard{single_threaded_};
    ++count_;
    at_end_ = count_ == cache_info_->Size();
    if (at_end_)
      this->EndIter();
    else
      this->Fetch();
    return *this;
  }
  void Reset(BatchParam const& p) final { this->Reset(nullptr, p); }
  void Reset(Context const* ctx, BatchParam const& param) {
    TryLockGuard guard{single_threaded_};
    need_host_ = !param.for_cuts && (!ctx || !ctx->IsMPS());
    auto p = param;
    p.n_prefetch_batches = 2;
    this->ResetCache(p);
  }
  auto const& BatchPtr() const { return batch_ptr_; }
  bst_idx_t NonMissing() const { return nnz_; }
};
}  // namespace xgboost::data
