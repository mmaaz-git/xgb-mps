/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#include "gradient_index_page_source.h"

#include <memory>   // for make_shared
#include <utility>  // for move

#include "../common/hist_util.h"  // for HistogramCuts
#include "gradient_index.h"       // for GHistIndexMatrix
#include "proxy_dmatrix.h"        // for DispatchAny

#if defined(XGBOOST_USE_METAL)
#include "metal_quantile.h"
#endif

namespace xgboost::data {
void GradientIndexPageSource::Fetch() {
  if (!this->ReadCache()) {
    auto ctx = Context{};
    ctx.Init(Args{{"nthread", std::to_string(nthreads_)}, {"device", use_metal_ ? "mps" : "cpu"}});
    // source is initialized to be the 0th page during construction, so when count_ is 0
    // there's no need to increment the source.
    if (this->count_ != 0 && !this->sync_) {
      // The mixin doesn't sync the source if `sync_` is false, we need to sync it
      // ourselves.
      ++(*source_);
    }
    // This is not read from cache so we still need it to be synced with sparse page source.
    CHECK_EQ(this->count_, this->source_->Iter());
    auto const& csr = this->source_->Page();
    CHECK_NE(this->cuts_.Values().size(), 0);
    this->page_.reset(new GHistIndexMatrix{&ctx, *csr, feature_types_, cuts_, max_bin_per_feat_,
                                           is_dense_, sparse_thresh_});
    this->WriteCache();
  }
}

void ExtGradientIndexPageSource::Fetch() {
  if (!this->ReadCache()) {
    CHECK_EQ(count_, source_->Iter());
    CHECK_NE(cuts_.Values().size(), 0);
#if defined(XGBOOST_USE_METAL)
    if (ctx_->IsMPS()) {
      CHECK_LT(count_, batch_nnz_.size())
          << "Different number of batches returned between iterations";
      page_.reset();
      page_ = metal_impl::MakeIndexBatch(ctx_, proxy_, missing_, p_, cuts_, *info_,
                                         base_rows_.at(count_ + 1) - base_rows_.at(count_),
                                         base_rows_[count_], batch_nnz_[count_]);
      info_->Extend(proxy_->Info(), false, true);
      this->WriteCache();
      return;
    }
#endif
    cpu_impl::DispatchAny(proxy_, [this](auto const& value) {
      CHECK(this->proxy_->Ctx()->IsCPU()) << "All batches must use the same device type.";
      auto h_feature_types = proxy_->Info().feature_types.ConstHostSpan();
      // This does three things:
      // - Generate CSR matrix for gradient index.
      // - Generate the column matrix for gradient index.
      // - Concatenate the meta info.
      common::HistogramCuts cuts{this->cuts_};
      CHECK_EQ(this->cuts_.MaxCategory(), cuts.MaxCategory());
      if (this->cuts_.HasCategorical()) {
        CHECK(!h_feature_types.empty());
      }
      this->page_.reset();
      // The external iterator has the data when the `next` method is called. Therefore,
      // it's one step ahead of this source.

      // FIXME(jiamingy): For now, we use the `info->IsDense()` to represent all batches
      // similar to the sparse DMatrix source. We should use per-batch property with proxy
      // DMatrix info instead. This requires more fine-grained tests.
      this->page_ =
          std::make_shared<GHistIndexMatrix>(value.NumRows(), this->base_rows_.at(source_->Iter()),
                                             std::move(cuts), this->p_.max_bin, info_->IsDense());
      bst_idx_t prev_sum = 0;
      bst_idx_t rbegin = 0;
      // Use `value.NumRows()` for the size of a single batch. Unlike the
      // `IterativeDMatrix`, external memory doesn't concatenate the pages.
      this->page_->PushAdapterBatch(ctx_, rbegin, prev_sum, value, this->missing_, h_feature_types,
                                    this->p_.sparse_thresh, value.NumRows());
      this->page_->PushAdapterBatchColumns(ctx_, value, this->missing_, rbegin);
      this->info_->Extend(proxy_->Info(), false, false);
    });
    this->WriteCache();
  }
}
void ExtGradientIndexPageSource::EndIter() {
  if (ctx_->IsMPS() && !cache_info_->written) {
    CHECK_EQ(count_, batch_nnz_.size())
        << "Different number of batches returned between iterations";
  }
  ExtQantileSourceMixin::EndIter();
}
}  // namespace xgboost::data
