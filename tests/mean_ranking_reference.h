// Transitional integration oracle: CUDA-source mean pairs + the shared host
// LambdaGrad formula, with the existing unrounded accumulation/bias updates.
// This intentionally does NOT claim CUDA gradient parity: CUDA rounding factors,
// MAP actual-label prefixes and weight multiplication are separate open work.
// Exact sampled pairs and the CCCL distribution have independent tests.
#pragma once
#include <cmath>
#include <memory>
#include <string>

#include "../xgboost/src/objective/lambdarank_obj.h"
#include "cuda_ranking_pairs.h"
#include "xgboost/json.h"

namespace cuda_reference {
class MeanRankingReference {
  xgboost::Context ctx_;
  xgboost::ltr::LambdaRankParam param_;
  std::unique_ptr<xgboost::ltr::RankingCache> cache_;
  std::string objective_;
  std::vector<double> positive_, negative_;

 public:
  MeanRankingReference(std::string objective, xgboost::Args const& args,
                       xgboost::MetaInfo const& info)
      : objective_{std::move(objective)} {
    ctx_.Init({{"device", "cpu"}, {"nthread", "2"}});
    param_.UpdateAllowUnknown(args);
    if (objective_ == "rank:ndcg")
      cache_ = std::make_unique<xgboost::ltr::NDCGCache>(&ctx_, info, param_);
    else
      cache_ = std::make_unique<xgboost::ltr::RankingCache>(&ctx_, info, param_);
    if (param_.lambdarank_unbiased) {
      positive_.assign(cache_->MaxPositionSize(), 1.);
      negative_ = positive_;
    }
  }
  void Gradient(xgboost::HostDeviceVector<float> const& predictions, xgboost::MetaInfo const& info,
                std::uint32_t seed, xgboost::linalg::Matrix<xgboost::GradientPair>* output) {
    using namespace xgboost;
    output->Reshape(info.num_row_, 1);
    auto out = output->HostView().Values();
    std::fill(out.begin(), out.end(), GradientPair{});
    auto groups = cache_->DataGroupPtr(&ctx_);
    auto tp = linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<double const>{positive_},
                                     positive_.size());
    auto tm = linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<double const>{negative_},
                                     negative_.size());
    std::vector<double> li(positive_.size()), lj(negative_.size());
    for (std::size_t g = 0; g + 1 < groups.size(); ++g) {
      auto begin = groups[g], n = groups[g + 1] - begin;
      auto pred = predictions.ConstHostSpan().subspan(begin, n);
      auto label = info.labels.HostView().Slice(linalg::Range(begin, begin + n), 0);
      std::vector<float> labels(n);
      std::vector<std::size_t> rank(n);
      std::iota(rank.begin(), rank.end(), 0);
      std::stable_sort(rank.begin(), rank.end(), [&](auto a, auto b) { return pred[a] > pred[b]; });
      for (std::size_t i = 0; i < n; ++i) labels[i] = label(i);
      std::vector<double> nrel(n), acc(n), gli(n), glj(n);
      double hits = 0, precision = 0;
      for (std::size_t i = 0; i < n; ++i) {
        // Existing CPU-style MAP statistics; actual-label CUDA scan is pending.
        if (labels[rank[i]] == 1.f) {
          ++hits;
          precision += 1. / double(i + 1);
        }
        nrel[i] = hits;
        acc[i] = precision;
      }
      auto delta = [&](float yh, float yl, std::size_t rh, std::size_t rl) {
        if (objective_ == "rank:ndcg") {
          auto* ndcg = static_cast<ltr::NDCGCache*>(cache_.get());
          auto inverse = ndcg->InvIDCG(&ctx_)(g);
          auto discount = ndcg->Discount(&ctx_);
          return param_.ndcg_exp_gain ? obj::DeltaNDCG<true>(yh, yl, rh, rl, inverse, discount)
                                      : obj::DeltaNDCG<false>(yh, yl, rh, rl, inverse, discount);
        }
        if (objective_ == "rank:map") {
          if (rh > rl) {
            std::swap(rh, rl);
            std::swap(yh, yl);
          }
          return obj::DeltaMAP(yh, yl, rh, rl, nrel, acc);
        }
        return 1.;
      };
      for (auto pair :
           MeanPairs(labels, rank, seed + static_cast<std::uint32_t>(g), param_.NumPair())) {
        std::size_t high = pair[0], low = pair[1];
        if (labels[rank[high]] == labels[rank[low]]) continue;
        if (labels[rank[high]] < labels[rank[low]]) std::swap(high, low);
        double cost = 0;
        auto calc = [&](auto unbiased, auto normalized) {
          return obj::LambdaGrad<decltype(unbiased)::value, decltype(normalized)::value>(
              label, pred, rank, high, low, delta, tp, tm, &cost);
        };
        auto pg = param_.lambdarank_unbiased ? (param_.lambdarank_score_normalization
                                                    ? calc(std::true_type{}, std::true_type{})
                                                    : calc(std::true_type{}, std::false_type{}))
                                             : (param_.lambdarank_score_normalization
                                                    ? calc(std::false_type{}, std::true_type{})
                                                    : calc(std::false_type{}, std::false_type{}));
        auto hi = rank[high], lo = rank[low];
        out[begin + hi] += pg;
        out[begin + lo] += obj::Repulse(pg);
        if (hi < positive_.size() && lo < positive_.size()) {
          if (negative_[lo] >= obj::Eps64()) gli[hi] += cost / negative_[lo];
          if (positive_[hi] >= obj::Eps64()) glj[lo] += cost / positive_[hi];
        }
      }
      for (std::size_t i = 0; i < std::min<std::size_t>(n, positive_.size()); ++i) {
        li[i] += gli[i];
        lj[i] += glj[i];
      }
      float norm = param_.lambdarank_normalization ? 1. / double(param_.NumPair()) : 1.;
      float weight = info.weights_.Empty() ? 1.f : info.weights_.ConstHostVector()[g];
      for (std::size_t i = begin; i < begin + n; ++i)
        out[i] = ((out[i] * norm) * weight) * cache_->WeightNorm();
    }
    for (std::size_t i = 0; i < positive_.size(); ++i) {
      if (li[0] >= obj::Eps64()) positive_[i] = std::pow(li[i] / li[0], param_.Regularizer());
      if (lj[0] >= obj::Eps64()) negative_[i] = std::pow(lj[i] / lj[0], param_.Regularizer());
    }
  }
  void SaveBias(xgboost::Json* config) const {
    using namespace xgboost;
    (*config)["ti+"] = F32Array(positive_.size());
    (*config)["tj-"] = F32Array(negative_.size());
    for (std::size_t i = 0; i < positive_.size(); ++i) {
      get<F32Array>((*config)["ti+"])[i] = positive_[i];
      get<F32Array>((*config)["tj-"])[i] = negative_[i];
    }
  }
};
}  // namespace cuda_reference
