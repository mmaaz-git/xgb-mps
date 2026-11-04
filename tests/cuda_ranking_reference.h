// Host CUDA-formula ranking oracle. No NVIDIA execution. Pair construction is
// independent of the shader. DeltaNDCG/MAP use upstream shared functions; sigmoid
// rounds a binary64 exp to FP32 explicitly, as tested by metal_ranking_math.mm.
// CUB scan association and CUDA libdevice rounding are not bitwise claims.
// The single global unbiased factor uses max(query factors), an explicit
// deterministic resolution of CUDA's concurrent writes to d_cost_rounding[0].
#pragma once
#include <cmath>
#include <memory>
#include <string>

#include "../xgboost/src/objective/lambdarank_obj.h"
#include "cuda_ranking_pairs.h"
#include "cuda_ranking_rounding.h"
#include "xgboost/json.h"
namespace cuda_reference {
class RankingReference {
  xgboost::Context ctx_;
  xgboost::ltr::LambdaRankParam param_;
  std::unique_ptr<xgboost::ltr::RankingCache> cache_;
  std::string objective_;
  std::vector<double> positive_, negative_;

 public:
  struct Statistics {
    float gr, hr, norm, weight;
    double cost_rounding, max_cost, tracked_cost;
  };
  std::vector<Statistics> statistics;
  double cost_rounding{0};
  std::vector<double> const& Positive() const { return positive_; }
  std::vector<double> const& Negative() const { return negative_; }
  void SetBias(std::vector<double> positive, std::vector<double> negative) {
    positive_ = std::move(positive);
    negative_ = std::move(negative);
  }
  RankingReference(std::string objective, xgboost::Args const& args, xgboost::MetaInfo const& info)
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
    struct Pair {
      float gradient, hessian;
      double cost;
      std::uint32_t high, low;
    };
    struct Group {
      std::vector<Pair> pairs;
      float gr, hr, norm, weight;
    };
    output->Reshape(info.num_row_, 1);
    auto out = output->HostView().Values();
    std::fill(out.begin(), out.end(), GradientPair{});
    auto groups = cache_->DataGroupPtr(&ctx_);
    std::vector<Group> state(groups.size() - 1);
    statistics.resize(state.size());
    double minimum = std::numeric_limits<double>::infinity();
    for (auto v : positive_) minimum = std::min(minimum, std::abs(v));
    for (auto v : negative_) minimum = std::min(minimum, std::abs(v));
    double global_rounding = 0;
    for (std::size_t g = 0; g < state.size(); ++g) {
      auto& query = state[g];
      auto begin = groups[g], n = groups[g + 1] - begin;
      auto pred = predictions.ConstHostSpan().subspan(begin, n);
      auto label = info.labels.HostView().Slice(linalg::Range(begin, begin + n), 0);
      std::vector<float> labels(n);
      std::vector<std::size_t> rank(n);
      std::iota(rank.begin(), rank.end(), 0);
      std::stable_sort(rank.begin(), rank.end(), [&](auto a, auto b) { return pred[a] > pred[b]; });
      for (std::size_t i = 0; i < n; ++i) labels[i] = label(i);
      std::vector<double> nrel(n), acc(n);
      double hits = 0, precision = 0;
      for (std::size_t i = 0; i < n; ++i) {
        hits += double(labels[rank[i]]);
        precision += double(labels[rank[i]]) / double(i + 1);
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
      auto pairs = param_.IsMean() ? MeanPairs(labels, rank, seed + static_cast<std::uint32_t>(g),
                                               param_.NumPair())
                                   : std::vector<RankingPair>{};
      if (!param_.IsMean())
        for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, param_.NumPair()); ++i)
          for (std::uint32_t j = i + 1; j < n; ++j) pairs.push_back({i, j});
      float max_g = 0, max_h = 0;
      double max_cost = 0, tracked_cost = 0, sum_lambda = 0;
      for (auto pair : pairs) {
        std::size_t high = pair[0], low = pair[1];
        if (labels[rank[high]] == labels[rank[low]]) continue;
        if (labels[rank[high]] < labels[rank[low]]) std::swap(high, low);
        auto hi = rank[high], lo = rank[low];
        float difference = pred[hi] - pred[lo];
        float exponential = static_cast<float>(std::exp(double(std::min(-difference, 88.7f))));
        float denominator = (exponential + 1.f) + 1e-16f;
        double sigmoid = static_cast<float>(1. / double(denominator));
        double weight = std::abs(delta(labels[hi], labels[lo], high, low));
        if (param_.lambdarank_score_normalization && pred[rank.front()] != pred[rank.back()])
          weight /= double(std::abs(difference)) + .01;
        double cost = param_.lambdarank_unbiased ? std::log(1. / (1. - sigmoid)) * weight : 0.;
        double gradient = (sigmoid - 1.) * weight;
        double hessian = (std::max(sigmoid * (1. - sigmoid), obj::Eps64()) * weight) * 2.;
        if (hi < positive_.size() && lo < negative_.size() && positive_[hi] >= obj::Eps64() &&
            negative_[lo] >= obj::Eps64()) {
          gradient /= positive_[hi] * negative_[lo];
          hessian /= positive_[hi] * negative_[lo];
        }
        Pair value{float(gradient), float(hessian), cost, std::uint32_t(hi), std::uint32_t(lo)};
        query.pairs.push_back(value);
        max_g = std::max(max_g, std::abs(value.gradient));
        max_h = std::max(max_h, std::abs(value.hessian));
        max_cost = std::max(max_cost, std::abs(cost));
        if (hi < positive_.size() && lo < negative_.size())
          tracked_cost = std::max(tracked_cost, std::abs(cost));
        sum_lambda += -2. * double(value.gradient);
      }
      query.gr = RankingRounding(float(n) * max_g, n);
      query.hr = RankingRounding(float(n) * max_h, n);
      query.norm = !param_.lambdarank_normalization ? 1.f
                   : param_.IsMean()                ? float(1. / double(param_.NumPair()))
                   : sum_lambda > 0                 ? float(std::log2(1. + sum_lambda) / sum_lambda)
                                                    : 1.f;
      float weight = info.weights_.Empty() ? 1.f : info.weights_.ConstHostVector()[g];
      query.weight = float(double(weight) * cache_->WeightNorm());
      auto rounding = param_.lambdarank_unbiased
                          ? RankingRounding(double(float(n)) * (max_cost / minimum), n)
                          : 0.;
      global_rounding = std::max(global_rounding, rounding);
      statistics[g] = {query.gr, query.hr, query.norm,  query.weight,
                       rounding, max_cost, tracked_cost};
    }
    cost_rounding = global_rounding;
    std::vector<double> li(positive_.size()), lj(negative_.size());
    for (std::size_t g = 0; g < state.size(); ++g) {
      auto const& query = state[g];
      auto begin = groups[g], n = groups[g + 1] - begin;
      std::vector<double> gli(n), glj(n);
      for (auto const& pair : query.pairs) {
        float hessian = RankingTruncate(query.hr, pair.hessian);
        out[begin + pair.high] += GradientPair{RankingTruncate(query.gr, pair.gradient), hessian};
        out[begin + pair.low] += GradientPair{RankingTruncate(query.gr, -pair.gradient), hessian};
        if (pair.high < positive_.size() && pair.low < negative_.size()) {
          if (negative_[pair.low] >= obj::Eps64())
            gli[pair.high] += RankingTruncate(global_rounding, pair.cost / negative_[pair.low]);
          if (positive_[pair.high] >= obj::Eps64())
            glj[pair.low] += RankingTruncate(global_rounding, pair.cost / positive_[pair.high]);
        }
      }
      for (std::size_t i = 0; i < std::min<std::size_t>(n, positive_.size()); ++i) {
        li[i] += gli[i];
        lj[i] += glj[i];
      }
      for (std::size_t i = begin; i < begin + n; ++i) out[i] = (out[i] * query.norm) * query.weight;
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
