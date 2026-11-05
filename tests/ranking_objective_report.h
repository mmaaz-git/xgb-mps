// Public objective observations shared by native Metal exports and CUDA replay.
#pragma once
#include <memory>
#include "ranking_objective_io.h"
#include "../xgboost/src/common/ranking_utils.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/objective.h"

namespace ranking_objective {
inline char const* Objective(unsigned id) {
  Require(id < 3, "Invalid ranking objective");
  char const* names[]{"rank:pairwise", "rank:ndcg", "rank:map"}; return names[id];
}
inline xgboost::Args Arguments(Spec const& s) {
  return {{"lambdarank_pair_method", s.mean ? "mean" : "topk"},
          {"lambdarank_num_pair_per_sample", std::to_string(s.pairs)},
          {"lambdarank_normalization", s.normalized ? "true" : "false"},
          {"lambdarank_score_normalization", s.score ? "true" : "false"},
          {"ndcg_exp_gain", s.exponential ? "true" : "false"},
          {"lambdarank_unbiased", s.unbiased ? "true" : "false"},
          {"lambdarank_bias_norm", std::to_string(s.bias_norm)}};
}
inline Observation Observe(xgboost::Context* ctx, xgboost::ObjFunction const* objective,
                           xgboost::linalg::Matrix<xgboost::GradientPair>* gradients, Spec const& spec) {
  using namespace xgboost;
  Observation out;
  for (auto value : gradients->HostView().Values()) {
    out.gradients.push_back(value.GetGrad()); out.gradients.push_back(value.GetHess());
  }
  if (spec.unbiased) {
    Json config{Object{}}; objective->SaveConfig(&config);
    auto const& positive = get<F32Array>(config["ti+"]);
    auto const& negative = get<F32Array>(config["tj-"]);
    out.positive.assign(positive.begin(), positive.end()); out.negative.assign(negative.begin(), negative.end());
  }
  auto copy = ctx->Rng(); out.next_rng = static_cast<std::uint32_t>(copy());
  return out;
}
class Exporter {
  std::vector<Spec> specs_;
  unsigned index_{0};
  Writer inputs_, observed_;
 public:
  Exporter(std::string const& prefix, unsigned suite)
      : specs_{Specs(suite)}, inputs_{prefix+".inputs.bin",0,suite}, observed_{prefix+".metal.bin",1,suite} {}
  void Add(std::string const& objective, xgboost::Args const& args, xgboost::Context* ctx,
           xgboost::MetaInfo const& info, xgboost::HostDeviceVector<float> const& predictions,
           xgboost::linalg::Matrix<xgboost::GradientPair>* gradients, xgboost::ObjFunction const* obj,
           unsigned sequence, unsigned iteration, bool batched, unsigned mode, std::uint32_t seed) {
    Require(index_ < specs_.size(), "Too many native objective exports");
    xgboost::ltr::LambdaRankParam p; p.UpdateAllowUnknown(args);
    Spec s = specs_[index_];
    Require(objective == Objective(s.objective) && sequence == s.sequence && iteration == s.iteration &&
            info.num_row_ == s.rows && info.labels.Shape(1) == 1 &&
            batched == s.batched && mode == s.mode && seed == s.seed &&
            p.IsMean() == s.mean && p.NumPair() == s.pairs && p.lambdarank_normalization == s.normalized &&
            p.lambdarank_score_normalization == s.score && p.ndcg_exp_gain == s.exponential &&
            p.lambdarank_unbiased == s.unbiased && p.lambdarank_bias_norm == s.bias_norm,
            "Actual objective parameters differ from shared case matrix");
    s.group_ptr = info.group_ptr_;
    Input in{s, predictions.ConstHostVector(), info.labels.Data()->ConstHostVector(), info.weights_.ConstHostVector()};
    auto observation = Observe(ctx, obj, gradients, s);
    inputs_.Add(in); observed_.Add(in, observation); ++index_;
  }
  void Finish() { inputs_.Finish(); observed_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc, char** argv, unsigned suite) {
  Require(argc == 1 || (argc == 3 && std::string(argv[1]) == "--report-prefix"),
          "Usage: native-test [--report-prefix new-prefix]");
  if (argc == 1) return {};
  std::string prefix = argv[2];
  Require(!std::filesystem::exists(prefix+".inputs.bin") && !std::filesystem::exists(prefix+".metal.bin"),
          "Use a new objective report prefix");
  return std::make_unique<Exporter>(prefix, suite);
}
}  // namespace ranking_objective
