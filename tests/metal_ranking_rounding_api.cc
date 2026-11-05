// Production bounds/cost factor/MAP prefixes, independently checked before
// checking gradients. In particular untracked positions must enter cost bounds.
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

#include "../xgboost/src/common/metal_ranking.h"
#include "cuda_ranking_reference.h"
#include "ranking_rounding_io.h"
using namespace xgboost;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
bool Close(double a, double b, double tol = 2e-12) {
  return a == b || (std::isnan(a) && std::isnan(b)) ||
         std::abs(a - b) <= tol * std::max(1., std::abs(b));
}
int main(int argc, char** argv) {
  try {
    Check(argc == 1 || (argc == 3 && std::string(argv[1]) == "--report-prefix"),
          "Usage: metal_ranking_rounding_api_test [--report-prefix new-prefix]");
    std::unique_ptr<ranking_rounding::Writer> inputs, report;
    if (argc == 3) {
      std::string prefix = argv[2];
      Check(!std::filesystem::exists(prefix+".inputs.bin") &&
            !std::filesystem::exists(prefix+".metal.bin"), "Use a new report prefix");
      inputs = std::make_unique<ranking_rounding::Writer>(prefix+".inputs.bin", 0);
      report = std::make_unique<ranking_rounding::Writer>(prefix+".metal.bin", 1);
    }
    std::mt19937 rng(39281);
    std::size_t cases = 0, fields = 0, prefixes = 0, untracked = 0, combined = 0,
                global_not_last = 0;
    Context cpu;
    cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
    for (std::string objective : {"rank:pairwise", "rank:map"})
      for (bool mean : {false, true})
        for (bool normalized : {false, true})
          for (std::uint32_t largest : {257u, 65537u})
            for (int mode = 0; mode < 4; ++mode) {
              MetaInfo info;
              info.group_ptr_ = {0, 0, 128, 128, 128 + largest, 129 + largest};
              info.num_row_ = info.group_ptr_.back();
              info.labels.Reshape(info.num_row_, 1);
              info.weights_.HostVector() = {2.f, 1e-20f, 3.f, 1e20f, 1.f};
              HostDeviceVector<float> pred(info.num_row_, 0.f);
              for (std::size_t g = 0; g + 1 < info.group_ptr_.size(); ++g) {
                auto begin = info.group_ptr_[g], n = info.group_ptr_[g + 1] - begin;
                for (std::size_t i = 0; i < n; ++i) {
                  pred.HostVector()[begin + i] = float(int(rng() % 17) - 8) / 7;
                  info.labels.HostView()(begin + i, 0) = float(rng() % 2);
                  if (mode == 0) {
                    pred.HostVector()[begin + i] = i == 64 ? 6.f : i == 65 ? -6.f : 0.f;
                    info.labels.HostView()(begin + i, 0) = i < 32 ? 0.f : float(i % 2 == 0);
                  }
                  if (mode == 1) {
                    float ys[] = {1.f, .99999994f, 1e-7f, 0.f};
                    info.labels.HostView()(begin + i, 0) = ys[i % 4];
                  }
                }
              }
              if (mode == 1)
                info.weights_.HostVector() = {8.897944450378418f, 3.939098358154297f,
                                              6.1226606369018555f, 1.4291059970855713f,
                                              9.536523818969727f};
              Args args{{"lambdarank_pair_method", mean ? "mean" : "topk"},
                        {"lambdarank_num_pair_per_sample", "8"},
                        {"lambdarank_unbiased", "true"},
                        {"lambdarank_normalization", normalized ? "true" : "false"},
                        {"lambdarank_score_normalization", mode == 3 ? "true" : "false"},
                        {"lambdarank_bias_norm", "1.5"}};
              ltr::LambdaRankParam param;
              param.UpdateAllowUnknown(args);
              ltr::RankingCache cache(&cpu, info, param);
              cuda_reference::RankingReference reference(objective, args, info);
              auto positive = reference.Positive(), negative = reference.Negative();
              if (mode == 2) positive.back() = 0.;
              if (mode == 3) {
                positive.back() = 1e-12;
                negative[0] = 1e-8;
              }
              ranking_rounding::Input exported;
              if (report) {
                exported.shape = {objective == "rank:map", mean, normalized, largest,
                                  static_cast<std::uint32_t>(mode),
                                  static_cast<std::uint32_t>(info.num_row_),
                                  static_cast<std::uint32_t>(positive.size())};
                exported.predictions = pred.ConstHostVector();
                exported.labels = info.labels.Data()->ConstHostVector();
                exported.weights = info.weights_.ConstHostVector();
                exported.positive = positive; exported.negative = negative;
              }
              reference.SetBias(positive, negative);
              linalg::Matrix<GradientPair> expected;
              reference.Gradient(pred, info, 0xffffffffu, &expected);
              std::vector<GradientPair> output(info.num_row_);
              std::vector<common::metal::RankingGroupStatistics> stats(info.group_ptr_.size() - 1);
              std::vector<std::array<double, 2>> prefix(objective == "rank:map" ? info.num_row_
                                                                                : 0);
              double factor = -1.;
              common::metal::RankingGradient(
                  objective == "rank:map" ? common::metal::RankingLossKind::kMAP
                                          : common::metal::RankingLossKind::kPairwise,
                  pred.ConstHostSpan(), info.labels.HostView().Slice(linalg::All(), 0),
                  info.group_ptr_, info.weights_.ConstHostSpan(), 8, normalized, mode == 3, true,
                  cache.WeightNorm(), {}, {}, output, mean, 0xffffffffu,
                  {positive, negative, param.Regularizer()}, {}, {stats, {&factor, 1}, prefix});
              Check(factor == reference.cost_rounding,
                    "Global cost factor differs from max(query factors)");
              global_not_last += factor != stats.back().cost_rounding;
              if (mode == 2) Check(factor == 1., "Nonfinite frexp exponent-zero policy differs");
              for (std::size_t g = 0; g < stats.size(); ++g) {
                auto e = reference.statistics[g];
                auto a = stats[g];
                Check(a.gradient_rounding == e.gr && a.hessian_rounding == e.hr,
                      "Gradient rounding bounds differ");
                Check(Close(a.normalization, e.norm, 2e-7), "Normalization differs");
                Check(a.weight == e.weight, "Combined binary64 weight multiplier differs");
                Check(a.cost_rounding == e.cost_rounding, "Query cost rounding bound differs");
                untracked += e.max_cost > e.tracked_cost;
                auto w = info.weights_.ConstHostVector()[g];
                combined += a.weight != w * float(cache.WeightNorm());
                fields += 5;
                auto begin = info.group_ptr_[g], n = info.group_ptr_[g + 1] - begin;
                if (!prefix.empty()) {
                  std::vector<std::size_t> rank(n);
                  std::iota(rank.begin(), rank.end(), 0);
                  std::stable_sort(rank.begin(), rank.end(), [&](auto i, auto j) {
                    return pred.ConstHostVector()[begin + i] > pred.ConstHostVector()[begin + j];
                  });
                  double hits = 0, precision = 0;
                  for (std::size_t i = 0; i < n; ++i) {
                    double label = info.labels.HostView()(begin + rank[i], 0);
                    hits += label;
                    precision += label / double(i + 1);
                    Check(
                        Close(prefix[begin + i][0], hits) && Close(prefix[begin + i][1], precision),
                        "Actual-label MAP prefix differs");
                    prefixes += 2;
                  }
                }
              }
              for (std::size_t i = 0; i < output.size(); ++i) {
                auto e = expected.HostView().Values()[i], a = output[i];
                Check(
                    Close(a.GetGrad(), e.GetGrad(), 2e-5) && Close(a.GetHess(), e.GetHess(), 2e-5),
                    "Source gradient differs");
              }
              for (std::size_t i = 0; i < positive.size(); ++i) {
                Check(Close(positive[i], reference.Positive()[i]) &&
                          Close(negative[i], reference.Negative()[i]),
                      "Binary64 bias state differs");
              }
              if (report) {
                ranking_rounding::Observation observed;
                observed.global_cost_rounding = factor; observed.weight_norm = cache.WeightNorm();
                for (auto g : stats)
                  observed.groups.push_back({g.normalization, g.gradient_rounding,
                      g.hessian_rounding, g.weight, g.cost_rounding});
                for (auto g : output) {
                  observed.gradients.push_back(g.GetGrad()); observed.gradients.push_back(g.GetHess());
                }
                for (auto p : prefix) observed.prefix.insert(observed.prefix.end(), p.begin(), p.end());
                observed.positive = positive; observed.negative = negative;
                inputs->Case(exported); report->Case(exported); report->Observe(observed);
              }
              ++cases;
              if (cases % 8 == 0) {
                std::printf(
                    "Checked %zu production rounding cases / %zu bound fields / %zu prefix "
                    "fields\n",
                    cases, fields, prefixes);
                std::fflush(stdout);
              }
            }
    std::printf("Coverage: untracked=%zu combined=%zu global=%zu\n", untracked, combined,
                global_not_last);
    Check(untracked && combined && global_not_last,
          "Untracked-cost/combined-weight/global-bound fixtures missing");
    if (report) { inputs->Finish(); report->Finish(); }
    std::printf(
        "PASS %zu production cases / %zu bounds / %zu prefixes; %zu untracked maxima, %zu "
        "combined-weight witnesses, %zu global-not-last factors\n",
        cases, fields, prefixes, untracked, combined, global_not_last);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
