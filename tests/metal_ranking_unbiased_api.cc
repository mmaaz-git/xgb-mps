// Compare repeated unbiased objective calls and serialized position-bias state.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/objective.h"
using namespace xgboost;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void CheckBatched() {
  Context cpu, gpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "8"}});
  gpu.Init({{"device", "mps"}, {"nthread", "8"}});
  MetaInfo info;
  constexpr size_t n = 524288, query = 128;
  info.num_row_ = n;
  for (size_t i = 0; i <= n; i += query) info.group_ptr_.push_back(i);
  info.labels.Reshape(n, 1);
  HostDeviceVector<float> predictions(n, 0);
  std::mt19937 rng(92183);
  for (size_t i = 0; i < n; ++i) {
    info.labels.HostView()(i, 0) = float(rng() % 5);
    predictions.HostVector()[i] = float(int(rng() % 1001) - 500) / 200;
  }
  for (std::string method : {"topk", "mean"}) {
    Args args{{"lambdarank_pair_method", method},
              {"lambdarank_num_pair_per_sample", "8"},
              {"lambdarank_unbiased", "true"}};
    std::unique_ptr<ObjFunction> co{ObjFunction::Create("rank:pairwise", &cpu)},
        go{ObjFunction::Create("rank:pairwise", &gpu)};
    co->Configure(args);
    go->Configure(args);
    for (int iteration = 0; iteration < 2; ++iteration) {
      cpu.Rng().seed(12345 + iteration);
      gpu.Rng().seed(12345 + iteration);
      linalg::Matrix<GradientPair> expected, actual;
      co->GetGradient(predictions, info, iteration, &expected);
      auto before = common::metal::GetStatistics();
      go->GetGradient(predictions, info, iteration, &actual);
      auto after = common::metal::GetStatistics();
      Check(after.ranking_pair_batches - before.ranking_pair_batches > 1,
            "Large unbiased test did not exercise multiple batches");
      auto a = actual.HostView().Values(), e = expected.HostView().Values();
      double max_error = 0;
      for (size_t i = 0; i < n; ++i)
        for (int field = 0; field < 2; ++field) {
          double x = field ? a[i].GetHess() : a[i].GetGrad(),
                 y = field ? e[i].GetHess() : e[i].GetGrad();
          max_error = std::max(max_error, std::abs(x - y) / std::max(1., std::abs(y)));
          Check(std::isfinite(x) && std::abs(x - y) <= 2e-6 + 2e-5 * std::abs(y),
                "Batched unbiased gradient differs");
        }
      std::printf(
          "PASS batched unbiased %s iteration %d / %zu rows / %llu batches; max scaled error "
          "%.9g\n",
          method.c_str(), iteration, n,
          static_cast<unsigned long long>(after.ranking_pair_batches - before.ranking_pair_batches),
          max_error);
      std::fflush(stdout);
    }
  }
}
int main() {
  try {
    CheckBatched();
    Context cpu, gpu;
    cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
    gpu.Init({{"device", "mps"}, {"nthread", "2"}});
    std::mt19937 rng(11939);
    size_t cases = 0, rows = 0, bias_values = 0, exact = 0;
    double max_gradient_error = 0, max_bias_error = 0;
    for (std::string objective : {"rank:pairwise", "rank:ndcg", "rank:map"})
      for (std::string method : {"topk", "mean"})
        for (uint32_t k : {1, 8, 32})
          for (bool normalized : {false, true})
            for (bool score_normalized : {false, true})
              for (double bias_norm : {0., 1., 2.5})
                for (int mode = 0; mode < 4; ++mode) {
                  MetaInfo info;
                  info.group_ptr_ = {0, 1, 8, 39, 72, 329};
                  info.num_row_ = 329;
                  info.labels.Reshape(info.num_row_, 1);
                  auto labels = info.labels.HostView();
                  HostDeviceVector<float> pred(info.num_row_, 0);
                  auto& values = pred.HostVector();
                  for (size_t i = 0; i < values.size(); ++i) {
                    labels(i, 0) = mode == 1                 ? 0.f
                                   : objective == "rank:map" ? float(rng() % 2)
                                                             : float(rng() % 5);
                    values[i] = mode == 2 ? 0.f : float(int(rng() % 1001) - 500) / 151;
                  }
                  if (mode == 3) info.weights_.HostVector() = {0.f, .5f, 1.f, 2.f, 7.f};
                  Args args{{"lambdarank_pair_method", method},
                            {"lambdarank_num_pair_per_sample", std::to_string(k)},
                            {"lambdarank_unbiased", "true"},
                            {"lambdarank_normalization", normalized ? "true" : "false"},
                            {"lambdarank_score_normalization", score_normalized ? "true" : "false"},
                            {"lambdarank_bias_norm", std::to_string(bias_norm)}};
                  std::unique_ptr<ObjFunction> co{ObjFunction::Create(objective, &cpu)},
                      go{ObjFunction::Create(objective, &gpu)};
                  co->Configure(args);
                  go->Configure(args);
                  for (int iteration = 0; iteration < 3; ++iteration) {
                    cpu.Rng().seed(8912 + cases);
                    gpu.Rng().seed(8912 + cases);
                    linalg::Matrix<GradientPair> expected, actual;
                    co->GetGradient(pred, info, iteration, &expected);
                    auto before = common::metal::GetStatistics();
                    go->GetGradient(pred, info, iteration, &actual);
                    auto after = common::metal::GetStatistics();
                    Check(
                        after.ranking_gradient_dispatches - before.ranking_gradient_dispatches == 1,
                        "Unbiased gradient did not execute on GPU");
                    Check(after.ranking_bias_dispatches - before.ranking_bias_dispatches == 1,
                          "Bias update did not execute on GPU");
                    auto e = expected.HostView().Values(), a = actual.HostView().Values();
                    for (size_t i = 0; i < a.size(); ++i)
                      for (int field = 0; field < 2; ++field) {
                        double x = field ? a[i].GetHess() : a[i].GetGrad(),
                               y = field ? e[i].GetHess() : e[i].GetGrad();
                        if (x == y) {
                          ++exact;
                          continue;
                        }
                        if (std::isnan(x) && std::isnan(y)) continue;
                        double error = std::abs(x - y) / std::max(1., std::abs(y));
                        max_gradient_error = std::max(max_gradient_error, error);
                        if (!std::isfinite(error) || std::abs(x - y) > 2e-6 + 2e-5 * std::abs(y)) {
                          std::fprintf(stderr,
                                       "%s %s k=%u norm=%d score=%d bias=%g mode=%d iter=%d "
                                       "row=%zu field=%d: %.9g != %.9g\n",
                                       objective.c_str(), method.c_str(), k, normalized,
                                       score_normalized, bias_norm, mode, iteration, i, field, x,
                                       y);
                          throw std::runtime_error("Unbiased gradient differs");
                        }
                      }
                    Json cj{Object{}}, gj{Object{}};
                    co->SaveConfig(&cj);
                    go->SaveConfig(&gj);
                    for (auto key : {"ti+", "tj-"}) {
                      auto const& c = get<F32Array>(cj[key]);
                      auto const& g = get<F32Array>(gj[key]);
                      Check(c.size() == g.size(), "Serialized bias shape differs");
                      for (size_t i = 0; i < c.size(); ++i) {
                        ++bias_values;
                        double x = g[i], y = c[i];
                        if (x == y || (std::isnan(x) && std::isnan(y))) continue;
                        double error = std::abs(x - y) / std::max(1., std::abs(y));
                        max_bias_error = std::max(max_bias_error, error);
                        if (!std::isfinite(error) || std::abs(x - y) > 2e-6 + 2e-5 * std::abs(y)) {
                          std::fprintf(stderr,
                                       "%s %s k=%u bias=%g mode=%d iter=%d %s[%zu]: %.9g != %.9g\n",
                                       objective.c_str(), method.c_str(), k, bias_norm, mode,
                                       iteration, key, i, x, y);
                          throw std::runtime_error("Serialized bias differs");
                        }
                      }
                    }
                    ++cases;
                    rows += values.size();
                    // Feed exactly the same changing predictions to both objective states.
                    if (mode != 1)
                      for (auto& v : values) v += float(int(rng() % 101) - 50) / 300;
                  }
                  if (cases % 216 == 0) {
                    std::printf("Checked %zu unbiased calls / %zu rows / %zu bias values\n", cases,
                                rows, bias_values);
                    std::fflush(stdout);
                  }
                }
    std::printf(
        "PASS %zu unbiased calls / %zu rows / %zu bias values / %zu exact gradient fields; maximum "
        "scaled gradient error %.9g, bias error %.9g\n",
        cases, rows, bias_values, exact, max_gradient_error, max_bias_error);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
