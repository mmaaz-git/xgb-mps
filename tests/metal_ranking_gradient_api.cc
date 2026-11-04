// Actual objectives/counters against CUDA-source pairs, rounding and gradient formulas.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "cuda_ranking_reference.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"

using namespace xgboost;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  try {
    Context cpu, gpu;
    cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
    gpu.Init({{"device", "mps"}, {"nthread", "2"}});
    std::mt19937 rng(9293);
    size_t cases = 0, rows = 0, exact = 0, subnormals = 0;
    uint64_t pair_count = 0;
    double max_error = 0;
    auto run = [&](std::string const& objective, std::vector<bst_group_t> const& sizes, uint32_t k,
                   bool normalized, bool score_normalized, int mode, bool batched,
                   bool mean = false) {
      MetaInfo info;
      info.group_ptr_.push_back(0);
      uint64_t expected_pairs = 0;
      for (auto n : sizes) {
        info.group_ptr_.push_back(info.group_ptr_.back() + n);
        auto m = std::min(n, k);
        expected_pairs += mean ? uint64_t(k) * n : uint64_t(m) * n - uint64_t(m) * (m + 1ull) / 2;
      }
      size_t n = info.group_ptr_.back();
      info.num_row_ = n;
      info.labels.Reshape(n, 1);
      auto labels = info.labels.HostView();
      HostDeviceVector<float> preds;
      auto& values = preds.HostVector();
      values.resize(n);
      bool exponential = mode != 3;
      for (size_t i = 0; i < n; ++i) {
        values[i] = float(int(rng() % 2001) - 1000) / 173;
        if (mode == 1) values[i] = i % 2 ? -0.f : 0.f;
        if (mode == 2 && i % 7 == 0) values[i] = i % 2 ? 100 : -100;
        float y = objective == "rank:map" ? float(rng() % 2) : float(rng() % 5);
        if (mode == 3 && objective != "rank:map") y = float(int(rng() % 15) - 1) / 3;
        if (mode == 3 && objective == "rank:map" && i % 17 == 0) y = 1e-7f;
        if (mode == 4) y = 0;
        labels(i, 0) = y;
      }
      if (mode >= 2) {
        auto& w = info.weights_.HostVector();
        w.resize(sizes.size());
        for (size_t i = 0; i < w.size(); ++i)
          w[i] = mode == 3 ? (i % 2 ? 1e20f : 1e-20f) : .1f + float(rng() % 31) / 7;
      }
      Args args{{"lambdarank_pair_method", mean ? "mean" : "topk"},
                {"lambdarank_num_pair_per_sample", std::to_string(k)},
                {"lambdarank_normalization", normalized ? "true" : "false"},
                {"lambdarank_score_normalization", score_normalized ? "true" : "false"},
                {"ndcg_exp_gain", exponential ? "true" : "false"}};
      std::unique_ptr<ObjFunction> go{ObjFunction::Create(objective, &gpu)};

      go->Configure(args);
      linalg::Matrix<GradientPair> expected, actual;
      cpu.Rng().seed(17891u + cases);
      gpu.Rng().seed(17891u + cases);
      cuda_reference::RankingReference reference(objective, args, info);
      reference.Gradient(preds, info, mean ? static_cast<std::uint32_t>(cpu.Rng()()) : 0,
                         &expected);
      auto before = common::metal::GetStatistics();
      go->GetGradient(preds, info, 0, &actual);
      auto after = common::metal::GetStatistics();
      Check(after.ranking_gradient_dispatches - before.ranking_gradient_dispatches == 1,
            "Ranking objective did not complete gradients on GPU");
      Check(after.ranking_pairs - before.ranking_pairs == expected_pairs,
            "Ranking pair count differs");
      Check((after.ranking_pair_batches - before.ranking_pair_batches > 1) == batched,
            "Ranking pair batch coverage differs");
      auto e = expected.HostView().Values(), a = actual.HostView().Values();
      Check(a.size() == e.size(), "Gradient shape differs");
      for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&e[i], &a[i], sizeof(GradientPair)) == 0) ++exact;
        for (int field = 0; field < 2; ++field) {
          double x = field ? a[i].GetHess() : a[i].GetGrad();
          double y = field ? e[i].GetHess() : e[i].GetGrad();
          if (y != 0 && std::abs(y) < std::numeric_limits<float>::min()) {
            ++subnormals;
            Check(std::abs(x - y) <=
                      2 * double(std::numeric_limits<float>::denorm_min()) + 2e-5 * std::abs(y),
                  "Subnormal ranking gradient was lost or inaccurate");
          }
          if (x == y || (std::isnan(x) && std::isnan(y))) continue;
          double error = std::abs(x - y) / std::max(1., std::abs(y));
          max_error = std::max(max_error, error);
          if (!std::isfinite(error) || std::abs(x - y) > 2e-6 + 2e-5 * std::abs(y)) {
            std::fprintf(
                stderr,
                "%s n=%zu k=%u norm=%d score_norm=%d mode=%d row=%zu field=%d: %.9g != %.9g\n",
                objective.c_str(), n, k, normalized, score_normalized, mode, i, field, x, y);
            throw std::runtime_error("Ranking gradient differs beyond FP32 tolerance");
          }
        }
      }
      ++cases;
      rows += n;
      pair_count += expected_pairs;
    };
    for (bool mean : {false, true}) {
      for (std::string objective : {"rank:pairwise", "rank:ndcg", "rank:map"}) {
        for (std::vector<bst_group_t> sizes :
             {std::vector<bst_group_t>{1}, {7, 31, 33}, {255, 256, 257}, {4097, 31, 1}}) {
          for (uint32_t k : {1, 8, 32})
            for (bool norm : {false, true})
              for (bool score_norm : {false, true})
                for (int mode = 0; mode < 5; ++mode)
                  run(objective, sizes, k, norm, score_norm, mode, false, mean);
          std::printf("Checked %zu gradient cases / %zu rows / %llu pairs\n", cases, rows,
                      static_cast<unsigned long long>(pair_count));
          std::fflush(stdout);
        }
      }
      run("rank:pairwise", std::vector<bst_group_t>(2560, 128), 32, true, true, 0, true, mean);
    }
    std::printf(
        "PASS %zu ranking gradient cases / %zu rows / %llu pairs; %zu exact gradient pairs, max "
        "scaled error %.9g; batched storage verified\n",
        cases, rows, static_cast<unsigned long long>(pair_count), exact, max_error);
    Check(subnormals > 0, "Subnormal gradient cases were not exercised");
    std::printf("Checked %zu subnormal gradient/Hessian fields with a two-ULP absolute tolerance\n",
                subnormals);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
