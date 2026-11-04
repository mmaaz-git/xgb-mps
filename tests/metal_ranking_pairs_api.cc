// Trace the exact production sampler, including raw (noncanonical) rank order.
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ranking.h"
#include "cuda_ranking_pairs.h"
using namespace xgboost;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  try {
    std::mt19937 rng(881927);
    std::size_t cases = 0, checked = 0, reversed = 0, homogeneous = 0;
    auto run = [&](std::vector<bst_group_t> sizes, std::uint32_t k, std::uint32_t seed, int mode,
                   bool batched) {
      std::vector<bst_group_t> groups{0};
      for (auto n : sizes) groups.push_back(groups.back() + n);
      std::size_t n = groups.back();
      std::vector<float> predictions(n), labels(n * 3, -999.f);
      for (std::size_t i = 0; i < n; ++i) {
        predictions[i] = mode == 0 ? (i % 2 ? 0.f : -0.f) : float(int(rng() % 23) - 11);
        float y = float(rng() % 5);
        if (mode == 1) y = 0;
        if (mode == 2) {
          float edge[] = {-0.f,
                          0.f,
                          std::numeric_limits<float>::denorm_min(),
                          -std::numeric_limits<float>::denorm_min(),
                          1.e-7f,
                          -1.e-7f};
          y = edge[rng() % 6];
        }
        if (mode == 3) y = float(int(rng() % 1001) - 500) / 173.f;
        labels[3 * i] = y;
      }
      auto label = linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{labels}, n, 3)
                       .Slice(linalg::All(), 0);
      std::vector<GradientPair> gradients(n);
      std::vector<cuda_reference::RankingPair> trace(n * k);
      auto before = common::metal::GetStatistics();
      common::metal::RankingGradient(common::metal::RankingLossKind::kPairwise, predictions, label,
                                     groups, {}, k, true, true, true, 1., {}, {}, gradients, true,
                                     seed, {}, trace);
      auto after = common::metal::GetStatistics();
      if (n) {
        Check(after.ranking_gradient_dispatches == before.ranking_gradient_dispatches + 1,
              "Production objective did not dispatch");
        Check((after.ranking_pair_batches - before.ranking_pair_batches > 1) == batched,
              "Production batch reuse coverage differs");
      }
      for (std::size_t g = 0; g < sizes.size(); ++g) {
        std::vector<std::size_t> rank(sizes[g]);
        std::iota(rank.begin(), rank.end(), 0);
        std::stable_sort(rank.begin(), rank.end(), [&](auto a, auto b) {
          return predictions[groups[g] + a] > predictions[groups[g] + b];
        });
        std::vector<float> y(sizes[g]);
        for (std::size_t i = 0; i < y.size(); ++i) y[i] = labels[3 * (groups[g] + i)];
        auto expected = cuda_reference::MeanPairs(y, rank, seed + static_cast<std::uint32_t>(g), k);
        for (std::size_t i = 0; i < expected.size(); ++i) {
          auto actual = trace[std::size_t{groups[g]} * k + i];
          if (actual != expected[i]) {
            std::fprintf(
                stderr, "case=%zu seed=%u group=%zu k=%u mode=%d pair=%zu: (%u,%u) != (%u,%u)\n",
                cases, seed, g, k, mode, i, actual[0], actual[1], expected[i][0], expected[i][1]);
            throw std::runtime_error("Production pairs differ from CUDA source");
          }
          reversed += actual[0] > actual[1];
          homogeneous += actual[0] == actual[1];
          ++checked;
        }
      }
      ++cases;
    };
    for (std::uint32_t seed : {0u, 1u, 2147483647u, 2147483648u, 0xfffffffeu, 0xffffffffu}) {
      for (std::uint32_t k : {1u, 8u, 32u})
        for (int mode = 0; mode < 4; ++mode)
          run({0, 1, 0, 7, 31, 33, 0, 255, 256, 257, 4097, 0}, k, seed, mode, false);
      std::printf("Checked %zu cases / %zu exact raw CUDA pairs\n", cases, checked);
      std::fflush(stdout);
    }
    run({0, 0}, 8, 0, 0, false);
    run(std::vector<bst_group_t>(512, 257), 32, 0xffffffffu, 2, true);
    Check(reversed > 0 && homogeneous > 0, "Raw orientation/homogeneous coverage missing");
    std::printf(
        "PASS %zu production cases / %zu exact pairs / %zu reversed / %zu homogeneous; "
        "strides, empty queries, seed wrap and reused batch buffers verified\n",
        cases, checked, reversed, homogeneous);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
