// Trace the exact production sampler, including raw (noncanonical) rank order.
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ranking.h"
#include "cuda_ranking_pairs.h"
#include "rng_contract_fixtures.h"
using namespace xgboost;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  try {
    std::size_t cases = 0, checked = 0, reversed = 0, homogeneous = 0;
    auto run = [&](rng_contracts::PairCase const& c) {
      auto const& sizes = c.sizes;
      auto k = c.count, seed = c.seed;
      auto mode = c.mode;
      auto batched = c.batched;
      std::vector<bst_group_t> groups{0};
      for (auto n : sizes) groups.push_back(groups.back() + n);
      std::size_t n = groups.back();
      auto const& predictions = c.predictions;
      auto const& labels = c.labels;
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
    rng_contracts::PairCases([&](auto const& c) {
      run(c);
      if (cases % 12 == 0) {
        std::printf("Checked %zu cases / %zu exact raw CUDA pairs\n", cases, checked);
        std::fflush(stdout);
      }
    });
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
