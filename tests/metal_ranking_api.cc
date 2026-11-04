// Production grouped Metal sorting against independent stable CPU sorting.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ranking.h"

void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
// Source oracle: metric/rank_metric.cu. Small, exactly reducible fixtures isolate
// operation order and branch rules; they do not assume a particular CUB tree.
void CheckCudaRules() {
  using namespace xgboost;
  using common::metal::RankingMetricKind;
  size_t checked = 0;
  auto run = [&](RankingMetricKind kind, std::vector<float> labels, std::vector<bst_group_t> groups,
                 std::vector<float> weights, uint32_t topk, bool minus,
                 std::vector<double> discounts, std::vector<double> inverse, double expected,
                 char const* message) {
    std::vector<float> scores(labels.size());
    for (size_t i = 0; i < scores.size(); ++i) scores[i] = -float(i);
    auto score_view =
        linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{scores}, scores.size());
    auto label_view =
        linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{labels}, labels.size());
    auto result = common::metal::RankingScore(kind, score_view, label_view, groups, weights, topk,
                                              minus, false, discounts, inverse);
    Check(result.score == expected || (std::isnan(result.score) && std::isnan(expected)), message);
    double sum_weight = weights.empty() ? groups.size() - 1 : 0;
    for (float w : weights) sum_weight += w;
    Check(result.weight == sum_weight, "CUDA metric weight denominator changed");
    ++checked;
  };
  run(RankingMetricKind::kPrecision, {}, {0, 0, 0}, {2, 3}, 10, false, {}, {}, 0,
      "CUDA precision: empty queries must contribute zero");
  run(RankingMetricKind::kPrecision, {1}, {0, 1}, {2}, 0, false, {}, {}, 0,
      "CUDA precision: topk zero must contribute zero");
  float tiny = std::numeric_limits<float>::denorm_min();
  run(RankingMetricKind::kPrecision, {tiny}, {0, 1}, {.5f}, 1, false, {}, {}, double(tiny) * .5,
      "CUDA precision must retain FP64 label/weight terms");
  float y = .99999994f, w = .7f;
  run(RankingMetricKind::kPrecision, {y, 0, 0}, {0, 3}, {w}, 3, false, {}, {},
      double(y) / 3 * double(w), "CUDA precision must divide each term before weighting");
  for (bool minus : {false, true}) {
    for (float weight : {0.f, 2.f}) {
      run(RankingMetricKind::kMAP, {0, 0}, {0, 0, 2, 2}, {weight, weight, weight}, 10, minus, {},
          {}, minus ? 0 : 3, "CUDA MAP fallback must be unweighted, including empties");
    }
  }
  double area = 2.75;
  Check(area * double(w) / 3 != area / 3 * double(w), "MAP order fixture no longer distinguishes");
  run(RankingMetricKind::kMAP, {1, 1, 0, 1}, {0, 4}, {w}, 4, false, {}, {}, area * double(w) / 3,
      "CUDA MAP must multiply weight before dividing by relevance");
  run(RankingMetricKind::kMAP, {1}, {0, 1}, {2}, 0, false, {}, {}, NAN,
      "CUDA MAP positive relevance with topk zero must produce NaN");
  run(RankingMetricKind::kMAP, {0}, {0, 1}, {2}, 0, false, {}, {}, 1,
      "CUDA MAP no relevance with topk zero must use unweighted fallback");
  Check((1. + .1) * .3 != .3 + .1 * .3, "NDCG order fixture no longer distinguishes");
  run(RankingMetricKind::kNDCG, {1, 1}, {0, 2}, {}, 2, false, {1, .1}, {.3}, (1. + .1) * .3,
      "CUDA NDCG must apply inverse IDCG after DCG reduction");
  run(RankingMetricKind::kNDCG, {1}, {0, 1}, {2}, 1, false, {1}, {NAN}, NAN,
      "CUDA NDCG NaN inverse must propagate");
  for (double inverse : {0., -1.}) {
    run(RankingMetricKind::kNDCG, {1}, {0, 1}, {2}, 1, false, {1}, {inverse}, 1,
        "CUDA NDCG nonpositive inverse fallback must remain unweighted");
  }
  std::printf("PASS %zu exact CUDA-source ranking edge/order cases\n", checked);
  std::fflush(stdout);
}
void CheckScores() {
  using namespace xgboost;
  using common::metal::RankingMetricKind;
  std::mt19937 rng(7182);
  std::vector<std::vector<bst_group_t>> layouts{
      {}, {0, 0}, {1}, {0, 31, 0, 257, 1}, {4096, 32, 0}, {0, 4097, 1}, {65537, 1, 256}};
  size_t cases = 0, checked = 0;
  double max_difference = 0;
  for (auto const& sizes : layouts) {
    std::vector<bst_group_t> groups{0};
    for (auto size : sizes) groups.push_back(groups.back() + size);
    size_t n = groups.back(),
           maximum = sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end());
    std::vector<double> discounts(maximum);
    for (size_t i = 0; i < maximum; ++i) discounts[i] = 1 / std::log2(double(i) + 2);
    for (int kind = 0; kind < 4; ++kind) {
      for (size_t stride : {1, 3}) {
        for (uint32_t topk : {0u, 1u, 10u, ~0u}) {
          for (int mode = 0; mode < 4; ++mode) {
            std::vector<float> pred(n * stride, NAN), label(n * stride, NAN), weight;
            if (mode) weight.resize(sizes.size());
            for (auto& w : weight) w = mode == 1 ? 0 : float(rng() % 101) / 7;
            if (mode == 3)
              for (auto& w : weight) w = std::ldexp(1.f, int(rng() % 241) - 120);
            for (size_t i = 0; i < n; ++i) {
              pred[i * stride] = float(int(rng() % 101) - 50) / 7;
              constexpr float binary[]{0, 1, 0.99999994f, 1.e-7f, -1.e-7f};
              constexpr float linear[]{-.4f, 0, .7f, 1.2f, 2.8f, 6.4f};
              constexpr float exponential[]{0, 1, 2, 5, 15, 31, 1.e-7f, 1.0000001f};
              float y = (kind == 0 || kind == 3) ? binary[rng() % 5]
                        : kind == 1              ? exponential[rng() % 8]
                                                 : linear[rng() % 6];
              if (kind == 0 && i % 17 == 0) y = std::numeric_limits<float>::denorm_min();
              if (kind == 3 && mode == 2) y = float(rng() % 2);
              label[i * stride] = mode == 1 ? 0 : y;
            }
            auto scores =
                linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{pred}, n, stride)
                    .Slice(linalg::All(), 0);
            auto labels = linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{label},
                                                 n, stride)
                              .Slice(linalg::All(), 0);
            std::vector<double> inverse(sizes.size(), 0);
            double expected_score = 0, expected_weight = 0;
            bool minus = (mode & 1) != 0;
            for (size_t g = 0; g < sizes.size(); ++g) {
              auto begin = groups[g], count = sizes[g], k = std::min(topk, count);
              std::vector<size_t> order(count);
              std::iota(order.begin(), order.end(), 0);
              auto gain = [&](float y) {
                return kind == 1 ? double((1u << uint32_t(y)) - 1) : double(y);
              };
              if (kind == 1 || kind == 2) {
                std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                  return labels(begin + a) > labels(begin + b);
                });
                double ideal = 0;
                for (size_t i = 0; i < k; ++i)
                  ideal += discounts[i] * gain(labels(begin + order[i]));
                inverse[g] = ideal == 0 ? 0 : 1 / ideal;
              }
              std::iota(order.begin(), order.end(), 0);
              std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                return scores(begin + a) > scores(begin + b);
              });
              float w = weight.empty() ? 1 : weight[g];
              double score = 0;
              if (kind == 3) {
                double hits = 0;
                for (size_t i = 0; i < count; ++i) {
                  auto y = labels(begin + order[i]);
                  hits += y;
                  if (i < k) score += hits / double(i + 1) * y;
                }
                score =
                    hits > 0 ? (score * double(w)) / std::min(hits, double(topk)) : (minus ? 0 : 1);
              } else {
                for (size_t i = 0; i < k; ++i) {
                  auto y = labels(begin + order[i]);
                  score += kind == 0 ? double(y) / double(k) * double(w) : discounts[i] * gain(y);
                }
                if (kind != 0)
                  score = inverse[g] <= 0 ? (minus ? 0 : 1) : (score * inverse[g]) * double(w);
              }
              expected_score += score;
              expected_weight += w;
            }
            auto before = common::metal::GetStatistics();
            auto result = common::metal::RankingScore(kind == 0   ? RankingMetricKind::kPrecision
                                                      : kind == 3 ? RankingMetricKind::kMAP
                                                                  : RankingMetricKind::kNDCG,
                                                      scores, labels, groups, weight, topk, minus,
                                                      kind == 1, discounts, inverse);
            auto after = common::metal::GetStatistics();
            Check(after.ranking_metric_dispatches - before.ranking_metric_dispatches ==
                      uint64_t(!sizes.empty()),
                  "Ranking score did not run on GPU");
            Check(after.group_sort_dispatches - before.group_sort_dispatches == uint64_t(n != 0),
                  "Ranking score did not use GPU sort");
            auto close = [&](double actual, double expected) {
              if (actual == expected || (std::isnan(actual) && std::isnan(expected))) return;
              double error = std::abs(actual - expected) / std::max(1., std::abs(expected));
              max_difference = std::max(max_difference, error);
              if (!std::isfinite(error) || error > 2e-12) {
                std::fprintf(stderr,
                             "score rows=%zu kind=%d stride=%zu k=%u mode=%d: %.17g != %.17g\n", n,
                             kind, stride, topk, mode, actual, expected);
                throw std::runtime_error("Ranking score differs from CUDA-source formulas");
              }
            };
            close(result.score, expected_score);
            close(result.weight, expected_weight);
            ++cases;
            checked += n;
          }
        }
      }
    }
    std::printf("Checked %zu ranking score cases / %zu rows\n", cases, checked);
    std::fflush(stdout);
  }
  std::printf(
      "PASS %zu precision/NDCG/MAP cases / %zu rows; max scaled CUDA-source difference %.9g\n",
      cases, checked, max_difference);
}

int main() {
  try {
    CheckCudaRules();
    using namespace xgboost;
    std::vector<std::vector<bst_group_t>> cases{{},
                                                {0, 0, 0},
                                                {1},
                                                {31, 0, 32, 33},
                                                {255, 256, 0, 1},
                                                {257, 0, 1, 256},
                                                {0, 65537, 0},
                                                {65537, 1, 4097, 0, 257}};
    std::vector<bst_group_t> many(4097), skew(65537, 1);
    for (size_t g = 0; g < many.size(); ++g) many[g] = g % 257;
    skew[32768] = 4097;
    cases.push_back({0, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 0});
    cases.push_back({4097, 1});
    cases.push_back(many);
    cases.push_back(skew);
    std::mt19937 rng(761);
    size_t checked = 0, count = 0, small = 0, radix = 0;
    for (auto const& sizes : cases) {
      std::vector<bst_group_t> groups{0};
      for (auto size : sizes) groups.push_back(groups.back() + size);
      size_t n = groups.back();
      bool large = std::any_of(sizes.begin(), sizes.end(), [](auto v) { return v > 4096; });
      for (size_t stride : {1, 3}) {
        for (int mode = 0; mode < 5; ++mode) {
          std::vector<float> data(n * stride, std::numeric_limits<float>::quiet_NaN());
          for (size_t i = 0; i < n; ++i) {
            float v = float(int(rng() % 101) - 50) / 7;
            if (mode == 1) v = i % 3 ? -0.0f : 0.0f;
            if (mode == 2) v = float(i);
            if (mode == 3) v = -float(i);
            if (mode == 4) {
              constexpr uint32_t bits[]{0,          0x80000000, 1,          0x80000001,
                                        0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
                                        0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff};
              auto b = bits[rng() % (sizeof(bits) / sizeof(bits[0]))];
              std::memcpy(&v, &b, sizeof(v));
            }
            data[i * stride] = v;
          }
          auto values =
              linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{data}, n, stride)
                  .Slice(linalg::All(), 0);
          std::vector<size_t> actual(n), expected(n);
          for (size_t g = 0; g < sizes.size(); ++g) {
            auto begin = expected.begin() + groups[g], end = expected.begin() + groups[g + 1];
            std::iota(begin, end, 0);
            std::stable_sort(begin, end, [&](size_t a, size_t b) {
              return values(groups[g] + a) > values(groups[g] + b);
            });
          }
          auto before = common::metal::GetStatistics();
          common::metal::SortGroups(values, groups, actual);
          auto after = common::metal::GetStatistics();
          Check(after.group_sort_dispatches - before.group_sort_dispatches == uint64_t(n != 0),
                "Grouped sort did not complete on GPU");
          Check(after.radix_sort_dispatches - before.radix_sort_dispatches == uint64_t(n && large),
                "Grouped sort selected an unexpected path");
          if (actual != expected) {
            for (size_t i = 0; i < n; ++i)
              if (actual[i] != expected[i]) {
                std::fprintf(stderr,
                             "rows=%zu groups=%zu stride=%zu mode=%d index=%zu: %zu != %zu\n", n,
                             sizes.size(), stride, mode, i, actual[i], expected[i]);
                break;
              }
            throw std::runtime_error("Grouped sort differs from stable CPU order");
          }
          checked += n;
          ++count;
          if (n) (large ? radix : small)++;
        }
      }
      std::printf("Checked %zu cases / %zu sorted rows\n", count, checked);
      std::fflush(stdout);
    }
    std::printf("PASS %zu grouped sort cases / %zu rows; %zu local-query and %zu radix launches\n",
                count, checked, small, radix);
    CheckScores();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
