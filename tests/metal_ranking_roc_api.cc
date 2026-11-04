// Actual GPU grouped ROC compared with CUDA-source pair enumeration and compatible CPU cases.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_auc.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "xgboost/context.h"

namespace xgboost::metric {
double GroupRankingROC(Context const*, common::Span<float const>, linalg::VectorView<float const>,
                       float);
}
using namespace xgboost;
using common::Span;
std::string current;
size_t cases = 0, checked_rows = 0;
uint64_t enumerated_pairs = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Close(double actual, double expected) {
  if (std::isnan(actual) && std::isnan(expected)) return;
  if (actual == expected) return;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > 2e-12 * std::max(1.0, std::abs(expected))) {
    std::fprintf(stderr, "%s: actual %.17g expected %.17g\n", current.c_str(), actual, expected);
    throw std::runtime_error("Grouped ROC differs from reference");
  }
}

double PairReference(std::vector<float> const& scores, std::vector<float> const& labels,
                     float weight) {
  if (scores.size() < 3) return 0;
  // Work directly in input order. Label comparisons choose which document is
  // earlier; equal labels retain input order. No merge or inversion algorithm.
  float squared = weight * weight;
  double area = 0, total = 0;
  for (size_t i = 0; i < scores.size(); ++i)
    for (size_t j = i + 1; j < scores.size(); ++j) {
      bool reverse = labels[i] < labels[j];
      float difference = reverse ? scores[j] - scores[i] : scores[i] - scores[j];
      float value = difference > 0 ? 1.f : (difference == 0 ? 0.5f : 0.f);
      area += value * squared;
      total += squared;
      ++enumerated_pairs;
    }
  // metric/auc.cu::RankingAUC writes zero unless the scanned pair weight is >0.
  // This differs from CPU for NaN query weights (private API only).
  return total > 0 ? area / total : 0;
}

void Case(size_t n, int layout, int mode) {
  current = "rows=" + std::to_string(n) + " layout=" + std::to_string(layout) +
            " mode=" + std::to_string(mode);
  std::vector<bst_group_t> groups{0};
  for (size_t length :
       {n, size_t(0), n / 2, size_t(1), size_t(2), size_t(3), size_t(31), size_t(0)})
    groups.push_back(groups.back() + length);
  size_t rows = groups.back(), stride = layout == 0 ? 1 : (layout == 1 ? 3 : 7);
  std::vector<float> storage_p(rows * stride + 8, -971), storage_y(rows * stride + 8, -193);
  linalg::VectorView<float> predictions{
      Span{storage_p.data() + 4, rows * stride}, {rows}, {stride}, DeviceOrd::CPU()};
  linalg::VectorView<float> labels{
      Span{storage_y.data() + 4, rows * stride}, {rows}, {stride}, DeviceOrd::CPU()};
  std::vector<float> weights(mode ? groups.size() - 1 : 0);
  std::mt19937 rng(1179 + mode);
  for (size_t row = 0; row < rows; ++row) {
    float p = float(int(rng() % 8191) - 4095) / 93;
    float y = float(rng() % 11);
    if (mode == 1) p = float(rng() % 5);
    if (mode == 2) p = row % 2 ? -0.0f : 0.0f;
    if (mode == 3) {
      p = std::ldexp(p, -145);
      y = std::ldexp(y, -149);
    }
    if (mode == 4) {
      if (row % 5 == 0) p = INFINITY;
      if (row % 5 == 1) p = -INFINITY;
      if (row % 5 == 2) p = NAN;
    }
    if (mode == 9) y = 1;
    if (mode == 10) y /= 11;
    predictions(row) = p;
    labels(row) = y;
  }
  for (size_t g = 0; g < weights.size(); ++g) {
    float w = float(g + 1) / 7;
    if (mode == 5) w = g % 2 ? -0.f : 0.f;
    if (mode == 6) w = std::ldexp(float(1 + g % 3), -74);
    if (mode == 7) w = std::ldexp(float(g + 1), 55);
    if (mode == 8) w = std::ldexp(float(g + 1), -100);
    if (mode == 10) w = std::ldexp(w, int((g * 17) % 121) - 60);
    weights[g] = w;
  }
  auto before_p = storage_p, before_y = storage_y;
  std::vector<common::metal::RankingAUCSums> output(groups.size() + 3, {-397, 973});
  Span<common::metal::RankingAUCSums> individual{output.data() + 2, groups.size() - 1};
  auto before = common::metal::GetStatistics();
  auto result = common::metal::RankingROCAUC(predictions, labels, groups, weights, individual);
  auto after = common::metal::GetStatistics();
  Check(after.ranking_roc_auc_dispatches == before.ranking_roc_auc_dispatches + 1,
        "Missing GPU ROC dispatch");
  Check(after.group_sort_dispatches == before.group_sort_dispatches + uint64_t(rows != 0),
        "Missing GPU label sort");
  Check(std::memcmp(before_p.data(), storage_p.data(), storage_p.size() * 4) == 0,
        "Predictions modified");
  Check(std::memcmp(before_y.data(), storage_y.data(), storage_y.size() * 4) == 0,
        "Labels modified");
  Check(output.front().area == -397 && output.back().valid_groups == 973, "Output guard changed");
  Context cpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  double sum = 0, valid = 0;
  for (size_t g = 0; g + 1 < groups.size(); ++g) {
    size_t count = groups[g + 1] - groups[g];
    if (count < 3) {
      Close(individual[g].area, 0);
      Close(individual[g].valid_groups, 0);
      continue;
    }
    std::vector<float> p(count), y(count);
    for (size_t i = 0; i < count; ++i) {
      p[i] = predictions(groups[g] + i);
      y[i] = labels(groups[g] + i);
    }
    float w = weights.empty() ? 1 : weights[g];
    auto expected = metric::GroupRankingROC(&cpu, p, linalg::MakeVec(y), w);
    Close(individual[g].area, expected);
    Close(individual[g].area, PairReference(p, y, w));
    Close(individual[g].valid_groups, 1);
    sum += expected;
    ++valid;
  }
  Close(result.area, sum);
  Close(result.valid_groups, valid);
  ++cases;
  checked_rows += rows;
  if (cases % 30 == 0) {
    std::printf("Checked %zu cases / %zu rows / %llu enumerated pairs\n", cases, checked_rows,
                (unsigned long long)enumerated_pairs);
    std::fflush(stdout);
  }
}

void LargeAnalytic() {
  constexpr size_t n = 1000003;
  std::vector<float> labels(n, 0), predictions(n);
  std::vector<bst_group_t> groups{0, n};
  for (int mode = 0; mode < 3; ++mode) {
    for (size_t i = 0; i < n; ++i)
      predictions[i] = mode == 0 ? float(i) : (mode == 1 ? float(n - i) : 1.f);
    current = "million-row analytic mode=" + std::to_string(mode);
    auto result = common::metal::RankingROCAUC(linalg::MakeVec(predictions),
                                               linalg::MakeVec(labels), groups, {});
    Close(result.area, mode == 0 ? 0 : (mode == 1 ? 1 : 0.5));
    Close(result.valid_groups, 1);
    std::printf("PASS million-row analytic case %d: %llu implied pairs, 64-bit counts\n", mode,
                (unsigned long long)(uint64_t(n) * (n - 1) / 2));
    std::fflush(stdout);
  }
}

void CudaEdgeContracts() {
  auto previous_pairs = enumerated_pairs;
  current = "CUDA grouped ROC finite query weight with overflowing FP32 square";
  auto empty = common::metal::RankingROCAUC(linalg::MakeVec(std::vector<float>{}),
                                            linalg::MakeVec(std::vector<float>{}),
                                            std::vector<bst_group_t>{0}, {});
  Close(empty.area, 0);
  Close(empty.valid_groups, 0);
  std::vector<float> p{0, 1, 2}, y{0, 1, 2}, w{1e30f};
  std::vector<bst_group_t> groups{0, 3};
  Context cpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "1"}});
  bool cpu_error = false;
  try {
    metric::GroupRankingROC(&cpu, p, linalg::MakeVec(y), w[0]);
  } catch (std::exception const&) {
    cpu_error = true;
  }
  Check(cpu_error, "Expected CPU-only overflow rejection");
  auto result = common::metal::RankingROCAUC(linalg::MakeVec(p), linalg::MakeVec(y), groups, w);
  Close(result.area, PairReference(p, y, w[0]));
  Close(result.valid_groups, 1);

  // Many queries in one command exercise the source weight branch, independently
  // enumerated FP32 pair terms, exceptional predictions, stable label ties and
  // short-query exclusions. Nonfinite/negative weights are private API tests;
  // public DMatrix metadata continues to reject them.
  std::vector<float> edge_weights{0,         -0.f,       1,          .7f,   -.7f,
                                  0x1p-100f, 0x1.8p-75f, 0x1.4p-74f, 1e19f, 1e30f,
                                  INFINITY,  -INFINITY,  NAN};
  std::vector<float> predictions, labels, weights;
  std::vector<bst_group_t> offsets{0};
  std::vector<common::metal::RankingAUCSums> expected;
  double expected_sum = 0, expected_valid = 0;
  for (size_t n : {0, 1, 2, 3, 31, 257}) {
    for (float weight : edge_weights) {
      for (int mode = 0; mode < 5; ++mode) {
        std::vector<float> query_p(n), query_y(n);
        for (size_t i = 0; i < n; ++i) {
          query_y[i] = mode == 4 ? 1 : float((i * 3) % 11);
          query_p[i] = float((i * 7) % 13);
          if (mode == 1) query_p[i] = i % 2 ? -0.f : 0.f;
          if (mode == 2) query_p[i] = std::ldexp(float(int(i % 5) - 2), -149);
          if (mode == 3) {
            float values[]{INFINITY, -INFINITY, NAN, INFINITY, -INFINITY, 0};
            query_p[i] = values[i % 6];
          }
        }
        double area = PairReference(query_p, query_y, weight), valid = n >= 3;
        expected.push_back({area, valid});
        expected_sum += area;
        expected_valid += valid;
        predictions.insert(predictions.end(), query_p.begin(), query_p.end());
        labels.insert(labels.end(), query_y.begin(), query_y.end());
        weights.push_back(weight);
        offsets.push_back(predictions.size());
      }
    }
  }
  auto before_p = predictions, before_y = labels, before_w = weights;
  auto before_offsets = offsets;
  std::vector<common::metal::RankingAUCSums> guarded(expected.size() + 2, {-397, 973});
  Span<common::metal::RankingAUCSums> individual{guarded.data() + 1, expected.size()};
  auto before = common::metal::GetStatistics();
  result = common::metal::RankingROCAUC(linalg::MakeVec(predictions), linalg::MakeVec(labels),
                                        offsets, weights, individual);
  auto after = common::metal::GetStatistics();
  Check(after.ranking_roc_auc_dispatches == before.ranking_roc_auc_dispatches + 1,
        "CUDA edge cases did not execute on GPU");
  Check(after.group_sort_dispatches == before.group_sort_dispatches + 1,
        "CUDA edge cases did not sort on GPU");
  auto unchanged = [](auto const& a, auto const& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
  };
  Check(unchanged(predictions, before_p) && unchanged(labels, before_y) &&
            unchanged(weights, before_w) && offsets == before_offsets,
        "CUDA edge inputs modified");
  Check(guarded.front().area == -397 && guarded.front().valid_groups == 973 &&
            guarded.back().area == -397 && guarded.back().valid_groups == 973,
        "CUDA edge output guards modified");
  for (size_t g = 0; g < expected.size(); ++g) {
    current = "CUDA grouped ROC edge query=" + std::to_string(g);
    Close(individual[g].area, expected[g].area);
    Close(individual[g].valid_groups, expected[g].valid_groups);
  }
  Close(result.area, expected_sum);
  Close(result.valid_groups, expected_valid);
  std::printf(
      "PASS %zu CUDA grouped ROC edge queries / %zu rows / %llu enumerated pairs; "
      "NaN/overflow/underflow weights, "
      "FP32 pair terms, short queries, guards and GPU counters\n",
      expected.size(), predictions.size(), (unsigned long long)(enumerated_pairs - previous_pairs));
  enumerated_pairs = previous_pairs;
}

int main() {
  try {
    CudaEdgeContracts();
    for (size_t n : {0, 1, 2, 3, 31, 256, 257, 1025})
      for (int layout = 0; layout < 3; ++layout)
        for (int mode = 0; mode < 11; ++mode) Case(n, layout, mode);
    for (size_t n : {4097, 8193})
      for (int mode : {0, 1, 6, 9}) Case(n, 2, mode);
    LargeAnalytic();
    std::printf(
        "PASS %zu native grouped ROC cases / %zu rows / %llu independently enumerated pairs, plus "
        "3 million-row analytic cases and CUDA empty/overflow checks\n",
        cases, checked_rows, (unsigned long long)enumerated_pairs);
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), error.what());
    return 1;
  }
}
