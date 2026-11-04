// Production ranking validation/cache initialization against CPU references.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ranking.h"

using namespace xgboost;
namespace metal = common::metal;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
auto View(std::vector<float> const& values, size_t n, size_t stride) {
  return linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{values}, n, stride)
      .Slice(linalg::All(), 0);
}
uint32_t Flags(float v, bool ndcg) {
  if (!std::isfinite(v)) return ndcg ? 3 : 1;
  if (!ndcg) return !(std::abs(v) < 1e-6f || std::abs(v - 1.f) < 1e-6f);
  return uint32_t(v < 0 || std::abs(std::floor(v) - v) > 1e-6f) | (v >= 32 ? 2 : 0);
}
void ValidateLabels() {
  size_t cases = 0, rows = 0;
  std::vector<float> boundary{0,
                              -0.f,
                              std::numeric_limits<float>::denorm_min(),
                              -std::numeric_limits<float>::denorm_min(),
                              1e-6f,
                              -1e-6f,
                              1 - 1e-6f,
                              1 + 1e-6f,
                              1,
                              31,
                              32,
                              31.5f,
                              -1,
                              1e30f,
                              INFINITY,
                              -INFINITY,
                              NAN};
  auto anchors = boundary;
  for (float v : anchors)
    if (std::isfinite(v)) {
      boundary.push_back(std::nextafter(v, -INFINITY));
      boundary.push_back(std::nextafter(v, INFINITY));
    }
  for (bool ndcg : {false, true})
    for (size_t stride : {1, 3}) {
      auto kind = ndcg ? metal::RankingLabelKind::kNDCG : metal::RankingLabelKind::kBinary;
      auto run = [&](std::vector<float> const& values, size_t n, uint32_t expected) {
        auto before = metal::GetStatistics();
        auto actual = metal::CheckRankingLabels(View(values, n, stride), kind);
        auto after = metal::GetStatistics();
        Check(after.ranking_label_dispatches - before.ranking_label_dispatches == uint64_t(n != 0),
              "Ranking validation did not complete on GPU");
        if (actual != expected) {
          std::fprintf(stderr, "labels n=%zu stride=%zu ndcg=%d actual=%u expected=%u\n", n, stride,
                       ndcg, actual, expected);
          throw std::runtime_error("Ranking validation differs from CPU");
        }
        ++cases;
        rows += n;
      };
      for (float v : boundary) {
        std::vector<float> values(stride, NAN);
        values[0] = v;
        run(values, 1, Flags(v, ndcg));
      }
      for (size_t n : {0, 1, 31, 32, 255, 256, 257, 65537}) {
        std::vector<float> values(n * stride, NAN);
        for (size_t i = 0; i < n; ++i) values[i * stride] = i % 2;
        run(values, n, 0);
        if (!n) continue;
        for (size_t bad : {size_t{0}, n / 2, n - 1}) {
          auto copy = values;
          copy[bad * stride] = -std::numeric_limits<float>::denorm_min();
          run(copy, n, ndcg ? 1 : 0);
          copy[bad * stride] = 32;
          copy[(bad + n / 2) % n * stride] = -.5f;
          uint32_t expected = 0;
          for (size_t i = 0; i < n; ++i) expected |= Flags(copy[i * stride], ndcg);
          run(copy, n, expected);
        }
      }
    }
  std::printf("PASS %zu ranking label cases / %zu values; boundary, strided and padded lanes\n",
              cases, rows);
}
void ValidateCache() {
  std::mt19937 rng(8183);
  std::vector<std::vector<bst_group_t>> layouts{
      {}, {0, 0}, {1}, {0, 31, 0, 257, 1}, {4096, 32, 0}, {0, 4097, 1}, {65537, 1, 256}};
  size_t cases = 0, rows = 0, discount_checks = 0;
  double max_discount_error = 0, max_gain_error = 0;
  for (auto const& sizes : layouts) {
    std::vector<bst_group_t> groups{0};
    for (auto size : sizes) groups.push_back(groups.back() + size);
    size_t n = groups.back(),
           maximum = sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end());
    std::vector<double> expected_discounts(maximum);
    for (size_t i = 0; i < maximum; ++i) expected_discounts[i] = 1 / std::log2(double(i) + 2);
    for (bool exponential : {false, true})
      for (size_t stride : {1, 3})
        for (uint32_t topk : {0u, 1u, 10u, ~0u})
          for (int mode = 0; mode < 3; ++mode) {
            std::vector<float> values(n * stride, NAN);
            for (size_t i = 0; i < n; ++i) {
              float value = exponential ? float(rng() % 32) : float(int(rng() % 211) - 10) / 7;
              if (mode == 1) value = 0;
              if (mode == 2 && !exponential) value = std::ldexp(value, int(rng() % 241) - 120);
              values[i * stride] = value;
            }
            auto labels = View(values, n, stride);
            std::vector<double> discounts(maximum, NAN), inverse(sizes.size(), NAN);
            auto before = metal::GetStatistics();
            metal::InitNDCG(labels, groups, topk, exponential, discounts, inverse);
            auto after = metal::GetStatistics();
            Check(after.ranking_cache_dispatches - before.ranking_cache_dispatches ==
                      uint64_t(!sizes.empty()),
                  "NDCG cache did not complete on GPU");
            Check(after.group_sort_dispatches - before.group_sort_dispatches == uint64_t(n != 0),
                  "NDCG cache did not sort labels on GPU");
            for (size_t i = 0; i < maximum; ++i) {
              double error = std::abs(discounts[i] / expected_discounts[i] - 1);
              Check(std::isfinite(error) && error < 2e-14, "NDCG discount differs from CPU");
              max_discount_error = std::max(max_discount_error, error);
              ++discount_checks;
            }
            for (size_t g = 0; g < sizes.size(); ++g) {
              std::vector<size_t> order(sizes[g]);
              std::iota(order.begin(), order.end(), 0);
              std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                return labels(groups[g] + a) > labels(groups[g] + b);
              });
              double ideal = 0, absolute = 0;
              for (size_t i = 0; i < std::min(size_t(topk), order.size()); ++i) {
                auto y = labels(groups[g] + order[i]);
                double gain = exponential ? double((1u << uint32_t(y)) - 1) : double(y);
                double term = expected_discounts[i] * gain;
                ideal += term;
                absolute += std::abs(term);
              }
              double actual = inverse[g] == 0 ? 0 : 1 / inverse[g];
              double error = std::abs(actual - ideal) / std::max(1e-300, absolute);
              if (!std::isfinite(error) || error > 2e-12 || (ideal == 0 && inverse[g] != 0)) {
                std::fprintf(stderr, "cache n=%zu g=%zu exp=%d k=%u mode=%d gain %.17g != %.17g\n",
                             n, g, exponential, topk, mode, actual, ideal);
                throw std::runtime_error(
                    "NDCG ideal gain differs from CPU beyond summation tolerance");
              }
              max_gain_error = std::max(max_gain_error, error);
            }
            ++cases;
            rows += n;
          }
    std::printf("Checked %zu NDCG cache cases / %zu rows\n", cases, rows);
    std::fflush(stdout);
  }
  // Exact and near cancellation with valid linear gains. Report the conditioning
  // explicitly instead of claiming a relative inverse-gain tolerance near zero.
  for (std::vector<float> values :
       {std::vector<float>{1, 0, -2}, std::vector<float>{1, -float(std::log2(3.0))}}) {
    std::vector<bst_group_t> groups{0, static_cast<bst_group_t>(values.size())};
    std::vector<double> discounts(values.size()), inverse(1);
    metal::InitNDCG(View(values, values.size(), 1), groups, ~0u, false, discounts, inverse);
    double cpu_ideal = 0, magnitude = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      double term = values[i] / std::log2(double(i + 2));
      cpu_ideal += term;
      magnitude += std::abs(term);
    }
    // Apple arm64 long double has only 53 significand bits. This independent
    // near-zero oracle was calculated with Python Decimal at 80-digit precision:
    // 1 - Decimal.from_float(float32(log2(3))) * Decimal(2).ln()/Decimal(3).ln().
    double reference = values.size() == 3 ? 0 : 0x1.24ab5a39eab8ap-27;
    double actual = inverse[0] == 0 ? 0 : 1 / inverse[0];
    Check(std::abs(actual - reference) <= 2e-14 * magnitude,
          "Ill-conditioned linear ideal gain inaccurate");
    Check(reference != 0 || inverse[0] == 0, "Exact zero ideal gain lost");
    std::printf("Linear cancellation: CPU ideal %.17g, GPU %.17g, Decimal oracle %.17g\n",
                cpu_ideal, actual, reference);
  }
  std::printf(
      "PASS %zu NDCG cache cases / %zu rows / %zu discounts; max discount relative %.9g, gain "
      "scaled %.9g\n",
      cases, rows, discount_checks, max_discount_error, max_gain_error);
}
int main() {
  try {
    ValidateLabels();
    ValidateCache();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
