// Grouped PR against the production CPU curve and independent log1p integration.
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
#include <tuple>
#include <vector>

#include "../xgboost/src/common/metal_auc.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/optional_weight.h"
#include "../xgboost/src/metric/auc.h"
#include "xgboost/context.h"

namespace xgboost::metric {
std::tuple<double, double, double> BinaryPRAUC(Context const*, common::Span<float const>,
                                               linalg::VectorView<float const>,
                                               common::OptionalWeights);
}
using namespace xgboost;
using common::Span;
std::string current;
size_t cases = 0, checked_rows = 0, checked_groups = 0;
double maximum_difference = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Close(double actual, double expected) {
  if (actual == expected || (std::isnan(actual) && std::isnan(expected))) return;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > 2e-12 + 2e-12 * std::abs(expected)) {
    std::fprintf(stderr, "%s: actual %.17g expected %.17g\n", current.c_str(), actual, expected);
    throw std::runtime_error("Grouped PR differs from reference");
  }
  maximum_difference = std::max(maximum_difference, std::abs(actual - expected));
}
double Reference(std::vector<float> const& predictions, std::vector<float> const& labels,
                 float weight) {
  // Independent continuous precision/recall integral using log1p, not the CPU
  // difference-of-logs formula. Apple arm64 long double has binary64 precision.
  long double total = 0, negative = 0;
  for (float y : labels) {
    total += y * weight;
    negative += (1.f - y) * weight;
  }
  if (!(total > 0) || !(negative > 0) || !std::isfinite(total) || !std::isfinite(negative))
    return NAN;
  std::vector<size_t> order(labels.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](auto a, auto b) { return predictions[a] > predictions[b]; });
  long double fp = 0, tp = 0, area = 0;
  for (size_t begin = 0; begin < order.size();) {
    size_t end = begin + 1;
    while (end < order.size() && predictions[order[begin]] == predictions[order[end]]) ++end;
    long double df = 0, dt = 0;
    for (size_t i = begin; i < end; ++i) {
      float label = labels[order[i]];
      df += i == 0 ? (1.0 - label) * weight : (1.f - label) * weight;
      dt += label * weight;
    }
    if (dt > 0) {
      auto a = 1 + df / dt;
      if (fp + tp == 0)
        area += dt / (a * total);
      else
        area += (dt - (fp - (a - 1) * tp) / a * std::log1p(a * dt / (fp + tp))) / (a * total);
    }
    fp += df;
    tp += dt;
    begin = end;
  }
  return double(area);
}

void Case(std::vector<uint32_t> const& lengths, int layout, int mode) {
  std::vector<bst_group_t> groups{0};
  for (auto n : lengths) groups.push_back(groups.back() + n);
  size_t rows = groups.back(), stride = layout == 0 ? 1 : (layout == 1 ? 3 : 7);
  current = "rows=" + std::to_string(rows) + " layout=" + std::to_string(layout) +
            " mode=" + std::to_string(mode);
  std::vector<float> p_storage(rows * stride + 8, -971), y_storage(rows * stride + 8, -193);
  linalg::VectorView<float> predictions{
      Span{p_storage.data() + 4, rows * stride}, {rows}, {stride}, DeviceOrd::CPU()};
  linalg::VectorView<float> labels{
      Span{y_storage.data() + 4, rows * stride}, {rows}, {stride}, DeviceOrd::CPU()};
  std::vector<float> weights(mode ? lengths.size() : 0);
  std::mt19937 rng(2871 + mode);
  for (size_t g = 0; g < lengths.size(); ++g) {
    float weight = float(g + 1) / 7;
    if (mode == 2) weight = std::ldexp(weight, int((g * 31) % 221) - 110);
    if (mode == 3) weight = std::numeric_limits<float>::denorm_min() * float(1 + g % 7);
    if (mode == 4) weight = 1e37f * float(1 + g % 17);
    if (mode == 6) weight = 0;
    if (mode == 7) {
      float choices[] = {1e-30f, 1e30f, 1.f, std::numeric_limits<float>::denorm_min(), 1e-10f, 3.f};
      weight = choices[g % 6];
    }
    if (mode == 9) {
      float choices[] = {INFINITY, NAN, -1.f, 0.f, 1.f};
      weight = choices[g % 5];
    }
    if (mode == 10) weight = std::numeric_limits<float>::denorm_min();
    if (mode == 11) weight = 1;
    if (!weights.empty()) weights[g] = weight;
    for (size_t r = groups[g]; r < groups[g + 1]; ++r) {
      float pred = float(int(rng() % 8191) - 4095) / 93;
      float label = float(rng() % 2);
      if (mode == 1 || mode == 11) pred = r % 2 ? -0.f : 0.f;
      if (mode == 2) {
        pred = float(rng() % 7);
        label = float(rng() % 17) / 16;
      }
      if (mode == 3) pred = std::ldexp(pred, -145);
      if (mode == 4) {
        if (r % 7 == 0) pred = INFINITY;
        if (r % 7 == 1) pred = -INFINITY;
      }
      if (mode == 5) label = float(g % 2);
      if (mode == 7) {
        float choices[] = {
            0, 1, std::nextafter(1.f, 0.f), 1e-30f, std::numeric_limits<float>::denorm_min(), .5f};
        label = choices[r % 6];
      }
      if (mode == 8 && g % 2 && r == groups[g]) label = NAN;
      if (mode == 10) label = std::nextafter(1.f, 0.f);
      if (mode == 11) label = 1e-30f;
      predictions(r) = pred;
      labels(r) = label;
    }
  }
  auto p_before = p_storage, y_before = y_storage;
  std::vector<common::metal::RankingAUCSums> individual(lengths.size() + 4, {-397, 973});
  Span<common::metal::RankingAUCSums> out{individual.data() + 2, lengths.size()};
  auto before = common::metal::GetStatistics();
  auto actual = common::metal::RankingPRAUC(predictions, labels, groups, weights, out);
  auto after = common::metal::GetStatistics();
  Check(after.ranking_pr_auc_dispatches ==
            before.ranking_pr_auc_dispatches + uint64_t(!lengths.empty()),
        "Missing grouped PR GPU dispatch");
  Check(after.group_sort_dispatches == before.group_sort_dispatches + uint64_t(rows != 0),
        "Missing GPU score sort");
  Check(std::memcmp(p_before.data(), p_storage.data(), p_storage.size() * 4) == 0,
        "Input predictions modified");
  Check(std::memcmp(y_before.data(), y_storage.data(), y_storage.size() * 4) == 0,
        "Input labels modified");
  Check(individual.front().area == -397 && individual.back().valid_groups == 973,
        "Output guard modified");
  Context cpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  double sum = 0, valid = 0;
  for (size_t g = 0; g < lengths.size(); ++g) {
    std::vector<float> p(lengths[g]), y(lengths[g]);
    for (size_t i = 0; i < lengths[g]; ++i) {
      p[i] = predictions(groups[g] + i);
      y[i] = labels(groups[g] + i);
    }
    float w = weights.empty() ? 1 : weights[g];
    double expected =
        std::get<2>(metric::BinaryPRAUC(&cpu, p, linalg::MakeVec(y), common::OptionalWeights{w}));
    bool is_valid = !std::isnan(expected);
    current += " group=" + std::to_string(g);
    Close(out[g].valid_groups, is_valid);
    Close(out[g].area, is_valid ? expected : 0);
    auto reference = Reference(p, y, w);
    Close(out[g].area, std::isnan(reference) ? 0 : reference);
    if (mode == 11 && !p.empty()) {
      double exact = double(1e-30f) / (1 + double(1e-30f));
      Check(std::abs(out[g].area / exact - 1) < 2e-12, "Tiny tied-score PR lost relative accuracy");
    }
    if (is_valid) {
      sum += expected;
      ++valid;
    }
    ++checked_groups;
  }
  Close(actual.area, sum);
  Close(actual.valid_groups, valid);
  ++cases;
  checked_rows += rows;
  if (cases % 24 == 0 || rows > 500000) {
    std::printf("Checked %zu PR cases / %zu rows / %zu query results\n", cases, checked_rows,
                checked_groups);
    std::fflush(stdout);
  }
}

void InvalidLabels() {
  std::vector<float> predictions{1, 0, 2}, labels{0, 1, 0};
  std::vector<bst_group_t> groups{0, 1, 1, 3};
  for (float invalid : {-std::numeric_limits<float>::denorm_min(), std::nextafter(1.f, 2.f), -1.f,
                        2.f, INFINITY, -INFINITY}) {
    Check(metric::PRAUCLabelInvalid{}(invalid), "CPU label validation oracle");
    labels[2] = invalid;
    bool rejected = false;
    try {
      common::metal::RankingPRAUC(linalg::MakeVec(predictions), linalg::MakeVec(labels), groups,
                                  {});
    } catch (std::exception const&) {
      rejected = true;
    }
    Check(rejected, "GPU failed to reject invalid PR label");
  }
  std::puts("PASS six GPU label-validation errors, including negative subnormal");
}

int main() {
  try {
    for (uint32_t n : {0u, 1u, 31u, 255u, 256u, 257u, 4097u, 65537u})
      for (int layout = 0; layout < 3; ++layout)
        for (int mode = 0; mode < 12; ++mode)
          Case({0, 1, 2, 3, 31, 32, 33, 255, 256, 257, 513, 0, n, n + 1, n / 2, 0}, layout, mode);
    // Carry propagation across three scan levels and query boundaries in every
    // part of a SIMD group, including an invalid query containing NaN labels.
    for (int mode : {0, 2, 3, 7, 8, 10, 11}) Case({1, 31, 1000003, 0, 65537, 257, 1}, 2, mode);
    Case({}, 0, 0);
    Case({0, 0, 0}, 0, 0);
    InvalidLabels();
    std::printf(
        "PASS %zu production grouped PR cases / %zu rows / %zu query results; max CPU/reference "
        "absolute difference %.9g\n",
        cases, checked_rows, checked_groups, maximum_difference);
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), error.what());
    return 1;
  }
}
