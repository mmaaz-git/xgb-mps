// Grouped PR against CUDA-source formulas and independent log1p integration.
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
#include "grouped_pr_contract_io.h"

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
std::unique_ptr<grouped_pr::Exporter> report;
void Record(std::vector<float> const& p, std::vector<float> const& y,
            std::vector<bst_group_t> const& groups, std::vector<float> const& weights,
            common::metal::RankingAUCSums actual,
            std::vector<common::metal::RankingAUCSums> const& individual = {}, bool rejected=false) {
  if (!report) return;
  grouped_pr::Input in{report->NextSpec(), groups, p, y, weights};
  grouped_pr::Observation observed;
  observed.rejected = rejected;
  if (!rejected) {
    observed.original = {actual.area, actual.valid_groups};
    for (auto pair : individual) {
      observed.guards.push_back(pair.area); observed.guards.push_back(pair.valid_groups);
    }
    auto compact = grouped_pr::Compact(in);
    auto combined = common::metal::RankingPRAUC(linalg::MakeVec(compact.predictions),
        linalg::MakeVec(compact.labels), compact.ptr, compact.weights);
    observed.common = {combined.area, combined.valid_groups};
    for (std::size_t g=0; g+1<compact.ptr.size(); ++g) {
      auto begin=compact.ptr[g], end=compact.ptr[g+1];
      std::vector<float> gp(compact.predictions.begin()+begin,compact.predictions.begin()+end);
      std::vector<float> gy(compact.labels.begin()+begin,compact.labels.begin()+end);
      std::vector<float> gw;
      if (!compact.weights.empty()) gw.push_back(compact.weights[g]);
      auto single=common::metal::RankingPRAUC(linalg::MakeVec(gp),linalg::MakeVec(gy),
                                             std::vector<bst_group_t>{0,end-begin},gw);
      observed.isolated.push_back(single.area); observed.isolated.push_back(single.valid_groups);
    }
  }
  report->Add(in,observed);
}
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
// Host source oracle for metric/auc.cu RankingPRAUC/RankingPRAUCImpl. Reduction
// association is serial here; it does not emulate a CUB architecture or libdevice.
common::metal::RankingAUCSums CudaReference(std::vector<float> const& predictions,
                                            std::vector<float> const& labels, float weight) {
  if (labels.size() < 2) return {0, 0};  // Empty-query safety extension as well.
  double total = 0;
  for (float y : labels) total += y * weight;
  std::vector<size_t> order(labels.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return predictions[a] > predictions[b]; });
  double fp = 0, tp = 0, fp_prev = 0, tp_prev = 0, area = 0;
  for (size_t i = 0; i < order.size(); ++i) {
    float label = labels[order[i]];
    float negative = (1.0 - double(label)) * double(weight), positive = label * weight;
    fp += negative;
    tp += positive;
    if (i + 1 < order.size() && predictions[order[i]] == predictions[order[i + 1]]) continue;
    double previous = tp_prev / total, recall = tp / total;
    double h = tp == tp_prev ? 0 : (fp - fp_prev) / (tp - tp_prev);
    double a = h + 1, b = tp == tp_prev ? 0 : (fp_prev - h * tp_prev) / total;
    area += b != 0 ? (recall - previous -
                      b / a * (std::log(a * recall + b) - std::log(a * previous + b))) /
                         a
                   : (recall - previous) / a;
    fp_prev = fp;
    tp_prev = tp;
  }
  return fp * tp > 0 ? common::metal::RankingAUCSums{area, 1} : common::metal::RankingAUCSums{0, 0};
}
void CudaEdgeRules() {
  std::vector<float> predictions{0}, labels{.5f};
  std::vector<bst_group_t> groups{0, 1};
  auto result = common::metal::RankingPRAUC(linalg::MakeVec(predictions), linalg::MakeVec(labels),
                                            groups, {});
  Check(result.valid_groups == 0 && result.area == 0,
        "CUDA grouped PR excludes one-row queries, including fractional labels");
  Record(predictions, labels, groups, {}, result);
  size_t checked = 1;
  // All tied predictions need one interval: no log/reduction-tree ambiguity.
  for (float label : {.25f, .1f, std::nextafter(1.f, 0.f), 1e-30f}) {
    for (float weight : {.7f, 1.f, std::numeric_limits<float>::denorm_min()}) {
      predictions = {0, 0};
      labels = {label, label};
      groups = {0, 2};
      std::vector<float> weights{weight};
      auto expected = CudaReference(predictions, labels, weight);
      result = common::metal::RankingPRAUC(linalg::MakeVec(predictions), linalg::MakeVec(labels),
                                           groups, weights);
      Check(result.valid_groups == expected.valid_groups && result.area == expected.area,
            "CUDA grouped PR must round every FP term once after binary64 arithmetic");
      Record(predictions, labels, groups, weights, result);
      ++checked;
    }
  }
  // Nonfinite/negative weights are rejected by public metadata. The direct API
  // still follows CUDA's arithmetic and FP*TP validity rather than extra flags.
  for (float weight : {INFINITY, NAN, -1.f}) {
    predictions = {0, 0};
    labels = {.25f, .75f};
    groups = {0, 2};
    std::vector<float> weights{weight};
    auto expected = CudaReference(predictions, labels, weight);
    result = common::metal::RankingPRAUC(linalg::MakeVec(predictions), linalg::MakeVec(labels),
                                         groups, weights);
    Check(result.valid_groups == expected.valid_groups, "CUDA grouped PR FP*TP validity");
    Close(result.area, expected.area);
    Record(predictions, labels, groups, weights, result);
    ++checked;
  }
  std::printf("PASS %zu CUDA grouped PR edge/cast cases\n", checked);
}
double Reference(std::vector<float> const& predictions, std::vector<float> const& labels,
                 float weight) {
  // Independent continuous precision/recall integral using log1p, not the CPU
  // difference-of-logs formula. Apple arm64 long double has binary64 precision.
  long double total = 0, negative = 0;
  for (float y : labels) {
    total += y * weight;
    negative += float((1.0 - double(y)) * double(weight));
  }
  if (labels.size() < 2 || !(total > 0) || !(negative > 0) || !std::isfinite(total) ||
      !std::isfinite(negative))
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
      df += float((1.0 - double(label)) * double(weight));
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
    auto expected = CudaReference(p, y, w);
    current += " group=" + std::to_string(g);
    Close(out[g].valid_groups, expected.valid_groups);
    Close(out[g].area, expected.area);
    // Binary labels and finite nonnegative weights have compatible CPU term
    // arithmetic. Keep those independent production comparisons unchanged.
    if (w >= 0 && std::isfinite(w) &&
        std::all_of(y.begin(), y.end(), [](float v) { return v == 0 || v == 1; })) {
      double cpu_value =
          std::get<2>(metric::BinaryPRAUC(&cpu, p, linalg::MakeVec(y), common::OptionalWeights{w}));
      Close(out[g].valid_groups, !std::isnan(cpu_value));
      Close(out[g].area, std::isnan(cpu_value) ? 0 : cpu_value);
    }
    auto reference = Reference(p, y, w);
    if (expected.valid_groups && std::isfinite(reference)) Close(out[g].area, reference);
    if (mode == 11 && p.size() >= 2) {
      double exact = double(1e-30f) / (1 + double(1e-30f));
      Check(std::abs(out[g].area / exact - 1) < 2e-12, "Tiny tied-score PR lost relative accuracy");
    }
    sum += expected.area;
    valid += expected.valid_groups;
    ++checked_groups;
  }
  Close(actual.area, sum);
  Close(actual.valid_groups, valid);
  Record(p_storage, y_storage, groups, weights, actual, individual);
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
    Record(predictions, labels, groups, {}, {}, {}, rejected);
  }
  std::puts("PASS six GPU label-validation errors, including negative subnormal");
}

int main(int argc, char** argv) {
  try {
    report = grouped_pr::ReportOption(argc, argv);
    CudaEdgeRules();
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
    if (report) report->Finish();
    std::printf(
        "PASS %zu production grouped PR cases / %zu rows / %zu query results; max "
        "CUDA-source/integral "
        "absolute difference %.9g\n",
        cases, checked_rows, checked_groups, maximum_difference);
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), error.what());
    return 1;
  }
}
