// Production GPU AUC compared with upstream CPU and independent weighted pairs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../xgboost/src/common/metal_auc.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/optional_weight.h"
#include "xgboost/context.h"
namespace xgboost::metric {
std::tuple<double, double, double> BinaryPRAUC(Context const*, common::Span<float const>,
                                               linalg::VectorView<float const>,
                                               common::OptionalWeights);
std::tuple<double, double, double> BinaryROCAUC(Context const*, common::Span<float const>,
                                                linalg::VectorView<float const>,
                                                common::OptionalWeights);
}  // namespace xgboost::metric
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
size_t current_n, current_targets, current_target;
int current_layout, current_mode;
void Close(double actual, double expected, char const* field = "pair", double atol = 0) {
  if (actual == expected || (std::isnan(actual) && std::isnan(expected))) return;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > atol + 2e-12 * std::abs(expected)) {
    std::fprintf(
        stderr,
        "n=%zu targets=%zu layout=%d mode=%d target=%zu field=%s actual %.17g, expected %.17g\n",
        current_n, current_targets, current_layout, current_mode, current_target, field, actual,
        expected);
    throw std::runtime_error("ROC-AUC differs from CPU beyond reduction-order tolerance");
  }
}
long double ReferencePR(std::vector<float> const& scores, std::vector<float> const& labels,
                        std::vector<float> const& weights) {
  // Independent long-double integral of precision over recall using log1p.
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return scores[a] > scores[b]; });
  long double total = 0, negative = 0;
  for (size_t i = 0; i < scores.size(); ++i) {
    auto w = weights.empty() ? 1.0f : weights[i];
    total += labels[i] * w;
    negative += (1.0f - labels[i]) * w;
  }
  if (total <= 0 || negative <= 0) return std::numeric_limits<long double>::quiet_NaN();
  long double fp = 0, tp = 0, area = 0;
  for (size_t begin = 0; begin < order.size();) {
    size_t end = begin + 1;
    while (end < order.size() && scores[order[end]] == scores[order[begin]]) ++end;
    long double df = 0, dt = 0;
    for (size_t k = begin; k < end; ++k) {
      auto i = order[k];
      auto w = weights.empty() ? 1.0f : weights[i];
      df += k == 0 ? (1.0 - labels[i]) * w : (1.0f - labels[i]) * w;
      dt += labels[i] * w;
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
  return area;
}
int main() {
  try {
    using namespace xgboost;
    Context cpu;
    cpu.Init({{"nthread", "2"}, {"device", "cpu"}});
    std::mt19937 rng(718);
    size_t cases = 0, checked = 0, pair_cases = 0, pr_references = 0;
    double max_pr_difference = 0;
    size_t ill_conditioned_differences = 0;
    for (size_t n : {0, 1, 2, 31, 255, 256, 257, 4097, 65537})
      for (size_t targets : {1, 3})
        for (int layout = 0; layout < 3; ++layout)
          for (int mode = 0; mode < 8; ++mode) {
            current_n = n;
            current_targets = targets;
            current_layout = layout;
            current_mode = mode;
            bool multiclass = targets == 3 && mode % 2;
            size_t columns = multiclass ? 1 : targets;
            size_t shape[2]{n, targets}, label_shape[2]{n, columns};
            size_t stride[2]{
                layout == 1 ? 1 : targets * (layout == 2 ? 2 : 1) + (layout == 2 ? 3 : 0),
                layout == 1 ? std::max(n, size_t(1)) : (layout == 2 ? 2 : 1)};
            size_t label_stride[2]{
                layout == 1 ? 1 : columns * (layout == 2 ? 2 : 1) + (layout == 2 ? 3 : 0),
                stride[1]};
            std::vector<float> predictions(n * stride[0] + targets * stride[1] + 1, -999);
            std::vector<float> labels(n * label_stride[0] + columns * label_stride[1] + 1, -999);
            std::vector<float> weights(mode >= 2 ? n : 0);
            linalg::MatrixView<float> p{predictions, shape, stride, DeviceOrd::CPU()};
            linalg::MatrixView<float> y{labels, label_shape, label_stride, DeviceOrd::CPU()};
            for (size_t row = 0; row < n; ++row) {
              for (size_t t = 0; t < targets; ++t) {
                float v = float(int(rng() % 32767) - 16383);
                if (mode == 1) v = row % 2 ? -0.0f : 0.0f;
                if (mode == 2) v = float(rng() % 7);
                if (mode == 3) v = std::ldexp(v, -149);  // nonzero subnormal scores
                if (mode == 4 && row % 37 == 0) v = row % 2 ? INFINITY : -INFINITY;
                p(row, t) = v;
              }
              for (size_t t = 0; t < columns; ++t) {
                float v = float(rng() % (multiclass ? targets : 2));
                if (mode == 5 && !multiclass) v = float(rng() % 17) / 16;
                if (multiclass && mode == 3 && row % 7 == 0)
                  v = std::numeric_limits<float>::denorm_min();
                if (mode == 6) v = 1;
                y(row, t) = v;
              }
              if (!weights.empty()) {
                float w = float(rng() % 31) / 8;
                if (mode == 3) w = std::ldexp(float(rng() % 127), -149);
                if (mode == 4) w = std::ldexp(w, int(rng() % 241) - 120);
                if (mode == 7) w = 0;
                weights[row] = w;
              }
            }
            std::vector<common::metal::ROCAUCSums> result(targets);
            auto before = common::metal::GetStatistics();
            common::metal::ROCAUC(p, y, weights, multiclass, result);
            auto after = common::metal::GetStatistics();
            Check(after.roc_auc_dispatches - before.roc_auc_dispatches == size_t(n != 0),
                  "ROC dispatch counter mismatch");
            Check(after.radix_sort_dispatches - before.radix_sort_dispatches == size_t(n != 0),
                  "ROC did not use one batched GPU sort");
            std::vector<common::metal::ROCAUCSums> pr_result(targets);
            auto pr_before = common::metal::GetStatistics();
            common::metal::PRAUC(p, y, weights, multiclass, pr_result);
            auto pr_after = common::metal::GetStatistics();
            Check(pr_after.pr_auc_dispatches - pr_before.pr_auc_dispatches == size_t(n != 0),
                  "PR-AUC dispatch counter mismatch");
            Check(
                pr_after.radix_sort_dispatches - pr_before.radix_sort_dispatches == size_t(n != 0),
                "PR-AUC did not use one batched GPU sort");
            for (size_t t = 0; t < targets; ++t) {
              current_target = t;
              std::vector<float> scores(n), response(n);
              for (size_t row = 0; row < n; ++row) {
                scores[row] = p(row, t);
                response[row] = multiclass ? float(y(row, 0) == t) : y(row, t);
              }
              double fp = 0, tp = 0, area = 0;
              if (n)
                std::tie(fp, tp, area) = metric::BinaryROCAUC(
                    &cpu, scores, linalg::MakeVec(response.data(), n, DeviceOrd::CPU()),
                    common::OptionalWeights{weights});
              auto expected_pr = metric::BinaryPRAUC(
                  &cpu, scores, linalg::MakeVec(response.data(), n, DeviceOrd::CPU()),
                  common::OptionalWeights{weights});
              Close(pr_result[t].fp, std::get<0>(expected_pr), "PR fp");
              Close(pr_result[t].tp, std::get<1>(expected_pr), "PR tp");
              // Preserve accuracy of the continuous integral when the upstream
              // difference-of-logs expression loses precision at extreme weights.
              if (mode == 4) {
                auto expected = std::get<2>(expected_pr), actual = pr_result[t].area;
                if (std::isfinite(actual) &&
                    (!std::isfinite(expected) || std::abs(actual - expected) > 2e-10))
                  ++ill_conditioned_differences;
              } else
                Close(pr_result[t].area, std::get<2>(expected_pr), "PR area", 2e-10);
              if (mode != 4 && std::isfinite(pr_result[t].area))
                max_pr_difference = std::max(
                    max_pr_difference, std::abs(pr_result[t].area - std::get<2>(expected_pr)));
              if (n <= 257) {
                Close(pr_result[t].area, double(ReferencePR(scores, response, weights)),
                      "PR independent integral", 2e-10);
                ++pr_references;
              }
              Close(result[t].fp, fp, "fp");
              Close(result[t].tp, tp, "tp");
              if (mode == 4 && fp > 0 && tp > 0) {
                // With a 240-bit weight range, upstream's differences of cumulative
                // sums can lose a tiny trapezoid. Bound the observable normalized
                // AUC error, rather than relative error of a near-zero area.
                Close(result[t].area / (fp * tp), area / (fp * tp), "normalized area", 2e-12);
              } else
                Close(result[t].area, area, "area");
              if (n <= 257 && mode < 5) {
                long double pairs = 0;
                for (size_t i = 0; i < n; ++i)
                  if (response[i] == 1)
                    for (size_t j = 0; j < n; ++j)
                      if (response[j] == 0) {
                        long double wi = weights.empty() ? 1 : weights[i];
                        long double wj = weights.empty() ? 1 : weights[j];
                        pairs += wi * wj *
                                 (scores[i] > scores[j]    ? 1
                                  : scores[i] == scores[j] ? 0.5L
                                                           : 0);
                      }
                if (mode == 4 && fp > 0 && tp > 0)
                  Close(result[t].area / (fp * tp),
                        double(pairs / (static_cast<long double>(fp) * tp)), "normalized pair",
                        2e-12);
                else
                  Close(result[t].area, double(pairs));
                ++pair_cases;
              }
              checked += n;
            }
            ++cases;
            if (cases % 144 == 0) {
              std::printf("Checked %zu ROC cases / %zu score-label pairs\n", cases, checked);
              std::fflush(stdout);
            }
          }
    std::printf(
        "PASS %zu production ROC-AUC cases / %zu score-label pairs, %zu independent "
        "weighted-pair checks; layouts, ties, weights and three-level scans\n",
        cases, checked, pair_cases);
    std::printf(
        "PASS %zu production PR-AUC cases / %zu score-label pairs, %zu long-double "
        "integral checks; maximum CPU difference %.9g\n",
        cases, checked, pr_references, max_pr_difference);
    {
      size_t shape[2]{2, 1};
      std::vector<float> scores{1, 0}, labels{0, 1}, weights{1e30f, 1};
      common::metal::ROCAUCSums result;
      common::metal::PRAUC({scores, shape, DeviceOrd::CPU()}, {labels, shape, DeviceOrd::CPU()},
                           weights, false, {&result, 1});
      Close(result.area, 1.0 / (2.0 * weights[0]), "tiny PR area");
      auto cpu_result = metric::BinaryPRAUC(&cpu, scores,
          linalg::MakeVec(labels.data(), labels.size(), DeviceOrd::CPU()), common::OptionalWeights{weights});
      Check(std::get<2>(cpu_result) == 1.0, "Update the documented upstream cancellation example");
      std::printf("PASS extreme-weight PR integral %.17g; upstream CPU %.17g\n", result.area, std::get<2>(cpu_result));
    }
    std::printf("Extreme-weight PR cases with unstable CPU differences: %zu\n",
                ill_conditioned_differences);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
