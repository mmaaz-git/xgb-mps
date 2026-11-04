// CUDA-source multiclass expressions and production dispatch checks. The
// primitive oracle rounds host binary64 exp to FP32; it is not libdevice.
#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/math.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ops.h"
#include "cuda_metric_reference.h"
using namespace xgboost;
namespace metal = common::metal;
using namespace cuda_metric;
std::size_t cases = 0, fields = 0, double_sum_witnesses = 0, kahan_witnesses = 0;
std::size_t initial_max_witnesses = 0, argmax_witnesses = 0, nonfinite_fields = 0;
std::uint64_t source_softmax_max_ulp = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Equal(float actual, float expected, char const* name, std::size_t row, unsigned k) {
  ++fields;
  nonfinite_fields += !std::isfinite(expected);
  if (Ulps(actual, expected)) {
    std::fprintf(stderr, "%s row%zu class%u actual %.9g [%08x] expected %.9g [%08x]\n", name, row,
                 k, actual, Cast<unsigned>(actual), expected, Cast<unsigned>(expected));
    throw std::runtime_error("Source-expression result differs");
  }
}
std::vector<float> Probabilities(std::vector<float> const& input, std::size_t begin,
                                 unsigned classes, bool gradient) {
  float maximum = gradient ? std::numeric_limits<float>::min() : input[begin];
  for (unsigned k = gradient ? 0 : 1; k < classes; ++k)
    maximum = std::fmax(input[begin + k], maximum);
  std::vector<float> terms(classes);
  double sum = 0;
  float float_sum = 0, kahan_sum = 0, correction = 0;
  for (unsigned k = 0; k < classes; ++k) {
    terms[k] = Exp(Sub(input[begin + k], maximum));
    sum += terms[k];
    float_sum = Add(float_sum, terms[k]);
    float value = Sub(terms[k], correction), next = Add(kahan_sum, value);
    correction = Sub(Sub(next, kahan_sum), value);
    kahan_sum = next;
  }
  if (std::isfinite(sum)) {
    double_sum_witnesses += float(sum) != float_sum;
    kahan_witnesses += float(sum) != kahan_sum;
  }
  for (auto& term : terms) term = Div(term, float(sum));
  return terms;
}
void Run(unsigned classes, std::size_t rows, unsigned mode, bool weighted, std::mt19937& rng) {
  auto n = rows * classes;
  std::vector<float> input(n), labels(rows), weights(rows);
  std::uniform_real_distribution<float> ordinary(-8, 8), broad(-105, 0);
  for (std::size_t row = 0; row < rows; ++row) {
    labels[row] = rng() % classes;
    weights[row] = float(rng() % 1000) / 127;
    switch (row % 8) {
      case 0:
        weights[row] = 0;
        break;
      case 1:
        weights[row] = 0x1p-149f;
        break;
      case 2:
        weights[row] = 0x1.fffffcp-127f;
        break;
      case 3:
        weights[row] = 1e30f;
        break;
    }
    for (unsigned k = 0; k < classes; ++k) {
      float value = ordinary(rng);
      if (mode == 1) value = broad(rng);
      if (mode == 2)
        value = Cast<float>(unsigned(rng() % 0x1000000) | (rng() % 2 ? 0x80000000u : 0));
      if (mode == 3) value = -90.f - float(row % 40);  // Positive-min initial max matters.
      if (mode == 4) value = k == row % classes ? 80.f : -80.f;
      if (mode == 5) value = k % 2 ? -0.f : 0.f;
      if (mode == 6) value = k % 2 ? -INFINITY : -3.f;
      if (mode == 7) value = NAN;
      if (mode == 8) value = k == 0 ? NAN : float(k);
      if (mode == 9) value = k == classes - 1 ? INFINITY : float(k);
      if (mode == 10) value = -INFINITY;
      // Integer-exponent terms span the rounding boundary of a FP32 sum.
      if (mode == 11) value = float(-int(rng() % 40) * std::log(2.));
      input[row * classes + k] = value;
    }
  }
  auto saved_input = input, saved_labels = labels, saved_weights = weights;
  std::vector<GradientPair> expected(n), output(n + 2, GradientPair{-71, -91});
  std::vector<float> softmax(n), argmax(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    auto begin = row * classes;
    auto probs = Probabilities(input, begin, classes, true);
    auto transformed = Probabilities(input, begin, classes, false);
    float weight = weighted ? weights[row] : 1.f;
    for (unsigned k = 0; k < classes; ++k) {
      float hess = std::fmax(Mul(Mul(Mul(2, probs[k]), Sub(1, probs[k])), weight), 1e-16f);
      float grad = Mul(labels[row] == k ? Sub(probs[k], 1) : probs[k], weight);
      expected[begin + k] = {grad, hess};
      softmax[begin + k] = transformed[k];
      initial_max_witnesses += Ulps(probs[k], transformed[k]) != 0;
    }
    auto start = input.begin() + begin, end = start + classes;
    argmax[row] = common::FindMaxIndex(start, end) - start;
    // The old maximum=-FLT_MAX loop skipped a leading NaN.
    unsigned old_index = 0;
    float old_maximum = -std::numeric_limits<float>::max();
    for (unsigned k = 0; k < classes; ++k)
      if (input[begin + k] > old_maximum) old_maximum = input[begin + (old_index = k)];
    argmax_witnesses += argmax[row] != old_index;
    // Execute the unmodified source helper on host as a separate libm check.
    std::vector<float> source(start, end);
    common::Softmax(source.begin(), source.end());
    for (unsigned k = 0; k < classes; ++k) {
      auto ulp = Ulps(source[k], transformed[k]);
      source_softmax_max_ulp = std::max(source_softmax_max_ulp, ulp);
      Check(ulp <= 4, "Host source expf softmax differs by more than four ULP");
    }
  }
  auto before = metal::GetStatistics();
  metal::Gradients(metal::Objective::kSoftmax, input, labels,
                   weighted ? common::Span<float const>{weights} : common::Span<float const>{},
                   classes, 0, 1, {}, {output.data() + 1, n});
  auto after = metal::GetStatistics();
  Check(after.objective_dispatches == before.objective_dispatches + 1, "Missing GPU gradients");
  Check(output.front().GetGrad() == -71 && output.front().GetHess() == -91 &&
            output.back().GetGrad() == -71 && output.back().GetHess() == -91,
        "Gradient output guard overwritten");
  for (std::size_t i = 0; i < n; ++i) {
    Equal(output[i + 1].GetGrad(), expected[i].GetGrad(), "gradient", i / classes, i % classes);
    Equal(output[i + 1].GetHess(), expected[i].GetHess(), "Hessian", i / classes, i % classes);
  }
  auto transformed = input;
  metal::TransformPredictions(metal::Transform::kSoftmax, transformed, classes);
  for (std::size_t i = 0; i < n; ++i)
    Equal(transformed[i], softmax[i], "softmax", i / classes, i % classes);
  transformed = input;
  metal::TransformPredictions(metal::Transform::kArgmax, transformed, classes);
  for (std::size_t row = 0; row < rows; ++row)
    Equal(transformed[row], argmax[row], "argmax", row, 0);
  Check(
      std::memcmp(transformed.data() + rows, input.data() + rows, (n - rows) * sizeof(float)) == 0,
      "Argmax changed tail outside its output span");
  Check(metal::GetStatistics().transform_dispatches == before.transform_dispatches + 2,
        "Missing GPU transforms");
  Check(std::memcmp(input.data(), saved_input.data(), n * sizeof(float)) == 0 &&
            labels == saved_labels && weights == saved_weights,
        "Input buffer was modified");
  // Exercise the production alias fallback with the new row dispatch.
  if (mode == 1 && weighted && classes == 17) {
    std::vector<GradientPair> alias(n + 2, GradientPair{-71, -91});
    auto* storage = reinterpret_cast<float*>(alias.data() + 1);
    std::memcpy(storage, input.data(), n * sizeof(float));
    metal::Gradients(metal::Objective::kSoftmax, {storage, n}, labels, weights, classes, 0, 1, {},
                     {alias.data() + 1, n});
    for (std::size_t i = 0; i < n; ++i) {
      Equal(alias[i + 1].GetGrad(), expected[i].GetGrad(), "alias gradient", i / classes,
            i % classes);
      Equal(alias[i + 1].GetHess(), expected[i].GetHess(), "alias Hessian", i / classes,
            i % classes);
    }
    Check(alias.front().GetGrad() == -71 && alias.back().GetHess() == -91, "Alias guard changed");
  }
  ++cases;
}
int main() {
  try {
    std::mt19937 rng(43920);
    for (unsigned classes : {1u, 2u, 3u, 17u, 37u, 257u, 1025u, 4097u}) {
      for (unsigned mode = 0; mode < 12; ++mode)
        for (bool weighted : {false, true})
          Run(classes, classes > 1000 ? 17 : 257, mode, weighted, rng);
      std::printf("PASS classes=%u: %zu cases / %zu fields\n", classes, cases, fields);
      std::fflush(stdout);
    }
    auto before = metal::GetStatistics();
    metal::Gradients(metal::Objective::kSoftmax, {}, {}, {}, 3, 0, 1, {}, {});
    metal::TransformPredictions(metal::Transform::kSoftmax, {}, 3);
    metal::TransformPredictions(metal::Transform::kArgmax, {}, 3);
    Check(before.objective_dispatches == metal::GetStatistics().objective_dispatches &&
              before.transform_dispatches == metal::GetStatistics().transform_dispatches,
          "Empty operation dispatched");
    std::vector<float> malformed(6, 0);
    std::vector<GradientPair> untouched(6, GradientPair{-71, -91});
    bool rejected = false;
    try {
      metal::Gradients(metal::Objective::kSoftmax, malformed, malformed, {}, 3, 0, 1, {},
                       untouched);
    } catch (std::exception const&) {
      rejected = true;
    }
    Check(rejected, "Multiclass must have one label per row");
    Check(before.objective_dispatches == metal::GetStatistics().objective_dispatches,
          "Invalid shape dispatched");
    for (auto const& pair : untouched)
      Check(pair.GetGrad() == -71 && pair.GetHess() == -91, "Invalid shape changed output");
    Check(double_sum_witnesses && kahan_witnesses && initial_max_witnesses && argmax_witnesses,
          "Stress corpus did not distinguish the source arithmetic/edge cases");
    std::printf(
        "PASS %zu cases / %zu fields; %zu nonfinite fields; denominator witnesses "
        "FP32=%zu Kahan=%zu; initial-max=%zu argmax=%zu; host source max ULP=%llu\n",
        cases, fields, nonfinite_fields, double_sum_witnesses, kahan_witnesses,
        initial_max_witnesses, argmax_witnesses,
        static_cast<unsigned long long>(source_softmax_max_ulp));
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
