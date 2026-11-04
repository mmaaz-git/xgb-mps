// Production objective/transform arithmetic against extracted CUDA expressions.
#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../results/cuda_objective_reference_generated.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ops.h"
#include "cuda_metric_reference.h"
using namespace xgboost;
namespace metal = common::metal;
using namespace cuda_metric;
struct Math {
  static float Exp(float x) { return float(std::exp(double(x))); }
  static float Log1p(float x) { return float(std::log1p(double(x))); }
  static float Sqrt(float x) { return float(std::sqrt(double(x))); }
  static double Pow(float x, int y) { return std::pow(double(x), double(y)); }
  // CUDA fmaxf treats either NaN as missing data. Apple std::fmax instead
  // propagates a signaling NaN; use the documented CUDA predicate here.
  static float FMax(float x, float y) {
    if (std::isnan(x)) return y;
    if (std::isnan(y)) return x;
    return std::fmax(x, y);
  }
};
struct NativeMath : Math {
  static float FMax(float x, float y) { return std::fmax(x, y); }
  static float Exp(float x) { return std::exp(x); }
  static float Log1p(float x) { return std::log1p(x); }
  static float Sqrt(float x) { return std::sqrt(x); }
};
using Ref = objective_reference::Ref<Math>;
using Native = objective_reference::Ref<NativeMath>;
static_assert(std::is_same_v<decltype(std::pow(0.f, 2)), double>);
std::size_t cases = 0, fields = 0, subnormal_fields = 0, nonfinite_fields = 0;
std::size_t numerator_witnesses = 0, denominator_witnesses = 0, hinge_witnesses = 0;
std::size_t native_differences = 0, native_class_differences = 0;
double native_max_absolute = 0, native_max_scaled_error = 0;
std::size_t native_tolerance_exceeded = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Equal(float actual, float expected, unsigned kind, std::size_t index, char const* name) {
  ++fields;
  auto magnitude = Cast<unsigned>(expected) & 0x7fffffffu;
  subnormal_fields += magnitude && magnitude < 0x800000u;
  nonfinite_fields += !std::isfinite(expected);
  if (Ulps(actual, expected)) {
    std::fprintf(stderr, "%s kind%u idx%zu actual %.9g[%08x] expected %.9g[%08x]\n", name, kind,
                 index, actual, Cast<unsigned>(actual), expected, Cast<unsigned>(expected));
    throw std::runtime_error("Source arithmetic differs");
  }
}
void NativeDifference(float ideal, float native) {
  native_differences += Ulps(ideal, native) != 0;
  native_class_differences +=
      std::isnan(ideal) != std::isnan(native) || std::isinf(ideal) != std::isinf(native);
  if (std::isfinite(ideal) && std::isfinite(native)) {
    double error = std::abs(double(ideal) - double(native));
    native_max_absolute = std::max(native_max_absolute, error);
    double scaled = error / std::max(1., std::abs(double(native)));
    native_max_scaled_error = std::max(native_max_scaled_error, scaled);
    native_tolerance_exceeded += scaled > 3e-5;
  }
}
float TestValue(unsigned mode, std::size_t i, std::mt19937& rng) {
  constexpr float edges[]{0.f,
                          -0.f,
                          0x1p-149f,
                          -0x1p-149f,
                          0x1.fffffcp-127f,
                          -0x1.fffffcp-127f,
                          0x1p-126f,
                          -0x1p-126f,
                          0x1.fffffep127f,
                          -0x1.fffffep127f,
                          INFINITY,
                          -INFINITY,
                          NAN,
                          -1.f,
                          -.999999f,
                          -.5f,
                          .5f,
                          1.f,
                          0x1p-26f,
                          -88.7f,
                          88.7f,
                          -104.f,
                          -90.f,
                          20.f,
                          1e-6f};
  if (mode == 1) return std::uniform_real_distribution<float>(-110, 100)(rng);
  if (mode == 2) return Cast<float>(unsigned(rng() % 0x1000000) | (rng() % 2 ? 0x80000000u : 0));
  if (mode == 3) return Cast<float>(unsigned(rng()));
  if (mode == 4) return edges[i % std::size(edges)];
  if (mode == 6) return i % 2 ? -1.f : 1.f;
  return std::uniform_real_distribution<float>(-4, 4)(rng);
}
void Run(unsigned kind, unsigned targets, unsigned mode, bool weighted, std::mt19937& rng) {
  std::size_t rows = mode == 7 ? 4097 : 257, n = rows * targets;
  unsigned label_targets = kind == 9 ? 1 : targets;
  std::vector<float> input(n), labels(rows * label_targets), weights(weighted ? rows : 0),
      alpha(targets);
  float parameter = kind == 4 ? 1.3f : kind == 5 ? .4f : 1.3f;
  if (kind == 4) {
    constexpr float slopes[]{1.3f, 1e-10f, 1e10f, 0x1p-75f, 0x1p75f, -1.3f, 1e-20f, 1.f};
    parameter = slopes[mode];
  }
  if (kind == 6) parameter = mode % 3 == 0 ? 1.f : mode % 3 == 1 ? 1.5f : std::nextafter(2.f, 1.f);
  float scale = kind == 0 || kind == 1 || kind == 3 ? (mode % 2 ? 1.7f : .31f) : 1.f;
  for (unsigned j = 0; j < targets; ++j) alpha[j] = float(j + 1) / (targets + 1);
  for (std::size_t i = 0; i < n; ++i) input[i] = TestValue(mode, i, rng);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    labels[i] = TestValue(mode, i + 7, rng);
    if (mode == 0 || mode == 1 || mode == 2 || mode == 7) {
      if (kind == 1 || kind == 7) labels[i] = i % 2;
      if (kind == 2 || kind == 3 || kind == 5 || kind == 6) labels[i] = std::abs(labels[i]);
    }
    if (i % 19 == 0) labels[i] = 1.f;  // Exercise scale_pos_weight.
    if (mode == 6 && kind == 7) labels[i] = 0x1p-26f;
    if (mode == 5) {
      float pred = input[label_targets == 1 ? i * targets : i];
      labels[i] = kind == 1                             ? Ref::Sigmoid(pred)
                  : kind == 3 || kind == 5 || kind == 6 ? Math::Exp(pred)
                                                        : pred;
      if (kind == 2) input[i] = labels[i] = std::abs(pred);
      if (i % 2) labels[i] = std::nextafter(labels[i], INFINITY);
    }
  }
  for (std::size_t row = 0; row < weights.size(); ++row) {
    constexpr float edge_weights[]{0.f, 0x1p-149f, 1e-40f, 1e-20f, 1.f, 1e20f, 1e30f};
    weights[row] = row % 8 < 7 ? edge_weights[row % 8] : float(rng() % 1000) / 127;
  }
  auto saved_input = input, saved_labels = labels, saved_weights = weights;
  std::vector<GradientPair> expected(n), output(n + 2, GradientPair{-71, -91});
  for (std::size_t i = 0; i < rows; ++i) {
    for (unsigned j = 0; j < targets; ++j) {
      auto index = i * targets + j;
      auto value = Ref::Pair(kind, input, labels, weights, alpha, targets, label_targets, i, j,
                             parameter, scale);
      expected[index] = value;
      auto native = Native::Pair(kind, input, labels, weights, alpha, targets, label_targets, i, j,
                                 parameter, scale);
      NativeDifference(value.GetGrad(), native.GetGrad());
      NativeDifference(value.GetHess(), native.GetHess());
      float pred = input[index], y = labels[i * label_targets + (label_targets == 1 ? 0 : j)];
      float w = weighted ? weights[i] : 1.f;
      if (kind == 2) {
        pred = Math::FMax(pred, -1 + 1e-6);
        auto a = Math::Log1p(pred), b = Math::Log1p(y), base = Add(pred, 1.f);
        float numerator = Add(Add(-a, b), 1.f);
        float reordered = Math::FMax(float(double(Sub(1, Sub(a, b))) / Math::Pow(base, 2)), 1e-6f);
        float squared = Math::FMax(Div(numerator, Mul(base, base)), 1e-6f);
        numerator_witnesses += Ulps(Mul(reordered, w), value.GetHess()) != 0;
        denominator_witnesses += Ulps(Mul(squared, w), value.GetHess()) != 0;
      }
      if (kind == 7) {
        float sign = Sub(Mul(y, 2), 1);
        float g = Mul(pred, sign) < 1 ? Mul(-sign, w) : 0.f;
        float h = Mul(pred, sign) < 1 ? w : std::numeric_limits<float>::min();
        hinge_witnesses += Ulps(g, value.GetGrad()) != 0 || Ulps(h, value.GetHess()) != 0;
      }
    }
  }
  auto before = metal::GetStatistics();
  metal::Gradients(static_cast<metal::Objective>(kind), input, labels, weights, targets, parameter,
                   scale, alpha, {output.data() + 1, n});
  Check(metal::GetStatistics().objective_dispatches == before.objective_dispatches + 1,
        "Missing GPU objective");
  Check(output.front().GetGrad() == -71 && output.front().GetHess() == -91 &&
            output.back().GetGrad() == -71 && output.back().GetHess() == -91,
        "Output guard changed");
  for (std::size_t i = 0; i < n; ++i) {
    if (Ulps(output[i + 1].GetGrad(), expected[i].GetGrad()) ||
        Ulps(output[i + 1].GetHess(), expected[i].GetHess())) {
      auto row = i / targets;
      std::fprintf(
          stderr,
          "targets%u mode%u weighted%d pred%.9g label%.9g weight%.9g parameter%.9g scale%.9g\n",
          targets, mode, weighted, input[i],
          labels[row * label_targets + (label_targets == 1 ? 0 : i % targets)],
          weighted ? weights[row] : 1.f, parameter, scale);
    }
    Equal(output[i + 1].GetGrad(), expected[i].GetGrad(), kind, i, "gradient");
    Equal(output[i + 1].GetHess(), expected[i].GetHess(), kind, i, "Hessian");
  }
  Check(std::memcmp(input.data(), saved_input.data(), n * 4) == 0 &&
            std::memcmp(labels.data(), saved_labels.data(), labels.size() * 4) == 0 &&
            weights == saved_weights,
        "Input changed");
  if (weighted && targets == 3 && mode == 0) {
    std::vector<GradientPair> alias(n + 2, GradientPair{-71, -91});
    auto* storage = reinterpret_cast<float*>(alias.data() + 1);
    std::memcpy(storage, input.data(), n * 4);
    metal::Gradients(static_cast<metal::Objective>(kind), {storage, n}, labels, weights, targets,
                     parameter, scale, alpha, {alias.data() + 1, n});
    for (std::size_t i = 0; i < n; ++i) {
      Equal(alias[i + 1].GetGrad(), expected[i].GetGrad(), kind, i, "alias gradient");
      Equal(alias[i + 1].GetHess(), expected[i].GetHess(), kind, i, "alias Hessian");
    }
    Check(alias.front().GetGrad() == -71 && alias.back().GetHess() == -91, "Alias guard changed");
  }
  ++cases;
}
void Transforms(unsigned targets, unsigned mode, std::mt19937& rng) {
  std::size_t rows = mode == 7 ? 4097 : 257, n = rows * targets;
  std::vector<float> input(n);
  for (std::size_t i = 0; i < n; ++i) input[i] = TestValue(mode, i, rng);
  for (unsigned kind : {0u, 1u, 2u, 3u, 6u, 7u}) {
    auto expected = input;
    for (std::size_t row = 0; row < rows; ++row) {
      if (kind == 7)
        Ref::Quantile(expected, row, targets);
      else
        for (unsigned k = 0; k < targets; ++k) {
          auto i = row * targets + k;
          if (kind == 1) expected[i] = Ref::Sigmoid(input[i]);
          if (kind == 2) expected[i] = Math::Exp(input[i]);
          if (kind == 3) expected[i] = input[i] > 0.0 ? 1.f : 0.f;
          if (kind == 6 && k)
            expected[i] = Add(expected[i - 1], Add(1e-6f, Ref::SoftPlus(input[i])));
        }
    }
    std::vector<float> output(n + 2, -91);
    std::copy(input.begin(), input.end(), output.begin() + 1);
    auto before = metal::GetStatistics();
    metal::TransformPredictions(static_cast<metal::Transform>(kind), {output.data() + 1, n},
                                targets);
    Check(metal::GetStatistics().transform_dispatches == before.transform_dispatches + (kind != 0),
          "Transform dispatch count");
    Check(output.front() == -91 && output.back() == -91, "Transform guard changed");
    for (std::size_t i = 0; i < n; ++i) {
      Equal(output[i + 1], expected[i], kind, i, "transform");
      if (kind == 0 || kind == 7)
        Check(Cast<unsigned>(output[i + 1]) == Cast<unsigned>(expected[i]),
              "Copy/sort must preserve all bits");
    }
    ++cases;
  }
}
int main() {
  try {
    std::mt19937 rng(12681);
    for (unsigned kind : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 9u}) {
      for (unsigned targets : {1u, 3u, 17u})
        for (unsigned mode = 0; mode < 8; ++mode)
          for (bool weighted : {false, true}) Run(kind, targets, mode, weighted, rng);
      std::printf("PASS kind=%u: %zu cases / %zu fields\n", kind, cases, fields);
      std::fflush(stdout);
    }
    for (unsigned targets : {1u, 3u, 17u, 257u})
      for (unsigned mode = 0; mode < 8; ++mode) Transforms(targets, mode, rng);
    auto before_empty = metal::GetStatistics();
    for (unsigned kind : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 9u})
      metal::Gradients(static_cast<metal::Objective>(kind), {}, {}, {}, 1, 1, 1, {}, {});
    for (unsigned kind : {0u, 1u, 2u, 3u, 6u, 7u})
      metal::TransformPredictions(static_cast<metal::Transform>(kind), {}, 1);
    Check(metal::GetStatistics().objective_dispatches == before_empty.objective_dispatches &&
              metal::GetStatistics().transform_dispatches == before_empty.transform_dispatches,
          "Empty objective/transform dispatched");
    Check(denominator_witnesses && hinge_witnesses && subnormal_fields,
          "Missing source-arithmetic witnesses");
    std::printf(
        "PASS %zu cases / %zu exact fields; subnormal=%zu nonfinite=%zu; witnesses "
        "numerator=%zu denominator=%zu hinge=%zu; native-libm differences=%zu "
        "classification=%zu max_abs=%.9g max_scaled=%.9g beyond_3e-5=%zu\n",
        cases, fields, subnormal_fields, nonfinite_fields, numerator_witnesses,
        denominator_witnesses, hinge_witnesses, native_differences, native_class_differences,
        native_max_absolute, native_max_scaled_error, native_tolerance_exceeded);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
