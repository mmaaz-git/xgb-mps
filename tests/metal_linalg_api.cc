// CUDA-source class-run packing, FP32 reductions and vector cast boundaries.
// Reduction accuracy is checked against independent long-double sums. CUB's
// architecture-specific association is not asserted by these tests.
#include <algorithm>
#include <array>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/linalg_op.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_linalg.h"
#include "../xgboost/src/common/metal_reduction.h"
#include "../xgboost/src/common/stats.h"
#include "cuda_metric_reference.h"
using namespace xgboost;
namespace metal = common::metal;
using namespace cuda_metric;
std::size_t cases = 0, fields = 0, packed_witnesses = 0;
double worst_bound = 0;
std::array<std::uint64_t, 4> max_vector_ulp{};
std::size_t double_scale_witnesses = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
auto Vector(std::vector<float>& v, std::size_t n, std::size_t stride = 1) {
  std::size_t shape[1]{n}, strides[1]{stride};
  return linalg::VectorView<float>{common::Span<float>{v}, shape, strides, DeviceOrd::CPU()};
}
auto Matrix(std::vector<float> const& v, std::size_t n, std::size_t stride = 1) {
  std::size_t shape[2]{n, 1}, strides[2]{stride, 1};
  return linalg::MatrixView<float const>{common::Span<float const>{v}, shape, strides,
                                         DeviceOrd::CPU()};
}
void Bound(float actual, long double expected, long double magnitude, std::size_t n, bool exact) {
  ++fields;
  if (std::isnan(double(expected))) {
    Check(std::isnan(actual), "Expected NaN reduction");
    return;
  }
  if (std::isinf(float(expected))) {
    Check(actual == float(expected), "Expected infinite reduction");
    return;
  }
  if (exact) {
    Check(actual == float(expected), "Exact integer/power-of-two reduction differs");
    return;
  }
  // Each term's path has at most ceil(n/256) local adds + 8 tree additions.
  double k = (n + 255) / 256 + 8;
  double eps = std::numeric_limits<float>::epsilon();
  double bound = k * eps / (1 - k * eps) * double(magnitude);
  double error = std::abs(double(actual) - double(expected));
  double ratio = error / std::max(bound, 1e-300);
  worst_bound = std::max(worst_bound, ratio);
  if (ratio > 1 || !std::isfinite(actual))
    std::fprintf(stderr, "Sum actual %.9g expected %.17g bound %.9g n%zu\n", actual,
                 double(expected), bound, n);
  Check(std::isfinite(actual) && ratio <= 1, "FP32 sum exceeds association bound");
}
void Histogram(Context const& ctx, std::size_t n, unsigned bins, unsigned mode, std::mt19937& rng) {
  std::size_t stride = 1 + mode % 3;
  std::vector<float> labels(n * stride + 4, -71), weights(n), output(bins * stride + 4, -91);
  for (std::size_t i = 0; i < n; ++i) {
    unsigned label = rng() % bins;
    if (mode % 2) label = (label / 3) * 3;
    labels[i * stride] = label;
    if (!label && i % 2) labels[i * stride] = -0.f;
    weights[i] = 1 + rng() % 7;
    if (mode == 2) weights[i] = (rng() % 13) * 0x1p-140f;
    if (mode == 3) weights[i] = float(.1 + (rng() % 1000) / 23.);
    if (mode == 4) weights[i] = 0;
    if (mode == 5) weights[i] = std::ldexp(float(1 + rng() % 7), int(rng() % 101) - 50);
    if (mode == 6) weights[i] = 1e37f;
  }
  auto saved_labels = labels;
  auto saved_weights = weights;
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](auto a, auto b) { return labels[a * stride] < labels[b * stride]; });
  std::vector<long double> expected(bins, 0), magnitude(bins, 0);
  std::vector<std::size_t> counts(bins, 0);
  unsigned run = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (i && labels[order[i] * stride] != labels[order[i - 1] * stride]) ++run;
    float weight = mode == 0 ? 1.f : weights[order[i]];
    expected[run] += weight;
    magnitude[run] += std::abs(weight);
    counts[run]++;
    if (run != unsigned(labels[order[i] * stride])) ++packed_witnesses;
  }
  auto before = metal::GetStatistics();
  auto optional = common::OptionalWeights{mode ? common::Span<float const>{weights}
                                               : common::Span<float const>{}};
  linalg::SmallHistogram(&ctx, Matrix(labels, n, stride), optional, Vector(output, bins, stride));
  auto after = metal::GetStatistics();
  Check(after.small_histogram_dispatches == before.small_histogram_dispatches + 1,
        "Histogram missed GPU dispatch");
  Check(after.radix_sort_dispatches == before.radix_sort_dispatches + bool(n),
        "Histogram missed GPU sort");
  for (unsigned b = 0; b < bins; ++b)
    Bound(output[b * stride], expected[b], magnitude[b], counts[b], mode <= 2 || mode == 4);
  for (std::size_t i = 0; i < output.size(); ++i)
    if (i % stride || i / stride >= bins) Check(output[i] == -91, "Histogram output guard changed");
  Check(labels == saved_labels && weights == saved_weights, "Histogram modified input");
  ++cases;
}
void Vectors(Context const& ctx, std::mt19937& rng) {
  std::vector<float> input{0,
                           -0.f,
                           1,
                           -1,
                           INFINITY,
                           -INFINITY,
                           NAN,
                           1e-40f,
                           -1e-40f,
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::min()};
  for (std::size_t i = 0; i < 65537; ++i) input.push_back(Cast<float>(std::uint32_t(rng())));
  for (unsigned stride : {1u, 3u}) {
    for (unsigned op = 0; op < 4; ++op) {
      for (double parameter :
           {0., -0., 1., -1., 1. / 3, 1e-40, 1e-300, 1e300, double(INFINITY), double(NAN)}) {
        std::vector<float> values(input.size() * stride + 3, -81), expected = values;
        for (std::size_t i = 0; i < input.size(); ++i) {
          float v = input[i], result;
          if (op < 2)
            result = float(double(v) * (op ? 1. / parameter : parameter));
          else if (op == 2)
            result = float(std::log(double(Add(v, float(parameter)))));
          else
            result = Sub(v, float(parameter));
          if (op == 0 && std::isfinite(result) && std::isfinite(Mul(v, float(parameter))) &&
              result != Mul(v, float(parameter)))
            ++double_scale_witnesses;
          values[i * stride] = v;
          expected[i * stride] = result;
        }
        auto before = metal::GetStatistics();
        auto view = Vector(values, input.size(), stride);
        if (op == 0) linalg::VecScaMul(&ctx, view, parameter);
        if (op == 1) linalg::VecScaDiv(&ctx, view, parameter);
        if (op == 2) linalg::LogE(&ctx, view, float(parameter));
        if (op == 3) metal::VectorArithmetic(metal::VectorOp::kSubtract, view, parameter);
        Check(metal::GetStatistics().vector_arithmetic_dispatches ==
                  before.vector_arithmetic_dispatches + 1,
              "Vector operation missed GPU dispatch");
        for (std::size_t i = 0; i < values.size(); ++i) {
          auto ulp = Ulps(values[i], expected[i]);
          max_vector_ulp[op] = std::max(max_vector_ulp[op], ulp);
          if (ulp > (op == 2 ? 1 : 0))
            std::fprintf(stderr,
                         "Vector op%u parameter%.17g index%zu actual%.9g expected%.9g ULP%llu\n",
                         op, parameter, i, values[i], expected[i], (unsigned long long)ulp);
          Check(ulp <= (op == 2 ? 1 : 0), "Vector arithmetic/source cast mismatch");
          ++fields;
        }
        ++cases;
      }
    }
  }
}
void Means(Context const& ctx, std::mt19937& rng) {
  for (std::size_t n : {1ul, 32ul, 257ul, 4097ul, 65537ul, 262145ul})
    for (unsigned mode = 0; mode < 5; ++mode) {
      std::vector<float> input(n * 3 + 4, -93);
      long double sum = 0, magnitude = 0;
      for (std::size_t i = 0; i < n; ++i) {
        float v = float(int(rng() % 1000) - 500) / 7;
        if (mode == 1) v = float(int(rng() % 9) - 4) * 0x1p-130f;
        if (mode == 2) v = 13.f;
        if (mode == 3) v = std::ldexp(v, int(rng() % 201) - 100);
        if (mode == 4) v = 0;
        input[i * 3] = v;
        float term = Div(v, float(n));
        sum += term;
        magnitude += std::abs(term);
      }
      auto before = metal::GetStatistics();
      linalg::Vector<float> output;
      auto matrix = Matrix(input, n, 3);
      common::Mean(&ctx, matrix.Slice(linalg::All(), 0), &output);
      auto after = metal::GetStatistics();
      Check(after.float_mean_dispatches == before.float_mean_dispatches + 1 &&
                after.mean_dispatches == before.mean_dispatches + 1,
            "FP32 mean missed GPU dispatch");
      Bound(output.HostView()(0), sum, magnitude, n, mode == 4);
      ++cases;
    }
  for (unsigned targets : {3u, 7u}) {
    std::size_t n = 257, stride0 = 2 * targets + 1, stride1 = 2;
    std::vector<float> input(n * stride0 + 4, -75), output(targets * 3 + 4, -81);
    std::vector<long double> sums(targets, 0), magnitudes(targets, 0);
    for (std::size_t r = 0; r < n; ++r)
      for (unsigned t = 0; t < targets; ++t) {
        float v = float(int(rng() % 500) - 250) / 7;
        input[r * stride0 + t * stride1] = v;
        float term = Div(v, float(n));
        sums[t] += term;
        magnitudes[t] += std::abs(term);
      }
    std::size_t shape[2]{n, targets}, strides[2]{stride0, stride1};
    linalg::MatrixView<float const> view{common::Span<float const>{input}, shape, strides,
                                         DeviceOrd::CPU()};
    metal::Mean(metal::MeanKind::kFloat32, view, {}, Vector(output, targets, 3));
    for (unsigned t = 0; t < targets; ++t) Bound(output[t * 3], sums[t], magnitudes[t], n, false);
    for (std::size_t i = 0; i < output.size(); ++i)
      if (i % 3 || i / 3 >= targets) Check(output[i] == -81, "Mean output guard changed");
    ++cases;
  }
  // A transparent association witness: this tree's FP32 cancellation loses the
  // small term, whereas the former binary64 sum would retain it. Not a CUB oracle.
  std::vector<float> input(256, 0);
  input[0] = 0x1p24f;
  input[64] = -0x1p24f;
  input[128] = 1;
  linalg::Vector<float> out;
  common::Mean(&ctx, Matrix(input, input.size()).Slice(linalg::All(), 0), &out);
  Check(out.HostView()(0) == 0, "Mean still uses wider accumulation");
  ++cases;
}
void Guards(Context const& ctx) {
  for (auto labels :
       {std::vector<float>{0, 1, 2, 3}, std::vector<float>{0, NAN}, std::vector<float>{INFINITY}}) {
    std::vector<float> output(3, -77);
    bool caught = false;
    try {
      linalg::SmallHistogram(&ctx, Matrix(labels, labels.size()), common::OptionalWeights{1.f},
                             Vector(output, 3));
    } catch (std::exception const&) {
      caught = true;
    }
    Check(caught && output == std::vector<float>(3, -77),
          "Invalid run/label guard failed or modified output");
    ++cases;
  }
  std::vector<float> labels{7, 7, 9, 9, 9}, output(4, -1);
  linalg::SmallHistogram(&ctx, Matrix(labels, labels.size()), common::OptionalWeights{.5f},
                         Vector(output, 4));
  Check(output == std::vector<float>({1, 1.5f, 0, 0}), "Source packed runs/default weights differ");
  ++cases;
  std::vector<float> empty, marker{91};
  auto before = metal::GetStatistics();
  linalg::VecScaMul(&ctx, Vector(empty, 0), 2.);
  linalg::SmallHistogram(&ctx, Matrix(empty, 0), common::OptionalWeights{1.f}, Vector(empty, 0));
  Check(
      metal::GetStatistics().vector_arithmetic_dispatches == before.vector_arithmetic_dispatches &&
          metal::GetStatistics().small_histogram_dispatches == before.small_histogram_dispatches,
      "Empty operation dispatched");
}
int main() {
  try {
    Context ctx;
    ctx.Init({{"device", "mps"}, {"nthread", "2"}});
    std::mt19937 rng(712);
    for (std::size_t n : {0ul, 1ul, 257ul, 4097ul, 65537ul}) {
      for (unsigned bins : {1u, 3u, 33u, 257u})
        for (unsigned mode = 0; mode < 7; ++mode) Histogram(ctx, n, bins, mode, rng);
      std::printf("Histogram checked rows%zu: %zu cases\n", n, cases);
      std::fflush(stdout);
    }
    Histogram(ctx, 262145, 1025, 3, rng);
    Vectors(ctx, rng);
    Means(ctx, rng);
    Guards(ctx);
    std::printf(
        "PASS CUDA linalg: %zu cases / %zu fields / %zu packed-class witnesses; maximum "
        "reduction-bound ratio %.9g\n",
        cases, fields, packed_witnesses, worst_bound);
    Check(double_scale_witnesses > 0, "Corpus must distinguish double scalar multiplication");
    std::printf(
        "Vector max scale/div/log/sub ULP %llu/%llu/%llu/%llu; %zu binary64-scalar witnesses\n",
        (unsigned long long)max_vector_ulp[0], (unsigned long long)max_vector_ulp[1],
        (unsigned long long)max_vector_ulp[2], (unsigned long long)max_vector_ulp[3],
        double_scale_witnesses);
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "Linalg FAIL: %s\n", error.what());
    return 1;
  }
}
