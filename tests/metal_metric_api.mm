// Production metric terms, recursive reductions and GPU label validation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ops.h"
#include "../xgboost/src/common/metal_survival.h"
#include "cuda_metric_reference.h"
using namespace xgboost;
namespace metal = common::metal;
using namespace cuda_metric;
std::size_t cases = 0, fields = 0, exact = 0;
double max_reduction_ratio = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
char const* names[] = {"rmse",
                       "rmsle",
                       "mae",
                       "mape",
                       "logloss",
                       "error",
                       "poisson-nloglik",
                       "gamma-deviance",
                       "gamma-nloglik",
                       "tweedie-nloglik",
                       "mphe",
                       "mlogloss",
                       "merror"};
struct Params {
  std::uint64_t n;
  std::uint32_t targets, kind, weighted, padding;
  float parameter, reserved;
};
struct QParams {
  std::uint64_t n;
  std::uint32_t targets, levels, weighted, expectile;
};
using Pair = std::array<float, 2>;
struct GPU {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  id<MTLComputePipelineState> terms, quantiles, math;
  explicit GPU(char const* path) {
    std::ifstream file(path);
    std::string source{std::istreambuf_iterator<char>(file), {}};
    Check(!source.empty(), "Missing production shader");
    source += R"(
kernel void test_metric_terms(device const uint* p [[buffer(0)]], device const uint* y [[buffer(1)]],
                                device const uint* w [[buffer(2)]], device uint2* out [[buffer(3)]],
                                constant xgb_metric_math::MetricParams& shape [[buffer(4)]],
                                uint row [[thread_position_in_grid]]) {
  bool invalid = false;
  out[row] = xgb_metric_math::MetricTerm(p, y, w, shape, row, invalid);
  if (invalid) out[row] = uint2(0x7fc00000u);
}
kernel void test_quantile_terms(device const float* p [[buffer(0)]], device const float* y [[buffer(1)]],
                                  device const float* w [[buffer(2)]], device const float* a [[buffer(3)]],
                                  device uint2* out [[buffer(4)]], constant QuantileMetricParams& shape [[buffer(5)]],
                                  uint index [[thread_position_in_grid]]) {
  using namespace xgb_metric_math;
  uint row = index / (shape.targets * shape.levels), level = (index / shape.targets) % shape.levels;
  out[index] = QuantileTerm(From(p[index]), From(y[ulong(row) * shape.targets + index % shape.targets]),
                             From(a[level]), From(shape.weighted ? w[row] : 1.f), shape.expectile);
}
kernel void test_metric_math(device const uint* input [[buffer(0)]], device uint* out [[buffer(1)]],
                               uint row [[thread_position_in_grid]]) {
  using namespace xgb_metric_math;
  F x{input[row]};
  out[ulong(row)*5] = Log(x).bits;
  out[ulong(row)*5+1] = Log1p(x).bits;
  out[ulong(row)*5+2] = Exp(x).bits;
  out[ulong(row)*5+3] = Sqrt(x).bits;
  out[ulong(row)*5+4] = LogGamma(x).bits;
})";
    device = MTLCreateSystemDefaultDevice();
    Check(device != nil, "Metal is required");
    auto options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    NSError* error = nil;
    library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                   options:options
                                     error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    auto pipeline = [&](NSString* name) {
      auto result = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                                          error:&error];
      if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
      return result;
    };
    terms = pipeline(@"test_metric_terms");
    quantiles = pipeline(@"test_quantile_terms");
    math = pipeline(@"test_metric_math");
    queue = [device newCommandQueue];
  }
  id<MTLBuffer> Buffer(void const* bytes, std::size_t size) {
    auto result = [device newBufferWithLength:std::max(size, std::size_t{16})
                                      options:MTLResourceStorageModeShared];
    Check(result != nil, "Buffer allocation failed");
    if (bytes && size) std::memcpy(result.contents, bytes, size);
    return result;
  }
  void Finish(id<MTLCommandBuffer> command, id<MTLComputeCommandEncoder> e, std::size_t n) {
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error(command.error.localizedDescription.UTF8String);
  }
  std::vector<Pair> Terms(std::vector<float> const& p, std::vector<float> const& y,
                          std::vector<float> const& w, Params shape) {
    auto pb = Buffer(p.data(), p.size() * 4), yb = Buffer(y.data(), y.size() * 4),
         wb = Buffer(w.data(), w.size() * 4);
    auto output = Buffer(nullptr, shape.n * 8);
    auto command = [queue commandBuffer];
    auto e = [command computeCommandEncoder];
    [e setComputePipelineState:terms];
    [e setBuffer:pb offset:0 atIndex:0];
    [e setBuffer:yb offset:0 atIndex:1];
    [e setBuffer:wb offset:0 atIndex:2];
    [e setBuffer:output offset:0 atIndex:3];
    [e setBytes:&shape length:sizeof(shape) atIndex:4];
    Finish(command, e, shape.n);
    auto begin = static_cast<Pair const*>(output.contents);
    return {begin, begin + shape.n};
  }
};
void Reduction(std::array<double, 2> actual, std::vector<Pair> const& terms,
               metal::Statistics const& before) {
  for (int channel = 0; channel < 2; ++channel) {
    Sum expected;
    for (auto const& pair : terms) expected.Add(pair[channel]);
    double value = expected.Result(),
           bound = 32 * std::numeric_limits<double>::epsilon() * expected.magnitude;
    bool ok = value == actual[channel] || (std::isnan(value) && std::isnan(actual[channel]));
    if (!ok && std::isfinite(value) && std::isfinite(actual[channel])) {
      double ratio = std::abs(actual[channel] - value) / std::max(bound, 1e-300);
      max_reduction_ratio = std::max(max_reduction_ratio, ratio);
      ok = ratio <= 1;
    }
    if (!ok)
      std::fprintf(stderr, "Reduction actual %.17g expected %.17g bound %.17g\n", actual[channel],
                   value, bound);
    Check(ok, "Binary64 metric reduction differs");
  }
  auto after = metal::GetStatistics();
  std::uint64_t passes = 0;
  for (auto blocks = (terms.size() + 255) / 256; blocks > 1; blocks = (blocks + 255) / 256)
    ++passes;
  Check(after.metric_reduction_passes == before.metric_reduction_passes + passes,
        "GPU reduction level count differs");
  Check(after.metric_final_readback_bytes == before.metric_final_readback_bytes + 16,
        "Metric readback was not the final binary64 pair");
  ++cases;
}
void RowField(float actual, float expected, bool algebra) {
  auto ulps = Ulps(actual, expected);
  ++fields;
  if (ulps == 0) ++exact;
  if (ulps > (algebra ? 0u : 4u)) {
    std::fprintf(stderr, "Row actual %.9g (%08x) expected %.9g (%08x), ULP %llu\n", actual,
                 Cast<unsigned>(actual), expected, Cast<unsigned>(expected),
                 (unsigned long long)ulps);
    throw std::runtime_error("CUDA-source row formula differs");
  }
}
void Ordinary(GPU& gpu, unsigned kind, unsigned rows, unsigned mode) {
  std::mt19937 rng(3761 + mode);
  bool multi = kind >= 11;
  unsigned targets = multi ? 7 : 3;
  std::vector<float> y(rows * (multi ? 1 : targets)), p(rows * targets), w(mode ? rows : 0);
  float edge[] = {0, -0.f,  0x1p-149f, -0x1p-149f,      0x1p-126f, 1,        -1,
                  2, -2.5f, 10000,     0x1.fffffep127f, -INFINITY, INFINITY, NAN};
  for (auto& v : y) v = float(rng() % 937 + 1) / 1000;
  for (auto& v : p) v = float(rng() % 937 + 1) / 1000;
  for (auto& v : w) v = float(rng() % 31) / 17;
  if (mode == 1) {
    for (auto& v : y) v = Cast<float>(std::uint32_t(rng()));
    for (auto& v : p) v = Cast<float>(std::uint32_t(rng()));
  }
  if (mode == 2) {
    for (auto& v : y) v = Cast<float>(std::uint32_t((rng() % 0x800000u) | (rng() & 0x80000000u)));
    for (auto& v : p) v = Cast<float>(std::uint32_t((rng() % 0x800000u) | (rng() & 0x80000000u)));
    for (auto& v : w) v *= 1e30f;
  }
  if (mode == 3)
    for (auto& v : w) v *= 1e37f;
  if (mode == 4)
    for (auto& v : w) v = 0;
  if (mode == 5) {
    for (auto& v : y) v = edge[rng() % 14];
    for (auto& v : p) v = edge[rng() % 14];
    for (auto& v : w) v = edge[rng() % 14];
  }
  if (mode == 6) {
    for (auto& v : y) v = std::nextafter(1.f, rng() % 2 ? 2.f : 0.f);
    for (auto& v : p) v = std::nextafter(1.f, rng() % 2 ? 2.f : 0.f);
    for (auto& v : w) v = rng() % 2 ? 1e-40f : -1e-40f;
  }
  if (multi)
    for (auto& v : y) {
      auto k = rng() % targets;
      v = k ? float(k) + .75f : -.75f;
    }
  float parameter = kind == 9 ? 1.3712345f : kind == 10 ? 1.3f : .371234567f;
  Params shape{y.size(), targets, kind, !w.empty(), 0, parameter, 0};
  auto terms = gpu.Terms(p, y, w, shape);
  for (std::size_t i = 0; i < y.size(); ++i) {
    auto row = multi ? i : i / targets;
    float weight = w.empty() ? 1 : w[row], loss;
    if (multi) {
      int label = int(y[i]);
      auto begin = p.data() + row * targets;
      if (kind == 11) {
        // Keep the source comparison explicit for signaling NaNs; an ARM
        // fmaxnm lowering has different sNaN behavior from this conditional.
        volatile bool above = begin[label] > 1e-16f;
        loss = -Log(above ? begin[label] : 1e-16f);
      } else {
        unsigned best = 0;
        for (unsigned k = 1; k < targets; ++k)
          if (begin[k] > begin[best]) best = k;
        loss = float(best != unsigned(label));
      }
    } else
      loss = Loss(kind, y[i], p[i], parameter);
    try {
      RowField(terms[i][0], Mul(loss, weight),
               kind == 0 || kind == 2 || kind == 3 || kind == 5 || kind == 10 || kind == 12);
      RowField(terms[i][1], weight, true);
    } catch (...) {
      std::fprintf(stderr, "kind %u rows %u mode %u index %zu y %.9g p %.9g w %.9g\n", kind, rows,
                   mode, i, y[i], p[i], weight);
      throw;
    }
  }
  auto before = metal::GetStatistics();
  auto actual = metal::MetricSums(names[kind], p, y, w, targets, parameter);
  Reduction(actual, terms, before);
  Check(metal::GetStatistics().metric_label_dispatches == before.metric_label_dispatches + multi,
        "Missing GPU multiclass validation");
}
void Quantiles(GPU& gpu, bool expectile, unsigned rows, unsigned mode) {
  unsigned targets = 3, levels = 7;
  std::mt19937 rng(5871 + mode);
  std::vector<float> y(rows * targets), p(rows * targets * levels), w(rows), a(levels);
  for (auto& v : y) v = float(int(rng() % 1000) - 500) / 123;
  for (auto& v : p) v = float(int(rng() % 1000) - 500) / 123;
  for (auto& v : w) v = float(rng() % 31) / 17;
  for (unsigned i = 0; i < levels; ++i) a[i] = float(i + 1) / (levels + 1);
  if (mode == 1) {
    for (auto& v : y) v *= 1e-40f;
    for (auto& v : p) v *= 1e-40f;
    for (auto& v : w) v *= 1e30f;
  }
  if (mode == 2)
    for (auto& v : a) v *= 1e-40f;
  if (mode == 3)
    for (auto& v : w) v *= 1e37f;
  if (mode == 4) {
    p[0] = NAN;
    p.back() = INFINITY;
  }
  QParams shape{p.size(), targets, levels, 1, expectile};
  auto pb = gpu.Buffer(p.data(), p.size() * 4), yb = gpu.Buffer(y.data(), y.size() * 4),
       wb = gpu.Buffer(w.data(), w.size() * 4), ab = gpu.Buffer(a.data(), a.size() * 4);
  auto out = gpu.Buffer(nullptr, p.size() * 8);
  auto command = [gpu.queue commandBuffer];
  auto e = [command computeCommandEncoder];
  [e setComputePipelineState:gpu.quantiles];
  [e setBuffer:pb offset:0 atIndex:0];
  [e setBuffer:yb offset:0 atIndex:1];
  [e setBuffer:wb offset:0 atIndex:2];
  [e setBuffer:ab offset:0 atIndex:3];
  [e setBuffer:out offset:0 atIndex:4];
  [e setBytes:&shape length:sizeof(shape) atIndex:5];
  gpu.Finish(command, e, p.size());
  auto begin = static_cast<Pair const*>(out.contents);
  std::vector<Pair> terms(begin, begin + p.size());
  for (std::size_t i = 0; i < p.size(); ++i) {
    auto row = i / (targets * levels), level = (i / targets) % levels;
    RowField(terms[i][0],
             Quantile(expectile, p[i], y[row * targets + i % targets], a[level], w[row]), true);
    RowField(terms[i][1], w[row], true);
  }
  auto before = metal::GetStatistics();
  auto actual = metal::QuantileMetricSums(expectile, p, y, w, a, targets);
  Reduction(actual, terms, before);
}
void Math(GPU& gpu) {
  std::vector<float> values{0,     -0.f, 0x1p-149f, -0x1p-149f,  0x1p-126f, 1,         2,
                            -1,    -2,   -2.5f,     -3.1435809f, .75f,      1.25f,     1.75f,
                            2.25f, 16,   88.7f,     -104,        INFINITY,  -INFINITY, NAN};
  auto anchors = values;
  for (auto x : anchors)
    if (std::isfinite(x)) {
      values.push_back(std::nextafter(x, INFINITY));
      values.push_back(std::nextafter(x, -INFINITY));
    }
  std::mt19937 rng(3721);
  for (int i = 0; i < 80000; ++i) values.push_back(Cast<float>(std::uint32_t(rng())));
  for (int i = 0; i < 20000; ++i) values.push_back(float(int(rng() % 200000) - 110000) / 1000);
  auto input = gpu.Buffer(values.data(), values.size() * 4),
       out = gpu.Buffer(nullptr, values.size() * 20);
  auto command = [gpu.queue commandBuffer];
  auto e = [command computeCommandEncoder];
  [e setComputePipelineState:gpu.math];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:out offset:0 atIndex:1];
  gpu.Finish(command, e, values.size());
  auto output = static_cast<float const*>(out.contents);
  std::uint64_t maximum[5]{}, limits[5]{1, 1, 1, 0, 6};
  std::size_t failures = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    float x = values[i],
          expected[]{Log(x), float(std::log1p(double(x))), Exp(x), Sqrt(x), Gamma(x)};
    for (int k = 0; k < 5; ++k) {
      auto ulps = Ulps(output[i * 5 + k], expected[k]);
      maximum[k] = std::max(maximum[k], ulps);
      if (ulps > limits[k] && failures++ < 10)
        std::fprintf(stderr, "Math k%d input %.9g actual %.9g expected %.9g ULP %llu\n", k, x,
                     output[i * 5 + k], expected[k], (unsigned long long)ulps);
      if (x == 0 && (k == 1 || k == 3))
        Check(Cast<unsigned>(output[i * 5 + k]) == Cast<unsigned>(expected[k]), "Lost signed zero");
    }
  }
  std::printf("Metric math: %zu inputs / %zu fields; max log/log1p/exp/sqrt/lgamma ULP "
              "%llu/%llu/%llu/%llu/%llu\n",
              values.size(), values.size() * 5, (unsigned long long)maximum[0],
              (unsigned long long)maximum[1], (unsigned long long)maximum[2],
              (unsigned long long)maximum[3], (unsigned long long)maximum[4]);
  Check(failures == 0, "Metric transcendental budget failed");
}
void Survival() {
  constexpr std::size_t n = 65537;
  std::vector<float> p(n, 0), y(n, 1), w(n);
  for (unsigned distribution = 0; distribution < 3; ++distribution)
    for (float weight : {0.f, 1e-40f, 1.f, 1e37f}) {
      std::fill(w.begin(), w.end(), weight);
      for (bool accuracy : {false, true}) {
        auto before = metal::GetStatistics();
        auto actual = metal::SurvivalMetricSums(distribution, 1, accuracy, p, y, y, w);
        double loss = accuracy            ? 1
                      : distribution == 0 ? .5 * std::log(2 * std::acos(-1.))
                      : distribution == 1 ? std::log(4.)
                                          : 1;
        double expected = loss * double(weight) * n;
        Check(actual[1] == double(weight) * n, "Survival weight sum differs");
        Check(std::abs(actual[0] - expected) <= 2e-14 * std::max(std::abs(expected), 1e-300),
              "Survival analytic sum differs");
        auto after = metal::GetStatistics();
        Check(after.metric_final_readback_bytes == before.metric_final_readback_bytes + 16 &&
                  after.metric_reduction_passes == before.metric_reduction_passes + 2,
              "Survival final reduction missing");
        ++cases;
      }
    }
}
void Guards() {
  // AFT's source lower<=0 guard is ordered. NaN bounds reach the probability
  // expression, then fmax clamps NaN likelihood to epsilon. Fully open bounds
  // still have unit likelihood, even with a NaN prediction.
  for (unsigned distribution = 0; distribution < 3; ++distribution) {
    for (float weight : {0.f, 1.f}) {
      std::vector<float> p{0, 0, 0, 0, NAN}, lower{NAN, NAN, 1, NAN, 0},
          upper{1, INFINITY, NAN, NAN, INFINITY}, w(5, weight);
      auto before = metal::GetStatistics();
      auto sum = metal::SurvivalMetricSums(distribution, 1, false, p, lower, upper, w);
      auto expected = -4 * std::log(1e-12) * weight;
      Check(std::abs(sum[0] - expected) <= 2e-14 * std::max(1., std::abs(expected)) &&
                sum[1] == 5 * weight,
            "AFT ordered-NaN metric guard differs from CUDA source");
      Check(metal::GetStatistics().metric_final_readback_bytes ==
                before.metric_final_readback_bytes + 16,
            "AFT NaN-bound metric did not use GPU sums");
      ++cases;
    }
  }
  for (auto name : {"mlogloss", "merror"})
    for (float label : {-1.f, 3.f, NAN, INFINITY, -INFINITY, 2147483648.f}) {
      std::vector<float> p{.2, .3, .5}, y{label};
      bool bad = false;
      auto before = metal::GetStatistics();
      try {
        metal::MetricSums(name, p, y, {}, 3);
      } catch (dmlc::Error const&) {
        bad = true;
      }
      Check(bad && metal::GetStatistics().metric_label_dispatches ==
                       before.metric_label_dispatches + 1,
            "Invalid label did not fail on GPU");
    }
  auto before = metal::GetStatistics();
  for (auto name : names)
    Check(metal::MetricSums(name, {}, {}, {}, 1) == std::array<double, 2>{0, 0},
          "Empty metric differs");
  Check(metal::GetStatistics().metric_dispatches == before.metric_dispatches,
        "Empty metric dispatched");
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production shader");
      GPU gpu(argv[1]);
      Math(gpu);
      for (unsigned kind = 0; kind < 13; ++kind) {
        for (unsigned n : {1u, 257u, 4097u})
          for (unsigned mode = 0; mode < 7; ++mode) Ordinary(gpu, kind, n, mode);
        Ordinary(gpu, kind, 65537, 0);
        std::printf("Checked %s; %zu cases / %zu row fields\n", names[kind], cases, fields);
        std::fflush(stdout);
      }
      Ordinary(gpu, 2, 262145, 3);
      for (bool expectile : {false, true}) {
        for (unsigned n : {1u, 257u, 4097u})
          for (unsigned mode = 0; mode < 5; ++mode) Quantiles(gpu, expectile, n, mode);
        Quantiles(gpu, expectile, 65537, 0);
      }
      Survival();
      Guards();
      std::printf("PASS CUDA metric formulas/reductions: %zu cases / %zu fields (%zu exact); "
                  "maximum reduction-bound ratio %.9g\n",
                  cases, fields, exact, max_reduction_ratio);
      return 0;
    } catch (std::exception const& error) {
      std::fprintf(stderr, "Metric FAIL: %s\n", error.what());
      return 1;
    }
  }
}
