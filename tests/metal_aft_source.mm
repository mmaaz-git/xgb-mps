// Replay actual upstream AFT expressions using independently checked GPU math
// intermediates. Separately retain ordinary CPU-equivalence checks and report
// all sensitive-tail differences; source replay covers those rows too.
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
#include "../results/cuda_aft_reference_generated.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_survival.h"
#include "../xgboost/src/common/survival_util.h"
#include "cuda_metric_reference.h"
using namespace xgboost;
namespace metal = common::metal;
using cuda_metric::Cast;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
bool Same(double a, double b) {
  return Cast<std::uint64_t>(a) == Cast<std::uint64_t>(b) || (std::isnan(a) && std::isnan(b));
}
std::uint64_t Ulps(double a, double b) {
  if (a == b || (std::isnan(a) && std::isnan(b))) return 0;
  if (!std::isfinite(a) || !std::isfinite(b)) return UINT64_MAX;
  auto key = [](double x) {
    auto bits = Cast<std::uint64_t>(x);
    return bits >> 63 ? ~bits : bits ^ (std::uint64_t{1} << 63);
  };
  auto aa = key(a), bb = key(b);
  return std::max(aa, bb) - std::min(aa, bb);
}
bool Close(float a, float b) {
  return a == b || (std::isnan(a) && std::isnan(b)) ||
         (std::isfinite(a) && std::isfinite(b) &&
          std::abs(double(a) - b) <= 1e-6 + 2e-5 * std::abs(double(b)));
}
struct Trace {
  double log[2], z[2], probability[2][4], math[2][4], gradient, hessian, likelihood;
};
static_assert(sizeof(Trace) == 23 * 8);
struct MathReplay {
  inline static double const* values;
  static double Exp(double x) {
    if (Same(x, values[0])) return values[1];
    Check(Same(x, values[2]), "Source exponential argument differs from GPU");
    return values[3];
  }
  static double Erf(double x) {
    Check(Same(x, values[2]), "Source erf argument differs from GPU");
    return values[3];
  }
};
template <typename Original>
struct DistributionReplay {
  using Source = Original;
  inline static Trace const* trace;
  inline static std::array<double, 2> bounds;
  static double Log(double x) {
    if (Same(x, bounds[0])) return trace->log[0];
    Check(Same(x, bounds[1]), "Unknown source log argument");
    return trace->log[1];
  }
  static double Field(double z, unsigned field) {
    if (Same(z, trace->z[0])) return trace->probability[0][field];
    Check(Same(z, trace->z[1]), "Source z-score differs from GPU");
    return trace->probability[1][field];
  }
  static double PDF(double z) { return Field(z, 0); }
  static double CDF(double z) { return Field(z, 1); }
  static double GradPDF(double z) { return Field(z, 2); }
  static double HessPDF(double z) { return Field(z, 3); }
};
struct GPU {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
  explicit GPU(char const* path) {
    std::ifstream file(path);
    std::string source{std::istreambuf_iterator<char>(file), {}};
    Check(!source.empty(), "Missing production shader");
    source += R"(
struct TestAFTTrace { ulong logs[2], z[2], probability[2][4], math[2][4], gradient, hessian, likelihood; };
kernel void test_aft_source(device const float4* input [[buffer(0)]],
                            device TestAFTTrace* output [[buffer(1)]],
                            constant uint2& shape [[buffer(2)]], uint row [[thread_position_in_grid]]) {
  using namespace xgb_survival;
  float4 in = input[row];
  D l = From(in.x), u = From(in.y), prediction = From(in.z), sigma = From(in.w);
  auto value = Evaluate(shape.x, l, u, prediction, sigma);
  output[row].gradient = value.gradient.bits;
  output[row].hessian = value.hessian.bits;
  output[row].likelihood = value.likelihood.bits;
  for (uint i = 0; i < 2; ++i) {
    D logarithm = Log(i ? u : l), z = (logarithm - prediction) / sigma;
    auto p = Probability(shape.x, z);
    output[row].logs[i] = logarithm.bits;
    output[row].z[i] = z.bits;
    output[row].probability[i][0] = p.pdf.bits;
    output[row].probability[i][1] = p.cdf.bits;
    output[row].probability[i][2] = p.grad.bits;
    output[row].probability[i][3] = p.hess.bits;
    D a = shape.x ? z : -z * z / two, e = Exp(a);
    D b = shape.x ? -e : z / D{0x3ff6a09e667f3bcdul};
    output[row].math[i][0] = a.bits;
    output[row].math[i][1] = e.bits;
    output[row].math[i][2] = b.bits;
    output[row].math[i][3] = (shape.x ? Exp(b) : Erf(b)).bits;
  }
})";
    device = MTLCreateSystemDefaultDevice();
    Check(device != nil, "Metal is required");
    auto options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    NSError* error = nil;
    auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                        options:options
                                          error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"test_aft_source"]
                                              error:&error];
    Check(pipeline != nil, "Missing diagnostic pipeline");
    queue = [device newCommandQueue];
  }
  std::vector<Trace> Run(unsigned distribution, float scale, std::vector<float> const& lo,
                         std::vector<float> const& hi, std::vector<float> const& pred) {
    std::vector<std::array<float, 4>> input(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) input[i] = {lo[i], hi[i], pred[i], scale};
    auto in = [device newBufferWithBytes:input.data()
                                  length:input.size() * sizeof(input[0])
                                 options:MTLResourceStorageModeShared];
    auto out = [device newBufferWithLength:lo.size() * sizeof(Trace)
                                   options:MTLResourceStorageModeShared];
    Check(in && out, "Diagnostic buffer allocation failed");
    auto command = [queue commandBuffer];
    auto e = [command computeCommandEncoder];
    std::array<unsigned, 2> shape{distribution, 0};
    [e setComputePipelineState:pipeline];
    [e setBuffer:in offset:0 atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    [e setBytes:shape.data() length:sizeof(shape) atIndex:2];
    [e dispatchThreads:MTLSizeMake(lo.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error(command.error.localizedDescription.UTF8String);
    auto begin = static_cast<Trace const*>(out.contents);
    return {begin, begin + lo.size()};
  }
};
std::size_t cases = 0, rows = 0, math_fields = 0, source_fields = 0, sensitive = 0,
            cpu_differences = 0, production_differences = 0;
std::array<std::uint64_t, 3> max_ulp{};
void Primitive(double actual, double expected, unsigned kind) {
  auto ulp = Ulps(actual, expected);
  max_ulp[kind] = std::max(max_ulp[kind], ulp);
  ++math_fields;
  if (ulp > 4)
    std::fprintf(stderr, "Math op%u actual %.17g ref %.17g ULP %llu\n", kind, actual, expected,
                 (unsigned long long)ulp);
  Check(ulp <= 4, "AFT primitive differs by more than four binary64 ULP");
}
template <typename Source, typename Probability>
void Verify(GPU& gpu, unsigned kind, float scale, std::vector<float> const& lo,
            std::vector<float> const& hi, std::vector<float> const& pred,
            std::vector<float> const& weights, bool inspect) {
  auto traces = gpu.Run(kind, scale, lo, hi, pred);
  std::vector<GradientPair> actual(lo.size());
  auto before = metal::GetStatistics();
  metal::SurvivalGradients(kind, scale, pred, lo, hi, weights, actual);
  auto after = metal::GetStatistics();
  Check(after.survival_gradient_dispatches == before.survival_gradient_dispatches + 1,
        "Missing production AFT GPU dispatch");
  if (!inspect)
    Check(after.survival_cpu_corrections == before.survival_cpu_corrections &&
              after.survival_cpu_time_ns == before.survival_cpu_time_ns,
          "CPU AFT correction remains");
  using Replay = DistributionReplay<Source>;
  for (std::size_t i = 0; i < lo.size(); ++i) {
    auto const& trace = traces[i];
    for (unsigned side = 0; side < 2; ++side) {
      Primitive(trace.log[side], std::log(double(side ? hi[i] : lo[i])), 0);
      auto m = trace.math[side];
      Primitive(m[1], std::exp(m[0]), 1);
      Primitive(m[3], kind ? std::exp(m[2]) : std::erf(m[2]), kind ? 1 : 2);
      MathReplay::values = m;
      double z = trace.z[side];
      std::array<double, 4> probability{Probability::PDF(z), Probability::CDF(z),
                                        Probability::GradPDF(z), Probability::HessPDF(z)};
      for (unsigned k = 0; k < 4; ++k) {
        Check(Same(trace.probability[side][k], probability[k]),
              "GPU probability differs from upstream expressions");
        ++source_fields;
      }
    }
    Replay::trace = &trace;
    Replay::bounds = {lo[i], hi[i]};
    double g = aft_replay::AFTLoss<Replay>::Gradient(lo[i], hi[i], pred[i], scale),
           h = aft_replay::AFTLoss<Replay>::Hessian(lo[i], hi[i], pred[i], scale);
    if (!Same(g, trace.gradient) || !Same(h, trace.hessian))
      std::fprintf(stderr,
                   "Source kind%u scale%g row%zu lo%g hi%g pred%g: g %.17g/%.17g h %.17g/%.17g\n",
                   kind, scale, i, lo[i], hi[i], pred[i], trace.gradient, g, trace.hessian, h);
    Check(Same(g, trace.gradient) && Same(h, trace.hessian),
          "Raw GPU gradient differs from upstream expressions");
    source_fields += 2;
    float w = weights.empty() ? 1.f : weights[i];
    std::array<float, 2> ref{cuda_metric::Mul(float(g), w), cuda_metric::Mul(float(h), w)},
        got{actual[i].GetGrad(), actual[i].GetHess()},
        cpu{cuda_metric::Mul(float(common::AFTLoss<Source>::Gradient(lo[i], hi[i], pred[i], scale)),
                             w),
            cuda_metric::Mul(float(common::AFTLoss<Source>::Hessian(lo[i], hi[i], pred[i], scale)),
                             w)};
    bool tail = lo[i] != hi[i] && !(trace.likelihood > 1e-5f);
    sensitive += tail;
    for (unsigned k = 0; k < 2; ++k) {
      bool equal = Same(double(got[k]), double(ref[k]));
      production_differences += !equal;
      if (!inspect) Check(equal, "Production gradient differs from raw GPU source result");
      bool close = Close(ref[k], cpu[k]);
      cpu_differences += !close;
      if (!tail) Check(close, "Well-conditioned AFT CPU tolerance exceeded");
    }
  }
  ++cases;
  rows += lo.size();
}
template <typename Source, typename Probability>
void Corpus(GPU& gpu, unsigned kind, std::mt19937& rng, bool inspect) {
  std::uniform_real_distribution<float> uniform(0, 1);
  for (float scale : {.1f, .5f, 1.f, 2.f, 10.f}) {
    for (unsigned mode = 0; mode < 5; ++mode) {
      std::vector<float> lo(4097), hi(4097), p(4097), w(4097);
      for (std::size_t i = 0; i < lo.size(); ++i) {
        float y = std::exp(-30.f + 60.f * uniform(rng));
        lo[i] = hi[i] = y;
        switch (i % 7) {
          case 1:
            hi[i] = INFINITY;
            break;
          case 2:
            lo[i] = 0;
            break;
          case 3:
            hi[i] = y * std::exp(.01f + 3.f * uniform(rng));
            break;
          case 4:
            hi[i] = std::nextafter(y, INFINITY);
            break;
          case 5:
            lo[i] = 0;
            hi[i] = INFINITY;
            break;
        }
        if (mode == 4) lo[i] = hi[i] = y;
        p[i] = std::log(y) + scale * (-50.f + 100.f * uniform(rng));
        if (i % 101 == 0) p[i] = i % 2 ? 1000 : -1000;
        w[i] = mode == 2 ? 0 : .1f + 3.f * uniform(rng);
        if (mode == 3) w[i] *= 1e-40f;
      }
      if (!mode) w.clear();
      Verify<Source, Probability>(gpu, kind, scale, lo, hi, p, w, inspect);
    }
    std::printf("Source checked distribution%u scale%g: %zu rows\n", kind, scale, rows);
    std::fflush(stdout);
  }
  // Cartesian domain guards, NaN/Inf, signed zero, invalid/reversed bounds and
  // invalid scales. These are direct API/source checks, not recommendations.
  std::vector<float> lo, hi, p, w;
  for (float l : {NAN, -INFINITY, -1.f, -0.f, 0.f, 1e-40f, 1.f, INFINITY})
    for (float u : {NAN, -INFINITY, -1.f, -0.f, 0.f, 1e-40f, 1.f, INFINITY})
      for (float v : {NAN, -INFINITY, -1000.f, -0.f, 0.f, 1000.f, INFINITY}) {
        lo.push_back(l);
        hi.push_back(u);
        p.push_back(v);
        w.push_back(w.size() % 3 == 0 ? 0.f : w.size() % 3 == 1 ? 1.f : 1e-40f);
      }
  for (float scale : {.1f, 1.f, 0.f, -1.f, 1e-40f, INFINITY, NAN})
    Verify<Source, Probability>(gpu, kind, scale, lo, hi, p, w, inspect);
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2 || argc == 3, "Pass shader path [--inspect]");
      bool inspect = argc == 3 && std::string(argv[2]) == "--inspect";
      GPU gpu(argv[1]);
      std::mt19937 rng(373);
      Corpus<common::NormalDistribution, aft_replay::NormalDistribution<MathReplay>>(gpu, 0, rng,
                                                                                     inspect);
      Corpus<common::LogisticDistribution, aft_replay::LogisticDistribution<MathReplay>>(
          gpu, 1, rng, inspect);
      Corpus<common::ExtremeDistribution, aft_replay::ExtremeDistribution<MathReplay>>(gpu, 2, rng,
                                                                                       inspect);
      std::printf("PASS upstream AFT replay: %zu cases / %zu rows / %zu exact source fields / %zu "
                  "math fields; max log/exp/erf ULP %llu/%llu/%llu\n",
                  cases, rows, source_fields, math_fields, (unsigned long long)max_ulp[0],
                  (unsigned long long)max_ulp[1], (unsigned long long)max_ulp[2]);
      std::printf("Sensitive rows %zu; CPU tolerance differences %zu fields; production/raw "
                  "differences %zu fields; inspection=%d\n",
                  sensitive, cpu_differences, production_differences, inspect);
      return 0;
    } catch (std::exception const& error) {
      std::fprintf(stderr, "AFT source FAIL: %s\n", error.what());
      return 1;
    }
  }
}
