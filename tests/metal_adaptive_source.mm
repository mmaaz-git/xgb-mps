// CUDA expression replay with independently checked FP32 math primitives.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <vector>
#include "../results/cuda_objective_reference_generated.h"
#include "cuda_metric_reference.h"
#include "math_contract_io.h"
using namespace cuda_metric;
struct Params {
  std::uint64_t rows;
  std::uint32_t targets, label_targets, weighted, kind, blocks, padding;
};
struct ReplayMath {
  inline static float result, first, second;
  static float Hypot(float a, float b) {
    if (Ulps(a, first) || Ulps(b, second)) throw std::runtime_error("Hypot arguments differ");
    return result;
  }
  static float Tanh(float x) {
    if (Ulps(x, first)) throw std::runtime_error("Tanh argument differs");
    return result;
  }
  static float FMax(float a, float b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return std::fmax(a, b);
  }
};
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
float Hypot(float a, float b) {
  if (std::isinf(a) || std::isinf(b)) return INFINITY;
  return float(std::hypot(double(a), double(b)));
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      auto report = math_contracts::ReportOption(argc, argv, 8);
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
kernel void adaptive_source_math(device const float* pred [[buffer(0)]],
    device const float* labels [[buffer(1)]], device const float* scales [[buffer(2)]],
    device uint4* output [[buffer(3)]], constant AdaptiveParams& p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  using namespace xgb_objective_math;
  F r=From(pred[i])-From(labels[i]), s=From(scales[i]);
  F x=r/(From(.04f)*s);
  F math=p.kind==0?Hypot(s,r):Tanh(x);
  output[i]=uint4(math.bits,p.kind==0?s.bits:x.bits,r.bits,0);
}
kernel void adaptive_primitive_math(device const uint2* input [[buffer(0)]],
    device uint2* output [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  using namespace xgb_objective_math;
  output[i]=uint2(Hypot(F{input[i].x},F{input[i].y}).bits,Tanh(F{input[i].x}).bits);
})";
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No Metal device");
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [&](NSString* name) {
        auto f = [lib newFunctionWithName:name];
        auto p = [device newComputePipelineStateWithFunction:f error:&error];
        Check(p != nil, "No pipeline");
        return p;
      };
      auto gradient = pipeline(@"xgb_adaptive_objective"),
           probe = pipeline(@"adaptive_source_math"), math = pipeline(@"adaptive_primitive_math");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, std::size_t bytes) {
        auto b = [device newBufferWithLength:std::max<std::size_t>(16, bytes)
                                     options:MTLResourceStorageModeShared];
        Check(b != nil, "Allocation");
        if (data && bytes) std::memcpy(b.contents, data, bytes);
        return b;
      };
      std::mt19937 rng(653);
      std::vector<float> edges{0.f,
                               -0.f,
                               0x1p-149f,
                               -0x1p-149f,
                               0x1p-126f,
                               -0x1p-126f,
                               .25f,
                               -.25f,
                               .04f,
                               1.f,
                               -1.f,
                               10.f,
                               -10.f,
                               INFINITY,
                               -INFINITY,
                               NAN,
                               std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::lowest(),
                               1e-6f,
                               3e-4f};
      std::vector<std::array<float, 2>> inputs;
      for (float a : edges)
        for (float b : edges) inputs.push_back({a, b});
      for (unsigned i = 0; i < 500000; ++i) {
        float a = Cast<float>(unsigned(rng())), b = Cast<float>(unsigned(rng()));
        if (i % 3 == 0) a = std::uniform_real_distribution<float>(-11, 11)(rng);
        if (i % 7 == 0) b = std::nextafter(std::abs(a), INFINITY);
        inputs.push_back({a, b});
      }
      auto in = upload(inputs.data(), inputs.size() * 8), out = upload(nullptr, inputs.size() * 8);
      auto cmd = [queue commandBuffer];
      auto e = [cmd computeCommandEncoder];
      [e setComputePipelineState:math];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(inputs.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      Check(cmd.status == MTLCommandBufferStatusCompleted, "GPU math failed");
      auto actual = static_cast<float const*>(out.contents);
      std::uint64_t max_hypot = 0, max_tanh = 0;
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto h = Ulps(actual[2 * i], Hypot(inputs[i][0], inputs[i][1]));
        auto t = Ulps(actual[2 * i + 1], float(std::tanh(double(inputs[i][0]))));
        max_hypot = std::max(max_hypot, h);
        max_tanh = std::max(max_tanh, t);
        if (h > 1 || t > 1)
          std::fprintf(stderr, "math idx%zu input %.9g %.9g actual %.9g %.9g ULP%llu/%llu\n", i,
                       inputs[i][0], inputs[i][1], actual[2 * i], actual[2 * i + 1],
                       (unsigned long long)h, (unsigned long long)t);
        Check(h <= 1 && t <= 1, "Primitive exceeds one FP32 ULP");
      }
      std::printf("PASS %zu primitive fields; max hypot/tanh ULP %llu/%llu\n", inputs.size() * 2,
                  (unsigned long long)max_hypot, (unsigned long long)max_tanh);
      std::fflush(stdout);
      if (report) {
        math_contracts::Words input_words, observations;
        for (std::size_t i = 0; i < inputs.size(); ++i)
          for (unsigned k = 0; k < 2; ++k) {
            input_words.push_back(Cast<std::uint32_t>(inputs[i][k]));
            observations.push_back(Cast<std::uint32_t>(actual[2*i+k]));
          }
        report->Add(3, input_words, observations);
      }
      std::size_t fields = 0, cases = 0;
      for (unsigned kind : {0u, 1u})
        for (unsigned mode = 0; mode < 6; ++mode) {
          // One target per element permits independent supplied scales, isolating
          // source expression arithmetic from the separately checked reductions.
          std::size_t n = 65537;
          Params p{1, std::uint32_t(n), std::uint32_t(n), 1, kind, 1, 0};
          std::vector<float> pred(n), labels(n), weights(1), scales(n), alpha(n);
          weights[0] = mode == 1 ? 0.f : mode == 2 ? 0x1p-140f : mode == 3 ? 1e30f : 1.f;
          for (std::size_t i = 0; i < n; ++i) {
            pred[i] = mode == 0 ? std::uniform_real_distribution<float>(-100, 100)(rng)
                                : Cast<float>(unsigned(rng()));
            labels[i] = mode == 0 ? std::uniform_real_distribution<float>(-100, 100)(rng)
                                  : Cast<float>(unsigned(rng()));
            scales[i] = mode == 0 ? std::uniform_real_distribution<float>(.001, 100)(rng)
                                  : std::abs(Cast<float>(unsigned(rng())));
            alpha[i] = float(rng() % 1001) / 1000;
            if (mode == 4) {
              pred[i] = edges[i % edges.size()];
              labels[i] = edges[(i / edges.size()) % edges.size()];
              scales[i] = edges[(i / (edges.size() * edges.size())) % edges.size()];
            }
            if (mode == 5) {
              pred[i] = float(int(i % 11) - 5) * 0x1p-140f;
              labels[i] = 0;
              scales[i] = float(i % 13) * 0x1p-140f;
              alpha[i] = i % 2 ? .5f : 0x1p-140f;
            }
          }
          auto bp = upload(pred.data(), n * 4), by = upload(labels.data(), n * 4),
               bw = upload(weights.data(), 4), bs = upload(scales.data(), n * 4),
               ba = upload(alpha.data(), n * 4), bo = upload(nullptr, n * 8),
               bm = upload(nullptr, n * 16);
          cmd = [queue commandBuffer];
          e = [cmd computeCommandEncoder];
          [e setComputePipelineState:gradient];
          [e setBuffer:bp offset:0 atIndex:0];
          [e setBuffer:by offset:0 atIndex:1];
          [e setBuffer:bw offset:0 atIndex:2];
          [e setBuffer:bs offset:0 atIndex:3];
          [e setBuffer:ba offset:0 atIndex:4];
          [e setBuffer:bo offset:0 atIndex:5];
          [e setBytes:&p length:sizeof(p) atIndex:6];
          [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [e endEncoding];
          e = [cmd computeCommandEncoder];
          [e setComputePipelineState:probe];
          [e setBuffer:bp offset:0 atIndex:0];
          [e setBuffer:by offset:0 atIndex:1];
          [e setBuffer:bs offset:0 atIndex:2];
          [e setBuffer:bm offset:0 atIndex:3];
          [e setBytes:&p length:sizeof(p) atIndex:4];
          [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [e endEncoding];
          [cmd commit];
          [cmd waitUntilCompleted];
          Check(cmd.status == MTLCommandBufferStatusCompleted, "GPU gradient failed");
          auto pairs = static_cast<float const*>(bo.contents),
               values = static_cast<float const*>(bm.contents);
          for (std::size_t i = 0; i < n; ++i) {
            ReplayMath::result = values[4 * i];
            ReplayMath::first = values[4 * i + 1];
            ReplayMath::second = values[4 * i + 2];
            float oracle = kind ? float(std::tanh(double(ReplayMath::first)))
                                : Hypot(ReplayMath::first, ReplayMath::second);
            Check(Ulps(ReplayMath::result, oracle) <= 1,
                  "Actual primitive argument exceeds one ULP");
            auto expected = objective_reference::Ref<ReplayMath>::Adaptive(
                kind, pred[i], labels[i], scales[i], weights[0], alpha[i]);
            auto g = Ulps(pairs[2 * i], expected.GetGrad()),
                 h = Ulps(pairs[2 * i + 1], expected.GetHess());
            if (g || h)
              std::fprintf(stderr,
                           "kind%u mode%u idx%zu p%.9g y%.9g s%.9g w%.9g a%.9g actual%.9g/%.9g "
                           "expected%.9g/%.9g\n",
                           kind, mode, i, pred[i], labels[i], scales[i], weights[0], alpha[i],
                           pairs[2 * i], pairs[2 * i + 1], expected.GetGrad(), expected.GetHess());
            Check(!g && !h, "Extracted source composition differs");
            fields += 3;
          }
          ++cases;
          std::printf("PASS kind%u mode%u: %zu checked fields\n", kind, mode, fields);
          std::fflush(stdout);
        }
      if (report) report->Finish();
      std::printf("PASS %zu source cases / %zu fields; %zu primitive fields\n", cases, fields,
                  inputs.size() * 2);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL %s\n", e.what());
      return 1;
    }
  }
}
