// Check ranking's sensitive FP32 sigmoid against CPU expf and division.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
uint32_t Bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return u;
}
float Value(uint32_t u) {
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::runtime_error("Pass production Metal source");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
        kernel void test_rank_math(device const uint* input [[buffer(0)]],
                                   device uint2* output [[buffer(1)]],
                                   uint tid [[thread_position_in_grid]]) {
          using namespace xgb_survival;
          uint bits = input[tid];
          D x = From(as_type<float>(bits ^ 0x80000000u)), cap = From(88.7f);
          if (x > cap) x = cap;
          uint e = as_type<uint>(xgb_dto_float(Exp(x)));
          uint denom = xgb_rank_fadd(xgb_rank_fadd(e, 0x3f800000u), as_type<uint>(1e-16f));
          uint precise = as_type<uint>(xgb_dto_float(one / From(as_type<float>(denom))));
          output[tid] = uint2(xgb_rank_sigmoid(bits), precise);
        })";
      auto device = MTLCreateSystemDefaultDevice();
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [device
          newComputePipelineStateWithFunction:[library newFunctionWithName:@"test_rank_math"]
                                        error:&error];
      if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
      std::mt19937 rng(18628);
      std::vector<uint32_t> inputs{0, 0x80000000u, 1, 0x80000001u, 0x7f800000u, 0xff800000u};
      for (int i = 0; i < 1000000; ++i) {
        inputs.push_back(Bits(float(int(rng() % 220000001) - 110000000) / 1000000.f));
        inputs.push_back(Bits(float(int(rng() % 200000001) - 100000000) / 100000000.f));
        uint32_t bits = rng();
        if ((bits & 0x7f800000u) != 0x7f800000u) inputs.push_back(bits);
      }
      auto in = [device newBufferWithBytes:inputs.data()
                                    length:inputs.size() * 4
                                   options:MTLResourceStorageModeShared];
      auto out = [device newBufferWithLength:inputs.size() * 8
                                     options:MTLResourceStorageModeShared];
      auto queue = [device newCommandQueue];
      auto command = [queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(inputs.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
      auto actual = static_cast<uint32_t const*>(out.contents);
      size_t cpu_diff = 0, precise_diff = 0, double_cpu_diff = 0;
      uint32_t max_cpu_ulp = 0;
      for (size_t i = 0; i < inputs.size(); ++i) {
        float x = std::min(-Value(inputs[i]), 88.7f);
        uint32_t expected = Bits(1.f / (std::exp(x) + 1.f + 1e-16f));
        auto got = actual[2 * i], precise = actual[2 * i + 1];
        float double_exp = static_cast<float>(std::exp(double(x)));
        uint32_t double_reference = Bits(1.f / (double_exp + 1.f + 1e-16f));
        double_cpu_diff += got != double_reference;
        if (got != expected) {
          ++cpu_diff;
          max_cpu_ulp = std::max(max_cpu_ulp, got > expected ? got - expected : expected - got);
          if (cpu_diff < 6)
            std::printf("CPU discrepancy x=%.9g: GPU=%08x CPU=%08x precise=%08x\n",
                        Value(inputs[i]), got, expected, precise);
        }
        if (got != precise) {
          ++precise_diff;
          if (precise_diff < 6)
            std::printf("Expansion discrepancy x=%.9g: GPU=%08x precise=%08x\n", Value(inputs[i]),
                        got, precise);
        }
      }
      std::printf("Checked %zu sigmoid inputs: %zu CPU differences (max %u ULP), %zu "
                  "binary64-oracle differences\n",
                  inputs.size(), cpu_diff, max_cpu_ulp, precise_diff);
      std::printf("CPU binary64-exp oracle differences: %zu\n", double_cpu_diff);
      // Apple's dynamic expf differs from correctly rounded exp in some cases;
      // both independent binary64-exp oracles must still agree exactly.
      if (precise_diff || double_cpu_diff || max_cpu_ulp > 4)
        throw std::runtime_error("Ranking sigmoid exceeded rounding contract");
    } catch (std::exception const& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}
