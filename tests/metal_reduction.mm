// Differential checks for root-gradient sums and unregularized stump weights.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Pair {
  float g, h;
};
struct Sum {
  double g, h;
};
struct Params {
  uint64_t rows, stride0, stride1;
  uint32_t targets, blocks;
};
static_assert(sizeof(Params) == 32 && sizeof(Sum) == 16 && sizeof(Pair) == 8);
void Require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
// Neumaier accumulation provides an independent high-accuracy reference.
struct AccurateSum {
  double sum{0}, correction{0}, magnitude{0};
  void Add(double value) {
    double next = sum + value;
    correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
    magnitude += std::abs(value);
  }
  double Value() const { return sum + correction; }
};
float FromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [&](NSString* name) {
        auto f = [library newFunctionWithName:name];
        auto p = [device newComputePipelineStateWithFunction:f error:&error];
        Require(p != nil, "Missing reduction pipeline");
        return p;
      };
      auto reduce = pipeline(@"xgb_gradient_sum_reduce"),
           finish = pipeline(@"xgb_gradient_sum_finish"), weight = pipeline(@"xgb_stump_weight");
      auto queue = [device newCommandQueue];
      auto allocate = [&](size_t bytes) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      };
      size_t cases = 0, pairs = 0;
      std::mt19937 rng(904);
      for (uint64_t rows : {1, 31, 32, 255, 256, 257, 2048, 4097, 65537}) {
        for (uint32_t targets : {1, 3, 7})
          for (int layout = 0; layout < 3; ++layout) {
            uint64_t stride0 = layout == 1 ? 1 : targets * (layout == 2 ? 2 : 1);
            uint64_t stride1 = layout == 1 ? rows + 3 : (layout == 2 ? 2 : 1);
            for (int mode = 0; mode < 5; ++mode) {
              std::vector<Pair> input((rows - 1) * stride0 + (targets - 1) * stride1 + 4,
                                      {7.5f, -8.5f});
              std::vector<AccurateSum> expected_g(targets), expected_h(targets);
              for (uint64_t row = 0; row < rows; ++row)
                for (uint32_t t = 0; t < targets; ++t) {
                  Pair value{float(int(rng() % 2048) - 1024), float(rng() % 21)};
                  if (mode == 1)
                    value = {FromBits((rng() & 0x807fffffu)), FromBits(rng() & 0x007fffffu)};
                  if (mode == 2) {
                    value.g /= 3.17f;
                    value.h /= 7.13f;
                  }
                  if (mode == 3) {
                    value.g = std::ldexp(value.g, int(rng() % 225) - 120);
                    value.h = std::ldexp(value.h, int(rng() % 225) - 120);
                  }
                  if (mode == 4) value.h = t % 2 ? 0 : -value.h;
                  input[row * stride0 + t * stride1] = value;
                  expected_g[t].Add(value.g);
                  expected_h[t].Add(value.h);
                }
              auto in = [device newBufferWithBytes:input.data()
                                            length:input.size() * sizeof(Pair)
                                           options:MTLResourceStorageModeShared];
              Params p{rows, stride0, stride1, targets, uint32_t((rows + 255) / 256)};
              auto partial = allocate(size_t(p.blocks) * targets * sizeof(Sum));
              auto output = allocate(targets * sizeof(Sum)),
                   weights = allocate(targets * sizeof(float));
              auto command = [queue commandBuffer];
              auto e = [command computeCommandEncoder];
              [e setComputePipelineState:reduce];
              [e setBuffer:in offset:0 atIndex:0];
              [e setBuffer:partial offset:0 atIndex:1];
              [e setBytes:&p length:sizeof(p) atIndex:2];
              [e dispatchThreadgroups:MTLSizeMake(p.blocks, targets, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
              [e endEncoding];
              e = [command computeCommandEncoder];
              [e setComputePipelineState:finish];
              [e setBuffer:partial offset:0 atIndex:0];
              [e setBuffer:output offset:0 atIndex:1];
              [e setBytes:&p length:sizeof(p) atIndex:2];
              [e dispatchThreadgroups:MTLSizeMake(targets, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
              [e endEncoding];
              e = [command computeCommandEncoder];
              [e setComputePipelineState:weight];
              [e setBuffer:output offset:0 atIndex:0];
              [e setBuffer:weights offset:0 atIndex:1];
              [e dispatchThreads:MTLSizeMake(targets, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
              [e endEncoding];
              [command commit];
              [command waitUntilCompleted];
              Require(command.status == MTLCommandBufferStatusCompleted, "GPU reduction failed");
              auto sums = static_cast<Sum const*>(output.contents);
              auto actual_weight = static_cast<float const*>(weights.contents);
              for (uint32_t t = 0; t < targets; ++t) {
                if (mode == 0 || mode == 1 || mode == 4) {
                  Require(sums[t].g == expected_g[t].Value() && sums[t].h == expected_h[t].Value(),
                          "Exactly representable sum mismatch");
                } else {
                  Require(std::abs(sums[t].g - expected_g[t].Value()) <=
                                  expected_g[t].magnitude * 8e-14 &&
                              std::abs(sums[t].h - expected_h[t].Value()) <=
                                  expected_h[t].magnitude * 8e-14,
                          "Sum differs from compensated CPU reference");
                }
                // Given the exact returned sums, the final weight must match CPU bit-for-bit.
                float reference = float(-sums[t].g / std::max(sums[t].h, double(1.0e-6f)));
                Require(std::memcmp(&reference, &actual_weight[t], sizeof(float)) == 0,
                        "Stump weight differs from CPU binary64 division and FP32 rounding");
              }
              Require(std::memcmp(in.contents, input.data(), input.size() * sizeof(Pair)) == 0,
                      "Input or padding modified by reduction");
              pairs += rows * targets;
              ++cases;
            }
          }
        printf("Checked root/stump reductions for %llu rows\n", (unsigned long long)rows);
      }
      printf("PASS %zu gradient/stump cases / %zu gradient pairs on %s\n", cases, pairs,
             device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
