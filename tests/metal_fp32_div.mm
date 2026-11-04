// Exact division used by sketch pruning, checked against CPU hardware division.
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

uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}
float Value(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}
struct Input {
  uint32_t a, b;
};
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::runtime_error("Pass the production Metal shader");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
        kernel void test_div(device const uint2* input [[buffer(0)]],
          device uint* output [[buffer(1)]], uint row [[thread_position_in_grid]]) {
          output[row + 1] = as_type<uint>(xgb_fdiv_preserve(
            as_type<float>(input[row].x), as_type<float>(input[row].y)));
        })";
      auto device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("No Metal device");
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline =
          [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"test_div"]
                                                error:&error];
      if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto queue = [device newCommandQueue];
      std::vector<Input> inputs;
      std::vector<uint32_t> edge{0,
                                 1,
                                 2,
                                 3,
                                 0x7ffffe,
                                 0x7fffff,
                                 0x800000,
                                 0x800001,
                                 Bits(.5f),
                                 Bits(1.f) - 1,
                                 Bits(1.f),
                                 Bits(1.f) + 1,
                                 Bits(2.f),
                                 Bits(3.f),
                                 Bits(65537.f),
                                 0x7f7fffff,
                                 0x7f800000,
                                 0x7fc00000,
                                 0x7f800001};
      auto positive = edge;
      for (auto bits : positive) edge.push_back(bits | 0x80000000u);
      for (auto a : edge)
        for (auto b : edge) inputs.push_back({a, b});
      std::mt19937 rng(73163);
      for (uint32_t i = 0; i < 4000000; ++i) inputs.push_back({rng(), rng()});
      // All exponent pairs, with mantissas near power-of-two rounding boundaries.
      for (uint32_t ea = 0; ea < 255; ++ea)
        for (uint32_t eb = 0; eb < 255; ++eb)
          for (uint32_t m : {0u, 1u, 0x3fffffu, 0x400000u, 0x7ffffeu, 0x7fffffu}) {
            inputs.push_back({(ea << 23) | m, (eb << 23) | m});
            inputs.push_back({(ea << 23) | m, (eb << 23) | (m ^ 0x7fffffu)});
          }
      // Pruning's integer denominators and exact subnormal half-way cases.
      for (uint32_t n = 1; n <= 131073; ++n) {
        inputs.push_back({rng(), Bits(float(n))});
        inputs.push_back({Bits(float(n - 1)), Bits(float(n))});
        for (auto b : {Bits(2.f), Bits(4.f), Bits(8.f)}) {
          inputs.push_back({n, b});
          inputs.push_back({n | 0x80000000u, b});
        }
      }
      std::size_t differences = 0;
      constexpr uint32_t guard = 0x97abcdefu;
      for (std::size_t begin = 0; begin < inputs.size(); begin += 1048576) {
        std::size_t n = std::min<std::size_t>(1048576, inputs.size() - begin);
        auto input = [device newBufferWithBytes:inputs.data() + begin
                                         length:n * sizeof(Input)
                                        options:MTLResourceStorageModeShared];
        auto output = [device newBufferWithLength:(n + 2) * 4 options:MTLResourceStorageModeShared];
        auto actual = static_cast<uint32_t*>(output.contents);
        std::fill_n(actual, n + 2, guard);
        auto command = [queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:pipeline];
        [e setBuffer:input offset:0 atIndex:0];
        [e setBuffer:output offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
          throw std::runtime_error(command.error.localizedDescription.UTF8String);
        if (actual[0] != guard || actual[n + 1] != guard)
          throw std::runtime_error("Output guard changed");
        if (std::memcmp(input.contents, inputs.data() + begin, n * sizeof(Input)))
          throw std::runtime_error("Source changed");
        for (std::size_t i = 0; i < n; ++i) {
          auto p = inputs[begin + i];
          float expected = Value(p.a) / Value(p.b);
          bool equal = std::isnan(expected) ? std::isnan(Value(actual[i + 1]))
                                            : Bits(expected) == actual[i + 1];
          if (!equal && differences++ < 8)
            std::printf("Mismatch %08x/%08x CPU=%08x GPU=%08x\n", p.a, p.b, Bits(expected),
                        actual[i + 1]);
        }
        std::printf("Checked FP32 division %zu/%zu inputs\n", begin + n, inputs.size());
        std::fflush(stdout);
      }
      std::printf("FP32 division: %zu inputs, %zu mismatches\n", inputs.size(), differences);
      if (differences) throw std::runtime_error("Division changed CPU rounding");
      std::puts(
          "PASS exact finite/infinite/zero division, NaN classification, source and output guards");
    } catch (std::exception const& error) {
      std::fprintf(stderr, "FAIL: %s\n", error.what());
      return 1;
    }
  }
}
