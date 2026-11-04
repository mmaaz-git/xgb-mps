// The column-sampling float log must retain ties in CPU-computed ES priorities.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::runtime_error("Pass the production Metal shader");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
        kernel void test_column_log(device const uint* input [[buffer(0)]],
          device uint* output [[buffer(1)]], uint row [[thread_position_in_grid]]) {
          output[row] = as_type<uint>(xgb_column_logf(as_type<float>(input[row])));
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
          [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"test_column_log"]
                                                error:&error];
      if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto queue = [device newCommandQueue];
      std::vector<uint32_t> inputs{0, Bits(1.f), Bits(0x1p-32f)};
      // Sample both the uniform draw distribution and the entire logarithmic
      // exponent domain. Exercise every table boundary across all 32 exponents.
      std::mt19937 rng(198271);
      for (uint32_t i = 0; i < 4000000; ++i) {
        inputs.push_back(Bits(float(rng()) * 0x1p-32f));
        inputs.push_back(Bits(0x1p-32f) + rng() % (Bits(1.f) - Bits(0x1p-32f)));
      }
      for (uint32_t exponent = 95; exponent < 127; ++exponent)
        for (uint32_t index = 0; index < 128; ++index)
          for (int offset = -2; offset <= 2; ++offset)
            inputs.push_back((exponent << 23) + (index << 16) + 0x8000 + offset);
      for (uint32_t i = 0; i < 65536; ++i) inputs.push_back(Bits(1.f) - i);
      std::size_t differences = 0;
      uint32_t max_ulps = 0;
      for (std::size_t begin = 0; begin < inputs.size(); begin += 1048576) {
        std::size_t n = std::min<std::size_t>(1048576, inputs.size() - begin);
        auto input = [device newBufferWithBytes:inputs.data() + begin
                                         length:n * 4
                                        options:MTLResourceStorageModeShared];
        auto output = [device newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
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
        auto actual = static_cast<uint32_t const*>(output.contents);
        for (std::size_t i = 0; i < n; ++i) {
          auto expected = Bits(std::log(Value(inputs[begin + i])));
          if (actual[i] != expected) {
            if (differences++ < 5)
              std::printf("Mismatch input=%08x CPU=%08x GPU=%08x\n", inputs[begin + i], expected,
                          actual[i]);
            auto ulps = actual[i] > expected ? actual[i] - expected : expected - actual[i];
            max_ulps = std::max(max_ulps, ulps);
          }
        }
        std::printf("Checked column log %zu/%zu inputs\n", begin + n, inputs.size());
        std::fflush(stdout);
      }
      std::printf("Column log: %zu inputs, %zu mismatches, maximum %u FP32 ULP\n", inputs.size(),
                  differences, max_ulps);
      if (differences) throw std::runtime_error("Column log changed CPU score rounding");
      std::puts("PASS column log CPU compatibility, zero/one and range-reduction boundaries");
    } catch (std::exception const& error) {
      std::fprintf(stderr, "FAIL: %s\n", error.what());
      return 1;
    }
  }
}
