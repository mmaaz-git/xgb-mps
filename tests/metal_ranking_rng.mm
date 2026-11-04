// Validate production sampling against Apple's actual standard-library engine
// and distribution, including 31/32-bit ranges and rejection boundaries.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::runtime_error("Pass production Metal source");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
        kernel void test_rank_rng(device const uint2* input [[buffer(0)]],
                                   device uint2* output [[buffer(1)]],
                                   uint tid [[thread_position_in_grid]]) {
          uint state = input[tid].x % 2147483647u;
          if (!state) state = 1;
          for (uint i = 0; i < 256; ++i) {
            uint value = xgb_rank_uniform(state, input[tid].y), copy = state;
            output[ulong(tid) * 256 + i] = uint2(value, xgb_rank_minstd(copy));
          }
        })";
      auto device = MTLCreateSystemDefaultDevice();
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline =
          [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"test_rank_rng"]
                                                error:&error];
      if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
      std::vector<uint32_t> seeds{0,          1,           2,           2147483646,
                                  2147483647, 2147483648u, 4294967294u, 4294967295u};
      std::mt19937 rng(78223);
      for (int i = 0; i < 32; ++i) seeds.push_back(rng());
      std::vector<uint32_t> ranges{1,           2,           3,          17,    127,
                                   129,         257,         4097,       65537, 2147483646u,
                                   2147483647u, 2147483648u, 4294967295u};
      for (uint32_t bit = 2; bit < 32; ++bit) {
        ranges.push_back((1u << bit) - 1);
        ranges.push_back(1u << bit);
        ranges.push_back((1u << bit) + 1);
      }
      std::vector<uint32_t> inputs, expected;
      for (uint32_t seed : seeds)
        for (uint32_t range : ranges) {
          inputs.push_back(seed);
          inputs.push_back(range);
          std::minstd_rand engine(seed);
          std::uniform_int_distribution<size_t> dist(0, size_t(range) - 1);
          for (int i = 0; i < 256; ++i) {
            expected.push_back(dist(engine));
            auto copy = engine;
            expected.push_back(copy());
          }
        }
      auto in = [device newBufferWithBytes:inputs.data()
                                    length:inputs.size() * 4
                                   options:MTLResourceStorageModeShared];
      auto out = [device newBufferWithLength:expected.size() * 4
                                     options:MTLResourceStorageModeShared];
      auto queue = [device newCommandQueue];
      auto command = [queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(inputs.size() / 2, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
      auto actual = static_cast<uint32_t const*>(out.contents);
      for (size_t i = 0; i < expected.size(); ++i)
        if (actual[i] != expected[i]) {
          size_t case_id = i / 512;
          std::fprintf(stderr, "seed=%u range=%u draw=%zu field=%zu: %u != %u\n",
                       inputs[2 * case_id], inputs[2 * case_id + 1], i / 2 % 256, i % 2, actual[i],
                       expected[i]);
          throw std::runtime_error("Sampling sequence differs from libc++");
        }
      std::printf("PASS %zu seed/range cases / %zu sampled values and engine states, bit-exact "
                  "with Apple libc++\n",
                  inputs.size() / 2, expected.size() / 2);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}
