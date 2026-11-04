// Validate the production CUDA ranking distribution, including rejection,
// singleton ranges, independent pair streams and 64-bit engine discard.
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
#include "cuda_ranking_pairs.h"
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::runtime_error("Pass production Metal source");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
        kernel void test_rank_rng(device const ulong4* input [[buffer(0)]],
                                   device ulong2* output [[buffer(1)]],
                                   uint tid [[thread_position_in_grid]]) {
          for (uint i = 0; i < 16; ++i) {
            ulong position = input[tid].z + i;
            uint value = xgb_rank_uniform(uint(input[tid].x), position, uint(input[tid].y));
            output[ulong(tid) * 16 + i] = ulong2(value, position);
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
      std::vector<uint64_t> positions{0,
                                      1,
                                      2,
                                      3,
                                      15,
                                      255,
                                      0xffffffffULL,
                                      0x100000000ULL,
                                      0x10000000003ULL,
                                      0x7fffffffffffffefULL};
      std::vector<uint64_t> inputs, expected;
      for (uint32_t seed : seeds)
        for (uint32_t range : ranges)
          for (auto start : positions) {
            inputs.insert(inputs.end(), {seed, range, start, 0});
            for (uint64_t i = 0; i < 16; ++i) {
              auto position = start + i;
              expected.push_back(cuda_reference::RankingUniform(seed, &position, range));
              expected.push_back(position);
            }
          }
      auto in = [device newBufferWithBytes:inputs.data()
                                    length:inputs.size() * 8
                                   options:MTLResourceStorageModeShared];
      auto out = [device newBufferWithLength:expected.size() * 8
                                     options:MTLResourceStorageModeShared];
      auto queue = [device newCommandQueue];
      auto command = [queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(inputs.size() / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
      auto actual = static_cast<uint64_t const*>(out.contents);
      for (size_t i = 0; i < expected.size(); ++i)
        if (actual[i] != expected[i]) {
          size_t case_id = i / 32;
          std::fprintf(stderr, "seed=%llu range=%llu start=%llu draw=%zu field=%zu: %llu != %llu\n",
                       inputs[4 * case_id], inputs[4 * case_id + 1], inputs[4 * case_id + 2],
                       i / 2 % 16, i % 2, actual[i], expected[i]);
          throw std::runtime_error("Sampling sequence differs from CUDA-source oracle");
        }
      std::printf("PASS %zu seed/range/discard cases / %zu exact CUDA-source draws and positions\n",
                  inputs.size() / 4, expected.size() / 2);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}
