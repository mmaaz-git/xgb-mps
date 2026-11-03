// Compare the original serial prefix and production SIMD prefix in one process.
// Every output is checked against independent CPU integer prefixes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct Split {
  uint64_t begin, end;
  uint32_t feature;
  int32_t split_bin;
  uint32_t default_left, categorical, cat_begin, cat_size, block_begin, block_end;
};
struct Params {
  uint64_t rows, base;
  uint32_t bin_bytes, dense, nodes, blocks;
};
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};

      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      id<MTLComputePipelineState> pipelines[2];
      NSArray<NSString*>* names = @[ @"xgb_partition_prefix_serial", @"xgb_partition_prefix" ];
      for (int i = 0; i < 2; ++i) {
        pipelines[i] =
            [device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]]
                                                  error:&error];
        Check(pipelines[i] != nil, "Missing pipeline");
      }
      auto queue = [device newCommandQueue];
      std::mt19937 rng(481);
      auto upload = [&](void const* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
      };
      constexpr uint32_t blocks = 3907, repeats = 5, dispatches = 50;
      std::printf("{\"device\":\"%s\",\"blocks\":%u,\"repeats\":%u,\"dispatches_per_measurement\":%"
                  "u,\"cases\":[\n",
                  device.name.UTF8String, blocks, repeats, dispatches);
      bool comma = false;
      for (uint32_t nodes : {1u, 2u, 8u, 32u, 64u, 128u, 256u})
        for (bool skew : {false, true}) {
          if (nodes == 1 && skew) continue;
          std::vector<uint32_t> counts(blocks), expected(blocks), guard(blocks + 16, 0xa5a5a5a5u);
          for (auto& x : counts) x = rng() % 257;
          std::vector<Split> splits(nodes);
          std::vector<uint64_t> expected_left(nodes), left_guard(nodes + 16, 0xababababababababull);
          uint32_t cursor = 0;
          for (uint32_t node = 0; node < nodes; ++node) {
            uint32_t end = (node + 1) * blocks / nodes;
            if (skew)
              end = node == 0 ? blocks * 95 / 100
                              : (node + 1 == nodes
                                     ? blocks
                                     : blocks * 95 / 100 +
                                           ((blocks - blocks * 95 / 100) * node / (nodes - 1)));
            splits[node].block_begin = cursor;
            splits[node].block_end = end;
            uint32_t total = 0;
            for (uint32_t i = cursor; i < end; ++i) {
              expected[i] = total;
              total += counts[i];
            }
            expected_left[node] = total;
            cursor = end;
          }
          auto c = upload(counts.data(), counts.size() * 4),
               sp = upload(splits.data(), splits.size() * sizeof(Split)),
               out = upload(guard.data(), guard.size() * 4),
               left = upload(left_guard.data(), left_guard.size() * 8);
          Params p{1000000, 0, 1, 1, nodes, blocks};
          std::array<std::vector<double>, 2> times;
          for (int trial = -1; trial < int(repeats); ++trial) {
            std::array<int, 2> order{0, 1};
            std::shuffle(order.begin(), order.end(), rng);
            for (int variant : order) {
              auto command = [queue commandBuffer];
              for (uint32_t j = 0; j < dispatches; ++j) {
                auto e = [command computeCommandEncoder];
                [e setComputePipelineState:pipelines[variant]];
                [e setBuffer:sp offset:0 atIndex:0];
                [e setBuffer:c offset:0 atIndex:1];
                [e setBuffer:out offset:0 atIndex:2];
                [e setBuffer:left offset:0 atIndex:3];
                [e setBytes:&p length:sizeof(p) atIndex:4];
                if (variant)
                  [e dispatchThreadgroups:MTLSizeMake(nodes, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else
                  [e dispatchThreads:MTLSizeMake(nodes, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
              }
              [command commit];
              [command waitUntilCompleted];
              Check(command.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
              auto result = static_cast<uint32_t const*>(out.contents);
              auto totals = static_cast<uint64_t const*>(left.contents);
              Check(std::equal(expected.begin(), expected.end(), result),
                    "Prefix differs from CPU");
              Check(std::equal(expected_left.begin(), expected_left.end(), totals),
                    "Total differs from CPU");
              for (size_t j = blocks; j < guard.size(); ++j)
                Check(result[j] == guard[j], "Prefix guard overwritten");
              for (size_t j = nodes; j < left_guard.size(); ++j)
                Check(totals[j] == left_guard[j], "Total guard overwritten");
              if (trial >= 0)
                times[variant].push_back((command.GPUEndTime - command.GPUStartTime) * 1e6 /
                                         dispatches);
            }
          }
          if (comma) std::printf(",\n");
          comma = true;
          std::printf("{\"nodes\":%u,\"skewed\":%s,\"variants\":[", nodes, skew ? "true" : "false");
          for (int i = 0; i < 2; ++i) {
            if (i) std::printf(",");
            std::printf("{\"name\":\"%s\",\"gpu_microseconds\":[", i ? "simd" : "serial");
            for (size_t j = 0; j < times[i].size(); ++j)
              std::printf("%s%.6f", j ? "," : "", times[i][j]);
            std::sort(times[i].begin(), times[i].end());
            std::printf("],\"median_microseconds\":%.6f}", times[i][times[i].size() / 2]);
          }
          std::printf("]}");
          std::fflush(stdout);
          std::fprintf(stderr, "Validated %u nodes, skew=%d\n", nodes, skew);
        }
      std::printf("\n]}\n");
      return 0;
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
