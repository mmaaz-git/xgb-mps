// Measure actual sequential-prefix kernels, preserving the CPU FP32 addition order.
// clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc \
//   scripts/metal_sampling_prefix_benchmark.mm -framework Metal -framework Foundation \
//   -o results/metal_sampling_prefix_benchmark
// Pass candidate.metal baseline.metal. Timings exclude sort, staging and compilation;
// compare full training separately. Every output/checkpoint is checked after every run.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
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

uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc != 3) throw std::runtime_error("Pass candidate shader and baseline shader");
      auto device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("A visible Metal GPU is required");
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto load = [&](char const* path) {
        std::ifstream input(path);
        if (!input.good()) throw std::runtime_error("Missing shader");
        std::string source{std::istreambuf_iterator<char>(input), {}};
        auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                            options:options
                                              error:&error];
        if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
        return library;
      };
      auto candidate = load(argv[1]), baseline = load(argv[2]);
      auto pipeline = [&](id<MTLLibrary> library, NSString* name) {
        auto result = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                                            error:&error];
        if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
        return result;
      };
      id<MTLComputePipelineState> pipelines[] = {
          pipeline(baseline, @"xgb_quantile_weight_cdf"),
          pipeline(candidate, @"xgb_quantile_weight_cdf"),
          pipeline(candidate, @"xgb_gradient_sampling_prefix")};
      char const* names[] = {"baseline_full_prefix", "vector_full_prefix", "vector_checkpoints"};
      auto queue = [device newCommandQueue];
      std::mt19937 rng(123);
      std::printf("{\"device\":\"%s\",\"repeats\":5,\"scope\":\"GPU prefix only; excludes sort, "
                  "staging and compilation\",\"cases\":[\n",
                  device.name.UTF8String);
      bool first = true;
      for (uint32_t rows : {257u, 100000u, 1000000u}) {
        std::vector<float> weights(rows), reference(rows);
        std::vector<uint32_t> order(rows), records(2 * rows);
        for (auto& weight : weights) weight = float(rng() % 100000) / 7919.f;
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](auto a, auto b) { return weights[a] < weights[b]; });
        float sum = 0;
        for (uint32_t i = 0; i < rows; ++i) {
          records[2 * i] = Bits(weights[order[i]]) ^ 0x80000000u;
          records[2 * i + 1] = order[i];
          sum += weights[order[i]];
          reference[i] = sum;
        }
        auto sorted = [device newBufferWithBytes:records.data()
                                          length:records.size() * 4
                                         options:MTLResourceStorageModeShared];
        auto weight = [device newBufferWithBytes:weights.data()
                                          length:weights.size() * 4
                                         options:MTLResourceStorageModeShared];
        auto output = [device newBufferWithLength:rows * 4 options:MTLResourceStorageModeShared];
        std::vector<double> times[3];
        for (int repeat = -1; repeat < 5; ++repeat) {
          std::vector<int> variants{0, 1, 2};
          std::shuffle(variants.begin(), variants.end(), rng);
          for (int variant : variants) {
            auto command = [queue commandBuffer];
            auto encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipelines[variant]];
            [encoder setBuffer:sorted offset:0 atIndex:0];
            if (variant == 2) {
              [encoder setBuffer:output offset:0 atIndex:1];
              [encoder setBytes:&rows length:4 atIndex:2];
            } else {
              [encoder setBuffer:weight offset:0 atIndex:1];
              [encoder setBuffer:output offset:0 atIndex:2];
              [encoder setBytes:&rows length:4 atIndex:3];
            }
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
              throw std::runtime_error(command.error.localizedDescription.UTF8String);
            auto values = static_cast<float*>(output.contents);
            for (uint32_t i = 0; i < rows; ++i) {
              if (variant == 2 && i % 256 != 255 && i + 1 != rows) continue;
              if (Bits(values[variant == 2 ? i / 256 : i]) != Bits(reference[i]))
                throw std::runtime_error("Prefix differs from CPU");
            }
            if (repeat >= 0) times[variant].push_back(command.GPUEndTime - command.GPUStartTime);
          }
        }
        for (int variant = 0; variant < 3; ++variant) {
          if (!first) std::printf(",\n");
          first = false;
          std::printf("{\"rows\":%u,\"kernel\":\"%s\",\"gpu_seconds\":[", rows, names[variant]);
          for (size_t i = 0; i < times[variant].size(); ++i)
            std::printf("%s%.9f", i ? "," : "", times[variant][i]);
          std::printf("]}");
        }
        std::fprintf(stderr, "Validated %u rows, all variants/repeats\n", rows);
      }
      std::puts("\n],\"status\":\"complete\"}");
      return 0;
    } catch (std::exception const& error) {
      std::fprintf(stderr, "FAIL: %s\n", error.what());
      return 1;
    }
  }
}
