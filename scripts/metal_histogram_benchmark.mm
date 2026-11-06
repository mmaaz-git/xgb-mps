// Compare histogram workgroup sizes within one process, validating every result.
// clang++ -O2 -std=c++17 -fobjc-arc scripts/metal_histogram_benchmark.mm \
//   -framework Metal -framework Foundation -o results/metal_histogram_benchmark
// results/metal_histogram_benchmark xgboost/src/common/metal_histogram.metal >
// results/histogram.json
// Optional comparison: pass candidate.metal baseline.metal bins_per_feature.
// This alternates both kernels and checks every sum against the same CPU oracle.
// Append --ellpack to compare the compressed row-major page path (1-8 bit symbols).
// Append --ellpack ROWS FEATURES to compare global/shared/tiled paths on that shape.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct Pair {
  int64_t g, h;
};
struct Params {
  uint64_t rows, base, page_rows;
  uint32_t bins, features, width, dense, nodes, column_major, rows_per_lane;
};
static_assert(sizeof(Params) == 56);
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2 || argc == 4 || ((argc == 5 || argc == 7) && std::string(argv[4]) == "--ellpack"),
            "Pass candidate shader [baseline shader bins_per_feature [--ellpack [ROWS FEATURES]]]");
      bool compare = argc >= 4;
      bool compressed = argc >= 5 && std::string(argv[4]) == "--ellpack";
      bool all_paths = argc == 7;
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No Metal GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto name = compressed ? @"xgb_histogram_dense_shared_ellpack" : @"xgb_histogram_dense_shared";
      auto function = [library newFunctionWithName:name];
      auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      Check(pipeline != nil, "Missing histogram pipeline");
      id<MTLComputePipelineState> baseline_pipeline = nil;
      id<MTLComputePipelineState> global_pipeline = nil, baseline_global_pipeline = nil,
                                  tiled_pipeline = nil;
      if (all_paths) {
        global_pipeline = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"xgb_histogram_ellpack"] error:&error];
        tiled_pipeline = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"xgb_histogram_dense_tiled_ellpack"] error:&error];
        Check(global_pipeline != nil && tiled_pipeline != nil, "Missing global/tiled histogram pipeline");
      }
      if (compare) {
        std::ifstream baseline_file(argv[2]);
        Check(baseline_file.good(), "Missing baseline shader");
        std::string baseline_source{std::istreambuf_iterator<char>(baseline_file), {}};
        auto baseline_library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:baseline_source.c_str()]
                                 options:options
                                   error:&error];
        if (!baseline_library) throw std::runtime_error(error.localizedDescription.UTF8String);
        auto baseline_function =
            [baseline_library newFunctionWithName:name];
        baseline_pipeline = [device newComputePipelineStateWithFunction:baseline_function
                                                                  error:&error];
        Check(baseline_pipeline != nil, "Missing baseline histogram pipeline");
        if (all_paths) {
          baseline_global_pipeline = [device newComputePipelineStateWithFunction:
              [baseline_library newFunctionWithName:@"xgb_histogram_ellpack"] error:&error];
          Check(baseline_global_pipeline != nil, "Missing baseline global histogram pipeline");
        }
      }
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
      };
      uint64_t n = all_paths ? std::stoull(argv[5]) : 1000000;
      uint32_t features = all_paths ? std::stoul(argv[6]) : 32;
      Check(n > 0 && n <= UINT32_MAX && features >= 4, "Rows/features out of range");
      uint32_t per_feature = compare ? std::stoi(argv[3]) : 256;
      Check(per_feature >= 1 && per_feature <= 256, "This benchmark uses 8-bit indices");
      uint32_t symbol_bits = 1;
      while ((uint64_t{1} << symbol_bits) < per_feature) ++symbol_bits;
      uint32_t bins = per_feature * features;
      std::mt19937 rng(123);
      std::vector<uint8_t> indices(n * features);
      std::vector<Pair> gradients(n);
      std::vector<uint64_t> rowptr(n + 1);
      std::vector<uint32_t> offsets(features + 1);
      for (auto& index : indices) index = rng() % per_feature;
      for (uint64_t r = 0; r < n; ++r) {
        gradients[r] = {(int64_t(rng() % (1u << 24)) - (1 << 23)) * (int64_t(1) << 16), int64_t(1)
                                                                                            << 40};
        rowptr[r] = r * features;
      }
      rowptr[n] = n * features;
      for (uint32_t f = 0; f <= features; ++f) offsets[f] = f * per_feature;
      std::vector<uint8_t> packed;
      if (compressed) {
        packed.resize(8 + (n * features * symbol_bits + 7) / 8, 0);
        for (uint64_t r = 0; r < n; ++r) {
          for (uint32_t f = 0; f < features; ++f) {
            uint32_t value = indices[f * n + r];
            for (uint32_t b = 0; b < symbol_bits; ++b) {
              uint64_t bit = (r * features + f) * symbol_bits + b;
              packed[8 + bit / 8] |= ((value >> (symbol_bits - b - 1)) & 1) << (7 - bit % 8);
            }
          }
        }
      }
      auto index = compressed ? upload(packed.data(), packed.size()) : upload(indices.data(), indices.size());
      auto gradient = upload(gradients.data(), gradients.size() * sizeof(Pair)),
           ptr = upload(rowptr.data(), rowptr.size() * sizeof(uint64_t)),
           offset = upload(offsets.data(), offsets.size() * sizeof(uint32_t));
      printf("{\"device\":\"%s\",\"rows\":%llu,\"features\":%u,\"ellpack\":%s,\"repeats\":5,\"cases\":[\n",
             device.name.UTF8String, (unsigned long long)n, features, compressed ? "true" : "false");
      for (uint32_t layout = 0; layout < 5; ++layout) {
        uint32_t nodes = layout == 0 ? 1 : (layout == 1 ? 2 : (layout == 2 ? 8 : 32));
        std::vector<uint32_t> membership(n);
        std::vector<uint64_t> node_ptr(nodes + 1, 0);
        for (uint64_t r = 0; r < n; ++r) {
          membership[r] = layout == 0 ? 0 : (layout < 4 ? r % nodes : (r % 20 ? 0 : 1 + r % 31));
          ++node_ptr[membership[r] + 1];
        }
        for (uint32_t i = 1; i <= nodes; ++i) node_ptr[i] += node_ptr[i - 1];
        std::vector<uint64_t> cursor(node_ptr), rows(n);
        for (uint64_t r = 0; r < n; ++r) rows[cursor[membership[r]]++] = r;
        std::vector<Pair> reference(size_t(nodes) * bins, {0, 0});
        for (uint64_t r = 0; r < n; ++r)
          for (uint32_t f = 0; f < features; ++f) {
            auto& sum =
                reference[size_t(membership[r]) * bins + f * per_feature + indices[f * n + r]];
            sum.g += gradients[r].g;
            sum.h += gradients[r].h;
          }
        auto row_ids = upload(rows.data(), rows.size() * sizeof(uint64_t)),
             boundaries = upload(node_ptr.data(), node_ptr.size() * sizeof(uint64_t));
        auto output = [device newBufferWithLength:reference.size() * sizeof(Pair)
                                          options:MTLResourceStorageModeShared];
        std::vector<uint32_t> choices = all_paths ? std::vector<uint32_t>{0, 0, 8, 32, 8, 32} :
            compare ? std::vector<uint32_t>{8, 32, 8, 32} : std::vector<uint32_t>{1, 8, 16, 32, 64};
        std::vector<std::vector<double>> gpu_times(choices.size());
        // One warmup for each geometry, followed by shuffled alternating trials.
        for (int repeat = -1; repeat < 5; ++repeat) {
          std::vector<size_t> order = all_paths ? std::vector<size_t>{0, 1, 2, 3, 4, 5} :
              compare ? std::vector<size_t>{0, 1, 2, 3} : std::vector<size_t>{0, 1, 2, 3, 4};
          std::shuffle(order.begin(), order.end(), rng);
          for (auto choice : order) {
            auto rows_per_lane = std::max(1u, choices[choice]);
            bool shared = !all_paths || choice >= 2;
            bool tiled = all_paths && choice >= 4;
            auto features_per_group = tiled ? 4u : 1u;
            std::vector<uint64_t> block_ptr{0};
            for (uint32_t node = 0; node < nodes; ++node)
              block_ptr.push_back(block_ptr.back() +
                                  (node_ptr[node + 1] - node_ptr[node] + 256 * rows_per_lane - 1) /
                                      (256 * rows_per_lane));
            auto blocks = upload(block_ptr.data(), block_ptr.size() * sizeof(uint64_t));
            Params p{n, 0, n, bins, features, 1, 1, nodes, 1, rows_per_lane};
            auto command = [queue commandBuffer];
            auto clear = [command blitCommandEncoder];
            [clear fillBuffer:output range:NSMakeRange(0, output.length) value:0];
            [clear endEncoding];
            auto e = [command computeCommandEncoder];
            auto selected = all_paths ? (choice == 0 ? baseline_global_pipeline :
                choice == 1 ? global_pipeline : tiled ? tiled_pipeline : pipeline) :
                compare && choice < 2 ? baseline_pipeline : pipeline;
            [e setComputePipelineState:selected];
            [e setBuffer:index offset:0 atIndex:0];
            [e setBuffer:ptr offset:0 atIndex:1];
            [e setBuffer:offset offset:0 atIndex:2];
            [e setBuffer:gradient offset:0 atIndex:3];
            [e setBuffer:row_ids offset:0 atIndex:4];
            [e setBuffer:output offset:0 atIndex:5];
            [e setBytes:&p length:sizeof(p) atIndex:6];
            [e setBuffer:boundaries offset:0 atIndex:7];
            [e setBuffer:boundaries offset:0 atIndex:8];
            if (shared) [e setBuffer:blocks offset:0 atIndex:9];
            if (compressed) {
              struct Compressed { uint64_t split; uint32_t stride, bits, symbols, dense, features, padding; };
              static_assert(sizeof(Compressed) == 32);
              Compressed page{index.length, features, symbol_bits, per_feature, 1, features, 0};
              auto slot = shared ? 10u : 9u;
              [e setBuffer:index offset:0 atIndex:slot];  // Unused tail, still a valid binding.
              [e setBytes:&page length:sizeof(page) atIndex:slot + 1];
            }
            if (shared) {
              [e setThreadgroupMemoryLength:per_feature * sizeof(Pair) * features_per_group atIndex:0];
              [e dispatchThreadgroups:MTLSizeMake(block_ptr.back(), (features + features_per_group - 1) / features_per_group, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
              [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [e endEncoding];
            [command commit];
            [command waitUntilCompleted];
            Check(command.status == MTLCommandBufferStatusCompleted, "Histogram command failed");
            Check(std::memcmp(output.contents, reference.data(), reference.size() * sizeof(Pair)) ==
                      0,
                  "Histogram differs from CPU integer sums");
            if (repeat >= 0) gpu_times[choice].push_back(command.GPUEndTime - command.GPUStartTime);
          }
        }
        if (layout) printf(",\n");
        printf("{\"layout\":\"%s\",\"nodes\":%u,\"timings\":[",
               layout == 0 ? "root" : (layout < 4 ? "balanced" : "skewed"), nodes);
        for (size_t c = 0; c < choices.size(); ++c) {
          if (c) printf(",");
          printf(
              "{\"variant\":\"%s\",\"bins_per_feature\":%u,\"rows_per_lane\":%u,\"gpu_seconds\":[",
              all_paths ? (c == 0 ? "baseline-global" : c == 1 ? "candidate-global" :
                           c < 4 ? "shared" : "tiled") :
              compare ? (c < 2 ? "baseline" : "candidate") : "current", per_feature, choices[c]);
          for (size_t r = 0; r < gpu_times[c].size(); ++r)
            printf("%s%.9f", r ? "," : "", gpu_times[c][r]);
          printf("]}");
        }
        printf("]}");
        fflush(stdout);
        fprintf(stderr, "Validated histogram geometry layout %u/5\n", layout + 1);
      }
      printf("\n]}\n");
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
