// Copyright 2026, XGBoost Contributors
// Isolated missing-value histogram profiling; not part of the regression suite.
// Every timed output must exactly equal an independent CPU integer histogram.
// clang++ -O2 -std=c++17 -fobjc-arc scripts/metal_sparse_histogram_benchmark.mm \
//   -framework Metal -framework Foundation -o results/sparse_histogram_benchmark
// results/sparse_histogram_benchmark xgboost/src/common/metal_histogram.metal \
//   1000000 128 256 .2 5 > results/sparse-histogram.jsonl
// Rows are sorted within each node except the final diagnostic unordered case.
// This profiles explicit kernel choices; production dispatch uses host guards.
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
struct Pair {
  int64_t g, h;
};
struct Params {
  uint64_t rows, base, page_rows;
  uint32_t bins, features, width, dense, nodes, tile, rows_per_lane;
};
struct Compressed {
  uint64_t split;
  uint32_t stride, bits, symbols, dense, features, padding;
};
static_assert(sizeof(Params) == 56 && sizeof(Compressed) == 32);
void Check(bool ok, char const *message) {
  if (!ok) throw std::runtime_error(message);
}
std::string Read(char const *path) {
  std::ifstream f(path);
  Check(f.good(), "Missing shader");
  return {std::istreambuf_iterator<char>(f), {}};
}
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      Check(argc == 7, "Pass shader rows features bins_per_feature "
                       "missing_fraction repeats");
      uint64_t n = std::stoull(argv[2]);
      uint32_t features = std::stoul(argv[3]), per_feature = std::stoul(argv[4]);
      double missing = std::stod(argv[5]);
      int repeats = std::stoi(argv[6]);
      Check(missing > 0 && missing < 1 && repeats > 0, "Invalid missing fraction or repeats");
      Check(n > 8192 && n <= 1000000 && features >= 4 && features <= 1024 && per_feature > 1 &&
                per_feature <= 256,
            "Invalid shape");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No GPU");
      auto queue = [device newCommandQueue];
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      auto compile = [&](char const *path) {
        auto s = Read(path);
        NSError *error = nil;
        auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:s.c_str()]
                                        options:options
                                          error:&error];
        if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
        return lib;
      };
      auto library = compile(argv[1]);
      auto pipeline = [&](id<MTLLibrary> lib, NSString *name) {
        NSError *error = nil;
        auto p = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:name]
                                                       error:&error];
        if (!p) throw std::runtime_error(error.localizedDescription.UTF8String);
        return p;
      };
      auto original = pipeline(library, @"xgb_histogram_ellpack");
      auto shared = pipeline(library, @"xgb_histogram_sparse_shared_ellpack");
      auto upload = [&](void const *data, size_t bytes) {
        auto b = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        Check(b != nil, "Buffer allocation failed");
        return b;
      };
      constexpr uint64_t base = 73;
      uint32_t bins = features * per_feature, null = bins, stride = 0;
      std::mt19937 rng(1701);
      std::vector<uint32_t> cells(n * features, null), counts(n, 0), offsets(features + 1);
      for (uint64_t r = 0; r < n; ++r) {
        for (uint32_t f = 0; f < features; ++f) {
          auto value = rng() % per_feature;
          bool absent = rng() % 10000 < uint32_t(missing * 10000);
          if (!absent) {
            cells[r * features + f] = f * per_feature + value;
            ++counts[r];
          }
        }
        stride = std::max(stride, counts[r]);
      }
      Check(stride < features, "This benchmark requires global-bin sparse "
                               "geometry (stride < features)");
      uint32_t bits = 1;
      while ((uint64_t{1} << bits) < uint64_t(bins) + 1) ++bits;
      std::vector<uint8_t> packed(8 + (n * stride * bits + 7) / 8, 0);
      auto put = [&](uint64_t index, uint32_t value) {
        if (bits == 16) {
          packed[8 + index * 2] = value >> 8;
          packed[9 + index * 2] = value;
          return;
        }
        for (uint32_t b = 0; b < bits; ++b) {
          uint64_t bit = index * bits + b;
          packed[8 + bit / 8] |= ((value >> (bits - b - 1)) & 1) << (7 - bit % 8);
        }
      };
      for (uint64_t r = 0; r < n; ++r) {
        uint32_t j = 0;
        for (uint32_t f = 0; f < features; ++f) {
          auto v = cells[r * features + f];
          if (v != null) put(r * stride + j++, v);
        }
        while (j < stride) put(r * stride + j++, null);
      }
      std::vector<Pair> gradients(n + base, {0, 0});
      for (uint64_t r = 0; r < n; ++r) {
        auto &g = gradients[r + base];
        g = {(int64_t(rng() % (1 << 24)) - (1 << 23)) * (int64_t{1} << 16), int64_t{1} << 40};
        if (r % 19 == 0) g = {0, 0};
        if (r % 23 == 0) g.h = -g.h;
        if (r % 29 == 0) g.g += 0xffffffffll;
      }
      std::iota(offsets.begin(), offsets.end(), 0);
      for (auto &v : offsets) v *= per_feature;
      size_t split = packed.size() / 2 + 1;
      auto head = upload(packed.data(), split),
           tail = upload(packed.data() + split, packed.size() - split);
      auto grad = upload(gradients.data(), gradients.size() * sizeof(Pair)),
           offset = upload(offsets.data(), offsets.size() * 4);
      Compressed c{split, stride, bits, bins + 1, 0, features, 0};
      struct Choice {
        char const *name;
        int type;
        uint32_t parameter, rows;
      };
      std::vector<Choice> choices = {{"global", 0, 0, 0},
                                     {"shared512-r32", 4, 512, 32},
                                     {"shared1024-r32", 4, 1024, 32},
                                     {"shared2048-r8", 4, 2048, 8},
                                     {"shared2048-r32", 4, 2048, 32}};
      for (uint32_t layout = 0; layout < 5; ++layout) {
        char const *names[] = {"root", "balanced8", "skew32", "partial32", "unordered8"};
        uint32_t nodes = layout == 0 ? 1 : layout == 1 || layout == 4 ? 8 : 32;
        std::vector<std::vector<uint64_t>> members(nodes);
        for (uint64_t r = 0; r < n; ++r) {
          if (layout == 3 && r % 3 == 0) continue;
          uint32_t node = layout == 2 ? (r % 20 ? 0 : 1 + r % 31) : rng() % nodes;
          members[node].push_back(r + base);
        }
        if (layout == 4)
          for (auto &m : members) std::shuffle(m.begin(), m.end(), rng);
        std::vector<uint64_t> node_ptr{0}, row_begin, rows;
        for (auto const &m : members) {
          rows.insert(rows.end(), 3, base);
          row_begin.push_back(rows.size());
          rows.insert(rows.end(), m.begin(), m.end());
          node_ptr.push_back(node_ptr.back() + m.size());
        }
        std::vector<Pair> expected(size_t(nodes) * bins, {0, 0});
        for (uint32_t node = 0; node < nodes; ++node)
          for (auto row : members[node])
            for (uint32_t f = 0; f < features; ++f) {
              uint32_t bin = cells[(row - base) * features + f];
              if (bin == null) continue;
              auto &s = expected[size_t(node) * bins + bin];
              s.g += gradients[row].g;
              s.h += gradients[row].h;
            }
        auto rowids = upload(rows.data(), rows.size() * 8),
             boundaries = upload(node_ptr.data(), node_ptr.size() * 8),
             begins = upload(row_begin.data(), row_begin.size() * 8);
        auto output = [device newBufferWithLength:expected.size() * sizeof(Pair)
                                          options:MTLResourceStorageModeShared];
        Check(output != nil, "No histogram buffer");
        std::vector<size_t> order(choices.size());
        std::iota(order.begin(), order.end(), 0);
        for (int repeat = -1; repeat < repeats; ++repeat) {
          std::shuffle(order.begin(), order.end(), rng);
          for (auto choice : order) {
            auto q = choices[choice];
            auto command = [queue commandBuffer];
            std::memset(output.contents, 0xa5, output.length);
            auto clear = [command blitCommandEncoder];
            [clear fillBuffer:output range:NSMakeRange(0, output.length) value:0];
            [clear endEncoding];
            auto e = [command computeCommandEncoder];
            auto pipe = q.type == 0 ? original : shared;
            [e setComputePipelineState:pipe];
            Params p{node_ptr.back(),
                     base,
                     n,
                     bins,
                     features,
                     4,
                     0,
                     nodes,
                     q.parameter,
                     q.type == 4 ? q.rows : q.parameter};
            [e setBuffer:head offset:0 atIndex:0];
            [e setBuffer:boundaries offset:0 atIndex:1];
            [e setBuffer:offset offset:0 atIndex:2];
            [e setBuffer:grad offset:0 atIndex:3];
            [e setBuffer:rowids offset:0 atIndex:4];
            [e setBuffer:output offset:0 atIndex:5];
            [e setBytes:&p length:sizeof(p) atIndex:6];
            [e setBuffer:boundaries offset:0 atIndex:7];
            [e setBuffer:begins offset:0 atIndex:8];
            if (q.type == 4) {
              std::vector<uint64_t> blocks{0};
              for (auto &m : members)
                blocks.push_back(blocks.back() + (m.size() + 256 * q.rows - 1) / (256 * q.rows));
              auto block = upload(blocks.data(), blocks.size() * 8);
              [e setBuffer:block offset:0 atIndex:9];
              [e setBuffer:tail offset:0 atIndex:10];
              [e setBytes:&c length:sizeof(c) atIndex:11];
              Check(q.parameter * sizeof(Pair) <= device.maxThreadgroupMemoryLength,
                    "Shared tile exceeds device limit");
              [e setThreadgroupMemoryLength:q.parameter * sizeof(Pair) atIndex:0];
              [e dispatchThreadgroups:MTLSizeMake(blocks.back(),
                                                  (bins + q.parameter - 1) / q.parameter, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
              [e setBuffer:tail offset:0 atIndex:9];
              [e setBytes:&c length:sizeof(c) atIndex:10];
              [e dispatchThreads:MTLSizeMake(p.rows, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [e endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
              throw std::runtime_error(command.error.localizedDescription.UTF8String);
            Check(
                std::memcmp(output.contents, expected.data(), expected.size() * sizeof(Pair)) == 0,
                "Histogram differs from CPU exact integer oracle");
            if (repeat >= 0)
              printf("{\"layout\":\"%s\",\"variant\":\"%s\",\"repeat\":%d,"
                     "\"rows\":%llu,\"features\":%u,\"bins_per_feature\":%u,"
                     "\"stride\":%u,\"bits\":%u,\"missing\":%.3f,\"gpu_ms\":%."
                     "9f,\"exact\":true}\n",
                     names[layout], q.name, repeat, (unsigned long long)n, features, per_feature,
                     stride, bits, missing, (command.GPUEndTime - command.GPUStartTime) * 1000);
            fflush(stdout);
          }
        }
      }
    } catch (std::exception const &e) {
      fprintf(stderr, "FAIL %s\n", e.what());
      return 1;
    }
  }
  return 0;
}
