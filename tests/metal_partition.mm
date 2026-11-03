// Stable partition differential tests of the production Metal kernels.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
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
  int32_t bin;
  uint32_t missing, cat, cat_begin, cat_size, block_begin, block_end;
};
struct Block {
  uint64_t begin, end;
  uint32_t node, pad;
};
struct Params {
  uint64_t rows, base;
  uint32_t width, dense, nodes, blocks;
};
void Require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production shader source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal device");
      std::ifstream file(argv[1]);
      Require(file.good(), "Cannot read shader");
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [&](NSString* name) {
        auto fn = [library newFunctionWithName:name];
        auto p = [device newComputePipelineStateWithFunction:fn error:&error];
        Require(p != nil, "Pipeline creation failed");
        return p;
      };
      auto classify = pipeline(@"xgb_partition_classify"),
           prefix = pipeline(@"xgb_partition_prefix"), serial_prefix = pipeline(@"xgb_partition_prefix_serial"), scatter = pipeline(@"xgb_partition_scatter"),
           transpose = pipeline(@"xgb_transpose_index");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, size_t bytes) {
        auto b = [device newBufferWithLength:std::max<size_t>(bytes, 16)
                                     options:MTLResourceStorageModeShared];
        Require(b != nil, "Buffer allocation failed");
        if (bytes) memcpy(b.contents, data, bytes);
        return b;
      };
      auto complete = [&](id<MTLCommandBuffer> command) {
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
          throw std::runtime_error(command.error.localizedDescription.UTF8String);
      };
      int cases = 0;
      for (uint32_t width : {1u, 2u, 4u})
        for (bool dense : {false, true}) {
          if (!dense && width != 4) continue;
          for (bool categorical : {false, true})
            for (bool missing_left : {false, true})
              for (uint64_t base : {0ull, 17ull})
                for (uint64_t active :
                     {1ull, 255ull, 256ull, 257ull, 511ull, 512ull, 513ull, 4097ull, 131071ull, 131075ull}) {
                  uint64_t n = active + 3;
                  constexpr uint32_t features = 5;
                  uint32_t bins = width == 1 ? 9 : 513;
                  std::mt19937 random(78);
                  std::vector<uint8_t> indices;
                  std::vector<uint64_t> rowptr{0};
                  std::vector<uint32_t> offsets;
                  std::vector<float> cuts;
                  for (uint32_t f = 0; f < features; ++f) {
                    offsets.push_back(f * bins);
                    for (uint32_t b = 0; b < bins; ++b) cuts.push_back(float(b));
                  }
                  offsets.push_back(features * bins);
                  std::vector<int32_t> values(n * features, -1);
                  for (uint64_t r = 0; r < n; ++r) {
                    for (uint32_t f = 0; f < features; ++f) {
                      if (!dense && random() % 4 == 0) continue;
                      uint32_t bin = random() % bins;
                      values[r * features + f] = bin;
                      uint32_t encoded = dense ? bin : offsets[f] + bin;
                      for (uint32_t b = 0; b < width; ++b)
                        indices.push_back(uint8_t(encoded >> (b * 8)));
                    }
                    rowptr.push_back(indices.size() / width);
                  }
                  std::vector<uint64_t> rows(n);
                  std::iota(rows.begin(), rows.end(), base);
                  std::shuffle(rows.begin(), rows.end(), random);
                  auto expected = rows;
                  std::vector<uint32_t> categories((bins + 31) / 32, 0);
                  for (uint32_t b = 0; b < bins; ++b)
                    if (b % 3 == 1) categories[b / 32] |= 1u << (31 - b % 32);
                  std::vector<Split> splits{
                      {active / 2, active, 3, int32_t(3 * bins + bins / 2), uint32_t(missing_left),
                       uint32_t(categorical), 0, uint32_t(categories.size()), 0, 0},
                      {0, 0, 1, -1, uint32_t(missing_left), uint32_t(categorical), 0,
                       uint32_t(categories.size()), 0, 0},
                      {0, active / 2, 1, -1, uint32_t(missing_left), uint32_t(categorical), 0,
                       uint32_t(categories.size()), 0, 0}};
                  std::vector<uint64_t> expected_counts;
                  std::vector<Block> blocks;
                  for (uint32_t i = 0; i < splits.size(); ++i) {
                    auto& split = splits[i];
                    split.block_begin = blocks.size();
                    for (uint64_t b = split.begin; b < split.end; b += 256)
                      blocks.push_back({b, std::min(b + 256, split.end), i, 0});
                    split.block_end = blocks.size();
                    auto mid = std::stable_partition(
                        expected.begin() + split.begin, expected.begin() + split.end,
                        [&](uint64_t row) {
                          int32_t value = values[(row - base) * features + split.feature];
                          if (value < 0) return missing_left;
                          if (categorical) return value % 3 != 1;
                          return int64_t(offsets[split.feature]) + value <= split.bin;
                        });
                    expected_counts.push_back(mid - (expected.begin() + split.begin));
                  }
                  auto idx = upload(indices.data(), indices.size()),
                       rp = upload(rowptr.data(), rowptr.size() * 8),
                       off = upload(offsets.data(), offsets.size() * 4),
                       cut = upload(cuts.data(), cuts.size() * 4);
                  if (dense) {
                    auto output = [device newBufferWithLength:indices.size()
                                                      options:MTLResourceStorageModeShared];
                    struct {
                      uint64_t rows;
                      uint32_t features, width;
                    } p{n, features, width};
                    auto cmd = [queue commandBuffer];
                    auto e = [cmd computeCommandEncoder];
                    [e setComputePipelineState:transpose];
                    [e setBuffer:idx offset:0 atIndex:0];
                    [e setBuffer:output offset:0 atIndex:1];
                    [e setBytes:&p length:sizeof(p) atIndex:2];
                    [e dispatchThreadgroups:MTLSizeMake((features + 31) / 32, (n + 31) / 32, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
                    [e endEncoding];
                    complete(cmd);
                    auto actual = static_cast<uint8_t const*>(output.contents);
                    for (uint64_t r = 0; r < n; ++r)
                      for (uint32_t f = 0; f < features; ++f)
                        for (uint32_t b = 0; b < width; ++b)
                          Require(actual[(f * n + r) * width + b] ==
                                      indices[(r * features + f) * width + b],
                                  "Transpose differs from CPU");
                    idx = output;
                  }
                  auto row = upload(rows.data(), rows.size() * 8),
                       split = upload(splits.data(), splits.size() * sizeof(Split)),
                       block = upload(blocks.data(), blocks.size() * sizeof(Block)),
                       cats = upload(categories.data(), categories.size() * 4);
                  std::vector<uint32_t> zero(n, 0);
                  std::vector<uint64_t> zero64(n, 0);
                  auto ranks = upload(zero.data(), n * 4), counts = upload(zero.data(), n * 4),
                       prefixes = upload(zero.data(), n * 4), left = upload(zero64.data(), n * 8),
                       saved = upload(zero64.data(), n * 8);
                  Params p{n,
                           base,
                           width,
                           uint32_t(dense),
                           uint32_t(splits.size()),
                           uint32_t(blocks.size())};
                  auto command = [queue commandBuffer];
                  auto e = [command computeCommandEncoder];
                  [e setComputePipelineState:classify];
                  [e setBuffer:idx offset:0 atIndex:0];
                  [e setBuffer:rp offset:0 atIndex:1];
                  [e setBuffer:off offset:0 atIndex:2];
                  [e setBuffer:cut offset:0 atIndex:3];
                  [e setBuffer:row offset:0 atIndex:4];
                  [e setBuffer:split offset:0 atIndex:5];
                  [e setBuffer:block offset:0 atIndex:6];
                  [e setBuffer:cats offset:0 atIndex:7];
                  [e setBuffer:ranks offset:0 atIndex:8];
                  [e setBuffer:counts offset:0 atIndex:9];
                  [e setBuffer:saved offset:0 atIndex:10];
                  [e setBytes:&p length:sizeof(p) atIndex:11];
                  [e dispatchThreadgroups:MTLSizeMake(blocks.size(), 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                  [e endEncoding];
                  e = [command computeCommandEncoder];
                  bool serial = std::all_of(splits.begin(), splits.end(), [](auto const& s) { return s.block_end-s.block_begin <= 16; });
                  [e setComputePipelineState:serial ? serial_prefix : prefix];
                  [e setBuffer:split offset:0 atIndex:0];
                  [e setBuffer:counts offset:0 atIndex:1];
                  [e setBuffer:prefixes offset:0 atIndex:2];
                  [e setBuffer:left offset:0 atIndex:3];
                  [e setBytes:&p length:sizeof(p) atIndex:4];
                  if (serial) {
                    [e dispatchThreads:MTLSizeMake(splits.size(), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                  } else {
                    [e dispatchThreadgroups:MTLSizeMake(splits.size(), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                  }
                  [e endEncoding];
                  e = [command computeCommandEncoder];
                  [e setComputePipelineState:scatter];
                  [e setBuffer:row offset:0 atIndex:0];
                  [e setBuffer:saved offset:0 atIndex:1];
                  [e setBuffer:split offset:0 atIndex:2];
                  [e setBuffer:block offset:0 atIndex:3];
                  [e setBuffer:ranks offset:0 atIndex:4];
                  [e setBuffer:prefixes offset:0 atIndex:5];
                  [e setBuffer:left offset:0 atIndex:6];
                  [e dispatchThreadgroups:MTLSizeMake(blocks.size(), 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                  [e endEncoding];
                  complete(command);
                  auto actual = static_cast<uint64_t const*>(row.contents),
                       actual_counts = static_cast<uint64_t const*>(left.contents);
                  Require(std::equal(expected.begin(), expected.end(), actual),
                          "Metal partition differs from CPU stable partition");
                  Require(std::equal(expected_counts.begin(), expected_counts.end(), actual_counts),
                          "Left counts differ from CPU");
                  ++cases;
                }
          printf("PASS width=%u dense=%d; cumulative %d partition cases\n", width, dense, cases);
        }
      printf("%d exact GPU partition cases passed on %s\n", cases, device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
