// Standalone differential test of the actual production shader, including
// concurrent negative additions, low-word carry, sparse rows and page offsets.
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

struct Row {
  uint64_t row;
  uint32_t node, pad;
};
struct Pair {
  int64_t g, h;
};
struct Params {
  uint64_t rows, base, page_rows;
  uint32_t bins, features, width, dense, nodes, pad, rows_per_lane;
};
static_assert(sizeof(Row) == 16 && sizeof(Pair) == 16 && sizeof(Params) == 56);
void Require(bool ok, char const *what) {
  if (!ok) throw std::runtime_error(what);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass the production .metal source path");
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      Require(dev != nil, "No visible Metal device");
      std::ifstream file(argv[1]);
      Require(file.good(), "Cannot read production shader");
      std::string source{std::istreambuf_iterator<char>(file), {}};
      NSError *error = nil;
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      auto library = [dev newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                       options:options
                                         error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto kernel = [library newFunctionWithName:@"xgb_histogram"];
      auto pipeline = [dev newComputePipelineStateWithFunction:kernel error:&error];
      Require(pipeline != nil, "Cannot create pipeline");
      auto shared_kernel = [library newFunctionWithName:@"xgb_histogram_dense_shared"];
      auto shared_pipeline = [dev newComputePipelineStateWithFunction:shared_kernel error:&error];
      Require(shared_pipeline != nil, "Cannot create shared pipeline");
      auto queue = [dev newCommandQueue];
      auto upload = [&](void const *ptr, size_t size) -> id<MTLBuffer> {
        auto b = [dev newBufferWithLength:std::max<size_t>(size, 16)
                                  options:MTLResourceStorageModeShared];
        Require(b != nil, "Buffer allocation failed");
        if (size) memcpy(b.contents, ptr, size);
        return b;
      };
      int cases = 0;
      for (uint32_t width : {1u, 2u, 4u}) {
        for (bool dense : {true, false}) {
          // Sparse indices in XGBoost are always global uint32 indices.
          if (!dense && width != 4) continue;
          for (bool shared : {false, true}) {
            if (shared && !dense) continue;
            for (uint64_t base : {0ull, 11ull}) {
              for (bool contention : {false, true}) {
                for (uint32_t rows_per_lane :
                     shared ? std::vector<uint32_t>{1, 8, 32, 64} : std::vector<uint32_t>{8}) {
                  for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
                    uint64_t n = shared ? 16385 : 4097;
                    constexpr uint32_t features = 7, nodes = 3;
                    uint32_t per_feature =
                        mode >= 3 ? (mode % 2 == 1 ? 1 : 2) : (width == 1 ? 17 : 513);
                    uint32_t bins = per_feature * features;
                    std::mt19937 rng(71);
                    std::vector<uint8_t> indices;
                    std::vector<uint64_t> rowptr{0};
                    std::vector<uint32_t> offsets;
                    for (uint32_t f = 0; f < features; ++f) offsets.push_back(f * per_feature);
                    offsets.push_back(bins);
                    std::vector<Pair> gradients(n + base);
                    std::vector<Pair> reference(nodes * bins, {0, 0});
                    std::vector<Row> rows;
                    for (uint64_t r = 0; r < n; ++r) {
                      // Large signed integers exercise both high-word addition and
                      // carry.
                      auto signed_value = static_cast<int64_t>(rng()) - (int64_t{1} << 31);
                      Pair g{signed_value * (int64_t{1} << 17) + int64_t(r % 11),
                             (int64_t{1} << 35) + int64_t(rng())};
                      if (mode == 1 || mode == 5 || mode == 6) {
                        switch (r % 8) {
                          case 0:
                            g = {0, 0};
                            break;
                          case 1:
                            g = {int64_t(r + 1), 0};
                            break;
                          case 2:
                            g = {-int64_t(r + 1), int64_t{1} << 40};
                            break;
                          case 3:
                            g = {int64_t{1} << 40, int64_t{1} << 40};
                            break;
                          case 4:
                            g = {0, int64_t{1} << 40};
                            break;
                          case 5:
                            g = {-(int64_t{1} << 32), 0};
                            break;
                          case 6:
                            g = {0xffffffffll, 0};
                            break;
                          case 7:
                            g = {1, 0};
                            break;
                        }
                      } else if (mode == 2 || mode >= 7)
                        g = {0, 0};
                      gradients[r + base] = g;
                      uint32_t node =
                          (mode == 1 || mode == 5 || mode == 6) ? (r < 32 ? 2 : 0) : r % nodes;
                      rows.push_back({r + base, node, 0});
                      for (uint32_t f = 0; f < features; ++f) {
                        if (!dense && (r % 13 == 0 || rng() % 3 == 0)) continue;
                        uint32_t local = contention ? 0 : rng() % per_feature;
                        uint32_t global = offsets[f] + local;
                        uint32_t stored = dense ? local : global;
                        for (uint32_t byte = 0; byte < width; ++byte)
                          indices.push_back(uint8_t(stored >> (8 * byte)));
                        reference[node * bins + global].g += g.g;
                        reference[node * bins + global].h += g.h;
                      }
                      rowptr.push_back(indices.size() / width);
                    }
                    std::stable_sort(rows.begin(), rows.end(),
                                     [](Row a, Row b) { return a.node < b.node; });
                    std::vector<uint64_t> node_ptr(nodes + 1, 0);
                    for (auto row : rows) ++node_ptr[row.node + 1];
                    std::vector<uint64_t> block_ptr{0};
                    for (size_t node = 1; node < node_ptr.size(); ++node) {
                      block_ptr.push_back(block_ptr.back() +
                                          (node_ptr[node] + 256 * rows_per_lane - 1) /
                                              (256 * rows_per_lane));
                      node_ptr[node] += node_ptr[node - 1];
                    }
                    auto idx = upload(indices.data(), indices.size());
                    auto ptr = upload(rowptr.data(), rowptr.size() * sizeof(uint64_t));
                    auto off = upload(offsets.data(), offsets.size() * sizeof(uint32_t));
                    auto gp = upload(gradients.data(), gradients.size() * sizeof(Pair));
                    std::vector<uint64_t> row_ids;
                    for (auto row : rows) row_ids.push_back(row.row);
                    auto rs = upload(row_ids.data(), row_ids.size() * sizeof(uint64_t));
                    std::vector<Pair> zero(reference.size(), {0, 0});
                    auto out = upload(zero.data(), zero.size() * sizeof(Pair));
                    Params p{n, base,         n, bins, features, width, uint32_t(dense), nodes,
                             0, rows_per_lane};
                    auto command = [queue commandBuffer];
                    auto encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:shared ? shared_pipeline : pipeline];
                    [encoder setBuffer:idx offset:0 atIndex:0];
                    [encoder setBuffer:ptr offset:0 atIndex:1];
                    [encoder setBuffer:off offset:0 atIndex:2];
                    [encoder setBuffer:gp offset:0 atIndex:3];
                    [encoder setBuffer:rs offset:0 atIndex:4];
                    [encoder setBuffer:out offset:0 atIndex:5];
                    [encoder setBytes:&p length:sizeof(p) atIndex:6];
                    auto np = upload(node_ptr.data(), node_ptr.size() * sizeof(uint64_t));
                    [encoder setBuffer:np offset:0 atIndex:7];
                    [encoder setBuffer:np offset:0 atIndex:8];
                    if (shared) {
                      [encoder setThreadgroupMemoryLength:per_feature * sizeof(Pair) atIndex:0];
                      auto bp = upload(block_ptr.data(), block_ptr.size() * sizeof(uint64_t));
                      [encoder setBuffer:bp offset:0 atIndex:9];
                      [encoder dispatchThreadgroups:MTLSizeMake(block_ptr.back(), features, 1)
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else {
                      [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    [encoder endEncoding];
                    [command commit];
                    [command waitUntilCompleted];
                    Require(command.status == MTLCommandBufferStatusCompleted,
                            "GPU command failed");
                    auto actual = static_cast<Pair const *>(out.contents);
                    for (size_t i = 0; i < reference.size(); ++i) {
                      if (actual[i].g != reference[i].g || actual[i].h != reference[i].h) {
                        fprintf(stderr,
                                "Mismatch: width=%u dense=%d base=%llu "
                                "contention=%d mode=%u bin=%zu\n",
                                width, dense, (unsigned long long)base, contention, mode, i);
                        throw std::runtime_error("GPU histogram differs from "
                                                 "independent integer reference");
                      }
                    }
                    ++cases;
                    printf("PASS case %d: width=%u dense=%d shared=%d base=%llu "
                           "contention=%d mode=%u rows_per_lane=%u\n",
                           cases, width, dense, shared, (unsigned long long)base, contention, mode,
                           rows_per_lane);
                  }
                }
              }
            }
          }
        }
      }
      auto quantize_function = [library newFunctionWithName:@"xgb_quantize_gradients"];
      auto quantize_pipeline = [dev newComputePipelineStateWithFunction:quantize_function
                                                                  error:&error];
      Require(quantize_pipeline != nil, "Cannot create quantization pipeline");
      for (int exponent : {-100, -40, 0, 30, 100, 200}) {
        std::mt19937 gen(9);
        std::vector<float> input;
        std::vector<int64_t> reference;
        auto append = [&](float value) {
          double scaled = std::ldexp(double(value), exponent);
          if (!std::isfinite(scaled) || std::abs(scaled) > std::ldexp(1.0, 60)) return;
          // Interleaved gradient/Hessian with different signs.
          input.push_back(value);
          input.push_back(-value);
          reference.push_back(static_cast<int64_t>(scaled));
          auto h = static_cast<int64_t>(-scaled);
          if (-value > 0.f && h == 0) h = 1;
          reference.push_back(h);
        };
        for (float value :
             {0.0f, -0.0f, 0.5f, -0.5f, 1.5f, -1.5f, std::numeric_limits<float>::denorm_min(),
              std::numeric_limits<float>::min(), std::numeric_limits<float>::max()})
          append(value);
        for (size_t i = 0; i < 100000; ++i) {
          uint32_t bits = gen();
          float value;
          memcpy(&value, &bits, sizeof(value));
          append(value);
        }
        auto in = upload(input.data(), input.size() * sizeof(float));
        auto out = [dev newBufferWithLength:reference.size() * sizeof(int64_t)
                                    options:MTLResourceStorageModeShared];
        uint64_t count = input.size() / 2;
        struct {
          int32_t g, h;
          uint32_t valid, pad;
        } scale{exponent, exponent, 1, 0};
        auto scales = upload(&scale, sizeof(scale));
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:quantize_pipeline];
        [encoder setBuffer:in offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBuffer:scales offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        Require(command.status == MTLCommandBufferStatusCompleted, "Quantization command failed");
        auto actual = static_cast<int64_t const *>(out.contents);
        for (size_t i = 0; i < reference.size(); ++i)
          Require(actual[i] == reference[i],
                  "GPU quantization differs from CUDA truncation/positive-curvature rule");
        printf("PASS quantization exponent %d: %zu values (including subnormals)\n", exponent,
               reference.size());
      }
      printf("%d actual-GPU histogram cases passed on %s\n", cases, dev.name.UTF8String);
      return 0;
    } catch (std::exception const &e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
