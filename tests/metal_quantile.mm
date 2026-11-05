// Independent full-output checks for integer prefixes, stable sorting and quantiles.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct Shape {
  uint64_t rows, stride0, stride1;
  uint32_t targets, blocks;
};
struct Radix {
  uint32_t rows, targets, blocks, shift;
};
struct Request {
  uint32_t target, padding;
  double alpha;
};
struct Params {
  uint32_t rows, targets, weighted, requests;
};
struct Record {
  uint32_t key, row;
};
static_assert(sizeof(Shape) == 32 && sizeof(Request) == 16 && sizeof(Radix) == 16);
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
uint32_t Bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, 4);
  return result;
}
float Float(uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, 4);
  return result;
}
struct GPU {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> init, rank, scatter, scan, add, cdf, select;
  explicit GPU(char const* path) {
    device = MTLCreateSystemDefaultDevice();
    Check(device != nil, "No Metal GPU");
    queue = [device newCommandQueue];
    std::ifstream file(path);
    std::string source{std::istreambuf_iterator<char>(file), {}};
    auto options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    NSError* error = nil;
    auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                        options:options
                                          error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    auto pipeline = [&](NSString* name) {
      auto f = [library newFunctionWithName:name];
      auto p = [device newComputePipelineStateWithFunction:f error:&error];
      Check(p != nil, "Missing sort/quantile pipeline");
      return p;
    };
    init = pipeline(@"xgb_sort_columns_init");
    rank = pipeline(@"xgb_radix_rank");
    scatter = pipeline(@"xgb_radix_scatter");
    scan = pipeline(@"xgb_prefix_u32");
    add = pipeline(@"xgb_prefix_add_u32");
    cdf = pipeline(@"xgb_quantile_weight_cdf");
    select = pipeline(@"xgb_quantile_select");
  }
  id<MTLBuffer> Allocate(size_t bytes) {
    return [device newBufferWithLength:std::max<size_t>(bytes, 16)
                               options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> Upload(void const* data, size_t bytes) {
    auto result = Allocate(bytes);
    if (bytes) std::memcpy(result.contents, data, bytes);
    return result;
  }
  void Finish(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    Check(command.status == MTLCommandBufferStatusCompleted, "Sort/quantile command failed");
  }
  id<MTLBuffer> Prefix(id<MTLCommandBuffer> command, id<MTLBuffer> input, uint32_t count) {
    uint32_t groups = (uint64_t(count) + 255) / 256;
    auto output = Allocate(size_t(count) * 4), totals = Allocate(size_t(groups) * 4);
    auto e = [command computeCommandEncoder];
    [e setComputePipelineState:scan];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBuffer:totals offset:0 atIndex:2];
    [e setBytes:&count length:sizeof(count) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    if (groups > 1) {
      auto upper = Prefix(command, totals, groups);
      e = [command computeCommandEncoder];
      [e setComputePipelineState:add];
      [e setBuffer:output offset:0 atIndex:0];
      [e setBuffer:upper offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    return output;
  }
  id<MTLBuffer> Sort(id<MTLCommandBuffer> command, id<MTLBuffer> input, Shape shape) {
    auto source = Allocate(shape.rows * shape.targets * sizeof(Record)),
         target = Allocate(source.length);
    auto ranks = Allocate(shape.rows * shape.targets * 4);
    uint32_t count = 16 * shape.blocks * shape.targets;
    auto counts = Allocate(size_t(count) * 4);
    auto e = [command computeCommandEncoder];
    [e setComputePipelineState:init];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:source offset:0 atIndex:1];
    [e setBytes:&shape length:sizeof(shape) atIndex:2];
    [e dispatchThreads:MTLSizeMake(shape.rows, shape.targets, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    for (uint32_t shift = 0; shift < 32; shift += 4) {
      Radix p{uint32_t(shape.rows), shape.targets, shape.blocks, shift};
      e = [command computeCommandEncoder];
      [e setComputePipelineState:rank];
      [e setBuffer:source offset:0 atIndex:0];
      [e setBuffer:ranks offset:0 atIndex:1];
      [e setBuffer:counts offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(shape.blocks, shape.targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto prefix = Prefix(command, counts, count);
      e = [command computeCommandEncoder];
      [e setComputePipelineState:scatter];
      [e setBuffer:source offset:0 atIndex:0];
      [e setBuffer:ranks offset:0 atIndex:1];
      [e setBuffer:prefix offset:0 atIndex:2];
      [e setBuffer:target offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreads:MTLSizeMake(shape.rows, shape.targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      std::swap(source, target);
    }
    return source;
  }
};
float Quantile(std::vector<float> const& values, std::vector<uint32_t> const& order,
               std::vector<float> const& weights, Shape shape, Request request,
               std::vector<float>* cdf) {
  auto val = [&](size_t i) {
    return values[order[i] * shape.stride0 + request.target * shape.stride1];
  };
  if (!weights.empty()) {
    cdf->resize(order.size());
    (*cdf)[0] = weights[order[0]];
    for (size_t i = 1; i < order.size(); ++i) (*cdf)[i] = (*cdf)[i - 1] + weights[order[i]];
    // CUDA WeightedQuantileSegOp keeps both the threshold and its comparison
    // in binary64. The CPU WeightedQuantile helper instead narrows to float.
    double threshold = cdf->back() * request.alpha;
    size_t i = std::lower_bound(cdf->begin(), cdf->end(), threshold) - cdf->begin();
    return val(std::min(i, order.size() - 1));
  }
  double n = order.size(), alpha = request.alpha;
  if (alpha <= 1 / (n + 1)) return val(0);
  if (alpha >= n / (n + 1)) return val(order.size() - 1);
  double x = alpha * (n + 1), k = std::floor(x) - 1, d = (x - 1) - k;
  float v0 = val(size_t(k)), v1 = val(size_t(k) + 1);
  float delta = v1 - v0;
  return float(v0 + d * delta);
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production Metal shader");
      GPU gpu(argv[1]);
      std::mt19937 rng(191);
      size_t prefix_values = 0, sorted_values = 0, cdf_values = 0, quantiles = 0, cases = 0;
      for (uint32_t n : {1, 31, 32, 255, 256, 257, 4097, 65537, 1000003}) {
        for (bool wrap : {false, true}) {
          std::vector<uint32_t> input(n), expected(n);
          uint32_t sum = 0;
          for (uint32_t i = 0; i < n; ++i) {
            input[i] = wrap ? rng() : rng() % 17;
            expected[i] = sum;
            sum += input[i];
          }
          auto data = gpu.Upload(input.data(), input.size() * 4);
          auto command = [gpu.queue commandBuffer];
          auto output = gpu.Prefix(command, data, n);
          gpu.Finish(command);
          Check(std::memcmp(output.contents, expected.data(), expected.size() * 4) == 0,
                "Integer exclusive prefix differs from CPU modular sum");
          prefix_values += n;
        }
      }
      printf("Checked %zu integer prefix values\n", prefix_values);
      for (uint32_t rows : {1, 31, 32, 255, 256, 257, 4097, 65537}) {
        for (uint32_t targets : {1, 3})
          for (int layout = 0; layout < 3; ++layout) {
            uint64_t stride0 = layout == 1 ? 1 : (layout == 2 ? targets * 2 : targets);
            uint64_t stride1 = layout == 1 ? rows + 3 : (layout == 2 ? 2 : 1);
            Shape shape{rows, stride0, stride1, targets, (rows + 255) / 256};
            for (int mode = 0; mode < 5; ++mode) {
              std::vector<float> values((rows - 1) * stride0 + (targets - 1) * stride1 + 4,
                                        -91.25f);
              std::vector<float> weights(rows);
              for (uint32_t row = 0; row < rows; ++row) {
                weights[row] = float(rng() % 97) / 7;
                if (mode == 1) weights[row] = Float(rng() & 0x007fffffu);
                if (mode == 2) weights[row] = row % 7 ? 0.0f : -0.0f;
                if (mode == 3) weights[row] = std::ldexp(weights[row], int(rng() % 201) - 100);
                if (mode == 4) weights[row] = row % 3 ? std::numeric_limits<float>::max() : 0;
                for (uint32_t t = 0; t < targets; ++t) {
                  float v = float(int(rng() % 1031) - 515) / 3;
                  if (mode == 1) v = Float(rng() & 0x807fffffu);
                  if (mode == 2) v = row % 2 ? 0.0f : -0.0f;
                  if (mode == 3) v = float(rows - row);
                  if (mode == 4) {
                    if (row % 5 == 0) v = std::numeric_limits<float>::max();
                    if (row % 5 == 1) v = -std::numeric_limits<float>::max();
                    if (row % 5 == 2) v = std::numeric_limits<float>::infinity();
                    if (row % 5 == 3) v = -std::numeric_limits<float>::infinity();
                  }
                  values[row * stride0 + t * stride1] = v;
                }
              }
              std::vector<std::vector<uint32_t>> order(targets, std::vector<uint32_t>(rows));
              for (uint32_t t = 0; t < targets; ++t) {
                std::iota(order[t].begin(), order[t].end(), 0);
                std::stable_sort(order[t].begin(), order[t].end(), [&](uint32_t a, uint32_t b) {
                  return values[a * stride0 + t * stride1] < values[b * stride0 + t * stride1];
                });
              }
              auto input = gpu.Upload(values.data(), values.size() * sizeof(float));
              auto command = [gpu.queue commandBuffer];
              auto sorted = gpu.Sort(command, input, shape);
              gpu.Finish(command);
              auto actual = static_cast<Record const*>(sorted.contents);
              for (uint32_t t = 0; t < targets; ++t)
                for (uint32_t row = 0; row < rows; ++row) {
                  if (actual[size_t(t) * rows + row].row != order[t][row]) {
                    fprintf(stderr, "Sort mismatch rows=%u target=%u layout=%d mode=%d row=%u\n",
                            rows, t, layout, mode, row);
                    throw std::runtime_error("Stable GPU sort differs from std::stable_sort");
                  }
                }
              sorted_values += size_t(rows) * targets;
              for (uint32_t weighted : {0u, 1u}) {
                std::vector<Request> requests;
                for (uint32_t t = 0; t < targets; ++t)
                  for (double alpha : {0.0, 1e-9, 1 / (double(rows) + 1), .1, .5, .9,
                                       double(rows) / (double(rows) + 1), 1 - 1e-9, 1.0})
                    requests.push_back({t, 0, alpha});
                std::vector<float> expected, empty_weights;
                std::vector<float> all_cdf(size_t(rows) * targets), cpu_cdf;
                for (auto request : requests) {
                  expected.push_back(Quantile(values, order[request.target],
                                              weighted ? weights : empty_weights, shape, request,
                                              &cpu_cdf));
                  if (weighted)
                    std::copy(cpu_cdf.begin(), cpu_cdf.end(),
                              all_cdf.begin() + size_t(request.target) * rows);
                }
                command = [gpu.queue commandBuffer];
                auto cdf = gpu.Allocate(all_cdf.size() * 4);
                if (weighted) {
                  auto w = gpu.Upload(weights.data(), weights.size() * 4);
                  auto e = [command computeCommandEncoder];
                  [e setComputePipelineState:gpu.cdf];
                  [e setBuffer:sorted offset:0 atIndex:0];
                  [e setBuffer:w offset:0 atIndex:1];
                  [e setBuffer:cdf offset:0 atIndex:2];
                  [e setBytes:&rows length:sizeof(rows) atIndex:3];
                  [e dispatchThreadgroups:MTLSizeMake(targets, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                  [e endEncoding];
                }
                auto task = gpu.Upload(requests.data(), requests.size() * sizeof(Request));
                auto output = gpu.Allocate(requests.size() * 4);
                Params p{rows, targets, weighted, uint32_t(requests.size())};
                auto e = [command computeCommandEncoder];
                [e setComputePipelineState:gpu.select];
                [e setBuffer:sorted offset:0 atIndex:0];
                [e setBuffer:input offset:0 atIndex:1];
                [e setBuffer:cdf offset:0 atIndex:2];
                [e setBuffer:task offset:0 atIndex:3];
                [e setBuffer:output offset:0 atIndex:4];
                [e setBytes:&p length:sizeof(p) atIndex:5];
                [e setBytes:&shape length:sizeof(shape) atIndex:6];
                [e dispatchThreads:MTLSizeMake(requests.size(), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
                gpu.Finish(command);
                if (weighted) {
                  auto actual_cdf = static_cast<uint32_t const*>(cdf.contents);
                  for (size_t i = 0; i < all_cdf.size(); ++i)
                    Check(actual_cdf[i] == Bits(all_cdf[i]),
                          "Weighted CDF differs from sequential CPU FP32 addition");
                  cdf_values += all_cdf.size();
                }
                auto result = static_cast<float const*>(output.contents);
                for (size_t i = 0; i < requests.size(); ++i) {
                  if (!(std::isnan(result[i]) && std::isnan(expected[i])) &&
                      Bits(result[i]) != Bits(expected[i])) {
                    fprintf(stderr,
                            "Quantile mismatch rows=%u target=%u layout=%d mode=%d weighted=%u "
                            "alpha=%.17g actual=%.9g expected=%.9g\n",
                            rows, requests[i].target, layout, mode, weighted, requests[i].alpha,
                            result[i], expected[i]);
                    throw std::runtime_error(
                        "GPU quantile differs from CPU interpolation/selection");
                  }
                }
                quantiles += requests.size();
              }
              Check(std::memcmp(input.contents, values.data(), values.size() * 4) == 0,
                    "Sort/quantile modified original values or stride padding");
              ++cases;
            }
          }
        printf("Checked full stable sorts and quantiles for %u rows\n", rows);
        fflush(stdout);
      }
      printf("PASS %zu sort/quantile cases / %zu stable records / %zu exact CDF entries / %zu "
             "exact quantiles / %zu prefix values on %s\n",
             cases, sorted_values, cdf_values, quantiles, prefix_values,
             gpu.device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
