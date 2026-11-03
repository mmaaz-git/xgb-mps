// Mean kernels checked against independent compensated CPU arithmetic.
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
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct Shape {
  uint64_t rows, stride0, stride1;
  uint32_t targets, blocks;
};
struct Params {
  Shape shape;
  uint32_t weighted, float32_terms;
};
struct Pair {
  double g, h;
};
static_assert(sizeof(Shape) == 32 && sizeof(Params) == 40);
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
struct AccurateSum {
  double sum{0}, correction{0}, magnitude{0};
  void Add(double value) {
    double next = sum + value;
    correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
    magnitude += std::abs(value);
  }
  double Value() const { return sum + correction; }
};
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production Metal source");
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
      auto pipeline = [&](NSString* name) {
        auto f = [library newFunctionWithName:name];
        auto p = [device newComputePipelineStateWithFunction:f error:&error];
        Check(p != nil, "Missing mean pipeline");
        return p;
      };
      auto weight_reduce = pipeline(@"xgb_mean_weight_reduce"),
           mean_reduce = pipeline(@"xgb_mean_reduce"),
           finish = pipeline(@"xgb_gradient_sum_finish"), result = pipeline(@"xgb_mean_result");
      auto queue = [device newCommandQueue];
      auto allocate = [&](size_t bytes) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      };
      auto upload = [&](void const* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
      };
      auto finish_sums = [&](id<MTLCommandBuffer> command, id<MTLBuffer> partial, Shape shape) {
        auto sum = allocate(shape.targets * sizeof(Pair));
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:finish];
        [e setBuffer:partial offset:0 atIndex:0];
        [e setBuffer:sum offset:0 atIndex:1];
        [e setBytes:&shape length:sizeof(shape) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(shape.targets, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        return sum;
      };
      std::mt19937 rng(918);
      size_t cases = 0, values_checked = 0;
      for (uint64_t rows : {1, 31, 32, 255, 256, 257, 4097, 65537}) {
        for (uint32_t targets : {1, 3, 7})
          for (int layout = 0; layout < 3; ++layout) {
            uint64_t stride0 = layout == 1 ? 1 : (layout == 2 ? targets * 2 : targets);
            uint64_t stride1 = layout == 1 ? rows + 3 : (layout == 2 ? 2 : 1);
            for (uint32_t kind = 0; kind < 3; ++kind)
              for (int mode = 0; mode < 5; ++mode) {
                Params p{{rows, stride0, stride1, targets, uint32_t((rows + 255) / 256)},
                         uint32_t(kind == 2),
                         uint32_t(kind == 0)};
                std::vector<float> values((rows - 1) * stride0 + (targets - 1) * stride1 + 4,
                                          -77.5f),
                    weights(rows);
                AccurateSum sum_weight;
                for (uint64_t row = 0; row < rows; ++row) {
                  float w = float(1 + rng() % 31) / 7;
                  if (mode == 1) w = 0;
                  if (mode == 2) w *= 1e-30f;
                  if (mode == 3) w = std::ldexp(w, int(rng() % 201) - 100);
                  weights[row] = w;
                  sum_weight.Add(w);
                  for (uint32_t t = 0; t < targets; ++t) {
                    float v = float(int(rng() % 2048) - 1024) / 3;
                    if (mode == 1) v = 0;
                    if (mode == 2)
                      v = float(int(rng() % 127) - 63) * std::numeric_limits<float>::denorm_min();
                    if (mode == 3) v = std::ldexp(v, int(rng() % 201) - 100);
                    if (mode == 4) v = 127 * std::ldexp(1.0f, -30);
                    values[row * stride0 + t * stride1] = v;
                  }
                }
                auto input = upload(values.data(), values.size() * sizeof(float)),
                     w = upload(weights.data(), weights.size() * sizeof(float));
                auto denominator = allocate(sizeof(Pair));
                auto command = [queue commandBuffer];
                if (p.weighted) {
                  auto partial = allocate(p.shape.blocks * sizeof(Pair));
                  auto e = [command computeCommandEncoder];
                  [e setComputePipelineState:weight_reduce];
                  [e setBuffer:w offset:0 atIndex:0];
                  [e setBuffer:partial offset:0 atIndex:1];
                  [e setBytes:&p.shape length:sizeof(p.shape) atIndex:2];
                  [e dispatchThreadgroups:MTLSizeMake(p.shape.blocks, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                  [e endEncoding];
                  auto shape = p.shape;
                  shape.targets = 1;
                  denominator = finish_sums(command, partial, shape);
                }
                auto partial = allocate(size_t(p.shape.blocks) * targets * sizeof(Pair));
                auto e = [command computeCommandEncoder];
                [e setComputePipelineState:mean_reduce];
                [e setBuffer:input offset:0 atIndex:0];
                [e setBuffer:w offset:0 atIndex:1];
                [e setBuffer:denominator offset:0 atIndex:2];
                [e setBuffer:partial offset:0 atIndex:3];
                [e setBytes:&p length:sizeof(p) atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake(p.shape.blocks, targets, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
                auto sums = finish_sums(command, partial, p.shape);
                auto output = allocate(targets * sizeof(float));
                e = [command computeCommandEncoder];
                [e setComputePipelineState:result];
                [e setBuffer:sums offset:0 atIndex:0];
                [e setBuffer:output offset:0 atIndex:1];
                [e dispatchThreads:MTLSizeMake(targets, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
                [command commit];
                [command waitUntilCompleted];
                Check(command.status == MTLCommandBufferStatusCompleted, "Mean command failed");
                auto actual = static_cast<float const*>(output.contents);
                if (p.weighted) {
                  double actual_weight = static_cast<Pair const*>(denominator.contents)->g;
                  Check(
                      std::abs(actual_weight - sum_weight.Value()) <= sum_weight.magnitude * 8e-14,
                      "Weight sum differs from CPU compensated reference");
                }
                double divisor = p.weighted
                                     ? sum_weight.Value()
                                     : (p.float32_terms ? double(float(rows)) : double(rows));
                for (uint32_t t = 0; t < targets; ++t) {
                  AccurateSum reference;
                  for (uint64_t row = 0; row < rows; ++row) {
                    double term =
                        divisor > 0 ? double(values[row * stride0 + t * stride1]) / divisor : 0;
                    if (p.weighted) term *= weights[row];
                    if (p.float32_terms) term = float(term);
                    reference.Add(term);
                  }
                  float expected = float(reference.Value());
                  double error = std::abs(double(actual[t]) - expected);
                  double tolerance = std::abs(double(expected)) * 2e-7 +
                                     reference.magnitude * 1e-13 +
                                     2 * double(std::numeric_limits<float>::denorm_min());
                  Check(error <= tolerance, "Mean differs from compensated CPU arithmetic");
                  if (mode == 4)
                    Check(actual[t] == expected, "Constant mean not preserved exactly");
                }
                Check(
                    std::memcmp(input.contents, values.data(), values.size() * sizeof(float)) == 0,
                    "Mean kernel modified input or padding");
                ++cases;
                values_checked += rows * targets;
              }
          }
        printf("Checked means for %llu rows\n", (unsigned long long)rows);
      }
      printf("PASS %zu mean cases / %zu values on %s\n", cases, values_checked,
             device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
