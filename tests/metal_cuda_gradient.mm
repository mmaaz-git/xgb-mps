// Independent arithmetic checks for the CUDA-formula sampling path.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <vector>
#include "cuda_gradient_reference.h"
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
std::uint64_t Bits(double v) {
  std::uint64_t b;
  std::memcpy(&b, &v, 8);
  return b;
}
double Value(std::uint64_t b) {
  double v;
  std::memcpy(&v, &b, 8);
  return v;
}
struct UniformParams {
  std::uint64_t rows, stride0, stride1;
  std::uint32_t targets, seed;
  float probability;
  std::uint32_t pad;
};
struct State {
  int exponent;
  std::uint32_t error, index;
  float threshold;
};
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No GPU");
      auto queue = [device newCommandQueue];
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
    kernel void test_mvs_fma(device const ulong4* input [[buffer(0)]],device ulong* output [[buffer(1)]],uint row [[thread_position_in_grid]]) {
      ulong4 v=input[row]; output[row]=xgb_positive_dfma({v.x},{v.y},{v.z}).bits;
    }
    kernel void test_mvs_conversion(device const ulong2* input [[buffer(0)]],device ulong* output [[buffer(1)]],device atomic_uint* errors [[buffer(2)]],uint row [[thread_position_in_grid]]) {
      ulong2 v=input[row]; output[row]=xgb_cuda_quantize_double({v.x},int(v.y),errors+row);
    }
  )";
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [&](NSString* name) {
        auto result = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                                            error:&error];
        Check(result != nil, "No pipeline");
        return result;
      };
      auto buffer = [&](void const* bytes, std::size_t size) {
        auto result = bytes
                          ? [device newBufferWithBytes:bytes
                                                length:size
                                               options:MTLResourceStorageModeShared]
                          : [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        Check(result != nil, "No buffer");
        return result;
      };
      auto finish = [&](id<MTLCommandBuffer> command) {
        [command commit];
        [command waitUntilCompleted];
        Check(command.status == MTLCommandBufferStatusCompleted, "GPU command failed");
      };
      std::mt19937_64 rng(1711);
      std::vector<std::array<std::uint64_t, 4>> inputs;
      // Entire normal domain used by norms, cancellation-free alignments, exact ties,
      // and zero shortcuts. Hardware std::fma supplies a separately implemented oracle.
      std::vector<double> edge{
          0,       0x1p-298, 0x1p-200, 0x1p-53, 1, 0x1.0000000000001p0, 0x1.fffffffffffffp0,
          0x1p100, 0x1p256};
      for (auto b : edge)
        for (auto c : edge) inputs.push_back({Bits(double(.1f)), Bits(b), Bits(c), 0});
      for (std::size_t i = 0; i < 200000; ++i) {
        double a = i % 2 ? double(.1f)
                         : std::ldexp(double(rng() >> 11) / 0x1p53 + .5, int(rng() % 61) - 30);
        double b = std::ldexp(double(rng() >> 11) / 0x1p53 + .5, int(rng() % 555) - 298);
        double c = std::ldexp(double(rng() >> 11) / 0x1p53 + .5, int(rng() % 555) - 298);
        inputs.push_back({Bits(a), Bits(b), Bits(c), 0});
      }
      auto in = buffer(inputs.data(), inputs.size() * 32), out = buffer(nullptr, inputs.size() * 8);
      auto command = [queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline(@"test_mvs_fma")];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(inputs.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      finish(command);
      auto actual = static_cast<std::uint64_t const*>(out.contents);
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto v = inputs[i];
        auto expected = std::fma(Value(v[0]), Value(v[1]), Value(v[2]));
        if (actual[i] != Bits(expected)) {
          std::fprintf(stderr, "FMA %zu %a * %a + %a: %a != %a\n", i, Value(v[0]), Value(v[1]),
                       Value(v[2]), Value(actual[i]), expected);
          throw std::runtime_error("FMA bits differ");
        }
      }
      std::printf("PASS %zu binary64 norm FMA cases\n", inputs.size());
      std::vector<std::array<std::uint64_t, 2>> conversions;
      std::vector<double> values{0.,
                                 -0.,
                                 0x1p-1074,
                                 -0x1p-1074,
                                 0x1p-1022,
                                 0.5,
                                 -0.5,
                                 1.,
                                 -1.,
                                 0x1p63,
                                 -0x1p63,
                                 std::nextafter(0x1p63, 0.),
                                 std::nextafter(-0x1p63, -INFINITY),
                                 INFINITY,
                                 -INFINITY,
                                 NAN};
      for (auto value : values)
        for (int scale : {-1074, -100, -64, -1, 0, 1, 63, 100, 1074})
          conversions.push_back({Bits(value), std::uint64_t(scale)});
      for (std::size_t i = 0; i < 100000; ++i)
        conversions.push_back({rng(), std::uint64_t(int(rng() % 2201) - 1100)});
      in = buffer(conversions.data(), conversions.size() * 16);
      out = buffer(nullptr, conversions.size() * 8);
      auto errors = buffer(nullptr, conversions.size() * 4);
      std::memset(errors.contents, 0, conversions.size() * 4);
      command = [queue commandBuffer];
      e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline(@"test_mvs_conversion")];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e setBuffer:errors offset:0 atIndex:2];
      [e dispatchThreads:MTLSizeMake(conversions.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      finish(command);
      actual = static_cast<std::uint64_t const*>(out.contents);
      auto flags = static_cast<std::uint32_t const*>(errors.contents);
      for (std::size_t i = 0; i < conversions.size(); ++i) {
        double value = std::ldexp(Value(conversions[i][0]), int(conversions[i][1]));
        bool valid = std::isfinite(value) && value >= -0x1p63 && value < 0x1p63;
        Check(flags[i] == (valid ? 0u : 2u), "Integer conversion overflow flag differs");
        if (valid)
          Check(actual[i] == std::uint64_t(static_cast<std::int64_t>(value)),
                "Integer conversion differs");
      }
      std::printf("PASS %zu binary64/int64 conversion and error cases\n", conversions.size());
      for (std::size_t targets : {1, 16, 17, 31, 32, 33, 511, 512, 513, 4095, 4096, 4097, 16385}) {
        std::size_t rows = 7;
        std::vector<float> terms(rows * targets), expected(rows);
        for (std::size_t row = 0; row < rows; ++row) {
          std::vector<float> values(targets);
          for (std::size_t t = 0; t < targets; ++t) {
            float value = std::ldexp(float(rng() % 65535) / 113, int(rng() % 180) - 149);
            values[t] = terms[t * rows + row] = value;
          }
          expected[row] = cuda_reference::Norm(values);
        }
        struct {
          std::uint64_t rows;
          std::uint32_t targets, budget;
        } params{rows, static_cast<std::uint32_t>(targets), 0};
        in = buffer(terms.data(), terms.size() * 4);
        out = buffer(nullptr, rows * 4);
        command = [queue commandBuffer];
        e = [command computeCommandEncoder];
        [e setComputePipelineState:pipeline(targets <= 16 ? @"xgb_cuda_mvs_norm_small"
                                                          : @"xgb_cuda_mvs_norm")];
        [e setBuffer:in offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e setBytes:&params length:sizeof(params) atIndex:2];
        if (targets <= 16)
          [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else
          [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(targets <= 512 ? 32 : 256, 1, 1)];
        [e endEncoding];
        finish(command);
        Check(!std::memcmp(out.contents, expected.data(), rows * 4),
              "CCCL norm reduction order differs");
      }
      std::puts("PASS 91 norms across 13 CCCL segment widths, through 16385 targets");
      auto scan = pipeline(@"xgb_prefix_u64"), add = pipeline(@"xgb_prefix_u64_add");
      // Recursive levels, block edges and unsigned carries are checked independently.
      for (std::uint64_t count : {1, 255, 256, 257, 65536, 65537, 1000003}) {
        std::vector<std::uint64_t> data(count), expected(count);
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < count; ++i) {
          data[i] = rng();
          sum += data[i];
          expected[i] = sum;
        }
        auto source_buffer = buffer(data.data(), count * 8);
        command = [queue commandBuffer];
        auto encode = [&](auto&& self, id<MTLBuffer> input, std::uint64_t n) -> id<MTLBuffer> {
          std::uint64_t blocks = (n + 255) / 256;
          auto result = buffer(nullptr, n * 8), totals = buffer(nullptr, blocks * 8);
          auto encoder = [command computeCommandEncoder];
          [encoder setComputePipelineState:scan];
          [encoder setBuffer:input offset:0 atIndex:0];
          [encoder setBuffer:result offset:0 atIndex:1];
          [encoder setBuffer:totals offset:0 atIndex:2];
          [encoder setBytes:&n length:8 atIndex:3];
          [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [encoder endEncoding];
          if (blocks > 1) {
            auto upper = self(self, totals, blocks);
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:add];
            [encoder setBuffer:result offset:0 atIndex:0];
            [encoder setBuffer:upper offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
          }
          return result;
        };
        auto result = encode(encode, source_buffer, count);
        finish(command);
        Check(!std::memcmp(result.contents, expected.data(), count * 8),
              "Recursive integer prefix differs");
        std::printf("PASS %llu exact integer prefixes\n", count);
      }
      // Every accepted/rejected integer coordinate, including rescaling without an
      // intervening FP32 round trip. This complements production root checks.
      auto apply = pipeline(@"xgb_cuda_mvs_integer");
      std::size_t checked = 0;
      for (std::size_t targets : {1, 3, 17})
        for (int mode : {0, 1, 2, 3}) {
          std::size_t n = 1027;
          std::vector<cuda_reference::Pair> pairs(n * targets);
          for (auto& p : pairs) {
            p = {float(int(rng() % 65535) - 32767) / 113, float(rng() % 65535) / 271};
            if (mode == 1) {
              p.g = std::ldexp(p.g, -70);
              p.h = std::ldexp(p.h, -70);
            }
            if (mode == 2) {
              p.g = std::ldexp(p.g, int(rng() % 120) - 80);
              p.h = std::ldexp(p.h, int(rng() % 120) - 80);
            }
            if (mode == 3) p = {-2, 3};
          }
          cuda_reference::GradientSampling ref(pairs, n, targets, .37f, 9939);
          Check(!ref.invalid_rescale, "Invalid primitive fixture");
          auto norms = buffer(ref.norms.data(), n * 4);
          State state{0, 0, 0, ref.threshold};
          auto saved = buffer(&state, sizeof(state));
          for (std::size_t t = 0; t < targets; ++t) {
            std::vector<cuda_reference::IntegerPair> column(n), expected(n);
            for (std::size_t r = 0; r < n; ++r) {
              column[r] = ref.original[r * targets + t];
              expected[r] = ref.sampled[r * targets + t];
            }
            auto fixed = buffer(column.data(), n * 16);
            UniformParams p{n, 0, 0, 0, 9939, .37f, 0};
            struct {
              int g, h;
              unsigned valid, pad;
            } scale{ref.scales[t].g, ref.scales[t].h, 1, 0};
            auto scales = buffer(&scale, sizeof(scale));
            command = [queue commandBuffer];
            e = [command computeCommandEncoder];
            [e setComputePipelineState:apply];
            [e setBuffer:fixed offset:0 atIndex:0];
            [e setBytes:&p length:sizeof(p) atIndex:1];
            [e setBuffer:scales offset:0 atIndex:2];
            [e setBuffer:norms offset:0 atIndex:3];
            [e setBuffer:saved offset:0 atIndex:4];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            finish(command);
            Check(static_cast<State const*>(saved.contents)->error == 0,
                  "Unexpected rescaling error");
            Check(!std::memcmp(fixed.contents, expected.data(), n * 16),
                  "Rescaled integer pair differs");
            checked += n;
          }
        }
      std::printf("PASS %zu individual quantized sampling/rescaling pairs\n", checked);
      return 0;
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL %s\n", e.what());
      return 1;
    }
  }
}
