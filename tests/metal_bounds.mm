// Production quantization checked against scalar translations of CUDA's
// quantiser.cu::Clip/BuildQuantiserFromPair and quantiser.cuh::ToFixedPointImpl.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
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

struct Scale {
  int32_t g, h;
  uint32_t valid, padding;
};
static_assert(sizeof(Scale) == 16);
void Require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
float FromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
// Independent CPU floating-point reference; the shader decodes the exponent
// with integer bits. Preserve the CUDA frexp behavior at exact powers of two.
int ReferenceExponent(double positive, double negative, uint64_t rows) {
  double bound = std::max(positive, negative);
  double delta = bound / (1.0 - 2.0 * double(rows) * std::numeric_limits<double>::epsilon());
  int exponent = 0;
  std::frexp(delta, &exponent);
  double rounding = std::ldexp(1.0, exponent);
  double to_float = rounding / double(int64_t{1} << 62);
  double to_fixed = 1.0 / to_float;
  int factor_exponent = 0;
  Require(std::frexp(to_fixed, &factor_exponent) == .5, "Non-power-of-two reference factor");
  return factor_exponent - 1;
}
int64_t ReferenceFixed(float value, int exponent, bool hessian) {
  auto fixed = static_cast<int64_t>(double(value) * std::ldexp(1.0, exponent));
  return hessian && value > 0 && fixed == 0 ? 1 : fixed;
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass the production Metal shader");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal GPU");
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
        auto function = [library newFunctionWithName:name];
        auto result = [device newComputePipelineStateWithFunction:function error:&error];
        Require(result != nil, "Missing gradient-bound pipeline");
        return result;
      };
      auto reduce = pipeline(@"xgb_gradient_bounds"), sum = pipeline(@"xgb_gradient_sum_finish"),
           finish = pipeline(@"xgb_gradient_scale"), quantize = pipeline(@"xgb_quantize_gradients");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
      };
      auto allocate = [&](size_t bytes) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      };
      auto complete = [&](id<MTLCommandBuffer> command) {
        [command commit];
        [command waitUntilCompleted];
        Require(command.status == MTLCommandBufferStatusCompleted,
                "Gradient bounds command failed");
      };
      auto encode_scale = [&](id<MTLCommandBuffer> command, id<MTLBuffer> maximum,
                              id<MTLBuffer> scale, uint64_t rows) {
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:finish];
        [e setBuffer:maximum offset:0 atIndex:0];
        [e setBuffer:scale offset:0 atIndex:1];
        [e setBytes:&rows length:sizeof(rows) atIndex:2];
        [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [e endEncoding];
      };
      auto check_scale = [&](Scale const& scale, std::array<double, 4> const& totals,
                             uint64_t rows) {
        bool valid = rows <= UINT32_MAX;
        for (auto v : totals) valid &= std::isfinite(v);
        Require(scale.valid == valid, "Incorrect finite-gradient/row-limit validation");
        if (valid) {
          Require(scale.g == ReferenceExponent(totals[0], totals[2], rows),
                  "Gradient scale mismatch");
          Require(scale.h == ReferenceExponent(totals[1], totals[3], rows),
                  "Hessian scale mismatch");
        } else {
          Require(scale.g == 0 && scale.h == 0, "Invalid values must have inert scales");
        }
      };
      size_t reduction_cases = 0, values = 0, scale_cases = 0, positive_minima = 0;
      std::mt19937 rng(708);
      for (uint64_t rows : {0, 1, 31, 32, 255, 256, 257, 2048, 4097, 65537, 1000003}) {
        for (int mode = 0; mode < 9; ++mode) {
          std::vector<uint32_t> input(rows * 2, 0);
          std::array<double, 4> expected{};
          std::array<long double, 4> accurate{};
          for (size_t i = 0; i < input.size(); ++i) {
            uint32_t bits = rng();
            if ((bits & 0x7f800000u) == 0x7f800000u) bits &= ~0x00800000u;
            if (mode == 0) bits &= 0x80000000u;
            if (mode == 2) bits &= 0x807fffffu;
            if (mode == 3) bits = 0x7f7fffffu - uint32_t(i % 32);
            if (mode == 4 && i % 97) bits = 0;
            if (mode == 5 && i == input.size() - 2) bits = 0xff800000u;
            if (mode == 6 && i == input.size() - 1) bits = 0x7fc12345u;
            // A few large values, tiny positive Hessians, and tiny negative
            // gradients distinguish CUDA truncation/minimum from old rounding.
            if (mode == 7) bits = i < 2 ? 0x60000000u : (i % 2 ? 1u : 0x80000001u);
            if (mode == 8) bits = i % 3 ? 0xbf800000u : 0x3e800000u;
            input[i] = bits;
            auto slot = i % 2 + (bits >> 31) * 2;
            auto magnitude = std::abs(double(FromBits(bits)));
            expected[slot] += magnitude;
            accurate[slot] += magnitude;
          }
          uint64_t zero[4]{0, 0, 0, 0};
          auto in =
              input.empty() ? allocate(8) : upload(input.data(), input.size() * sizeof(uint32_t));
          auto blocks = (rows + 255) / 256;
          auto partial = allocate(std::max<uint64_t>(blocks, 1) * 32);
          auto totals = upload(zero, sizeof(zero)), scale = allocate(sizeof(Scale));
          // Guards surround the GPU output, including the no-dispatch empty case.
          std::vector<int64_t> guard(input.size() + 4, INT64_C(0x1234567890abcdef));
          auto output = upload(guard.data(), guard.size() * sizeof(int64_t));
          auto command = [queue commandBuffer];
          if (rows) {
            auto e = [command computeCommandEncoder];
            [e setComputePipelineState:reduce];
            [e setBuffer:in offset:0 atIndex:0];
            [e setBuffer:partial offset:0 atIndex:1];
            [e setBytes:&rows length:sizeof(rows) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            struct {
              uint64_t rows, stride0, stride1;
              uint32_t targets, blocks;
            } p{rows, 0, 0, 2, uint32_t(blocks)};
            e = [command computeCommandEncoder];
            [e setComputePipelineState:sum];
            [e setBuffer:partial offset:0 atIndex:0];
            [e setBuffer:totals offset:0 atIndex:1];
            [e setBytes:&p length:sizeof(p) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(2, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
          }
          encode_scale(command, totals, scale, rows);
          if (rows) {
            auto e = [command computeCommandEncoder];
            [e setComputePipelineState:quantize];
            [e setBuffer:in offset:0 atIndex:0];
            [e setBuffer:output offset:16 atIndex:1];
            [e setBytes:&rows length:sizeof(rows) atIndex:2];
            [e setBuffer:scale offset:0 atIndex:3];
            [e dispatchThreads:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
          }
          complete(command);
          auto actual = static_cast<double const*>(totals.contents);
          auto scales = *static_cast<Scale const*>(scale.contents);
          check_scale(scales, expected, rows);
          for (size_t i = 0; i < 4; ++i) {
            if (std::isfinite(expected[i])) {
              // Reduction topology is backend-specific. For dyadic narrow-range
              // fixtures every addition is exact; otherwise bound FP64 rounding
              // against independently accumulated nonnegative magnitudes.
              auto error = std::abs(static_cast<long double>(actual[i]) - accurate[i]);
              auto bound =
                  accurate[i] * std::numeric_limits<double>::epsilon() * (2. + double(rows));
              Require(error <= bound, "Signed bound reduction exceeds FP64 error bound");
              if (mode == 0 || mode == 2 || mode == 3 || mode == 8)
                Require(actual[i] == expected[i], "Exactly representable bound sum differs");
            } else {
              Require(!std::isfinite(actual[i]), "Invalid gradient not propagated");
            }
          }
          auto quantized = static_cast<int64_t const*>(output.contents);
          for (size_t i = 0; i < input.size(); ++i) {
            auto reference = scales.valid ? ReferenceFixed(FromBits(input[i]),
                                                           i % 2 ? scales.h : scales.g, i % 2)
                                          : 0;
            Require(quantized[i + 2] == reference, "Quantization differs from CUDA scalar formula");
            if (scales.valid && i % 2 && FromBits(input[i]) > 0 &&
                std::ldexp(double(FromBits(input[i])), scales.h) < 1.)
              ++positive_minima;
          }
          Require(quantized[0] == guard[0] && quantized[1] == guard[1] &&
                      quantized[input.size() + 2] == guard[0] &&
                      quantized[input.size() + 3] == guard[0],
                  "Quantizer wrote outside output");
          values += input.size();
          ++reduction_cases;
        }
        printf("Checked CUDA quantizer bounds for %llu rows\n", (unsigned long long)rows);
        std::fflush(stdout);
      }
      std::vector<double> boundaries{0.,
                                     std::ldexp(1., -149),
                                     std::ldexp(1., -126),
                                     .5,
                                     1.,
                                     std::nextafter(1., 0.),
                                     std::nextafter(1., 2.),
                                     std::ldexp(1., 160),
                                     double(std::numeric_limits<float>::max()),
                                     std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN()};
      for (uint64_t rows : {0ull, 1ull, 2ull, 3ull, 16777215ull, 16777216ull, 16777217ull,
                            2147483647ull, 4294967295ull, 4294967296ull}) {
        for (auto g : boundaries)
          for (auto h : boundaries) {
            std::array<double, 4> bounds{g, h / 4, g / 2, h};
            auto input = upload(bounds.data(), sizeof(bounds)), scale = allocate(sizeof(Scale));
            auto command = [queue commandBuffer];
            encode_scale(command, input, scale, rows);
            complete(command);
            check_scale(*static_cast<Scale const*>(scale.contents), bounds, rows);
            ++scale_cases;
          }
      }
      Require(positive_minima > 0, "Positive-curvature preservation not exercised");
      printf("PASS %zu CUDA-formula bound reductions / %zu quantized values / %zu positive minima "
             "and %zu scale boundary cases on %s\n",
             reduction_cases, values, positive_minima, scale_cases, device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
