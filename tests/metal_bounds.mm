// Production gradient bounds and quantization checked against CPU arithmetic.
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
// Brute-force the largest legal exponent using CPU frexp and exact 128-bit
// comparisons, independently of the shader's integer logarithm algorithm.
int ReferenceExponent(uint32_t bits, uint64_t rows) {
  if (bits == 0 || rows == 0) return 0;
  int exponent;
  double fraction = std::frexp(double(FromBits(bits)), &exponent);
  auto mantissa = uint64_t(std::ldexp(fraction, 24));
  __uint128_t product = __uint128_t(mantissa) * rows;
  auto fits = [&](int candidate) {
    int shift = candidate + exponent - 24;
    if (shift > 60) return false;
    if (shift < -67) return true;
    return shift >= 0 ? (product << shift) <= (__uint128_t(1) << 60)
                      : product <= (__uint128_t(1) << (60 - shift));
  };
  int result = 0;
  while (!fits(result)) --result;
  while (fits(result + 1)) ++result;
  return result;
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
      auto reduce = pipeline(@"xgb_gradient_maximum"), finish = pipeline(@"xgb_gradient_scale"),
           quantize = pipeline(@"xgb_quantize_gradients");
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
      auto check_scale = [&](Scale const& scale, uint32_t const* maximum, uint64_t rows) {
        bool valid = std::isfinite(FromBits(maximum[0])) && std::isfinite(FromBits(maximum[1]));
        Require(scale.valid == valid, "Incorrect finite-gradient validation");
        if (valid) {
          Require(scale.g == ReferenceExponent(maximum[0], rows), "Gradient scale mismatch");
          Require(scale.h == ReferenceExponent(maximum[1], rows), "Hessian scale mismatch");
        } else {
          Require(scale.g == 0 && scale.h == 0, "Invalid values must have inert scales");
        }
      };
      size_t reduction_cases = 0, values = 0, scale_cases = 0;
      std::mt19937 rng(708);
      for (uint64_t rows : {1, 31, 32, 255, 256, 257, 2048, 4097, 65537}) {
        for (int mode = 0; mode < 7; ++mode) {
          std::vector<uint32_t> input(rows * 2, 0);
          uint32_t expected[2]{0, 0};
          for (size_t i = 0; i < input.size(); ++i) {
            uint32_t bits = rng();
            if ((bits & 0x7f800000u) == 0x7f800000u) bits &= ~0x00800000u;
            if (mode == 0) bits &= 0x80000000u;
            if (mode == 2) bits &= 0x807fffffu;
            if (mode == 3) bits = 0x7f7fffffu - uint32_t(i % 32);
            if (mode == 4 && i % 97) bits = 0;
            if (mode == 5 && i == input.size() - 2) bits = 0xff800000u;
            if (mode == 6 && i == input.size() - 1) bits = 0x7fc12345u;
            input[i] = bits;
            expected[i % 2] = std::max(expected[i % 2], bits & 0x7fffffffu);
          }
          uint32_t zero[2]{0, 0};
          auto in = upload(input.data(), input.size() * sizeof(uint32_t));
          auto maximum = upload(zero, sizeof(zero)), scale = allocate(sizeof(Scale));
          auto output = allocate(input.size() * sizeof(int64_t));
          auto command = [queue commandBuffer];
          auto e = [command computeCommandEncoder];
          [e setComputePipelineState:reduce];
          [e setBuffer:in offset:0 atIndex:0];
          [e setBuffer:maximum offset:0 atIndex:1];
          [e setBytes:&rows length:sizeof(rows) atIndex:2];
          [e dispatchThreadgroups:MTLSizeMake((rows + 255) / 256, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [e endEncoding];
          encode_scale(command, maximum, scale, rows);
          e = [command computeCommandEncoder];
          [e setComputePipelineState:quantize];
          [e setBuffer:in offset:0 atIndex:0];
          [e setBuffer:output offset:0 atIndex:1];
          [e setBytes:&rows length:sizeof(rows) atIndex:2];
          [e setBuffer:scale offset:0 atIndex:3];
          [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [e endEncoding];
          complete(command);
          auto actual = static_cast<uint32_t const*>(maximum.contents);
          Require(actual[0] == expected[0] && actual[1] == expected[1],
                  "Maximum reduction mismatch");
          auto scales = *static_cast<Scale const*>(scale.contents);
          check_scale(scales, expected, rows);
          auto quantized = static_cast<int64_t const*>(output.contents);
          for (size_t i = 0; i < input.size(); ++i) {
            auto reference = scales.valid ? std::llround(std::ldexp(double(FromBits(input[i])),
                                                                    i % 2 ? scales.h : scales.g))
                                          : 0;
            Require(quantized[i] == reference, "Bounded quantization differs from CPU llround");
          }
          values += input.size();
          ++reduction_cases;
        }
        printf("Checked gradient bounds for %llu rows\n", (unsigned long long)rows);
      }
      std::vector<uint32_t> boundaries{0,           1,           0x007fffffu, 0x00800000u,
                                       0x3f000000u, 0x3f800000u, 0x3f800001u, 0x3fffffffu,
                                       0x7f7fffffu, 0x7f800000u, 0x7f800001u, 0x7fffffffu};
      for (uint64_t rows : {0ull, 1ull, 2ull, 3ull, 16777215ull, 16777216ull, 16777217ull,
                            2147483647ull, 4294967295ull}) {
        for (auto g : boundaries)
          for (auto h : boundaries) {
            uint32_t maximum[2]{g, h};
            auto input = upload(maximum, sizeof(maximum)), scale = allocate(sizeof(Scale));
            auto command = [queue commandBuffer];
            encode_scale(command, input, scale, rows);
            complete(command);
            check_scale(*static_cast<Scale const*>(scale.contents), maximum, rows);
            ++scale_cases;
          }
      }
      printf("PASS %zu gradient-bound reductions / %zu quantized values and %zu scale boundary "
             "cases on %s\n",
             reduction_cases, values, scale_cases, device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
