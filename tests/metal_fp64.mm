// Test software binary64 against independent hardware CPU arithmetic.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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
void Require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
uint64_t Bits(double value) {
  uint64_t out;
  memcpy(&out, &value, 8);
  return out;
}
double Value(uint64_t bits) {
  double out;
  memcpy(&out, &bits, 8);
  return out;
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
 kernel void test_fp64(device const ulong2* input [[buffer(0)]],device ulong4* output [[buffer(1)]],device uint2* converted [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
   XGBDouble a{input[tid].x},b{input[tid].y};
   output[tid]=ulong4(xgb_dadd(a,b).bits,xgb_dsub(a,b).bits,xgb_dmul(a,b).bits,xgb_ddiv(a,b).bits);
   converted[tid]=uint2(as_type<uint>(xgb_dto_float(a)),as_type<uint>(xgb_dto_float(b)));
 }
 kernel void test_from_float(device const uint* input [[buffer(0)]],device ulong* output [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
   output[tid]=xgb_dfrom_float(as_type<float>(input[tid])).bits;
 })";
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto function = [library newFunctionWithName:@"test_fp64"];
      auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      Require(pipeline != nil, "No pipeline");
      auto queue = [device newCommandQueue];
      std::mt19937_64 rng(1128);
      std::vector<uint64_t> inputs;
      std::vector<uint64_t> edge{0,
                                 0x8000000000000000ull,
                                 1,
                                 0x8000000000000001ull,
                                 0xfffffffffffffull,
                                 0x10000000000000ull,
                                 0x7fefffffffffffffull,
                                 0x7ff0000000000000ull,
                                 0xfff0000000000000ull,
                                 0x7ff8000000000000ull,
                                 Bits(1),
                                 Bits(-1),
                                 Bits(0.5),
                                 Bits(2)};
      for (auto a : edge)
        for (auto b : edge) {
          inputs.push_back(a);
          inputs.push_back(b);
        }
      for (size_t i = 0; i < 100000; ++i) {
        inputs.push_back(rng());
        inputs.push_back(rng());
      }
      auto count = inputs.size() / 2;
      auto in = [device newBufferWithBytes:inputs.data()
                                    length:inputs.size() * 8
                                   options:MTLResourceStorageModeShared];
      auto out = [device newBufferWithLength:count * 32 options:MTLResourceStorageModeShared];
      auto converted = [device newBufferWithLength:count * 8 options:MTLResourceStorageModeShared];
      auto cmd = [queue commandBuffer];
      auto e = [cmd computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      [e setBuffer:converted offset:0 atIndex:2];
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      Require(cmd.status == MTLCommandBufferStatusCompleted, "GPU failed");
      printf("GPU arithmetic duration: %.3f ms\n", (cmd.GPUEndTime - cmd.GPUStartTime) * 1e3);
      auto actual = static_cast<uint64_t const*>(out.contents);
      auto floats = static_cast<uint32_t const*>(converted.contents);
      for (size_t i = 0; i < count; ++i) {
        volatile double a = Value(inputs[2 * i]), b = Value(inputs[2 * i + 1]);
        double reference[4]{a + b, a - b, a * b, a / b};
        for (size_t op = 0; op < 4; ++op)
          if (!(std::isnan(reference[op]) && std::isnan(Value(actual[4 * i + op]))) &&
              Bits(reference[op]) != actual[4 * i + op]) {
            fprintf(stderr,
                    "Mismatch i=%zu op=%zu a=%a b=%a actual=%a expected=%a bits=%llx/%llx\n", i, op,
                    double(a), double(b), Value(actual[4 * i + op]), reference[op],
                    actual[4 * i + op], Bits(reference[op]));
            throw std::runtime_error("Binary64 arithmetic mismatch");
          }
        for (size_t k = 0; k < 2; ++k) {
          float expected = float(Value(inputs[2 * i + k]));
          uint32_t bits;
          memcpy(&bits, &expected, 4);
          float value;
          memcpy(&value, &floats[2 * i + k], 4);
          Require((std::isnan(expected) && std::isnan(value)) || bits == floats[2 * i + k],
                  "Binary64 to float mismatch");
        }
      }
      std::vector<uint32_t> floats_in;
      for (size_t i = 0; i < 100000; ++i) floats_in.push_back(uint32_t(rng()));
      for (uint32_t x : {0u, 0x80000000u, 1u, 0x7fffffu, 0x800000u, 0x7f800000u, 0xff800000u})
        floats_in.push_back(x);
      auto raw = [device newBufferWithBytes:floats_in.data()
                                     length:floats_in.size() * 4
                                    options:MTLResourceStorageModeShared];
      auto wide = [device newBufferWithLength:floats_in.size() * 8
                                      options:MTLResourceStorageModeShared];
      function = [library newFunctionWithName:@"test_from_float"];
      pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      cmd = [queue commandBuffer];
      e = [cmd computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:raw offset:0 atIndex:0];
      [e setBuffer:wide offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(floats_in.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      Require(cmd.status == MTLCommandBufferStatusCompleted, "GPU conversion failed");
      actual = static_cast<uint64_t const*>(wide.contents);
      for (size_t i = 0; i < floats_in.size(); ++i) {
        float value;
        memcpy(&value, &floats_in[i], 4);
        double expected = value;
        Require(
            (std::isnan(expected) && std::isnan(Value(actual[i]))) || Bits(expected) == actual[i],
            "Float to binary64 mismatch");
      }
      printf("PASS %zu binary64 arithmetic operations, %zu narrowing and %zu widening conversions "
             "on %s\n",
             count * 4, count * 2, floats_in.size(), device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
