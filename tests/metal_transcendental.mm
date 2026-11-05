// Validate GPU binary64 exp/log/erf against independent CPU libm, including
// tails.
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
#include "math_contract_io.h"
uint64_t Bits(double value) {
  uint64_t b;
  std::memcpy(&b, &value, 8);
  return b;
}
double Value(uint64_t b) {
  double v;
  std::memcpy(&v, &b, 8);
  return v;
}
void Require(bool ok, char const *message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      auto report = math_contracts::ReportOption(argc, argv, 4);
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
    kernel void test_transcendental(device const ulong* in [[buffer(0)]],
      device ulong4* out [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
      XGBDouble x{in[tid]};
      out[tid]=ulong4(xgb_survival::Exp(x).bits,xgb_survival::Log(x).bits,
                       xgb_survival::Erf(x).bits,0);
    })";
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError *error = nil;
      auto lib = [device
          newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                       options:options
                         error:&error];
      if (!lib)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline =
          [device newComputePipelineStateWithFunction:
                      [lib newFunctionWithName:@"test_transcendental"]
                                                error:&error];
      Require(pipeline != nil, "No pipeline");
      auto queue = [device newCommandQueue];
      std::vector<uint64_t> in;
      for (double v : {0.,
                       -0.,
                       1.,
                       -1.,
                       0.84375,
                       1.25,
                       6.,
                       8.,
                       -6.,
                       -8.,
                       709.782712893384,
                       -745.1332191019411,
                       2.857142857142857,
                       1e-300,
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::denorm_min(),
                       std::numeric_limits<double>::min(),
                       std::numeric_limits<double>::max()}) {
        in.push_back(Bits(v));
        in.push_back(Bits(std::nextafter(v, -INFINITY)));
        in.push_back(Bits(std::nextafter(v, INFINITY)));
      }
      std::mt19937_64 rng(927);
      for (size_t i = 0; i < 20000; ++i) {
        in.push_back(rng());
        in.push_back(
            Bits(std::uniform_real_distribution<double>(-746, 710)(rng)));
        in.push_back(Bits(std::uniform_real_distribution<double>(-8, 8)(rng)));
        in.push_back(Bits(
            std::uniform_real_distribution<double>(0.999999, 1.000001)(rng)));
      }
      auto input = [device newBufferWithBytes:in.data()
                                       length:in.size() * 8
                                      options:MTLResourceStorageModeShared];
      auto output = [device newBufferWithLength:in.size() * 32
                                        options:MTLResourceStorageModeShared];
      auto command = [queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:output offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(in.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      Require(command.status == MTLCommandBufferStatusCompleted,
              "GPU execution failed");
      auto result = static_cast<uint64_t const *>(output.contents);
      std::array<uint64_t, 3> max_ulp{};
      size_t mismatches = 0;
      for (size_t i = 0; i < in.size(); ++i) {
        double x = Value(in[i]);
        std::array<double, 3> expected{std::exp(x), std::log(x), std::erf(x)};
        for (size_t k = 0; k < 3; ++k) {
          double actual = Value(result[4 * i + k]);
          if (std::isnan(expected[k]) && std::isnan(actual))
            continue;
          auto a = result[4 * i + k], b = Bits(expected[k]);
          auto ulp = a > b ? a - b : b - a;
          max_ulp[k] = std::max(max_ulp[k], ulp);
          if (ulp > 4 || (std::isinf(actual) != std::isinf(expected[k]))) {
            if (++mismatches <= 10)
              std::fprintf(
                  stderr,
                  "op%zu x=%.17g actual=%.17g expected=%.17g ULP=%llu\n", k, x,
                  actual, expected[k], (unsigned long long)ulp);
          }
        }
      }
      std::printf("Checked %zu exp/log/erf values; maximum ULP %llu/%llu/%llu; "
                  "failures %zu\n",
                  in.size() * 3, (unsigned long long)max_ulp[0],
                  (unsigned long long)max_ulp[1],
                  (unsigned long long)max_ulp[2], mismatches);
      Require(mismatches == 0,
              "Transcendental error exceeds four binary64 ULP");
      if (report) {
        math_contracts::Words observations;
        for (std::size_t i = 0; i < in.size(); ++i)
          for (unsigned k = 0; k < 3; ++k) observations.push_back(result[4*i+k]);
        report->Add(2, in, observations);
        report->Finish();
      }
      std::printf("PASS on %s\n", device.name.UTF8String);
      return 0;
    } catch (std::exception const &e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
