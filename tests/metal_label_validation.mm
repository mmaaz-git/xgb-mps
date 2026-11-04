// Production label predicates and API versus upstream CPU label checks.
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
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ops.h"
#include "../xgboost/src/objective/regression_loss.h"

using namespace xgboost;
namespace metal = xgboost::common::metal;
void Check(bool ok, char const* text) {
  if (!ok) throw std::runtime_error(text);
}
float FloatFromBits(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
bool CPU(float x, unsigned domain, int64_t classes) {
  switch (domain) {
    case 0:
      return obj::LinearSquareLoss::CheckLabel(x);
    case 1:
      return obj::LogisticRegression::CheckLabel(x);
    case 2:
      return obj::SquaredLogError::CheckLabel(x);
    case 3:
      return obj::GammaDeviance::CheckLabel(x);
    case 4:
      return obj::PoissonLabel::CheckLabel(x);
    case 5:
      return x >= 0 && x < classes && std::floor(x) == x;
  }
  throw std::runtime_error("Unknown domain");
}
void Predicates(char const* path) {
  std::ifstream f(path);
  std::string source{std::istreambuf_iterator<char>(f), {}};
  source += R"(
    kernel void test_labels(device const uint* input [[buffer(0)]],
      device uint* output [[buffer(1)]], constant float& classes [[buffer(2)]],
      uint i [[thread_position_in_grid]]) {
      uint mask = 0;
      for (uint d = 0; d < 6; ++d) mask |= uint(xgb_valid_label(input[i], d, classes)) << d;
      output[i+1] = mask;
    })";
  auto device = MTLCreateSystemDefaultDevice();
  Check(device != nil, "No Metal device");
  auto options = [MTLCompileOptions new];
  options.fastMathEnabled = NO;
  NSError* error = nil;
  auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
  if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
  auto pipeline =
      [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"test_labels"]
                                            error:&error];
  Check(pipeline != nil, "No label pipeline");
  auto queue = [device newCommandQueue];
  std::vector<uint32_t> input{0,          1,          2,          0x7fffff,   0x800000,
                              0x800001,   0x3f7fffff, 0x3f800000, 0x3f800001, 0x4b7fffff,
                              0x4b800000, 0x4b800001, 0x4effffff, 0x4f000000, 0x4f000001,
                              0x7f7fffff, 0x7f800000, 0x7f800001, 0x7fc12345};
  auto positives = input;
  for (auto bits : positives) input.push_back(bits | 0x80000000u);
  std::mt19937 rng(76281);
  for (uint32_t i = 0; i < 4000000; ++i) input.push_back(rng());
  for (uint32_t e = 0; e < 256; ++e)
    for (uint32_t m : {0u, 1u, 0x3fffffu, 0x400000u, 0x7ffffeu, 0x7fffffu}) {
      input.push_back((e << 23) | m);
      input.push_back(0x80000000u | (e << 23) | m);
    }
  std::size_t checked = 0;
  for (int64_t classes : {int64_t{1}, int64_t{3}, int64_t{16777217}, int64_t{2147483647},
                          std::numeric_limits<int64_t>::max()}) {
    for (std::size_t begin = 0; begin < input.size(); begin += 1048576) {
      auto n = std::min<std::size_t>(1048576, input.size() - begin);
      auto in = [device newBufferWithBytes:input.data() + begin
                                    length:n * 4
                                   options:MTLResourceStorageModeShared];
      auto out = [device newBufferWithLength:(n + 2) * 4 options:MTLResourceStorageModeShared];
      auto actual = static_cast<uint32_t*>(out.contents);
      std::fill_n(actual, n + 2, 0x917abcdeu);
      auto command = [queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:in offset:0 atIndex:0];
      [e setBuffer:out offset:0 atIndex:1];
      float bound = static_cast<float>(classes);
      [e setBytes:&bound length:4 atIndex:2];
      [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      Check(command.status == MTLCommandBufferStatusCompleted, "Label GPU execution failed");
      Check(actual[0] == 0x917abcdeu && actual[n + 1] == 0x917abcdeu,
            "Output guard changed");
      Check(!std::memcmp(in.contents, input.data() + begin, n * 4), "Input changed");
      for (std::size_t i = 0; i < n; ++i) {
        uint32_t expected = 0;
        for (uint32_t d = 0; d < 6; ++d)
          expected |= uint32_t(CPU(FloatFromBits(input[begin + i]), d, classes)) << d;
        if (actual[i + 1] != expected) {
          std::printf("label %08x classes %lld: GPU %x CPU %x\n", input[begin + i],
                      static_cast<long long>(classes), actual[i + 1], expected);
          throw std::runtime_error("Predicate differs from CPU");
        }
      }
      checked += n * 6;
    }
    std::printf("Checked %zu individual label predicates\n", checked);
    std::fflush(stdout);
  }
  std::printf("PASS %zu exact individual label predicates\n", checked);
}
void API() {
  std::size_t cases = 0, values = 0;
  for (unsigned d = 0; d < 6; ++d)
    for (std::size_t rows : {0u, 1u, 31u, 257u, 4097u, 1000003u})
      for (std::size_t columns : {1u, 3u})
        for (int layout = 0; layout < 3; ++layout) {
          std::size_t shape[2]{rows, columns};
          std::size_t strides[2]{layout == 1 ? 1 : columns * 2 + 3, layout == 1 ? rows + 5 : 2};
          if (layout == 2) strides[0] = 0;  // Broadcast row view.
          std::vector<float> storage(rows * strides[0] + columns * strides[1] + 2, NAN);
          linalg::MatrixView<float> view{storage, shape, strides, DeviceOrd::CPU()};
          for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < columns; ++j) view(i, j) = d == 5 ? float(j) : .25f;
          auto check = [&] {
            auto original = storage;
            bool expected = true;
            for (std::size_t i = 0; i < rows; ++i)
              for (std::size_t j = 0; j < columns; ++j) expected &= CPU(view(i, j), d, 3);
            auto before = metal::GetStatistics();
            bool actual = metal::ValidateLabels(static_cast<metal::LabelDomain>(d), view, 3);
            auto after = metal::GetStatistics();
            Check(actual == expected, "Production validation differs from CPU");
            Check(after.label_validation_dispatches - before.label_validation_dispatches ==
                      bool(rows && d),
                  "Missing label validation dispatch");
            Check(!rows || !d ||
                      after.label_validation_gpu_time_ns > before.label_validation_gpu_time_ns,
                  "Missing label validation GPU duration");
            Check(!std::memcmp(storage.data(), original.data(), storage.size() * 4),
                  "Source/guard changed");
            ++cases;
            values += rows * columns;
          };
          check();
          if (rows) {
            view(rows - 1, columns - 1) = d == 2 ? -1.f : d == 3 ? -0.f : -FloatFromBits(1);
            check();
            view(rows / 2, 0) = NAN;
            check();
          }
        }
  std::printf("PASS %zu production validation cases / %zu tensor values; row/column/broadcast "
              "strides and guards\n",
              cases, values);
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass the production shader");
      Predicates(argv[1]);
      API();
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
