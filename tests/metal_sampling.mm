// Match actual libc++ Bernoulli draws and XGBoost's parallel LCG skip-ahead.
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
struct Pair {
  float g, h;
};
struct Params {
  uint64_t rows, stride0, stride1, seed;
  uint32_t targets, threads;
  float probability;
  uint32_t padding;
};
static_assert(sizeof(Params) == 48);
using Engine = std::linear_congruential_engine<uint64_t, 16807, 0, uint64_t(1) << 63>;
uint64_t Skip(uint64_t exponent, uint64_t seed) {
  uint64_t result = 1, base = 16807, mod = uint64_t(1) << 63;
  while (exponent) {
    if (exponent % 2) result = (result * base) % mod;
    base = (base * base) % mod;
    exponent >>= 1;
  }
  return result * seed % mod;
}
void Require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto function = [lib newFunctionWithName:@"xgb_uniform_sample"];
      auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      Require(pipeline != nil, "No pipeline");
      auto queue = [device newCommandQueue];
      size_t cases = 0, draws = 0;
      std::vector<uint64_t> seeds{0,
                                  1,
                                  17,
                                  0xffffffff,
                                  4409460005719528983ull,
                                  202226012707858921ull,
                                  4611686018427387904ull,
                                  9021146024146916887ull,
                                  4813912031135246825ull};
      std::vector<float> probabilities{
          0, std::numeric_limits<float>::denorm_min(), 0.1f, 0.5f, 0.9f, std::nextafter(1.0f, 0.0f),
          1};
      for (uint64_t rows : {1, 31, 32, 257, 1027})
        for (uint32_t targets : {1, 3})
          for (bool column_major : {false, true})
            for (uint64_t seed : seeds)
              for (float probability : probabilities)
                for (uint32_t threads : {1, 8, 2048}) {
                  Params p{rows,
                           column_major ? 1 : targets + 1,
                           column_major ? rows + 2 : 1,
                           seed,
                           targets,
                           threads,
                           probability,
                           0};
                  std::vector<Pair> expected((rows - 1) * p.stride0 + (targets - 1) * p.stride1 + 5,
                                             {99, -73});
                  for (uint64_t i = 0; i < rows; ++i)
                    for (uint32_t j = 0; j < targets; ++j)
                      expected[2 + i * p.stride0 + j * p.stride1] = {float(i % 31) - 15.25f,
                                                                     float(j + 1)};
                  auto data = [device newBufferWithBytes:expected.data()
                                                  length:expected.size() * sizeof(Pair)
                                                 options:MTLResourceStorageModeShared];
                  auto command = [queue commandBuffer];
                  auto e = [command computeCommandEncoder];
                  [e setComputePipelineState:pipeline];
                  [e setBuffer:data offset:2 * sizeof(Pair) atIndex:0];
                  [e setBytes:&p length:sizeof(p) atIndex:1];
                  [e dispatchThreads:MTLSizeMake(rows, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                  [e endEncoding];
                  [command commit];
                  [command waitUntilCompleted];
                  Require(command.status == MTLCommandBufferStatusCompleted, "GPU sampling failed");
                  uint64_t chunk = rows / threads;
                  for (uint32_t t = 0; t < threads; ++t) {
                    uint64_t begin = t * chunk, end = t == threads - 1 ? rows : begin + chunk;
                    Engine engine(Skip(begin, seed));
                    std::bernoulli_distribution coin(probability);
                    for (uint64_t i = begin; i < end; ++i)
                      if (!coin(engine))
                        for (uint32_t j = 0; j < targets; ++j)
                          expected[2 + i * p.stride0 + j * p.stride1] = {0, 0};
                  }
                  if (memcmp(data.contents, expected.data(), expected.size() * sizeof(Pair))) {
                    fprintf(stderr,
                            "rows=%llu targets=%u column_major=%d seed=%llu probability=%g "
                            "threads=%u\n",
                            rows, targets, column_major, seed, probability, threads);
                    throw std::runtime_error(
                        "GPU mask or gradient writes differ from CPU Bernoulli reference");
                  }
                  ++cases;
                  draws += rows;
                  if (cases % 1000 == 0) {
                    printf("Checked %zu sampling cases\n", cases);
                    fflush(stdout);
                  }
                }
      printf("PASS %zu exact sampling cases and %zu Bernoulli draws, including zero seeds, "
             "rounding boundaries, strides and guard regions on %s\n",
             cases, draws, device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
