#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_philox_reference.h"
#include "rng_contract_fixtures.h"
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
using Query = rng_contracts::PhiloxQuery;
struct Result {
  std::uint64_t word;
  std::uint32_t weight, padding;
};
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass the production shader");
      Check(cuda_reference::Philox(20111115, 9999) == 3409172418970261260ULL,
            "Host oracle failed CCCL's published 10000th-value check");
      std::ifstream stream(argv[1]);
      std::string source{std::istreambuf_iterator<char>{stream}, {}};
      source += R"(
struct Query { ulong row; uint seed, padding; };
struct Result { ulong word; uint weight, padding; };
kernel void probe_cuda_rng(device const Query* in [[buffer(0)]],
                           device Result* out [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = {xgb_cuda_philox(in[tid].seed, in[tid].row),
              as_type<uint>(xgb_cuda_random_weight(in[tid].seed, in[tid].row)), 0};
})";
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No GPU");
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [device
          newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe_cuda_rng"]
                                        error:&error];
      Check(pipeline != nil, "No pipeline");
      auto queries = rng_contracts::PhiloxQueries();
      auto input = [device newBufferWithBytes:queries.data()
                                       length:queries.size() * sizeof(Query)
                                      options:MTLResourceStorageModeShared];
      auto output = [device newBufferWithLength:queries.size() * sizeof(Result)
                                        options:MTLResourceStorageModeShared];
      auto queue = [device newCommandQueue];
      auto command = [queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:pipeline];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:output offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(queries.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      Check(command.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
      auto actual = static_cast<Result const*>(output.contents);
      for (std::size_t i = 0; i < queries.size(); ++i) {
        auto query = queries[i];
        auto weight = cuda_reference::Weight(query.seed, query.row);
        Check(actual[i].word == cuda_reference::Philox(query.seed, query.row),
              "Philox word mismatch");
        Check(!std::memcmp(&actual[i].weight, &weight, sizeof(float)),
              "Uniform float bits mismatch");
      }
      std::printf(
          "PASS: %zu exact Philox words/uniform draws, published check, 64-bit skip boundaries\n",
          queries.size());
      return 0;
    } catch (std::exception const& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
