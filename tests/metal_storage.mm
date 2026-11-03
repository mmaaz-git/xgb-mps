// GPU histogram storage operations compared with independent CPU arithmetic.
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
  double g, h;
};
struct Raw {
  int64_t g, h;
};
struct Dequantize {
  uint64_t count;
  int32_t gu, hu;
  uint32_t bins, add;
};
struct Subtract {
  uint64_t parent, sibling, output, bins;
};
struct Gather {
  uint64_t source;
  uint32_t output, bins, stride, padding;
};
static_assert(sizeof(Dequantize) == 24 && sizeof(Subtract) == 32 && sizeof(Gather) == 24);
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
      auto pipeline = [&](NSString* name) {
        auto f = [lib newFunctionWithName:name];
        auto p = [device newComputePipelineStateWithFunction:f error:&error];
        Require(p != nil, "Missing pipeline");
        return p;
      };
      auto dequantize = pipeline(@"xgb_histogram_dequantize"),
           subtract = pipeline(@"xgb_histogram_subtract"),
           gather = pipeline(@"xgb_histogram_gather");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
      };
      auto finish = [&](id<MTLCommandBuffer> command) {
        [command commit];
        [command waitUntilCompleted];
        Require(command.status == MTLCommandBufferStatusCompleted, "GPU command failed");
      };
      std::mt19937_64 rng(955);
      size_t converted = 0, subtracted = 0, gathered = 0, cases = 0;
      for (uint32_t bins : {1, 33, 257, 1025})
        for (int unit : {-200, -100, -32, 0, 32, 100, 200}) {
          std::vector<Raw> raw(bins * 3);
          for (auto& v : raw) {
            v.g = int64_t(rng());
            v.h = int64_t(rng());
          }
          raw[0] = {0, std::numeric_limits<int64_t>::min()};
          raw[1] = {std::numeric_limits<int64_t>::max(), (int64_t(1) << 53) + 1};
          raw[2] = {-((int64_t(1) << 53) + 3), 1};
          // Non-contiguous and non-monotonic node offsets; padding must remain intact.
          uint64_t offsets[3]{2 * bins + 11, 3, bins + 7};
          std::vector<Pair> expected(3 * bins + 17, {17.5, -23.25});
          auto input = upload(raw.data(), raw.size() * sizeof(Raw));
          auto output = upload(expected.data(), expected.size() * sizeof(Pair));
          auto offset = upload(offsets, sizeof(offsets));
          for (uint32_t add : {0u, 1u}) {
            Dequantize p{raw.size(), unit, -unit, bins, add};
            auto command = [queue commandBuffer];
            auto e = [command computeCommandEncoder];
            [e setComputePipelineState:dequantize];
            [e setBuffer:input offset:0 atIndex:0];
            [e setBuffer:output offset:0 atIndex:1];
            [e setBuffer:offset offset:0 atIndex:2];
            [e setBytes:&p length:sizeof(p) atIndex:3];
            [e dispatchThreads:MTLSizeMake(raw.size(), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            finish(command);
            for (size_t i = 0; i < raw.size(); ++i) {
              auto out = offsets[i / bins] + i % bins;
              Pair v{double(raw[i].g) * std::ldexp(1.0, unit),
                     double(raw[i].h) * std::ldexp(1.0, -unit)};
              if (add) {
                expected[out].g += v.g;
                expected[out].h += v.h;
              } else
                expected[out] = v;
            }
            auto actual = static_cast<Pair const*>(output.contents);
            for (size_t i = 0; i < expected.size(); ++i)
              Require(actual[i].g == expected[i].g && actual[i].h == expected[i].h,
                      "Scaled integer conversion/page accumulation mismatch or out-of-range write");
            converted += raw.size();
            ++cases;
          }
          // Preserve sources and padding while writing differences into disjoint spans.
          std::vector<Pair> storage(expected);
          storage.resize(expected.size() + bins * 2 + 9, {17.5, -23.25});
          Subtract jobs[2]{{offsets[0], offsets[1], expected.size() + 1, bins},
                           {offsets[1], offsets[2], expected.size() + bins + 5, bins}};
          auto all = upload(storage.data(), storage.size() * sizeof(Pair)),
               tasks = upload(jobs, sizeof(jobs));
          auto command = [queue commandBuffer];
          auto e = [command computeCommandEncoder];
          [e setComputePipelineState:subtract];
          [e setBuffer:all offset:0 atIndex:0];
          [e setBuffer:tasks offset:0 atIndex:1];
          [e dispatchThreads:MTLSizeMake(bins, 2, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [e endEncoding];
          finish(command);
          for (auto t : jobs)
            for (size_t i = 0; i < bins; ++i)
              storage[t.output + i] = {storage[t.parent + i].g - storage[t.sibling + i].g,
                                       storage[t.parent + i].h - storage[t.sibling + i].h};
          auto actual = static_cast<Pair const*>(all.contents);
          for (size_t i = 0; i < storage.size(); ++i)
            Require(actual[i].g == storage[i].g && actual[i].h == storage[i].h,
                    "Histogram subtraction mismatch or out-of-range write");
          subtracted += 2 * bins;
          ++cases;
          for (uint32_t stride : {1, 3, 17}) {
            std::vector<Gather> jobs;
            std::vector<Pair> packed(3 * bins * stride + 19, {17.5, -23.25});
            for (uint32_t j = 0; j < 3; ++j) {
              uint32_t out = j * bins * stride + 7;
              jobs.push_back({offsets[j], out, bins, stride, 0});
              for (uint32_t i = 0; i < bins; ++i)
                packed[out + i * stride] = expected[offsets[j] + i];
            }
            std::vector<Pair> cleared(packed.size(), {17.5, -23.25});
            auto target = upload(cleared.data(), cleared.size() * sizeof(Pair));
            auto metadata = upload(jobs.data(), jobs.size() * sizeof(Gather));
            command = [queue commandBuffer];
            e = [command computeCommandEncoder];
            [e setComputePipelineState:gather];
            [e setBuffer:output offset:0 atIndex:0];
            [e setBuffer:target offset:0 atIndex:1];
            [e setBuffer:metadata offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(jobs.size(), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            finish(command);
            actual = static_cast<Pair const*>(target.contents);
            for (size_t i = 0; i < packed.size(); ++i)
              Require(actual[i].g == packed[i].g && actual[i].h == packed[i].h,
                      "Histogram gather mismatch or out-of-range write");
            gathered += 3 * bins;
            ++cases;
          }
        }
      printf("PASS %zu histogram storage cases: %zu converted/accumulated pairs, %zu subtracted "
             "pairs, %zu gathered pairs on %s\n",
             cases, converted, subtracted, gathered, device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
