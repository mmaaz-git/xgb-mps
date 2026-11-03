// Verify compressed bin indices and hit counts against CPU standard algorithms.
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
struct Entry {
  uint32_t feature;
  float value;
};
struct Params {
  uint64_t count;
  uint32_t features, width, dense, padding;
};
static_assert(sizeof(Entry) == 8 && sizeof(Params) == 24);
void Require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal device");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto kernel = [library newFunctionWithName:@"xgb_bin_entries"];
      auto pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
      Require(pipeline != nil, "Missing binning kernel");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
      };
      std::mt19937 rng(39);
      size_t cases = 0, entries_checked = 0, counts_checked = 0;
      for (uint32_t width : {1u, 2u, 4u})
        for (bool dense : {false, true}) {
          if (!dense && width != 4) continue;
          uint32_t per_feature = width == 1 ? 17 : (width == 2 ? 257 : 65537);
          for (bool mixed : {false, true}) {
            std::vector<uint32_t> ptrs{0};
            std::vector<float> cuts;
            std::vector<uint8_t> categorical(7, 0);
            float tiny = std::numeric_limits<float>::denorm_min();
            std::vector<float> special{-std::numeric_limits<float>::max(),
                                       -1.0f,
                                       -tiny * 3,
                                       -tiny,
                                       -0.0f,
                                       0.0f,
                                       tiny,
                                       tiny * 3,
                                       1.0f,
                                       std::numeric_limits<float>::max()};
            for (uint32_t f = 0; f < 7; ++f) {
              categorical[f] = mixed && f % 2;
              for (uint32_t i = 0; i < per_feature; ++i)
                cuts.push_back(categorical[f] ? float(i * 2)
                                              : (i < special.size()
                                                     ? special[i]
                                                     : float(int(rng() % 200000) - 100000) / 7));
              std::sort(cuts.begin() + ptrs.back(), cuts.end());
              ptrs.push_back(cuts.size());
            }
            for (uint64_t size : {1, 31, 32, 255, 256, 257, 8193})
              for (int mode = 0; mode < 3; ++mode) {
                std::vector<Entry> entries(size);
                std::vector<uint8_t> expected(32 + size * width, 0xa5);
                std::vector<uint32_t> counts(cuts.size() + 8, 7);
                uint32_t expected_status = 0;
                for (size_t i = 0; i < entries.size(); ++i) {
                  uint32_t f = rng() % 7;
                  float value = categorical[f]
                                    ? float(int(rng() % (per_feature * 2 + 4)) - 2) + 0.75f
                                    : special[i % special.size()];
                  if (mode == 1) value = cuts[ptrs[f] + rng() % per_feature];
                  if (mode == 2) {
                    if (i % 3 == 0)
                      value = std::numeric_limits<float>::infinity();
                    else if (i % 3 == 1)
                      value = -std::numeric_limits<float>::infinity();
                    else
                      f = 7;
                  }
                  entries[i] = {f, value};
                  if (std::isinf(value)) {
                    expected_status |= 1;
                    continue;
                  }
                  if (f >= 7) {
                    expected_status |= 2;
                    continue;
                  }
                  auto begin = cuts.begin() + ptrs[f], end = cuts.begin() + ptrs[f + 1];
                  auto it = categorical[f] ? std::lower_bound(begin, end, float(int32_t(value)))
                                           : std::upper_bound(begin, end, value);
                  if (it == end) --it;
                  uint32_t global = it - cuts.begin(), bin = dense ? global - ptrs[f] : global;
                  std::memcpy(expected.data() + 16 + i * width, &bin, width);
                  ++counts[4 + global];
                }
                auto input = upload(entries.data(), entries.size() * sizeof(Entry));
                auto offsets = upload(ptrs.data(), ptrs.size() * sizeof(uint32_t));
                auto values = upload(cuts.data(), cuts.size() * sizeof(float));
                auto types = upload(categorical.data(), categorical.size());
                std::vector<uint8_t> initial(expected.size(), 0xa5);
                std::vector<uint32_t> initial_counts(counts.size(), 7);
                uint32_t zero = 0;
                auto output = upload(initial.data(), initial.size());
                auto hits = upload(initial_counts.data(), initial_counts.size() * sizeof(uint32_t));
                auto status = upload(&zero, sizeof(zero));
                Params p{size, 7, width, uint32_t(dense), 0};
                auto command = [queue commandBuffer];
                auto e = [command computeCommandEncoder];
                [e setComputePipelineState:pipeline];
                [e setBuffer:input offset:0 atIndex:0];
                [e setBuffer:offsets offset:0 atIndex:1];
                [e setBuffer:values offset:0 atIndex:2];
                [e setBuffer:types offset:0 atIndex:3];
                [e setBuffer:output offset:16 atIndex:4];
                [e setBuffer:hits offset:16 atIndex:5];
                [e setBuffer:status offset:0 atIndex:6];
                [e setBytes:&p length:sizeof(p) atIndex:7];
                [e dispatchThreads:MTLSizeMake(size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
                [command commit];
                [command waitUntilCompleted];
                Require(command.status == MTLCommandBufferStatusCompleted,
                        "Binning command failed");
                Require(std::memcmp(output.contents, expected.data(), expected.size()) == 0,
                        "Bin indices or guard regions differ from CPU search");
                Require(std::memcmp(hits.contents, counts.data(),
                                    counts.size() * sizeof(uint32_t)) == 0,
                        "Hit counts or count guard regions differ from CPU");
                Require(*static_cast<uint32_t const*>(status.contents) == expected_status,
                        "Invalid entries were not detected correctly");
                ++cases;
                entries_checked += size;
                counts_checked += cuts.size();
              }
            printf("Checked binning width=%u dense=%d categorical=%d\n", width, dense, mixed);
          }
        }
      printf("PASS %zu exact binning cases / %zu entries / %zu hit counts on %s\n", cases,
             entries_checked, counts_checked, device.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
