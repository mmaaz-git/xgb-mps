// Standalone differential test of the actual production shader, including
// concurrent negative additions, low-word carry, sparse rows and page offsets.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Row { uint64_t row; uint32_t node, pad; };
struct Pair { int64_t g, h; };
struct Params { uint64_t rows, base; uint32_t bins, features, width, dense; };
static_assert(sizeof(Row) == 16 && sizeof(Pair) == 16 && sizeof(Params) == 32);
void Require(bool ok, char const* what) { if (!ok) throw std::runtime_error(what); }

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass the production .metal source path");
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      Require(dev != nil, "No visible Metal device");
      std::ifstream file(argv[1]);
      Require(file.good(), "Cannot read production shader");
      std::string source{std::istreambuf_iterator<char>(file), {}};
      NSError* error = nil;
      auto library = [dev newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                       options:nil error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto kernel = [library newFunctionWithName:@"xgb_histogram"];
      auto pipeline = [dev newComputePipelineStateWithFunction:kernel error:&error];
      Require(pipeline != nil, "Cannot create pipeline");
      auto queue = [dev newCommandQueue];
      auto upload = [&](void const* ptr, size_t size) -> id<MTLBuffer> {
        auto b = [dev newBufferWithLength:std::max<size_t>(size, 16) options:MTLResourceStorageModeShared];
        Require(b != nil, "Buffer allocation failed");
        if (size) memcpy(b.contents, ptr, size);
        return b;
      };
      int cases = 0;
      for (uint32_t width : {1u, 2u, 4u}) {
        for (bool dense : {true, false}) {
          // Sparse indices in XGBoost are always global uint32 indices.
          if (!dense && width != 4) continue;
          for (uint64_t base : {0ull, 11ull}) {
            for (bool contention : {false, true}) {
              constexpr uint64_t n = 4097;
              constexpr uint32_t features = 7, nodes = 3;
              uint32_t per_feature = width == 1 ? 17 : 513;
              uint32_t bins = per_feature * features;
              std::mt19937 rng(71);
              std::vector<uint8_t> indices;
              std::vector<uint64_t> rowptr{0};
              std::vector<uint32_t> offsets;
              for (uint32_t f = 0; f < features; ++f) offsets.push_back(f * per_feature);
              std::vector<Pair> gradients(n + base);
              std::vector<Pair> reference(nodes * bins, {0, 0});
              std::vector<Row> rows;
              for (uint64_t r = 0; r < n; ++r) {
                // Large signed integers exercise both high-word addition and carry.
                auto signed_value = static_cast<int64_t>(rng()) - (int64_t{1} << 31);
                Pair g{signed_value * (int64_t{1} << 17) + int64_t(r % 11),
                       (int64_t{1} << 35) + int64_t(rng())};
                gradients[r + base] = g;
                uint32_t node = r % nodes;
                rows.push_back({r + base, node, 0});
                for (uint32_t f = 0; f < features; ++f) {
                  if (!dense && (r % 13 == 0 || rng() % 3 == 0)) continue;
                  uint32_t local = contention ? 0 : rng() % per_feature;
                  uint32_t global = offsets[f] + local;
                  uint32_t stored = dense ? local : global;
                  for (uint32_t byte = 0; byte < width; ++byte)
                    indices.push_back(uint8_t(stored >> (8 * byte)));
                  reference[node * bins + global].g += g.g;
                  reference[node * bins + global].h += g.h;
                }
                rowptr.push_back(indices.size() / width);
              }
              auto idx = upload(indices.data(), indices.size());
              auto ptr = upload(rowptr.data(), rowptr.size() * sizeof(uint64_t));
              auto off = upload(offsets.data(), offsets.size() * sizeof(uint32_t));
              auto gp = upload(gradients.data(), gradients.size() * sizeof(Pair));
              auto rs = upload(rows.data(), rows.size() * sizeof(Row));
              std::vector<Pair> zero(reference.size(), {0, 0});
              auto out = upload(zero.data(), zero.size() * sizeof(Pair));
              Params p{n, base, bins, features, width, uint32_t(dense)};
              auto command = [queue commandBuffer];
              auto encoder = [command computeCommandEncoder];
              [encoder setComputePipelineState:pipeline];
              [encoder setBuffer:idx offset:0 atIndex:0];
              [encoder setBuffer:ptr offset:0 atIndex:1];
              [encoder setBuffer:off offset:0 atIndex:2];
              [encoder setBuffer:gp offset:0 atIndex:3];
              [encoder setBuffer:rs offset:0 atIndex:4];
              [encoder setBuffer:out offset:0 atIndex:5];
              [encoder setBytes:&p length:sizeof(p) atIndex:6];
              [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
              [encoder endEncoding];
              [command commit];
              [command waitUntilCompleted];
              Require(command.status == MTLCommandBufferStatusCompleted, "GPU command failed");
              auto actual = static_cast<Pair const*>(out.contents);
              for (size_t i = 0; i < reference.size(); ++i) {
                if (actual[i].g != reference[i].g || actual[i].h != reference[i].h) {
                  fprintf(stderr, "Mismatch: width=%u dense=%d base=%llu contention=%d bin=%zu\n",
                          width, dense, (unsigned long long)base, contention, i);
                  throw std::runtime_error("GPU histogram differs from independent integer reference");
                }
              }
              ++cases;
              printf("PASS case %d: width=%u dense=%d base=%llu contention=%d\n",
                     cases, width, dense, (unsigned long long)base, contention);
            }
          }
        }
      }
      printf("%d actual-GPU histogram cases passed on %s\n", cases, dev.name.UTF8String);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
