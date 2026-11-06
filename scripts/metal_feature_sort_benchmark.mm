// Compare full-width and bounded feature-key radix sorts using production kernels.
// clang++ -O2 -std=c++17 -fobjc-arc scripts/metal_feature_sort_benchmark.mm \
//   -framework Metal -framework Foundation -o results/feature-sort-benchmark
// results/feature-sort-benchmark xgboost/src/common/metal_histogram.metal > results/feature-sort.json
// This measures only feature-key sorting, excluding the preceding value sort
// and the rest of sketch construction. Every output matches CPU stable_sort exactly.
#define main xgb_quantile_reference_test_main
#include "../tests/metal_quantile.mm"
#undef main
#include <array>

id<MTLBuffer> SortFeatures(GPU& gpu, id<MTLCommandBuffer> command,
                          id<MTLBuffer> source, uint32_t rows, uint32_t bits) {
  uint32_t blocks = (rows + 255) / 256;
  auto target = gpu.Allocate(source.length), ranks = gpu.Allocate(size_t(rows) * 4);
  auto counts = gpu.Allocate(size_t(blocks) * 16 * 4);
  for (uint32_t shift = 0; shift < bits; shift += 4) {
    Radix p{rows, 1, blocks, shift};
    auto e = [command computeCommandEncoder];
    [e setComputePipelineState:gpu.rank];
    [e setBuffer:source offset:0 atIndex:0];
    [e setBuffer:ranks offset:0 atIndex:1];
    [e setBuffer:counts offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    auto prefix = gpu.Prefix(command, counts, 16 * blocks);
    e = [command computeCommandEncoder];
    [e setComputePipelineState:gpu.scatter];
    [e setBuffer:source offset:0 atIndex:0];
    [e setBuffer:ranks offset:0 atIndex:1];
    [e setBuffer:prefix offset:0 atIndex:2];
    [e setBuffer:target offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    std::swap(source, target);
  }
  return source;
}
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production shader source");
      GPU gpu(argv[1]);
      std::mt19937 rng(1703);
      constexpr uint32_t rows = 1 << 22;
      std::printf("{\"device\":\"%s\",\"records\":%u,\"repeats\":5,\"cases\":[\n",
                  gpu.device.name.UTF8String, rows);
      bool comma = false;
      for (uint32_t columns : {28u, 128u, 257u, 4097u}) {
        std::vector<Record> input(rows);
        for (uint32_t i = 0; i < rows; ++i) input[i] = {uint32_t(rng() % columns), i};
        auto expected = input;
        std::stable_sort(expected.begin(), expected.end(), [](auto a, auto b) {
          return a.key < b.key;
        });
        uint32_t bits = 0;
        for (auto key = columns - 1; key; key >>= 1) ++bits;
        std::vector<double> times[2];
        for (int trial = -1; trial < 5; ++trial) {
          std::array<int, 2> order{0, 1};
          std::shuffle(order.begin(), order.end(), rng);
          for (int variant : order) {
            @autoreleasepool {
              auto source = gpu.Upload(input.data(), input.size() * sizeof(Record));
              auto command = [gpu.queue commandBuffer];
              auto output = SortFeatures(gpu, command, source, rows, variant ? bits : 32);
              gpu.Finish(command);
              Check(std::memcmp(output.contents, expected.data(), expected.size() * sizeof(Record)) == 0,
                    "Feature ordering differs from CPU stable_sort");
              if (trial >= 0) times[variant].push_back(command.GPUEndTime - command.GPUStartTime);
            }
          }
        }
        std::printf("%s{\"features\":%u,\"key_bits\":%u,\"gpu_seconds\":{", comma ? ",\n" : "",
                    columns, bits);
        comma = true;
        for (int v = 0; v < 2; ++v) {
          std::printf("%s\"%s\":[", v ? "," : "", v ? "bounded" : "full32");
          for (size_t i = 0; i < times[v].size(); ++i)
            std::printf("%s%.9f", i ? "," : "", times[v][i]);
          std::printf("]");
        }
        std::printf("}}");
        std::fflush(stdout);
        std::fprintf(stderr, "Validated %u feature IDs / all repeats against CPU stable_sort\n", columns);
      }
      std::printf("\n]}\n");
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
  return 0;
}
