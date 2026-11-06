// Stage timings for dense compressed row partitioning, with exact CPU checks.
// clang++ -O2 -std=c++17 -fobjc-arc scripts/metal_partition_benchmark.mm \
//   -framework Metal -framework Foundation -o results/partition-benchmark
// results/partition-benchmark candidate.metal [baseline.metal [--swap]] > results/partition.json
// --swap compares a candidate using separate scatter output with the baseline copy.
// Stages use separate command buffers to expose their GPU time; this is a
// diagnostic, not an end-to-end training benchmark. Every timed output is checked.
#define main xgb_partition_reference_test_main
#include "../tests/metal_partition.mm"
#undef main
#include <array>
#include <cstring>

struct Compressed {
  uint64_t split;
  uint32_t stride, bits, symbols, dense, features, padding;
};
static_assert(sizeof(Compressed) == 32);

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      bool swap = argc == 4 && std::string(argv[3]) == "--swap";
      Require(argc == 2 || argc == 3 || swap, "Pass candidate shader [baseline shader [--swap]]");
      int variants = argc >= 3 ? 2 : 1;
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal device");
      auto queue = [device newCommandQueue];
      id<MTLComputePipelineState> pipelines[2][3];
      NSArray<NSString*>* names = @[@"xgb_partition_classify_ellpack",
                                    @"xgb_partition_prefix", @"xgb_partition_scatter"];
      for (int variant = 0; variant < variants; ++variant) {
        std::ifstream file(argv[variant + 1]);
        Require(file.good(), "Cannot read shader");
        std::string source{std::istreambuf_iterator<char>(file), {}};
        auto options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        NSError* error = nil;
        auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                            options:options error:&error];
        if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
        for (int stage = 0; stage < 3; ++stage) {
          pipelines[variant][stage] = [device newComputePipelineStateWithFunction:
                                      [library newFunctionWithName:names[stage]] error:&error];
          Require(pipelines[variant][stage] != nil, "Pipeline creation failed");
        }
      }
      auto upload = [&](void const* data, size_t bytes) {
        auto buffer = [device newBufferWithLength:std::max<size_t>(bytes, 16)
                                          options:MTLResourceStorageModeShared];
        Require(buffer != nil, "Allocation failed");
        if (data && bytes) std::memcpy(buffer.contents, data, bytes);
        return buffer;
      };
      std::mt19937 rng(1702);
      std::printf("{\"device\":\"%s\",\"repeats\":5,\"candidate_swaps_rows\":%s,\"cases\":[\n",
                  device.name.UTF8String, swap ? "true" : "false");
      bool comma = false;
      for (uint64_t n : {1000000ull, 10500000ull}) {
        @autoreleasepool {
          uint32_t features = n == 1000000 ? 128 : 28;
          std::vector<uint8_t> data(8 + n * features);
          for (size_t i = 8; i < data.size(); ++i) data[i] = rng();
          std::vector<uint32_t> offsets(features + 1);
          for (uint32_t f = 0; f <= features; ++f) offsets[f] = f * 256;
          auto index = upload(data.data(), data.size());
          auto off = upload(offsets.data(), offsets.size() * 4);
          auto dummy = upload(nullptr, 16);
          auto row = upload(nullptr, n * 8), saved = upload(nullptr, n * 8);
          auto rank = upload(nullptr, n * 4);
          Compressed compressed{data.size(), features, 8, 256, 1, features, 0};
          for (uint32_t nodes : {1u, 8u, 32u}) {
            @autoreleasepool {
              std::vector<uint64_t> rows(n);
              std::iota(rows.begin(), rows.end(), 0);
              // A stable partition preserves ascending order within each leaf,
              // but leaves select scattered source rows after earlier splits.
              if (nodes > 1) std::shuffle(rows.begin(), rows.end(), rng);
              std::vector<Split> splits;
              std::vector<Block> blocks;
              for (uint32_t node = 0; node < nodes; ++node) {
                uint64_t begin = n * node / nodes, end = n * (node + 1) / nodes;
                if (nodes > 1) std::sort(rows.begin() + begin, rows.begin() + end);
                uint32_t f = (node * 7 + 3) % features;
                splits.push_back({begin, end, f, int32_t(offsets[f] + 127), 0, 0, 0, 0,
                                  uint32_t(blocks.size()), 0});
                for (auto b = begin; b < end; b += 256)
                  blocks.push_back({b, std::min(b + 256, end), node, 0});
                splits.back().block_end = blocks.size();
              }
              auto expected = rows;
              std::vector<uint64_t> totals(nodes);
              for (uint32_t node = 0; node < nodes; ++node) {
                auto const& s = splits[node];
                auto middle = std::stable_partition(expected.begin() + s.begin,
                    expected.begin() + s.end, [&](uint64_t r) {
                      return data[8 + r * features + s.feature] <= 127;
                    });
                totals[node] = middle - (expected.begin() + s.begin);
              }
              auto split = upload(splits.data(), splits.size() * sizeof(Split));
              auto block = upload(blocks.data(), blocks.size() * sizeof(Block));
              auto counts = upload(nullptr, blocks.size() * 4);
              auto prefix = upload(nullptr, blocks.size() * 4);
              auto left = upload(nullptr, nodes * 8);
              Params p{n, 0, 1, 1, nodes, uint32_t(blocks.size())};
              std::vector<std::array<double, 3>> times[2];
              for (int repeat = -1; repeat < 5; ++repeat) {
                std::vector<int> order;
                for (int v = 0; v < variants; ++v) order.push_back(v);
                std::shuffle(order.begin(), order.end(), rng);
                for (int v : order) {
                  bool swapping = swap && v == 0;
                  std::memcpy(row.contents, rows.data(), n * 8);
                  std::array<double, 3> elapsed;
                  for (int stage = 0; stage < 3; ++stage) {
                    auto cmd = [queue commandBuffer];
                    auto e = [cmd computeCommandEncoder];
                    [e setComputePipelineState:pipelines[v][stage]];
                    if (stage == 0) {
                      id<MTLBuffer> buffers[] = {index, dummy, off, dummy, row, split, block,
                                                dummy, rank, counts, swapping ? row : saved};
                      for (int i = 0; i < 11; ++i) [e setBuffer:buffers[i] offset:0 atIndex:i];
                      [e setBytes:&p length:sizeof(p) atIndex:11];
                      [e setBuffer:dummy offset:0 atIndex:12];
                      [e setBytes:&compressed length:sizeof(compressed) atIndex:13];
                    } else if (stage == 1) {
                      id<MTLBuffer> buffers[] = {split, counts, prefix, left};
                      for (int i = 0; i < 4; ++i) [e setBuffer:buffers[i] offset:0 atIndex:i];
                      [e setBytes:&p length:sizeof(p) atIndex:4];
                    } else {
                      id<MTLBuffer> buffers[] = {swapping ? saved : row, swapping ? row : saved,
                                                split, block, rank, prefix, left};
                      for (int i = 0; i < 7; ++i) [e setBuffer:buffers[i] offset:0 atIndex:i];
                    }
                    [e dispatchThreadgroups:MTLSizeMake(stage == 1 ? nodes : blocks.size(), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [e endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    Require(cmd.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
                    elapsed[stage] = cmd.GPUEndTime - cmd.GPUStartTime;
                  }
                  Require(std::equal(expected.begin(), expected.end(),
                                     static_cast<uint64_t const*>((swapping ? saved : row).contents)),
                          "Row order differs");
                  Require(std::equal(totals.begin(), totals.end(),
                                     static_cast<uint64_t const*>(left.contents)), "Node counts differ");
                  if (repeat >= 0) times[v].push_back(elapsed);
                }
              }
              if (comma) std::printf(",\n");
              comma = true;
              std::printf("{\"rows\":%llu,\"features\":%u,\"nodes\":%u,\"variants\":[",
                          static_cast<unsigned long long>(n), features, nodes);
              for (int v = 0; v < variants; ++v) {
                std::printf("%s{\"name\":\"%s\",\"classify_prefix_scatter_seconds\":[",
                            v ? "," : "", v ? "baseline" : "candidate");
                for (size_t i = 0; i < times[v].size(); ++i)
                  std::printf("%s[%.9f,%.9f,%.9f]", i ? "," : "", times[v][i][0],
                              times[v][i][1], times[v][i][2]);
                std::printf("]}");
              }
              std::printf("]}");
              std::fflush(stdout);
              std::fprintf(stderr, "Validated %llu rows / %u nodes / all variants and repeats\n",
                           static_cast<unsigned long long>(n), nodes);
            }
          }
        }
      }
      std::printf("\n]}\n");
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
  return 0;
}
