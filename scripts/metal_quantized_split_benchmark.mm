// Timings and exact CUDA-formula checks for original, reuse-only and screened splits.
#define main xgb_existing_cuda_split_test_main
#include "../tests/metal_cuda_split.mm"
#undef main
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 3, "Pass production and baseline Metal sources");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No Metal device");
      id<MTLComputePipelineState> pipelines[3];
      char const* names[] = {"baseline", "reuse_only", "screened"};
      for (int variant = 0; variant < 3; ++variant) {
        std::ifstream f(argv[variant ? 1 : 2]);
        Check(f.good(), "Cannot open Metal source");
        std::string source{std::istreambuf_iterator<char>(f), {}};
        auto options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        options.preprocessorMacros = @{@"XGB_METAL_SPLIT_FILTER" : @(variant == 2)};
        NSError* error = nil;
        auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                            options:options error:&error];
        if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
        pipelines[variant] = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"xgb_evaluate_quantized_splits"] error:&error];
        Check(pipelines[variant] != nil, "Missing integer split pipeline");
      }
      auto queue = [device newCommandQueue];
      std::mt19937_64 rng(1947);
      std::printf("{\"device\":\"%s\",\"bins\":256,\"repeats\":5,\"cases\":[\n",
                  device.name.UTF8String);
      bool first = true;
      for (bool missing : {false, true}) {
        for (std::uint32_t count : {128u, 1024u, 8192u}) {
          ref::Scale scale{32, 32};
          ref::Parameters params{1, 0, 0, 1};
          std::vector<ref::Pair> hist;
          std::vector<ref::Task> tasks;
          std::vector<ref::Result> expected;
          for (std::uint32_t t = 0; t < count; ++t) {
            auto begin = std::uint32_t(hist.size());
            ref::Pair total{};
            for (std::uint32_t i = 0; i < 256; ++i) {
              ref::Pair value{(std::int64_t(rng() % 20001) - 10000) * (1LL << 27),
                              std::int64_t(1 + rng() % 2000) * (1LL << 29)};
              if (t % 19 == 0) value.g = 2 * value.h;  // Near-tied objectives.
              hist.push_back(value);
              total = ref::Add(total, value);
            }
            if (missing) total = ref::Add(total, {45LL << 32, 50LL << 32});
            tasks.push_back({begin, std::uint32_t(hist.size()), 0, 256, total,
                             -INFINITY, INFINITY, 0});
            std::vector<std::uint32_t> order;
            expected.push_back(ref::Evaluate(hist, tasks.back(), scale, params, &order));
          }
          auto upload = [&](void const* data, std::size_t bytes) {
            auto b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            Check(b != nil, "Buffer allocation failed");
            if (data) std::memcpy(b.contents, data, bytes);
            return b;
          };
          auto h = upload(hist.data(), hist.size() * sizeof(ref::Pair));
          auto t = upload(tasks.data(), tasks.size() * sizeof(ref::Task));
          auto o = upload(nullptr, tasks.size() * sizeof(ref::Result));
          auto order = upload(nullptr, hist.size() * sizeof(std::uint32_t));
          auto weights = upload(nullptr, hist.size() * sizeof(float));
          auto scans = upload(nullptr, hist.size() * sizeof(ref::Pair));
          std::vector<double> seconds[3];
          for (int repeat = -1; repeat < 5; ++repeat) {
            std::vector<int> variants{0, 1, 2};
            std::shuffle(variants.begin(), variants.end(), rng);
            for (auto variant : variants) {
              auto command = [queue commandBuffer];
              auto e = [command computeCommandEncoder];
              [e setComputePipelineState:pipelines[variant]];
              [e setBuffer:h offset:0 atIndex:0];
              [e setBuffer:t offset:0 atIndex:1];
              [e setBytes:&params length:sizeof(params) atIndex:2];
              [e setBuffer:o offset:0 atIndex:3];
              [e setBuffer:order offset:0 atIndex:4];
              [e setBuffer:weights offset:0 atIndex:5];
              [e setBuffer:scans offset:0 atIndex:6];
              [e setBytes:&scale length:sizeof(scale) atIndex:7];
              [e dispatchThreadgroups:MTLSizeMake(count, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
              [e endEncoding];
              [command commit];
              [command waitUntilCompleted];
              Check(command.status == MTLCommandBufferStatusCompleted, "Split command failed");
              Check(std::memcmp(o.contents, expected.data(), expected.size() * sizeof(ref::Result)) == 0,
                    "GPU result differs from full-precision CUDA-formula oracle");
              if (repeat >= 0) seconds[variant].push_back(command.GPUEndTime - command.GPUStartTime);
            }
          }
          std::printf("%s{\"feature_tasks\":%u,\"missing\":%s,\"gpu_seconds\":{",
                      first ? "" : ",\n", count, missing ? "true" : "false");
          first = false;
          for (int variant = 0; variant < 3; ++variant) {
            std::printf("%s\"%s\":[", variant ? "," : "", names[variant]);
            for (std::size_t i = 0; i < seconds[variant].size(); ++i)
              std::printf("%s%.9f", i ? "," : "", seconds[variant][i]);
            std::printf("]");
          }
          std::printf("}}");
          std::fflush(stdout);
          std::fprintf(stderr, "Checked %u feature tasks, missing=%d, all three variants\n", count, missing);
        }
      }
      std::printf("\n]}\n");
      return 0;
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
