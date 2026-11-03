// Paired kernel timings: screening enabled/disabled, both checked against CPU.
#define main xgb_existing_split_test_main
#include "../tests/metal_split.mm"
#undef main
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      id<MTLComputePipelineState> pipelines[2];
      for (int enabled = 0; enabled < 2; ++enabled) {
        auto options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        options.preprocessorMacros = @{@"XGB_METAL_SPLIT_FILTER" : @(enabled)};
        NSError* error = nil;
        auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                        options:options
                                          error:&error];
        if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
        pipelines[enabled] = [device
            newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xgb_evaluate_splits"]
                                          error:&error];
        Require(pipelines[enabled] != nil, "Split pipeline failed");
      }
      auto queue = [device newCommandQueue];
      std::mt19937 rng(911);
      std::printf("{\"device\":\"%s\",\"bins_per_feature\":256,\"repeats\":5,\"cases\":[\n",
                  device.name.UTF8String);
      bool first = true;
      for (uint32_t mode = 0; mode < 6; ++mode) {
        uint32_t count = mode % 3 == 0 ? 128 : mode % 3 == 1 ? 1024 : 8192;
        bool missing = mode >= 3;
        Params p{1, 0, 0, 1};
        std::vector<Pair> hist;
        std::vector<Task> tasks;
        std::vector<Result> expected;
        for (uint32_t t = 0; t < count; ++t) {
          auto begin = uint32_t(hist.size());
          Pair total{0, 0};
          for (uint32_t b = 0; b < 256; ++b) {
            Pair v{double(int(rng() % 20001) - 10000) / 32, double(1 + rng() % 2000) / 8};
            // Include flat/near-tied objectives, where retaining many candidates is required.
            if (t % 19 == 0) v.g = v.h * 3.141592653589793;
            hist.push_back(v);
            total = Add(total, v);
          }
          if (missing) total = Add(total, {45, 50});
          Task task{begin,
                    uint32_t(hist.size()),
                    0,
                    256,
                    total.g,
                    total.h,
                    Gain(total, Weight(total, p), p),
                    -INFINITY,
                    INFINITY,
                    0};
          tasks.push_back(task);
          std::vector<uint32_t> order;
          expected.push_back(Reference(hist, task, p, order));
        }
        auto upload = [&](void const* ptr, size_t bytes) {
          return [device newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
        };
        auto alloc = [&](size_t bytes) {
          return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };
        auto input = upload(hist.data(), hist.size() * sizeof(Pair));
        auto metadata = upload(tasks.data(), tasks.size() * sizeof(Task));
        auto output = alloc(count * sizeof(Result)), order = alloc(hist.size() * 4),
             weights = alloc(hist.size() * 4), scans = alloc(hist.size() * 32);
        std::vector<double> elapsed[2];
        for (int repeat = -1; repeat < 5; ++repeat) {
          std::vector<int> variants{0, 1};
          std::shuffle(variants.begin(), variants.end(), rng);
          for (auto enabled : variants) {
            auto command = [queue commandBuffer];
            auto e = [command computeCommandEncoder];
            [e setComputePipelineState:pipelines[enabled]];
            [e setBuffer:input offset:0 atIndex:0];
            [e setBuffer:metadata offset:0 atIndex:1];
            [e setBytes:&p length:sizeof(p) atIndex:2];
            [e setBuffer:output offset:0 atIndex:3];
            [e setBuffer:order offset:0 atIndex:4];
            [e setBuffer:weights offset:0 atIndex:5];
            [e setBuffer:scans offset:0 atIndex:6];
            [e dispatchThreadgroups:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e endEncoding];
            [command commit];
            [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted, "Split command failed");
            Require(std::memcmp(output.contents, expected.data(),
                                expected.size() * sizeof(Result)) == 0,
                    "Screened/unscreened split differs from CPU");
            if (repeat >= 0) elapsed[enabled].push_back(command.GPUEndTime - command.GPUStartTime);
          }
        }
        if (!first) std::printf(",\n");
        first = false;
        std::printf("{\"feature_tasks\":%u,\"missing\":%s,\"gpu_seconds\":{", count,
                    missing ? "true" : "false");
        for (int enabled = 0; enabled < 2; ++enabled) {
          std::printf("%s\"%s\":[", enabled ? "," : "", enabled ? "screened" : "original");
          for (size_t i = 0; i < elapsed[enabled].size(); ++i)
            std::printf("%s%.9f", i ? "," : "", elapsed[enabled][i]);
          std::printf("]");
        }
        std::printf("}}");
        std::fflush(stdout);
        std::fprintf(stderr, "Validated CPU and both GPU variants for %u features\n", count);
      }
      std::printf("\n]}\n");
      return 0;
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
