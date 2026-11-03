// Check the conservative pruning intervals against the original CPU gain formula.
#define main xgb_existing_split_test_main
#include "metal_split.mm"
#undef main
struct BoundInput {
  double g, h;
  float lambda;
  uint32_t padding{0};
};
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      source += R"(
struct XGBBoundInput { ulong g, h; float lambda; uint padding; };
kernel void xgb_test_split_bound(device const XGBBoundInput* input [[buffer(0)]],
                                 device float2* output [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]]) {
  auto v = input[tid];
  output[tid] = xgb_unconstrained_gain_bound(DoublePair{{v.g}, {v.h}}, v.lambda);
}
)";
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto kernel = [device
          newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xgb_test_split_bound"]
                                        error:&error];
      Require(kernel != nil, "Missing bound kernel");
      std::mt19937 rng(653);
      std::vector<BoundInput> data;
      for (size_t i = 0; i < 1000000; ++i) {
        double g = std::ldexp(1.0 + double(rng()) / 4294967296.0, int(rng() % 140) - 70);
        double h = std::ldexp(1.0 + double(rng()) / 4294967296.0, int(rng() % 140) - 70);
        if (i % 2) {
          g = std::ldexp(g, -std::ilogb(g) + int(rng() % 50) - 25);
          h = std::ldexp(h, -std::ilogb(h) + int(rng() % 50) - 25);
        }
        if (i % 3) g = -g;
        if (i % 47 == 0) g = i % 2 ? -0.0 : 0.0;
        if (i % 53 == 0) h = 0;
        if (i % 67 == 0) g = std::numeric_limits<double>::denorm_min();
        float lambda = i % 5 ? float(std::ldexp(1.0, int(rng() % 80) - 40)) : 0.0f;
        data.push_back({g, h, lambda});
      }
      for (double g : {1e-8, 1e8, 0.0, 1.0, -1.0})
        for (double h : {1e-8, 1e8, 0.0, 1.0})
          for (float lambda : {0.0f, 1e-8f, 1.0f, 1e8f, 1e9f})
            for (int direction : {-1, 0, 1}) {
              double gg = direction ? std::nextafter(g, direction * INFINITY) : g;
              double hh = direction ? std::nextafter(h, direction * INFINITY) : h;
              data.push_back({gg, hh, lambda});
            }
      auto input = [device newBufferWithBytes:data.data()
                                       length:data.size() * sizeof(BoundInput)
                                      options:MTLResourceStorageModeShared];
      auto output = [device newBufferWithLength:data.size() * 2 * sizeof(float)
                                        options:MTLResourceStorageModeShared];
      auto command = [[device newCommandQueue] commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:kernel];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:output offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(data.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      Require(command.status == MTLCommandBufferStatusCompleted, "Bound command failed");
      auto bounds = static_cast<float const*>(output.contents);
      size_t finite = 0, unbounded = 0;
      for (size_t i = 0; i < data.size(); ++i) {
        auto v = data[i];
        Params p{v.lambda, 0, 0, 0};
        Pair pair{v.g, v.h};
        float actual = Gain(pair, Weight(pair, p), p), lo = bounds[2 * i], hi = bounds[2 * i + 1];
        if (lo == -INFINITY && hi == INFINITY) {
          ++unbounded;
          continue;
        }
        ++finite;
        if (!(lo <= actual && actual <= hi)) {
          std::fprintf(stderr, "bound case %zu g=%a h=%a lambda=%a actual=%a bounds=[%a,%a]\n", i,
                       v.g, v.h, v.lambda, actual, lo, hi);
          throw std::runtime_error("Conservative gain interval excluded CPU gain");
        }
      }
      Require(finite > 100000 && unbounded > 100000, "Did not exercise both bound paths");
      std::printf(
          "PASS %zu CPU gain containment checks: %zu finite intervals, %zu unbounded fallbacks\n",
          data.size(), finite, unbounded);
      return 0;
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
