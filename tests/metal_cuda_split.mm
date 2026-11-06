// Exact GPU differential tests against a separate host CUDA-formula oracle.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include "cuda_split_reference.h"
namespace ref = cuda_split_reference;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
std::size_t cases = 0, bins_checked = 0, cancellation_cases = 0;
int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No Metal device");
      std::ifstream f(argv[1]);
      std::string source{std::istreambuf_iterator<char>(f), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto pipeline = [device newComputePipelineStateWithFunction:
                                  [library newFunctionWithName:@"xgb_evaluate_quantized_splits"]
                                                            error:&error];
      Check(pipeline != nil, "Missing integer split pipeline");
      auto queue = [device newCommandQueue];
      auto upload = [&](void const* data, std::size_t bytes) {
        auto b = [device newBufferWithLength:std::max<std::size_t>(bytes, 16)
                                     options:MTLResourceStorageModeShared];
        if (bytes && data) std::memcpy(b.contents, data, bytes);
        return b;
      };
      std::mt19937_64 random(731972);
      auto run = [&](std::vector<ref::Pair> hist, std::vector<ref::Task> tasks, ref::Scale scale,
                     ref::Parameters params) {
        std::vector<ref::Result> expected;
        std::vector<std::vector<std::uint32_t>> orders(tasks.size());
        for (std::size_t i = 0; i < tasks.size(); ++i)
          expected.push_back(ref::Evaluate(hist, tasks[i], scale, params, &orders[i]));
        auto h = upload(hist.data(), hist.size() * sizeof(ref::Pair));
        auto t = upload(tasks.data(), tasks.size() * sizeof(ref::Task));
        auto o = upload(nullptr, (tasks.size() + 2) * sizeof(ref::Result));
        std::memset(o.contents, 0xa5, o.length);
        auto order = upload(nullptr, (hist.size() + 2) * sizeof(std::uint32_t));
        std::memset(order.contents, 0xa5, order.length);
        auto weights = upload(nullptr, hist.size() * sizeof(float));
        auto scans = upload(nullptr, hist.size() * sizeof(ref::Pair));
        auto command = [queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:pipeline];
        [e setBuffer:h offset:0 atIndex:0];
        [e setBuffer:t offset:0 atIndex:1];
        [e setBytes:&params length:sizeof(params) atIndex:2];
        [e setBuffer:o offset:sizeof(ref::Result) atIndex:3];
        [e setBuffer:order offset:sizeof(std::uint32_t) atIndex:4];
        [e setBuffer:weights offset:0 atIndex:5];
        [e setBuffer:scans offset:0 atIndex:6];
        [e setBytes:&scale length:sizeof(scale) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
          throw std::runtime_error(command.error.localizedDescription.UTF8String);
        auto actual = static_cast<ref::Result const*>(o.contents) + 1;
        auto sorted = static_cast<std::uint32_t const*>(order.contents) + 1;
        for (std::size_t i = 0; i < tasks.size(); ++i) {
          if (std::memcmp(&actual[i], &expected[i], sizeof(ref::Result))) {
            auto const& a = actual[i];
            auto const& b = expected[i];
            std::fprintf(stderr,
                         "case=%zu n=%u kind=%u scale=%d/%d constraints=%d bounds=%g/%g "
                         "p=%g/%g/%g/%g\nactual bin=%d dir=%u partition=%u gain=%.9g (%a) "
                         "sums=%lld/%lld,%lld/%lld float=%a/%a,%a/%a\n"
                         "expect bin=%d dir=%u partition=%u gain=%.9g (%a) "
                         "sums=%lld/%lld,%lld/%lld float=%a/%a,%a/%a\n",
                         cases + i, tasks[i].end - tasks[i].begin, tasks[i].kind, scale.ge,
                         scale.he, tasks[i].monotone, tasks[i].lower, tasks[i].upper, params.lambda,
                         params.alpha, params.delta, params.minimum, a.bin, a.missing, a.partition,
                         a.gain, a.gain, a.left.g, a.left.h, a.right.g, a.right.h, a.lf.g, a.lf.h,
                         a.rf.g, a.rf.h, b.bin, b.missing, b.partition, b.gain, b.gain, b.left.g,
                         b.left.h, b.right.g, b.right.h, b.lf.g, b.lf.h, b.rf.g, b.rf.h);
            throw std::runtime_error("CUDA-formula split result differs (exact comparison)");
          }
          if (tasks[i].kind == 2)
            for (std::size_t j = 0; j < orders[i].size(); ++j) {
              if (orders[i][j] != sorted[tasks[i].begin + j]) {
                auto a = sorted[tasks[i].begin + j], b = orders[i][j];
                auto wa = ref::Weight(ref::Dequantize(hist[tasks[i].begin + a], scale), params);
                auto wb = ref::Weight(ref::Dequantize(hist[tasks[i].begin + b], scale), params);
                auto gpu_weights = static_cast<float const*>(weights.contents);
                std::fprintf(stderr,
                             "category case=%zu n=%zu j=%zu actual=%u expected=%u "
                             "cpu_weights=%a/%a gpu_weights=%a/%a scale=%d/%d\n",
                             cases + i, orders[i].size(), j, a, b, wa, wb,
                             gpu_weights[tasks[i].begin + a], gpu_weights[tasks[i].begin + b],
                             scale.ge, scale.he);
                throw std::runtime_error("Stable category order differs");
              }
            }
        }
        auto guard = static_cast<unsigned char const*>(o.contents);
        for (std::size_t i = 0; i < sizeof(ref::Result); ++i) {
          Check(guard[i] == 0xa5, "Leading result guard changed");
          Check(guard[(tasks.size() + 1) * sizeof(ref::Result) + i] == 0xa5,
                "Trailing result guard changed");
        }
        Check(sorted[-1] == 0xa5a5a5a5u && sorted[hist.size()] == 0xa5a5a5a5u,
              "Category guard changed");
        cases += tasks.size();
        bins_checked += hist.size();
      };
      // Multiple features/tasks per dispatch; empty/singleton features, partial
      // warps and multiple chunks. Exact residual bits sit below FP64 precision.
      for (int variant = 0; variant < 24; ++variant) {
        ref::Scale scale{variant % 3 == 0 ? 0 : 48, variant % 2 ? 48 : 0};
        ref::Parameters params{variant % 4 == 0 ? 0.f : 1.25f, variant % 4 == 1 ? .3f : 0.f,
                               variant % 4 == 2 ? .75f : 0.f, variant % 4 == 3 ? 3.f : 0.f};
        std::vector<ref::Pair> hist;
        std::vector<ref::Task> tasks;
        for (std::uint32_t n : {0u, 1u, 2u, 3u, 31u, 32u, 33u, 127u, 257u, 1025u}) {
          for (std::uint32_t kind = 0; kind < 3; ++kind) {
            auto begin = std::uint32_t(hist.size());
            ref::Pair total{};
            for (std::uint32_t i = 0; i < n; ++i) {
              auto unit = std::int64_t{1} << (variant % 3 == 0 ? 0 : 48);
              auto hunit = std::int64_t{1} << (variant % 2 ? 48 : 0);
              ref::Pair value{
                  (std::int64_t(random() % 33) - 16) * unit + std::int64_t(random() % 7),
                  std::int64_t(random() % 5) * hunit + std::int64_t(random() % 3)};
              if (variant % 6 == 0) value = {std::int64_t(i % 2 ? -4 : 4) * unit, hunit};
              hist.push_back(value);
              total = ref::Add(total, value);
            }
            if (variant % 3 == 2) total = ref::Add(total, {137, 71});
            if (variant == 7)
              total = ref::Add(total, {std::int64_t{1} << 54, std::int64_t{1} << 54});
            tasks.push_back({begin, std::uint32_t(hist.size()), kind,
                             variant % 4 == 0   ? 1u
                             : variant % 4 == 1 ? 3u
                                                : 256u,
                             total, variant % 5 == 0 ? -.25f : -INFINITY,
                             variant % 5 == 0 ? .5f : INFINITY, variant % 3 - 1});
          }
        }
        run(hist, tasks, scale, params);
      }
      // Dense numeric screening: preserve winners across scan chunks, flat and
      // near-tied gains, zero curvature, and bounded/unbounded gain intervals.
      // The independent oracle evaluates every candidate at full precision.
      for (ref::Scale scale : {ref::Scale{32, 32}, {0, 0}, {-32, -32}, {80, 80}, {48, 8}}) {
        for (float lambda : {0.f, 1.f, 1e-8f, 1e8f, 1e9f}) {
          std::vector<ref::Pair> hist;
          std::vector<ref::Task> tasks;
          for (std::uint32_t n : {31u, 32u, 33u, 127u, 255u, 256u, 257u, 1024u}) {
            for (int pattern = 0; pattern < 6; ++pattern) {
              auto begin = std::uint32_t(hist.size());
              ref::Pair total{};
              for (std::uint32_t i = 0; i < n; ++i) {
                ref::Pair value{(std::int64_t(random() % 20001) - 10000) * (1LL << 20),
                                std::int64_t(1 + random() % 2000) * (1LL << 20)};
                if (pattern == 1) value.g = 0;
                if (pattern == 2 || pattern == 3) value.g = 2 * value.h;
                if (pattern == 3 && i == n / 2) value.g += 1;  // Near-tie below FP32 precision.
                if (pattern == 4) value.h = 0;
                if (pattern == 5) value.g = i % 2 ? -1 : 1;
                hist.push_back(value);
                total = ref::Add(total, value);
              }
              tasks.push_back({begin, std::uint32_t(hist.size()), 0, n, total,
                               -INFINITY, INFINITY, 0});
            }
          }
          run(hist, tasks, scale, {lambda, 0, 0, 0});
        }
      }
      // Integer cancellation must survive an FP64-rounded prefix: +/-2^60
      // around a one-unit contribution. Cover both numeric and categorical.
      for (int e : {-80, 0, 54, 100}) {
        for (std::uint32_t kind = 0; kind < 3; ++kind) {
          for (std::uint32_t offset : {0u, 30u, 31u, 32u, 126u, 1023u}) {
            std::vector<ref::Pair> hist(offset, {0, 0});
            hist.insert(
                hist.end(),
                {{std::int64_t{1} << 60, 1}, {1, 1}, {-(std::int64_t{1} << 60), 1}, {-2, 1}});
            std::vector<ref::Task> tasks{
                {0, std::uint32_t(hist.size()), kind, 2048, {-1, 4}, -INFINITY, INFINITY, 0}};
            run(hist, tasks, {60 - e, 0}, {1, 0, 0, 0});
            ++cancellation_cases;
          }
        }
      }
      std::printf("PASS CUDA-formula scalar splits: %zu cases, %zu bins, %zu cancellation cases; "
                  "exact integer children, materialized sums, gains, bins, missing directions, "
                  "partitions, category order and guards. No NVIDIA execution comparison.\n",
                  cases, bins_checked, cancellation_cases);
      return 0;
    } catch (std::exception const& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
