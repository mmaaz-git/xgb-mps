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
std::size_t cases = 0, bins_checked = 0, cancellation_cases = 0, screened_features = 0;
struct BoundInput {
  ref::Pair left, right;
  ref::Scale scale;
  ref::Parameters params;
};
static_assert(sizeof(BoundInput) == 64);

void CheckQuantizedBounds(id<MTLDevice> device, id<MTLLibrary> library) {
  NSError* error = nil;
  auto pipeline = [device newComputePipelineStateWithFunction:
      [library newFunctionWithName:@"test_quantized_bound"] error:&error];
  Check(pipeline != nil, "Missing quantized bound check pipeline");
  std::mt19937_64 random(994791);
  std::vector<BoundInput> data;
  for (std::size_t i = 0; i < 1000000; ++i) {
    auto grad = [&]() { return ref::Signed(random()); };
    auto hess = [&]() { return std::int64_t(random() >> 1); };
    ref::Scale scale{int(35 + random() % 60), int(35 + random() % 60)};
    if (i % 3 == 0) scale = {int(random() % 340) - 170, int(random() % 340) - 170};
    BoundInput v{{grad(), hess()}, {grad(), hess()}, scale,
                 {i % 5 ? float(std::ldexp(1.0, int(random() % 50) - 25)) : 0.f,
                  0, 0, i % 7 ? 0.f : 1.f}};
    if (i % 11 == 0) v.left.g = 0;
    if (i % 13 == 0) v.right.h = -v.right.h;
    if (i % 17 == 0) v.left.h = 0;
    if (i % 19 == 0) {
      v.params.minimum = float(ref::Dequantize(v.left, scale).h);
      if (i % 2)
        v.params.minimum = std::nextafter(v.params.minimum, i % 4 == 1 ? INFINITY : -INFINITY);
    }
    if (i % 23 == 0) v.params.lambda = 1e9f;
    if (i % 29 == 0) v.params.minimum = std::numeric_limits<float>::denorm_min();
    data.push_back(v);
  }
  // Include integer->double->float double rounding on either side of FP32
  // midpoints, including above 2^53. Test both usable intervals and uncertainty
  // at min_child_weight; the latter must never create a spurious lower bound.
  auto midpoint_begin = data.size();
  for (int shift = 1; shift <= 39; ++shift) {
    for (std::uint64_t mantissa : {0x800000ULL, 0x800001ULL, 0xfffffeULL, 0xffffffULL}) {
      auto mid = std::int64_t((2 * mantissa + 1) << (shift - 1));
      for (std::int64_t offset : {-257LL, -2LL, -1LL, 0LL, 1LL, 2LL, 257LL}) {
        for (int sign : {-1, 1}) {
          for (float lambda : {0.f, 1.f, 1e-8f, 1e8f, 1e9f}) {
            BoundInput v{{sign * (mid + offset), mid + offset},
                         {-(mid - offset), mid - offset},
                         {shift + 23, shift + 23}, {lambda, 0, 0, 0}};
            data.push_back(v);
            for (int direction : {-1, 0, 1}) {
              v.params.minimum = float(ref::Dequantize(v.left, v.scale).h);
              if (direction)
                v.params.minimum = std::nextafter(v.params.minimum, direction * INFINITY);
              data.push_back(v);
            }
          }
        }
      }
    }
  }
  auto midpoint_end = data.size();
  for (int exponent : {-200, -150, -149, -148, -127, -126, -125, -97, -96, -95,
                        0, 95, 96, 97, 125, 126, 127, 148, 149, 150, 200}) {
    for (std::int64_t raw : {0LL, 1LL, -1LL, 2LL, -2LL, (1LL << 62), -(1LL << 62)}) {
      for (float minimum : {0.f, 1e-8f, 1.f, 1e8f, INFINITY})
        data.push_back({{raw, raw}, {-raw, raw}, {exponent, exponent}, {1, 0, 0, minimum}});
    }
  }
  // Both sides of the supported finite interval endpoints.
  for (auto edge : {1e-8f, 1e8f}) {
    int exponent = edge < 1.f ? 60 : 20;
    auto center = std::int64_t(std::ldexp(double(edge), exponent));
    for (std::int64_t offset : {-2049LL, -1025LL, -1LL, 0LL, 1LL, 1025LL, 2049LL}) {
      for (float lambda : {0.f, 1.f, 1e8f, 1e9f})
        data.push_back({{center + offset, center + offset}, {-center, center},
                        {exponent, exponent}, {lambda, 0, 0, 0}});
    }
  }
  for (auto raw : {std::numeric_limits<std::int64_t>::min(),
                   std::numeric_limits<std::int64_t>::max()}) {
    for (int exponent : {-97, -96, 0, 63, 96, 97})
      data.push_back({{raw, 1LL << 62}, {raw, 1LL << 62},
                      {exponent, exponent}, {1, 0, 0, 0}});
  }
  auto input = [device newBufferWithBytes:data.data() length:data.size() * sizeof(BoundInput)
                                  options:MTLResourceStorageModeShared];
  auto output = [device newBufferWithLength:data.size() * sizeof(float) * 2
                                   options:MTLResourceStorageModeShared];
  auto command = [[device newCommandQueue] commandBuffer];
  auto encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(data.size(), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  Check(command.status == MTLCommandBufferStatusCompleted, "Quantized bound command failed");
  auto bounds = static_cast<float const*>(output.contents);
  std::size_t finite = 0, unbounded = 0, invalid = 0, midpoint_finite = 0;
  for (std::size_t i = 0; i < data.size(); ++i) {
    auto v = data[i];
    auto l = ref::Dequantize(v.left, v.scale), r = ref::Dequantize(v.right, v.scale);
    float lo = bounds[2 * i], hi = bounds[2 * i + 1];
    bool valid = l.h > 0 && r.h > 0 && l.h >= v.params.minimum && r.h >= v.params.minimum;
    if (!valid) {
      Check(lo == -INFINITY, "Invalid candidate could raise split cutoff");
      ++invalid;
    }
    if (lo == -INFINITY && hi == INFINITY) {
      ++unbounded;
      continue;
    }
    ref::Task task{0, 0, 0, 0, ref::Add(v.left, v.right), -INFINITY, INFINITY, 0};
    float actual = ref::SplitGain(v.left, v.right, v.scale, task, v.params);
    if (!(lo <= actual && actual <= hi)) {
      std::fprintf(stderr, "bound=%zu scale=%d/%d min=%a gain=%a interval=%a/%a valid=%d\n",
                   i, v.scale.ge, v.scale.he, v.params.minimum, actual, lo, hi, valid);
      throw std::runtime_error("Quantized bound excluded full-precision CUDA-formula gain");
    }
    ++finite;
    if (i >= midpoint_begin && i < midpoint_end && v.params.minimum == 0) ++midpoint_finite;
  }
  Check(finite > 100000 && unbounded > 100000 && invalid > 10000 && midpoint_finite > 8000,
        "Insufficient quantized bound/fallback/midpoint coverage");
  std::printf("PASS %zu quantized gain intervals: %zu finite, %zu unbounded, %zu invalid "
               "cannot raise cutoff, %zu finite midpoint cases\n",
               data.size(), finite, unbounded, invalid, midpoint_finite);
}
struct FloatComparisonInput {
  std::uint32_t a, b;
};
void CheckFloatOrdering(id<MTLDevice> device, id<MTLLibrary> library) {
  NSError* error = nil;
  auto pipeline = [device newComputePipelineStateWithFunction:
      [library newFunctionWithName:@"test_float_comparison"] error:&error];
  Check(pipeline != nil, "Missing FP32 comparison check pipeline");
  std::mt19937 random(391047);
  std::vector<FloatComparisonInput> data;
  for (std::size_t i = 0; i < 1000000; ++i) {
    auto a = random(), b = random();
    data.push_back({a, b});
    if (i % 10 == 0) {
      data.push_back({a, a});
      data.push_back({a, a + 1});
      data.push_back({a, a - 1});
      data.push_back({a, a ^ 0x80000000u});
    }
  }
  std::vector<std::uint32_t> special{0, 1, 2, 0x7fffff, 0x800000, 0x800001,
      0x3f7fffff, 0x3f800000, 0x3f800001, 0x7f7fffff, 0x7f800000,
      0x7f800001, 0x7fc00000, 0x7fffffff};
  auto count = special.size();
  for (std::size_t i = 0; i < count; ++i) special.push_back(special[i] | 0x80000000u);
  for (auto a : special) for (auto b : special) data.push_back({a, b});
  auto input = [device newBufferWithBytes:data.data() length:data.size() * sizeof(FloatComparisonInput)
                                  options:MTLResourceStorageModeShared];
  auto output = [device newBufferWithLength:data.size() * sizeof(FloatComparisonInput)
                                   options:MTLResourceStorageModeShared];
  auto command = [[device newCommandQueue] commandBuffer];
  auto encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(data.size(), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  Check(command.status == MTLCommandBufferStatusCompleted, "FP32 comparison command failed");
  auto results = static_cast<FloatComparisonInput const*>(output.contents);
  for (std::size_t i = 0; i < data.size(); ++i) {
    float a, b;
    std::memcpy(&a, &data[i].a, sizeof(a));
    std::memcpy(&b, &data[i].b, sizeof(b));
    if (results[i].a != std::uint32_t(a < b) || results[i].b != std::uint32_t(a <= b)) {
      std::fprintf(stderr, "pair=%zu a=%08x b=%08x actual=%u/%u expected=%u/%u\n",
                   i, data[i].a, data[i].b, results[i].a, results[i].b,
                   unsigned(a < b), unsigned(a <= b));
      throw std::runtime_error("FP32 comparison differs from native IEEE ordering");
    }
  }
  std::printf("PASS %zu exact FP32 less-than/less-equal pairs; signed zeros, subnormals, "
               "infinities, signaling/quiet NaNs, adjacent values and random bit patterns\n",
               data.size());
}

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Check(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Check(device != nil, "No Metal device");
      std::ifstream f(argv[1]);
      std::string source{std::istreambuf_iterator<char>(f), {}};
      source += R"(
struct BoundCheckInput { ulong2 left, right; GradientScale scale; SplitParameters params; };
kernel void test_quantized_bound(device const BoundCheckInput* data [[buffer(0)]],
                                 device float2* output [[buffer(1)]],
                                 uint i [[thread_position_in_grid]]) {
  auto v = data[i];
  output[i] = xgb_quantized_split_gain_bound(v.left, v.right, v.scale, v.params);
}
kernel void test_float_comparison(device const uint2* input [[buffer(0)]],
                                  device uint2* output [[buffer(1)]],
                                  uint i [[thread_position_in_grid]]) {
  float a = as_type<float>(input[i].x), b = as_type<float>(input[i].y);
  output[i] = uint2(xgb_cuda_float_lt(a, b), xgb_cuda_float_le(a, b));
}
)";
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:options
                                            error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      CheckQuantizedBounds(device, library);
      CheckFloatOrdering(device, library);
      auto pipeline = [device newComputePipelineStateWithFunction:
                                  [library newFunctionWithName:@"xgb_evaluate_quantized_splits"]
                                                            error:&error];
      Check(pipeline != nil, "Missing integer split pipeline");
      auto resident_pipeline = [device newComputePipelineStateWithFunction:
          [library newFunctionWithName:@"xgb_evaluate_resident_quantized_splits"] error:&error];
      Check(resident_pipeline != nil, "Missing resident integer split pipeline");
      auto bounds_pipeline = [device newComputePipelineStateWithFunction:
          [library newFunctionWithName:@"xgb_feature_gain_bounds"] error:&error];
      auto prune_pipeline = [device newComputePipelineStateWithFunction:
          [library newFunctionWithName:@"xgb_feature_gain_prune"] error:&error];
      Check(bounds_pipeline && prune_pipeline, "Missing cross-feature screening pipelines");
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
        auto ranges = upload(nullptr, tasks.size() * 2 * sizeof(float));
        // Two independent nodes and an empty node exercise range isolation.
        std::uint32_t half = tasks.size() / 2, count = tasks.size();
        std::uint32_t nodes[]{0, half, 0, 1, half, count, 1, 1, count, count, 2, 1};
        auto node_input = upload(nodes, sizeof(nodes));
        struct Segment { std::uint64_t source; std::uint32_t output, bins, stride, padding; };
        static_assert(sizeof(Segment) == 24);
        std::vector<Segment> segments(tasks.size());
        std::vector<ref::Pair> cache;
        // Reverse feature storage independently of task order. Every evaluation
        // must leave the cache unchanged for subsequent nodes/dispatches.
        for (std::size_t i = tasks.size(); i-- > 0;) {
          auto task = tasks[i];
          segments[i] = {cache.size(), task.begin, task.end - task.begin, 1, 0};
          cache.insert(cache.end(), hist.begin() + task.begin, hist.begin() + task.end);
        }
        auto resident_hist = upload(cache.data(), cache.size() * sizeof(ref::Pair));
        auto gather = upload(segments.data(), segments.size() * sizeof(Segment));
        for (bool resident : {false, true}) for (bool screened : {false, true}) {
          std::memcpy(t.contents, tasks.data(), tasks.size() * sizeof(ref::Task));
          auto command = [queue commandBuffer];
          if (screened) {
            auto e = [command computeCommandEncoder];
            [e setComputePipelineState:bounds_pipeline];
            [e setBuffer:resident_hist offset:0 atIndex:0];
            [e setBuffer:t offset:0 atIndex:1];
            [e setBytes:&params length:sizeof(params) atIndex:2];
            [e setBuffer:ranges offset:0 atIndex:3];
            [e setBytes:&scale length:sizeof(scale) atIndex:4];
            [e setBuffer:gather offset:0 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding];
            e = [command computeCommandEncoder];
            [e setComputePipelineState:prune_pipeline];
            [e setBuffer:ranges offset:0 atIndex:0];
            [e setBuffer:t offset:0 atIndex:1];
            [e setBuffer:node_input offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(3, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding];
          }
          auto e = [command computeCommandEncoder];
          [e setComputePipelineState:resident ? resident_pipeline : pipeline];
          [e setBuffer:resident ? resident_hist : h offset:0 atIndex:0];
          [e setBuffer:t offset:0 atIndex:1];
          [e setBytes:&params length:sizeof(params) atIndex:2];
          [e setBuffer:o offset:sizeof(ref::Result) atIndex:3];
          [e setBuffer:order offset:sizeof(std::uint32_t) atIndex:4];
          [e setBuffer:weights offset:0 atIndex:5];
          [e setBuffer:scans offset:0 atIndex:6];
          [e setBytes:&scale length:sizeof(scale) atIndex:7];
          if (resident) [e setBuffer:gather offset:0 atIndex:8];
          [e dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
          [e endEncoding];
          [command commit];
          [command waitUntilCompleted];
          if (command.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
          auto actual = static_cast<ref::Result const*>(o.contents) + 1;
          auto sorted = static_cast<std::uint32_t const*>(order.contents) + 1;
          auto flags = static_cast<ref::Task const*>(t.contents);
          auto intervals = static_cast<float const*>(ranges.contents);
          for (std::size_t i = 0; i < tasks.size(); ++i) {
            if (screened && expected[i].bin >= 0) {
              Check(intervals[2*i] <= expected[i].gain && expected[i].gain <= intervals[2*i+1],
                    "Feature gain interval excludes exhaustive oracle maximum");
            }
            if (flags[i].padding) {
              auto begin = i < half ? 0u : half, end = i < half ? half : count;
              float maximum = -std::numeric_limits<float>::max();
              for (auto j = begin; j < end; ++j) maximum = std::max(maximum, expected[j].gain);
              Check(expected[i].gain < maximum, "Screening discarded a winning feature or tie");
              Check(actual[i].bin == -1, "Screened feature remained a valid candidate");
              ++screened_features;
              continue;
            }
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
          Check(std::memcmp(resident_hist.contents, cache.data(), cache.size()*sizeof(ref::Pair)) == 0,
                "Resident evaluator mutated the histogram cache");
          cases += tasks.size();
          bins_checked += hist.size();
        }
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
      // Curvature exactly at, just below and just above min_child_weight.
      // An empty leading bin matches numeric quantile histograms; duplicate
      // features ensure tied maxima are never discarded across scan chunks.
      for (float minimum : {0.f, 1.f, std::nextafter(1.f, 0.f), std::nextafter(1.f, INFINITY)}) {
        std::vector<ref::Pair> hist;
        std::vector<ref::Task> tasks;
        for (unsigned f = 0; f < 64; ++f) {
          auto begin = std::uint32_t(hist.size());
          ref::Pair total{};
          for (unsigned i = 0; i < 257; ++i) {
            ref::Pair value{i < 128 ? (1LL << 32) : -(1LL << 32), 1LL << 32};
            if (i == 0) value = {0, 0};
            if (f % 3 == 0) value.g /= 2;
            hist.push_back(value); total = ref::Add(total, value);
          }
          tasks.push_back({begin, std::uint32_t(hist.size()), 0, 257, total, -INFINITY, INFINITY, 0});
        }
        run(hist, tasks, {32, 32}, {1, 0, 0, minimum});
      }
      Check(screened_features > 0, "Cross-feature checks never exercised rejection");
      std::printf("PASS %zu conservatively screened features; exhaustive winners/ties retained\n", screened_features);
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
