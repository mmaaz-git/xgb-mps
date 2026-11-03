// Independent CPU differential checks for the production Metal split evaluator.
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
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct Pair {
  double g, h;
};
struct Task {
  uint32_t begin, end, kind, limit;
  double pg, ph;
  float root, lower, upper;
  int32_t monotone;
};
struct Params {
  float lambda, alpha, delta, minimum;
};
struct Result {
  double lg, lh, rg, rh;
  float gain;
  int32_t bin;
  uint32_t missing, partition;
};
static_assert(sizeof(Task) == 48 && sizeof(Result) == 48);
void Require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
Pair Add(Pair a, Pair b) { return {a.g + b.g, a.h + b.h}; }
Pair Sub(Pair a, Pair b) { return {a.g - b.g, a.h - b.h}; }
float Weight(Pair s, Params p, float lower = -INFINITY, float upper = INFINITY) {
  if (s.h <= 0) return 0;
  double threshold = s.g > p.alpha ? s.g - p.alpha : (s.g < -p.alpha ? s.g + p.alpha : 0);
  double w = -threshold / (s.h + p.lambda);
  if (p.delta && std::abs(w) > p.delta) w = std::copysign(p.delta, w);
  return std::clamp(float(w), lower, upper);
}
float Gain(Pair s, float w, Params p) {
  if (s.h <= 0) return 0;
  float square = w * w;
  return -(2.0 * s.g * w + (s.h + p.lambda) * square + 2.0 * p.alpha * std::abs(w));
}
Result Reference(std::vector<Pair> const& hist, Task t, Params p, std::vector<uint32_t>& order) {
  Result best{0, 0, 0, 0, 0, -1, 0, 0};
  Pair parent{t.pg, t.ph}, sum{0, 0};
  uint32_t n = t.end - t.begin;
  auto consider = [&](Pair left, Pair right, int bin, uint32_t missing, uint32_t partition) {
    if (!(left.h > 0 && right.h > 0) || left.h < p.minimum || right.h < p.minimum) return;
    float wl = Weight(left, p, t.lower, t.upper), wr = Weight(right, p, t.lower, t.upper);
    if ((t.monotone > 0 && !(wl <= wr)) || (t.monotone < 0 && !(wl >= wr))) return;
    float gain = (Gain(left, wl, p) + Gain(right, wr, p)) - t.root;
    if (std::isfinite(gain) && gain > best.gain)
      best = {left.g, left.h, right.g, right.h, gain, bin, missing, partition};
  };
  if (t.kind == 0) {
    for (uint32_t i = 0; i < n; ++i) {
      sum = Add(sum, hist[t.begin + i]);
      consider(sum, Sub(parent, sum), i, 0, 0);
    }
    if (sum.g != parent.g || sum.h != parent.h) {
      sum = {0, 0};
      for (int i = n - 1; i >= 0; --i) {
        sum = Add(sum, hist[t.begin + i]);
        consider(Sub(parent, sum), sum, i, 1, 0);
      }
    }
  } else if (t.kind == 1) {
    for (uint32_t i = 0; i < n; ++i) sum = Add(sum, hist[t.begin + i]);
    Pair missing = Sub(parent, sum);
    for (uint32_t i = 0; i < n; ++i) {
      Pair right = hist[t.begin + i];
      consider(Sub(parent, right), right, i, 1, 0);
      right = Add(right, missing);
      consider(Sub(parent, right), right, i, 0, 0);
    }
  } else {
    order.resize(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
      return Weight(hist[t.begin + a], p) < Weight(hist[t.begin + b], p);
    });
    uint32_t limit = std::min(n, t.limit);
    for (uint32_t i = 0; i + 1 < limit; ++i) {
      sum = Add(sum, hist[t.begin + order[i]]);
      consider(Sub(parent, sum), sum, i, 1, i + 1);
    }
    sum = {0, 0};
    for (int i = n - 1; i > int(n - limit); --i) {
      sum = Add(sum, hist[t.begin + order[i]]);
      consider(sum, Sub(parent, sum), i, 0, i);
    }
  }
  return best;
}
struct MultiTask {
  uint32_t begin, end, kind, limit, parent_begin, targets;
  double root;
  int32_t monotone;
  uint32_t constrained;
};
struct Bounds {
  float lower, upper;
};
static_assert(sizeof(MultiTask) == 40);
double PreciseWeight(Pair s, Params p) {
  if (s.h <= 0) return 0;
  double threshold = s.g > p.alpha ? s.g - p.alpha : (s.g < -p.alpha ? s.g + p.alpha : 0);
  double w = -threshold / (s.h + p.lambda);
  if (p.delta && std::abs(w) > p.delta) w = std::copysign(p.delta, w);
  return w;
}
template <typename W>
double PreciseGain(Pair s, W w, Params p) {
  W square = w * w;
  return -(2.0 * s.g * w + (s.h + p.lambda) * square + 2.0 * p.alpha * std::abs(w));
}
double UnconstrainedGain(Pair s, Params p) {
  if (s.h <= 0) return 0;
  if (p.delta) return PreciseGain(s, PreciseWeight(s, p), p);
  double g = s.g;
  if (p.alpha) g = g > p.alpha ? g - p.alpha : (g < -p.alpha ? g + p.alpha : 0);
  return g * g / (s.h + p.lambda);
}
struct VectorReference {
  Result split{0, 0, 0, 0, 0, -1, 0, 0};
  std::vector<Pair> left, right;
  std::vector<uint32_t> order;
};
VectorReference ReferenceMulti(std::vector<Pair> const& hist, MultiTask t, Params p,
                               std::vector<Pair> const& parents,
                               std::vector<Bounds> const& bounds) {
  VectorReference best;
  uint32_t k = t.targets, n = (t.end - t.begin) / k;
  std::vector<Pair> left(k), right(k), sum(k);
  best.order.resize(n);
  std::iota(best.order.begin(), best.order.end(), 0);
  auto parent = [&](uint32_t target) { return parents[t.parent_begin + target]; };
  auto consider = [&](int bin, uint32_t missing, uint32_t partition) {
    double gain = 0, lh = 0, rh = 0;
    for (uint32_t j = 0; j < k; ++j) {
      lh += left[j].h;
      rh += right[j].h;
      if (!t.constrained) {
        gain += UnconstrainedGain(left[j], p);
        gain += UnconstrainedGain(right[j], p);
      }
    }
    lh /= double(k);
    rh /= double(k);
    if (!(lh > 0 && rh > 0) || lh < p.minimum || rh < p.minimum) return;
    if (t.constrained)
      for (uint32_t j = 0; j < k; ++j) {
        auto b = bounds[t.parent_begin + j];
        float wl = Weight(left[j], p, b.lower, b.upper), wr = Weight(right[j], p, b.lower, b.upper);
        if ((t.monotone > 0 && !(wl <= wr)) || (t.monotone < 0 && !(wl >= wr))) {
          Params pooled = p;
          pooled.alpha = 2.0f * p.alpha;
          pooled.lambda = 2.0f * p.lambda;
          wl = Weight(Add(left[j], right[j]), pooled, b.lower, b.upper);
          wr = wl;
        }
        gain += PreciseGain(left[j], wl, p);
        gain += PreciseGain(right[j], wr, p);
      }
    float change = gain - t.root;
    if (std::isfinite(change) && change > best.split.gain) {
      best.split = {0, 0, 0, 0, change, bin, missing, partition};
      best.left = left;
      best.right = right;
    }
  };
  if (t.kind == 0) {
    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < k; ++j) {
        sum[j] = Add(sum[j], hist[t.begin + i * k + j]);
        left[j] = sum[j];
        right[j] = Sub(parent(j), sum[j]);
      }
      consider(i, 0, 0);
    }
    bool missing = false;
    for (uint32_t j = 0; j < k; ++j) missing |= sum[j].g != parent(j).g || sum[j].h != parent(j).h;
    if (missing) {
      sum.assign(k, {0, 0});
      for (int i = n - 1; i >= 0; --i) {
        for (uint32_t j = 0; j < k; ++j) {
          sum[j] = Add(sum[j], hist[t.begin + i * k + j]);
          right[j] = sum[j];
          left[j] = Sub(parent(j), sum[j]);
        }
        consider(i, 1, 0);
      }
    }
  } else if (t.kind == 1) {
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t j = 0; j < k; ++j) sum[j] = Add(sum[j], hist[t.begin + i * k + j]);
    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < k; ++j) {
        right[j] = hist[t.begin + i * k + j];
        left[j] = Sub(parent(j), right[j]);
      }
      consider(i, 1, 0);
      for (uint32_t j = 0; j < k; ++j) {
        right[j] = Add(right[j], Sub(parent(j), sum[j]));
        left[j] = Sub(parent(j), right[j]);
      }
      consider(i, 0, 0);
    }
  } else {
    std::vector<double> scores(n, 0);
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t j = 0; j < k; ++j) {
        float product = Weight(parent(j), p) * Weight(hist[t.begin + i * k + j], p);
        scores[i] += product;
      }
    std::stable_sort(best.order.begin(), best.order.end(),
                     [&](auto a, auto b) { return scores[a] < scores[b]; });
    uint32_t limit = std::min(n, t.limit);
    for (uint32_t i = 0; i + 1 < limit; ++i) {
      for (uint32_t j = 0; j < k; ++j) {
        sum[j] = Add(sum[j], hist[t.begin + best.order[i] * k + j]);
        right[j] = sum[j];
        left[j] = Sub(parent(j), sum[j]);
      }
      consider(i, 1, i + 1);
    }
    sum.assign(k, {0, 0});
    for (int i = n - 1; i > int(n - limit); --i) {
      for (uint32_t j = 0; j < k; ++j) {
        sum[j] = Add(sum[j], hist[t.begin + best.order[i] * k + j]);
        left[j] = sum[j];
        right[j] = Sub(parent(j), sum[j]);
      }
      consider(i, 0, i);
    }
  }
  return best;
}
void CheckMulti(id<MTLDevice> device, id<MTLLibrary> lib, id<MTLCommandQueue> queue) {
  NSError* error = nil;
  auto function = [lib newFunctionWithName:@"xgb_evaluate_multi_splits"];
  auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  Require(pipeline != nil, "No vector split pipeline");
  std::mt19937_64 rng(667);
  size_t checked = 0;
  for (Params p : std::vector<Params>{{1, 0, 0, 1}, {0, 0, 0, 0}, {3, 0.5, 0.2, 4}}) {
    std::vector<Pair> hist, parents;
    std::vector<Bounds> bounds;
    std::vector<MultiTask> tasks;
    std::vector<VectorReference> expected;
    for (uint32_t n : {1, 2, 7, 33, 129, 257})
      for (uint32_t k : {1, 2, 3, 5, 17})
        for (uint32_t kind = 0; kind < 3; ++kind)
          for (int c = 0; c < 8; ++c) {
            uint32_t begin = hist.size(), pb = parents.size();
            std::vector<Pair> sum(k, {0, 0});
            for (uint32_t i = 0; i < n; ++i)
              for (uint32_t j = 0; j < k; ++j) {
                double scale = std::ldexp(1.0, (c % 3 - 1) * 20);
                Pair bin{double(int(rng() % 2001) - 1000) / 8 * scale,
                         double(rng() % 201) / 8 * scale};
                if (c == 3) bin = {0, 1};
                if (c == 4) bin = {i % 2 ? 1.0 : -1.0, 1};
                if (c == 5 && j == 0) bin = {0, 0};
                if (c >= 6) {
                  bin.g = std::ldexp(bin.g, int(rng() % 40) - 20);
                  bin.h = std::ldexp(bin.h, int(rng() % 40) - 20);
                }
                hist.push_back(bin);
                sum[j] = Add(sum[j], bin);
              }
            double root = 0;
            for (uint32_t j = 0; j < k; ++j) {
              if (c % 2)
                sum[j] = Add(sum[j], Pair{double(int(rng() % 101) - 50), double(rng() % 100)});
              parents.push_back(sum[j]);
              bounds.push_back({c % 4 == 0 ? -0.1f : -INFINITY, c % 4 == 0 ? 0.3f : INFINITY});
              root += PreciseGain(sum[j], Weight(sum[j], p), p);
            }
            MultiTask t{begin,     uint32_t(hist.size()), kind, c % 2 ? 16u : 4096u, pb, k, root,
                        c % 3 - 1, uint32_t(c % 3 != 0)};
            tasks.push_back(t);
            expected.push_back(ReferenceMulti(hist, t, p, parents, bounds));
          }
    auto buffer = [&](void const* data, size_t bytes) {
      return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    };
    auto allocate = [&](size_t bytes) {
      return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    };
    auto input = buffer(hist.data(), hist.size() * sizeof(Pair)),
         metadata = buffer(tasks.data(), tasks.size() * sizeof(MultiTask));
    auto parent = buffer(parents.data(), parents.size() * sizeof(Pair)),
         bound = buffer(bounds.data(), bounds.size() * sizeof(Bounds));
    auto output = allocate(tasks.size() * sizeof(Result)), order = allocate(hist.size() * 4),
         scores = allocate(hist.size() * 8), scans = allocate(hist.size() * 32),
         children = allocate(parents.size() * 2 * sizeof(Pair));
    auto cmd = [queue commandBuffer];
    auto e = [cmd computeCommandEncoder];
    [e setComputePipelineState:pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:metadata offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e setBuffer:output offset:0 atIndex:3];
    [e setBuffer:order offset:0 atIndex:4];
    [e setBuffer:scores offset:0 atIndex:5];
    [e setBuffer:scans offset:0 atIndex:6];
    [e setBuffer:parent offset:0 atIndex:7];
    [e setBuffer:bound offset:0 atIndex:8];
    [e setBuffer:children offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [e endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    Require(cmd.status == MTLCommandBufferStatusCompleted, "Vector split GPU failure");
    auto actual = static_cast<Result const*>(output.contents);
    auto sorted = static_cast<uint32_t const*>(order.contents);
    auto sums = static_cast<Pair const*>(children.contents);
    for (size_t i = 0; i < tasks.size(); ++i) {
      auto a = actual[i], b = expected[i].split;
      auto t = tasks[i];
      if (a.bin != b.bin || a.missing != b.missing || a.partition != b.partition ||
          a.gain != b.gain) {
        fprintf(stderr,
                "vector case=%zu kind=%u targets=%u n=%u split=%d/%d default=%u/%u gain=%a/%a\n", i,
                t.kind, t.targets, (t.end - t.begin) / t.targets, a.bin, b.bin, a.missing,
                b.missing, a.gain, b.gain);
        throw std::runtime_error("Vector split selection differs");
      }
      if (a.bin >= 0)
        for (uint32_t j = 0; j < t.targets; ++j) {
          Pair l = sums[2 * t.parent_begin + j], r = sums[2 * t.parent_begin + t.targets + j],
               el = expected[i].left[j], er = expected[i].right[j];
          Require(l.g == el.g && l.h == el.h && r.g == er.g && r.h == er.h,
                  "Vector child sums differ");
        }
      if (t.kind == 2)
        for (uint32_t j = 0; j < (t.end - t.begin) / t.targets; ++j)
          Require(sorted[t.begin + j] == expected[i].order[j],
                  "Vector category scores/order differ");
      ++checked;
    }
    printf("Checked %zu vector split tasks\n", checked);
    fflush(stdout);
  }
  printf("PASS %zu vector split selections, child sums and category orders\n", checked);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      Require(argc == 2, "Pass production Metal source");
      auto device = MTLCreateSystemDefaultDevice();
      Require(device != nil, "No Metal GPU");
      std::ifstream file(argv[1]);
      std::string source{std::istreambuf_iterator<char>(file), {}};
      auto options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&error];
      if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String);
      auto function = [lib newFunctionWithName:@"xgb_evaluate_splits"];
      auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      Require(pipeline != nil, "No split pipeline");
      auto queue = [device newCommandQueue];
      std::mt19937_64 rng(329);
      size_t checked = 0, category_orders = 0;
      for (Params p : std::vector<Params>{
               {1, 0, 0, 1}, {0, 0, 0, 0}, {3, 2, 0, 4}, {1, 0.5, 0.2, 0}, {0.1, 5, 2, 10}}) {
        std::vector<Pair> hist;
        std::vector<Task> tasks;
        std::vector<Result> expected;
        std::vector<std::vector<uint32_t>> orders;
        for (uint32_t n : {1, 2, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 1025})
          for (uint32_t kind = 0; kind < 3; ++kind)
            for (int c = 0; c < 18; ++c) {
              uint32_t begin = hist.size();
              Pair sum{0, 0};
              for (uint32_t i = 0; i < n; ++i) {
                double scale = std::ldexp(1.0, (c % 3 - 1) * 28);
                Pair bin{double(int(rng() % 2001) - 1000) / 8 * scale,
                         double(rng() % 201) / 8 * scale};
                if (c == 3) bin = {0, 1};  // Exact gain and category-order ties.
                if (c == 4) bin = {i % 2 ? 1.0 : -1.0, 1};
                if (c == 5) bin.h = 0;
                if (c >= 12) {
                  bin.g = std::ldexp(bin.g, int(rng() % 80) - 40);
                  bin.h = std::ldexp(bin.h, int(rng() % 80) - 40);
                }
                hist.push_back(bin);
                sum = Add(sum, bin);
              }
              if (c % 2) sum = Add(sum, Pair{double(int(rng() % 101) - 50), double(rng() % 100)});
              Task t{begin,
                     uint32_t(hist.size()),
                     kind,
                     c % 2 ? 64u : 4096u,
                     sum.g,
                     sum.h,
                     Gain(sum, Weight(sum, p), p),
                     c % 4 == 0 ? -0.1f : -INFINITY,
                     c % 4 == 0 ? 0.3f : INFINITY,
                     c % 3 - 1};
              tasks.push_back(t);
              orders.emplace_back();
              expected.push_back(Reference(hist, t, p, orders.back()));
            }
        auto buffer = [&](void const* data, size_t bytes) {
          return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        };
        auto allocate = [&](size_t bytes) {
          return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };
        auto input = buffer(hist.data(), hist.size() * sizeof(Pair));
        auto metadata = buffer(tasks.data(), tasks.size() * sizeof(Task));
        auto output = allocate(tasks.size() * sizeof(Result)), order = allocate(hist.size() * 4),
             weights = allocate(hist.size() * 4), scans = allocate(hist.size() * 32);
        auto command = [queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:pipeline];
        [e setBuffer:input offset:0 atIndex:0];
        [e setBuffer:metadata offset:0 atIndex:1];
        [e setBytes:&p length:sizeof(p) atIndex:2];
        [e setBuffer:output offset:0 atIndex:3];
        [e setBuffer:order offset:0 atIndex:4];
        [e setBuffer:weights offset:0 atIndex:5];
        [e setBuffer:scans offset:0 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [e endEncoding];
        [command commit];
        [command waitUntilCompleted];
        Require(command.status == MTLCommandBufferStatusCompleted, "GPU dispatch failed");
        auto actual = static_cast<Result const*>(output.contents);
        auto sorted = static_cast<uint32_t const*>(order.contents);
        for (size_t i = 0; i < tasks.size(); ++i) {
          auto a = actual[i], b = expected[i];
          if (a.bin != b.bin || a.missing != b.missing || a.partition != b.partition ||
              a.gain != b.gain || a.lg != b.lg || a.lh != b.lh || a.rg != b.rg || a.rh != b.rh) {
            fprintf(stderr,
                    "case %zu kind=%u n=%u: split %d/%d missing %u/%u gain %a/%a "
                    "stats=(%a,%a,%a,%a)/(%a,%a,%a,%a)\n",
                    i, tasks[i].kind, tasks[i].end - tasks[i].begin, a.bin, b.bin, a.missing,
                    b.missing, a.gain, b.gain, a.lg, a.lh, a.rg, a.rh, b.lg, b.lh, b.rg, b.rh);
            throw std::runtime_error("Split differs from CPU reference");
          }
          if (tasks[i].kind == 2) {
            for (size_t j = 0; j < orders[i].size(); ++j)
              Require(sorted[tasks[i].begin + j] == orders[i][j], "Category ordering differs");
            ++category_orders;
          }
          ++checked;
        }
        printf("Checked %zu split tasks (%zu stable category orders)\n", checked, category_orders);
        fflush(stdout);
      }
      printf("PASS %zu exact split selections, gains, child sums and %zu category orders on %s\n",
             checked, category_orders, device.name.UTF8String);
      CheckMulti(device, lib, queue);
      return 0;
    } catch (std::exception const& e) {
      fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
