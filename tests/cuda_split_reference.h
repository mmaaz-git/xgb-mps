// Independent host execution of scalar CUDA split formulas/enumeration.
// This is a source-derived oracle, not a captured NVIDIA execution result.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>
namespace cuda_split_reference {
struct Pair {
  std::int64_t g, h;
};
struct Floating {
  double g, h;
};
struct Scale {
  std::int32_t ge, he;
  std::uint32_t valid{1}, padding{0};
};
struct Task {
  std::uint32_t begin, end, kind, limit;
  Pair parent;
  float lower, upper;
  std::int32_t monotone;
  std::uint32_t padding{0};
};
struct Parameters {
  float lambda, alpha, delta, minimum;
};
struct Result {
  Pair left{}, right{};
  float gain{-std::numeric_limits<float>::max()};
  std::int32_t bin{-1};
  std::uint32_t missing{0}, partition{0};
  Floating lf{}, rf{};
};
static_assert(sizeof(Task) == 48 && sizeof(Result) == 80 && sizeof(Scale) == 16);
inline std::int64_t Signed(std::uint64_t bits) {
  std::int64_t value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
inline Pair Add(Pair a, Pair b) {
  return {Signed(std::uint64_t(a.g) + std::uint64_t(b.g)),
          Signed(std::uint64_t(a.h) + std::uint64_t(b.h))};
}
inline Pair Sub(Pair a, Pair b) {
  return {Signed(std::uint64_t(a.g) - std::uint64_t(b.g)),
          Signed(std::uint64_t(a.h) - std::uint64_t(b.h))};
}
inline Floating Dequantize(Pair value, Scale scale) {
  return {std::ldexp(double(value.g), -scale.ge), std::ldexp(double(value.h), -scale.he)};
}
inline float Weight(Floating s, Parameters p, float lower = -INFINITY, float upper = INFINITY) {
  if (s.h <= 0) return 0;
  double threshold = s.g > p.alpha ? s.g - p.alpha : s.g < -p.alpha ? s.g + p.alpha : 0;
  double weight = -threshold / (s.h + p.lambda);
  if (p.delta && std::abs(weight) > p.delta) weight = std::copysign(p.delta, weight);
  return std::clamp(float(weight), lower, upper);
}
inline float Gain(Floating s, float w, Parameters p) {
  if (s.h <= 0) return 0;
  // CUDA TreeEvaluator calls CalcGainGivenWeight<double, float>: w*w is FP32.
  float square = w * w;
  return -(2.0 * s.g * w + (s.h + p.lambda) * square + 2.0 * p.alpha * std::abs(w));
}
inline float SplitGain(Pair l, Pair r, Scale scale, Task task, Parameters p) {
  auto left = Dequantize(l, scale), right = Dequantize(r, scale);
  if (!(left.h > 0 && right.h > 0 && left.h >= p.minimum && right.h >= p.minimum)) return -INFINITY;
  auto lw = Weight(left, p, task.lower, task.upper), rw = Weight(right, p, task.lower, task.upper);
  if ((task.monotone > 0 && !(lw <= rw)) || (task.monotone < 0 && !(lw >= rw))) return -INFINITY;
  return Gain(left, lw, p) + Gain(right, rw, p);
}
inline Result Evaluate(std::vector<Pair> const& hist, Task task, Scale scale, Parameters p,
                       std::vector<std::uint32_t>* sorted) {
  auto n = task.end - task.begin;
  Pair total{};
  for (auto i = task.begin; i < task.end; ++i) total = Add(total, hist[i]);
  Pair missing = Sub(task.parent, total);
  Result best;
  auto update = [&](float gain, Pair left, Pair right, std::uint32_t bin, bool ml,
                    std::uint32_t partition) {
    if (gain > best.gain) {
      best = {left,
              right,
              gain,
              std::int32_t(bin),
              std::uint32_t(ml),
              partition,
              Dequantize(left, scale),
              Dequantize(right, scale)};
    }
  };
  if (task.kind < 2) {
    Pair prefix{};
    for (std::uint32_t i = 0; i < n; ++i) {
      Pair scan = task.kind == 0 ? prefix : Sub(total, hist[task.begin + i]);
      Pair with_missing = Add(scan, missing);
      float left = SplitGain(with_missing, Sub(task.parent, with_missing), scale, task, p);
      float right = SplitGain(scan, Sub(task.parent, scan), scale, task, p);
      bool ml = left > right;
      Pair l = ml ? with_missing : scan;
      update(ml ? left : right, l, Sub(task.parent, l), i, ml, 0);
      prefix = Add(prefix, hist[task.begin + i]);
    }
  } else {
    sorted->resize(n);
    std::iota(sorted->begin(), sorted->end(), 0);
    std::stable_sort(sorted->begin(), sorted->end(), [&](auto a, auto b) {
      return Weight(Dequantize(hist[task.begin + a], scale), p) <
             Weight(Dequantize(hist[task.begin + b], scale), p);
    });
    std::uint32_t limit = std::min(n, task.limit);
    Pair prefix{};
    for (std::uint32_t i = 0; i + 1 < limit; ++i) {
      prefix = Add(prefix, hist[task.begin + (*sorted)[i]]);
      Pair left = Sub(task.parent, prefix);
      update(SplitGain(left, prefix, scale, task, p), left, prefix, i, true, i + 1);
    }
    Pair suffix{};
    for (int i = int(n) - 1; i > int(n - limit); --i) {
      suffix = Add(suffix, hist[task.begin + (*sorted)[i]]);
      Pair right = Sub(task.parent, suffix);
      update(SplitGain(suffix, right, scale, task, p), suffix, right, i, false, i);
    }
  }
  return best;
}
}  // namespace cuda_split_reference
