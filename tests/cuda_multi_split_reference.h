// Independent host oracle for CUDA multi_evaluate_splits.cu. No NVIDIA execution.
#pragma once
#include "cuda_split_reference.h"
namespace cuda_multi_split_reference {
using cuda_split_reference::Add;
using cuda_split_reference::Dequantize;
using cuda_split_reference::Floating;
using cuda_split_reference::Pair;
using cuda_split_reference::Parameters;
using cuda_split_reference::Scale;
using cuda_split_reference::Sub;
struct Task {
  std::uint32_t begin, end, kind, limit, parent_begin, targets;
  std::int32_t monotone;
  std::uint32_t constrained, one_pass{0}, padding{0};
};
struct Bounds {
  float lower, upper;
};
struct Result {
  float gain{-std::numeric_limits<float>::max()};
  std::int32_t bin{-1};
  std::uint32_t missing{0}, partition{0};
};
static_assert(sizeof(Task) == 40 && sizeof(Result) == 16);
inline double WeightDouble(Floating s, Parameters p) {
  if (s.h <= 0) return 0;
  double threshold = s.g > p.alpha ? s.g - p.alpha : s.g < -p.alpha ? s.g + p.alpha : 0;
  double w = -threshold / (s.h + p.lambda);
  if (p.delta && std::abs(w) > p.delta) w = std::copysign(p.delta, w);
  return w;
}
inline float VectorWeight(Floating s, Parameters p, Bounds bounds = {-INFINITY, INFINITY}) {
  return std::clamp(float(WeightDouble(s, p)), bounds.lower, bounds.upper);
}
template <typename T>
double GainWeight(Floating s, T w, Parameters p) {
  T square = w * w;  // Float for constrained leaves, double in unconstrained max-delta path.
  return -(2.0 * s.g * w + (s.h + p.lambda) * square + 2.0 * p.alpha * std::abs(w));
}
inline double Gain(Floating s, Parameters p) {
  if (s.h <= 0) return 0;
  if (p.delta == 0) {
    double g = s.g > p.alpha ? s.g - p.alpha : s.g < -p.alpha ? s.g + p.alpha : 0;
    return g * g / (s.h + p.lambda);
  }
  return GainWeight(s, WeightDouble(s, p), p);
}
inline double SplitGain(std::vector<Pair> const& left, std::vector<Pair> const& right,
                        std::vector<Scale> const& scales, Task task, Parameters p,
                        std::vector<Bounds> const& bounds) {
  double gain = 0, lh = 0, rh = 0;
  for (std::size_t t = 0; t < left.size(); ++t) {
    auto l = Dequantize(left[t], scales[t]), r = Dequantize(right[t], scales[t]);
    lh += l.h;
    rh += r.h;
    if (!task.constrained) {
      gain += Gain(l, p);
      gain += Gain(r, p);
    }
  }
  lh /= left.size();
  rh /= left.size();
  if (!(lh > 0 && rh > 0 && lh >= p.minimum && rh >= p.minimum)) return -INFINITY;
  if (task.constrained) {
    for (std::size_t t = 0; t < left.size(); ++t) {
      auto l = Dequantize(left[t], scales[t]), r = Dequantize(right[t], scales[t]);
      auto b = bounds[task.parent_begin + t];
      float lw = VectorWeight(l, p, b), rw = VectorWeight(r, p, b);
      if ((task.monotone > 0 && !(lw <= rw)) || (task.monotone < 0 && !(lw >= rw))) {
        auto pooled = p;
        pooled.alpha = 2.f * p.alpha;
        pooled.lambda = 2.f * p.lambda;
        lw = rw = VectorWeight({l.g + r.g, l.h + r.h}, pooled, b);
      }
      gain += GainWeight(l, lw, p);
      gain += GainWeight(r, rw, p);
    }
  }
  return gain;
}
inline Result Evaluate(std::vector<Pair> const& hist, std::vector<Pair> const& parents,
                       std::vector<Scale> const& scales, std::vector<Bounds> const& bounds,
                       Task task, Parameters p, std::vector<std::uint32_t>* sorted,
                       std::vector<Pair>* children) {
  auto n = (task.end - task.begin) / task.targets, nt = task.targets;
  sorted->resize(n);
  std::iota(sorted->begin(), sorted->end(), 0);
  if (task.kind == 2) {
    std::vector<double> scores(n, 0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t t = 0; t < nt; ++t) {
        auto pw = VectorWeight(Dequantize(parents[task.parent_begin + t], scales[t]), p);
        auto cw = VectorWeight(Dequantize(hist[task.begin + i * nt + t], scales[t]), p);
        scores[i] += cw * pw;
      }
    std::stable_sort(sorted->begin(), sorted->end(),
                     [&](auto a, auto b) { return scores[a] < scores[b]; });
  }
  std::vector<Pair> total(nt);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t t = 0; t < nt; ++t) total[t] = Add(total[t], hist[task.begin + i * nt + t]);
  Result best;
  children->clear();
  auto limit = task.kind == 2 ? (std::min(n, task.limit) > 0 ? std::min(n, task.limit) - 1 : 0) : n;
  // Two separate phases, including one-hot: all missing-right candidates before
  // any missing-left candidates. Never interleave directions per category.
  for (unsigned phase = 0; phase < 2; ++phase) {
    if (task.kind == 0 &&
        ((phase == 0 && task.one_pass == 2) || (phase == 1 && task.one_pass == 1)))
      continue;
    bool missing = task.kind == 2 ? phase == 0 : phase == 1;
    std::vector<Pair> scan(nt);
    for (std::uint32_t base = 0; base < limit; base += 32) {
      double chunk_gain = -INFINITY;
      std::uint32_t chunk_bin = 0;
      std::vector<Pair> chunk_left, chunk_right;
      for (std::uint32_t i = base; i < std::min(base + 32, limit); ++i) {
        auto bin = task.kind == 1 ? i : phase == 0 ? i : n - 1 - i;
        auto read = task.kind == 2 ? (*sorted)[bin] : bin;
        std::vector<Pair> left(nt), right(nt);
        for (std::size_t t = 0; t < nt; ++t) {
          auto parent = parents[task.parent_begin + t];
          auto value = hist[task.begin + read * nt + t];
          scan[t] = Add(scan[t], value);
          Pair child = task.kind == 1 ? (missing ? value : Sub(total[t], value)) : scan[t];
          if (missing) {
            right[t] = child;
            left[t] = Sub(parent, child);
          } else {
            left[t] = child;
            right[t] = Sub(parent, child);
          }
        }
        auto gain = SplitGain(left, right, scales, task, p, bounds);
        if (gain > chunk_gain) {
          chunk_gain = gain;
          chunk_bin = bin;
          chunk_left = left;
          chunk_right = right;
        }
      }
      // Crucial CUDA boundary: double argmax, then implicit FP32 conversion
      // into MultiSplitCandidate::Update, whose comparison is strict.
      if (!std::isinf(chunk_gain) && float(chunk_gain) > best.gain) {
        best = {float(chunk_gain), std::int32_t(chunk_bin), std::uint32_t(missing),
                task.kind == 2 ? chunk_bin + std::uint32_t(missing) : 0};
        *children = chunk_left;
        children->insert(children->end(), chunk_right.begin(), chunk_right.end());
      }
    }
  }
  return best;
}
}  // namespace cuda_multi_split_reference
