// Host oracle for src/tree/gpu_hist/{sampler.cu,quantiser.cuh}.
// CUDA source semantics, not measured CUDA results. Norm reduction models the
// CCCL SM60-SM90 fixed-segment policy at commit
// 3d0d040ac0dc5907daf2a0731edb378e5562099d, with the documented FMA model.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_philox_reference.h"
namespace cuda_reference {
struct Pair {
  float g, h;
};
struct IntegerPair {
  std::int64_t g, h;
};
struct Scale {
  int g, h;
};
inline int Exponent(double positive, double negative, std::size_t rows) {
  int exponent = 0;
  std::frexp(std::max(positive, negative) /
                 (1. - 2. * double(rows) * std::numeric_limits<double>::epsilon()),
             &exponent);
  return 62 - exponent;
}
inline float Add(float a, float b) {
  volatile float result = a + b;
  return result;
}
inline float Multiply(float a, float b) {
  volatile float result = a * b;
  return result;
}
inline float Divide(float a, float b) {
  volatile float result = a / b;
  return result;
}
inline float Norm(std::vector<float> const& terms) {
  auto targets = terms.size();
  float total = terms[0];
  if (targets <= 16) {
    for (std::size_t i = 1; i < targets; ++i) total = Add(total, terms[i]);
  } else {
    std::size_t width = targets <= 512 ? 32 : 256;
    std::vector<float> lanes(width, 0.f);
    for (std::size_t i = 0; i < targets; ++i) lanes[i % width] = Add(lanes[i % width], terms[i]);
    for (std::size_t offset = 1; offset < 32; offset *= 2) {
      auto previous = lanes;
      for (std::size_t i = 0; i < width; ++i)
        if (i % 32 + offset < 32) lanes[i] = Add(previous[i], previous[i + offset]);
    }
    total = lanes[0];
    for (std::size_t warp = 1; warp < width / 32; ++warp) total = Add(total, lanes[warp * 32]);
  }
  return std::sqrt(total);
}
inline float Probability(float norm, float threshold) {
  if (std::isinf(threshold)) return 0.f;
  if (std::abs(threshold) < 1e-6f) threshold = std::copysign(1e-6f, threshold);
  return Divide(norm, threshold);
}
struct GradientSampling {
  std::size_t rows, targets;
  float fraction, threshold{std::numeric_limits<float>::infinity()};
  std::uint32_t seed;
  bool invalid_rescale{false};
  std::vector<Scale> scales;
  std::vector<IntegerPair> original, sampled;
  std::vector<float> norms;
  GradientSampling(std::vector<Pair> const& source, std::size_t n, std::size_t t, float p,
                   std::uint32_t s)
      : rows{n}, targets{t}, fraction{p}, seed{s}, scales(t), original(n * t), norms(n) {
    for (std::size_t c = 0; c < t; ++c) {
      double positive[2]{}, negative[2]{};
      for (std::size_t r = 0; r < n; ++r) {
        auto pair = source[r * t + c];
        float values[]{pair.g, pair.h};
        for (int k = 0; k < 2; ++k)
          if (values[k] >= 0)
            positive[k] += values[k];
          else
            negative[k] -= values[k];
      }
      auto scale = scales[c] = {Exponent(positive[0], negative[0], n),
                                Exponent(positive[1], negative[1], n)};
      for (std::size_t r = 0; r < n; ++r) {
        auto pair = source[r * t + c];
        auto& fixed = original[r * t + c];
        fixed = {static_cast<std::int64_t>(std::ldexp(double(pair.g), scale.g)),
                 static_cast<std::int64_t>(std::ldexp(double(pair.h), scale.h))};
        if (pair.h > 0 && !fixed.h) fixed.h = 1;
      }
    }
    sampled = original;
    if (p >= 1) return;
    std::vector<float> terms(t);
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t c = 0; c < t; ++c) {
        auto fixed = original[r * t + c];
        auto scale = scales[c];
        double g = std::ldexp(double(fixed.g), -scale.g), h = std::ldexp(double(fixed.h), -scale.h);
        terms[c] = static_cast<float>(std::fma(double(.1f), h * h, g * g));
      }
      norms[r] = Norm(terms);
    }
    auto budget = static_cast<std::size_t>(float(n) * p);
    if (budget) {
      auto sorted = norms;
      std::sort(sorted.begin(), sorted.end());
      double bound = 0;
      for (float norm : sorted) bound += norm;
      if (!std::isfinite(bound)) throw std::runtime_error("Oracle needs finite FP32 norms");
      int exponent = Exponent(bound, 0., n);
      sorted.push_back(std::numeric_limits<float>::infinity());
      threshold = sorted[1];  // CUDA min_element's first-index fallback.
      std::int64_t cumulative = 0;
      for (std::size_t i = 0; i < n; ++i) {
        cumulative += static_cast<std::int64_t>(std::ldexp(double(sorted[i]), exponent));
        float prefix = static_cast<float>(std::ldexp(double(cumulative), -exponent));
        float denominator = float(budget) - float(n - i - 1);
        if (denominator <= 0) continue;
        float candidate = Divide(prefix, denominator);
        if (candidate > sorted[i] && candidate <= sorted[i + 1]) {
          threshold = candidate;
          break;
        }
      }
    }
    for (std::size_t r = 0; r < n; ++r) {
      float prob = Probability(norms[r], threshold);
      bool keep = Selected(seed, r, prob);
      float reciprocal = prob >= 1 ? 1.f : (keep ? Divide(1.f, prob) : 0.f);
      for (std::size_t c = 0; c < t; ++c) {
        auto fixed = original[r * t + c];
        auto scale = scales[c];
        auto& out = sampled[r * t + c];
        if (!keep) {
          out = {};
          continue;
        }
        double g = std::ldexp(double(fixed.g), -scale.g) * double(reciprocal);
        double h = std::ldexp(double(fixed.h), -scale.h) * double(reciprocal);
        double qg = std::ldexp(g, scale.g), qh = std::ldexp(h, scale.h);
        if (!(qg >= -0x1p63 && qg < 0x1p63 && qh >= -0x1p63 && qh < 0x1p63)) {
          invalid_rescale = true;
          out = {};
          continue;
        }
        out = {static_cast<std::int64_t>(qg), static_cast<std::int64_t>(qh)};
        if (h > 0 && !out.h) out.h = 1;
      }
    }
  }
  Pair Replay(std::size_t row, Pair pair) const {
    if (fraction >= 1) return pair;
    float prob = Probability(norms[row], threshold);
    if (!Selected(seed, row, prob)) return {};
    if (prob >= 1) return pair;
    float reciprocal = Divide(1.f, prob);
    return {Multiply(pair.g, reciprocal), Multiply(pair.h, reciprocal)};
  }
  std::pair<double, double> Sum(std::size_t target) const {
    std::uint64_t g = 0, h = 0;
    for (std::size_t r = 0; r < rows; ++r) {
      g += sampled[r * targets + target].g;
      h += sampled[r * targets + target].h;
    }
    std::int64_t signed_g, signed_h;
    std::memcpy(&signed_g, &g, 8);
    std::memcpy(&signed_h, &h, 8);
    return {std::ldexp(double(signed_g), -scales[target].g),
            std::ldexp(double(signed_h), -scales[target].h)};
  }
};
}  // namespace cuda_reference
