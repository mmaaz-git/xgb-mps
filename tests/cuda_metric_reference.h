// CUDA-source FP32 row expressions, with explicit host binary64 libm rounding.
// This is a scalar formula oracle, not NVIDIA/libdevice execution.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cuda_metric {
inline float R(float value) {
  volatile float result = value;
  return result;
}
inline float Add(float a, float b) { return R(a + b); }
inline float Sub(float a, float b) { return R(a - b); }
inline float Mul(float a, float b) { return R(a * b); }
inline float Div(float a, float b) { return R(a / b); }
inline float Log(float x) { return static_cast<float>(std::log(double(x))); }
inline float Exp(float x) { return static_cast<float>(std::exp(double(x))); }
inline float Sqrt(float x) { return static_cast<float>(std::sqrt(double(x))); }
inline float Gamma(float x) {
  if (std::isinf(x)) return INFINITY;
  return static_cast<float>(std::lgamma(double(x)));
}
inline float Loss(unsigned kind, float y, float p, float parameter) {
  switch (kind) {
    case 0: {
      auto d = Sub(y, p);
      return Mul(d, d);
    }
    case 1: {
      auto d = Sub(float(std::log1p(double(y))), float(std::log1p(double(p))));
      return Mul(d, d);
    }
    case 2:
      return std::abs(Sub(y, p));
    case 3:
      return std::abs(Div(Sub(y, p), y));
    case 4: {
      auto xlogy = [](float x, float v) { return x == 0 ? 0.f : Mul(x, Log(std::max(v, 1e-16f))); };
      return Add(xlogy(-y, p), xlogy(-Sub(1, y), Sub(1, p)));
    }
    case 5:
      return p > parameter ? Sub(1, y) : y;
    case 6:
      if (p < 1e-16f) p = 1e-16f;
      return Sub(Add(Gamma(Add(y, 1)), p), Mul(Log(p), y));
    case 7:
      p = Add(p, 1e-6f);
      y = Add(y, 1e-6f);
      return Sub(Add(Log(Div(p, y)), Div(y, p)), 1);
    case 8: {
      p = std::max(p, 1e-6f);
      float theta = float(-1. / double(p)), b = -Log(-theta);
      return -Add(Div(Sub(Mul(y, theta), b), 1), 0);
    }
    case 9: {
      auto a = Div(Mul(y, Exp(Mul(Sub(1, parameter), Log(p)))), Sub(1, parameter));
      auto b = Div(Exp(Mul(Sub(2, parameter), Log(p))), Sub(2, parameter));
      return Add(-a, b);
    }
    case 10: {
      auto d = Div(Sub(y, p), parameter);
      return Mul(Mul(parameter, parameter), Sub(Sqrt(Add(1, Mul(d, d))), 1));
    }
  }
  return NAN;
}
inline float Quantile(bool expectile, float p, float y, float a, float w) {
  if (expectile) {
    auto d = Sub(p, y), scale = d >= 0 ? Sub(1, a) : a;
    return Mul(Mul(Mul(scale, d), d), w);
  }
  auto d = Sub(y, p), sign = float(d >= 0);
  return Mul(Sub(Mul(Mul(a, sign), d), Mul(Mul(Sub(1, a), Sub(1, sign)), d)), w);
}
template <typename T, typename U>
T Cast(U x) {
  static_assert(sizeof(T) == sizeof(U));
  T out;
  std::memcpy(&out, &x, sizeof(T));
  return out;
}
inline std::uint32_t Key(float x) {
  auto bits = Cast<std::uint32_t>(x);
  return bits >> 31 ? ~bits : bits ^ 0x80000000u;
}
inline std::uint64_t Ulps(float a, float b) {
  if (a == b || (std::isnan(a) && std::isnan(b))) return 0;
  if (!std::isfinite(a) || !std::isfinite(b)) return std::numeric_limits<std::uint64_t>::max();
  auto x = Key(a), y = Key(b);
  return std::uint64_t{std::max(x, y)} - std::min(x, y);
}
// Neumaier compensated reference; finite binary64 sums of FP32 row terms.
struct Sum {
  double value{0}, correction{0}, magnitude{0};
  void Add(double x) {
    magnitude += std::abs(x);
    auto total = value + x;
    if (std::isfinite(total) && std::isfinite(value) && std::isfinite(x))
      correction += std::abs(value) >= std::abs(x) ? (value - total) + x : (x - total) + value;
    value = total;
  }
  double Result() const { return value + correction; }
};
}  // namespace cuda_metric
