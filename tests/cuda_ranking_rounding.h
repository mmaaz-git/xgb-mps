// Scalar interpretation of common/deterministic.cuh, with an explicit exponent
// zero for nonfinite frexp inputs (NVIDIA leaves that exponent unspecified).
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
namespace cuda_reference {
template <typename T>
T RankingRounding(T maximum, std::uint32_t rows) {
  T delta = maximum / (T{1} - (T{2} * static_cast<T>(rows)) * std::numeric_limits<T>::epsilon());
  int exponent = 0;
  if (std::isfinite(delta)) std::frexp(delta, &exponent);
  return std::ldexp(T{1}, exponent);
}
template <typename T>
T RankingTruncate(T rounding, T value) {
  // Explicit stores inhibit reassociation even in a caller built for speed.
  volatile T sum = rounding + value;
  return sum - rounding;
}
}  // namespace cuda_reference
