// Original column-sampling test matrix, shared without changing its ordering.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace rng_contracts {
inline std::vector<std::uint32_t> ColumnFeatures(std::uint32_t n, int mode) {
  std::vector<std::uint32_t> features(n);
  for (std::uint32_t i = 0; i < n; ++i) features[i] = i * 3 + 1;
  std::reverse(features.begin(), features.end());
  if (mode == 0 && n > 2) {
    features[0] = 0xffffffffu;
    features[1] = 0x80000001u;
  }
  return features;
}
template <typename Fn> void ColumnCases(Fn run) {
  for (std::uint32_t n : {1u, 2u, 3u, 31u, 32u, 33u, 255u, 256u, 257u, 623u, 624u, 625u,
                          2047u, 2048u, 2049u, 4095u, 4096u, 4097u, 65537u}) {
    for (int mode = 0; mode < 7; ++mode) {
      run(n, 1, 0, mode);
      run(n, std::max(1u, n/3), 7, mode);
      run(n, n, 0xffffffffu, mode);
    }
  }
  for (std::uint32_t seed = 1; seed <= 48; ++seed) {
    run(1031, 517, seed, 0);
    run(1031, 517, seed, 2);
  }
  for (std::uint32_t seed : {2147483646u, 2147483647u, 2147483648u, 0xfffffffeu})
    for (std::uint32_t n : {1u, 255u, 256u, 257u, 2049u}) run(n, std::max(1u, n/2), seed, 0);
  for (int mode : {0, 2, 3, 6}) run(1000003, 500003, 9852, mode);
}
template <typename Fn> void ColumnHierarchies(Fn run) {
  for (std::uint32_t n : {3u, 257u, 4097u})
    for (bool weighted : {false, true})
      for (auto rates : {std::array<float, 3>{1, 1, 1}, {0.7f, 1, 1}, {1, 0.7f, 1},
                         {1, 1, 0.7f}, {0.9f, 0.7f, 0.5f}})
        run(n, rates[0], rates[1], rates[2], weighted);
}
}  // namespace rng_contracts
