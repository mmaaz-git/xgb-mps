// Host source oracle for common/random.cu. This is not NVIDIA execution.
// Thrust shuffle algorithm: NVIDIA/cccl commit
// 3d0d040ac0dc5907daf2a0731edb378e5562099d,
// thrust/system/detail/generic/shuffle.inl and thrust/detail/random_bijection.h.
// Feistel mapping derived from Thrust, Copyright (c) 2008-2025 NVIDIA Corporation,
// licensed under Apache-2.0, as is XGBoost.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "cuda_philox_reference.h"

namespace cuda_reference {
inline std::array<std::uint32_t, 24> ShuffleKeys(std::uint32_t seed) {
  // Independent explicit LCG, instead of the production std::minstd_rand.
  std::uint64_t state = seed % 2147483647u;
  if (!state) state = 1;
  std::array<std::uint32_t, 24> keys;
  for (auto& key : keys) {
    state = (state * 48271) % 2147483647u;
    // Thrust uniform_int<uint32> uses uniform_real<double>(0, UINT32_MAX+1).
    key = static_cast<std::uint32_t>((double(state - 1) / 2147483646.0) * 0x1p32);
  }
  return keys;
}
inline std::vector<std::uint32_t> Shuffle(std::uint32_t rows, std::uint32_t seed) {
  unsigned bits = 8;
  while ((std::uint64_t{1} << bits) < rows) ++bits;
  auto keys = ShuffleKeys(seed);
  unsigned r_bits = (bits + 1) / 2, l_bits = bits / 2;
  std::uint32_t r_mask = (1u << r_bits) - 1, l_mask = (1u << l_bits) - 1;
  std::vector<std::uint32_t> indices;
  indices.reserve(rows);
  for (std::uint64_t position = 0; position < (std::uint64_t{1} << bits); ++position) {
    std::uint32_t left = position >> r_bits, right = position & r_mask;
    for (auto key : keys) {
      std::uint64_t product = 0xD2B74407B1CE6E93ULL * left;
      auto next_left = (std::uint32_t(product >> 32) ^ key ^ right) & l_mask;
      auto next_right =
          ((std::uint32_t(product) << (r_bits - l_bits)) | (right >> l_bits)) & r_mask;
      left = next_left;
      right = next_right;
    }
    std::uint32_t permuted = (left << r_bits) | right;
    if (permuted < rows) indices.push_back(permuted);
  }
  return indices;
}
inline std::vector<std::uint32_t> SampleFeatures(std::vector<std::uint32_t> const& features,
                                                 std::vector<float> const& weights,
                                                 std::uint32_t seed, std::size_t count) {
  std::vector<std::uint32_t> order(features.size());
  if (weights.empty()) {
    order = Shuffle(features.size(), seed);
  } else {
    std::iota(order.begin(), order.end(), 0);
    std::vector<float> keys(features.size());
    for (std::size_t i = 0; i < features.size(); ++i)
      keys[i] = std::log(Weight(seed, i)) / std::max(weights[features[i]], 1e-6f);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return keys[a] > keys[b]; });
  }
  std::vector<std::uint32_t> result(count);
  for (std::size_t i = 0; i < count; ++i) result[i] = features[order[i]];
  std::sort(result.begin(), result.end());
  return result;
}
}  // namespace cuda_reference
