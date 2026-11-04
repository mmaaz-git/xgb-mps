// Host interpretation of lambdarank_obj.cuh::MakePairsOp::WithSampling, modern
// CCCL branch. This does not run CUDA. The distribution is separately checked
// against pinned, unmodified NVIDIA headers by cuda_ranking_upstream.cc.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include "cuda_philox_reference.h"

namespace cuda_reference {
inline std::uint32_t RankingUniform(std::uint32_t seed, std::uint64_t* position,
                                    std::uint32_t range) {
  if (range == 1) return 0;
  std::uint64_t mask = 1;
  while (mask < std::uint64_t{range} - 1) mask = (mask << 1) | 1;
  std::uint64_t draw;
  do {
    draw = Philox(seed, (*position)++) & mask;
  } while (draw >= range);
  return static_cast<std::uint32_t>(draw);
}
using RankingPair = std::array<std::uint32_t, 2>;
inline std::vector<RankingPair> MeanPairs(std::vector<float> const& labels,
                                          std::vector<std::size_t> const& rank, std::uint32_t seed,
                                          std::uint32_t count) {
  std::vector<std::size_t> label_order(rank.size());
  std::iota(label_order.begin(), label_order.end(), 0);
  std::stable_sort(label_order.begin(), label_order.end(),
                   [&](auto a, auto b) { return labels[rank[a]] > labels[rank[b]]; });
  std::vector<RankingPair> output(rank.size() * count);
  // Linear bucket traversal is deliberately independent of the shader's two
  // binary searches. Each sample/repeat still starts a fresh Philox engine.
  for (std::size_t first = 0; first < rank.size();) {
    auto last = first + 1;
    while (last < rank.size() &&
           labels[rank[label_order[last]]] == labels[rank[label_order[first]]])
      ++last;
    auto outside = first + rank.size() - last;
    if (outside) {
      for (auto i = first; i < last; ++i) {
        for (std::uint32_t repeat = 0; repeat < count; ++repeat) {
          auto index = i * count + repeat;
          std::uint64_t position = index;
          std::size_t other = RankingUniform(seed, &position, outside);
          if (other >= first) other += last - first;
          output[index] = {static_cast<std::uint32_t>(label_order[i]),
                           static_cast<std::uint32_t>(label_order[other])};
        }
      }
    }
    first = last;
  }
  return output;
}
}  // namespace cuda_reference
