// Shared inputs from the existing Metal RNG/pair tests. Integer standard engines
// and explicit FP32 operations keep the original corpus reproducible.
#pragma once
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace rng_contracts {
struct PhiloxQuery {
  std::uint64_t row;
  std::uint32_t seed, padding;
};
inline std::vector<PhiloxQuery> PhiloxQueries() {
  std::vector<PhiloxQuery> queries;
  for (std::uint32_t seed : {0u, 1u, 17u, 20111115u, 0x80000000u, 0xffffffffu}) {
    for (std::uint64_t row = 0; row < 10001; ++row) queries.push_back({row, seed, 0});
    for (unsigned bit : {32, 40, 62, 63})
      for (int offset = -5; offset <= 5; ++offset)
        queries.push_back({(std::uint64_t{1} << bit) + offset, seed, 0});
    queries.push_back({UINT64_MAX, seed, 0});
  }
  std::mt19937_64 rng{803};
  for (unsigned i = 0; i < 1000000; ++i) queries.push_back({rng(), std::uint32_t(rng()), 0});
  return queries;
}

// Packed as seed/range/start/padding to preserve the existing Metal ulong4 ABI.
inline std::vector<std::uint64_t> RankingQueries() {
  std::vector<std::uint32_t> seeds{0, 1, 2, 2147483646, 2147483647, 2147483648u,
                                  4294967294u, 4294967295u};
  std::mt19937 rng(78223);
  for (int i = 0; i < 32; ++i) seeds.push_back(rng());
  std::vector<std::uint32_t> ranges{1, 2, 3, 17, 127, 129, 257, 4097, 65537,
                                   2147483646u, 2147483647u, 2147483648u, 4294967295u};
  for (std::uint32_t bit = 2; bit < 32; ++bit) {
    ranges.push_back((1u << bit) - 1);
    ranges.push_back(1u << bit);
    ranges.push_back((1u << bit) + 1);
  }
  std::vector<std::uint64_t> positions{0, 1, 2, 3, 15, 255, 0xffffffffULL,
      0x100000000ULL, 0x10000000003ULL, 0x7fffffffffffffefULL};
  std::vector<std::uint64_t> inputs;
  for (auto seed : seeds)
    for (auto range : ranges)
      for (auto start : positions) inputs.insert(inputs.end(), {seed, range, start, 0});
  return inputs;
}

struct PairCase {
  std::vector<std::uint32_t> sizes;
  std::uint32_t count, seed;
  int mode;
  bool batched;
  std::vector<float> predictions, labels;  // Original labels have stride three.
};
template <typename Fn> void PairCases(Fn run) {
  std::mt19937 rng(881927);
  auto make = [&](std::vector<std::uint32_t> sizes, std::uint32_t count,
                  std::uint32_t seed, int mode, bool batched) {
    PairCase c{sizes, count, seed, mode, batched, {}, {}};
    std::size_t n = 0;
    for (auto size : sizes) n += size;
    c.predictions.resize(n);
    c.labels.assign(n * 3, -999.f);
    for (std::size_t i = 0; i < n; ++i) {
      c.predictions[i] = mode == 0 ? (i % 2 ? 0.f : -0.f) : float(int(rng() % 23) - 11);
      float y = float(rng() % 5);
      if (mode == 1) y = 0;
      if (mode == 2) {
        float edge[] = {-0.f, 0.f, std::numeric_limits<float>::denorm_min(),
                       -std::numeric_limits<float>::denorm_min(), 1.e-7f, -1.e-7f};
        y = edge[rng() % 6];
      }
      if (mode == 3) y = float(int(rng() % 1001) - 500) / 173.f;
      c.labels[3 * i] = y;
    }
    run(c);
  };
  for (std::uint32_t seed : {0u, 1u, 2147483647u, 2147483648u, 0xfffffffeu, 0xffffffffu})
    for (std::uint32_t count : {1u, 8u, 32u})
      for (int mode = 0; mode < 4; ++mode)
        make({0, 1, 0, 7, 31, 33, 0, 255, 256, 257, 4097, 0}, count, seed, mode, false);
  make({0, 0}, 8, 0, 0, false);
  make(std::vector<std::uint32_t>(512, 257), 32, 0xffffffffu, 2, true);
}
}  // namespace rng_contracts
