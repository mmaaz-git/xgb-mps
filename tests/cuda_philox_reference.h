// Independent host oracle for CUDA RandomWeight's modern CCCL branch. This is
// not an NVIDIA device execution comparison. Host 128-bit products differ from
// the shader's manual wide multiplication; the engine's published check value
// anchors its counter/key/output ordering.
// https://github.com/NVIDIA/cccl/blob/3d0d040ac0dc5907daf2a0731edb378e5562099d/libcudacxx/include/cuda/std/__random/philox_engine.h
#pragma once
#include <array>
#include <cstdint>

namespace cuda_reference {
inline std::uint64_t Philox(std::uint32_t seed, std::uint64_t position) {
  std::array<std::uint64_t, 4> words{position / 4, 0, 0, 0};
  std::uint64_t k0 = seed, k1 = 0;
  for (unsigned round = 0; round != 10; ++round) {
    __uint128_t p0 = __uint128_t{words[2]} * 0xca5a826395121157ULL;
    __uint128_t p1 = __uint128_t{words[0]} * 0xd2e7470ee14c6c93ULL;
    words = {std::uint64_t(p0 >> 64) ^ words[1] ^ k0, std::uint64_t(p0),
             std::uint64_t(p1 >> 64) ^ words[3] ^ k1, std::uint64_t(p1)};
    k0 += 0x9e3779b97f4a7c15ULL;
    k1 += 0xbb67ae8584caa73bULL;
  }
  return words[position % 4];
}
inline float Weight(std::uint32_t seed, std::uint64_t row) {
  // CCCL generate_canonical<float> with a full 64-bit engine: one engine call,
  // conversion to float, then division. Retain its possible rounded endpoint 1.
  return static_cast<float>(Philox(seed, row)) / 0x1p64f;
}
inline bool Selected(std::uint32_t seed, std::uint64_t row, float probability) {
  return probability >= 1.f || (probability > 0.f && Weight(seed, row) <= probability);
}
}  // namespace cuda_reference
