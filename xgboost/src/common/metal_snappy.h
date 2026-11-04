/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "xgboost/span.h"

namespace xgboost::common::metal {
// Raw Snappy blocks, not the framed stream format. Offsets include both endpoints;
// decoded_sizes describes the allocation contract, independently of block headers.
struct SnappyBlocks {
  std::vector<std::uint8_t> data;
  std::vector<std::size_t> offsets{0};
  std::vector<std::uint32_t> decoded_sizes;
};
// Host exchange boundaries for the GPU codec. Cache integration uses the same
// kernels directly on Metal buffers, without these diagnostic byte readbacks.
SnappyBlocks CompressSnappy(Span<std::uint8_t const> input, std::size_t chunk_bytes = 1 << 21);
std::vector<std::uint8_t> DecompressSnappy(SnappyBlocks const& input);
}  // namespace xgboost::common::metal
