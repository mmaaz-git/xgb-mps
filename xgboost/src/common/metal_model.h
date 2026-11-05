/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include "xgboost/span.h"

namespace xgboost::common::metal {
enum class ModelBufferKind : std::uint32_t { kWeights, kCategories };
struct ModelCopy {
  void const* source;
  std::size_t destination, count;  // Offsets and counts in 32-bit words.
};
class ModelSnapshot {
 public:
  virtual ~ModelSnapshot() = default;
  virtual void const* Data() const = 0;
};
// GPU snapshot before model growth invalidates an aliased input. Guarded host
// sources can be shared; incompatible host allocations retain copied fallback.
std::unique_ptr<ModelSnapshot> SnapshotModelBuffer(void const* source, std::size_t words);
// Bit-preserving GPU copies. Destination ranges must be disjoint. Aliased
// sources are snapshotted before any destination writes; holes are preserved.
void CopyModelBuffer(void* output, std::size_t words, Span<ModelCopy const> copies,
                     ModelBufferKind kind);
// CUDA multi_target_tree_model.cu: one FP32 multiply per new child weight.
void ScaleModelWeights(Span<float> weights, float eta);
}  // namespace xgboost::common::metal
