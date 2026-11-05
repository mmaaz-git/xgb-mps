/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost {
class GHistIndexMatrix;
namespace common::metal {
class LeafSums;
class QuantizedGradients;
// Immutable GPU copy owned by the corresponding gradient-index page. Destruction
// and explicit invalidation follow the source page, including external memory.
class QuantizedPage {
 public:
  virtual ~QuantizedPage() = default;
};
struct PartitionSplit {
  std::uint64_t begin, end;
  std::uint32_t feature;
  std::int32_t split_bin;
  std::uint32_t default_left, categorical, cat_begin, cat_size;
  std::uint32_t block_begin, block_end;
};
struct LeafRange {
  std::uint64_t begin, end;
  std::int32_t node;
  std::uint32_t padding{0};
};
struct LeafSumSegment {
  std::uint64_t begin, end;
  std::uint32_t output_leaf, padding{0};
};
class DeviceRows {
 public:
  virtual ~DeviceRows() = default;
  virtual void Reset(std::size_t rows, bst_idx_t base) = 0;
  virtual std::vector<std::uint64_t> Partition(GHistIndexMatrix const& page,
                                               std::vector<PartitionSplit> splits,
                                               Span<std::uint32_t const> categories) = 0;
  virtual void CopyToHost(Span<bst_idx_t> output) const = 0;
  virtual void LeafPositions(std::vector<LeafRange> leaves,
                             linalg::MatrixView<GradientPair const> gradients,
                             Span<bst_node_t> output,
                             QuantizedGradients const* sampling = nullptr) = 0;
  virtual void AddLeafGradients(Span<LeafSumSegment const> leaves,
                                linalg::MatrixView<GradientPair const> gradients,
                                LeafSums* output) = 0;
};
std::unique_ptr<DeviceRows> MakeDeviceRows();
}  // namespace common::metal
}  // namespace xgboost
