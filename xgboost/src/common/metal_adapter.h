/** Copyright 2026, XGBoost Contributors */
#pragma once
#include "metal_input.h"

namespace xgboost::common::metal {
// One CUDA-sized flattened adapter window. Compact entries and row pointers
// stay on the GPU between decoding, sketching and bin assignment.
class AdapterInput {
 public:
  virtual ~AdapterInput() = default;
  virtual bst_idx_t Rows() const = 0;
  virtual bst_idx_t BaseRow() const = 0;
  virtual bst_idx_t NonMissing() const = 0;
  virtual bst_idx_t RowStride() const = 0;
  virtual bst_idx_t SketchRows() const = 0;
  virtual void WriteIndex(Span<FeatureType const> types, GHistIndexMatrix* output,
                          bst_idx_t row_base, bst_idx_t entry_offset) const = 0;
  // Diagnostics only; production does not read raw feature values back.
  virtual void Readback(SparsePage* output) const = 0;
};
std::unique_ptr<AdapterInput> PrepareAdapter(data::ArrayAdapter const& adapter, float missing,
                                             bst_idx_t begin, bst_idx_t count,
                                             enc::MappingView mapping = {});
std::unique_ptr<AdapterInput> PrepareAdapter(data::ColumnarAdapter const& adapter, float missing,
                                             bst_idx_t begin, bst_idx_t count,
                                             enc::MappingView mapping = {});
}  // namespace xgboost::common::metal
