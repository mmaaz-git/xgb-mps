/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <memory>

#include "metal_adapter.h"
#include "xgboost/string_view.h"

namespace xgboost::common::metal {
// Owned GPU snapshot of a dense strided Metal tensor. Import never dereferences
// its opaque MTLBuffer handle as host data. The producer must complete its work
// before ImportTensor; the snapshot completes before this call returns.
class Tensor {
 public:
  virtual ~Tensor() = default;
  virtual bst_idx_t NumRows() const = 0;
  virtual bst_feature_t NumColumns() const = 0;
};
std::shared_ptr<Tensor> ImportTensor(StringView descriptor);
std::unique_ptr<AdapterInput> PrepareAdapter(Tensor const& input, float missing, bst_idx_t begin,
                                             bst_idx_t count, enc::MappingView mapping = {});
std::unique_ptr<InputBatch> PrepareInput(Tensor const& input, float missing, std::size_t begin,
                                         std::size_t rows, std::uint32_t features,
                                         enc::MappingView mapping);
}  // namespace xgboost::common::metal
