/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <memory>

#include "metal_adapter.h"
#include "xgboost/string_view.h"

namespace xgboost::common::metal {
// Owned GPU snapshot of a dense tensor or heterogeneous columnar table. Import never dereferences
// its opaque MTLBuffer handle as host data. The producer must complete its work
// before ImportTensor; the snapshot completes before this call returns.
class Tensor {
 public:
  virtual ~Tensor() = default;
  virtual bst_idx_t NumRows() const = 0;
  virtual bst_feature_t NumColumns() const = 0;
  virtual enc::HostColumnsView Cats() const { return {}; }
  virtual enc::HostColumnsView RefCats() const { return {}; }
  bool HasCategorical() const { return Cats().n_total_cats != 0; }
  bool HasRefCategorical() const { return RefCats().n_total_cats != 0; }
};
std::shared_ptr<Tensor> ImportTensor(StringView descriptor);
// CUDA CopyToSparsePage: GPU missing filtering, counts, prefix and entry packing.
// Entries retain Metal storage; row offsets publish the host view used by CUDA's
// sizing/window consumers. Explicit CPU access lazily reads back entry payloads.
void CopyTensorToSparsePage(Tensor const& input, float missing, SparsePage* output);
std::unique_ptr<AdapterInput> PrepareAdapter(Tensor const& input, float missing, bst_idx_t begin,
                                             bst_idx_t count, enc::MappingView mapping = {});
std::unique_ptr<InputBatch> PrepareInput(Tensor const& input, float missing, std::size_t begin,
                                         std::size_t rows, std::uint32_t features,
                                         enc::MappingView mapping);
}  // namespace xgboost::common::metal
