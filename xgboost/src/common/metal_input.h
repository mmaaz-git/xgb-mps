/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <memory>

#include "../encoder/ordinal.h"
#include "xgboost/data.h"

namespace xgboost {
class GHistIndexMatrix;
namespace data {
class DenseAdapter;
class ArrayAdapter;
class ColumnarAdapter;
class CSRArrayAdapter;
}  // namespace data
namespace common::metal {
// Completed GPU input chunk, consumed directly by prediction and SHAP.
class InputBatch {
 public:
  virtual ~InputBatch() = default;
  virtual std::size_t Rows() const = 0;
  virtual std::uint32_t Features() const = 0;
  // Explicit diagnostic readback; production consumers keep the data on GPU.
  virtual void Readback(Span<float> output) const = 0;
};
std::unique_ptr<InputBatch> PrepareInput(HostSparsePageView page, std::size_t begin,
                                         std::size_t rows, std::uint32_t features,
                                         enc::MappingView mapping);
std::unique_ptr<InputBatch> PrepareInput(GHistIndexMatrix const& page,
                                         Span<FeatureType const> types, std::size_t begin,
                                         std::size_t rows, std::uint32_t features,
                                         enc::MappingView mapping);
#define XGB_METAL_INPUT_ADAPTER(Type)                                                \
  std::unique_ptr<InputBatch> PrepareInput(data::Type const& adapter, float missing, \
                                           std::size_t begin, std::size_t rows,      \
                                           std::uint32_t features, enc::MappingView mapping);
XGB_METAL_INPUT_ADAPTER(DenseAdapter)
XGB_METAL_INPUT_ADAPTER(ArrayAdapter)
XGB_METAL_INPUT_ADAPTER(ColumnarAdapter)
XGB_METAL_INPUT_ADAPTER(CSRArrayAdapter)
#undef XGB_METAL_INPUT_ADAPTER
}  // namespace common::metal
}  // namespace xgboost
