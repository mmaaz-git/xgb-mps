/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <memory>
#include <vector>

#include "dlpack/dlpack.h"
#include "xgboost/base.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// An owned GPU snapshot, independent of the learner's mutable prediction cache.
class Output {
 public:
  virtual ~Output() = default;
  virtual DLTensor View() const = 0;
  virtual std::size_t Size() const = 0;
  virtual void Readback(Span<float> output) const = 0;
  virtual std::shared_ptr<Output> Clone() const = 0;
};
std::shared_ptr<Output> MakeOutput(float const* values, Span<bst_ulong const> shape);
// Each export owns a distinct managed descriptor; deleters retain the allocation.
void* ExportOutput(std::shared_ptr<Output> output, bool versioned, bool copy);
// Optional CPython capsule destructor. Resolves the running interpreter's C API;
// native clients only need ExportOutput and the standard DLPack deleter.
void* OutputCapsuleDestructor();
}  // namespace xgboost::common::metal
