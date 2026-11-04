/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstdint>
#include <memory>

#include "xgboost/base.h"
#include "xgboost/linalg.h"
#include "xgboost/string_view.h"

namespace xgboost::common::metal {
// Immutable original per-target bounds/scales, without allocating integer rows.
// Value-gradient leaf fitting retains these scales BEFORE float sampling replay.
class GradientQuantizer {
 public:
  virtual ~GradientQuantizer() = default;
  virtual std::size_t Rows() const = 0;
  virtual std::size_t Targets() const = 0;
};
std::shared_ptr<GradientQuantizer> MakeGradientQuantizer(
    linalg::MatrixView<GradientPair const> original);

// Per-tree, GPU-owned integer gradients and their original per-target scales.
// Histogram builders retain this state explicitly. Floating-point replay is for
// row-position bookkeeping/value gradients, never an input to re-quantization.
class QuantizedGradients {
 public:
  virtual ~QuantizedGradients() = default;
  virtual std::size_t Rows() const = 0;
  virtual std::size_t Targets() const = 0;
  // Optional exact root statistics for split evaluation. Never reconstruct these
  // integers from the rounded floating-point view.
  virtual void Sum(linalg::VectorView<GradientPairPrecise> output,
                   Span<GradientPairInt64> integers = {}) const = 0;
  // CUDA's <cuda/std/random> branch (Philox4x64), with one draw per row.
  // Mutates the integer gradients exactly once; replay can be called repeatedly.
  virtual void SampleUniform(float probability, std::uint32_t seed) = 0;
  // CUDA gradient-based sampling on the original integers. Optional diagnostics
  // expose completed norms/threshold for source-derived local validation.
  virtual void SampleGradient(float probability, std::uint32_t seed,
                              Span<float> norms = {}, float* threshold = nullptr) = 0;
  virtual void ApplySampling(linalg::MatrixView<GradientPair> values) const = 0;
};
std::shared_ptr<QuantizedGradients> MakeQuantizedGradients(
    linalg::MatrixView<GradientPair const> gradients);

// Synchronous bit-preserving layout copies and Hessian extraction. Inputs can
// broadcast; output elements must be disjoint. Aliases use snapshot semantics.
// Guarded shared host views are used when eligible, with copied staging otherwise.
void CopyGradients(linalg::MatrixView<GradientPair const> input,
                   linalg::MatrixView<GradientPair> output);
void ExtractHessians(linalg::MatrixView<GradientPair const> input,
                     linalg::MatrixView<float> output);
// Native tensor descriptors have explicit device_type=8 tags, not host pointers.
// Preserve CUDA CustomGradHessOp casts and packing without clamping/validation
// of the numerical values. Shape/device validation precedes output publication.
void CopyNativeGradients(StringView grad, StringView hess, bst_idx_t rows,
                         linalg::Matrix<GradientPair>* output);
}  // namespace xgboost::common::metal
