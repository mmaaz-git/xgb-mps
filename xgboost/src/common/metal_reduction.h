/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>
#include <cstdint>

#include "xgboost/base.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
class HistogramBuffer;
// kFloat32 preserves FP32 per-term division and accumulation; parallel
// association can differ from CUB/CPU. Sample means use binary64 terms.
enum class MeanKind : std::uint32_t { kFloat32, kSample, kWeightedSample };
double Sum(Span<float const> values);
// CUDA numeric::Reduce uses plus<float>; its FP32 result is returned as double.
double SumFloat32(Span<float const> values);
void Mean(MeanKind kind, linalg::MatrixView<float const> values, Span<float const> weights,
          linalg::VectorView<float> output);
void SumGradients(linalg::MatrixView<GradientPair const> gradients,
                  linalg::VectorView<GradientPairPrecise> output);
void FitStump(linalg::MatrixView<GradientPair const> gradients, linalg::VectorView<float> output);
// Sum one feature's bins. Quantized caches sum integers first, then convert;
// their quantizer bounds require that each source row occurs at most once.
GradientPairPrecise SumHistogram(HistogramBuffer const* storage, std::size_t offset,
                                 std::size_t bins);
}  // namespace xgboost::common::metal
