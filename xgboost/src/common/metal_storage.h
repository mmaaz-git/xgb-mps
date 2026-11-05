/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// Shared Metal allocation: GPU kernels own histogram arithmetic, while existing
// host tree orchestration can inspect completed results without a staging copy.
// Quantized builds retain integer sums; Data() is a materialized binary64 view.
// Callers must not mutate that view once the buffer contains quantized histograms.
// Resize preserves both views. Page-zero builds replace reused cache ranges.
class HistogramBuffer {
 public:
  virtual ~HistogramBuffer() = default;
  virtual void Resize(std::size_t size) = 0;
  virtual Span<GradientPairPrecise> Data() = 0;
  virtual Span<GradientPairPrecise const> Data() const = 0;
};
std::unique_ptr<HistogramBuffer> MakeHistogramBuffer();

struct HistogramSubtraction {
  std::uint64_t parent, sibling, output, bins;
};
void SubtractHistograms(HistogramBuffer* buffer, Span<HistogramSubtraction const> tasks);
// Reduce the authoritative integer cache, then refresh its binary64 view.
void AllreduceHistograms(Context const* ctx, HistogramBuffer* buffer, std::size_t offset,
                         std::size_t bins);

struct HistogramGather {
  std::uint64_t source;
  std::uint32_t output, bins, stride, padding;
};
struct HistogramInput {
  HistogramBuffer const* source;
  std::vector<HistogramGather> segments;
};
}  // namespace xgboost::common::metal
