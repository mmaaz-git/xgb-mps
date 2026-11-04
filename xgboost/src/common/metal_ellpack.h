/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "hist_util.h"
#include "xgboost/context.h"
#include "xgboost/data.h"

namespace xgboost {
class GHistIndexMatrix;
namespace common {
class AlignedFileWriteStream;
class AlignedResourceReadStream;
}  // namespace common
namespace common::metal {
class AdapterInput;
struct EllpackInfo {
  std::uint32_t features, row_stride, symbols, bits;
  bool dense;
};
// CalcNumSymbols uses a GPU maximum over feature bin counts, as in CUDA.
EllpackInfo MakeEllpackInfo(HistogramCuts const& cuts, bst_idx_t row_stride, bool dense);
std::size_t EllpackBytes(bst_idx_t rows, EllpackInfo info);
std::uint64_t CacheWorkingSetBytes();

class CompressedPage {
 public:
  virtual ~CompressedPage() = default;
  virtual EllpackInfo Info() const = 0;
  virtual bst_idx_t Rows() const = 0;
  virtual bst_idx_t BaseRow() const = 0;
  virtual std::size_t SizeBytes() const = 0;
  virtual std::size_t HostBytes() const = 0;
  virtual std::size_t DeviceBytes() const = 0;
  virtual void Pack(GHistIndexMatrix const& input) = 0;
  // Append raw CSR rows directly into compressed symbols. Zero window_rows selects
  // a bounded device-budget window; a nonzero value may reduce it for diagnostics.
  virtual void PushCSR(SparsePage const& input, HistogramCuts const& cuts,
                       Span<FeatureType const> types, bst_idx_t window_rows = 0) = 0;
  // Ordered raw adapter windows write bins directly into the compressed buffer.
  virtual void BeginAdapter(HistogramCuts const& cuts, Span<FeatureType const> types) = 0;
  virtual void PushAdapter(AdapterInput const& input) = 0;
  virtual void EndAdapter() = 0;
  virtual void Copy(CompressedPage const& input, bst_idx_t row_offset) = 0;
  virtual void Commit(double host_ratio) = 0;
  // Count actual compressed symbols on GPU; only the scalar result returns to host.
  virtual bst_idx_t NonMissing() const = 0;
  // Acquire a GPU view without unpacking rows, bins, hits or host columns.
  virtual void View(HistogramCuts const& cuts, bst_bin_t max_bin, bool prefetch_copy,
                    GHistIndexMatrix* output) const = 0;
  virtual void Restore(Context const* ctx, HistogramCuts const& cuts, bst_bin_t max_bin,
                       double sparse_thresh, bool prefetch_copy,
                       GHistIndexMatrix* output) const = 0;
  // Diagnostic only. Production restoration reads compressed bytes on GPU.
  virtual std::vector<std::uint8_t> Readback() const = 0;
};
// CUDA EllpackPageRawFormat field order and alignment, with owned Metal buffers.
std::size_t WriteCompressedPage(GHistIndexMatrix const& page, AlignedFileWriteStream* output);
bool ReadCompressedPage(AlignedResourceReadStream* input, HistogramCuts const& cuts,
                        bst_bin_t max_bin, bool prefetch_copy, GHistIndexMatrix* output);
// Actual compressed non-missing symbols; only the final scalar returns to host.
bst_idx_t CompressedNonMissing(GHistIndexMatrix const& page);
// Explicit CPU access boundary. Idempotent; retains the compressed GPU view.
void MaterializeHost(Context const* ctx, double sparse_thresh, GHistIndexMatrix* page);
std::shared_ptr<CompressedPage> MakeCompressedPage(EllpackInfo info, bst_idx_t rows,
                                                   bst_idx_t base_row, std::size_t bytes = 0);

struct EllpackCachePlan {
  double host_ratio;
  std::int64_t min_page_bytes;
  std::vector<std::size_t> mapping, bytes;
  std::vector<bst_idx_t> rows;
};
// CUDA CalcCacheMapping/DftPageSizeHostRatio with the Metal working-set budget.
// Explicit budget argument permits deterministic validation without allocations.
EllpackCachePlan PlanEllpackCache(EllpackInfo info, Span<bst_idx_t const> batch_rows,
                                  bool validation, double host_ratio, std::int64_t min_page_bytes,
                                  std::uint64_t device_bytes);
}  // namespace common::metal
}  // namespace xgboost
