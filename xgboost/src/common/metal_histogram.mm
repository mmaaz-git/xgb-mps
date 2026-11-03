/** Copyright 2026, XGBoost Contributors */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../data/gradient_index.h"
#include "metal_histogram.h"
#include "metal_shader.h"
#include "xgboost/logging.h"

namespace xgboost::common::metal {
namespace {
std::atomic<std::uint64_t> dispatches{0};
std::atomic<std::uint64_t> processed_rows{0};

struct RowEntry {
  std::uint64_t row;
  std::uint32_t node;
  std::uint32_t padding{0};
};
static_assert(sizeof(RowEntry) == 16);
struct Params {
  std::uint64_t n_rows, base_rowid;
  std::uint32_t n_bins, n_features, bin_bytes, dense;
};
static_assert(sizeof(Params) == 32);
struct QuantizedPair {
  std::int64_t grad, hess;
};
static_assert(sizeof(QuantizedPair) == 16);

char const* ErrorText(NSError* error) {
  return error ? error.localizedDescription.UTF8String : "unknown Metal error";
}

class Runtime {
 public:
  id<MTLDevice> device;
  id<MTLComputePipelineState> pipeline;

  Runtime() {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      CHECK(device) << "device=mps requested, but no Metal device is available. "
                       "GPU access may be restricted by the execution sandbox.";
      MTLCompileOptions* options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      NSError* error = nil;
      auto source = [NSString stringWithUTF8String:kXGBoostMetalShader];
      id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
      CHECK(library) << "Metal shader compilation failed: " << ErrorText(error);
      id<MTLFunction> kernel = [library newFunctionWithName:@"xgb_histogram"];
      CHECK(kernel) << "Missing Metal histogram kernel.";
      pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
      CHECK(pipeline) << "Metal pipeline creation failed: " << ErrorText(error);
      LOG(INFO) << "Metal histogram backend: " << device.name.UTF8String;
    }
  }
};

Runtime& GetRuntime() {
  static Runtime runtime;
  return runtime;
}

id<MTLBuffer> Allocate(Runtime const& rt, std::size_t bytes) {
  auto length = std::max<std::size_t>(bytes, 16);
  CHECK_LE(length, rt.device.maxBufferLength) << "Metal buffer exceeds the device limit.";
  id<MTLBuffer> buffer = [rt.device newBufferWithLength:length
                                              options:MTLResourceStorageModeShared];
  CHECK(buffer) << "Metal buffer allocation failed for " << length << " bytes.";
  return buffer;
}

id<MTLBuffer> Upload(Runtime const& rt, void const* data, std::size_t bytes) {
  auto buffer = Allocate(rt, bytes);
  if (bytes != 0) std::memcpy(buffer.contents, data, bytes);
  return buffer;
}

struct Page {
  id<MTLBuffer> index, row_ptr, offsets;
};

class MetalHistogram final : public Histogram {
  Runtime& rt_;
  id<MTLCommandQueue> queue_;
  id<MTLBuffer> gradients_;
  std::vector<Page> pages_;
  double inverse_scale_[2]{1.0, 1.0};

  void UploadGradients(Span<GradientPair const> gradients) {
    // Use independent power-of-two scales. Bound every possible subset by
    // n_rows * max(abs(value)), with three signed bits of accumulation headroom.
    // CPU FP64 is intentional: Metal has no native FP64 arithmetic.
    double maxima[2]{0.0, 0.0};
    for (auto const& g : gradients) {
      CHECK(std::isfinite(g.GetGrad()) && std::isfinite(g.GetHess()))
          << "Metal histograms require finite gradients and Hessians.";
      maxima[0] = std::max(maxima[0], std::abs(static_cast<double>(g.GetGrad())));
      maxima[1] = std::max(maxima[1], std::abs(static_cast<double>(g.GetHess())));
    }
    double scale[2];
    for (int i = 0; i < 2; ++i) {
      double bound = maxima[i] * static_cast<double>(gradients.size());
      int exponent = bound > 0.0 ? 60 - static_cast<int>(std::ceil(std::log2(bound))) : 0;
      scale[i] = std::ldexp(1.0, exponent);
      inverse_scale_[i] = std::ldexp(1.0, -exponent);
    }
    gradients_ = Allocate(rt_, gradients.size() * sizeof(QuantizedPair));
    auto* out = static_cast<QuantizedPair*>(gradients_.contents);
    for (std::size_t i = 0; i < gradients.size(); ++i) {
      out[i] = {std::llround(gradients[i].GetGrad() * scale[0]),
                std::llround(gradients[i].GetHess() * scale[1])};
    }
  }

 public:
  MetalHistogram() : rt_{GetRuntime()} {
    queue_ = [rt_.device newCommandQueue];
    CHECK(queue_) << "Metal command queue creation failed.";
  }

  void Reset() override {
    gradients_ = nil;
    pages_.clear();
  }

  void Build(std::size_t page_idx, GHistIndexMatrix const& gidx,
             RowSetCollection const& rows, std::vector<bst_node_t> const& nodes,
             Span<GradientPair const> gradients, std::vector<GHistRow> const& output) override {
    @autoreleasepool {
      CHECK_EQ(nodes.size(), output.size());
      CHECK_LE(nodes.size(), std::numeric_limits<std::uint32_t>::max());
      if (!gradients_) this->UploadGradients(gradients);
      if (pages_.size() <= page_idx) pages_.resize(page_idx + 1);
      auto& page = pages_[page_idx];
      if (!page.index) {
        page.index = Upload(rt_, gidx.index.data<std::uint8_t>(),
                            gidx.index.Size() * gidx.index.GetBinTypeSize());
        page.row_ptr = Upload(rt_, gidx.row_ptr.data(), gidx.row_ptr.size() * sizeof(std::size_t));
        page.offsets = Upload(rt_, gidx.index.Offset(),
                              gidx.index.OffsetSize() * sizeof(std::uint32_t));
      }
      std::vector<RowEntry> work;
      for (std::size_t node = 0; node < nodes.size(); ++node) {
        for (auto row : rows[nodes[node]]) {
          CHECK_GE(row, gidx.base_rowid);
          CHECK_LT(row - gidx.base_rowid + 1, gidx.row_ptr.size());
          CHECK_LT(row, gradients.size());
          work.push_back({row, static_cast<std::uint32_t>(node), 0});
        }
      }
      CHECK_LE(work.size(), std::numeric_limits<std::uint32_t>::max());
      auto bins = static_cast<std::size_t>(gidx.cut.TotalBins());
      for (auto out : output) CHECK_EQ(out.size(), bins);
      if (work.empty() || bins == 0) return;
      CHECK_LE(nodes.size(), std::numeric_limits<std::size_t>::max() / bins / 16);
      auto bytes = nodes.size() * bins * sizeof(QuantizedPair);
      auto histogram = Allocate(rt_, bytes);
      auto row_buffer = Upload(rt_, work.data(), work.size() * sizeof(RowEntry));
      Params params{work.size(), gidx.base_rowid, static_cast<std::uint32_t>(bins),
                    static_cast<std::uint32_t>(gidx.cut.Ptrs().size() - 1),
                    static_cast<std::uint32_t>(gidx.index.GetBinTypeSize()),
                    static_cast<std::uint32_t>(gidx.IsDense())};
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      CHECK(command) << "Metal command buffer allocation failed.";
      id<MTLBlitCommandEncoder> clear = [command blitCommandEncoder];
      CHECK(clear) << "Metal blit encoder allocation failed.";
      [clear fillBuffer:histogram range:NSMakeRange(0, bytes) value:0];
      [clear endEncoding];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      CHECK(encoder) << "Metal compute encoder allocation failed.";
      [encoder setComputePipelineState:rt_.pipeline];
      [encoder setBuffer:page.index offset:0 atIndex:0];
      [encoder setBuffer:page.row_ptr offset:0 atIndex:1];
      [encoder setBuffer:page.offsets offset:0 atIndex:2];
      [encoder setBuffer:gradients_ offset:0 atIndex:3];
      [encoder setBuffer:row_buffer offset:0 atIndex:4];
      [encoder setBuffer:histogram offset:0 atIndex:5];
      [encoder setBytes:&params length:sizeof(params) atIndex:6];
      auto threads = std::min<NSUInteger>(256, rt_.pipeline.maxTotalThreadsPerThreadgroup);
      [encoder dispatchThreads:MTLSizeMake(work.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted)
          << "Metal histogram execution failed: " << ErrorText(command.error);
      auto const* values = static_cast<QuantizedPair const*>(histogram.contents);
      for (std::size_t node = 0; node < nodes.size(); ++node) {
        for (std::size_t bin = 0; bin < bins; ++bin) {
          auto const& v = values[node * bins + bin];
          output[node][bin] += GradientPairPrecise{v.grad * inverse_scale_[0],
                                                  v.hess * inverse_scale_[1]};
        }
      }
      dispatches.fetch_add(1, std::memory_order_relaxed);
      processed_rows.fetch_add(work.size(), std::memory_order_relaxed);
    }
  }
};
}  // namespace

std::unique_ptr<Histogram> MakeHistogram() { return std::make_unique<MetalHistogram>(); }
bool IsAvailable() {
  @autoreleasepool {
    return MTLCreateSystemDefaultDevice() != nil;
  }
}
Statistics GetStatistics() {
  return {dispatches.load(std::memory_order_relaxed), processed_rows.load(std::memory_order_relaxed)};
}
}  // namespace xgboost::common::metal
