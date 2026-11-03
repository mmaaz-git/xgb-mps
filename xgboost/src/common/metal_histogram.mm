/** Copyright 2026, XGBoost Contributors */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../data/gradient_index.h"
#include "metal_auc.h"
#include "metal_data.h"
#include "metal_histogram.h"
#include "metal_memory.h"
#include "metal_ops.h"
#include "metal_predictor.h"
#include "metal_quantile.h"
#include "metal_reduction.h"
#include "metal_sampling.h"
#include "metal_shader.h"
#include "metal_split.h"
#include "metal_survival.h"
#include "survival_util.h"
#include "xgboost/logging.h"

namespace xgboost::common::metal {
namespace {
std::atomic<std::uint64_t> survival_gradient_dispatches{0}, survival_metric_dispatches{0},
    survival_cpu_corrections{0}, survival_cpu_time_ns{0};
std::atomic<std::uint64_t> split_dispatches{0}, split_gpu_time_ns{0};
std::atomic<std::uint64_t> residual_scale_dispatches{0};
std::atomic<std::uint64_t> gradient_sum_dispatches{0}, gradient_sum_gpu_time_ns{0},
    stump_dispatches{0};
std::atomic<std::uint64_t> sampling_dispatches{0}, sampling_gpu_time_ns{0};
std::atomic<std::uint64_t> histogram_conversion_dispatches{0}, histogram_subtraction_dispatches{0},
    resident_split_dispatches{0};
std::atomic<std::uint64_t> histogram_shared_gradient_bytes{0}, leaf_shared_gradient_bytes{0},
    leaf_shared_output_bytes{0};
std::atomic<std::uint64_t> objective_shared_input_bytes{0}, objective_shared_output_bytes{0},
    objective_staging_bytes{0};
std::atomic<std::uint64_t> roc_auc_dispatches{0}, roc_auc_gpu_time_ns{0};
std::atomic<std::uint64_t> pr_auc_dispatches{0}, pr_auc_gpu_time_ns{0};
std::atomic<std::uint64_t> float_sum_dispatches{0}, float_sum_gpu_time_ns{0};
std::atomic<std::uint64_t> row_workspace_allocations{0}, row_workspace_reuses{0};
std::atomic<std::uint64_t> index_gpu_time_ns{0};
std::atomic<std::uint64_t> binning_dispatches{0}, binning_gpu_time_ns{0};
std::atomic<std::uint64_t> mean_dispatches{0}, mean_gpu_time_ns{0};
std::atomic<std::uint64_t> radix_sort_dispatches{0}, quantile_dispatches{0},
    quantile_gpu_time_ns{0};
std::atomic<std::uint64_t> index_uploads{0}, index_uploaded_bytes{0},
    resident_histogram_dispatches{0};
std::atomic<std::uint64_t> partition_dispatches{0}, leaf_partition_dispatches{0},
    row_init_dispatches{0}, partition_gpu_time_ns{0};
std::atomic<std::uint64_t> dispatches{0};
std::atomic<std::uint64_t> processed_rows{0};
std::atomic<std::uint64_t> shared_dispatches{0};
std::atomic<std::uint64_t> gpu_time_ns{0};
std::atomic<std::uint64_t> quantization_dispatches{0}, gradient_bound_dispatches{0};
std::atomic<std::uint64_t> metric_dispatches{0}, metric_gpu_time_ns{0};
std::atomic<std::uint64_t> prediction_dispatches{0}, prediction_gpu_time_ns{0},
    leaf_update_dispatches{0};
std::atomic<std::uint64_t> objective_dispatches{0}, transform_dispatches{0},
    objective_gpu_time_ns{0};
std::atomic<std::uint64_t> staging_time_ns{0}, wait_time_ns{0}, readback_time_ns{0};
using Clock = std::chrono::steady_clock;
std::uint64_t Elapsed(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

struct Params {
  std::uint64_t n_rows, base_rowid, page_rows;
  std::uint32_t n_bins, n_features, bin_bytes, dense, n_nodes, padding, rows_per_lane;
};
static_assert(sizeof(Params) == 56);
struct QuantizedPair {
  std::int64_t grad, hess;
};
static_assert(sizeof(QuantizedPair) == 16);

char const* ErrorText(NSError* error) {
  return error ? error.localizedDescription.UTF8String : "unknown Metal error";
}

struct GradientScale {
  std::int32_t grad_exponent, hess_exponent;
  std::uint32_t valid, padding;
};
static_assert(sizeof(GradientScale) == 16);

class Runtime {
 public:
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
  id<MTLComputePipelineState> split_pipeline, multi_split_pipeline;
  id<MTLComputePipelineState> dequantize_hist_pipeline, subtract_hist_pipeline,
      gather_hist_pipeline;
  id<MTLComputePipelineState> residual_reduce_pipeline, residual_finish_pipeline, adaptive_pipeline;
  id<MTLComputePipelineState> quantile_metric_pipeline, binning_pipeline;
  id<MTLComputePipelineState> mean_weight_pipeline, mean_pipeline, mean_result_pipeline;
  id<MTLComputePipelineState> sort_init_pipeline, radix_rank_pipeline, radix_scatter_pipeline,
      prefix_u32_pipeline, prefix_add_pipeline, quantile_cdf_pipeline, quantile_select_pipeline;
  id<MTLComputePipelineState> roc_sort_init_pipeline, roc_counts_pipeline, roc_area_pipeline,
      pr_area_pipeline, prefix_double_pair_pipeline, prefix_double_pair_add_pipeline;
  id<MTLComputePipelineState> aft_gradient_pipeline, aft_metric_pipeline, aft_transform_pipeline;
  id<MTLComputePipelineState> uniform_sampling_pipeline;
  id<MTLComputePipelineState> gradient_sum_pipeline, gradient_sum_finish_pipeline,
      histogram_sum_pipeline, stump_weight_pipeline;
  id<MTLComputePipelineState> shared_pipeline, gradient_maximum_pipeline, gradient_scale_pipeline;
  id<MTLComputePipelineState> quantize_pipeline, objective_pipeline, transform_pipeline,
      prediction_pipeline, leaf_pipeline, metric_pipeline;
  id<MTLComputePipelineState> row_iota_pipeline, partition_classify_pipeline,
      partition_prefix_pipeline, partition_prefix_serial_pipeline, partition_scatter_pipeline,
      leaf_positions_pipeline, transpose_index_pipeline;

  Runtime() {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      CHECK(device) << "device=mps requested, but no Metal device is available. "
                       "GPU access may be restricted by the execution sandbox.";
      queue = [device newCommandQueue];
      CHECK(queue) << "Metal command queue creation failed.";
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
      auto shared_kernel = [library newFunctionWithName:@"xgb_histogram_dense_shared"];
      CHECK(shared_kernel) << "Missing shared Metal histogram kernel.";
      shared_pipeline = [device newComputePipelineStateWithFunction:shared_kernel error:&error];
      CHECK(shared_pipeline) << "Metal shared pipeline creation failed: " << ErrorText(error);
      auto quantize_kernel = [library newFunctionWithName:@"xgb_quantize_gradients"];
      quantize_pipeline = [device newComputePipelineStateWithFunction:quantize_kernel error:&error];
      CHECK(quantize_pipeline) << ErrorText(error);
      auto make_pipeline = [&](NSString* name) {
        auto function = [library newFunctionWithName:name];
        auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        CHECK(pipeline) << ErrorText(error);
        return pipeline;
      };
      aft_gradient_pipeline = make_pipeline(@"xgb_aft_gradient");
      aft_metric_pipeline = make_pipeline(@"xgb_aft_metric");
      aft_transform_pipeline = make_pipeline(@"xgb_aft_transform");
      objective_pipeline = make_pipeline(@"xgb_objective");
      transform_pipeline = make_pipeline(@"xgb_transform");
      prediction_pipeline = make_pipeline(@"xgb_predict");
      leaf_pipeline = make_pipeline(@"xgb_add_leaf");
      metric_pipeline = make_pipeline(@"xgb_metric");
      row_iota_pipeline = make_pipeline(@"xgb_row_iota");
      partition_classify_pipeline = make_pipeline(@"xgb_partition_classify");
      partition_prefix_pipeline = make_pipeline(@"xgb_partition_prefix");
      partition_prefix_serial_pipeline = make_pipeline(@"xgb_partition_prefix_serial");
      partition_scatter_pipeline = make_pipeline(@"xgb_partition_scatter");
      leaf_positions_pipeline = make_pipeline(@"xgb_leaf_positions");
      transpose_index_pipeline = make_pipeline(@"xgb_transpose_index");
      split_pipeline = make_pipeline(@"xgb_evaluate_splits");
      multi_split_pipeline = make_pipeline(@"xgb_evaluate_multi_splits");
      dequantize_hist_pipeline = make_pipeline(@"xgb_histogram_dequantize");
      subtract_hist_pipeline = make_pipeline(@"xgb_histogram_subtract");
      gather_hist_pipeline = make_pipeline(@"xgb_histogram_gather");
      residual_reduce_pipeline = make_pipeline(@"xgb_residual_scale_reduce");
      residual_finish_pipeline = make_pipeline(@"xgb_residual_scale_finish");
      adaptive_pipeline = make_pipeline(@"xgb_adaptive_objective");
      quantile_metric_pipeline = make_pipeline(@"xgb_quantile_metric");
      uniform_sampling_pipeline = make_pipeline(@"xgb_uniform_sample");
      gradient_maximum_pipeline = make_pipeline(@"xgb_gradient_maximum");
      gradient_scale_pipeline = make_pipeline(@"xgb_gradient_scale");
      gradient_sum_pipeline = make_pipeline(@"xgb_gradient_sum_reduce");
      gradient_sum_finish_pipeline = make_pipeline(@"xgb_gradient_sum_finish");
      histogram_sum_pipeline = make_pipeline(@"xgb_histogram_sum");
      stump_weight_pipeline = make_pipeline(@"xgb_stump_weight");
      binning_pipeline = make_pipeline(@"xgb_bin_entries");
      mean_weight_pipeline = make_pipeline(@"xgb_mean_weight_reduce");
      mean_pipeline = make_pipeline(@"xgb_mean_reduce");
      mean_result_pipeline = make_pipeline(@"xgb_mean_result");
      sort_init_pipeline = make_pipeline(@"xgb_sort_columns_init");
      radix_rank_pipeline = make_pipeline(@"xgb_radix_rank");
      radix_scatter_pipeline = make_pipeline(@"xgb_radix_scatter");
      prefix_u32_pipeline = make_pipeline(@"xgb_prefix_u32");
      prefix_add_pipeline = make_pipeline(@"xgb_prefix_add_u32");
      quantile_cdf_pipeline = make_pipeline(@"xgb_quantile_weight_cdf");
      quantile_select_pipeline = make_pipeline(@"xgb_quantile_select");
      roc_sort_init_pipeline = make_pipeline(@"xgb_roc_sort_init");
      roc_counts_pipeline = make_pipeline(@"xgb_roc_counts");
      roc_area_pipeline = make_pipeline(@"xgb_roc_area");
      pr_area_pipeline = make_pipeline(@"xgb_pr_area");
      prefix_double_pair_pipeline = make_pipeline(@"xgb_prefix_double_pair");
      prefix_double_pair_add_pipeline = make_pipeline(@"xgb_prefix_double_pair_add");
      CHECK_EQ(partition_classify_pipeline.threadExecutionWidth, 32);
      LOG(INFO) << "Metal compute backend: " << device.name.UTF8String;
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

class MetalHistogramBuffer final : public HistogramBuffer {
  std::size_t size_{0};

 public:
  id<MTLBuffer> buffer;
  void Resize(std::size_t size) override {
    if (size <= size_) return;
    auto& rt = GetRuntime();
    CHECK_LE(size, rt.device.maxBufferLength / sizeof(GradientPairPrecise));
    if (!buffer || buffer.length / sizeof(GradientPairPrecise) < size) {
      auto capacity = std::min<std::size_t>(
          std::max(size, size_ * 2), rt.device.maxBufferLength / sizeof(GradientPairPrecise));
      auto next = Allocate(rt, capacity * sizeof(GradientPairPrecise));
      if (size_) {
        auto command = [rt.queue commandBuffer];
        CHECK(command);
        auto encoder = [command blitCommandEncoder];
        CHECK(encoder);
        [encoder copyFromBuffer:buffer
                   sourceOffset:0
                       toBuffer:next
              destinationOffset:0
                           size:size_ * sizeof(GradientPairPrecise)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      }
      buffer = next;
    }
    size_ = size;
  }
  Span<GradientPairPrecise> Data() override {
    return {static_cast<GradientPairPrecise*>(buffer.contents), size_};
  }
  Span<GradientPairPrecise const> Data() const override {
    return {static_cast<GradientPairPrecise const*>(buffer.contents), size_};
  }
};

void EnsureBuffer(Runtime const& rt, id<MTLBuffer> __strong* buffer, std::size_t bytes) {
  if (!*buffer || (*buffer).length < bytes) *buffer = Allocate(rt, bytes);
}

struct BufferHolder {
  id<MTLBuffer> value;
};
class MetalSplitWorkspace final : public SplitWorkspace {
 public:
  id<MTLBuffer> hist, input, output, order, weights, scans, parent, bounds, children;
  std::vector<BufferHolder> metadata;
};
id<MTLBuffer> Scratch(Runtime const& rt, id<MTLBuffer> __strong* buffer, std::size_t bytes,
                      void const* data = nullptr) {
  EnsureBuffer(rt, buffer, bytes);
  if (data && bytes) std::memcpy((*buffer).contents, data, bytes);
  return *buffer;
}
id<MTLBuffer> GatherHistograms(Runtime const& rt, Span<HistogramInput const> inputs,
                               std::size_t size, id<MTLCommandBuffer> command,
                               MetalSplitWorkspace* workspace) {
  static_assert(sizeof(HistogramGather) == 24);
  auto output = Scratch(rt, &workspace->hist, size * sizeof(GradientPairPrecise));
  workspace->metadata.resize(inputs.size());
  std::size_t index = 0;
  for (auto const& input : inputs) {
    auto& metadata = workspace->metadata[index++].value;
    if (input.segments.empty()) continue;
    CHECK(input.source);
    auto source = static_cast<MetalHistogramBuffer const*>(input.source)->buffer;
    for (auto const& segment : input.segments) {
      CHECK_LE(segment.source + segment.bins, input.source->Data().size());
      CHECK_GT(segment.stride, 0);
      CHECK(segment.bins == 0 ||
            std::uint64_t(segment.output) + std::uint64_t(segment.bins - 1) * segment.stride <
                size);
    }
    auto segments = Scratch(rt, &metadata, input.segments.size() * sizeof(HistogramGather),
                            input.segments.data());
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt.gather_hist_pipeline];
    [encoder setBuffer:source offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:segments offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(input.segments.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
  }
  return output;
}

class Page final : public QuantizedPage {
 public:
  id<MTLBuffer> index, row_ptr, offsets, cuts;
  Page(Runtime const& rt, GHistIndexMatrix const& gidx) {
    auto bytes = gidx.index.Size() * gidx.index.GetBinTypeSize();
    auto source = Upload(rt, gidx.index.data<std::uint8_t>(), bytes);
    index = source;
    if (gidx.IsDense() && gidx.Size() && gidx.Features()) {
      index = Allocate(rt, bytes);
      struct {
        std::uint64_t rows;
        std::uint32_t features, width;
      } p{gidx.Size(), static_cast<std::uint32_t>(gidx.Features()),
          static_cast<std::uint32_t>(gidx.index.GetBinTypeSize())};
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      auto encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:rt.transpose_index_pipeline];
      [encoder setBuffer:source offset:0 atIndex:0];
      [encoder setBuffer:index offset:0 atIndex:1];
      [encoder setBytes:&p length:sizeof(p) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake((p.features + 31) / 32, (p.rows + 31) / 32, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      auto elapsed = command.GPUEndTime - command.GPUStartTime;
      if (elapsed > 0)
        index_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                    std::memory_order_relaxed);
    }
    row_ptr = Upload(rt, gidx.row_ptr.data(), gidx.row_ptr.size() * sizeof(std::size_t));
    offsets = Upload(rt, gidx.cut.Ptrs().data(), gidx.cut.Ptrs().size() * sizeof(std::uint32_t));
    cuts = Upload(rt, gidx.cut.Values().data(), gidx.cut.Values().size() * sizeof(float));
    index_uploads.fetch_add(1, std::memory_order_relaxed);
    index_uploaded_bytes.fetch_add(bytes + gidx.row_ptr.size() * sizeof(std::size_t) +
                                       gidx.cut.Ptrs().size() * sizeof(std::uint32_t) +
                                       gidx.cut.Values().size() * sizeof(float),
                                   std::memory_order_relaxed);
  }
};
std::shared_ptr<Page> GetPage(Runtime const& rt, GHistIndexMatrix const& gidx) {
  auto cached = std::atomic_load(&gidx.metal_page);
  if (!cached) {
    std::shared_ptr<QuantizedPage> created = std::make_shared<Page>(rt, gidx);
    if (std::atomic_compare_exchange_strong(&gidx.metal_page, &cached, created))
      cached = std::move(created);
  }
  return std::static_pointer_cast<Page>(cached);
}

void CompleteTraining(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
  auto elapsed = command.GPUEndTime - command.GPUStartTime;
  if (elapsed > 0)
    partition_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                    std::memory_order_relaxed);
}
class MetalRows final : public DeviceRows {
  Runtime& rt_{GetRuntime()};
  id<MTLBuffer> rows_, saved_rows_, ranks_;
  id<MTLBuffer> splits_, blocks_, categories_, counts_, prefix_, left_;
  id<MTLBuffer> leaves_, leaf_gradients_, leaf_output_;
  id<MTLBuffer> Workspace(id<MTLBuffer> __strong* buffer, std::size_t bytes,
                          void const* data = nullptr) {
    bool allocated = !*buffer || (*buffer).length < bytes;
    auto result = Scratch(rt_, buffer, bytes, data);
    (allocated ? row_workspace_allocations : row_workspace_reuses)
        .fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  std::size_t size_{0};
  bst_idx_t base_{0};
  bool ready_{false};

 public:
  void Reset(std::size_t rows, bst_idx_t base) override {
    size_ = rows;
    base_ = base;
    ready_ = false;
  }
  id<MTLBuffer> EnsureReady(id<MTLCommandBuffer> command) {
    CHECK_LE(size_, std::numeric_limits<std::uint32_t>::max());
    Workspace(&rows_, size_ * sizeof(bst_idx_t));
    if (!ready_ && size_) {
      struct {
        std::uint64_t rows, base;
      } p{size_, base_};
      auto encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:rt_.row_iota_pipeline];
      [encoder setBuffer:rows_ offset:0 atIndex:0];
      [encoder setBytes:&p length:sizeof(p) atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(size_, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
    }
    return rows_;
  }
  void MarkReady() {
    if (!ready_ && size_) row_init_dispatches.fetch_add(1, std::memory_order_relaxed);
    ready_ = true;
  }
  void CopyToHost(Span<bst_idx_t> output) const override {
    CHECK_EQ(output.size(), size_);
    if (size_) {
      CHECK(ready_);
      std::memcpy(output.data(), rows_.contents, output.size_bytes());
    }
  }
  std::vector<std::uint64_t> Partition(GHistIndexMatrix const& gidx,
                                       std::vector<PartitionSplit> splits,
                                       Span<std::uint32_t const> categories) override {
    struct Block {
      std::uint64_t begin, end;
      std::uint32_t node, padding;
    };
    struct Params {
      std::uint64_t rows, base;
      std::uint32_t bin_bytes, dense, nodes, blocks;
    };
    static_assert(sizeof(Block) == 24 && sizeof(PartitionSplit) == 48);
    CHECK_EQ(gidx.Size(), size_);
    CHECK_EQ(gidx.base_rowid, base_);
    CHECK_LE(splits.size(), std::numeric_limits<std::uint32_t>::max());
    std::vector<Block> blocks;
    std::size_t largest_block_count = 0;
    for (std::size_t node = 0; node < splits.size(); ++node) {
      auto& split = splits[node];
      CHECK_LE(split.end, size_);
      CHECK_LE(split.begin, split.end);
      split.block_begin = blocks.size();
      for (auto begin = split.begin; begin < split.end; begin += 256)
        blocks.push_back(
            {begin, std::min(begin + 256, split.end), static_cast<std::uint32_t>(node), 0});
      split.block_end = blocks.size();
      largest_block_count =
          std::max(largest_block_count, std::size_t{split.block_end - split.block_begin});
    }
    bool serial_prefix = largest_block_count <= 16;
    std::vector<std::uint64_t> result(splits.size(), 0);
    if (blocks.empty()) return result;
    @autoreleasepool {
      auto page = GetPage(rt_, gidx);
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      this->EnsureReady(command);
      Workspace(&saved_rows_, size_ * sizeof(bst_idx_t));
      Workspace(&ranks_, size_ * sizeof(std::uint32_t));
      auto split_buffer =
          Workspace(&splits_, splits.size() * sizeof(PartitionSplit), splits.data());
      auto block_buffer = Workspace(&blocks_, blocks.size() * sizeof(Block), blocks.data());
      auto cats = Workspace(&categories_, categories.size_bytes(), categories.data());
      auto counts = Workspace(&counts_, blocks.size() * sizeof(std::uint32_t));
      auto prefix = Workspace(&prefix_, blocks.size() * sizeof(std::uint32_t));
      auto left = Workspace(&left_, splits.size() * sizeof(std::uint64_t));
      Params p{size_,
               base_,
               static_cast<std::uint32_t>(gidx.index.GetBinTypeSize()),
               static_cast<std::uint32_t>(gidx.IsDense()),
               static_cast<std::uint32_t>(splits.size()),
               static_cast<std::uint32_t>(blocks.size())};
      auto encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:rt_.partition_classify_pipeline];
      [encoder setBuffer:page->index offset:0 atIndex:0];
      [encoder setBuffer:page->row_ptr offset:0 atIndex:1];
      [encoder setBuffer:page->offsets offset:0 atIndex:2];
      [encoder setBuffer:page->cuts offset:0 atIndex:3];
      [encoder setBuffer:rows_ offset:0 atIndex:4];
      [encoder setBuffer:split_buffer offset:0 atIndex:5];
      [encoder setBuffer:block_buffer offset:0 atIndex:6];
      [encoder setBuffer:cats offset:0 atIndex:7];
      [encoder setBuffer:ranks_ offset:0 atIndex:8];
      [encoder setBuffer:counts offset:0 atIndex:9];
      [encoder setBuffer:saved_rows_ offset:0 atIndex:10];
      [encoder setBytes:&p length:sizeof(p) atIndex:11];
      [encoder dispatchThreadgroups:MTLSizeMake(blocks.size(), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:serial_prefix ? rt_.partition_prefix_serial_pipeline
                                                     : rt_.partition_prefix_pipeline];
      [encoder setBuffer:split_buffer offset:0 atIndex:0];
      [encoder setBuffer:counts offset:0 atIndex:1];
      [encoder setBuffer:prefix offset:0 atIndex:2];
      [encoder setBuffer:left offset:0 atIndex:3];
      [encoder setBytes:&p length:sizeof(p) atIndex:4];
      if (serial_prefix) {
        [encoder dispatchThreads:MTLSizeMake(splits.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      } else {
        [encoder dispatchThreadgroups:MTLSizeMake(splits.size(), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [encoder endEncoding];
      encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:rt_.partition_scatter_pipeline];
      [encoder setBuffer:rows_ offset:0 atIndex:0];
      [encoder setBuffer:saved_rows_ offset:0 atIndex:1];
      [encoder setBuffer:split_buffer offset:0 atIndex:2];
      [encoder setBuffer:block_buffer offset:0 atIndex:3];
      [encoder setBuffer:ranks_ offset:0 atIndex:4];
      [encoder setBuffer:prefix offset:0 atIndex:5];
      [encoder setBuffer:left offset:0 atIndex:6];
      [encoder dispatchThreadgroups:MTLSizeMake(blocks.size(), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      CompleteTraining(command);
      this->MarkReady();
      std::memcpy(result.data(), left.contents, result.size() * sizeof(std::uint64_t));
      partition_dispatches.fetch_add(1, std::memory_order_relaxed);
      return result;
    }
  }
  void LeafPositions(std::vector<LeafRange> leaves,
                     linalg::MatrixView<GradientPair const> gradients,
                     Span<bst_node_t> output) override {
    if (!size_) return;
    CHECK_GE(output.size(), base_ + size_);
    CHECK_GE(gradients.Shape(0), base_ + size_);
    std::sort(leaves.begin(), leaves.end(), [](auto const& a, auto const& b) {
      // Empty leaves must precede a nonempty leaf starting at the same offset.
      return a.begin < b.begin || (a.begin == b.begin && a.end < b.end);
    });
    std::size_t end = 0;
    for (auto const& leaf : leaves) {
      CHECK_EQ(leaf.begin, end);
      CHECK_GE(leaf.end, leaf.begin);
      end = leaf.end;
    }
    CHECK_EQ(end, size_);
    @autoreleasepool {
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      this->EnsureReady(command);
      auto leaf_buffer = Workspace(&leaves_, leaves.size() * sizeof(LeafRange), leaves.data());
      auto values = gradients.Values();
      auto g = TryBorrowMemory(rt_.device, values.data(), values.size_bytes(), 8);
      bool shared_gradient = g.buffer != nil;
      if (shared_gradient) {
        leaf_gradients_ = nil;
      } else if (base_ == 0 && size_ == gradients.Shape(0)) {
        g = {Workspace(&leaf_gradients_, values.size_bytes(), values.data()), 0};
      } else {
        // External-memory pages must not each retain an entire gradient matrix.
        leaf_gradients_ = nil;
        g = {Upload(rt_, values.data(), values.size_bytes()), 0};
      }
      auto destination = output.data() + base_;
      auto output_bytes = size_ * sizeof(bst_node_t);
      bool disjoint =
          !MemoryRangesOverlap(values.data(), values.size_bytes(), destination, output_bytes);
      auto out =
          disjoint ? TryBorrowMemory(rt_.device, destination, output_bytes, 4) : BorrowedMemory{};
      bool shared_output = out.buffer != nil;
      if (shared_output)
        leaf_output_ = nil;
      else
        out = {Workspace(&leaf_output_, output_bytes), 0};
      struct {
        std::uint64_t rows, base, stride0, stride1;
        std::uint32_t targets, leaves;
      } p{size_,
          base_,
          gradients.Stride(0),
          gradients.Stride(1),
          static_cast<std::uint32_t>(gradients.Shape(1)),
          static_cast<std::uint32_t>(leaves.size())};
      auto encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:rt_.leaf_positions_pipeline];
      [encoder setBuffer:rows_ offset:0 atIndex:0];
      [encoder setBuffer:leaf_buffer offset:0 atIndex:1];
      [encoder setBuffer:g.buffer offset:g.offset atIndex:2];
      [encoder setBuffer:out.buffer offset:out.offset atIndex:3];
      [encoder setBytes:&p length:sizeof(p) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(size_, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      CompleteTraining(command);
      this->MarkReady();
      if (!shared_output) std::memcpy(destination, out.buffer.contents, output_bytes);
      if (shared_gradient)
        leaf_shared_gradient_bytes.fetch_add(values.size_bytes(), std::memory_order_relaxed);
      if (shared_output)
        leaf_shared_output_bytes.fetch_add(output_bytes, std::memory_order_relaxed);
      leaf_partition_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
  }
};

class MetalHistogram final : public Histogram {
  Runtime& rt_;
  id<MTLCommandQueue> queue_;
  id<MTLBuffer> gradients_, raw_gradients_, histogram_buffer_, row_buffer_, node_buffer_,
      block_buffer_, gradient_maximum_, gradient_scale_;
  bool gradients_ready_{false};

  std::size_t UploadGradients(Span<GradientPair const> gradients, id<MTLCommandBuffer> command) {
    CHECK_LE(gradients.size(), std::numeric_limits<std::uint32_t>::max());
    if (!gradients_ || gradients_.length < gradients.size() * sizeof(QuantizedPair))
      gradients_ = Allocate(rt_, gradients.size() * sizeof(QuantizedPair));
    static_assert(sizeof(GradientPair) == 8);
    // Only the command retains a borrowed view. The class caches GPU-owned
    // quantized gradients and, when needed, a separate owned staging allocation.
    auto input = TryBorrowMemory(rt_.device, gradients.data(), gradients.size_bytes(), 8);
    auto shared_bytes = input.buffer ? gradients.size_bytes() : 0;
    if (!input.buffer) {
      input = {Scratch(rt_, &raw_gradients_, gradients.size_bytes(), gradients.data()), 0};
    } else {
      raw_gradients_ = nil;
    }
    EnsureBuffer(rt_, &gradient_maximum_, 2 * sizeof(std::uint32_t));
    EnsureBuffer(rt_, &gradient_scale_, sizeof(GradientScale));
    auto clear = [command blitCommandEncoder];
    CHECK(clear);
    [clear fillBuffer:gradient_maximum_ range:NSMakeRange(0, 2 * sizeof(std::uint32_t)) value:0];
    [clear endEncoding];
    std::uint64_t count = gradients.size();
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt_.gradient_maximum_pipeline];
    [encoder setBuffer:input.buffer offset:input.offset atIndex:0];
    [encoder setBuffer:gradient_maximum_ offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt_.gradient_scale_pipeline];
    [encoder setBuffer:gradient_maximum_ offset:0 atIndex:0];
    [encoder setBuffer:gradient_scale_ offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt_.quantize_pipeline];
    [encoder setBuffer:input.buffer offset:input.offset atIndex:0];
    [encoder setBuffer:gradients_ offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder setBuffer:gradient_scale_ offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    return shared_bytes;
  }

 public:
  MetalHistogram() : rt_{GetRuntime()} {
    queue_ = rt_.queue;
    CHECK(queue_) << "Metal command queue creation failed.";
  }

  void Reset() override { gradients_ready_ = false; }

  void Build(std::size_t page_idx, GHistIndexMatrix const& gidx, RowSetCollection const& rows,
             std::vector<bst_node_t> const& nodes, Span<GradientPair const> gradients,
             std::vector<GHistRow> const& output, HistogramBuffer* storage) override {
    @autoreleasepool {
      CHECK_EQ(nodes.size(), output.size());
      CHECK_LE(nodes.size(), std::numeric_limits<std::uint32_t>::max());
      auto staging_start = Clock::now();
      auto page = GetPage(rt_, gidx);
      std::vector<std::uint64_t> node_ptr{0};
      std::size_t max_node_rows = 0;
      for (auto node : nodes) {
        auto size = rows[node].Size();
        max_node_rows = std::max(max_node_rows, size);
        node_ptr.push_back(node_ptr.back() + size);
      }
      auto total_rows = node_ptr.back();
      CHECK_LE(total_rows, std::numeric_limits<std::uint32_t>::max());
      auto bins = static_cast<std::size_t>(gidx.cut.TotalBins());
      for (auto out : output) CHECK_EQ(out.size(), bins);
      if (total_rows == 0 || bins == 0) return;
      CHECK_LE(nodes.size(), std::numeric_limits<std::size_t>::max() / bins / 16);
      auto bytes = nodes.size() * bins * sizeof(QuantizedPair);
      if (!histogram_buffer_ || histogram_buffer_.length < bytes)
        histogram_buffer_ = Allocate(rt_, bytes);
      auto histogram = histogram_buffer_;
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      CHECK(command);
      auto device_rows = static_cast<MetalRows*>(rows.metal_rows.get());
      std::vector<std::uint64_t> row_begin;
      id<MTLBuffer> row_buffer;
      if (device_rows) {
        row_buffer = device_rows->EnsureReady(command);
        for (auto node : nodes)
          row_begin.push_back(rows[node].Size() ? rows[node].begin() - rows.Data()->data() : 0);
      } else {
        EnsureBuffer(rt_, &row_buffer_, total_rows * sizeof(bst_idx_t));
        row_buffer = row_buffer_;
        for (std::size_t node = 0; node < nodes.size(); ++node) {
          auto size = rows[nodes[node]].Size();
          row_begin.push_back(node_ptr[node]);
          if (size)
            std::memcpy(static_cast<bst_idx_t*>(row_buffer.contents) + node_ptr[node],
                        rows[nodes[node]].begin(), size * sizeof(bst_idx_t));
        }
      }
      auto row_begin_buffer =
          Upload(rt_, row_begin.data(), row_begin.size() * sizeof(std::uint64_t));
      auto node_bytes = node_ptr.size() * sizeof(std::uint64_t);
      if (!node_buffer_ || node_buffer_.length < node_bytes)
        node_buffer_ = Allocate(rt_, node_bytes);
      std::memcpy(node_buffer_.contents, node_ptr.data(), node_bytes);
      auto node_buffer = node_buffer_;
      Params params{total_rows, gidx.base_rowid, gidx.Size(), static_cast<std::uint32_t>(bins),
                    static_cast<std::uint32_t>(gidx.cut.Ptrs().size() - 1),
                    static_cast<std::uint32_t>(gidx.index.GetBinTypeSize()),
                    static_cast<std::uint32_t>(gidx.IsDense()),
                    static_cast<std::uint32_t>(nodes.size()),
                    static_cast<std::uint32_t>(gidx.IsDense()),
                    // Large, dense node populations benefit from amortizing merges;
                    // fragmented branches need smaller groups for memory locality.
                    max_node_rows > gidx.Size() / 4 ? 32u : 8u};

      bool quantized = !gradients_ready_;
      auto shared_gradient_bytes = quantized ? this->UploadGradients(gradients, command) : 0;
      id<MTLBlitCommandEncoder> clear = [command blitCommandEncoder];
      CHECK(clear) << "Metal blit encoder allocation failed.";
      [clear fillBuffer:histogram range:NSMakeRange(0, bytes) value:0];
      [clear endEncoding];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      CHECK(encoder) << "Metal compute encoder allocation failed.";
      std::size_t max_feature_bins = 0;
      auto const& cuts = gidx.cut.Ptrs();
      for (std::size_t f = 1; f < cuts.size(); ++f)
        max_feature_bins = std::max<std::size_t>(max_feature_bins, cuts[f] - cuts[f - 1]);
      auto shared_bytes = max_feature_bins * sizeof(QuantizedPair);
      bool use_shared = gidx.IsDense() && total_rows >= 16384 &&
                        shared_bytes <= rt_.device.maxThreadgroupMemoryLength &&
                        params.n_features <= 65535;
      auto pipeline = use_shared ? rt_.shared_pipeline : rt_.pipeline;
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:page->index offset:0 atIndex:0];
      [encoder setBuffer:page->row_ptr offset:0 atIndex:1];
      [encoder setBuffer:page->offsets offset:0 atIndex:2];
      [encoder setBuffer:gradients_ offset:0 atIndex:3];
      [encoder setBuffer:row_buffer offset:0 atIndex:4];
      [encoder setBuffer:histogram offset:0 atIndex:5];
      [encoder setBytes:&params length:sizeof(params) atIndex:6];
      auto threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
      [encoder setBuffer:node_buffer offset:0 atIndex:7];
      [encoder setBuffer:row_begin_buffer offset:0 atIndex:8];
      if (use_shared) {
        [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
        std::vector<std::uint64_t> block_ptr{0};
        for (std::size_t node = 0; node < nodes.size(); ++node) {
          auto size = node_ptr[node + 1] - node_ptr[node];
          block_ptr.push_back(block_ptr.back() + (size + threads * params.rows_per_lane - 1) /
                                                     (threads * params.rows_per_lane));
        }
        auto block_bytes = block_ptr.size() * sizeof(std::uint64_t);
        EnsureBuffer(rt_, &block_buffer_, block_bytes);
        std::memcpy(block_buffer_.contents, block_ptr.data(), block_bytes);
        [encoder setBuffer:block_buffer_ offset:0 atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(block_ptr.back(), params.n_features, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      } else {
        [encoder dispatchThreads:MTLSizeMake(total_rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      }
      [encoder endEncoding];
      CHECK(storage);
      auto shared = static_cast<MetalHistogramBuffer*>(storage);
      std::vector<std::uint64_t> output_offsets;
      for (auto out : output) output_offsets.push_back(out.data() - storage->Data().data());
      auto offset_buffer =
          Upload(rt_, output_offsets.data(), output_offsets.size() * sizeof(std::uint64_t));
      struct {
        std::uint64_t count;
        std::uint32_t bins, add;
      } dequantize{nodes.size() * bins, static_cast<std::uint32_t>(bins), page_idx != 0};
      CHECK_LE(dequantize.count, std::numeric_limits<std::uint32_t>::max());
      encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:rt_.dequantize_hist_pipeline];
      [encoder setBuffer:histogram offset:0 atIndex:0];
      [encoder setBuffer:shared->buffer offset:0 atIndex:1];
      [encoder setBuffer:offset_buffer offset:0 atIndex:2];
      [encoder setBytes:&dequantize length:sizeof(dequantize) atIndex:3];
      [encoder setBuffer:gradient_scale_ offset:0 atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(dequantize.count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      staging_time_ns.fetch_add(Elapsed(staging_start), std::memory_order_relaxed);
      auto wait_start = Clock::now();
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted)
          << "Metal histogram execution failed: " << ErrorText(command.error);
      if (device_rows) {
        device_rows->MarkReady();
        resident_histogram_dispatches.fetch_add(1, std::memory_order_relaxed);
      }
      wait_time_ns.fetch_add(Elapsed(wait_start), std::memory_order_relaxed);
      if (quantized) {
        CHECK(static_cast<GradientScale const*>(gradient_scale_.contents)->valid)
            << "Metal histograms require finite gradients and Hessians.";
        gradient_bound_dispatches.fetch_add(1, std::memory_order_relaxed);
        quantization_dispatches.fetch_add(1, std::memory_order_relaxed);
        histogram_shared_gradient_bytes.fetch_add(shared_gradient_bytes, std::memory_order_relaxed);
        gradients_ready_ = true;
      }
      histogram_conversion_dispatches.fetch_add(1, std::memory_order_relaxed);
      dispatches.fetch_add(1, std::memory_order_relaxed);
      processed_rows.fetch_add(total_rows, std::memory_order_relaxed);
      if (use_shared) shared_dispatches.fetch_add(1, std::memory_order_relaxed);
      auto elapsed = command.GPUEndTime - command.GPUStartTime;
      if (elapsed > 0)
        gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9), std::memory_order_relaxed);
    }
  }
};
}  // namespace

std::unique_ptr<SplitWorkspace> MakeSplitWorkspace() {
  return std::make_unique<MetalSplitWorkspace>();
}
std::unique_ptr<HistogramBuffer> MakeHistogramBuffer() {
  return std::make_unique<MetalHistogramBuffer>();
}
void SubtractHistograms(HistogramBuffer* storage, Span<HistogramSubtraction const> tasks) {
  static_assert(sizeof(HistogramSubtraction) == 32);
  if (tasks.empty()) return;
  CHECK(storage);
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto buffer = static_cast<MetalHistogramBuffer*>(storage)->buffer;
    std::uint64_t bins = 0;
    for (auto const& task : tasks) {
      bins = std::max(bins, task.bins);
      CHECK_LE(task.parent + task.bins, storage->Data().size());
      CHECK_LE(task.sibling + task.bins, storage->Data().size());
      CHECK_LE(task.output + task.bins, storage->Data().size());
    }
    if (!bins) return;
    auto input = Upload(rt, tasks.data(), tasks.size_bytes());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt.subtract_hist_pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(bins, tasks.size(), 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    histogram_subtraction_dispatches.fetch_add(1, std::memory_order_relaxed);
  }
}
std::unique_ptr<DeviceRows> MakeDeviceRows() { return std::make_unique<MetalRows>(); }
std::unique_ptr<Histogram> MakeHistogram() { return std::make_unique<MetalHistogram>(); }
bool IsAvailable() {
  @autoreleasepool {
    return MTLCreateSystemDefaultDevice() != nil;
  }
}
Statistics GetStatistics() {
  return {dispatches.load(std::memory_order_relaxed),
          processed_rows.load(std::memory_order_relaxed),
          shared_dispatches.load(std::memory_order_relaxed),
          gpu_time_ns.load(std::memory_order_relaxed),
          quantization_dispatches.load(std::memory_order_relaxed),
          staging_time_ns.load(std::memory_order_relaxed),
          wait_time_ns.load(std::memory_order_relaxed),
          readback_time_ns.load(std::memory_order_relaxed),
          objective_dispatches.load(std::memory_order_relaxed),
          transform_dispatches.load(std::memory_order_relaxed),
          objective_gpu_time_ns.load(std::memory_order_relaxed),
          prediction_dispatches.load(std::memory_order_relaxed),
          prediction_gpu_time_ns.load(std::memory_order_relaxed),
          leaf_update_dispatches.load(std::memory_order_relaxed),
          metric_dispatches.load(std::memory_order_relaxed),
          metric_gpu_time_ns.load(std::memory_order_relaxed),
          index_uploads.load(std::memory_order_relaxed),
          index_uploaded_bytes.load(std::memory_order_relaxed),
          resident_histogram_dispatches.load(std::memory_order_relaxed),
          partition_dispatches.load(std::memory_order_relaxed),
          leaf_partition_dispatches.load(std::memory_order_relaxed),
          row_init_dispatches.load(std::memory_order_relaxed),
          partition_gpu_time_ns.load(std::memory_order_relaxed),
          index_gpu_time_ns.load(std::memory_order_relaxed),
          split_dispatches.load(std::memory_order_relaxed),
          split_gpu_time_ns.load(std::memory_order_relaxed),
          residual_scale_dispatches.load(std::memory_order_relaxed),
          sampling_dispatches.load(std::memory_order_relaxed),
          sampling_gpu_time_ns.load(std::memory_order_relaxed),
          histogram_conversion_dispatches.load(std::memory_order_relaxed),
          histogram_subtraction_dispatches.load(std::memory_order_relaxed),
          resident_split_dispatches.load(std::memory_order_relaxed),
          gradient_bound_dispatches.load(std::memory_order_relaxed),
          gradient_sum_dispatches.load(std::memory_order_relaxed),
          gradient_sum_gpu_time_ns.load(std::memory_order_relaxed),
          stump_dispatches.load(std::memory_order_relaxed),
          binning_dispatches.load(std::memory_order_relaxed),
          binning_gpu_time_ns.load(std::memory_order_relaxed),
          mean_dispatches.load(std::memory_order_relaxed),
          mean_gpu_time_ns.load(std::memory_order_relaxed),
          radix_sort_dispatches.load(std::memory_order_relaxed),
          quantile_dispatches.load(std::memory_order_relaxed),
          quantile_gpu_time_ns.load(std::memory_order_relaxed),
          survival_gradient_dispatches.load(std::memory_order_relaxed),
          survival_metric_dispatches.load(std::memory_order_relaxed),
          survival_cpu_corrections.load(std::memory_order_relaxed),
          survival_cpu_time_ns.load(std::memory_order_relaxed),
          row_workspace_allocations.load(std::memory_order_relaxed),
          row_workspace_reuses.load(std::memory_order_relaxed),
          float_sum_dispatches.load(std::memory_order_relaxed),
          float_sum_gpu_time_ns.load(std::memory_order_relaxed),
          objective_shared_input_bytes.load(std::memory_order_relaxed),
          objective_shared_output_bytes.load(std::memory_order_relaxed),
          objective_staging_bytes.load(std::memory_order_relaxed),
          histogram_shared_gradient_bytes.load(std::memory_order_relaxed),
          leaf_shared_gradient_bytes.load(std::memory_order_relaxed),
          leaf_shared_output_bytes.load(std::memory_order_relaxed),
          roc_auc_dispatches.load(std::memory_order_relaxed),
          roc_auc_gpu_time_ns.load(std::memory_order_relaxed),
          pr_auc_dispatches.load(std::memory_order_relaxed),
          pr_auc_gpu_time_ns.load(std::memory_order_relaxed)};
}

namespace {
void CompleteOperation(id<MTLCommandBuffer> command, std::atomic<std::uint64_t>* counter) {
  [command commit];
  [command waitUntilCompleted];
  CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
  counter->fetch_add(1, std::memory_order_relaxed);
  auto elapsed = command.GPUEndTime - command.GPUStartTime;
  if (elapsed > 0)
    objective_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                    std::memory_order_relaxed);
}
}  // namespace
Objective ObjectiveFromName(char const* name) {
  if (!strcmp(name, "reg:squarederror")) return Objective::kSquare;
  if (!strcmp(name, "reg:logistic") || !strcmp(name, "binary:logistic") ||
      !strcmp(name, "binary:logitraw"))
    return Objective::kLogistic;
  if (!strcmp(name, "reg:gamma")) return Objective::kGamma;
  LOG(FATAL) << "Missing Metal objective: " << name;
  return Objective::kSquare;
}
Transform TransformFromName(char const* name) {
  if (!strcmp(name, "reg:logistic") || !strcmp(name, "binary:logistic")) return Transform::kSigmoid;
  if (!strcmp(name, "reg:gamma")) return Transform::kExp;
  return Transform::kIdentity;
}

namespace {
id<MTLBuffer> EncodeGradientSum(Runtime& rt, linalg::MatrixView<GradientPair const> gradients,
                                id<MTLCommandBuffer> command) {
  auto rows = gradients.Shape(0), targets = gradients.Shape(1);
  CHECK_GT(rows, 0);
  CHECK_GT(targets, 0);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(targets, 65535);
  auto values = gradients.Values();
  auto input = Upload(rt, values.data(), values.size_bytes());
  auto blocks = (rows + 255) / 256;
  auto partial = Allocate(rt, blocks * targets * sizeof(GradientPairPrecise));
  auto output = Allocate(rt, targets * sizeof(GradientPairPrecise));
  struct {
    std::uint64_t rows, stride0, stride1;
    std::uint32_t targets, blocks;
  } p{rows, gradients.Stride(0), gradients.Stride(1), static_cast<std::uint32_t>(targets),
      static_cast<std::uint32_t>(blocks)};
  static_assert(sizeof(p) == 32);
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:rt.gradient_sum_pipeline];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:partial offset:0 atIndex:1];
  [e setBytes:&p length:sizeof(p) atIndex:2];
  [e dispatchThreadgroups:MTLSizeMake(blocks, targets, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:rt.gradient_sum_finish_pipeline];
  [e setBuffer:partial offset:0 atIndex:0];
  [e setBuffer:output offset:0 atIndex:1];
  [e setBytes:&p length:sizeof(p) atIndex:2];
  [e dispatchThreadgroups:MTLSizeMake(targets, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  return output;
}
void CompleteGradientSum(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
  gradient_sum_dispatches.fetch_add(1, std::memory_order_relaxed);
  auto elapsed = command.GPUEndTime - command.GPUStartTime;
  if (elapsed > 0)
    gradient_sum_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                       std::memory_order_relaxed);
}
}  // namespace

void SumGradients(linalg::MatrixView<GradientPair const> gradients,
                  linalg::VectorView<GradientPairPrecise> output) {
  CHECK_EQ(gradients.Shape(1), output.Size());
  if (!gradients.Size()) {
    for (std::size_t t = 0; t < output.Size(); ++t) output(t) = GradientPairPrecise{};
    return;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto result = EncodeGradientSum(rt, gradients, command);
    CompleteGradientSum(command);
    auto data = static_cast<GradientPairPrecise const*>(result.contents);
    for (std::size_t t = 0; t < output.Size(); ++t) output(t) = data[t];
  }
}
void FitStump(linalg::MatrixView<GradientPair const> gradients, linalg::VectorView<float> output) {
  CHECK_EQ(gradients.Shape(1), output.Size());
  if (!gradients.Size()) {
    for (std::size_t t = 0; t < output.Size(); ++t) output(t) = 0.0f;
    return;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto sums = EncodeGradientSum(rt, gradients, command);
    auto result = Allocate(rt, output.Size() * sizeof(float));
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.stump_weight_pipeline];
    [e setBuffer:sums offset:0 atIndex:0];
    [e setBuffer:result offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake(output.Size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteGradientSum(command);
    auto data = static_cast<float const*>(result.contents);
    for (std::size_t t = 0; t < output.Size(); ++t) output(t) = data[t];
    stump_dispatches.fetch_add(1, std::memory_order_relaxed);
  }
}
GradientPairPrecise SumHistogram(HistogramBuffer const* storage, std::size_t offset,
                                 std::size_t bins) {
  CHECK(storage);
  CHECK_LE(offset, storage->Data().size());
  CHECK_LE(bins, storage->Data().size() - offset);
  if (!bins) return GradientPairPrecise{};
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto shared = static_cast<MetalHistogramBuffer const*>(storage);
    auto output = Allocate(rt, sizeof(GradientPairPrecise));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    std::uint64_t range[2]{offset, bins};
    [e setComputePipelineState:rt.histogram_sum_pipeline];
    [e setBuffer:shared->buffer offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBytes:range length:sizeof(range) atIndex:2];
    [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    CompleteGradientSum(command);
    return *static_cast<GradientPairPrecise const*>(output.contents);
  }
}

double Sum(Span<float const> values) {
  if (values.empty()) return 0.0;
  CHECK_LE(values.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, values.data(), values.size_bytes());
    auto blocks = (values.size() + 255) / 256;
    auto partial = Allocate(rt, blocks * sizeof(GradientPairPrecise));
    auto result = Allocate(rt, sizeof(GradientPairPrecise));
    struct {
      std::uint64_t rows, stride0, stride1;
      std::uint32_t targets, blocks;
    } p{values.size(), 1, 1, 1, static_cast<std::uint32_t>(blocks)};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.mean_weight_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:partial offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.gradient_sum_finish_pipeline];
    [e setBuffer:partial offset:0 atIndex:0];
    [e setBuffer:result offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    float_sum_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      float_sum_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                      std::memory_order_relaxed);
    return static_cast<GradientPairPrecise const*>(result.contents)->GetGrad();
  }
}

void Mean(MeanKind kind, linalg::MatrixView<float const> values, Span<float const> weights,
          linalg::VectorView<float> output) {
  bool weighted = kind == MeanKind::kWeightedSample;
  auto rows = values.Shape(0), targets = values.Shape(1);
  CHECK_EQ(output.Size(), std::max<std::size_t>(targets, 1));
  if (weighted) {
    CHECK_EQ(weights.size(), rows);
    CHECK_GT(rows, 0) << "weights must contain at least one non-zero value.";
  }
  if (!rows || (!targets && !weighted)) {
    for (std::size_t t = 0; t < output.Size(); ++t) output(t) = 0;
    return;
  }
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(targets, 65535);
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto input = Upload(rt, values.Values().data(), values.Values().size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    struct Shape {
      std::uint64_t rows, stride0, stride1;
      std::uint32_t targets, blocks;
    } shape{rows, values.Stride(0), values.Stride(1), static_cast<std::uint32_t>(targets),
            static_cast<std::uint32_t>((rows + 255) / 256)};
    static_assert(sizeof(Shape) == 32);
    auto finish = [&](id<MTLBuffer> partial, Shape p) {
      auto result = Allocate(rt, p.targets * sizeof(GradientPairPrecise));
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.gradient_sum_finish_pipeline];
      [e setBuffer:partial offset:0 atIndex:0];
      [e setBuffer:result offset:0 atIndex:1];
      [e setBytes:&p length:sizeof(p) atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(p.targets, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      return result;
    };
    auto denominator = Allocate(rt, sizeof(GradientPairPrecise));
    if (weighted) {
      auto partial = Allocate(rt, shape.blocks * sizeof(GradientPairPrecise));
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.mean_weight_pipeline];
      [e setBuffer:weight offset:0 atIndex:0];
      [e setBuffer:partial offset:0 atIndex:1];
      [e setBytes:&shape length:sizeof(shape) atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(shape.blocks, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto weight_shape = shape;
      weight_shape.targets = 1;
      denominator = finish(partial, weight_shape);
    }
    id<MTLBuffer> result;
    if (targets) {
      auto partial = Allocate(rt, shape.blocks * targets * sizeof(GradientPairPrecise));
      struct {
        Shape shape;
        std::uint32_t weighted, float32_terms;
      } p{shape, static_cast<std::uint32_t>(weighted),
          static_cast<std::uint32_t>(kind == MeanKind::kFloat32)};
      static_assert(sizeof(p) == 40);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.mean_pipeline];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:weight offset:0 atIndex:1];
      [e setBuffer:denominator offset:0 atIndex:2];
      [e setBuffer:partial offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake(shape.blocks, targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto sums = finish(partial, shape);
      result = Allocate(rt, targets * sizeof(float));
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.mean_result_pipeline];
      [e setBuffer:sums offset:0 atIndex:0];
      [e setBuffer:result offset:0 atIndex:1];
      [e dispatchThreads:MTLSizeMake(targets, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    if (weighted) {
      CHECK_GT(static_cast<GradientPairPrecise const*>(denominator.contents)->GetGrad(), 0.0)
          << "weights must contain at least one non-zero value.";
    }
    for (std::size_t t = 0; t < output.Size(); ++t)
      output(t) = targets ? static_cast<float const*>(result.contents)[t] : 0;
    mean_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      mean_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                 std::memory_order_relaxed);
  }
}

namespace {
id<MTLBuffer> EncodePrefixSum(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                              std::uint32_t count) {
  CHECK_GT(count, 0);
  auto groups = (std::uint64_t{count} + 255) / 256;
  auto output = Allocate(rt, std::size_t{count} * sizeof(std::uint32_t));
  auto totals = Allocate(rt, groups * sizeof(std::uint32_t));
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:rt.prefix_u32_pipeline];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:output offset:0 atIndex:1];
  [e setBuffer:totals offset:0 atIndex:2];
  [e setBytes:&count length:sizeof(count) atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  if (groups > 1) {
    auto upper = EncodePrefixSum(rt, command, totals, static_cast<std::uint32_t>(groups));
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.prefix_add_pipeline];
    [e setBuffer:output offset:0 atIndex:0];
    [e setBuffer:upper offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
  }
  return output;
}
struct SortShape {
  std::uint64_t rows, stride0, stride1;
  std::uint32_t targets, blocks;
};
static_assert(sizeof(SortShape) == 32);
id<MTLBuffer> EncodeColumnSort(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                               SortShape shape, bool descending = false) {
  auto size = shape.rows * shape.targets;
  CHECK_LE(size, std::numeric_limits<std::uint32_t>::max());
  auto source = Allocate(rt, size * 2 * sizeof(std::uint32_t));
  auto target = Allocate(rt, source.length);
  auto ranks = Allocate(rt, size * sizeof(std::uint32_t));
  auto count_size = std::uint64_t{shape.targets} * 16 * shape.blocks;
  CHECK_LE(count_size, std::numeric_limits<std::uint32_t>::max());
  auto counts = Allocate(rt, count_size * sizeof(std::uint32_t));
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:descending ? rt.roc_sort_init_pipeline : rt.sort_init_pipeline];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:source offset:0 atIndex:1];
  [e setBytes:&shape length:sizeof(shape) atIndex:2];
  [e dispatchThreads:MTLSizeMake(shape.rows, shape.targets, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  for (std::uint32_t shift = 0; shift < 32; shift += 4) {
    struct {
      std::uint32_t rows, targets, blocks, shift;
    } p{static_cast<std::uint32_t>(shape.rows), shape.targets, shape.blocks, shift};
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.radix_rank_pipeline];
    [e setBuffer:source offset:0 atIndex:0];
    [e setBuffer:ranks offset:0 atIndex:1];
    [e setBuffer:counts offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(shape.blocks, shape.targets, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    auto prefix = EncodePrefixSum(rt, command, counts, static_cast<std::uint32_t>(count_size));
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.radix_scatter_pipeline];
    [e setBuffer:source offset:0 atIndex:0];
    [e setBuffer:ranks offset:0 atIndex:1];
    [e setBuffer:prefix offset:0 atIndex:2];
    [e setBuffer:target offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(shape.rows, shape.targets, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    std::swap(source, target);
  }
  return source;
}
}  // namespace

namespace {
id<MTLBuffer> EncodeDoublePrefix(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                                 std::uint32_t rows, std::uint32_t targets) {
  auto blocks = (std::uint64_t{rows} + 255) / 256;
  auto output = Allocate(rt, std::size_t{rows} * targets * 2 * sizeof(double));
  auto totals = Allocate(rt, blocks * targets * 2 * sizeof(double));
  std::uint32_t shape[2]{rows, targets};
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:rt.prefix_double_pair_pipeline];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:output offset:0 atIndex:1];
  [e setBuffer:totals offset:0 atIndex:2];
  [e setBytes:&shape length:sizeof(shape) atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(blocks, targets, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  if (blocks > 1) {
    auto upper = EncodeDoublePrefix(rt, command, totals, blocks, targets);
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.prefix_double_pair_add_pipeline];
    [e setBuffer:output offset:0 atIndex:0];
    [e setBuffer:upper offset:0 atIndex:1];
    [e setBytes:&shape length:sizeof(shape) atIndex:2];
    [e dispatchThreads:MTLSizeMake(rows, targets, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
  }
  return output;
}
}  // namespace
namespace {
void AUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
         Span<float const> weights, bool multiclass, bool precision_recall,
         Span<ROCAUCSums> output) {
  auto rows = predictions.Shape(0), targets = predictions.Shape(1);
  CHECK_EQ(labels.Shape(0), rows);
  CHECK_EQ(labels.Shape(1), multiclass ? 1 : targets);
  CHECK_EQ(output.size(), targets);
  CHECK(weights.empty() || weights.size() == rows);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(targets, 65535);
  if (!rows || !targets) {
    std::fill(output.begin(), output.end(),
              precision_recall ? ROCAUCSums{1, 1, std::numeric_limits<double>::quiet_NaN()}
                               : ROCAUCSums{});
    return;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, predictions.Values().data(), predictions.Values().size_bytes());
    auto response = Upload(rt, labels.Values().data(), labels.Values().size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    SortShape shape{rows, predictions.Stride(0), predictions.Stride(1),
                    static_cast<std::uint32_t>(targets),
                    static_cast<std::uint32_t>((rows + 255) / 256)};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto sorted = EncodeColumnSort(rt, command, input, shape, true);
    auto counts = Allocate(rt, rows * targets * 2 * sizeof(double));
    struct {
      std::uint64_t rows, label_stride0, label_stride1;
      std::uint32_t targets, blocks, weighted, multiclass;
    } p{rows,
        labels.Stride(0),
        labels.Stride(1),
        shape.targets,
        shape.blocks,
        static_cast<std::uint32_t>(!weights.empty()),
        static_cast<std::uint32_t>(multiclass)};
    static_assert(sizeof(p) == 40);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.roc_counts_pipeline];
    [e setBuffer:sorted offset:0 atIndex:0];
    [e setBuffer:response offset:0 atIndex:1];
    [e setBuffer:weight offset:0 atIndex:2];
    [e setBuffer:counts offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(rows, targets, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    auto prefix = EncodeDoublePrefix(rt, command, counts, rows, targets);
    auto partial = Allocate(rt, shape.blocks * targets * 2 * sizeof(double));
    auto area = Allocate(rt, targets * 2 * sizeof(double));
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:precision_recall ? rt.pr_area_pipeline : rt.roc_area_pipeline];
    [e setBuffer:sorted offset:0 atIndex:0];
    [e setBuffer:prefix offset:0 atIndex:1];
    [e setBuffer:partial offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(shape.blocks, targets, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.gradient_sum_finish_pipeline];
    [e setBuffer:partial offset:0 atIndex:0];
    [e setBuffer:area offset:0 atIndex:1];
    [e setBytes:&shape length:sizeof(shape) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(targets, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    auto sums = static_cast<double const*>(prefix.contents);
    auto areas = static_cast<double const*>(area.contents);
    for (std::size_t t = 0; t < targets; ++t) {
      auto offset = 2 * ((t + 1) * rows - 1);
      output[t] = {sums[offset], sums[offset + 1], areas[2 * t]};
      bool invalid = output[t].fp <= 0 || output[t].tp <= 0;
      if (precision_recall) {
        output[t] = {1, 1, invalid ? std::numeric_limits<double>::quiet_NaN() : output[t].area};
      } else if (invalid)
        output[t] = {};
    }
    (precision_recall ? pr_auc_dispatches : roc_auc_dispatches)
        .fetch_add(1, std::memory_order_relaxed);
    radix_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0) {
      auto ns = static_cast<std::uint64_t>(elapsed * 1e9);
      (precision_recall ? pr_auc_gpu_time_ns : roc_auc_gpu_time_ns)
          .fetch_add(ns, std::memory_order_relaxed);
      metric_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
    }
  }
}

}  // namespace
void ROCAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
            Span<float const> weights, bool multiclass, Span<ROCAUCSums> output) {
  AUC(predictions, labels, weights, multiclass, false, output);
}
void PRAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
           Span<float const> weights, bool multiclass, Span<ROCAUCSums> output) {
  AUC(predictions, labels, weights, multiclass, true, output);
}

void Quantiles(linalg::MatrixView<float const> values, Span<float const> weights,
               Span<QuantileRequest const> requests, Span<float> output) {
  static_assert(sizeof(QuantileRequest) == 16);
  CHECK_EQ(requests.size(), output.size());
  CHECK_LE(requests.size(), std::numeric_limits<std::uint32_t>::max());
  if (requests.empty()) return;
  auto rows = values.Shape(0), targets = values.Shape(1);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(targets, 65535);
  CHECK(weights.empty() || weights.size() == rows);
  for (auto const& request : requests) {
    CHECK_LT(request.target, targets);
    CHECK_GE(request.alpha, 0);
    CHECK_LE(request.alpha, 1);
  }
  if (!rows) {
    std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
    return;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, values.Values().data(), values.Values().size_bytes());
    SortShape shape{rows, values.Stride(0), values.Stride(1), static_cast<std::uint32_t>(targets),
                    static_cast<std::uint32_t>((rows + 255) / 256)};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto sorted = EncodeColumnSort(rt, command, input, shape);
    auto cdf = Allocate(rt, weights.empty() ? 4 : rows * targets * sizeof(std::uint32_t));
    if (!weights.empty()) {
      auto weight = Upload(rt, weights.data(), weights.size_bytes());
      auto e = [command computeCommandEncoder];
      CHECK(e);
      auto n = static_cast<std::uint32_t>(rows);
      [e setComputePipelineState:rt.quantile_cdf_pipeline];
      [e setBuffer:sorted offset:0 atIndex:0];
      [e setBuffer:weight offset:0 atIndex:1];
      [e setBuffer:cdf offset:0 atIndex:2];
      [e setBytes:&n length:sizeof(n) atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(targets, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    auto task = Upload(rt, requests.data(), requests.size_bytes());
    auto result = Allocate(rt, output.size_bytes());
    struct {
      std::uint32_t rows, targets, weighted, requests;
    } p{static_cast<std::uint32_t>(rows), shape.targets,
        static_cast<std::uint32_t>(!weights.empty()), static_cast<std::uint32_t>(requests.size())};
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.quantile_select_pipeline];
    [e setBuffer:sorted offset:0 atIndex:0];
    [e setBuffer:input offset:0 atIndex:1];
    [e setBuffer:cdf offset:0 atIndex:2];
    [e setBuffer:task offset:0 atIndex:3];
    [e setBuffer:result offset:0 atIndex:4];
    [e setBytes:&p length:sizeof(p) atIndex:5];
    [e setBytes:&shape length:sizeof(shape) atIndex:6];
    [e dispatchThreads:MTLSizeMake(requests.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(output.data(), result.contents, output.size_bytes());
    radix_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
    quantile_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      quantile_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                     std::memory_order_relaxed);
  }
}

void BinEntries(Span<Entry const> entries, Span<FeatureType const> feature_types,
                GHistIndexMatrix* output) {
  CHECK(output);
  CHECK_EQ(entries.size(), output->index.Size());
  if (entries.empty()) return;
  static_assert(sizeof(Entry) == 8 && offsetof(Entry, fvalue) == 4);
  CHECK_LE(entries.size(), std::numeric_limits<std::uint32_t>::max());
  auto const& cuts = output->cut;
  CHECK(!cuts.Ptrs().empty());
  auto features = cuts.Ptrs().size() - 1;
  CHECK(feature_types.empty() || feature_types.size() == features);
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, entries.data(), entries.size_bytes());
    auto ptrs = Upload(rt, cuts.Ptrs().data(), cuts.Ptrs().size() * sizeof(std::uint32_t));
    auto values = Upload(rt, cuts.Values().data(), cuts.Values().size() * sizeof(float));
    std::vector<std::uint8_t> is_cat(features, 0);
    for (std::size_t f = 0; f < feature_types.size(); ++f)
      is_cat[f] = feature_types[f] == FeatureType::kCategorical;
    auto categories = Upload(rt, is_cat.data(), is_cat.size());
    auto bytes = entries.size() * output->index.GetBinTypeSize();
    auto bins = Allocate(rt, bytes);
    auto counts = Allocate(rt, output->hit_count.size() * sizeof(std::uint32_t));
    auto status = Allocate(rt, sizeof(std::uint32_t));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto clear = [command blitCommandEncoder];
    CHECK(clear);
    [clear fillBuffer:counts range:NSMakeRange(0, counts.length) value:0];
    [clear fillBuffer:status range:NSMakeRange(0, status.length) value:0];
    [clear endEncoding];
    struct {
      std::uint64_t count;
      std::uint32_t features, width, dense, padding;
    } p{entries.size(), static_cast<std::uint32_t>(features),
        static_cast<std::uint32_t>(output->index.GetBinTypeSize()),
        static_cast<std::uint32_t>(output->IsDense()), 0};
    static_assert(sizeof(p) == 24);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.binning_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:ptrs offset:0 atIndex:1];
    [e setBuffer:values offset:0 atIndex:2];
    [e setBuffer:categories offset:0 atIndex:3];
    [e setBuffer:bins offset:0 atIndex:4];
    [e setBuffer:counts offset:0 atIndex:5];
    [e setBuffer:status offset:0 atIndex:6];
    [e setBytes:&p length:sizeof(p) atIndex:7];
    [e dispatchThreads:MTLSizeMake(entries.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    auto flags = *static_cast<std::uint32_t const*>(status.contents);
    CHECK(!(flags & 1)) << error::InfInData();
    CHECK_EQ(flags, 0) << "Metal bin assignment found a feature without cut points.";
    std::memcpy(output->index.data<std::uint8_t>(), bins.contents, bytes);
    auto counts_data = static_cast<std::uint32_t const*>(counts.contents);
    for (std::size_t i = 0; i < output->hit_count.size(); ++i)
      output->hit_count[i] += counts_data[i];
    binning_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      binning_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                    std::memory_order_relaxed);
  }
}

void UniformSample(linalg::MatrixView<GradientPair> gradients, float probability,
                   std::uint64_t seed, std::uint32_t threads) {
  CHECK_GT(threads, 0);
  CHECK_GE(probability, 0.0f);
  CHECK_LE(probability, 1.0f);
  auto rows = gradients.Shape(0);
  if (!rows) return;
  CHECK_GT(gradients.Shape(1), 0);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto values = gradients.Values();
    auto data = Upload(rt, values.data(), values.size_bytes());
    struct {
      std::uint64_t rows, stride0, stride1, seed;
      std::uint32_t targets, threads;
      float probability;
      std::uint32_t padding;
    } p{rows,
        gradients.Stride(0),
        gradients.Stride(1),
        seed,
        static_cast<std::uint32_t>(gradients.Shape(1)),
        threads,
        probability,
        0};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.uniform_sampling_pipeline];
    [e setBuffer:data offset:0 atIndex:0];
    [e setBytes:&p length:sizeof(p) atIndex:1];
    [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(values.data(), data.contents, values.size_bytes());
    sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      sampling_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                     std::memory_order_relaxed);
  }
}

std::array<double, 2> QuantileMetricSums(bool expectile, Span<float const> predictions,
                                         Span<float const> labels, Span<float const> weights,
                                         Span<float const> alpha, std::uint32_t targets) {
  CHECK_GT(targets, 0);
  CHECK(!alpha.empty());
  CHECK_EQ(labels.size() % targets, 0);
  CHECK_EQ(predictions.size(), labels.size() * alpha.size());
  CHECK_LE(predictions.size(), std::numeric_limits<std::uint32_t>::max());
  auto rows = labels.size() / targets;
  CHECK(weights.empty() || weights.size() == rows);
  if (!rows) return {0, 0};
  @autoreleasepool {
    auto& rt = GetRuntime();
    CHECK_EQ(rt.quantile_metric_pipeline.threadExecutionWidth, 32);
    auto pred = Upload(rt, predictions.data(), predictions.size_bytes());
    auto label = Upload(rt, labels.data(), labels.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto levels = Upload(rt, alpha.data(), alpha.size_bytes());
    auto blocks = (predictions.size() + 255) / 256;
    auto partial = Allocate(rt, blocks * 2 * sizeof(double));
    struct {
      std::uint64_t count;
      std::uint32_t targets, levels, weighted, expectile;
    } p{predictions.size(), targets, static_cast<std::uint32_t>(alpha.size()), !weights.empty(),
        expectile};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.quantile_metric_pipeline];
    [e setBuffer:pred offset:0 atIndex:0];
    [e setBuffer:label offset:0 atIndex:1];
    [e setBuffer:weight offset:0 atIndex:2];
    [e setBuffer:levels offset:0 atIndex:3];
    [e setBuffer:partial offset:0 atIndex:4];
    [e setBytes:&p length:sizeof(p) atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      metric_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                   std::memory_order_relaxed);
    auto values = static_cast<double const*>(partial.contents);
    std::array<double, 2> result{0, 0};
    for (std::size_t i = 0; i < blocks; ++i) {
      result[0] += values[2 * i];
      result[1] += values[2 * i + 1];
    }
    return result;
  }
}

namespace {
struct SurvivalCommand {
  id<MTLCommandBuffer> command;
  id<MTLBuffer> result, correction_count, correction_rows;
  std::size_t blocks;
};
SurvivalCommand EncodeSurvival(bool metric, std::uint32_t distribution, float scale, bool accuracy,
                               Span<float const> predictions, Span<float const> lower,
                               Span<float const> upper, Span<float const> weights) {
  CHECK_LE(distribution, 2);
  CHECK_EQ(predictions.size(), lower.size());
  CHECK_EQ(predictions.size(), upper.size());
  CHECK(weights.empty() || weights.size() == predictions.size());
  CHECK_LE(predictions.size(), std::numeric_limits<std::uint32_t>::max());
  auto& rt = GetRuntime();
  auto pred = Upload(rt, predictions.data(), predictions.size_bytes());
  auto lo = Upload(rt, lower.data(), lower.size_bytes());
  auto hi = Upload(rt, upper.data(), upper.size_bytes());
  auto w = Upload(rt, weights.data(), weights.size_bytes());
  auto blocks = (predictions.size() + 255) / 256;
  auto result = Allocate(
      rt, metric ? blocks * 2 * sizeof(double) : predictions.size() * sizeof(GradientPair));
  auto count = metric ? nil : Allocate(rt, 4);
  auto indices = metric ? nil : Allocate(rt, predictions.size() * sizeof(std::uint32_t));
  struct {
    std::uint64_t rows;
    std::uint32_t distribution, weighted, accuracy;
    float scale;
  } p{predictions.size(), distribution, !weights.empty(), accuracy, scale};
  static_assert(sizeof(p) == 24);
  auto command = [rt.queue commandBuffer];
  CHECK(command);
  if (!metric) {
    auto clear = [command blitCommandEncoder];
    CHECK(clear);
    [clear fillBuffer:count range:NSMakeRange(0, 4) value:0];
    [clear endEncoding];
  }
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:metric ? rt.aft_metric_pipeline : rt.aft_gradient_pipeline];
  [e setBuffer:pred offset:0 atIndex:0];
  [e setBuffer:lo offset:0 atIndex:1];
  [e setBuffer:hi offset:0 atIndex:2];
  [e setBuffer:w offset:0 atIndex:3];
  [e setBuffer:result offset:0 atIndex:4];
  [e setBytes:&p length:sizeof(p) atIndex:5];
  if (!metric) {
    [e setBuffer:count offset:0 atIndex:6];
    [e setBuffer:indices offset:0 atIndex:7];
  }
  [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  return {command, result, count, indices, blocks};
}
template <typename Distribution>
void CorrectSurvival(Span<std::uint32_t const> rows, float scale, Span<float const> predictions,
                     Span<float const> lower, Span<float const> upper, Span<float const> weights,
                     Span<GradientPair> output) {
  for (auto row : rows) {
    float gradient = static_cast<float>(
        AFTLoss<Distribution>::Gradient(lower[row], upper[row], predictions[row], scale));
    float hessian = static_cast<float>(
        AFTLoss<Distribution>::Hessian(lower[row], upper[row], predictions[row], scale));
    float weight = weights.empty() ? 1.0f : weights[row];
    output[row] = GradientPair{gradient * weight, hessian * weight};
  }
}
}  // namespace
void SurvivalGradients(std::uint32_t distribution, float scale, Span<float const> predictions,
                       Span<float const> lower, Span<float const> upper, Span<float const> weights,
                       Span<GradientPair> output) {
  CHECK_EQ(predictions.size(), output.size());
  if (predictions.empty()) return;
  @autoreleasepool {
    auto op = EncodeSurvival(false, distribution, scale, false, predictions, lower, upper, weights);
    CompleteOperation(op.command, &objective_dispatches);
    std::memcpy(output.data(), op.result.contents, output.size_bytes());
    auto count = *static_cast<std::uint32_t const*>(op.correction_count.contents);
    CHECK_LE(count, predictions.size());
    if (count) {
      auto start = Clock::now();
      Span<std::uint32_t const> rows{static_cast<std::uint32_t const*>(op.correction_rows.contents),
                                     count};
      switch (distribution) {
        case 0:
          CorrectSurvival<NormalDistribution>(rows, scale, predictions, lower, upper, weights,
                                              output);
          break;
        case 1:
          CorrectSurvival<LogisticDistribution>(rows, scale, predictions, lower, upper, weights,
                                                output);
          break;
        case 2:
          CorrectSurvival<ExtremeDistribution>(rows, scale, predictions, lower, upper, weights,
                                               output);
          break;
      }
      survival_cpu_time_ns.fetch_add(Elapsed(start), std::memory_order_relaxed);
      survival_cpu_corrections.fetch_add(count, std::memory_order_relaxed);
    }
    survival_gradient_dispatches.fetch_add(1, std::memory_order_relaxed);
  }
}
std::array<double, 2> SurvivalMetricSums(std::uint32_t distribution, float scale, bool accuracy,
                                         Span<float const> predictions, Span<float const> lower,
                                         Span<float const> upper, Span<float const> weights) {
  if (predictions.empty()) return {0, 0};
  @autoreleasepool {
    auto op =
        EncodeSurvival(true, distribution, scale, accuracy, predictions, lower, upper, weights);
    [op.command commit];
    [op.command waitUntilCompleted];
    CHECK(op.command.status == MTLCommandBufferStatusCompleted) << ErrorText(op.command.error);
    auto values = static_cast<double const*>(op.result.contents);
    std::array<double, 2> result{0, 0};
    for (std::size_t i = 0; i < op.blocks; ++i) {
      result[0] += values[2 * i];
      result[1] += values[2 * i + 1];
    }
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    survival_metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = op.command.GPUEndTime - op.command.GPUStartTime;
    if (elapsed > 0)
      metric_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                   std::memory_order_relaxed);
    return result;
  }
}
void SurvivalTransform(bool to_margin, Span<float> values) {
  if (values.empty()) return;
  CHECK_LE(values.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, values.data(), values.size_bytes());
    auto output = Allocate(rt, values.size_bytes());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    std::uint32_t kind = to_margin;
    [e setComputePipelineState:rt.aft_transform_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBytes:&kind length:sizeof(kind) atIndex:2];
    [e dispatchThreads:MTLSizeMake(values.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &transform_dispatches);
    std::memcpy(values.data(), output.contents, values.size_bytes());
  }
}

void AdaptiveGradients(AdaptiveObjective objective, Span<float const> predictions,
                       Span<float const> labels, Span<float const> weights, std::uint32_t targets,
                       Span<float const> alpha, Span<GradientPair> output) {
  CHECK_GT(targets, 0);
  CHECK_EQ(predictions.size() % targets, 0);
  CHECK_EQ(predictions.size(), output.size());
  CHECK_LE(predictions.size(), std::numeric_limits<std::uint32_t>::max());
  auto rows = predictions.size() / targets;
  if (!rows) return;
  CHECK(weights.empty() || weights.size() == rows);
  CHECK(labels.size() == rows || labels.size() == predictions.size());
  if (objective == AdaptiveObjective::kQuantile) CHECK_EQ(alpha.size(), targets);
  struct {
    std::uint64_t rows;
    std::uint32_t targets, label_targets, weighted, kind, blocks, padding;
  } p{rows,
      targets,
      static_cast<std::uint32_t>(labels.size() / rows),
      !weights.empty(),
      static_cast<std::uint32_t>(objective),
      static_cast<std::uint32_t>((rows + 255) / 256),
      0};
  @autoreleasepool {
    auto& rt = GetRuntime();
    CHECK_EQ(rt.residual_reduce_pipeline.threadExecutionWidth, 32);
    auto pred = Upload(rt, predictions.data(), predictions.size_bytes());
    auto label = Upload(rt, labels.data(), labels.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto levels = Upload(rt, alpha.data(), alpha.size_bytes());
    auto partial = Allocate(rt, std::size_t(p.blocks) * targets * 2 * sizeof(std::uint64_t));
    auto scale = Allocate(rt, targets * sizeof(float)), out = Allocate(rt, output.size_bytes());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.residual_reduce_pipeline];
    [e setBuffer:pred offset:0 atIndex:0];
    [e setBuffer:label offset:0 atIndex:1];
    [e setBuffer:weight offset:0 atIndex:2];
    [e setBuffer:partial offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(p.blocks, targets, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.residual_finish_pipeline];
    [e setBuffer:partial offset:0 atIndex:0];
    [e setBuffer:scale offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(targets, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [e endEncoding];
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.adaptive_pipeline];
    [e setBuffer:pred offset:0 atIndex:0];
    [e setBuffer:label offset:0 atIndex:1];
    [e setBuffer:weight offset:0 atIndex:2];
    [e setBuffer:scale offset:0 atIndex:3];
    [e setBuffer:levels offset:0 atIndex:4];
    [e setBuffer:out offset:0 atIndex:5];
    [e setBytes:&p length:sizeof(p) atIndex:6];
    [e dispatchThreads:MTLSizeMake(predictions.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &objective_dispatches);
    residual_scale_dispatches.fetch_add(1, std::memory_order_relaxed);
    std::memcpy(output.data(), out.contents, output.size_bytes());
  }
}

void Gradients(Objective objective, Span<float const> predictions, Span<float const> labels,
               Span<float const> weights, std::uint32_t targets, float parameter,
               float scale_pos_weight, Span<float const> alpha, Span<GradientPair> output) {
  CHECK_NE(targets, 0);
  CHECK_EQ(predictions.size(), output.size());
  CHECK_EQ(predictions.size() % targets, 0);
  auto rows = predictions.size() / targets;
  if (rows == 0) return;
  CHECK(labels.size() == rows || labels.size() == predictions.size());
  CHECK(weights.empty() || weights.size() == rows);
  CHECK_LE(predictions.size(), std::numeric_limits<std::uint32_t>::max());
  if (objective == Objective::kExpectile) CHECK_EQ(alpha.size(), targets);
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto queue = rt.queue;
    CHECK(queue);
    auto command = [queue commandBuffer];
    CHECK(command);
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    struct BoundMemory {
      id<MTLBuffer> buffer;
      std::size_t offset;
      bool shared;
    };
    std::uint64_t shared_input = 0, shared_output = 0, staged = 0;
    auto bind = [&](void const* data, std::size_t bytes, std::size_t alignment, bool is_output,
                    bool allow_shared = true) -> BoundMemory {
      auto view =
          allow_shared ? TryBorrowMemory(rt.device, data, bytes, alignment) : BorrowedMemory{};
      if (view.buffer) {
        (is_output ? shared_output : shared_input) += bytes;
        return {view.buffer, view.offset, true};
      }
      staged += bytes;
      return {is_output ? Allocate(rt, bytes) : Upload(rt, data, bytes), 0, false};
    };
    auto overlaps_output = [&](Span<float const> input) {
      return MemoryRangesOverlap(input.data(), input.size_bytes(), output.data(),
                                 output.size_bytes());
    };
    bool output_disjoint = !overlaps_output(predictions) && !overlaps_output(labels) &&
                           !overlaps_output(weights) && !overlaps_output(alpha);
    auto pred = bind(predictions.data(), predictions.size_bytes(), 4, false);
    auto label = bind(labels.data(), labels.size_bytes(), 4, false);
    auto weight = bind(weights.data(), weights.size_bytes(), 4, false);
    auto levels = bind(alpha.data(), alpha.size_bytes(), 4, false);
    auto out = bind(output.data(), output.size_bytes(), 8, true, output_disjoint);
    struct {
      std::uint64_t n;
      std::uint32_t targets, kind, weighted, label_targets;
      float parameter, scale_pos_weight;
    } p{predictions.size(),
        targets,
        static_cast<std::uint32_t>(objective),
        !weights.empty(),
        static_cast<std::uint32_t>(labels.size() / rows),
        parameter,
        scale_pos_weight};
    [encoder setComputePipelineState:rt.objective_pipeline];
    [encoder setBuffer:pred.buffer offset:pred.offset atIndex:0];
    [encoder setBuffer:label.buffer offset:label.offset atIndex:1];
    [encoder setBuffer:weight.buffer offset:weight.offset atIndex:2];
    [encoder setBuffer:levels.buffer offset:levels.offset atIndex:3];
    [encoder setBuffer:out.buffer offset:out.offset atIndex:4];
    [encoder setBytes:&p length:sizeof(p) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(predictions.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    CompleteOperation(command, &objective_dispatches);
    if (!out.shared) std::memcpy(output.data(), out.buffer.contents, output.size_bytes());
    objective_shared_input_bytes.fetch_add(shared_input, std::memory_order_relaxed);
    objective_shared_output_bytes.fetch_add(shared_output, std::memory_order_relaxed);
    objective_staging_bytes.fetch_add(staged, std::memory_order_relaxed);
  }
}
void TransformPredictions(Transform transform, Span<float> values, std::uint32_t targets) {
  CHECK_NE(targets, 0);
  CHECK_EQ(values.size() % targets, 0);
  if (values.empty() || transform == Transform::kIdentity) return;
  CHECK_LE(values.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto queue = rt.queue;
    CHECK(queue);
    auto command = [queue commandBuffer];
    CHECK(command);
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    auto in = Upload(rt, values.data(), values.size_bytes());
    auto out = Allocate(rt, values.size_bytes());
    auto kind = static_cast<std::uint32_t>(transform);
    auto n = kind >= 4 ? values.size() / targets : values.size();
    struct {
      std::uint64_t n;
      std::uint32_t targets, kind;
    } p{n, targets, kind};
    [encoder setComputePipelineState:rt.transform_pipeline];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&p length:sizeof(p) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    CompleteOperation(command, &transform_dispatches);
    auto count = transform == Transform::kArgmax ? n : values.size();
    std::memcpy(values.data(), out.contents, count * sizeof(float));
  }
}

namespace {
class MetalPrediction final : public Prediction {
  Runtime& rt_;
  id<MTLCommandQueue> queue_;
  id<MTLBuffer> nodes_, trees_, categories_, leaves_;
  std::uint32_t n_trees_;

 public:
  explicit MetalPrediction(PredictionModel const& model) : rt_{GetRuntime()} {
    static_assert(sizeof(PredictionNode) == 40 && sizeof(PredictionTree) == 16);
    CHECK_LE(model.trees.size(), std::numeric_limits<std::uint32_t>::max());
    n_trees_ = model.trees.size();
    @autoreleasepool {
      queue_ = rt_.queue;
      CHECK(queue_);
      nodes_ = Upload(rt_, model.nodes.data(), model.nodes.size() * sizeof(PredictionNode));
      trees_ = Upload(rt_, model.trees.data(), model.trees.size() * sizeof(PredictionTree));
      categories_ =
          Upload(rt_, model.categories.data(), model.categories.size() * sizeof(std::uint32_t));
      leaves_ = Upload(rt_, model.leaves.data(), model.leaves.size() * sizeof(float));
    }
  }
  void Predict(Span<float const> data, std::size_t rows, std::uint32_t features,
               std::uint32_t groups, Span<float> predictions, bool leaf) override {
    CHECK_EQ(data.size(), rows * features);
    CHECK_EQ(predictions.size(), rows * groups);
    CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
    if (!rows || !n_trees_) return;
    @autoreleasepool {
      auto input = Upload(rt_, data.data(), data.size_bytes());
      auto output = Upload(rt_, predictions.data(), predictions.size_bytes());
      auto command = [queue_ commandBuffer];
      CHECK(command);
      auto encoder = [command computeCommandEncoder];
      CHECK(encoder);
      struct {
        std::uint64_t rows;
        std::uint32_t features, groups, trees, padding;
      } p{rows, features, groups, n_trees_, static_cast<std::uint32_t>(leaf)};
      [encoder setComputePipelineState:rt_.prediction_pipeline];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:nodes_ offset:0 atIndex:1];
      [encoder setBuffer:trees_ offset:0 atIndex:2];
      [encoder setBuffer:categories_ offset:0 atIndex:3];
      [encoder setBuffer:leaves_ offset:0 atIndex:4];
      [encoder setBuffer:output offset:0 atIndex:5];
      [encoder setBytes:&p length:sizeof(p) atIndex:6];
      [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      std::memcpy(predictions.data(), output.contents, predictions.size_bytes());
      prediction_dispatches.fetch_add(1, std::memory_order_relaxed);
      auto elapsed = command.GPUEndTime - command.GPUStartTime;
      if (elapsed > 0)
        prediction_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                         std::memory_order_relaxed);
    }
  }
};
}  // namespace
std::unique_ptr<Prediction> MakePrediction(PredictionModel const& model) {
  return std::make_unique<MetalPrediction>(model);
}

void AddLeafValues(Span<std::int32_t const> ids, Span<float const> weights, std::uint32_t targets,
                   std::size_t stride, Span<float> output) {
  if (ids.empty()) return;
  CHECK_NE(targets, 0);
  CHECK_EQ(weights.size() % targets, 0);
  CHECK_GE(stride, targets);
  CHECK_GE(output.size(), (ids.size() - 1) * stride + targets);
  CHECK_LE(ids.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto queue = rt.queue;
    CHECK(queue);
    auto command = [queue commandBuffer];
    CHECK(command);
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    auto input = Upload(rt, ids.data(), ids.size_bytes());
    auto leaves = Upload(rt, weights.data(), weights.size_bytes());
    auto out = Upload(rt, output.data(), output.size_bytes());
    struct {
      std::uint64_t rows, stride;
      std::uint32_t targets, padding;
    } p{ids.size(), stride, targets, 0};
    [encoder setComputePipelineState:rt.leaf_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:leaves offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:&p length:sizeof(p) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(ids.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(output.data(), out.contents, output.size_bytes());
    leaf_update_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      prediction_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                       std::memory_order_relaxed);
  }
}

std::array<double, 2> MetricSums(char const* name, Span<float const> predictions,
                                 Span<float const> labels, Span<float const> weights,
                                 std::uint32_t targets, float parameter) {
  std::uint32_t kind;
  if (!strcmp(name, "rmse"))
    kind = 0;
  else if (!strcmp(name, "rmsle"))
    kind = 1;
  else if (!strcmp(name, "mae"))
    kind = 2;
  else if (!strcmp(name, "mape"))
    kind = 3;
  else if (!strcmp(name, "logloss"))
    kind = 4;
  else if (!strncmp(name, "error", 5))
    kind = 5;
  else if (!strcmp(name, "poisson-nloglik"))
    kind = 6;
  else if (!strcmp(name, "gamma-deviance"))
    kind = 7;
  else if (!strcmp(name, "gamma-nloglik"))
    kind = 8;
  else if (!strncmp(name, "tweedie-nloglik", 14))
    kind = 9;
  else if (!strcmp(name, "mphe"))
    kind = 10;
  else if (!strcmp(name, "mlogloss"))
    kind = 11;
  else if (!strcmp(name, "merror"))
    kind = 12;
  else {
    LOG(FATAL) << "Missing Metal metric: " << name;
    return {};
  }
  if (labels.empty()) return {0.0, 0.0};
  CHECK_NE(targets, 0);
  bool multi = kind >= 11;
  auto rows = multi ? labels.size() : labels.size() / targets;
  CHECK_EQ(predictions.size(), multi ? labels.size() * targets : labels.size());
  CHECK(weights.empty() || weights.size() == rows);
  CHECK_LE(labels.size(), std::numeric_limits<std::uint32_t>::max());
  if (multi)
    for (auto y : labels) {
      CHECK(std::isfinite(y) && y >= 0 && y < targets)
          << "MultiClassEvaluation: label must be in [0, num_class).";
    }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto queue = rt.queue;
    CHECK(queue);
    auto command = [queue commandBuffer];
    CHECK(command);
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    auto pred = Upload(rt, predictions.data(), predictions.size_bytes());
    auto label = Upload(rt, labels.data(), labels.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto groups = (labels.size() + 255) / 256;
    auto partial = Allocate(rt, groups * 4 * sizeof(float));
    struct {
      std::uint64_t n;
      std::uint32_t targets, kind, weighted, padding;
      float parameter, reserved;
    } p{labels.size(), targets, kind, !weights.empty(), 0, parameter, 0};
    [encoder setComputePipelineState:rt.metric_pipeline];
    [encoder setBuffer:pred offset:0 atIndex:0];
    [encoder setBuffer:label offset:0 atIndex:1];
    [encoder setBuffer:weight offset:0 atIndex:2];
    [encoder setBuffer:partial offset:0 atIndex:3];
    [encoder setBytes:&p length:sizeof(p) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      metric_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                   std::memory_order_relaxed);
    auto values = static_cast<float const*>(partial.contents);
    std::array<double, 2> sum{0.0, 0.0};
    for (std::size_t i = 0; i < groups; ++i)
      for (std::size_t k = 0; k < 2; ++k) {
        double scale = values[i * 4 + 2 + k];
        if (!(scale > 0.0) || std::isinf(scale)) scale = 1.0;
        sum[k] += static_cast<double>(values[i * 4 + k]) * scale;
      }
    return sum;
  }
}

void EvaluateSplits(Span<HistogramInput const> histograms, Span<SplitTask const> tasks,
                    SplitParameters const& parameters, std::vector<SplitResult>* results,
                    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace) {
  static_assert(sizeof(SplitTask) == 48 && sizeof(SplitResult) == 48 &&
                sizeof(SplitParameters) == 16);
  results->resize(tasks.size());
  auto bins = tasks.empty() ? 0 : tasks.back().end;
  bool categorical =
      std::any_of(tasks.begin(), tasks.end(), [](auto const& task) { return task.kind == 2; });
  sorted_categories->resize(categorical ? bins : 0);
  if (tasks.empty()) return;
  CHECK_LE(tasks.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    CHECK(workspace);
    auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto hist = GatherHistograms(rt, histograms, bins, command, scratch);
    auto input = Scratch(rt, &scratch->input, tasks.size_bytes(), tasks.data());
    auto output = Scratch(rt, &scratch->output, tasks.size() * sizeof(SplitResult));
    auto order = Scratch(rt, &scratch->order, (categorical ? bins : 0) * sizeof(std::uint32_t));
    auto weights = Scratch(rt, &scratch->weights, (categorical ? bins : 0) * sizeof(float));
    auto scans = Scratch(rt, &scratch->scans, bins * 4 * sizeof(std::uint64_t));
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt.split_pipeline];
    [encoder setBuffer:hist offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBuffer:order offset:0 atIndex:4];
    [encoder setBuffer:weights offset:0 atIndex:5];
    [encoder setBuffer:scans offset:0 atIndex:6];
    CHECK_EQ(rt.split_pipeline.threadExecutionWidth, 32);
    auto threads = std::min<NSUInteger>(128, rt.split_pipeline.maxTotalThreadsPerThreadgroup);
    threads -= threads % 32;
    CHECK_GT(threads, 0);
    [encoder dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(results->data(), output.contents, tasks.size() * sizeof(SplitResult));
    for (auto const& task : tasks)
      if (task.kind == 2) {
        std::memcpy(sorted_categories->data() + task.begin,
                    static_cast<std::uint32_t const*>(order.contents) + task.begin,
                    (task.end - task.begin) * sizeof(std::uint32_t));
      }
    split_dispatches.fetch_add(1, std::memory_order_relaxed);
    resident_split_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      split_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                  std::memory_order_relaxed);
  }
}
void EvaluateMultiSplits(Span<HistogramInput const> histograms, Span<MultiSplitTask const> tasks,
                         Span<GradientPairPrecise const> parents, Span<SplitBounds const> bounds,
                         SplitParameters const& parameters, std::vector<SplitResult>* results,
                         std::vector<GradientPairPrecise>* children,
                         std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace) {
  static_assert(sizeof(MultiSplitTask) == 40 && sizeof(SplitBounds) == 8);
  results->resize(tasks.size());
  auto bins = tasks.empty() ? 0 : tasks.back().end;
  bool categorical =
      std::any_of(tasks.begin(), tasks.end(), [](auto const& task) { return task.kind == 2; });
  sorted_categories->resize(categorical ? bins : 0);
  children->resize(parents.size() * 2);
  if (tasks.empty()) return;
  CHECK_EQ(parents.size(), bounds.size());
  @autoreleasepool {
    auto& rt = GetRuntime();
    CHECK(workspace);
    auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto hist = GatherHistograms(rt, histograms, bins, command, scratch);
    auto input = Scratch(rt, &scratch->input, tasks.size_bytes(), tasks.data());
    auto parent = Scratch(rt, &scratch->parent, parents.size_bytes(), parents.data());
    auto bound = Scratch(rt, &scratch->bounds, bounds.size_bytes(), bounds.data());
    auto output = Scratch(rt, &scratch->output, tasks.size() * sizeof(SplitResult));
    auto child = Scratch(rt, &scratch->children, children->size() * sizeof(GradientPairPrecise));
    auto order = Scratch(rt, &scratch->order, (categorical ? bins : 0) * sizeof(std::uint32_t));
    auto scores = Scratch(rt, &scratch->weights, (categorical ? bins : 0) * sizeof(std::uint64_t));
    auto scans = Scratch(rt, &scratch->scans, bins * 4 * sizeof(std::uint64_t));
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt.multi_split_pipeline];
    [encoder setBuffer:hist offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBuffer:order offset:0 atIndex:4];
    [encoder setBuffer:scores offset:0 atIndex:5];
    [encoder setBuffer:scans offset:0 atIndex:6];
    [encoder setBuffer:parent offset:0 atIndex:7];
    [encoder setBuffer:bound offset:0 atIndex:8];
    [encoder setBuffer:child offset:0 atIndex:9];
    CHECK_EQ(rt.multi_split_pipeline.threadExecutionWidth, 32);
    auto threads = std::min<NSUInteger>(128, rt.multi_split_pipeline.maxTotalThreadsPerThreadgroup);
    threads -= threads % 32;
    CHECK_GT(threads, 0);
    [encoder dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(results->data(), output.contents, tasks.size() * sizeof(SplitResult));
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto const& task = tasks[i];
      if (results->at(i).bin >= 0)
        std::memcpy(children->data() + 2 * task.parent_begin,
                    static_cast<GradientPairPrecise const*>(child.contents) + 2 * task.parent_begin,
                    2 * task.targets * sizeof(GradientPairPrecise));
      if (task.kind == 2)
        std::memcpy(sorted_categories->data() + task.begin,
                    static_cast<std::uint32_t const*>(order.contents) + task.begin,
                    ((task.end - task.begin) / task.targets) * sizeof(std::uint32_t));
    }
    split_dispatches.fetch_add(1, std::memory_order_relaxed);
    resident_split_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      split_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                  std::memory_order_relaxed);
  }
}
}  // namespace xgboost::common::metal
