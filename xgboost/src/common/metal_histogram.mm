/** Copyright 2026, XGBoost Contributors */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include "../data/gradient_index.h"
#include "metal_auc.h"
#include "metal_data.h"
#include "metal_gradient.h"
#include "metal_histogram.h"
#include "metal_leaf.h"
#include "metal_linear.h"
#include "metal_memory.h"
#include "metal_ops.h"
#include "metal_predictor.h"
#include "metal_quantile.h"
#include "metal_ranking.h"
#include "metal_reduction.h"
#include "metal_sampling.h"
#include "metal_shader.h"
#include "metal_shap.h"
#include "metal_sketch.h"
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
std::atomic<std::uint64_t> gradient_sampling_dispatches{0}, gradient_sampling_gpu_time_ns{0};
std::atomic<std::uint64_t> sketch_finalize_dispatches{0}, sketch_finalize_columns{0},
    sketch_finalize_gpu_time_ns{0};
std::atomic<std::uint64_t> sketch_weight_dispatches{0}, sketch_sorted_dispatches{0},
    sketch_update_dispatches{0}, sketch_ingest_gpu_time_ns{0};
std::atomic<std::uint64_t> gradient_copy_dispatches{0}, hessian_extract_dispatches{0},
    gradient_state_gpu_time_ns{0}, gradient_state_shared_input_bytes{0},
    gradient_state_shared_output_bytes{0}, gradient_state_staging_bytes{0};
std::atomic<std::uint64_t> expectile_init_dispatches{0}, prediction_init_dispatches{0},
    initialization_gpu_time_ns{0};
std::atomic<std::uint64_t> label_validation_dispatches{0}, label_validation_gpu_time_ns{0};
std::atomic<std::uint64_t> leaf_sum_init_dispatches{0}, leaf_sum_dispatches{0},
    leaf_weight_dispatches{0}, leaf_fitting_gpu_time_ns{0};
std::atomic<std::uint64_t> linear_gradient_dispatches{0}, linear_residual_dispatches{0},
    linear_gpu_time_ns{0};
std::atomic<std::uint64_t> shap_dispatches{0}, shap_interaction_dispatches{0}, shap_gpu_time_ns{0};
std::atomic<std::uint64_t> base_score_dispatches{0}, base_score_gpu_time_ns{0};
std::atomic<std::uint64_t> column_init_dispatches{0}, column_sampling_dispatches{0},
    column_sampling_gpu_time_ns{0};
std::atomic<std::uint64_t> histogram_conversion_dispatches{0}, histogram_subtraction_dispatches{0},
    resident_split_dispatches{0};
std::atomic<std::uint64_t> histogram_shared_gradient_bytes{0}, leaf_shared_gradient_bytes{0},
    leaf_shared_output_bytes{0};
std::atomic<std::uint64_t> objective_shared_input_bytes{0}, objective_shared_output_bytes{0},
    objective_staging_bytes{0};
std::atomic<std::uint64_t> roc_auc_dispatches{0}, roc_auc_gpu_time_ns{0};
std::atomic<std::uint64_t> pr_auc_dispatches{0}, pr_auc_gpu_time_ns{0};
std::atomic<std::uint64_t> group_sort_dispatches{0}, group_sort_gpu_time_ns{0};
std::atomic<std::uint64_t> ranking_metric_dispatches{0}, ranking_metric_gpu_time_ns{0};
std::atomic<std::uint64_t> ranking_roc_auc_dispatches{0}, ranking_roc_auc_gpu_time_ns{0};
std::atomic<std::uint64_t> ranking_pr_auc_dispatches{0}, ranking_pr_auc_gpu_time_ns{0};
std::atomic<std::uint64_t> ranking_gradient_dispatches{0}, ranking_gradient_gpu_time_ns{0},
    ranking_pairs{0}, ranking_pair_batches{0}, ranking_bias_dispatches{0};
std::atomic<std::uint64_t> ranking_label_dispatches{0}, ranking_label_gpu_time_ns{0},
    ranking_cache_dispatches{0}, ranking_cache_gpu_time_ns{0};
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
  id<MTLComputePipelineState> group_sort_small_pipeline, group_sort_medium_pipeline, group_sort_key_pipeline,
      group_sort_export_pipeline, ranking_score_pipeline;
  id<MTLComputePipelineState> ranking_labels_pipeline, ndcg_discount_pipeline, ndcg_ideal_pipeline;
  id<MTLComputePipelineState> ranking_map_stats_pipeline, ranking_pairs_pipeline,
      ranking_normalize_pipeline, ranking_gather_pipeline, ranking_mean_labels_pipeline,
      ranking_mean_sample_pipeline, ranking_mean_gather_pipeline, ranking_bias_sum_pipeline,
      ranking_bias_finish_pipeline, ranking_bias_update_pipeline, ranking_bias_pairs_pipeline,
      ranking_bias_dispatch_pipeline;
  id<MTLComputePipelineState> aft_gradient_pipeline, aft_metric_pipeline, aft_transform_pipeline;
  id<MTLComputePipelineState> uniform_sampling_pipeline, sketch_finalize_pipeline;
  id<MTLComputePipelineState> sketch_weight_pipeline, sketch_sorted_pipeline, sketch_update_pipeline;
  id<MTLComputePipelineState> gradient_state_pipeline;
  id<MTLComputePipelineState> expectile_init_pipeline, broadcast_targets_pipeline;
  id<MTLComputePipelineState> label_validation_pipeline;
  id<MTLComputePipelineState> base_score_pipeline, shap_pipeline;
  id<MTLComputePipelineState> linear_gradient_pipeline, linear_residual_pipeline;
  id<MTLComputePipelineState> leaf_sum_pipeline, leaf_sum_finish_pipeline, leaf_weight_pipeline;
  id<MTLComputePipelineState> column_iota_pipeline, column_rng_pipeline, column_keys_pipeline,
      column_select_pipeline, column_sort_small_pipeline, column_output_pipeline,
      column_sample_small_pipeline;
  id<MTLComputePipelineState> gradient_sampling_norm_pipeline, gradient_sampling_prefix_pipeline,
      gradient_sampling_threshold_pipeline, gradient_sampling_apply_pipeline;
  id<MTLComputePipelineState> ranking_roc_init_pipeline, ranking_roc_merge_pipeline,
      ranking_roc_score_pipeline;
  id<MTLComputePipelineState> ranking_pr_counts_pipeline, ranking_pr_area_pipeline,
      ranking_pr_score_pipeline, prefix_segmented_pair_pipeline, prefix_segmented_pair_add_pipeline;
  id<MTLComputePipelineState> gradient_sum_pipeline, gradient_sum_finish_pipeline,
      histogram_sum_pipeline, stump_weight_pipeline;
  id<MTLComputePipelineState> shared_pipeline, gradient_bounds_pipeline, gradient_scale_pipeline;
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
      sketch_finalize_pipeline = make_pipeline(@"xgb_sketch_finalize");
      sketch_weight_pipeline = make_pipeline(@"xgb_sketch_weights");
      sketch_sorted_pipeline = make_pipeline(@"xgb_sketch_sorted");
      sketch_update_pipeline = make_pipeline(@"xgb_sketch_update");
      gradient_state_pipeline = make_pipeline(@"xgb_gradient_state");
      expectile_init_pipeline = make_pipeline(@"xgb_expectile_init");
      broadcast_targets_pipeline = make_pipeline(@"xgb_broadcast_targets");
      label_validation_pipeline = make_pipeline(@"xgb_validate_labels");
      base_score_pipeline = make_pipeline(@"xgb_base_score");
      shap_pipeline = make_pipeline(@"xgb_shap");
      linear_gradient_pipeline = make_pipeline(@"xgb_linear_gradient");
      linear_residual_pipeline = make_pipeline(@"xgb_linear_residual");
      leaf_sum_pipeline = make_pipeline(@"xgb_leaf_sum");
      leaf_sum_finish_pipeline = make_pipeline(@"xgb_leaf_sum_finish");
      leaf_weight_pipeline = make_pipeline(@"xgb_leaf_weight");
      column_iota_pipeline = make_pipeline(@"xgb_column_iota");
      column_rng_pipeline = make_pipeline(@"xgb_column_rng");
      column_keys_pipeline = make_pipeline(@"xgb_column_keys");
      column_select_pipeline = make_pipeline(@"xgb_column_select");
      column_sort_small_pipeline = make_pipeline(@"xgb_column_sort_small");
      column_output_pipeline = make_pipeline(@"xgb_column_output");
      column_sample_small_pipeline = make_pipeline(@"xgb_column_sample_small");

      gradient_bounds_pipeline = make_pipeline(@"xgb_gradient_bounds");
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
      group_sort_small_pipeline = make_pipeline(@"xgb_group_sort_small");
      group_sort_medium_pipeline = make_pipeline(@"xgb_group_sort_medium");
      group_sort_key_pipeline = make_pipeline(@"xgb_group_sort_key");
      group_sort_export_pipeline = make_pipeline(@"xgb_group_sort_export");
      ranking_score_pipeline = make_pipeline(@"xgb_ranking_score");
      ranking_labels_pipeline = make_pipeline(@"xgb_ranking_labels");
      ndcg_discount_pipeline = make_pipeline(@"xgb_ndcg_discount");
      ndcg_ideal_pipeline = make_pipeline(@"xgb_ndcg_ideal");
      ranking_map_stats_pipeline = make_pipeline(@"xgb_ranking_map_stats");
      ranking_pairs_pipeline = make_pipeline(@"xgb_ranking_pairs");
      ranking_normalize_pipeline = make_pipeline(@"xgb_ranking_normalize");
      ranking_gather_pipeline = make_pipeline(@"xgb_ranking_gather");
      ranking_mean_labels_pipeline = make_pipeline(@"xgb_ranking_mean_labels");
      ranking_mean_sample_pipeline = make_pipeline(@"xgb_ranking_mean_sample");
      ranking_mean_gather_pipeline = make_pipeline(@"xgb_ranking_mean_gather");
      ranking_bias_sum_pipeline = make_pipeline(@"xgb_ranking_bias_sum");
      ranking_bias_finish_pipeline = make_pipeline(@"xgb_ranking_bias_finish");
      ranking_bias_pairs_pipeline = make_pipeline(@"xgb_ranking_bias_pairs");
      ranking_bias_dispatch_pipeline = make_pipeline(@"xgb_ranking_bias_dispatch");
      ranking_bias_update_pipeline = make_pipeline(@"xgb_ranking_bias_update");
      gradient_sampling_norm_pipeline = make_pipeline(@"xgb_gradient_sampling_norm");
      gradient_sampling_prefix_pipeline = make_pipeline(@"xgb_gradient_sampling_prefix");
      gradient_sampling_threshold_pipeline = make_pipeline(@"xgb_gradient_sampling_threshold");
      gradient_sampling_apply_pipeline = make_pipeline(@"xgb_gradient_sampling_apply");
      ranking_roc_init_pipeline = make_pipeline(@"xgb_ranking_roc_init");
      ranking_roc_merge_pipeline = make_pipeline(@"xgb_ranking_roc_merge");
      ranking_roc_score_pipeline = make_pipeline(@"xgb_ranking_roc_score");
      ranking_pr_counts_pipeline = make_pipeline(@"xgb_ranking_pr_counts");
      ranking_pr_area_pipeline = make_pipeline(@"xgb_ranking_pr_area");
      ranking_pr_score_pipeline = make_pipeline(@"xgb_ranking_pr_score");
      prefix_segmented_pair_pipeline = make_pipeline(@"xgb_prefix_segmented_pair");
      prefix_segmented_pair_add_pipeline = make_pipeline(@"xgb_prefix_segmented_pair_add");
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
void CompleteOperation(id<MTLCommandBuffer> command, std::atomic<std::uint64_t>* counter,
                       std::atomic<std::uint64_t>* duration);
class MetalLeafSums final : public LeafSums {
 public:
  id<MTLBuffer> buffer;
  std::size_t leaves, targets;
  MetalLeafSums(std::size_t n_leaves, std::size_t n_targets)
      : leaves{n_leaves}, targets{n_targets} {
    CHECK_GT(targets, 0);
    CHECK_LE(targets, 65535);
    CHECK_LE(leaves, std::numeric_limits<std::uint32_t>::max());
    auto& rt = GetRuntime();
    CHECK_LE(leaves, rt.device.maxBufferLength / sizeof(GradientPairPrecise) / targets);
    @autoreleasepool {
      buffer = Allocate(rt, leaves * targets * sizeof(GradientPairPrecise));
      if (!leaves) return;
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      auto e = [command blitCommandEncoder];
      CHECK(e);
      [e fillBuffer:buffer
              range:NSMakeRange(0, leaves * targets * sizeof(GradientPairPrecise))
              value:0];
      [e endEncoding];
      CompleteOperation(command, &leaf_sum_init_dispatches, &leaf_fitting_gpu_time_ns);
    }
  }
  Span<GradientPairPrecise const> Data() const override {
    return {static_cast<GradientPairPrecise const*>(buffer.contents), leaves * targets};
  }
  void Weights(SplitParameters const& parameters, Span<SplitBounds const> bounds, float eta,
               Span<float> output) const override {
    CHECK_EQ(output.size(), leaves * targets);
    CHECK(bounds.empty() || bounds.size() == output.size());
    CHECK_LE(output.size(), std::numeric_limits<std::uint32_t>::max());
    if (output.empty()) return;
    @autoreleasepool {
      auto& rt = GetRuntime();
      auto limits = Upload(rt, bounds.data(), bounds.size_bytes());
      auto result = Allocate(rt, output.size_bytes());
      struct {
        SplitParameters regularization;
        float eta;
        std::uint32_t bounded;
      } p{parameters, eta, static_cast<std::uint32_t>(!bounds.empty())};
      static_assert(sizeof(p) == 24);
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.leaf_weight_pipeline];
      [e setBuffer:buffer offset:0 atIndex:0];
      [e setBuffer:limits offset:0 atIndex:1];
      [e setBuffer:result offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      [e dispatchThreads:MTLSizeMake(output.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &leaf_weight_dispatches, &leaf_fitting_gpu_time_ns);
      std::memcpy(output.data(), result.contents, output.size_bytes());
    }
  }
};

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
  void AddLeafGradients(Span<LeafSumSegment const> leaves,
                        linalg::MatrixView<GradientPair const> gradients,
                        LeafSums* output) override {
    CHECK(output);
    auto sums = static_cast<MetalLeafSums*>(output);
    CHECK_EQ(gradients.Shape(1), sums->targets);
    CHECK_LE(base_, gradients.Shape(0));
    CHECK_LE(size_, gradients.Shape(0) - base_);
    auto values = gradients.Values();
    if (gradients.Size()) {
      CHECK_GT(values.size(), 0);
      std::size_t last = 0;
      for (std::size_t d = 0; d < 2; ++d) {
        if (gradients.Stride(d)) {
          CHECK_LE(gradients.Shape(d) - 1, (values.size() - 1 - last) / gradients.Stride(d));
          last += (gradients.Shape(d) - 1) * gradients.Stride(d);
        }
      }
    }
    struct Block {
      std::uint64_t begin, end;
    };
    struct Segment {
      std::uint64_t begin, end;
      std::uint32_t output, padding;
    };
    std::vector<Block> blocks;
    std::vector<Segment> segments;
    std::vector<bool> seen(sums->leaves, false);
    for (auto const& leaf : leaves) {
      CHECK_LE(leaf.begin, leaf.end);
      CHECK_LE(leaf.end, size_);
      CHECK_LT(leaf.output_leaf, sums->leaves);
      CHECK(!seen[leaf.output_leaf]) << "Duplicate output leaf within a page.";
      seen[leaf.output_leaf] = true;
      auto begin = blocks.size();
      for (auto offset = leaf.begin; offset < leaf.end; offset += 4096) {
        blocks.push_back({offset, std::min<std::uint64_t>(leaf.end, offset + 4096)});
      }
      if (blocks.size() != begin) segments.push_back({begin, blocks.size(), leaf.output_leaf, 0});
    }
    if (blocks.empty()) return;
    CHECK_LE(blocks.size(), std::numeric_limits<std::uint32_t>::max());
    CHECK_LE(segments.size(), std::numeric_limits<std::uint32_t>::max());
    CHECK_LE(blocks.size(),
             rt_.device.maxBufferLength / sizeof(GradientPairPrecise) / sums->targets);
    @autoreleasepool {
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      this->EnsureReady(command);
      auto g = TryBorrowMemory(rt_.device, values.data(), values.size_bytes(), 8);
      if (!g.buffer || MemoryRangesOverlap(values.data(), values.size_bytes(), sums->Data().data(),
                                           sums->Data().size_bytes())) {
        g = {Upload(rt_, values.data(), values.size_bytes()), 0};
      }
      auto tasks = Upload(rt_, blocks.data(), blocks.size() * sizeof(Block));
      auto ranges = Upload(rt_, segments.data(), segments.size() * sizeof(Segment));
      auto partial = Allocate(rt_, blocks.size() * sums->targets * sizeof(GradientPairPrecise));
      struct {
        std::uint64_t stride0, stride1;
        std::uint32_t targets, padding;
      } p{gradients.Stride(0), gradients.Stride(1), static_cast<std::uint32_t>(sums->targets), 0};
      static_assert(sizeof(p) == 24 && sizeof(Block) == 16 && sizeof(Segment) == 24);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.leaf_sum_pipeline];
      [e setBuffer:rows_ offset:0 atIndex:0];
      [e setBuffer:g.buffer offset:g.offset atIndex:1];
      [e setBuffer:tasks offset:0 atIndex:2];
      [e setBuffer:partial offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake(blocks.size(), sums->targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.leaf_sum_finish_pipeline];
      [e setBuffer:partial offset:0 atIndex:0];
      [e setBuffer:ranges offset:0 atIndex:1];
      [e setBuffer:sums->buffer offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(segments.size(), sums->targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &leaf_sum_dispatches, &leaf_fitting_gpu_time_ns);
      this->MarkReady();
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
      block_buffer_, gradient_partial_, gradient_bounds_, gradient_scale_;
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
    std::uint64_t count = gradients.size();
    auto blocks = (count + 255) / 256;
    EnsureBuffer(rt_, &gradient_partial_, 2 * blocks * sizeof(GradientPairPrecise));
    EnsureBuffer(rt_, &gradient_bounds_, 2 * sizeof(GradientPairPrecise));
    EnsureBuffer(rt_, &gradient_scale_, sizeof(GradientScale));
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt_.gradient_bounds_pipeline];
    [encoder setBuffer:input.buffer offset:input.offset atIndex:0];
    [encoder setBuffer:gradient_partial_ offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    struct {
      std::uint64_t rows, stride0, stride1;
      std::uint32_t targets, blocks;
    } reduction{count, 0, 0, 2, static_cast<std::uint32_t>(blocks)};
    encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt_.gradient_sum_finish_pipeline];
    [encoder setBuffer:gradient_partial_ offset:0 atIndex:0];
    [encoder setBuffer:gradient_bounds_ offset:0 atIndex:1];
    [encoder setBytes:&reduction length:sizeof(reduction) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(2, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt_.gradient_scale_pipeline];
    [encoder setBuffer:gradient_bounds_ offset:0 atIndex:0];
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
          pr_auc_gpu_time_ns.load(std::memory_order_relaxed),
          group_sort_dispatches.load(std::memory_order_relaxed),
          group_sort_gpu_time_ns.load(std::memory_order_relaxed),
          ranking_metric_dispatches.load(std::memory_order_relaxed),
          ranking_metric_gpu_time_ns.load(std::memory_order_relaxed),
          ranking_label_dispatches.load(std::memory_order_relaxed),
          ranking_label_gpu_time_ns.load(std::memory_order_relaxed),
          ranking_cache_dispatches.load(std::memory_order_relaxed),
          ranking_cache_gpu_time_ns.load(std::memory_order_relaxed),
          ranking_gradient_dispatches.load(std::memory_order_relaxed),
          ranking_gradient_gpu_time_ns.load(std::memory_order_relaxed),
          ranking_pairs.load(std::memory_order_relaxed),
          ranking_pair_batches.load(std::memory_order_relaxed),
          ranking_bias_dispatches.load(std::memory_order_relaxed),
          gradient_sampling_dispatches.load(std::memory_order_relaxed),
          gradient_sampling_gpu_time_ns.load(std::memory_order_relaxed),
          ranking_roc_auc_dispatches.load(std::memory_order_relaxed),
          ranking_roc_auc_gpu_time_ns.load(std::memory_order_relaxed),
          ranking_pr_auc_dispatches.load(std::memory_order_relaxed),
          ranking_pr_auc_gpu_time_ns.load(std::memory_order_relaxed),
          column_init_dispatches.load(std::memory_order_relaxed),
          column_sampling_dispatches.load(std::memory_order_relaxed),
          column_sampling_gpu_time_ns.load(std::memory_order_relaxed),
          sketch_finalize_dispatches.load(std::memory_order_relaxed),
          sketch_finalize_columns.load(std::memory_order_relaxed),
          sketch_finalize_gpu_time_ns.load(std::memory_order_relaxed),
          expectile_init_dispatches.load(std::memory_order_relaxed),
          prediction_init_dispatches.load(std::memory_order_relaxed),
          initialization_gpu_time_ns.load(std::memory_order_relaxed),
          label_validation_dispatches.load(std::memory_order_relaxed),
          label_validation_gpu_time_ns.load(std::memory_order_relaxed),
          base_score_dispatches.load(std::memory_order_relaxed),
          base_score_gpu_time_ns.load(std::memory_order_relaxed),
          shap_dispatches.load(std::memory_order_relaxed),
          shap_interaction_dispatches.load(std::memory_order_relaxed),
          shap_gpu_time_ns.load(std::memory_order_relaxed),
          leaf_sum_init_dispatches.load(std::memory_order_relaxed),
          leaf_sum_dispatches.load(std::memory_order_relaxed),
          leaf_weight_dispatches.load(std::memory_order_relaxed),
          leaf_fitting_gpu_time_ns.load(std::memory_order_relaxed),
          linear_gradient_dispatches.load(std::memory_order_relaxed),
          linear_residual_dispatches.load(std::memory_order_relaxed),
          linear_gpu_time_ns.load(std::memory_order_relaxed),
          sketch_weight_dispatches.load(std::memory_order_relaxed),
          sketch_sorted_dispatches.load(std::memory_order_relaxed),
          sketch_update_dispatches.load(std::memory_order_relaxed),
          sketch_ingest_gpu_time_ns.load(std::memory_order_relaxed),
          gradient_copy_dispatches.load(std::memory_order_relaxed),
          hessian_extract_dispatches.load(std::memory_order_relaxed),
          gradient_state_gpu_time_ns.load(std::memory_order_relaxed),
          gradient_state_shared_input_bytes.load(std::memory_order_relaxed),
          gradient_state_shared_output_bytes.load(std::memory_order_relaxed),
          gradient_state_staging_bytes.load(std::memory_order_relaxed)};
}

namespace {
void CompleteOperation(id<MTLCommandBuffer> command, std::atomic<std::uint64_t>* counter,
                       std::atomic<std::uint64_t>* duration = &objective_gpu_time_ns) {
  [command commit];
  [command waitUntilCompleted];
  CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
  counter->fetch_add(1, std::memory_order_relaxed);
  auto elapsed = command.GPUEndTime - command.GPUStartTime;
  if (elapsed > 0)
    duration->fetch_add(static_cast<std::uint64_t>(elapsed * 1e9), std::memory_order_relaxed);
}
}  // namespace
std::unique_ptr<LeafSums> MakeLeafSums(std::size_t leaves, std::size_t targets) {
  return std::make_unique<MetalLeafSums>(leaves, targets);
}

namespace {
template <typename T>
void CheckGradientView(linalg::MatrixView<T> view, bool output) {
  auto rows = view.Shape(0), columns = view.Shape(1);
  if (!rows || !columns) return;
  auto a = view.Stride(0), b = view.Stride(1);
  auto max = std::numeric_limits<std::size_t>::max();
  CHECK(!a || rows - 1 <= max / a);
  CHECK(!b || columns - 1 <= max / b);
  auto row_extent = (rows - 1) * a, column_extent = (columns - 1) * b;
  CHECK_LE(column_extent, max - row_extent);
  CHECK_LT(row_extent + column_extent, view.Values().size());
  if (output) {
    CHECK(rows == 1 || a != 0) << "Overlapping output rows.";
    CHECK(columns == 1 || b != 0) << "Overlapping output columns.";
    if (rows > 1 && columns > 1) {
      auto x = a, y = b;
      while (y) {
        auto r = x % y;
        x = y;
        y = r;
      }
      CHECK(rows <= b / x || columns <= a / x) << "Overlapping output elements.";
    }
  }
}

template <typename T>
void GradientState(linalg::MatrixView<GradientPair const> input, linalg::MatrixView<T> output,
                   bool hessian) {
  CHECK_EQ(input.Shape(0), output.Shape(0));
  CHECK_EQ(input.Shape(1), output.Shape(1));
  CheckGradientView(input, false);
  CheckGradientView(output, true);
  auto rows = input.Shape(0), columns = input.Shape(1);
  if (!rows || !columns) return;
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max() / columns);
  auto source = input.Values();
  auto destination = output.Values();
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto in = TryBorrowMemory(rt.device, source.data(), source.size_bytes(), 8);
    bool shared_in = in.buffer != nil;
    if (!shared_in) in = {Upload(rt, source.data(), source.size_bytes()), 0};
    bool alias = MemoryRangesOverlap(source.data(), source.size_bytes(), destination.data(),
                                     destination.size_bytes());
    auto out = alias ? BorrowedMemory{}
                     : TryBorrowMemory(rt.device, destination.data(), destination.size_bytes(), 4);
    bool shared_out = out.buffer != nil;
    // Preserve output padding when staging, and defer aliased writes until the
    // kernel has finished reading the original source snapshot.
    if (!shared_out) out = {Upload(rt, destination.data(), destination.size_bytes()), 0};
    struct {
      std::uint64_t columns, in_row, in_column, out_row, out_column;
      std::uint32_t hessian, padding;
    } p{columns, input.Stride(0), input.Stride(1), output.Stride(0), output.Stride(1), hessian, 0};
    static_assert(sizeof(p) == 48);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.gradient_state_pipeline];
    [e setBuffer:in.buffer offset:in.offset atIndex:0];
    [e setBuffer:out.buffer offset:out.offset atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(rows * columns, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, hessian ? &hessian_extract_dispatches : &gradient_copy_dispatches,
                      &gradient_state_gpu_time_ns);
    if (!shared_out) std::memcpy(destination.data(), out.buffer.contents, destination.size_bytes());
    gradient_state_shared_input_bytes.fetch_add(shared_in ? source.size_bytes() : 0,
                                                std::memory_order_relaxed);
    gradient_state_shared_output_bytes.fetch_add(shared_out ? destination.size_bytes() : 0,
                                                 std::memory_order_relaxed);
    gradient_state_staging_bytes.fetch_add(
        (shared_in ? 0 : source.size_bytes()) + (shared_out ? 0 : destination.size_bytes()),
        std::memory_order_relaxed);
  }
}
}  // namespace

void CopyGradients(linalg::MatrixView<GradientPair const> input,
                   linalg::MatrixView<GradientPair> output) {
  GradientState(input, output, false);
}
void ExtractHessians(linalg::MatrixView<GradientPair const> input,
                     linalg::MatrixView<float> output) {
  GradientState(input, output, true);
}

namespace {
class MetalLinearState final : public LinearState {
  Runtime& rt_{GetRuntime()};
  id<MTLBuffer> entries_, gradients_;
  std::vector<std::size_t> offsets_;
  std::size_t rows_;
  std::uint32_t groups_;
  struct Params {
    std::uint64_t begin, count;
    std::uint32_t groups, group, bias;
    float delta;
  };
  Params Parameters(std::int32_t feature, std::uint32_t group, float delta = 0) const {
    CHECK_GE(feature, -1);
    CHECK_LT(group, groups_);
    if (feature == -1) return {0, rows_, groups_, group, 1, delta};
    CHECK_LT(static_cast<std::size_t>(feature) + 1, offsets_.size());
    return {offsets_[feature], offsets_[feature + 1] - offsets_[feature], groups_, group, 0, delta};
  }

 public:
  MetalLinearState(std::size_t rows, std::uint32_t groups, Span<bst_idx_t const> offsets,
                   Span<Entry const> entries, Span<GradientPair const> gradients)
      : offsets_{offsets.begin(), offsets.end()}, rows_{rows}, groups_{groups} {
    static_assert(sizeof(Entry) == 8 && sizeof(GradientPair) == 8 && sizeof(Params) == 32);
    CHECK_GT(groups, 0);
    CHECK(!offsets.empty());
    CHECK_LE(offsets.size() - 1, std::numeric_limits<std::int32_t>::max());
    CHECK_LE(rows, std::numeric_limits<std::size_t>::max() / groups);
    CHECK_EQ(gradients.size(), rows * groups);
    CHECK_EQ(offsets.front(), 0);
    CHECK_EQ(offsets.back(), entries.size());
    for (std::size_t f = 0; f + 1 < offsets.size(); ++f) {
      CHECK_LE(offsets[f], offsets[f + 1]);
      CHECK_LE(offsets[f + 1], entries.size());
      for (auto i = offsets[f]; i < offsets[f + 1]; ++i) {
        CHECK_LT(entries[i].index, rows);
        if (i != offsets[f]) CHECK_LE(entries[i - 1].index, entries[i].index);
      }
    }
    @autoreleasepool {
      entries_ = Upload(rt_, entries.data(), entries.size_bytes());
      gradients_ = Upload(rt_, gradients.data(), gradients.size_bytes());
    }
  }
  GradientPairPrecise Gradient(std::int32_t feature, std::uint32_t group) override {
    auto p = Parameters(feature, group);
    if (!p.count) return {};
    auto blocks = (p.count + 255) / 256;
    CHECK_LE(blocks, std::numeric_limits<std::uint32_t>::max());
    CHECK_LE(blocks, rt_.device.maxBufferLength / sizeof(GradientPairPrecise));
    @autoreleasepool {
      auto partial = Allocate(rt_, blocks * sizeof(GradientPairPrecise));
      auto result = Allocate(rt_, sizeof(GradientPairPrecise));
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.linear_gradient_pipeline];
      [e setBuffer:entries_ offset:0 atIndex:0];
      [e setBuffer:gradients_ offset:0 atIndex:1];
      [e setBuffer:partial offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      // Reuse the binary64 pair final reduction; this is independent of the
      // generic/root-gradient dispatch counters.
      struct {
        std::uint64_t rows, stride0, stride1;
        std::uint32_t targets, blocks;
      } finish{p.count, 0, 0, 1, static_cast<std::uint32_t>(blocks)};
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.gradient_sum_finish_pipeline];
      [e setBuffer:partial offset:0 atIndex:0];
      [e setBuffer:result offset:0 atIndex:1];
      [e setBytes:&finish length:sizeof(finish) atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &linear_gradient_dispatches, &linear_gpu_time_ns);
      return *static_cast<GradientPairPrecise const*>(result.contents);
    }
  }
  void UpdateResidual(std::int32_t feature, std::uint32_t group, float delta) override {
    auto p = Parameters(feature, group, delta);
    if (!p.count || delta == 0.f) return;
    CHECK_LE(p.count, std::numeric_limits<std::uint32_t>::max());
    @autoreleasepool {
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.linear_residual_pipeline];
      [e setBuffer:entries_ offset:0 atIndex:0];
      [e setBuffer:gradients_ offset:0 atIndex:1];
      [e setBytes:&p length:sizeof(p) atIndex:2];
      [e dispatchThreads:MTLSizeMake(p.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &linear_residual_dispatches, &linear_gpu_time_ns);
    }
  }
  void CopyGradients(Span<GradientPair> output) const override {
    CHECK_EQ(output.size(), rows_ * groups_);
    if (!output.empty()) std::memcpy(output.data(), gradients_.contents, output.size_bytes());
  }
};
}  // namespace
std::unique_ptr<LinearState> MakeLinearState(std::size_t rows, std::uint32_t groups,
                                             Span<bst_idx_t const> offsets,
                                             Span<Entry const> entries,
                                             Span<GradientPair const> gradients) {
  return std::make_unique<MetalLinearState>(rows, groups, offsets, entries, gradients);
}

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
id<MTLBuffer> EncodeRadixRecords(Runtime& rt, id<MTLCommandBuffer> command,
                                  id<MTLBuffer> source, SortShape shape,
                                  std::uint32_t key_bits = 32) {
  auto size = shape.rows * shape.targets;
  CHECK_LE(size, std::numeric_limits<std::uint32_t>::max());
  auto target = Allocate(rt, source.length);
  auto ranks = Allocate(rt, size * sizeof(std::uint32_t));
  auto count_size = std::uint64_t{shape.targets} * 16 * shape.blocks;
  CHECK_LE(count_size, std::numeric_limits<std::uint32_t>::max());
  auto counts = Allocate(rt, count_size * sizeof(std::uint32_t));
  for (std::uint32_t shift = 0; shift < key_bits; shift += 4) {
    struct {
      std::uint32_t rows, targets, blocks, shift;
    } p{static_cast<std::uint32_t>(shape.rows), shape.targets, shape.blocks, shift};
    auto e = [command computeCommandEncoder];
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
id<MTLBuffer> EncodeColumnSort(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                               SortShape shape, bool descending = false) {
  auto size = shape.rows * shape.targets;
  CHECK_LE(size, std::numeric_limits<std::uint32_t>::max());
  auto source = Allocate(rt, size * 2 * sizeof(std::uint32_t));
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:descending ? rt.roc_sort_init_pipeline : rt.sort_init_pipeline];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:source offset:0 atIndex:1];
  [e setBytes:&shape length:sizeof(shape) atIndex:2];
  [e dispatchThreads:MTLSizeMake(shape.rows, shape.targets, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  return EncodeRadixRecords(rt, command, source, shape);
}
}  // namespace

namespace {
std::uint32_t CheckGroupSort(linalg::VectorView<float const> values,
                             Span<bst_group_t const> groups) {
  static_assert(sizeof(bst_group_t) == 4 && sizeof(std::size_t) == 8);
  CHECK_GE(groups.size(), 1);
  CHECK_EQ(groups.front(), 0);
  CHECK_EQ(groups.back(), values.Size());
  CHECK_LE(values.Size(), std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(groups.size() - 1, std::numeric_limits<std::uint32_t>::max());
  std::uint32_t max_group_size = 0;
  for (std::size_t i = 1; i < groups.size(); ++i) {
    CHECK_LE(groups[i - 1], groups[i]);
    max_group_size = std::max(max_group_size, groups[i] - groups[i - 1]);
  }
  return max_group_size;
}
struct GroupOrder {
  id<MTLBuffer> records;
  bool radix;
};
GroupOrder EncodeGroupSort(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                           id<MTLBuffer> offsets, std::uint64_t rows, std::uint64_t stride,
                           std::uint32_t n_groups, std::uint32_t max_group_size) {
  struct {
    std::uint64_t rows, stride;
    std::uint32_t groups, padding;
  } p{rows, stride, n_groups, 0};
  static_assert(sizeof(p) == 24);
  id<MTLBuffer> records;
  std::size_t shared_count = 1;
  while (shared_count < max_group_size && shared_count < 4096) shared_count <<= 1;
  bool medium = max_group_size <= 4096 &&
                shared_count * 2 * sizeof(std::uint32_t) <= rt.device.maxThreadgroupMemoryLength;
  bool radix_path = max_group_size > 256 && !medium;
  if (max_group_size <= 256) {
    records = Allocate(rt, rows * 2 * sizeof(std::uint32_t));
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.group_sort_small_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBuffer:records offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    std::uint32_t width = 32;
    while (width < max_group_size) width <<= 1;
    [e dispatchThreadgroups:MTLSizeMake(p.groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [e endEncoding];
  } else if (medium) {
    records = Allocate(rt, rows * 2 * sizeof(std::uint32_t));
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.group_sort_medium_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBuffer:records offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:shared_count * 2 * sizeof(std::uint32_t) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(p.groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
  } else {
    SortShape shape{p.rows, p.stride, 1, 1, static_cast<std::uint32_t>((p.rows + 255) / 256)};
    records = EncodeColumnSort(rt, command, input, shape, true);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.group_sort_key_pipeline];
    [e setBuffer:records offset:0 atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    std::uint32_t bits = 0;
    for (auto key = p.groups - 1; key; key >>= 1) ++bits;
    // Stable group-key passes preserve the preceding score and row ordering.
    if (bits) records = EncodeRadixRecords(rt, command, records, shape, bits);
  }
  return {records, radix_path};
}
void RecordGroupSort(bool radix, double seconds) {
  group_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
  if (radix) radix_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
  if (seconds > 0)
    group_sort_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(seconds * 1e9),
                                     std::memory_order_relaxed);
}
}  // namespace
void SortGroups(linalg::VectorView<float const> values, Span<bst_group_t const> groups,
                Span<std::size_t> relative_indices) {
  auto max_group_size = CheckGroupSort(values, groups);
  CHECK_EQ(relative_indices.size(), values.Size());
  if (!values.Size()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, values.Values().data(), values.Values().size_bytes());
    auto offsets = Upload(rt, groups.data(), groups.size_bytes());
    auto output = Allocate(rt, relative_indices.size_bytes());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto order = EncodeGroupSort(rt, command, input, offsets, values.Size(), values.Stride(0),
                                 groups.size() - 1, max_group_size);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.group_sort_export_pipeline];
    [e setBuffer:order.records offset:0 atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBuffer:output offset:0 atIndex:2];
    [e dispatchThreads:MTLSizeMake(values.Size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(relative_indices.data(), output.contents, relative_indices.size_bytes());
    RecordGroupSort(order.radix, command.GPUEndTime - command.GPUStartTime);
  }
}

RankingAUCSums RankingROCAUC(linalg::VectorView<float const> predictions,
                             linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                             Span<float const> weights, Span<RankingAUCSums> individual) {
  auto max_group_size = CheckGroupSort(labels, groups);
  auto n_groups = groups.size() - 1;
  CHECK_EQ(predictions.Size(), labels.Size());
  CHECK(weights.empty() || weights.size() == n_groups);
  CHECK(individual.empty() || individual.size() == n_groups);
  if (!n_groups) return {0, 0};
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, predictions.Values().data(), predictions.Values().size_bytes());
    auto response = Upload(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = Upload(rt, groups.data(), groups.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto source = Allocate(rt, labels.Size() * 2 * sizeof(std::uint32_t));
    auto target = Allocate(rt, labels.Size() * 2 * sizeof(std::uint32_t));
    auto row_group = Allocate(rt, labels.Size() * sizeof(std::uint32_t));
    auto counts = Allocate(rt, labels.Size() * 2 * sizeof(std::uint32_t));
    auto partial = Allocate(rt, n_groups * sizeof(RankingAUCSums));
    auto output = Allocate(rt, sizeof(RankingAUCSums));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    struct {
      std::uint64_t rows, prediction_stride, width;
      std::uint32_t groups, weighted;
    } p{predictions.Size(), predictions.Stride(0), 1, static_cast<std::uint32_t>(n_groups),
        static_cast<std::uint32_t>(!weights.empty())};
    static_assert(sizeof(p) == 32 && sizeof(RankingAUCSums) == 16);
    GroupOrder order{Allocate(rt, 16), false};
    if (p.rows) {
      order = EncodeGroupSort(rt, command, response, offsets, labels.Size(), labels.Stride(0),
                              n_groups, max_group_size);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.ranking_roc_init_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:input offset:0 atIndex:1];
      [e setBuffer:offsets offset:0 atIndex:2];
      [e setBuffer:source offset:0 atIndex:3];
      [e setBuffer:row_group offset:0 atIndex:4];
      [e setBuffer:counts offset:0 atIndex:5];
      [e setBytes:&p length:sizeof(p) atIndex:6];
      [e dispatchThreads:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      for (; p.width < max_group_size; p.width <<= 1) {
        e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt.ranking_roc_merge_pipeline];
        [e setBuffer:source offset:0 atIndex:0];
        [e setBuffer:target offset:0 atIndex:1];
        [e setBuffer:offsets offset:0 atIndex:2];
        [e setBuffer:row_group offset:0 atIndex:3];
        [e setBuffer:counts offset:0 atIndex:4];
        [e setBytes:&p length:sizeof(p) atIndex:5];
        [e dispatchThreads:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        std::swap(source, target);
      }
    }
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.ranking_roc_score_pipeline];
    [e setBuffer:counts offset:0 atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBuffer:weight offset:0 atIndex:2];
    [e setBuffer:partial offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    SortShape shape{n_groups, 1, 1, 1, static_cast<std::uint32_t>(n_groups)};
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.gradient_sum_finish_pipeline];
    [e setBuffer:partial offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBytes:&shape length:sizeof(shape) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (p.rows) RecordGroupSort(order.radix, elapsed);
    ranking_roc_auc_dispatches.fetch_add(1, std::memory_order_relaxed);
    roc_auc_dispatches.fetch_add(1, std::memory_order_relaxed);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (elapsed > 0) {
      auto ns = static_cast<std::uint64_t>(elapsed * 1e9);
      ranking_roc_auc_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
      roc_auc_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
      metric_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
    }
    if (!individual.empty())
      std::memcpy(individual.data(), partial.contents, individual.size_bytes());
    auto result = *static_cast<RankingAUCSums const*>(output.contents);
    CHECK_LE(result.area, result.valid_groups * (1.0 + kRtEps))
        << "Invalid ranking ROC-AUC (group-weight square may have overflowed).";
    return result;
  }
}

std::uint32_t CheckRankingLabels(linalg::VectorView<float const> labels, RankingLabelKind kind) {
  CHECK(kind == RankingLabelKind::kBinary || kind == RankingLabelKind::kNDCG);
  CHECK_LE(labels.Size(), std::numeric_limits<std::uint32_t>::max());
  if (!labels.Size()) return 0;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, labels.Values().data(), labels.Values().size_bytes());
    std::uint32_t zero = 0;
    auto flags = Upload(rt, &zero, sizeof(zero));
    struct {
      std::uint64_t rows, stride;
      std::uint32_t kind, padding;
    } p{labels.Size(), labels.Stride(0), static_cast<std::uint32_t>(kind), 0};
    static_assert(sizeof(p) == 24);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.ranking_labels_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:flags offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((labels.Size() + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    ranking_label_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      ranking_label_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                          std::memory_order_relaxed);
    return *static_cast<std::uint32_t const*>(flags.contents);
  }
}

void InitNDCG(linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
              std::uint32_t topk, bool exponential, Span<double> discounts,
              Span<double> inverse_idcg) {
  auto max_group_size = CheckGroupSort(labels, groups);
  auto n_groups = groups.size() - 1;
  CHECK_EQ(discounts.size(), max_group_size);
  CHECK_EQ(inverse_idcg.size(), n_groups);
  if (!n_groups) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = Upload(rt, groups.data(), groups.size_bytes());
    auto discount = Allocate(rt, discounts.size_bytes());
    auto inverse = Allocate(rt, inverse_idcg.size_bytes());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    GroupOrder order{Allocate(rt, 16), false};
    if (labels.Size()) {
      order = EncodeGroupSort(rt, command, input, offsets, labels.Size(), labels.Stride(0),
                              n_groups, max_group_size);
    }
    if (max_group_size) {
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.ndcg_discount_pipeline];
      [e setBuffer:discount offset:0 atIndex:0];
      [e dispatchThreads:MTLSizeMake(max_group_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    struct {
      std::uint64_t rows, label_stride;
      std::uint32_t groups, topk, kind, weighted, minus, exponential;
    } p{labels.Size(),
        labels.Stride(0),
        static_cast<std::uint32_t>(n_groups),
        topk,
        0,
        0,
        0,
        static_cast<std::uint32_t>(exponential)};
    static_assert(sizeof(p) == 40);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.ndcg_ideal_pipeline];
    [e setBuffer:order.records offset:0 atIndex:0];
    [e setBuffer:input offset:0 atIndex:1];
    [e setBuffer:offsets offset:0 atIndex:2];
    [e setBuffer:discount offset:0 atIndex:3];
    [e setBuffer:inverse offset:0 atIndex:4];
    [e setBytes:&p length:sizeof(p) atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake((n_groups + 7) / 8, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    if (!discounts.empty())
      std::memcpy(discounts.data(), discount.contents, discounts.size_bytes());
    std::memcpy(inverse_idcg.data(), inverse.contents, inverse_idcg.size_bytes());
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (labels.Size()) RecordGroupSort(order.radix, elapsed);
    ranking_cache_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (elapsed > 0)
      ranking_cache_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                          std::memory_order_relaxed);
  }
}

void RankingGradient(RankingLossKind kind, Span<float const> predictions,
                      linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                      Span<float const> weights, std::uint32_t pairs, bool normalization,
                      bool score_normalization, bool exponential, double weight_norm,
                      Span<double const> discounts, Span<double const> inverse_idcg,
                      Span<GradientPair> output, bool mean, std::uint32_t seed, RankingBias bias) {
  auto values = linalg::MakeTensorView(DeviceOrd::CPU(), predictions, predictions.size());
  auto max_group_size = CheckGroupSort(values, groups);
  auto n_groups = groups.size() - 1;
  CHECK_EQ(bias.positive.size(), bias.negative.size());
  CHECK_LE(bias.positive.size(), std::numeric_limits<std::uint32_t>::max());
  auto positions = static_cast<std::uint32_t>(bias.positive.size());
  CHECK_GE(pairs, 1);
  CHECK_EQ(labels.Size(), predictions.size());
  CHECK_EQ(output.size(), predictions.size());
  CHECK(weights.empty() || weights.size() == n_groups);
  CHECK(kind == RankingLossKind::kPairwise || kind == RankingLossKind::kNDCG || kind == RankingLossKind::kMAP);
  if (kind == RankingLossKind::kNDCG) {
    CHECK_GE(discounts.size(), max_group_size);
    CHECK_EQ(inverse_idcg.size(), n_groups);
  }
  if (predictions.empty()) return;
  static_assert(sizeof(GradientPair) == 8);
  struct Task {
    std::uint64_t pair_begin, tile_begin;
    std::uint32_t begin, count, topk, padding;
  };
  static_assert(sizeof(Task) == 32);
  std::vector<Task> tasks(n_groups + 1);
  std::uint64_t total_pairs = 0, total_tiles = 0;
  for (std::size_t g = 0; g < n_groups; ++g) {
    auto count = groups[g + 1] - groups[g], k = mean ? pairs : std::min(count, pairs);
    tasks[g] = {total_pairs, total_tiles, groups[g], count, k, 0};
    auto n = std::uint64_t{k} * count - (mean ? 0 : std::uint64_t{k} * (k + 1ull) / 2);
    total_pairs += n;
    total_tiles += (n + 255) / 256;
  }
  tasks.back() = {total_pairs, total_tiles, groups.back(), 0, 0, 0};
  // Host work is query-size/launch metadata only. Actual pair indices, labels
  // and gradients are generated on GPU. Reuse a 64 MiB pair buffer across
  // batches; a single larger query is allowed up to the device buffer limit.
  auto budget_pairs = (64ull << 20) / (sizeof(GradientPair) + (mean ? 16 : 0) + (positions ? 20 : 0));
  std::vector<std::uint32_t> batches{0};
  std::uint64_t max_pairs = 0;
  for (std::size_t g = 0; g < n_groups; ++g) {
    if (g != batches.back() && tasks[g + 1].pair_begin - tasks[batches.back()].pair_begin > budget_pairs) {
      max_pairs = std::max(max_pairs, tasks[g].pair_begin - tasks[batches.back()].pair_begin);
      batches.push_back(g);
    }
  }
  max_pairs = std::max(max_pairs, total_pairs - tasks[batches.back()].pair_begin);
  batches.push_back(n_groups);
  @autoreleasepool {
    auto& rt = GetRuntime();
    CHECK_LE(max_pairs, rt.device.maxBufferLength / sizeof(GradientPair))
        << "A ranking query needs more pair storage than this Metal device supports.";
    auto input = Upload(rt, predictions.data(), predictions.size_bytes());
    auto response = Upload(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = Upload(rt, groups.data(), groups.size_bytes());
    auto task_buffer = Upload(rt, tasks.data(), tasks.size() * sizeof(Task));
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto discount = Upload(rt, discounts.data(), discounts.size_bytes());
    auto inverse = Upload(rt, inverse_idcg.data(), inverse_idcg.size_bytes());
    auto pair_buffer = Allocate(rt, max_pairs * sizeof(GradientPair));
    if (positions) CHECK_LE(max_pairs, std::numeric_limits<std::uint32_t>::max());
    auto selected = Allocate(rt, positions ? max_pairs * 4 : 4);
    auto selected_count = Allocate(rt, 16);
    auto bias_pair = Allocate(rt, positions ? max_pairs * 16 : 16);
    auto bias_row = Allocate(rt, positions ? predictions.size() * 16 : 16);
    auto bias_segments = (n_groups + 127) / 128;
    auto bias_partial = Allocate(rt, positions ? positions * bias_segments * 16ull : 16);
    auto bias_sum = Allocate(rt, positions ? positions * 16ull : 16);
    auto bias_positive = Upload(rt, bias.positive.data(), bias.positive.size_bytes());
    auto bias_negative = Upload(rt, bias.negative.data(), bias.negative.size_bytes());
    // Each sampled pair has two incidence nodes. A query's single generator
    // appends them in CPU order; rows can then accumulate independently.
    if (mean) CHECK_LE(max_pairs, (std::numeric_limits<std::uint32_t>::max() - 1ull) / 2);
    auto sampled = Allocate(rt, mean ? max_pairs * 8 : 8);
    auto links = Allocate(rt, mean ? max_pairs * 8 : 8);
    auto heads = Allocate(rt, mean ? predictions.size() * 4 : 4);
    auto tails = Allocate(rt, mean ? predictions.size() * 4 : 4);
    auto norms = Allocate(rt, n_groups * sizeof(float));
    auto result = Allocate(rt, output.size_bytes());
    auto map_stats = Allocate(rt, kind == RankingLossKind::kMAP ? predictions.size() * 2 * sizeof(double) : 16);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    id<MTLCounterSampleBuffer> timing = nil;
    std::vector<char const*> phases;
    if (auto requested = std::getenv("XGBOOST_METAL_PROFILE_RANKING"); requested && std::strcmp(requested, "1") == 0) {
      CHECK([rt.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
          << "This Metal device does not support compute-stage timestamp sampling.";
      auto desc = [MTLCounterSampleBufferDescriptor new];
      for (id<MTLCounterSet> set in rt.device.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) { desc.counterSet = set; break; }
      }
      CHECK(desc.counterSet) << "No Metal timestamp counter set is available.";
      desc.storageMode = MTLStorageModeShared;
      desc.sampleCount = 2 * (batches.size() * 8 + 16);
      NSError* error = nil;
      timing = [rt.device newCounterSampleBufferWithDescriptor:desc error:&error];
      CHECK(timing) << ErrorText(error);
    }
    auto encoder = [&](char const* phase) -> id<MTLComputeCommandEncoder> {
      if (!timing) return [command computeCommandEncoder];
      auto desc = [MTLComputePassDescriptor computePassDescriptor];
      auto attachment = desc.sampleBufferAttachments[0];
      attachment.sampleBuffer = timing;
      attachment.startOfEncoderSampleIndex = 2 * phases.size();
      attachment.endOfEncoderSampleIndex = 2 * phases.size() + 1;
      CHECK_LT(attachment.endOfEncoderSampleIndex, timing.sampleCount);
      phases.push_back(phase);
      return [command computeCommandEncoderWithDescriptor:desc];
    };
    auto order = EncodeGroupSort(rt, command, input, offsets, predictions.size(), 1, n_groups, max_group_size);
    GroupOrder label_order{};
    if (mean) {
      auto ranked_labels = Allocate(rt, predictions.size_bytes());
      auto e = encoder("mean_labels");
      CHECK(e);
      [e setComputePipelineState:rt.ranking_mean_labels_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:response offset:0 atIndex:1];
      [e setBuffer:ranked_labels offset:0 atIndex:2];
      std::uint64_t stride = labels.Stride(0);
      [e setBytes:&stride length:sizeof(stride) atIndex:3];
      [e dispatchThreads:MTLSizeMake(predictions.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      label_order = EncodeGroupSort(rt, command, ranked_labels, offsets, predictions.size(), 1, n_groups, max_group_size);
    }
    if (kind == RankingLossKind::kMAP) {
      struct {
        std::uint64_t rows, label_stride;
        std::uint32_t groups, topk, kind, weighted, minus, exponential;
      } p{predictions.size(), labels.Stride(0), static_cast<std::uint32_t>(n_groups), 0, 0, 0, 0, 0};
      auto e = encoder("map_prefix");
      CHECK(e);
      [e setComputePipelineState:rt.ranking_map_stats_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:response offset:0 atIndex:1];
      [e setBuffer:offsets offset:0 atIndex:2];
      [e setBuffer:map_stats offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake((n_groups + 7) / 8, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    for (std::size_t b = 1; b < batches.size(); ++b) {
      auto begin = batches[b - 1], end = batches[b];
      float norm_float = static_cast<float>(weight_norm);
      std::uint32_t norm_bits;
      std::memcpy(&norm_bits, &norm_float, sizeof(norm_bits));
      struct {
        std::uint64_t rows, label_stride, pair_base, tile_base;
        std::uint32_t group_begin, group_end, row_begin, kind;
        std::uint32_t normalized, score_normalized, exponential, weighted;
        std::uint32_t weight_norm, mean, seed, positions;
      } p{predictions.size(), labels.Stride(0), tasks[begin].pair_begin, tasks[begin].tile_begin,
           begin, end, groups[begin], static_cast<std::uint32_t>(kind),
           static_cast<std::uint32_t>(normalization), static_cast<std::uint32_t>(score_normalization),
           static_cast<std::uint32_t>(exponential), static_cast<std::uint32_t>(!weights.empty()), norm_bits, static_cast<std::uint32_t>(mean), seed, positions};
      static_assert(sizeof(p) == 80);
      if (mean) {
        auto e = encoder("sample_pairs");
        CHECK(e);
        [e setComputePipelineState:rt.ranking_mean_sample_pipeline];
        [e setBuffer:order.records offset:0 atIndex:0];
        [e setBuffer:label_order.records offset:0 atIndex:1];
        [e setBuffer:response offset:0 atIndex:2];
        [e setBuffer:task_buffer offset:0 atIndex:3];
        [e setBuffer:sampled offset:0 atIndex:4];
        [e setBuffer:links offset:0 atIndex:5];
        [e setBuffer:heads offset:0 atIndex:6];
        [e setBuffer:tails offset:0 atIndex:7];
        [e setBytes:&p length:sizeof(p) atIndex:8];
        [e dispatchThreads:MTLSizeMake(end - begin, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
      }
      if (positions) {
        auto blit = [command blitCommandEncoder];
        CHECK(blit);
        [blit fillBuffer:selected_count range:NSMakeRange(0, 16) value:0];
        [blit endEncoding];
      }
      auto tiles = tasks[end].tile_begin - tasks[begin].tile_begin;
      if (tiles) {
        auto e = encoder("pair_gradients");
        CHECK(e);
        [e setComputePipelineState:rt.ranking_pairs_pipeline];
        [e setBuffer:order.records offset:0 atIndex:0];
        [e setBuffer:input offset:0 atIndex:1];
        [e setBuffer:response offset:0 atIndex:2];
        [e setBuffer:task_buffer offset:0 atIndex:3];
        [e setBuffer:discount offset:0 atIndex:4];
        [e setBuffer:inverse offset:0 atIndex:5];
        [e setBuffer:map_stats offset:0 atIndex:6];
        [e setBuffer:pair_buffer offset:0 atIndex:7];
        [e setBytes:&p length:sizeof(p) atIndex:8];
        [e setBuffer:sampled offset:0 atIndex:9];
        [e setBuffer:bias_pair offset:0 atIndex:10];
        [e setBuffer:bias_positive offset:0 atIndex:11];
        [e setBuffer:bias_negative offset:0 atIndex:12];
        [e setBuffer:selected offset:0 atIndex:13];
        [e setBuffer:selected_count offset:0 atIndex:14];
        [e dispatchThreadgroups:MTLSizeMake(tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      if (positions) {
        auto e = encoder("bias_dispatch");
        CHECK(e);
        [e setComputePipelineState:rt.ranking_bias_dispatch_pipeline];
        [e setBuffer:selected_count offset:0 atIndex:0];
        [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [e endEncoding];
        e = encoder("bias_pairs");
        CHECK(e);
        [e setComputePipelineState:rt.ranking_bias_pairs_pipeline];
        [e setBuffer:order.records offset:0 atIndex:0];
        [e setBuffer:input offset:0 atIndex:1];
        [e setBuffer:response offset:0 atIndex:2];
        [e setBuffer:task_buffer offset:0 atIndex:3];
        [e setBuffer:discount offset:0 atIndex:4];
        [e setBuffer:inverse offset:0 atIndex:5];
        [e setBuffer:map_stats offset:0 atIndex:6];
        [e setBuffer:pair_buffer offset:0 atIndex:7];
        [e setBytes:&p length:sizeof(p) atIndex:8];
        [e setBuffer:sampled offset:0 atIndex:9];
        [e setBuffer:bias_pair offset:0 atIndex:10];
        [e setBuffer:bias_positive offset:0 atIndex:11];
        [e setBuffer:bias_negative offset:0 atIndex:12];
        [e setBuffer:selected offset:0 atIndex:13];
        [e setBuffer:selected_count offset:0 atIndex:14];
        [e dispatchThreadgroupsWithIndirectBuffer:selected_count indirectBufferOffset:4
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      auto e = encoder("normalize");
      CHECK(e);
      [e setComputePipelineState:rt.ranking_normalize_pipeline];
      [e setBuffer:pair_buffer offset:0 atIndex:0];
      [e setBuffer:task_buffer offset:0 atIndex:1];
      [e setBuffer:norms offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake((end - begin + 7) / 8, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto rows = std::uint64_t{groups[end]} - groups[begin];
      if (rows) {
        e = encoder("gather");
        CHECK(e);
        [e setComputePipelineState:mean ? rt.ranking_mean_gather_pipeline : rt.ranking_gather_pipeline];
        [e setBuffer:order.records offset:0 atIndex:0];
        [e setBuffer:task_buffer offset:0 atIndex:1];
        [e setBuffer:pair_buffer offset:0 atIndex:2];
        [e setBuffer:norms offset:0 atIndex:3];
        [e setBuffer:weight offset:0 atIndex:4];
        [e setBuffer:result offset:0 atIndex:5];
        [e setBytes:&p length:sizeof(p) atIndex:6];
        if (mean) {
          [e setBuffer:heads offset:0 atIndex:7];
          [e setBuffer:links offset:0 atIndex:8];
        }
        [e setBuffer:bias_pair offset:0 atIndex:9];
        [e setBuffer:bias_row offset:0 atIndex:10];
        [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
    }
    if (positions) {
      struct {
        std::uint32_t groups, positions;
        double regularizer;
        std::uint32_t segments, padding;
      } p{static_cast<std::uint32_t>(n_groups), positions, bias.regularizer,
          static_cast<std::uint32_t>(bias_segments), 0};
      static_assert(sizeof(p) == 24);
      auto e = encoder("bias_sum");
      CHECK(e);
      [e setComputePipelineState:rt.ranking_bias_sum_pipeline];
      [e setBuffer:bias_row offset:0 atIndex:0];
      [e setBuffer:offsets offset:0 atIndex:1];
      [e setBuffer:bias_segments == 1 ? bias_sum : bias_partial offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      [e dispatchThreads:MTLSizeMake(positions, bias_segments, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [e endEncoding];
      if (bias_segments > 1) {
        e = encoder("bias_finish");
        CHECK(e);
        [e setComputePipelineState:rt.ranking_bias_finish_pipeline];
        [e setBuffer:bias_partial offset:0 atIndex:0];
        [e setBuffer:bias_sum offset:0 atIndex:1];
        [e setBytes:&p length:sizeof(p) atIndex:2];
        [e dispatchThreads:MTLSizeMake(positions, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
      }
      e = encoder("bias_update");
      CHECK(e);
      [e setComputePipelineState:rt.ranking_bias_update_pipeline];
      [e setBuffer:bias_sum offset:0 atIndex:0];
      [e setBuffer:bias_positive offset:0 atIndex:1];
      [e setBuffer:bias_negative offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      [e dispatchThreads:MTLSizeMake(positions, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [e endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    if (timing) {
      auto data = [timing resolveCounterRange:NSMakeRange(0, 2 * phases.size())];
      CHECK(data && data.length == 2 * phases.size() * sizeof(MTLCounterResultTimestamp));
      auto samples = static_cast<MTLCounterResultTimestamp const*>(data.bytes);
      for (std::size_t i = 0; i < phases.size(); ++i) {
        auto begin = samples[2 * i].timestamp, end = samples[2 * i + 1].timestamp;
        CHECK(begin != MTLCounterErrorValue && end != MTLCounterErrorValue && end >= begin);
        LOG(CONSOLE) << "MPS ranking profile rows=" << predictions.size() << " phase=" << phases[i]
                  << " gpu_ticks=" << end - begin;
      }
    }
    std::memcpy(output.data(), result.contents, output.size_bytes());
    if (positions) {
      std::memcpy(bias.positive.data(), bias_positive.contents, bias.positive.size_bytes());
      std::memcpy(bias.negative.data(), bias_negative.contents, bias.negative.size_bytes());
      ranking_bias_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    RecordGroupSort(order.radix, elapsed);
    if (mean) RecordGroupSort(label_order.radix, 0);
    ranking_gradient_dispatches.fetch_add(1, std::memory_order_relaxed);
    ranking_pairs.fetch_add(total_pairs, std::memory_order_relaxed);
    ranking_pair_batches.fetch_add(batches.size() - 1, std::memory_order_relaxed);
    objective_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (elapsed > 0) {
      auto ns = static_cast<std::uint64_t>(elapsed * 1e9);
      ranking_gradient_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
      objective_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
    }
  }
}

RankingSums RankingScore(RankingMetricKind kind, linalg::VectorView<float const> predictions,
                         linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                         Span<float const> weights, std::uint32_t topk, bool minus,
                         bool exponential, Span<double const> discounts,
                         Span<double const> inverse_idcg) {
  auto max_group_size = CheckGroupSort(predictions, groups);
  auto n_groups = groups.size() - 1;
  CHECK_EQ(labels.Size(), predictions.Size());
  CHECK(weights.empty() || weights.size() == n_groups);
  CHECK(kind == RankingMetricKind::kPrecision || kind == RankingMetricKind::kNDCG ||
        kind == RankingMetricKind::kMAP);
  if (kind == RankingMetricKind::kNDCG) {
    CHECK_GE(discounts.size(), std::min(max_group_size, topk));
    CHECK_EQ(inverse_idcg.size(), n_groups);
  }
  if (!n_groups) return {0, 0};
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, predictions.Values().data(), predictions.Values().size_bytes());
    auto response = Upload(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = Upload(rt, groups.data(), groups.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto discount = Upload(rt, discounts.data(), discounts.size_bytes());
    auto inverse = Upload(rt, inverse_idcg.data(), inverse_idcg.size_bytes());
    auto partial = Allocate(rt, n_groups * 2 * sizeof(double));
    auto output = Allocate(rt, 2 * sizeof(double));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    GroupOrder order{Allocate(rt, 16), false};
    if (predictions.Size()) {
      order = EncodeGroupSort(rt, command, input, offsets, predictions.Size(),
                              predictions.Stride(0), n_groups, max_group_size);
    }
    struct {
      std::uint64_t rows, label_stride;
      std::uint32_t groups, topk, kind, weighted, minus, exponential;
    } p{predictions.Size(),
        labels.Stride(0),
        static_cast<std::uint32_t>(n_groups),
        topk,
        static_cast<std::uint32_t>(kind),
        static_cast<std::uint32_t>(!weights.empty()),
        static_cast<std::uint32_t>(minus),
        static_cast<std::uint32_t>(exponential)};
    static_assert(sizeof(p) == 40);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.ranking_score_pipeline];
    [e setBuffer:order.records offset:0 atIndex:0];
    [e setBuffer:response offset:0 atIndex:1];
    [e setBuffer:offsets offset:0 atIndex:2];
    [e setBuffer:weight offset:0 atIndex:3];
    [e setBuffer:discount offset:0 atIndex:4];
    [e setBuffer:inverse offset:0 atIndex:5];
    [e setBuffer:partial offset:0 atIndex:6];
    [e setBytes:&p length:sizeof(p) atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake((n_groups + 7) / 8, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    SortShape shape{n_groups, 1, 1, 1, static_cast<std::uint32_t>(n_groups)};
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.gradient_sum_finish_pipeline];
    [e setBuffer:partial offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBytes:&shape length:sizeof(shape) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    // Sort and score share a command; its duration overlaps both family counters.
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (predictions.Size()) RecordGroupSort(order.radix, elapsed);
    ranking_metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (elapsed > 0) {
      auto ns = static_cast<std::uint64_t>(elapsed * 1e9);
      ranking_metric_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
      metric_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
    }
    auto result = static_cast<double const*>(output.contents);
    return {result[0], result[1]};
  }
}

namespace {
id<MTLBuffer> EncodeSegmentedDoublePrefix(Runtime& rt, id<MTLCommandBuffer> command,
                                          id<MTLBuffer> input, id<MTLBuffer> ids,
                                          std::uint32_t rows) {
  auto blocks = (std::uint64_t{rows} + 255) / 256;
  auto output = Allocate(rt, std::size_t{rows} * 2 * sizeof(double));
  auto totals = Allocate(rt, blocks * 2 * sizeof(double));
  auto total_ids = Allocate(rt, blocks * sizeof(std::uint32_t));
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:rt.prefix_segmented_pair_pipeline];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:ids offset:0 atIndex:1];
  [e setBuffer:output offset:0 atIndex:2];
  [e setBuffer:totals offset:0 atIndex:3];
  [e setBuffer:total_ids offset:0 atIndex:4];
  [e setBytes:&rows length:sizeof(rows) atIndex:5];
  [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  if (blocks > 1) {
    auto upper = EncodeSegmentedDoublePrefix(rt, command, totals, total_ids, blocks);
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.prefix_segmented_pair_add_pipeline];
    [e setBuffer:output offset:0 atIndex:0];
    [e setBuffer:ids offset:0 atIndex:1];
    [e setBuffer:upper offset:0 atIndex:2];
    [e setBuffer:total_ids offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
  }
  return output;
}

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

RankingAUCSums RankingPRAUC(linalg::VectorView<float const> predictions,
                            linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                            Span<float const> weights, Span<RankingAUCSums> individual) {
  auto max_group_size = CheckGroupSort(predictions, groups);
  auto n_groups = groups.size() - 1;
  CHECK_EQ(predictions.Size(), labels.Size());
  CHECK(weights.empty() || weights.size() == n_groups);
  CHECK(individual.empty() || individual.size() == n_groups);
  if (!n_groups) return {0, 0};
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, predictions.Values().data(), predictions.Values().size_bytes());
    auto response = Upload(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = Upload(rt, groups.data(), groups.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto ids = Allocate(rt, predictions.Size() * sizeof(std::uint32_t));
    auto keys = Allocate(rt, predictions.Size() * sizeof(std::uint32_t));
    auto counts = Allocate(rt, predictions.Size() * 2 * sizeof(double));
    auto areas = Allocate(rt, predictions.Size() * sizeof(double));
    auto flags = Allocate(rt, (n_groups + 1) * sizeof(std::uint32_t));
    auto partial = Allocate(rt, n_groups * sizeof(RankingAUCSums));
    auto output = Allocate(rt, sizeof(RankingAUCSums));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto clear = [command blitCommandEncoder];
    CHECK(clear);
    [clear fillBuffer:flags range:NSMakeRange(0, flags.length) value:0];
    [clear endEncoding];
    struct {
      std::uint64_t rows, prediction_stride, label_stride;
      std::uint32_t groups, weighted;
    } p{predictions.Size(), predictions.Stride(0), labels.Stride(0),
        static_cast<std::uint32_t>(n_groups), static_cast<std::uint32_t>(!weights.empty())};
    static_assert(sizeof(p) == 32);
    GroupOrder order{Allocate(rt, 16), false};
    id<MTLBuffer> prefix = Allocate(rt, 16);
    if (p.rows) {
      order = EncodeGroupSort(rt, command, input, offsets, p.rows, predictions.Stride(0), n_groups,
                              max_group_size);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.ranking_pr_counts_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:input offset:0 atIndex:1];
      [e setBuffer:response offset:0 atIndex:2];
      [e setBuffer:weight offset:0 atIndex:3];
      [e setBuffer:offsets offset:0 atIndex:4];
      [e setBuffer:ids offset:0 atIndex:5];
      [e setBuffer:keys offset:0 atIndex:6];
      [e setBuffer:counts offset:0 atIndex:7];
      [e setBuffer:flags offset:0 atIndex:8];
      [e setBytes:&p length:sizeof(p) atIndex:9];
      [e dispatchThreads:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      prefix = EncodeSegmentedDoublePrefix(rt, command, counts, ids, p.rows);
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.ranking_pr_area_pipeline];
      [e setBuffer:keys offset:0 atIndex:0];
      [e setBuffer:prefix offset:0 atIndex:1];
      [e setBuffer:ids offset:0 atIndex:2];
      [e setBuffer:offsets offset:0 atIndex:3];
      [e setBuffer:flags offset:0 atIndex:4];
      [e setBuffer:areas offset:0 atIndex:5];
      [e dispatchThreads:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.ranking_pr_score_pipeline];
    [e setBuffer:areas offset:0 atIndex:0];
    [e setBuffer:prefix offset:0 atIndex:1];
    [e setBuffer:offsets offset:0 atIndex:2];
    [e setBuffer:flags offset:0 atIndex:3];
    [e setBuffer:partial offset:0 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    SortShape shape{n_groups, 1, 1, 1, static_cast<std::uint32_t>(n_groups)};
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.gradient_sum_finish_pipeline];
    [e setBuffer:partial offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBytes:&shape length:sizeof(shape) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (p.rows) RecordGroupSort(order.radix, elapsed);
    ranking_pr_auc_dispatches.fetch_add(1, std::memory_order_relaxed);
    pr_auc_dispatches.fetch_add(1, std::memory_order_relaxed);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (elapsed > 0) {
      auto ns = static_cast<std::uint64_t>(elapsed * 1e9);
      ranking_pr_auc_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
      pr_auc_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
      metric_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
    }
    CHECK_EQ(static_cast<std::uint32_t const*>(flags.contents)[n_groups], 0)
        << "PR-AUC supports only binary relevance for learning to rank.";
    if (!individual.empty())
      std::memcpy(individual.data(), partial.contents, individual.size_bytes());
    return *static_cast<RankingAUCSums const*>(output.contents);
  }
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

std::vector<WQSummaryContainer> FinalizeSketches(Span<SketchFinalization const> columns) {
  using Entry = WQSummary<>::Entry;
  static_assert(sizeof(Entry) == 16);
  CHECK_LE(columns.size(), std::numeric_limits<std::uint32_t>::max());
  std::vector<WQSummaryContainer> result(columns.size());
  if (columns.empty()) return result;
  struct Task {
    std::uint64_t begin, end, workspace;
    std::uint32_t capacity, intermediate_budget, final_budget, padding;
  };
  struct Level {
    std::uint64_t begin, size;
  };
  static_assert(sizeof(Task) == 40 && sizeof(Level) == 16);
  std::vector<Task> tasks;
  std::vector<Level> levels;
  std::vector<Entry> entries;
  std::size_t workspace_size = 0;
  for (auto const& column : columns) {
    CHECK_LE(column.intermediate_budget, std::numeric_limits<std::uint32_t>::max());
    CHECK_LE(column.final_budget, std::numeric_limits<std::uint32_t>::max());
    auto begin = levels.size();
    std::size_t total = 0, maximum = 0;
    for (auto const& level : column.levels) {
      CHECK_LE(level.Size(), std::numeric_limits<std::uint32_t>::max());
      auto data = level.Entries();
      levels.push_back({entries.size(), data.size()});
      entries.insert(entries.end(), data.begin(), data.end());
      total += data.size();
      maximum = std::max(maximum, data.size());
    }
    auto capacity = std::min(total, column.intermediate_budget + maximum);
    CHECK_LE(capacity, std::numeric_limits<std::uint32_t>::max());
    tasks.push_back({begin, levels.size(), workspace_size, static_cast<std::uint32_t>(capacity),
                     static_cast<std::uint32_t>(column.intermediate_budget),
                     static_cast<std::uint32_t>(column.final_budget), 0});
    workspace_size += capacity;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, entries.data(), entries.size() * sizeof(Entry));
    auto offsets = Upload(rt, levels.data(), levels.size() * sizeof(Level));
    auto metadata = Upload(rt, tasks.data(), tasks.size() * sizeof(Task));
    auto current = Allocate(rt, workspace_size * sizeof(Entry));
    auto merged = Allocate(rt, workspace_size * sizeof(Entry));
    auto sizes = Allocate(rt, columns.size() * sizeof(std::uint32_t));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.sketch_finalize_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBuffer:metadata offset:0 atIndex:2];
    [e setBuffer:current offset:0 atIndex:3];
    [e setBuffer:merged offset:0 atIndex:4];
    [e setBuffer:sizes offset:0 atIndex:5];
    CHECK_EQ(rt.sketch_finalize_pipeline.threadExecutionWidth, 32);
    [e dispatchThreadgroups:MTLSizeMake(columns.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    auto counts = static_cast<std::uint32_t const*>(sizes.contents);
    auto output = static_cast<Entry const*>(merged.contents);
    for (std::size_t i = 0; i < columns.size(); ++i) {
      CHECK_LE(counts[i], tasks[i].capacity);
      CHECK_LE(counts[i], columns[i].final_budget);
      result[i].Reserve(counts[i]);
      if (counts[i])
        std::memcpy(result[i].space.data(), output + tasks[i].workspace, counts[i] * sizeof(Entry));
      result[i].SetSize(counts[i]);
    }
    sketch_finalize_dispatches.fetch_add(1, std::memory_order_relaxed);
    sketch_finalize_columns.fetch_add(columns.size(), std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      sketch_finalize_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                            std::memory_order_relaxed);
  }
  return result;
}

std::vector<float> MakeSketchWeights(std::size_t rows, Span<float const> hessians,
                                     Span<float const> weights, Span<bst_group_t const> groups) {
  CHECK(hessians.empty() || hessians.size() == rows);
  if (!groups.empty()) {
    CHECK_GE(groups.size(), 2);
    CHECK_EQ(groups.front(), 0);
    CHECK_EQ(groups.back(), rows);
    CHECK(std::is_sorted(groups.begin(), groups.end()));
    CHECK(weights.empty() || weights.size() == groups.size() - 1);
  } else {
    CHECK(weights.empty() || weights.size() == rows);
  }
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(groups.size(), std::numeric_limits<std::uint32_t>::max());
  std::vector<float> result(rows);
  if (!rows) return result;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto h = Upload(rt, hessians.data(), hessians.size_bytes());
    auto w = Upload(rt, weights.data(), weights.size_bytes());
    auto g = Upload(rt, groups.data(), groups.size_bytes());
    auto out = Allocate(rt, rows * sizeof(float));
    struct {
      std::uint32_t hessian, weighted, groups, padding;
    } p{!hessians.empty(), !weights.empty(), static_cast<std::uint32_t>(groups.size()), 0};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.sketch_weight_pipeline];
    [e setBuffer:h offset:0 atIndex:0];
    [e setBuffer:w offset:0 atIndex:1];
    [e setBuffer:g offset:0 atIndex:2];
    [e setBuffer:out offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &sketch_weight_dispatches, &sketch_ingest_gpu_time_ns);
    std::memcpy(result.data(), out.contents, rows * sizeof(float));
  }
  return result;
}

std::vector<SortedSketchResult> MakeSortedSketches(Span<SortedSketchColumn const> columns,
                                                   Span<float const> weights) {
  CHECK_LE(columns.size(), std::numeric_limits<std::uint32_t>::max());
  std::vector<SortedSketchResult> results(columns.size());
  if (columns.empty()) return results;
  struct Task {
    std::uint64_t begin, count, output;
    std::uint32_t budget, padding;
  };
  static_assert(sizeof(Task) == 32);
  std::vector<Task> tasks;
  std::vector<::xgboost::Entry> entries;
  std::size_t capacity = 0;
  for (auto const& column : columns) {
    CHECK_GT(column.budget, 0);
    CHECK_LT(column.budget, std::numeric_limits<std::uint32_t>::max());
    CHECK_LE(column.entries.size(), std::numeric_limits<std::uint32_t>::max());
    for (std::size_t i = 0; i < column.entries.size(); ++i) {
      auto const& entry = column.entries[i];
      CHECK_LT(entry.index, weights.size());
      CHECK(!std::isnan(entry.fvalue));
      if (i) CHECK_LE(column.entries[i - 1].fvalue, entry.fvalue);
    }
    auto n = std::min(column.entries.size(), column.budget + 1);
    CHECK_LE(n, std::numeric_limits<std::size_t>::max() - capacity);
    tasks.push_back({entries.size(), column.entries.size(), capacity,
                     static_cast<std::uint32_t>(column.budget), 0});
    entries.insert(entries.end(), column.entries.begin(), column.entries.end());
    capacity += n;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    CHECK_LE(capacity, rt.device.maxBufferLength / sizeof(WQSummary<>::Entry));
    auto input = Upload(rt, entries.data(), entries.size() * sizeof(::xgboost::Entry));
    auto w = Upload(rt, weights.data(), weights.size_bytes());
    auto metadata = Upload(rt, tasks.data(), tasks.size() * sizeof(Task));
    auto output = Allocate(rt, capacity * sizeof(WQSummary<>::Entry));
    auto sizes = Allocate(rt, columns.size() * 2 * sizeof(std::uint64_t));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.sketch_sorted_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:w offset:0 atIndex:1];
    [e setBuffer:metadata offset:0 atIndex:2];
    [e setBuffer:output offset:0 atIndex:3];
    [e setBuffer:sizes offset:0 atIndex:4];
    [e dispatchThreads:MTLSizeMake(columns.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &sketch_sorted_dispatches, &sketch_ingest_gpu_time_ns);
    auto counts = static_cast<std::uint64_t const*>(sizes.contents);
    auto data = static_cast<WQSummary<>::Entry const*>(output.contents);
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto count = counts[2 * i];
      CHECK_LE(count, std::min(columns[i].entries.size(), columns[i].budget + 1));
      CHECK_LE(counts[2 * i + 1], columns[i].entries.size());
      results[i].num_elements = counts[2 * i + 1];
      auto& summary = results[i].summary;
      summary.Reserve(count);
      if (count)
        std::memcpy(summary.space.data(), data + tasks[i].output,
                    count * sizeof(WQSummary<>::Entry));
      summary.SetSize(count);
    }
  }
  return results;
}

std::vector<SketchUpdateResult> UpdateSketches(Span<SketchUpdate const> columns) {
  using Entry = WQSummary<>::Entry;
  CHECK_LE(columns.size(), std::numeric_limits<std::uint32_t>::max());
  std::vector<SketchUpdateResult> results(columns.size());
  if (columns.empty()) return results;
  struct Level {
    std::uint64_t begin, size;
  };
  struct Task {
    std::uint64_t levels, incoming, workspace, output, sizes;
    std::uint32_t nlevels, count, limit, padding;
  };
  static_assert(sizeof(Task) == 56);
  std::vector<Entry> entries;
  std::vector<Level> levels;
  std::vector<Task> tasks;
  std::size_t work_size = 0, output_size = 0, sizes_count = 0;
  for (auto const& column : columns) {
    CHECK_GT(column.limit, 0);
    CHECK_LE(column.limit, std::numeric_limits<std::uint32_t>::max() / 2);
    CHECK_LT(column.levels.size(), std::numeric_limits<std::uint32_t>::max());
    CHECK_LE(column.incoming.size(), std::numeric_limits<std::uint32_t>::max());
    auto begin = levels.size();
    for (auto const& level : column.levels) {
      CHECK_LE(level.Size(), column.limit);
      levels.push_back({entries.size(), level.Size()});
      auto source = level.Entries();
      entries.insert(entries.end(), source.begin(), source.end());
    }
    tasks.push_back({begin, entries.size(), work_size, output_size, sizes_count,
                     static_cast<std::uint32_t>(column.levels.size()),
                     static_cast<std::uint32_t>(column.incoming.size()),
                     static_cast<std::uint32_t>(column.limit), 0});
    entries.insert(entries.end(), column.incoming.begin(), column.incoming.end());
    auto n = column.levels.size() + 1;
    CHECK_LE(n, (std::numeric_limits<std::size_t>::max() - output_size) / column.limit);
    CHECK_LE(column.limit * 3, std::numeric_limits<std::size_t>::max() - work_size);
    output_size += n * column.limit;
    work_size += 3 * column.limit;
    sizes_count += n + 2;  // level sizes, used level count and final carry level
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    CHECK_LE(work_size, rt.device.maxBufferLength / sizeof(Entry));
    CHECK_LE(output_size, rt.device.maxBufferLength / sizeof(Entry));
    auto input = Upload(rt, entries.data(), entries.size() * sizeof(Entry));
    auto offsets = Upload(rt, levels.data(), levels.size() * sizeof(Level));
    auto metadata = Upload(rt, tasks.data(), tasks.size() * sizeof(Task));
    auto scratch = Allocate(rt, work_size * sizeof(Entry));
    auto output = Allocate(rt, output_size * sizeof(Entry));
    auto sizes = Allocate(rt, sizes_count * sizeof(std::uint32_t));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.sketch_update_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBuffer:metadata offset:0 atIndex:2];
    [e setBuffer:scratch offset:0 atIndex:3];
    [e setBuffer:output offset:0 atIndex:4];
    [e setBuffer:sizes offset:0 atIndex:5];
    [e dispatchThreads:MTLSizeMake(columns.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &sketch_update_dispatches, &sketch_ingest_gpu_time_ns);
    auto counts = static_cast<std::uint32_t const*>(sizes.contents);
    auto data = static_cast<Entry const*>(output.contents);
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto const& task = tasks[i];
      auto used = counts[task.sizes + task.nlevels + 1];
      auto carry = counts[task.sizes + task.nlevels + 2];
      CHECK_GE(used, task.nlevels);
      CHECK_LE(used, task.nlevels + 1);
      CHECK_LT(carry, used);
      results[i].carry_level = carry;
      results[i].levels.resize(used);
      for (std::size_t l = 0; l < used; ++l) {
        auto count = counts[task.sizes + l];
        CHECK_LE(count, task.limit);
        auto& summary = results[i].levels[l];
        summary.Reserve(count);
        if (count)
          std::memcpy(summary.space.data(), data + task.output + l * task.limit,
                      count * sizeof(Entry));
        summary.SetSize(count);
      }
    }
  }
  return results;
}

void InitFeatureSet(Span<bst_feature_t> features) {
  if (features.empty()) return;
  CHECK_LE(features.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto output = Allocate(rt, features.size_bytes());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.column_iota_pipeline];
    [e setBuffer:output offset:0 atIndex:0];
    [e dispatchThreads:MTLSizeMake(features.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(features.data(), output.contents, features.size_bytes());
    column_init_dispatches.fetch_add(1, std::memory_order_relaxed);
  }
}

void SampleFeatures(Span<bst_feature_t const> features, Span<float const> weights,
                    std::uint32_t seed, Span<bst_feature_t> selected, Span<float> uniforms,
                    Span<float> keys) {
  CHECK_LE(selected.size(), features.size());
  CHECK_LE(features.size(), std::numeric_limits<std::uint32_t>::max());
  CHECK(uniforms.empty() || (!weights.empty() && uniforms.size() == features.size()));
  CHECK(keys.empty() || (!weights.empty() && keys.size() == features.size()));
  if (selected.empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    bool fused = weights.empty() && features.size() <= 2048 &&
                 rt.device.maxThreadgroupMemoryLength >= 2048 * 8 + 624 * 4;
    auto input = Upload(rt, features.data(), features.size_bytes());
    auto output = Allocate(rt, selected.size_bytes());
    auto weight = fused ? nil : Upload(rt, weights.data(), weights.size_bytes());
    auto indices = fused ? nil : Allocate(rt, features.size_bytes());
    auto random = fused ? nil : Allocate(rt, weights.empty() ? 4 : features.size() * sizeof(float));
    auto scores = fused ? nil : Allocate(rt, weights.empty() ? 4 : features.size() * sizeof(float));
    auto invalid = fused ? nil : Allocate(rt, sizeof(std::uint32_t));
    auto records = fused ? nil : Allocate(rt, selected.size() * 2 * sizeof(std::uint32_t));
    struct {
      std::uint32_t rows, selected, seed, weighted;
      std::uint64_t weights;
    } p{static_cast<std::uint32_t>(features.size()), static_cast<std::uint32_t>(selected.size()),
        seed, static_cast<std::uint32_t>(!weights.empty()), weights.size()};
    static_assert(sizeof(p) == 24);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    if (!fused) {
      auto clear = [command blitCommandEncoder];
      CHECK(clear);
      [clear fillBuffer:invalid range:NSMakeRange(0, invalid.length) value:0];
      [clear endEncoding];
    }
    auto finish = [&] {
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      if (!fused)
        CHECK_EQ(*static_cast<std::uint32_t const*>(invalid.contents), 0)
            << "Feature index exceeds weights";
      std::memcpy(selected.data(), output.contents, selected.size_bytes());
      if (!uniforms.empty()) std::memcpy(uniforms.data(), random.contents, uniforms.size_bytes());
      if (!keys.empty()) std::memcpy(keys.data(), scores.contents, keys.size_bytes());
      column_sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
      auto elapsed = command.GPUEndTime - command.GPUStartTime;
      if (elapsed > 0)
        column_sampling_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                              std::memory_order_relaxed);
    };
    if (fused) {
      std::uint32_t width = 2;
      while (width < p.rows) width <<= 1;
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.column_sample_small_pipeline];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:output offset:0 atIndex:1];
      [e setBytes:&p length:sizeof(p) atIndex:2];
      [e setThreadgroupMemoryLength:width * 8 atIndex:0];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      finish();
      return;
    }
    if (!p.weighted) {
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.column_iota_pipeline];
      [e setBuffer:indices offset:0 atIndex:0];
      [e dispatchThreads:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.column_rng_pipeline];
    [e setBuffer:indices offset:0 atIndex:0];
    [e setBuffer:random offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    id<MTLBuffer> ranked = Allocate(rt, 8);
    if (p.weighted) {
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.column_keys_pipeline];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:weight offset:0 atIndex:1];
      [e setBuffer:random offset:0 atIndex:2];
      [e setBuffer:scores offset:0 atIndex:3];
      [e setBuffer:invalid offset:0 atIndex:4];
      [e setBytes:&p length:sizeof(p) atIndex:5];
      [e dispatchThreads:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      std::uint32_t bounds[]{0, p.rows};
      auto offsets = Upload(rt, bounds, sizeof(bounds));
      ranked = EncodeGroupSort(rt, command, scores, offsets, p.rows, 1, 1, p.rows).records;
    }
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.column_select_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:indices offset:0 atIndex:1];
    [e setBuffer:ranked offset:0 atIndex:2];
    [e setBuffer:records offset:0 atIndex:3];
    [e setBytes:&p.weighted length:sizeof(p.weighted) atIndex:4];
    [e dispatchThreads:MTLSizeMake(p.selected, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    // Dynamic threadgroup storage must be aligned to 16 bytes, even for one ID.
    std::uint32_t width = 2;
    while (width < p.selected && width < 4096) width <<= 1;
    if (p.selected <= 4096 && width * 8 <= rt.device.maxThreadgroupMemoryLength) {
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.column_sort_small_pipeline];
      [e setBuffer:records offset:0 atIndex:0];
      [e setBytes:&p.selected length:sizeof(p.selected) atIndex:1];
      [e setBytes:&width length:sizeof(width) atIndex:2];
      [e setThreadgroupMemoryLength:width * 8 atIndex:0];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    } else {
      SortShape shape{p.selected, 1, 1, 1,
                      static_cast<std::uint32_t>((std::uint64_t{p.selected} + 255) / 256)};
      records = EncodeRadixRecords(rt, command, records, shape);
    }
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.column_output_pipeline];
    [e setBuffer:records offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake(p.selected, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    finish();
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

void GradientSample(linalg::MatrixView<GradientPair const> split_gradients,
                    linalg::MatrixView<GradientPair> output, std::size_t sample_rows,
                    std::uint64_t seed, std::uint32_t threads, Span<float> norms,
                    float* threshold) {
  auto rows = split_gradients.Shape(0);
  CHECK_EQ(rows, output.Shape(0));
  CHECK_LE(sample_rows, rows);
  CHECK_GT(threads, 0);
  CHECK(norms.empty() || norms.size() == rows);
  if (!rows || sample_rows == rows) return;
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_GT(split_gradients.Shape(1), 0);
  CHECK_GT(output.Shape(1), 0);
  CHECK_LE(split_gradients.Shape(1), std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(output.Shape(1), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto source_values = split_gradients.Values();
    auto output_values = output.Values();
    auto source = sample_rows ? Upload(rt, source_values.data(), source_values.size_bytes()) : nil;
    bool same = source_values.data() == output_values.data() &&
                source_values.size() == output_values.size() &&
                split_gradients.Shape(1) == output.Shape(1) &&
                split_gradients.Stride(0) == output.Stride(0) &&
                split_gradients.Stride(1) == output.Stride(1);
    auto result =
        same && source ? source : Upload(rt, output_values.data(), output_values.size_bytes());
    auto regularized = Allocate(rt, sample_rows ? rows * sizeof(float) : sizeof(float));
    auto selected = Allocate(rt, sizeof(float));
    struct {
      std::uint64_t rows, source_stride0, source_stride1, output_stride0, output_stride1, seed;
      std::uint32_t source_targets, output_targets, threads, sample_rows;
    } p{rows,
        split_gradients.Stride(0),
        split_gradients.Stride(1),
        output.Stride(0),
        output.Stride(1),
        seed,
        static_cast<std::uint32_t>(split_gradients.Shape(1)),
        static_cast<std::uint32_t>(output.Shape(1)),
        threads,
        static_cast<std::uint32_t>(sample_rows)};
    static_assert(sizeof(p) == 64);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    if (sample_rows) {
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.gradient_sampling_norm_pipeline];
      [e setBuffer:source offset:0 atIndex:0];
      [e setBuffer:regularized offset:0 atIndex:1];
      [e setBytes:&p length:sizeof(p) atIndex:2];
      [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      SortShape shape{rows, 1, 1, 1, static_cast<std::uint32_t>((rows + 255) / 256)};
      auto sorted = EncodeColumnSort(rt, command, regularized, shape);
      auto cdf = Allocate(rt, ((rows + 255) / 256) * sizeof(float));
      auto n = static_cast<std::uint32_t>(rows);
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.gradient_sampling_prefix_pipeline];
      [e setBuffer:sorted offset:0 atIndex:0];
      [e setBuffer:cdf offset:0 atIndex:1];
      [e setBytes:&n length:sizeof(n) atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.gradient_sampling_threshold_pipeline];
      [e setBuffer:sorted offset:0 atIndex:0];
      [e setBuffer:regularized offset:0 atIndex:1];
      [e setBuffer:cdf offset:0 atIndex:2];
      [e setBuffer:selected offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [e endEncoding];
    }
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.gradient_sampling_apply_pipeline];
    [e setBuffer:result offset:0 atIndex:0];
    [e setBuffer:regularized offset:0 atIndex:1];
    [e setBuffer:selected offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(output_values.data(), result.contents, output_values.size_bytes());
    if (sample_rows && !norms.empty())
      std::memcpy(norms.data(), regularized.contents, norms.size_bytes());
    if (sample_rows && threshold) std::memcpy(threshold, selected.contents, sizeof(float));
    sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
    gradient_sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (sample_rows) radix_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0) {
      auto ns = static_cast<std::uint64_t>(elapsed * 1e9);
      sampling_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
      gradient_sampling_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
    }
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
bool ValidateLabels(LabelDomain domain, linalg::MatrixView<float const> labels,
                    std::int64_t n_classes) {
  CHECK_LE(static_cast<std::uint32_t>(domain),
           static_cast<std::uint32_t>(LabelDomain::kClassIndex));
  if (domain == LabelDomain::kAny || labels.Size() == 0) return true;
  CHECK_LE(labels.Shape(0), std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(labels.Shape(1), std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(labels.Size(), std::numeric_limits<std::uint32_t>::max());
  // Validate strides without overflowing the address calculation.
  auto available = labels.Values().size();
  CHECK_GT(available, 0);
  std::size_t last = 0;
  for (std::size_t d = 0; d < 2; ++d) {
    if (labels.Stride(d)) {
      CHECK_LE(labels.Shape(d) - 1, (available - 1 - last) / labels.Stride(d));
      last += (labels.Shape(d) - 1) * labels.Stride(d);
    }
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, labels.Values().data(), labels.Values().size_bytes());
    auto invalid = Allocate(rt, sizeof(std::uint32_t));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto clear = [command blitCommandEncoder];
    CHECK(clear);
    [clear fillBuffer:invalid range:NSMakeRange(0, invalid.length) value:0];
    [clear endEncoding];
    struct {
      std::uint64_t stride0, stride1;
      std::uint32_t columns, domain;
      float class_limit;
      std::uint32_t padding;
    } p{labels.Stride(0),
        labels.Stride(1),
        static_cast<std::uint32_t>(labels.Shape(1)),
        static_cast<std::uint32_t>(domain),
        static_cast<float>(n_classes),
        0};
    static_assert(sizeof(p) == 32);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.label_validation_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:invalid offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(labels.Size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &label_validation_dispatches, &label_validation_gpu_time_ns);
    return *static_cast<std::uint32_t const*>(invalid.contents) == 0;
  }
}
bool BaseScoreToMargin(BaseScoreTransform transform, Span<float> values) {
  auto kind = static_cast<std::uint32_t>(transform);
  CHECK_LE(kind, static_cast<std::uint32_t>(BaseScoreTransform::kPositiveLogarithm));
  if (transform == BaseScoreTransform::kIdentity || values.empty()) return true;
  CHECK_LE(values.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, values.data(), values.size_bytes());
    auto output = Allocate(rt, values.size_bytes());
    auto invalid = Allocate(rt, sizeof(std::uint32_t));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto clear = [command blitCommandEncoder];
    CHECK(clear);
    [clear fillBuffer:invalid range:NSMakeRange(0, invalid.length) value:0];
    [clear endEncoding];
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.base_score_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBuffer:invalid offset:0 atIndex:2];
    [e setBytes:&kind length:sizeof(kind) atIndex:3];
    [e dispatchThreads:MTLSizeMake(values.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &base_score_dispatches, &base_score_gpu_time_ns);
    if (*static_cast<std::uint32_t const*>(invalid.contents)) return false;
    std::memcpy(values.data(), output.contents, values.size_bytes());
    return true;
  }
}
void InitExpectileGradients(float mean, Span<float const> labels, Span<float const> weights,
                            Span<float const> alpha, Span<GradientPair> output) {
  CHECK(!alpha.empty());
  CHECK_LE(alpha.size(), std::numeric_limits<std::uint32_t>::max());
  CHECK_EQ(output.size() % alpha.size(), 0);
  CHECK_EQ(output.size() / alpha.size(), labels.size());
  CHECK(weights.empty() || weights.size() == labels.size());
  CHECK_LE(output.size(), std::numeric_limits<std::uint32_t>::max());
  if (output.empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto label = Upload(rt, labels.data(), labels.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto levels = Upload(rt, alpha.data(), alpha.size_bytes());
    auto out = Allocate(rt, output.size_bytes());
    struct {
      std::uint64_t count;
      std::uint32_t targets, weighted;
      float mean;
      std::uint32_t padding;
    } p{output.size(), static_cast<std::uint32_t>(alpha.size()), !weights.empty(), mean, 0};
    static_assert(sizeof(p) == 24);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.expectile_init_pipeline];
    [e setBuffer:label offset:0 atIndex:0];
    [e setBuffer:weight offset:0 atIndex:1];
    [e setBuffer:levels offset:0 atIndex:2];
    [e setBuffer:out offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(output.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &expectile_init_dispatches, &initialization_gpu_time_ns);
    std::memcpy(output.data(), out.contents, output.size_bytes());
  }
}
void BroadcastTargets(Span<float const> targets, Span<float> output) {
  CHECK(!targets.empty());
  CHECK_LE(targets.size(), std::numeric_limits<std::uint32_t>::max());
  CHECK_EQ(output.size() % targets.size(), 0);
  CHECK_LE(output.size(), std::numeric_limits<std::uint32_t>::max());
  if (output.empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, targets.data(), targets.size_bytes());
    auto out = Allocate(rt, output.size_bytes());
    auto n = static_cast<std::uint32_t>(targets.size());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.broadcast_targets_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    [e setBytes:&n length:sizeof(n) atIndex:2];
    [e dispatchThreads:MTLSizeMake(output.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &prediction_init_dispatches, &initialization_gpu_time_ns);
    std::memcpy(output.data(), out.contents, output.size_bytes());
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

namespace {
class MetalShap final : public Shap {
  Runtime& rt_;
  id<MTLBuffer> nodes_, trees_, categories_, leaves_, roots_, bases_;
  struct Params {
    std::uint32_t features, groups, trees, depth, interactions, margin;
    float quad_nodes[8], quad_weights[8];
  } params_{};
  std::size_t batch_rows_, row_output_, row_stack_;
  static constexpr std::size_t kFrameBytes = 88;

 public:
  MetalShap(ShapModel const& model, std::uint32_t features, std::uint32_t groups, bool interactions)
      : rt_{GetRuntime()} {
    static_assert(sizeof(ShapNode) == 48 && sizeof(ShapTree) == 16 && sizeof(Params) == 88);
    CHECK_GT(groups, 0);
    CHECK_GT(model.depth, 0);
    CHECK_LT(features, std::numeric_limits<std::uint32_t>::max());
    CHECK_LE(model.trees.size(), std::numeric_limits<std::uint32_t>::max());
    CHECK_EQ(model.root_means.size(), groups);
    CHECK_EQ(model.base_scores.size(), groups);
    params_ = {features,
               groups,
               static_cast<std::uint32_t>(model.trees.size()),
               model.depth,
               static_cast<std::uint32_t>(interactions),
               0,
               {},
               {}};
    std::copy_n(model.quadrature_nodes, 8, params_.quad_nodes);
    std::copy_n(model.quadrature_weights, 8, params_.quad_weights);
    auto product = [](std::size_t a, std::size_t b) {
      CHECK_LE(a, std::numeric_limits<std::size_t>::max() / b);
      return a * b;
    };
    auto columns = static_cast<std::size_t>(features) + 1;
    row_output_ = product(product(interactions ? product(columns, columns) : columns, groups), 4);
    row_stack_ = product(product(model.depth, groups), kFrameBytes);
    auto row_temporary = product(product(columns, groups), 4);
    auto row_input = product(columns, 4);
    auto row_margin = product(groups, 4);
    // Bound both the sum of scratch allocations and each Metal buffer, even for
    // interaction matrices much wider than ordinary prediction output.
    std::size_t row_bytes = 0;
    for (auto bytes : {row_output_, row_stack_, row_temporary, row_input, row_margin}) {
      CHECK_LE(bytes, rt_.device.maxBufferLength);
      CHECK_LE(bytes, std::numeric_limits<std::size_t>::max() - row_bytes);
      row_bytes += bytes;
    }
    auto budget = std::min<std::size_t>(64ull << 20, rt_.device.maxBufferLength);
    batch_rows_ = std::max<std::size_t>(1, budget / row_bytes);
    batch_rows_ = std::min<std::size_t>(batch_rows_, 4096);
    batch_rows_ =
        std::min<std::size_t>(batch_rows_, std::numeric_limits<std::uint32_t>::max() / groups);
    @autoreleasepool {
      // Metal validates the pointed-to struct size even when there are no trees.
      ShapNode empty{};
      nodes_ = Upload(rt_, model.nodes.empty() ? &empty : model.nodes.data(),
                      std::max<std::size_t>(1, model.nodes.size()) * sizeof(ShapNode));
      trees_ = Upload(rt_, model.trees.data(), model.trees.size() * sizeof(ShapTree));
      categories_ = Upload(rt_, model.categories.data(), model.categories.size() * 4);
      leaves_ = Upload(rt_, model.leaves.data(), model.leaves.size() * 4);
      roots_ = Upload(rt_, model.root_means.data(), model.root_means.size() * 4);
      bases_ = Upload(rt_, model.base_scores.data(), model.base_scores.size() * 4);
    }
  }
  std::size_t MaxBatchRows() const override { return batch_rows_; }
  void Predict(Span<float const> data, std::size_t rows, Span<float const> base_margin,
               Span<float> output) override {
    CHECK_LE(rows, batch_rows_);
    CHECK_EQ(data.size(), rows * params_.features);
    CHECK_EQ(output.size_bytes(), rows * row_output_);
    CHECK(base_margin.empty() || base_margin.size() == rows * params_.groups);
    if (!rows) return;
    @autoreleasepool {
      auto input = Upload(rt_, data.data(), data.size_bytes());
      auto bases =
          base_margin.empty() ? bases_ : Upload(rt_, base_margin.data(), base_margin.size_bytes());
      auto result = Allocate(rt_, output.size_bytes());
      auto frames = Allocate(rt_, rows * row_stack_);
      auto temporary =
          Allocate(rt_, rows * params_.groups * (std::size_t(params_.features) + 1) * 4);
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      auto p = params_;
      p.margin = !base_margin.empty();
      [e setComputePipelineState:rt_.shap_pipeline];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:nodes_ offset:0 atIndex:1];
      [e setBuffer:trees_ offset:0 atIndex:2];
      [e setBuffer:categories_ offset:0 atIndex:3];
      [e setBuffer:leaves_ offset:0 atIndex:4];
      [e setBuffer:roots_ offset:0 atIndex:5];
      [e setBuffer:bases offset:0 atIndex:6];
      [e setBuffer:result offset:0 atIndex:7];
      [e setBuffer:frames offset:0 atIndex:8];
      [e setBuffer:temporary offset:0 atIndex:9];
      [e setBytes:&p length:sizeof(p) atIndex:10];
      [e dispatchThreads:MTLSizeMake(rows * params_.groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [e endEncoding];
      CompleteOperation(command,
                        params_.interactions ? &shap_interaction_dispatches : &shap_dispatches,
                        &shap_gpu_time_ns);
      std::memcpy(output.data(), result.contents, output.size_bytes());
    }
  }
};
}  // namespace
std::unique_ptr<Shap> MakeShap(ShapModel const& model, std::uint32_t features, std::uint32_t groups,
                               bool interactions) {
  return std::make_unique<MetalShap>(model, features, groups, interactions);
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
