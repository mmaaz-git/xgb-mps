/** Copyright 2026, XGBoost Contributors */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "../data/gradient_index.h"
#include "metal_auc.h"
#include "metal_constraints.h"
#include "metal_cuda_sketch.h"
#include "metal_data.h"
#include "metal_gradient.h"
#include "metal_histogram.h"
#include "metal_leaf.h"
#include "metal_linear.h"
#include "metal_memory.h"
#include "metal_model.h"
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
std::atomic<std::uint64_t> cuda_sketch_dispatches{0}, cuda_sketch_batches{0},
    cuda_sketch_input_entries{0}, cuda_sketch_gpu_time_ns{0};
std::atomic<std::uint64_t> sketch_weight_dispatches{0}, sketch_sorted_dispatches{0},
    sketch_update_dispatches{0}, sketch_ingest_gpu_time_ns{0};
std::atomic<std::uint64_t> gradient_copy_dispatches{0}, hessian_extract_dispatches{0},
    gradient_state_gpu_time_ns{0}, gradient_state_shared_input_bytes{0},
    gradient_state_shared_output_bytes{0}, gradient_state_staging_bytes{0};
std::atomic<std::uint64_t> expectile_init_dispatches{0}, prediction_init_dispatches{0},
    initialization_gpu_time_ns{0};
std::atomic<std::uint64_t> label_validation_dispatches{0}, label_validation_gpu_time_ns{0};
std::atomic<std::uint64_t> leaf_integer_sum_dispatches{0};
std::atomic<std::uint64_t> cuda_integer_split_dispatches{0};
std::atomic<std::uint64_t> cuda_integer_multi_split_dispatches{0};
std::atomic<std::uint64_t> split_finalize_dispatches{0}, split_bounds_dispatches{0},
    split_bounds_init_dispatches{0};
std::atomic<std::uint64_t> split_category_dispatches{0}, split_category_nodes{0};
std::atomic<std::uint64_t> interaction_reset_dispatches{0}, interaction_split_dispatches{0},
    interaction_query_dispatches{0}, interaction_buffer_growths{0}, interaction_gpu_time_ns{0};
std::atomic<std::uint64_t> model_weight_copy_dispatches{0}, model_category_copy_dispatches{0},
    model_copy_words{0}, model_scale_dispatches{0}, model_state_gpu_time_ns{0};
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
std::atomic<std::uint64_t> cuda_gradient_sampling_dispatches{0};
std::atomic<std::uint64_t> quantized_gradient_sum_dispatches{0},
    cuda_uniform_sampling_dispatches{0};
std::atomic<std::uint64_t> histogram_integer_store_dispatches{0},
    histogram_integer_accumulation_dispatches{0}, histogram_integer_subtraction_dispatches{0},
    histogram_integer_sum_dispatches{0};
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
  id<MTLComputePipelineState> split_pipeline, multi_split_pipeline, quantized_split_pipeline;
  id<MTLComputePipelineState> quantized_multi_split_pipeline;
  id<MTLComputePipelineState> split_finalize_pipeline, split_bounds_pipeline,
      split_bounds_init_pipeline;
  id<MTLComputePipelineState> model_scale_pipeline;
  id<MTLComputePipelineState> split_category_pipeline;
  id<MTLComputePipelineState> interaction_restore_pipeline, interaction_split_pipeline,
      interaction_input_pipeline, interaction_filter_pipeline, interaction_scatter_pipeline;
  id<MTLComputePipelineState> dequantize_hist_pipeline, subtract_hist_pipeline,
      gather_hist_pipeline;
  id<MTLComputePipelineState> store_integer_hist_pipeline, subtract_integer_hist_pipeline,
      sum_integer_hist_pipeline;
  id<MTLComputePipelineState> residual_reduce_pipeline, residual_finish_pipeline, adaptive_pipeline;
  id<MTLComputePipelineState> quantile_metric_pipeline, binning_pipeline;
  id<MTLComputePipelineState> mean_weight_pipeline, mean_pipeline, mean_result_pipeline;
  id<MTLComputePipelineState> sort_init_pipeline, radix_rank_pipeline, radix_scatter_pipeline,
      prefix_u32_pipeline, prefix_add_pipeline, quantile_cdf_pipeline, quantile_select_pipeline;
  id<MTLComputePipelineState> roc_sort_init_pipeline, roc_counts_pipeline, roc_area_pipeline,
      pr_area_pipeline, pr_totals_pipeline, auc_result_pipeline, auc_normalize_pipeline,
      prefix_double_pair_pipeline, prefix_double_pair_add_pipeline;
  id<MTLComputePipelineState> group_sort_small_pipeline, group_sort_medium_pipeline,
      group_sort_key_pipeline, group_sort_export_pipeline, ranking_score_pipeline,
      ranking_map_labels_pipeline;
  id<MTLComputePipelineState> ranking_labels_pipeline, ndcg_discount_pipeline, ndcg_ideal_pipeline;
  id<MTLComputePipelineState> ranking_map_stats_pipeline, ranking_pairs_pipeline,
      ranking_normalize_pipeline, ranking_gather_pipeline, ranking_mean_labels_pipeline,
      ranking_mean_sample_pipeline, ranking_mean_links_pipeline, ranking_mean_gather_pipeline,
      ranking_bias_sum_pipeline, ranking_bias_finish_pipeline, ranking_bias_update_pipeline,
      ranking_bias_min_pipeline, ranking_cost_rounding_pipeline;
  id<MTLComputePipelineState> aft_gradient_pipeline, aft_metric_pipeline, aft_transform_pipeline;
  id<MTLComputePipelineState> cs_sort_keys_pipeline;
  id<MTLComputePipelineState> cs_bounds_pipeline;
  id<MTLComputePipelineState> cs_gather_pipeline;
  id<MTLComputePipelineState> cs_prefix_pipeline;
  id<MTLComputePipelineState> cs_prefix_add_pipeline;
  id<MTLComputePipelineState> cs_attach_pipeline;
  id<MTLComputePipelineState> cs_flags_pipeline;
  id<MTLComputePipelineState> cs_compact_pipeline;
  id<MTLComputePipelineState> cs_raw_ranks_pipeline;
  id<MTLComputePipelineState> cs_cut_counts_pipeline;
  id<MTLComputePipelineState> cs_prune_pipeline;
  id<MTLComputePipelineState> cs_scan_input_pipeline;
  id<MTLComputePipelineState> cs_merge_ptr_pipeline;
  id<MTLComputePipelineState> cs_merge_pipeline;
  id<MTLComputePipelineState> cs_fix_pipeline;
  id<MTLComputePipelineState> uniform_sampling_pipeline, sketch_finalize_pipeline;
  id<MTLComputePipelineState> sketch_weight_pipeline, sketch_sorted_pipeline, sketch_update_pipeline;
  id<MTLComputePipelineState> gradient_state_pipeline;
  id<MTLComputePipelineState> cuda_mvs_term_pipeline, cuda_mvs_norm_pipeline,
      cuda_mvs_norm_small_pipeline, cuda_mvs_bound_pipeline, cuda_mvs_scale_pipeline,
      cuda_mvs_quantize_pipeline, prefix_u64_pipeline, prefix_u64_add_pipeline,
      cuda_mvs_candidate_pipeline, cuda_mvs_select_pipeline, cuda_mvs_integer_pipeline,
      cuda_mvs_replay_pipeline;
  id<MTLComputePipelineState> quantized_sum_pipeline, quantized_sum_finish_pipeline,
      cuda_uniform_integer_pipeline, cuda_uniform_replay_pipeline;
  id<MTLComputePipelineState> expectile_init_pipeline, broadcast_targets_pipeline;
  id<MTLComputePipelineState> label_validation_pipeline;
  id<MTLComputePipelineState> base_score_pipeline, shap_pipeline;
  id<MTLComputePipelineState> linear_gradient_pipeline, linear_sum_pipeline,
      linear_residual_pipeline;
  id<MTLComputePipelineState> leaf_sum_pipeline, leaf_sum_finish_pipeline, leaf_weight_pipeline;
  id<MTLComputePipelineState> leaf_integer_sum_pipeline, leaf_integer_sum_finish_pipeline;
  id<MTLComputePipelineState> column_iota_pipeline, column_keys_pipeline, column_select_pipeline,
      column_sort_small_pipeline, column_output_pipeline, column_shuffle_init_pipeline,
      column_shuffle_compact_pipeline;
  id<MTLComputePipelineState> gradient_sampling_norm_pipeline, gradient_sampling_prefix_pipeline,
      gradient_sampling_threshold_pipeline, gradient_sampling_apply_pipeline;
  id<MTLComputePipelineState> ranking_roc_init_pipeline, ranking_roc_merge_pipeline,
      ranking_roc_score_pipeline;
  id<MTLComputePipelineState> ranking_pr_counts_pipeline, ranking_pr_totals_pipeline,
      ranking_pr_area_pipeline, ranking_pr_score_pipeline, prefix_segmented_pair_pipeline,
      prefix_segmented_pair_add_pipeline;
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
      quantized_split_pipeline = make_pipeline(@"xgb_evaluate_quantized_splits");
      quantized_multi_split_pipeline = make_pipeline(@"xgb_evaluate_quantized_multi_splits");
      split_finalize_pipeline = make_pipeline(@"xgb_finalize_split_nodes");
      split_bounds_pipeline = make_pipeline(@"xgb_update_split_bounds");
      split_bounds_init_pipeline = make_pipeline(@"xgb_init_split_bounds");
      model_scale_pipeline = make_pipeline(@"xgb_scale_model_weights");
      split_category_pipeline = make_pipeline(@"xgb_build_split_categories");
      interaction_restore_pipeline = make_pipeline(@"xgb_interaction_restore");
      interaction_split_pipeline = make_pipeline(@"xgb_interaction_split");
      interaction_input_pipeline = make_pipeline(@"xgb_interaction_input");
      interaction_filter_pipeline = make_pipeline(@"xgb_interaction_filter");
      interaction_scatter_pipeline = make_pipeline(@"xgb_interaction_scatter");
      multi_split_pipeline = make_pipeline(@"xgb_evaluate_multi_splits");
      dequantize_hist_pipeline = make_pipeline(@"xgb_histogram_dequantize");
      subtract_hist_pipeline = make_pipeline(@"xgb_histogram_subtract");
      gather_hist_pipeline = make_pipeline(@"xgb_histogram_gather");
      store_integer_hist_pipeline = make_pipeline(@"xgb_histogram_store_integer");
      subtract_integer_hist_pipeline = make_pipeline(@"xgb_histogram_subtract_integer");
      sum_integer_hist_pipeline = make_pipeline(@"xgb_histogram_sum_integer");
      residual_reduce_pipeline = make_pipeline(@"xgb_residual_scale_reduce");
      residual_finish_pipeline = make_pipeline(@"xgb_residual_scale_finish");
      adaptive_pipeline = make_pipeline(@"xgb_adaptive_objective");
      quantile_metric_pipeline = make_pipeline(@"xgb_quantile_metric");
      uniform_sampling_pipeline = make_pipeline(@"xgb_uniform_sample");
      cs_sort_keys_pipeline = make_pipeline(@"xgb_cs_sort_keys");
      cs_bounds_pipeline = make_pipeline(@"xgb_cs_bounds");
      cs_gather_pipeline = make_pipeline(@"xgb_cs_gather");
      cs_prefix_pipeline = make_pipeline(@"xgb_cs_prefix");
      cs_prefix_add_pipeline = make_pipeline(@"xgb_cs_prefix_add");
      cs_attach_pipeline = make_pipeline(@"xgb_cs_attach");
      cs_flags_pipeline = make_pipeline(@"xgb_cs_flags");
      cs_compact_pipeline = make_pipeline(@"xgb_cs_compact");
      cs_raw_ranks_pipeline = make_pipeline(@"xgb_cs_raw_ranks");
      cs_cut_counts_pipeline = make_pipeline(@"xgb_cs_cut_counts");
      cs_prune_pipeline = make_pipeline(@"xgb_cs_prune");
      cs_scan_input_pipeline = make_pipeline(@"xgb_cs_scan_input");
      cs_merge_ptr_pipeline = make_pipeline(@"xgb_cs_merge_ptr");
      cs_merge_pipeline = make_pipeline(@"xgb_cs_merge");
      cs_fix_pipeline = make_pipeline(@"xgb_cs_fix");
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
      linear_sum_pipeline = make_pipeline(@"xgb_linear_sum");
      linear_residual_pipeline = make_pipeline(@"xgb_linear_residual");
      leaf_integer_sum_pipeline = make_pipeline(@"xgb_leaf_integer_sum");
      leaf_integer_sum_finish_pipeline = make_pipeline(@"xgb_leaf_integer_sum_finish");
      leaf_sum_pipeline = make_pipeline(@"xgb_leaf_sum");
      leaf_sum_finish_pipeline = make_pipeline(@"xgb_leaf_sum_finish");
      leaf_weight_pipeline = make_pipeline(@"xgb_leaf_weight");
      column_iota_pipeline = make_pipeline(@"xgb_column_iota");
      column_keys_pipeline = make_pipeline(@"xgb_column_keys");
      column_select_pipeline = make_pipeline(@"xgb_column_select");
      column_sort_small_pipeline = make_pipeline(@"xgb_column_sort_small");
      column_output_pipeline = make_pipeline(@"xgb_column_output");
      column_shuffle_init_pipeline = make_pipeline(@"xgb_column_shuffle_init");
      column_shuffle_compact_pipeline = make_pipeline(@"xgb_column_shuffle_compact");

      cuda_mvs_term_pipeline = make_pipeline(@"xgb_cuda_mvs_term");
      cuda_mvs_norm_pipeline = make_pipeline(@"xgb_cuda_mvs_norm");
      cuda_mvs_norm_small_pipeline = make_pipeline(@"xgb_cuda_mvs_norm_small");
      cuda_mvs_bound_pipeline = make_pipeline(@"xgb_cuda_mvs_bound");
      cuda_mvs_scale_pipeline = make_pipeline(@"xgb_cuda_mvs_scale");
      cuda_mvs_quantize_pipeline = make_pipeline(@"xgb_cuda_mvs_quantize");
      prefix_u64_pipeline = make_pipeline(@"xgb_prefix_u64");
      prefix_u64_add_pipeline = make_pipeline(@"xgb_prefix_u64_add");
      cuda_mvs_candidate_pipeline = make_pipeline(@"xgb_cuda_mvs_candidate");
      cuda_mvs_select_pipeline = make_pipeline(@"xgb_cuda_mvs_select");
      cuda_mvs_integer_pipeline = make_pipeline(@"xgb_cuda_mvs_integer");
      cuda_mvs_replay_pipeline = make_pipeline(@"xgb_cuda_mvs_replay");
      quantized_sum_pipeline = make_pipeline(@"xgb_quantized_sum");
      quantized_sum_finish_pipeline = make_pipeline(@"xgb_quantized_sum_finish");
      cuda_uniform_integer_pipeline = make_pipeline(@"xgb_cuda_uniform_integer");
      cuda_uniform_replay_pipeline = make_pipeline(@"xgb_cuda_uniform_replay");
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
      pr_totals_pipeline = make_pipeline(@"xgb_pr_totals");
      auc_result_pipeline = make_pipeline(@"xgb_auc_result");
      auc_normalize_pipeline = make_pipeline(@"xgb_auc_normalize");
      group_sort_small_pipeline = make_pipeline(@"xgb_group_sort_small");
      group_sort_medium_pipeline = make_pipeline(@"xgb_group_sort_medium");
      group_sort_key_pipeline = make_pipeline(@"xgb_group_sort_key");
      group_sort_export_pipeline = make_pipeline(@"xgb_group_sort_export");
      ranking_score_pipeline = make_pipeline(@"xgb_ranking_score");
      ranking_map_labels_pipeline = make_pipeline(@"xgb_ranking_map_labels");
      ranking_labels_pipeline = make_pipeline(@"xgb_ranking_labels");
      ndcg_discount_pipeline = make_pipeline(@"xgb_ndcg_discount");
      ndcg_ideal_pipeline = make_pipeline(@"xgb_ndcg_ideal");
      ranking_map_stats_pipeline = make_pipeline(@"xgb_ranking_map_stats");
      ranking_pairs_pipeline = make_pipeline(@"xgb_ranking_pairs");
      ranking_normalize_pipeline = make_pipeline(@"xgb_ranking_normalize");
      ranking_gather_pipeline = make_pipeline(@"xgb_ranking_gather");
      ranking_mean_labels_pipeline = make_pipeline(@"xgb_ranking_mean_labels");
      ranking_mean_sample_pipeline = make_pipeline(@"xgb_ranking_mean_sample");
      ranking_mean_links_pipeline = make_pipeline(@"xgb_ranking_mean_links");
      ranking_mean_gather_pipeline = make_pipeline(@"xgb_ranking_mean_gather");
      ranking_bias_sum_pipeline = make_pipeline(@"xgb_ranking_bias_sum");
      ranking_bias_finish_pipeline = make_pipeline(@"xgb_ranking_bias_finish");
      ranking_bias_min_pipeline = make_pipeline(@"xgb_ranking_bias_min");
      ranking_cost_rounding_pipeline = make_pipeline(@"xgb_ranking_cost_rounding");
      ranking_bias_update_pipeline = make_pipeline(@"xgb_ranking_bias_update");
      gradient_sampling_norm_pipeline = make_pipeline(@"xgb_gradient_sampling_norm");
      gradient_sampling_prefix_pipeline = make_pipeline(@"xgb_gradient_sampling_prefix");
      gradient_sampling_threshold_pipeline = make_pipeline(@"xgb_gradient_sampling_threshold");
      gradient_sampling_apply_pipeline = make_pipeline(@"xgb_gradient_sampling_apply");
      ranking_roc_init_pipeline = make_pipeline(@"xgb_ranking_roc_init");
      ranking_roc_merge_pipeline = make_pipeline(@"xgb_ranking_roc_merge");
      ranking_roc_score_pipeline = make_pipeline(@"xgb_ranking_roc_score");
      ranking_pr_counts_pipeline = make_pipeline(@"xgb_ranking_pr_counts");
      ranking_pr_totals_pipeline = make_pipeline(@"xgb_ranking_pr_totals");
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
  // Integers remain authoritative after the first quantized build. `buffer` is
  // its materialized binary64 view for host orchestration/legacy split APIs.
  id<MTLBuffer> integers, scale;
  void UseQuantization(id<MTLBuffer> quantization) {
    CHECK(buffer);
    if (!integers) integers = Allocate(GetRuntime(), buffer.length);
    scale = quantization;
  }
  void Resize(std::size_t size) override {
    if (size <= size_) return;
    auto& rt = GetRuntime();
    CHECK_LE(size, rt.device.maxBufferLength / sizeof(GradientPairPrecise));
    if (!buffer || buffer.length / sizeof(GradientPairPrecise) < size) {
      auto capacity = std::min<std::size_t>(
          std::max(size, size_ * 2), rt.device.maxBufferLength / sizeof(GradientPairPrecise));
      auto next = Allocate(rt, capacity * sizeof(GradientPairPrecise));
      auto next_integers = integers ? Allocate(rt, capacity * sizeof(QuantizedPair)) : nil;
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
        if (integers) {
          [encoder copyFromBuffer:integers
                     sourceOffset:0
                         toBuffer:next_integers
                destinationOffset:0
                             size:size_ * sizeof(QuantizedPair)];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      }
      buffer = next;
      integers = next_integers;
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
  id<MTLBuffer> scales, integer_children;
  id<MTLBuffer> node_input, feature_input, node_parents, node_bounds, node_results, node_weights;
  id<MTLBuffer> category_input, category_cuts, category_output, category_error;
  std::size_t candidate_count{0};
  std::size_t candidate_child_count{0};
  bool vector_candidates{false};
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
                               MetalSplitWorkspace* workspace, bool quantized = false) {
  static_assert(sizeof(HistogramGather) == 24);
  auto output = Scratch(rt, &workspace->hist, size * sizeof(GradientPairPrecise));
  workspace->metadata.resize(inputs.size());
  std::size_t index = 0;
  for (auto const& input : inputs) {
    auto& metadata = workspace->metadata[index++].value;
    if (input.segments.empty()) continue;
    CHECK(input.source);
    auto storage = static_cast<MetalHistogramBuffer const*>(input.source);
    auto source = quantized ? storage->integers : storage->buffer;
    CHECK(source);
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
  id<MTLBuffer> buffer, integers, scales, error;
  std::shared_ptr<GradientQuantizer const> quantizer;
  std::size_t leaves, targets;
  MetalLeafSums(std::size_t n_leaves, std::size_t n_targets,
                std::shared_ptr<GradientQuantizer const> original_quantizer);
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
    if (sums->quantizer) CHECK_EQ(gradients.Shape(0), sums->quantizer->Rows());
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
      [e setComputePipelineState:sums->quantizer ? rt_.leaf_integer_sum_pipeline
                                                 : rt_.leaf_sum_pipeline];
      [e setBuffer:rows_ offset:0 atIndex:0];
      [e setBuffer:g.buffer offset:g.offset atIndex:1];
      [e setBuffer:tasks offset:0 atIndex:2];
      [e setBuffer:partial offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      if (sums->quantizer) {
        [e setBuffer:sums->scales offset:0 atIndex:5];
        [e setBuffer:sums->error offset:0 atIndex:6];
      }
      [e dispatchThreadgroups:MTLSizeMake(blocks.size(), sums->targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:sums->quantizer ? rt_.leaf_integer_sum_finish_pipeline
                                                 : rt_.leaf_sum_finish_pipeline];
      [e setBuffer:partial offset:0 atIndex:0];
      [e setBuffer:ranges offset:0 atIndex:1];
      [e setBuffer:sums->buffer offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      if (sums->quantizer) {
        [e setBuffer:sums->integers offset:0 atIndex:4];
        [e setBuffer:sums->scales offset:0 atIndex:5];
      }
      [e dispatchThreadgroups:MTLSizeMake(segments.size(), sums->targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &leaf_sum_dispatches, &leaf_fitting_gpu_time_ns);
      this->MarkReady();
      if (sums->quantizer) {
        CHECK_EQ(*static_cast<std::uint32_t const*>(sums->error.contents), 0)
            << "Leaf fitting requires finite sampled value gradients within the original "
               "quantizer's int64 range; CUDA's conversion is undefined outside this range.";
        leaf_integer_sum_dispatches.fetch_add(1, std::memory_order_relaxed);
      }
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

template <typename T>
void CheckGradientView(linalg::MatrixView<T> view, bool output);

class MetalQuantizedColumn {
  Runtime& rt_;
  id<MTLBuffer> raw_gradients_, gradient_partial_, gradient_bounds_;

 public:
  id<MTLBuffer> gradients_, gradient_scale_;
  explicit MetalQuantizedColumn(Runtime& rt) : rt_{rt} {}
  void ReleaseScratch() {
    raw_gradients_ = nil;
    gradient_partial_ = nil;
    gradient_bounds_ = nil;
  }
  std::size_t Encode(Span<GradientPair const> gradients, id<MTLCommandBuffer> command,
                     bool quantize) {
    CHECK_LE(gradients.size(), std::numeric_limits<std::uint32_t>::max());
    if (quantize && (!gradients_ || gradients_.length < gradients.size() * sizeof(QuantizedPair)))
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
    if (blocks) {
      [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
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
    if (quantize) {
      encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:rt_.quantize_pipeline];
      [encoder setBuffer:input.buffer offset:input.offset atIndex:0];
      [encoder setBuffer:gradients_ offset:0 atIndex:1];
      [encoder setBytes:&count length:sizeof(count) atIndex:2];
      [encoder setBuffer:gradient_scale_ offset:0 atIndex:3];
      if (count) {
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [encoder endEncoding];
    }
    return shared_bytes;
  }
};

std::vector<std::unique_ptr<MetalQuantizedColumn>> MakeGradientColumns(
    Runtime& rt, linalg::MatrixView<GradientPair const> gradients, bool quantize) {
  std::vector<std::unique_ptr<MetalQuantizedColumn>> columns;
  CheckGradientView(gradients, false);
  auto rows = gradients.Shape(0);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_GT(gradients.Shape(1), 0);
  CHECK_LE(gradients.Shape(1), 65535);
  // Quantization is per target. A GPU layout copy handles C order, padding and
  // broadcasts; its temporary raw floats never own the persistent tree state.
  linalg::Matrix<GradientPair> contiguous;
  if (gradients.Stride(0) != 1) {
    contiguous =
        linalg::Matrix<GradientPair>{gradients.Shape(), DeviceOrd::CPU(), linalg::Order::kF};
    CopyGradients(gradients, contiguous.HostView());
    gradients = contiguous.HostView();
  }
  auto command = [rt.queue commandBuffer];
  CHECK(command);
  std::size_t shared_bytes = 0;
  for (std::size_t t = 0; t < gradients.Shape(1); ++t) {
    auto column = std::make_unique<MetalQuantizedColumn>(rt);
    shared_bytes += column->Encode(gradients.Slice(linalg::All(), t).Values().subspan(0, rows),
                                   command, quantize);
    columns.push_back(std::move(column));
  }
  [command commit];
  [command waitUntilCompleted];
  CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
  for (auto const& column : columns) {
    CHECK(static_cast<GradientScale const*>(column->gradient_scale_.contents)->valid)
        << "Metal quantization requires finite gradients and Hessians.";
    column->ReleaseScratch();
  }
  gradient_bound_dispatches.fetch_add(columns.size(), std::memory_order_relaxed);
  if (quantize) quantization_dispatches.fetch_add(columns.size(), std::memory_order_relaxed);
  (quantize ? histogram_shared_gradient_bytes : leaf_shared_gradient_bytes)
      .fetch_add(shared_bytes, std::memory_order_relaxed);
  if (!quantize && command.GPUEndTime > command.GPUStartTime)
    leaf_fitting_gpu_time_ns.fetch_add(
        static_cast<std::uint64_t>((command.GPUEndTime - command.GPUStartTime) * 1e9),
        std::memory_order_relaxed);
  return columns;
}
class MetalGradientQuantizer final : public GradientQuantizer {
  std::size_t rows_;

 public:
  std::vector<std::unique_ptr<MetalQuantizedColumn>> columns;
  explicit MetalGradientQuantizer(linalg::MatrixView<GradientPair const> original)
      : rows_{original.Shape(0)}, columns{MakeGradientColumns(GetRuntime(), original, false)} {}
  std::size_t Rows() const override { return rows_; }
  std::size_t Targets() const override { return columns.size(); }
};

MetalLeafSums::MetalLeafSums(std::size_t n_leaves, std::size_t n_targets,
                             std::shared_ptr<GradientQuantizer const> original_quantizer)
    : quantizer{std::move(original_quantizer)}, leaves{n_leaves}, targets{n_targets} {
  CHECK_GT(targets, 0);
  CHECK_LE(targets, 65535);
  CHECK_LE(leaves, std::numeric_limits<std::uint32_t>::max());
  if (quantizer) CHECK_EQ(quantizer->Targets(), targets);
  auto& rt = GetRuntime();
  CHECK_LE(leaves, rt.device.maxBufferLength / sizeof(GradientPairPrecise) / targets);
  @autoreleasepool {
    auto bytes = leaves * targets * sizeof(GradientPairPrecise);
    buffer = Allocate(rt, bytes);
    if (quantizer) {
      integers = Allocate(rt, bytes);
      scales = Allocate(rt, targets * sizeof(GradientScale));
      std::uint32_t zero = 0;
      error = Upload(rt, &zero, sizeof(zero));
    }
    if (!leaves) return;
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command blitCommandEncoder];
    CHECK(e);
    [e fillBuffer:buffer range:NSMakeRange(0, bytes) value:0];
    if (quantizer) {
      [e fillBuffer:integers range:NSMakeRange(0, bytes) value:0];
      auto original = static_cast<MetalGradientQuantizer const*>(quantizer.get());
      for (std::size_t t = 0; t < targets; ++t) {
        [e copyFromBuffer:original->columns[t]->gradient_scale_
                 sourceOffset:0
                     toBuffer:scales
            destinationOffset:t * sizeof(GradientScale)
                         size:sizeof(GradientScale)];
      }
    }
    [e endEncoding];
    CompleteOperation(command, &leaf_sum_init_dispatches, &leaf_fitting_gpu_time_ns);
  }
}

struct CudaUniformParams {
  std::uint64_t rows, stride0, stride1;
  std::uint32_t targets, seed;
  float probability;
  std::uint32_t padding;
};
static_assert(sizeof(CudaUniformParams) == 40);

class MetalQuantizedGradients final : public QuantizedGradients {
  Runtime& rt_;
  std::size_t rows_;
  float probability_{1.f};
  std::uint32_t seed_{0};
  bool sampled_{false}, gradient_sampling_{false};
  id<MTLBuffer> norms_, sampling_state_;

 public:
  std::vector<std::unique_ptr<MetalQuantizedColumn>> columns;
  explicit MetalQuantizedGradients(linalg::MatrixView<GradientPair const> gradients)
      : rt_{GetRuntime()}, rows_{gradients.Shape(0)} {
    columns = MakeGradientColumns(rt_, gradients, true);
  }
  std::size_t Rows() const override { return rows_; }
  std::size_t Targets() const override { return columns.size(); }
  void Sum(linalg::VectorView<GradientPairPrecise> output,
           Span<GradientPairInt64> integers) const override {
    CHECK_EQ(output.Size(), columns.size());
    CHECK(integers.empty() || integers.size() == columns.size());
    @autoreleasepool {
      std::uint64_t n = rows_, blocks = (n + 255) / 256;
      auto partial = Allocate(rt_, blocks * sizeof(QuantizedPair));
      auto result = Allocate(rt_, columns.size() * sizeof(GradientPairPrecise));
      auto exact = Allocate(rt_, columns.size() * sizeof(GradientPairInt64));
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      for (std::size_t t = 0; t < columns.size(); ++t) {
        auto const& column = *columns[t];
        auto e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt_.quantized_sum_pipeline];
        [e setBuffer:column.gradients_ offset:0 atIndex:0];
        [e setBuffer:partial offset:0 atIndex:1];
        [e setBytes:&n length:sizeof(n) atIndex:2];
        if (blocks)
          [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt_.quantized_sum_finish_pipeline];
        [e setBuffer:partial offset:0 atIndex:0];
        [e setBuffer:result offset:t * sizeof(GradientPairPrecise) atIndex:1];
        [e setBytes:&blocks length:sizeof(blocks) atIndex:2];
        [e setBuffer:column.gradient_scale_ offset:0 atIndex:3];
        [e setBuffer:exact offset:t * sizeof(GradientPairInt64) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      auto values = static_cast<GradientPairPrecise const*>(result.contents);
      for (std::size_t t = 0; t < columns.size(); ++t) output(t) = values[t];
      if (!integers.empty()) std::memcpy(integers.data(), exact.contents, integers.size_bytes());
      gradient_sum_dispatches.fetch_add(1, std::memory_order_relaxed);
      quantized_gradient_sum_dispatches.fetch_add(1, std::memory_order_relaxed);
      auto elapsed = command.GPUEndTime - command.GPUStartTime;
      if (elapsed > 0)
        gradient_sum_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                           std::memory_order_relaxed);
    }
  }
  void SampleUniform(float probability, std::uint32_t seed) override {
    CHECK(!sampled_) << "Quantized gradients can be sampled only once per tree.";
    CHECK_GE(probability, 0.f);
    CHECK_LE(probability, 1.f);
    probability_ = probability;
    seed_ = seed;
    sampled_ = true;
    if (!rows_ || probability == 1.f) return;
    @autoreleasepool {
      CudaUniformParams p{rows_, 0, 0, 0, seed, probability, 0};
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      for (auto const& column : columns) {
        auto e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt_.cuda_uniform_integer_pipeline];
        [e setBuffer:column->gradients_ offset:0 atIndex:0];
        [e setBytes:&p length:sizeof(p) atIndex:1];
        [e dispatchThreads:MTLSizeMake(rows_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
      cuda_uniform_sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
      auto elapsed = command.GPUEndTime - command.GPUStartTime;
      if (elapsed > 0)
        sampling_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                       std::memory_order_relaxed);
    }
  }
  void SampleGradient(float probability, std::uint32_t seed, Span<float> norms,
                      float* threshold) override;
  void ApplySampling(linalg::MatrixView<GradientPair> values) const override {
    CHECK_EQ(values.Shape(0), rows_);
    CheckGradientView(values, true);
    if (!rows_ || probability_ == 1.f) return;
    @autoreleasepool {
      // Snapshot staging also preserves the caller's padding and offset guards.
      auto span = values.Values();
      auto data = Upload(rt_, span.data(), span.size_bytes());
      CudaUniformParams p{rows_,
                          values.Stride(0),
                          values.Stride(1),
                          static_cast<std::uint32_t>(values.Shape(1)),
                          seed_,
                          probability_,
                          0};
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:gradient_sampling_ ? rt_.cuda_mvs_replay_pipeline
                                                 : rt_.cuda_uniform_replay_pipeline];
      [e setBuffer:data offset:0 atIndex:0];
      [e setBytes:&p length:sizeof(p) atIndex:1];
      if (gradient_sampling_) {
        [e setBuffer:norms_ offset:0 atIndex:2];
        [e setBuffer:sampling_state_ offset:0 atIndex:3];
      }
      [e dispatchThreads:MTLSizeMake(rows_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      std::memcpy(span.data(), data.contents, span.size_bytes());
    }
  }
};

class MetalHistogram final : public Histogram {
  Runtime& rt_;
  id<MTLCommandQueue> queue_;
  id<MTLBuffer> histogram_buffer_, row_buffer_, node_buffer_, block_buffer_;
  std::shared_ptr<MetalQuantizedGradients const> gradients_;
  bst_target_t target_{0};

 public:
  MetalHistogram() : rt_{GetRuntime()} {
    queue_ = rt_.queue;
    CHECK(queue_) << "Metal command queue creation failed.";
  }

  void Reset() override {
    gradients_.reset();
    target_ = 0;
  }
  void BindQuantized(std::shared_ptr<QuantizedGradients const> gradients,
                     bst_target_t target) override {
    CHECK(gradients);
    CHECK_LT(target, gradients->Targets());
    gradients_ = std::static_pointer_cast<MetalQuantizedGradients const>(std::move(gradients));
    target_ = target;
  }

  void Build(std::size_t page_idx, GHistIndexMatrix const& gidx, RowSetCollection const& rows,
             std::vector<bst_node_t> const& nodes, Span<GradientPair const> gradients,
             std::vector<GHistRow> const& output, HistogramBuffer* storage) override {
    @autoreleasepool {
      CHECK_EQ(nodes.size(), output.size());
      CHECK_LE(nodes.size(), std::numeric_limits<std::uint32_t>::max());
      auto staging_start = Clock::now();
      if (!gradients_) {
        this->BindQuantized(MakeQuantizedGradients(linalg::MakeTensorView(
                                DeviceOrd::CPU(), gradients, gradients.size(), 1)),
                            0);
      }
      CHECK_EQ(gradients_->Rows(), gradients.size());
      auto const& quantized = *gradients_->columns[target_];
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
      if (nodes.empty() || bins == 0) return;
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
      [encoder setBuffer:quantized.gradients_ offset:0 atIndex:3];
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
      } else if (total_rows) {
        [encoder dispatchThreads:MTLSizeMake(total_rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      }
      [encoder endEncoding];
      CHECK(storage);
      auto shared = static_cast<MetalHistogramBuffer*>(storage);
      shared->UseQuantization(quantized.gradient_scale_);
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
      [encoder setComputePipelineState:rt_.store_integer_hist_pipeline];
      [encoder setBuffer:histogram offset:0 atIndex:0];
      [encoder setBuffer:shared->buffer offset:0 atIndex:1];
      [encoder setBuffer:offset_buffer offset:0 atIndex:2];
      [encoder setBytes:&dequantize length:sizeof(dequantize) atIndex:3];
      [encoder setBuffer:quantized.gradient_scale_ offset:0 atIndex:4];
      [encoder setBuffer:shared->integers offset:0 atIndex:5];
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
      histogram_conversion_dispatches.fetch_add(1, std::memory_order_relaxed);
      histogram_integer_store_dispatches.fetch_add(1, std::memory_order_relaxed);
      if (page_idx)
        histogram_integer_accumulation_dispatches.fetch_add(1, std::memory_order_relaxed);
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

std::shared_ptr<GradientQuantizer> MakeGradientQuantizer(
    linalg::MatrixView<GradientPair const> original) {
  @autoreleasepool {
    return std::make_shared<MetalGradientQuantizer>(original);
  }
}

std::shared_ptr<QuantizedGradients> MakeQuantizedGradients(
    linalg::MatrixView<GradientPair const> gradients) {
  @autoreleasepool {
    return std::make_shared<MetalQuantizedGradients>(gradients);
  }
}

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
    auto shared = static_cast<MetalHistogramBuffer*>(storage);
    bool quantized = shared->scale != nil;
    auto buffer = quantized ? shared->integers : shared->buffer;
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
    [encoder setComputePipelineState:quantized ? rt.subtract_integer_hist_pipeline
                                               : rt.subtract_hist_pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    if (quantized) {
      [encoder setBuffer:shared->buffer offset:0 atIndex:2];
      [encoder setBuffer:shared->scale offset:0 atIndex:3];
    }
    [encoder dispatchThreads:MTLSizeMake(bins, tasks.size(), 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    histogram_subtraction_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (quantized) histogram_integer_subtraction_dispatches.fetch_add(1, std::memory_order_relaxed);
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
          gradient_state_staging_bytes.load(std::memory_order_relaxed),
          histogram_integer_store_dispatches.load(std::memory_order_relaxed),
          histogram_integer_accumulation_dispatches.load(std::memory_order_relaxed),
          histogram_integer_subtraction_dispatches.load(std::memory_order_relaxed),
          histogram_integer_sum_dispatches.load(std::memory_order_relaxed),
          quantized_gradient_sum_dispatches.load(std::memory_order_relaxed),
          cuda_uniform_sampling_dispatches.load(std::memory_order_relaxed),
          cuda_gradient_sampling_dispatches.load(std::memory_order_relaxed),
          leaf_integer_sum_dispatches.load(std::memory_order_relaxed),
          cuda_integer_split_dispatches.load(std::memory_order_relaxed),
          cuda_integer_multi_split_dispatches.load(std::memory_order_relaxed),
          split_finalize_dispatches.load(std::memory_order_relaxed),
          split_bounds_dispatches.load(std::memory_order_relaxed),
          model_weight_copy_dispatches.load(std::memory_order_relaxed),
          model_category_copy_dispatches.load(std::memory_order_relaxed),
          model_copy_words.load(std::memory_order_relaxed),
          model_scale_dispatches.load(std::memory_order_relaxed),
          model_state_gpu_time_ns.load(std::memory_order_relaxed),
          split_category_dispatches.load(std::memory_order_relaxed),
          split_category_nodes.load(std::memory_order_relaxed),
          interaction_reset_dispatches.load(std::memory_order_relaxed),
          interaction_split_dispatches.load(std::memory_order_relaxed),
          interaction_query_dispatches.load(std::memory_order_relaxed),
          interaction_buffer_growths.load(std::memory_order_relaxed),
          interaction_gpu_time_ns.load(std::memory_order_relaxed),
          split_bounds_init_dispatches.load(std::memory_order_relaxed),
          cuda_sketch_dispatches.load(std::memory_order_relaxed),
          cuda_sketch_batches.load(std::memory_order_relaxed),
          cuda_sketch_input_entries.load(std::memory_order_relaxed),
          cuda_sketch_gpu_time_ns.load(std::memory_order_relaxed)};
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
void CopyModelBuffer(void* output, std::size_t words, Span<ModelCopy const> copies,
                     ModelBufferKind kind) {
  CHECK(kind == ModelBufferKind::kWeights || kind == ModelBufferKind::kCategories);
  CHECK_LE(words, std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t));
  CHECK(output || words == 0);
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  for (auto const& copy : copies) {
    CHECK_LE(copy.destination, words);
    CHECK_LE(copy.count, words - copy.destination);
    CHECK(copy.source || copy.count == 0);
    if (copy.count) ranges.emplace_back(copy.destination, copy.destination + copy.count);
  }
  std::sort(ranges.begin(), ranges.end());
  std::size_t copied = 0, end = 0;
  for (auto const& range : ranges) {
    CHECK_GE(range.first, end) << "Overlapping model copy destinations";
    copied += range.second - range.first;
    end = range.second;
  }
  if (!copied) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto bytes = words * sizeof(std::uint32_t);
    auto out = TryBorrowMemory(rt.device, output, bytes, alignof(std::uint32_t));
    bool copy_out = !out.buffer;
    if (copy_out) out = {Upload(rt, output, bytes), 0};
    std::vector<BorrowedMemory> sources;
    sources.reserve(copies.size());
    // Retain every source, including snapshots of aliases, until completion.
    for (auto const& copy : copies) {
      auto size = copy.count * sizeof(std::uint32_t);
      if (!size) {
        sources.push_back({});
        continue;
      }
      auto in = MemoryRangesOverlap(copy.source, size, output, bytes)
                    ? BorrowedMemory{}
                    : TryBorrowMemory(rt.device, copy.source, size, alignof(std::uint32_t));
      if (!in.buffer) in = {Upload(rt, copy.source, size), 0};
      sources.push_back(in);
    }
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto encoder = [command blitCommandEncoder];
    CHECK(encoder);
    for (std::size_t i = 0; i < copies.size(); ++i) {
      auto const& copy = copies[i];
      if (!copy.count) continue;
      [encoder copyFromBuffer:sources[i].buffer
                 sourceOffset:sources[i].offset
                     toBuffer:out.buffer
            destinationOffset:out.offset + copy.destination * sizeof(std::uint32_t)
                         size:copy.count * sizeof(std::uint32_t)];
    }
    [encoder endEncoding];
    CompleteOperation(command,
                      kind == ModelBufferKind::kWeights ? &model_weight_copy_dispatches
                                                        : &model_category_copy_dispatches,
                      &model_state_gpu_time_ns);
    if (copy_out) std::memcpy(output, out.buffer.contents, bytes);
    model_copy_words.fetch_add(copied, std::memory_order_relaxed);
  }
}

void ScaleModelWeights(Span<float> weights, float eta) {
  if (weights.empty()) return;
  CHECK(weights.data());
  CHECK_LE(weights.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto out = TryBorrowMemory(rt.device, weights.data(), weights.size_bytes(), alignof(float));
    bool copy_out = !out.buffer;
    if (copy_out) out = {Upload(rt, weights.data(), weights.size_bytes()), 0};
    struct {
      std::uint32_t count, eta;
    } p{static_cast<std::uint32_t>(weights.size()), 0};
    std::memcpy(&p.eta, &eta, sizeof(eta));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt.model_scale_pipeline];
    [encoder setBuffer:out.buffer offset:out.offset atIndex:0];
    [encoder setBytes:&p length:sizeof(p) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(p.count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    CompleteOperation(command, &model_scale_dispatches, &model_state_gpu_time_ns);
    if (copy_out) std::memcpy(weights.data(), out.buffer.contents, weights.size_bytes());
  }
}

std::unique_ptr<LeafSums> MakeLeafSums(
    std::size_t leaves, std::size_t targets,
    std::shared_ptr<GradientQuantizer const> original_quantizer) {
  return std::make_unique<MetalLeafSums>(leaves, targets, std::move(original_quantizer));
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
  GradientPair Gradient(std::int32_t feature, std::uint32_t group) override {
    auto p = Parameters(feature, group);
    if (!p.count) return {};
    // CUDA SumReduction takes a signed int length. Avoid silently narrowing it.
    CHECK_LE(p.count, std::numeric_limits<std::int32_t>::max());
    CHECK_EQ(rt_.linear_gradient_pipeline.threadExecutionWidth, 32);
    CHECK_EQ(rt_.linear_sum_pipeline.threadExecutionWidth, 32);
    auto blocks = (p.count + 255) / 256;
    @autoreleasepool {
      auto partial = Allocate(rt_, blocks * sizeof(GradientPair));
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
      // All reduction levels accumulate GradientPair (FP32), like CUDA's CUB
      // reduction. CUB tile/order choices vary with architecture and toolkit.
      while (blocks > 1) {
        auto next_blocks = (blocks + 255) / 256;
        auto next = Allocate(rt_, next_blocks * sizeof(GradientPair));
        e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt_.linear_sum_pipeline];
        [e setBuffer:partial offset:0 atIndex:0];
        [e setBuffer:next offset:0 atIndex:1];
        [e setBytes:&blocks length:sizeof(blocks) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(next_blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        partial = next;
        blocks = next_blocks;
      }
      CompleteOperation(command, &linear_gradient_dispatches, &linear_gpu_time_ns);
      return *static_cast<GradientPair const*>(partial.contents);
    }
  }
  void UpdateResidual(std::int32_t feature, std::uint32_t group, float delta) override {
    auto p = Parameters(feature, group, delta);
    if (!p.count || (p.bias && delta == 0.f)) return;
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
    bool quantized = shared->scale != nil;
    [e setComputePipelineState:quantized ? rt.sum_integer_hist_pipeline
                                         : rt.histogram_sum_pipeline];
    [e setBuffer:quantized ? shared->integers : shared->buffer offset:0 atIndex:0];
    if (quantized) [e setBuffer:shared->scale offset:0 atIndex:3];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBytes:range length:sizeof(range) atIndex:2];
    [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    CompleteGradientSum(command);
    if (quantized) histogram_integer_sum_dispatches.fetch_add(1, std::memory_order_relaxed);
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
class MetalInteractionConstraints final : public InteractionConstraints {
  Runtime& rt_;
  bool enabled_;
  std::uint32_t features_, words_;
  std::size_t capacity_{0};
  std::vector<std::uint32_t> group_counts_;
  id<MTLBuffer> group_ptr_, group_features_, feature_ptr_, feature_groups_;
  id<MTLBuffer> nodes_{nil}, feature_mask_{nil}, input_mask_{nil}, input_{nil}, flags_{nil},
      output_{nil}, count_{nil}, error_{nil};

  bool Grow(id<MTLCommandBuffer> command, std::size_t required) {
    if (required <= capacity_) return false;
    auto capacity = std::max(required, std::max<std::size_t>(1, capacity_ * 2));
    CHECK_LE(capacity, std::numeric_limits<std::size_t>::max() / 4 / words_);
    auto replacement = Allocate(rt_, capacity * words_ * 4);
    auto blit = [command blitCommandEncoder];
    CHECK(blit);
    [blit fillBuffer:replacement range:NSMakeRange(0, capacity * words_ * 4) value:0];
    if (capacity_) {
      [blit copyFromBuffer:nodes_
               sourceOffset:0
                   toBuffer:replacement
          destinationOffset:0
                       size:capacity_ * words_ * 4];
    }
    [blit endEncoding];
    nodes_ = replacement;
    capacity_ = capacity;
    return true;
  }
  void Finish(id<MTLCommandBuffer> command, std::atomic<std::uint64_t>* counter, bool grew) {
    CompleteOperation(command, counter, &interaction_gpu_time_ns);
    if (grew) interaction_buffer_growths.fetch_add(1, std::memory_order_relaxed);
  }
  std::vector<bst_feature_t> Filter(Span<bst_feature_t const> input, bst_node_t node, bool all) {
    CHECK_GE(node, 0);
    CHECK_LE(input.size(), std::numeric_limits<std::uint32_t>::max());
    if (!features_) return {};
    @autoreleasepool {
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      auto grew = Grow(command, std::size_t(node) + 1);
      auto mask = Scratch(rt_, &input_mask_, words_ * 4ul);
      auto source = Scratch(rt_, &input_, input.size_bytes(), input.data());
      auto flags = Scratch(rt_, &flags_, features_ * 4ul);
      auto output = Scratch(rt_, &output_, features_ * 4ul);
      std::uint32_t zero = 0;
      auto error = Scratch(rt_, &error_, 4, &zero);
      auto count = Scratch(rt_, &count_, 4);
      auto blit = [command blitCommandEncoder];
      CHECK(blit);
      [blit fillBuffer:mask range:NSMakeRange(0, words_ * 4ul) value:0];
      [blit endEncoding];
      struct {
        std::uint32_t features, words, node, count, all;
      } p{features_, words_, static_cast<std::uint32_t>(node),
          static_cast<std::uint32_t>(input.size()), all};
      if (!all && !input.empty()) {
        auto e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt_.interaction_input_pipeline];
        [e setBuffer:source offset:0 atIndex:0];
        [e setBuffer:mask offset:0 atIndex:1];
        [e setBytes:&p length:sizeof(p) atIndex:2];
        [e setBuffer:error offset:0 atIndex:3];
        [e dispatchThreads:MTLSizeMake(input.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.interaction_filter_pipeline];
      [e setBuffer:nodes_ offset:0 atIndex:0];
      [e setBuffer:mask offset:0 atIndex:1];
      [e setBuffer:flags offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      [e dispatchThreads:MTLSizeMake(features_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto prefix = EncodePrefixSum(rt_, command, flags, features_);
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.interaction_scatter_pipeline];
      [e setBuffer:flags offset:0 atIndex:0];
      [e setBuffer:prefix offset:0 atIndex:1];
      [e setBuffer:output offset:0 atIndex:2];
      [e setBuffer:count offset:0 atIndex:3];
      [e setBytes:&features_ length:sizeof(features_) atIndex:4];
      [e dispatchThreads:MTLSizeMake(features_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      Finish(command, &interaction_query_dispatches, grew);
      CHECK_EQ(*static_cast<std::uint32_t const*>(error.contents), 0)
          << "Interaction query feature is out of range";
      auto size = *static_cast<std::uint32_t const*>(count.contents);
      CHECK_LE(size, features_);
      auto data = static_cast<bst_feature_t const*>(output.contents);
      return {data, data + size};
    }
  }

 public:
  MetalInteractionConstraints(bst_feature_t features,
                              std::vector<std::vector<bst_feature_t>> const& groups, bool enabled)
      : rt_{GetRuntime()},
        enabled_{enabled},
        features_{features},
        words_{static_cast<std::uint32_t>((std::uint64_t{features} + 31) / 32)} {
    if (!enabled_) return;
    CHECK_LE(features, std::numeric_limits<std::int32_t>::max());
    CHECK_LE(groups.size(), std::numeric_limits<std::uint32_t>::max());
    // CUDA Configure builds both CSR mappings on the host. Only metadata is
    // constructed here; no allowed-feature masks or unions are computed on CPU.
    std::vector<std::uint64_t> group_ptr{0}, feature_ptr{0};
    std::vector<bst_feature_t> group_features, feature_groups;
    std::vector<std::vector<std::uint32_t>> memberships(features);
    for (std::size_t g = 0; g < groups.size(); ++g) {
      for (auto feature : groups[g]) {
        CHECK_LT(feature, features);
        group_features.push_back(feature);
        auto& ids = memberships[feature];
        if (ids.empty() || ids.back() != g) ids.push_back(g);
      }
      group_ptr.push_back(group_features.size());
    }
    for (auto const& ids : memberships) {
      group_counts_.push_back(ids.size());
      feature_groups.insert(feature_groups.end(), ids.begin(), ids.end());
      feature_ptr.push_back(feature_groups.size());
    }
    @autoreleasepool {
      group_ptr_ = Upload(rt_, group_ptr.data(), group_ptr.size() * 8);
      group_features_ = Upload(rt_, group_features.data(), group_features.size() * 4);
      feature_ptr_ = Upload(rt_, feature_ptr.data(), feature_ptr.size() * 8);
      feature_groups_ = Upload(rt_, feature_groups.data(), feature_groups.size() * 4);
    }
  }
  void Reset() override {
    if (!enabled_ || !features_) return;
    @autoreleasepool {
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      auto grew = Grow(command, 1);
      if (!grew) {
        auto blit = [command blitCommandEncoder];
        CHECK(blit);
        [blit fillBuffer:nodes_ range:NSMakeRange(0, capacity_ * words_ * 4) value:0];
        [blit endEncoding];
      }
      Finish(command, &interaction_reset_dispatches, grew);
    }
  }
  void Split(bst_node_t node, bst_feature_t feature, bst_node_t left, bst_node_t right) override {
    if (!enabled_) return;
    CHECK_GE(node, 0);
    CHECK_GE(left, 0);
    CHECK_GE(right, 0);
    CHECK_NE(node, left);
    CHECK_NE(node, right);
    CHECK_LT(feature, features_);
    @autoreleasepool {
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      auto grew = Grow(command, std::size_t(std::max({node, left, right})) + 1);
      auto mask = Scratch(rt_, &feature_mask_, words_ * 4ul);
      auto blit = [command blitCommandEncoder];
      CHECK(blit);
      [blit fillBuffer:mask range:NSMakeRange(0, words_ * 4ul) value:0];
      [blit endEncoding];
      if (group_counts_[feature]) {
        auto e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt_.interaction_restore_pipeline];
        [e setBuffer:group_ptr_ offset:0 atIndex:0];
        [e setBuffer:group_features_ offset:0 atIndex:1];
        [e setBuffer:feature_ptr_ offset:0 atIndex:2];
        [e setBuffer:feature_groups_ offset:0 atIndex:3];
        [e setBuffer:mask offset:0 atIndex:4];
        [e setBytes:&feature length:sizeof(feature) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(group_counts_[feature], 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      struct {
        std::uint32_t words, node, left, right, feature;
      } p{words_, static_cast<std::uint32_t>(node), static_cast<std::uint32_t>(left),
          static_cast<std::uint32_t>(right), feature};
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.interaction_split_pipeline];
      [e setBuffer:mask offset:0 atIndex:0];
      [e setBuffer:nodes_ offset:0 atIndex:1];
      [e setBytes:&p length:sizeof(p) atIndex:2];
      [e dispatchThreads:MTLSizeMake(words_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      Finish(command, &interaction_split_dispatches, grew);
    }
  }
  std::vector<bst_feature_t> Query(Span<bst_feature_t const> input, bst_node_t node) override {
    // CUDA preserves the original list, including order/duplicates, at root or
    // when disabled. QueryNode deliberately does not use this root bypass.
    if (!enabled_ || node == 0) {
      if (input.empty()) return {};
      return {input.begin(), input.end()};
    }
    return Filter(input, node, false);
  }
  std::vector<bst_feature_t> QueryNode(bst_node_t node) override {
    return enabled_ ? Filter({}, node, true) : std::vector<bst_feature_t>{};
  }
};
}  // namespace
std::unique_ptr<InteractionConstraints> MakeInteractionConstraints(
    bst_feature_t features, std::vector<std::vector<bst_feature_t>> const& groups, bool enabled) {
  return std::make_unique<MetalInteractionConstraints>(features, groups, enabled);
}
namespace {
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
struct CudaMvsParams {
  std::uint64_t rows;
  std::uint32_t targets, sample_rows;
};
struct CudaMvsState {
  std::int32_t exponent;
  std::uint32_t error, index;
  float threshold;
};
static_assert(sizeof(CudaMvsState) == 16 && sizeof(CudaMvsParams) == 16);
id<MTLBuffer> EncodePrefixU64(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                              std::uint64_t count) {
  CHECK_GT(count, 0);
  auto blocks = (count + 255) / 256;
  auto output = Allocate(rt, count * sizeof(std::uint64_t));
  auto totals = Allocate(rt, blocks * sizeof(std::uint64_t));
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:rt.prefix_u64_pipeline];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:output offset:0 atIndex:1];
  [e setBuffer:totals offset:0 atIndex:2];
  [e setBytes:&count length:sizeof(count) atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  if (blocks > 1) {
    auto upper = EncodePrefixU64(rt, command, totals, blocks);
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.prefix_u64_add_pipeline];
    [e setBuffer:output offset:0 atIndex:0];
    [e setBuffer:upper offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
  }
  return output;
}
void MetalQuantizedGradients::SampleGradient(float probability, std::uint32_t seed,
                                             Span<float> norms, float* threshold) {
  CHECK(!sampled_) << "Quantized gradients can be sampled only once per tree.";
  CHECK_GE(probability, 0.f);
  CHECK_LE(probability, 1.f);
  CHECK(norms.empty() || norms.size() == rows_);
  probability_ = probability;
  seed_ = seed;
  sampled_ = true;
  if (probability == 1.f) return;
  gradient_sampling_ = true;
  @autoreleasepool {
    auto count = std::uint64_t{rows_}, blocks = (count + 255) / 256;
    CudaMvsParams p{count, static_cast<std::uint32_t>(columns.size()),
                    static_cast<std::uint32_t>(count * probability)};
    CHECK_LE(p.sample_rows, count);
    CudaMvsState initial{0, 0, static_cast<std::uint32_t>(count),
                         std::numeric_limits<float>::infinity()};
    sampling_state_ = Upload(rt_, &initial, sizeof(initial));
    norms_ = Allocate(rt_, count * sizeof(float));
    if (!count) {
      if (threshold) *threshold = initial.threshold;
      return;
    }
    CHECK_LE(count, std::numeric_limits<std::size_t>::max() / columns.size() / sizeof(float));
    auto terms = Allocate(rt_, count * columns.size() * sizeof(float));
    auto partial = Allocate(rt_, blocks * sizeof(GradientPairPrecise));
    auto total = Allocate(rt_, sizeof(GradientPairPrecise));
    auto candidates = Allocate(rt_, count * sizeof(float));
    auto command = [rt_.queue commandBuffer];
    CHECK(command);
    for (std::size_t t = 0; t < columns.size(); ++t) {
      auto const& column = *columns[t];
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.cuda_mvs_term_pipeline];
      [e setBuffer:column.gradients_ offset:0 atIndex:0];
      [e setBuffer:column.gradient_scale_ offset:0 atIndex:1];
      [e setBuffer:terms offset:t * count * sizeof(float) atIndex:2];
      [e setBytes:&count length:sizeof(count) atIndex:3];
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:p.targets <= 16 ? rt_.cuda_mvs_norm_small_pipeline
                                               : rt_.cuda_mvs_norm_pipeline];
    [e setBuffer:terms offset:0 atIndex:0];
    [e setBuffer:norms_ offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    if (p.targets <= 16)
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    else
      [e dispatchThreadgroups:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(p.targets <= 512 ? 32 : 256, 1, 1)];
    [e endEncoding];
    auto sorted = EncodeColumnSort(rt_, command, norms_,
                                   SortShape{count, 1, 1, 1, static_cast<std::uint32_t>(blocks)});
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt_.cuda_mvs_bound_pipeline];
    [e setBuffer:sorted offset:0 atIndex:0];
    [e setBuffer:partial offset:0 atIndex:1];
    [e setBytes:&count length:sizeof(count) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    struct {
      std::uint64_t rows, stride0, stride1;
      std::uint32_t targets, blocks;
    } reduction{count, 0, 0, 1, static_cast<std::uint32_t>(blocks)};
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt_.gradient_sum_finish_pipeline];
    [e setBuffer:partial offset:0 atIndex:0];
    [e setBuffer:total offset:0 atIndex:1];
    [e setBytes:&reduction length:sizeof(reduction) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt_.cuda_mvs_scale_pipeline];
    [e setBuffer:total offset:0 atIndex:0];
    [e setBuffer:sampling_state_ offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    if (p.sample_rows) {
      auto fixed = Allocate(rt_, count * sizeof(std::uint64_t));
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.cuda_mvs_quantize_pipeline];
      [e setBuffer:sorted offset:0 atIndex:0];
      [e setBuffer:fixed offset:0 atIndex:1];
      [e setBuffer:sampling_state_ offset:0 atIndex:2];
      [e setBytes:&count length:sizeof(count) atIndex:3];
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto prefix = EncodePrefixU64(rt_, command, fixed, count);
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.cuda_mvs_candidate_pipeline];
      [e setBuffer:sorted offset:0 atIndex:0];
      [e setBuffer:prefix offset:0 atIndex:1];
      [e setBuffer:sampling_state_ offset:0 atIndex:2];
      [e setBuffer:candidates offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt_.cuda_mvs_select_pipeline];
    [e setBuffer:sorted offset:0 atIndex:0];
    [e setBuffer:candidates offset:0 atIndex:1];
    [e setBuffer:sampling_state_ offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    CudaUniformParams apply{count, 0, 0, 0, seed, probability, 0};
    for (auto const& column : columns) {
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.cuda_mvs_integer_pipeline];
      [e setBuffer:column->gradients_ offset:0 atIndex:0];
      [e setBytes:&apply length:sizeof(apply) atIndex:1];
      [e setBuffer:column->gradient_scale_ offset:0 atIndex:2];
      [e setBuffer:norms_ offset:0 atIndex:3];
      [e setBuffer:sampling_state_ offset:0 atIndex:4];
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    auto info = static_cast<CudaMvsState const*>(sampling_state_.contents);
    CHECK_EQ(info->error & 1u, 0)
        << "Gradient-based sampling requires finite FP32 norms; "
           "nonfinite norms make CUDA's float-to-integer prefix conversion undefined.";
    CHECK_EQ(info->error & 2u, 0)
        << "Gradient-based sampling rescaling exceeds int64 range; "
           "CUDA's float-to-integer conversion is undefined for these values.";
    if (!norms.empty()) std::memcpy(norms.data(), norms_.contents, norms.size_bytes());
    if (threshold) *threshold = info->threshold;
    sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
    gradient_sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
    cuda_gradient_sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
    radix_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0) {
      auto ns = static_cast<std::uint64_t>(elapsed * 1e9);
      sampling_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
      gradient_sampling_gpu_time_ns.fetch_add(ns, std::memory_order_relaxed);
    }
  }
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
    // Like cuda_impl::RankingAUC, preserve NaN from overflowing squared weights.
    // The shared metric finalizer checks finite scores after collective aggregation.
    return *static_cast<RankingAUCSums const*>(output.contents);
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

namespace {
id<MTLBuffer> EncodeSegmentedDoublePrefix(Runtime& rt, id<MTLCommandBuffer> command,
                                          id<MTLBuffer> input, id<MTLBuffer> ids,
                                          std::uint32_t rows);
}  // namespace

void RankingGradient(RankingLossKind kind, Span<float const> predictions,
                     linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                     Span<float const> weights, std::uint32_t pairs, bool normalization,
                     bool score_normalization, bool exponential, double weight_norm,
                     Span<double const> discounts, Span<double const> inverse_idcg,
                     Span<GradientPair> output, bool mean, std::uint32_t seed, RankingBias bias,
                     Span<std::array<std::uint32_t, 2>> mean_pair_trace,
                     RankingGradientTrace diagnostics) {
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
  CHECK(mean_pair_trace.empty() ||
        (mean && mean_pair_trace.size() == predictions.size() * std::uint64_t{pairs}));
  static_assert(sizeof(RankingGroupStatistics) == 32);
  CHECK(diagnostics.groups.empty() || diagnostics.groups.size() == n_groups);
  CHECK(diagnostics.global_cost_rounding.empty() ||
        (positions && diagnostics.global_cost_rounding.size() == 1));
  CHECK(diagnostics.map_prefix.empty() ||
        (kind == RankingLossKind::kMAP && diagnostics.map_prefix.size() == predictions.size()));
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
  auto budget_pairs =
      (64ull << 20) / (sizeof(GradientPair) + (mean ? 16 : 0) + (positions ? 24 : 0));
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
    auto costs = Allocate(rt, positions ? max_pairs * 8 : 8);
    auto minimum_bias = Allocate(rt, 8);
    auto cost_rounding = Allocate(rt, 8);
    auto bias_pair = Allocate(rt, positions ? max_pairs * 16 : 16);
    auto bias_row = Allocate(rt, positions ? predictions.size() * 16 : 16);
    auto bias_segments = (n_groups + 127) / 128;
    auto bias_partial = Allocate(rt, positions ? positions * bias_segments * 16ull : 16);
    auto bias_sum = Allocate(rt, positions ? positions * 16ull : 16);
    auto bias_positive = Upload(rt, bias.positive.data(), bias.positive.size_bytes());
    auto bias_negative = Upload(rt, bias.negative.data(), bias.negative.size_bytes());
    // Each sampled pair has two incidence nodes, linked in CUDA pair-index
    // order so rows can accumulate independently. Pair RNG is parallel.
    if (mean) CHECK_LE(max_pairs, (std::numeric_limits<std::uint32_t>::max() - 1ull) / 2);
    auto sampled = Allocate(rt, mean ? max_pairs * 8 : 8);
    auto trace = mean_pair_trace.empty() ? nil : Allocate(rt, mean_pair_trace.size_bytes());
    auto links = Allocate(rt, mean ? max_pairs * 8 : 8);
    auto heads = Allocate(rt, mean ? predictions.size() * 4 : 4);
    auto tails = Allocate(rt, mean ? predictions.size() * 4 : 4);
    auto norms = Allocate(rt, n_groups * 32);  // RankingGroupStats
    auto result = Allocate(rt, output.size_bytes());
    auto map_stats = Allocate(rt, 16);
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
      desc.sampleCount = 2 * (batches.size() * 12 + 16);
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
      auto values = Allocate(rt, predictions.size() * 16);
      auto ids = Allocate(rt, predictions.size() * 4);
      auto e = encoder("map_values");
      CHECK(e);
      [e setComputePipelineState:rt.ranking_map_stats_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:response offset:0 atIndex:1];
      [e setBuffer:offsets offset:0 atIndex:2];
      [e setBuffer:values offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e setBuffer:ids offset:0 atIndex:5];
      [e dispatchThreads:MTLSizeMake(predictions.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      map_stats = EncodeSegmentedDoublePrefix(rt, command, values, ids, predictions.size());
    }
    if (positions) {
      auto e = encoder("bias_minimum");
      CHECK(e);
      [e setComputePipelineState:rt.ranking_bias_min_pipeline];
      [e setBuffer:bias_positive offset:0 atIndex:0];
      [e setBuffer:bias_negative offset:0 atIndex:1];
      [e setBuffer:minimum_bias offset:0 atIndex:2];
      [e setBytes:&positions length:sizeof(positions) atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [e endEncoding];
    }
    // Unbiased CUDA bounds depend on all queries. Preserve bounded storage by
    // recomputing deterministic pairs in a second pass, after the global factor.
    auto passes = positions ? 2 : 1;
    for (int pass = 0; pass < passes; ++pass) {
      bool update = pass + 1 == passes;
      for (std::size_t b = 1; b < batches.size(); ++b) {
        auto begin = batches[b - 1], end = batches[b];
        struct {
          std::uint64_t rows, label_stride, pair_base, tile_base;
          std::uint32_t group_begin, group_end, row_begin, kind;
          std::uint32_t normalized, score_normalized, exponential, weighted;
          double weight_norm;
          std::uint32_t mean, seed, positions, update;
        } p{predictions.size(),
            labels.Stride(0),
            tasks[begin].pair_begin,
            tasks[begin].tile_begin,
            begin,
            end,
            groups[begin],
            static_cast<std::uint32_t>(kind),
            static_cast<std::uint32_t>(normalization),
            static_cast<std::uint32_t>(score_normalization),
            static_cast<std::uint32_t>(exponential),
            static_cast<std::uint32_t>(!weights.empty()),
            weight_norm,
            static_cast<std::uint32_t>(mean),
            seed,
            positions,
            static_cast<std::uint32_t>(update)};
        static_assert(sizeof(p) == 88);
        if (mean) {
          auto n_pairs = tasks[end].pair_begin - tasks[begin].pair_begin;
          if (n_pairs) {
            auto e = encoder("sample_pairs");
            CHECK(e);
            [e setComputePipelineState:rt.ranking_mean_sample_pipeline];
            [e setBuffer:order.records offset:0 atIndex:0];
            [e setBuffer:label_order.records offset:0 atIndex:1];
            [e setBuffer:response offset:0 atIndex:2];
            [e setBuffer:task_buffer offset:0 atIndex:3];
            [e setBuffer:sampled offset:0 atIndex:4];
            [e setBytes:&p length:sizeof(p) atIndex:5];
            [e dispatchThreads:MTLSizeMake(n_pairs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            if (trace && update) {
              auto blit = [command blitCommandEncoder];
              CHECK(blit);
              [blit copyFromBuffer:sampled
                       sourceOffset:0
                           toBuffer:trace
                  destinationOffset:p.pair_base * 8
                               size:n_pairs * 8];
              [blit endEncoding];
            }
          }
          if (update) {
            auto e = encoder("pair_links");
            CHECK(e);
            [e setComputePipelineState:rt.ranking_mean_links_pipeline];
            [e setBuffer:task_buffer offset:0 atIndex:0];
            [e setBuffer:sampled offset:0 atIndex:1];
            [e setBuffer:links offset:0 atIndex:2];
            [e setBuffer:heads offset:0 atIndex:3];
            [e setBuffer:tails offset:0 atIndex:4];
            [e setBytes:&p length:sizeof(p) atIndex:5];
            [e dispatchThreads:MTLSizeMake(end - begin, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e endEncoding];
          }
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
          [e setBuffer:costs offset:0 atIndex:13];
          [e setBuffer:cost_rounding offset:0 atIndex:14];
          [e dispatchThreadgroups:MTLSizeMake(tiles, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [e endEncoding];
        }
        if (!positions || !update) {
          auto e = encoder("gradient_statistics");
          CHECK(e);
          [e setComputePipelineState:rt.ranking_normalize_pipeline];
          [e setBuffer:pair_buffer offset:0 atIndex:0];
          [e setBuffer:task_buffer offset:0 atIndex:1];
          [e setBuffer:norms offset:0 atIndex:2];
          [e setBytes:&p length:sizeof(p) atIndex:3];
          [e setBuffer:costs offset:0 atIndex:4];
          [e setBuffer:minimum_bias offset:0 atIndex:5];
          [e setBuffer:weight offset:0 atIndex:6];
          [e dispatchThreadgroups:MTLSizeMake((end - begin + 7) / 8, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [e endEncoding];
        }
        auto rows = std::uint64_t{groups[end]} - groups[begin];
        if (update && rows) {
          auto e = encoder("gather");
          CHECK(e);
          [e setComputePipelineState:mean ? rt.ranking_mean_gather_pipeline
                                          : rt.ranking_gather_pipeline];
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
      if (positions && !update) {
        auto e = encoder("cost_rounding");
        CHECK(e);
        [e setComputePipelineState:rt.ranking_cost_rounding_pipeline];
        [e setBuffer:norms offset:0 atIndex:0];
        [e setBuffer:cost_rounding offset:0 atIndex:1];
        auto count = static_cast<std::uint32_t>(n_groups);
        [e setBytes:&count length:sizeof(count) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
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
    if (!diagnostics.groups.empty())
      std::memcpy(diagnostics.groups.data(), norms.contents, diagnostics.groups.size_bytes());
    if (!diagnostics.global_cost_rounding.empty())
      std::memcpy(diagnostics.global_cost_rounding.data(), cost_rounding.contents, sizeof(double));
    if (!diagnostics.map_prefix.empty())
      std::memcpy(diagnostics.map_prefix.data(), map_stats.contents,
                  diagnostics.map_prefix.size_bytes());
    if (trace) std::memcpy(mean_pair_trace.data(), trace.contents, mean_pair_trace.size_bytes());
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
    id<MTLBuffer> prefix = Allocate(rt, 16);
    if (kind == RankingMetricKind::kMAP && p.rows) {
      auto ids = Allocate(rt, p.rows * sizeof(std::uint32_t));
      auto values = Allocate(rt, p.rows * 2 * sizeof(double));
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.ranking_map_labels_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:response offset:0 atIndex:1];
      [e setBuffer:offsets offset:0 atIndex:2];
      [e setBuffer:ids offset:0 atIndex:3];
      [e setBuffer:values offset:0 atIndex:4];
      [e setBytes:&p length:sizeof(p) atIndex:5];
      [e dispatchThreads:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      prefix = EncodeSegmentedDoublePrefix(rt, command, values, ids, p.rows);
    }
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
    [e setBuffer:prefix offset:0 atIndex:8];
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
         Span<float const> weights, bool multiclass, bool precision_recall, void* output,
         std::size_t output_count, bool aggregate = false, AUCTrace* trace = nullptr) {
  auto rows = predictions.Shape(0), targets = predictions.Shape(1);
  CHECK_EQ(labels.Shape(0), rows);
  CHECK_EQ(labels.Shape(1), multiclass ? 1 : targets);
  CHECK_EQ(output_count, targets);
  CHECK(weights.empty() || weights.size() == rows);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(targets, 65535);
  if (!targets) return;
  CHECK(output);
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
    auto counts = Allocate(rt, rows * targets * 2 * sizeof(double));
    auto totals = Allocate(rt, targets * 2 * sizeof(double));
    auto area = Allocate(rt, targets * 2 * sizeof(double));
    auto result = Allocate(rt, targets * 3 * sizeof(double));
    id<MTLBuffer> prefix = Allocate(rt, 16);
    if (rows) {
      auto sorted = EncodeColumnSort(rt, command, input, shape, true);
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
      [e dispatchThreads:MTLSizeMake(rows, targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      prefix = EncodeDoublePrefix(rt, command, counts, rows, targets);
      if (precision_recall) {
        e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt.pr_totals_pipeline];
        [e setBuffer:sorted offset:0 atIndex:0];
        [e setBuffer:response offset:0 atIndex:1];
        [e setBuffer:weight offset:0 atIndex:2];
        [e setBuffer:totals offset:0 atIndex:3];
        [e setBytes:&p length:sizeof(p) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(targets, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      auto partial = Allocate(rt, shape.blocks * targets * 2 * sizeof(double));
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:precision_recall ? rt.pr_area_pipeline : rt.roc_area_pipeline];
      [e setBuffer:sorted offset:0 atIndex:0];
      [e setBuffer:prefix offset:0 atIndex:1];
      [e setBuffer:partial offset:0 atIndex:2];
      [e setBytes:&p length:sizeof(p) atIndex:3];
      if (precision_recall) [e setBuffer:totals offset:0 atIndex:4];
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
    }
    struct {
      std::uint64_t rows;
      std::uint32_t targets, precision_recall, aggregate, multiclass;
    } result_params{rows, static_cast<std::uint32_t>(targets),
                    static_cast<std::uint32_t>(precision_recall),
                    static_cast<std::uint32_t>(aggregate), static_cast<std::uint32_t>(multiclass)};
    static_assert(sizeof(result_params) == 24);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.auc_result_pipeline];
    [e setBuffer:prefix offset:0 atIndex:0];
    [e setBuffer:area offset:0 atIndex:1];
    [e setBuffer:totals offset:0 atIndex:2];
    [e setBuffer:result offset:0 atIndex:3];
    [e setBytes:&result_params length:sizeof(result_params) atIndex:4];
    [e dispatchThreads:MTLSizeMake(targets, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    if (trace) {
      auto copy = [](id<MTLBuffer> buffer, std::size_t count, std::vector<double>* out) {
        auto data = static_cast<double const*>(buffer.contents);
        out->assign(data, data + count);
      };
      copy(counts, 2 * rows * targets, &trace->terms);
      copy(prefix, 2 * rows * targets, &trace->prefixes);
      copy(totals, rows ? 2 * targets : 0, &trace->totals);
    }
    std::memcpy(output, result.contents, targets * 3 * sizeof(double));
    if (rows) {
      (precision_recall ? pr_auc_dispatches : roc_auc_dispatches)
          .fetch_add(1, std::memory_order_relaxed);
      radix_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
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
  static_assert(sizeof(ROCAUCSums) == 3 * sizeof(double));
  AUC(predictions, labels, weights, multiclass, false, output.data(), output.size());
}
void PRAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
           Span<float const> weights, bool multiclass, Span<ROCAUCSums> output, AUCTrace* trace) {
  AUC(predictions, labels, weights, multiclass, true, output.data(), output.size(), false, trace);
}
void MultiAUC(linalg::MatrixView<float const> predictions, linalg::MatrixView<float const> labels,
              Span<float const> weights, bool multiclass, bool is_roc, Span<double> output) {
  CHECK_EQ(output.size(), predictions.Shape(1) * 3);
  AUC(predictions, labels, weights, multiclass, !is_roc, output.data(), output.size() / 3, true);
}
double NormalizeAUC(Span<double const> summaries) {
  CHECK_EQ(summaries.size() % 3, 0);
  CHECK_LE(summaries.size() / 3, 65535);
  CHECK_GT(summaries.size(), 0);
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, summaries.data(), summaries.size_bytes());
    auto output = Allocate(rt, sizeof(double));
    std::uint32_t targets = summaries.size() / 3;
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.auc_normalize_pipeline];
    [e setBuffer:input offset:0 atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBytes:&targets length:sizeof(targets) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      metric_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                   std::memory_order_relaxed);
    return *static_cast<double const*>(output.contents);
  }
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
    auto flags = Allocate(rt, sizeof(std::uint32_t));
    auto totals = Allocate(rt, n_groups * sizeof(double));
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
    auto total_encoder = [command computeCommandEncoder];
    CHECK(total_encoder);
    [total_encoder setComputePipelineState:rt.ranking_pr_totals_pipeline];
    [total_encoder setBuffer:response offset:0 atIndex:0];
    [total_encoder setBuffer:weight offset:0 atIndex:1];
    [total_encoder setBuffer:offsets offset:0 atIndex:2];
    [total_encoder setBuffer:totals offset:0 atIndex:3];
    [total_encoder setBytes:&p length:sizeof(p) atIndex:4];
    [total_encoder dispatchThreadgroups:MTLSizeMake((n_groups + 7) / 8, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [total_encoder endEncoding];
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
      [e setBuffer:totals offset:0 atIndex:4];
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
    [e setBuffer:partial offset:0 atIndex:3];
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
    CHECK_EQ(static_cast<std::uint32_t const*>(flags.contents)[0], 0)
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
                    Span<float> keys, Span<bst_feature_t> permutation) {
  CHECK_LE(selected.size(), features.size());
  CHECK_LE(features.size(), std::numeric_limits<std::uint32_t>::max());
  CHECK(uniforms.empty() || (!weights.empty() && uniforms.size() == features.size()));
  CHECK(keys.empty() || (!weights.empty() && keys.size() == features.size()));
  CHECK(permutation.empty() || (weights.empty() && permutation.size() == features.size()));
  if (selected.empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = Upload(rt, features.data(), features.size_bytes());
    auto output = Allocate(rt, selected.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto indices = Allocate(rt, weights.empty() ? features.size_bytes() : 4);
    auto random = Allocate(rt, weights.empty() ? 4 : features.size() * sizeof(float));
    auto scores = Allocate(rt, weights.empty() ? 4 : features.size() * sizeof(float));
    auto invalid = Allocate(rt, sizeof(std::uint32_t));
    auto records = Allocate(rt, selected.size() * 2 * sizeof(std::uint32_t));
    struct {
      std::uint32_t rows, selected, seed, weighted;
      std::uint64_t weights;
    } p{static_cast<std::uint32_t>(features.size()), static_cast<std::uint32_t>(selected.size()),
        seed, static_cast<std::uint32_t>(!weights.empty()), weights.size()};
    static_assert(sizeof(p) == 24);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    {
      auto clear = [command blitCommandEncoder];
      CHECK(clear);
      [clear fillBuffer:invalid range:NSMakeRange(0, invalid.length) value:0];
      [clear endEncoding];
    }
    auto finish = [&] {
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      CHECK_EQ(*static_cast<std::uint32_t const*>(invalid.contents), 0)
          << "Feature index exceeds weights";
      std::memcpy(selected.data(), output.contents, selected.size_bytes());
      if (!uniforms.empty()) std::memcpy(uniforms.data(), random.contents, uniforms.size_bytes());
      if (!keys.empty()) std::memcpy(keys.data(), scores.contents, keys.size_bytes());
      if (!permutation.empty())
        std::memcpy(permutation.data(), indices.contents, permutation.size_bytes());
      column_sampling_dispatches.fetch_add(1, std::memory_order_relaxed);
      auto elapsed = command.GPUEndTime - command.GPUStartTime;
      if (elapsed > 0)
        column_sampling_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                              std::memory_order_relaxed);
    };
    if (!p.weighted) {
      std::uint32_t bits = 8;
      while ((std::uint64_t{1} << bits) < p.rows) ++bits;
      auto domain = std::uint64_t{1} << bits;
      struct {
        std::uint32_t rows, left_bits, right_bits, padding;
        std::uint32_t keys[24];
      } shuffle{p.rows, bits / 2, (bits + 1) / 2, 0, {}};
      static_assert(sizeof(shuffle) == 112);
      // Thrust constructs its round keys on host before launching the shuffle.
      // Its default engine is minstd_rand; uint32 uniform uses a binary64 real
      // draw over [0, UINT32_MAX+1), then truncates. Keep those rounding boundaries.
      std::minstd_rand rng(seed);
      for (auto& key : shuffle.keys)
        key = static_cast<std::uint32_t>((double(rng() - 1) / 2147483646.0) * 0x1p32);
      auto padded = Allocate(rt, domain * sizeof(std::uint32_t));
      auto flags = Allocate(rt, domain * sizeof(std::uint64_t));
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.column_shuffle_init_pipeline];
      [e setBuffer:padded offset:0 atIndex:0];
      [e setBuffer:flags offset:0 atIndex:1];
      [e setBytes:&shuffle length:sizeof(shuffle) atIndex:2];
      [e dispatchThreads:MTLSizeMake(domain, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto prefix = EncodePrefixU64(rt, command, flags, domain);
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.column_shuffle_compact_pipeline];
      [e setBuffer:padded offset:0 atIndex:0];
      [e setBuffer:prefix offset:0 atIndex:1];
      [e setBuffer:indices offset:0 atIndex:2];
      [e setBytes:&p.rows length:sizeof(p.rows) atIndex:3];
      [e dispatchThreads:MTLSizeMake(domain, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    id<MTLBuffer> ranked = Allocate(rt, 8);
    if (p.weighted) {
      auto e = [command computeCommandEncoder];
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
    auto e = [command computeCommandEncoder];
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

void EvaluateQuantizedSplits(Span<HistogramInput const> histograms,
                             Span<QuantizedSplitTask const> tasks,
                             SplitParameters const& parameters,
                             std::vector<QuantizedSplitResult>* results,
                             std::vector<std::uint32_t>* sorted_categories,
                             SplitWorkspace* workspace) {
  if (workspace) static_cast<MetalSplitWorkspace*>(workspace)->candidate_count = 0;
  static_assert(sizeof(QuantizedSplitTask) == 48 && sizeof(QuantizedSplitResult) == 80);
  static_assert(sizeof(GradientPairInt64) == 16);
  results->resize(tasks.size());
  auto bins = tasks.empty() ? 0 : tasks.back().end;
  bool categorical =
      std::any_of(tasks.begin(), tasks.end(), [](auto const& task) { return task.kind == 2; });
  if (sorted_categories) sorted_categories->resize(categorical ? bins : 0);
  if (tasks.empty()) return;
  CHECK(workspace);
  CHECK_EQ(histograms.size(), 1);
  CHECK(histograms[0].source);
  auto storage = static_cast<MetalHistogramBuffer const*>(histograms[0].source);
  CHECK(storage->integers);
  CHECK(storage->scale);
  CHECK_LE(tasks.size(), std::numeric_limits<std::uint32_t>::max());
  std::uint32_t previous = 0;
  for (auto const& task : tasks) {
    CHECK_EQ(task.begin, previous);
    CHECK_GE(task.end, task.begin);
    CHECK_LE(task.end - task.begin, std::numeric_limits<std::int32_t>::max());
    CHECK_LE(task.kind, 2);
    previous = task.end;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto hist = GatherHistograms(rt, histograms, bins, command, scratch, true);
    auto input = Scratch(rt, &scratch->input, tasks.size_bytes(), tasks.data());
    auto output = Scratch(rt, &scratch->output, tasks.size() * sizeof(QuantizedSplitResult));
    auto order = Scratch(rt, &scratch->order, (categorical ? bins : 0) * sizeof(std::uint32_t));
    auto weights = Scratch(rt, &scratch->weights, (categorical ? bins : 0) * sizeof(float));
    auto scans = Scratch(rt, &scratch->scans, bins * sizeof(GradientPairInt64));
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt.quantized_split_pipeline];
    [encoder setBuffer:hist offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBuffer:order offset:0 atIndex:4];
    [encoder setBuffer:weights offset:0 atIndex:5];
    [encoder setBuffer:scans offset:0 atIndex:6];
    [encoder setBuffer:storage->scale offset:0 atIndex:7];
    CHECK_EQ(rt.quantized_split_pipeline.threadExecutionWidth, 32);
    CHECK_GE(rt.quantized_split_pipeline.maxTotalThreadsPerThreadgroup, 32);
    [encoder dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(results->data(), output.contents, tasks.size() * sizeof(QuantizedSplitResult));
    for (auto const& task : tasks)
      if (sorted_categories && task.kind == 2) {
        std::memcpy(sorted_categories->data() + task.begin,
                    static_cast<std::uint32_t const*>(order.contents) + task.begin,
                    (task.end - task.begin) * sizeof(std::uint32_t));
      }
    split_dispatches.fetch_add(1, std::memory_order_relaxed);
    resident_split_dispatches.fetch_add(1, std::memory_order_relaxed);
    cuda_integer_split_dispatches.fetch_add(1, std::memory_order_relaxed);
    scratch->candidate_count = tasks.size();
    scratch->vector_candidates = false;
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      split_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                  std::memory_order_relaxed);
  }
}

void FinalizeSplitNodes(Span<SplitFinalizeNode const> nodes,
                        Span<SplitFinalizeFeature const> features,
                        Span<GradientPairPrecise const> parents, Span<SplitBounds const> bounds,
                        SplitParameters const& parameters, bool vector_leaf,
                        std::vector<SplitNodeResult>* results,
                        std::vector<SplitNodeWeights>* weights, SplitWorkspace* workspace) {
  static_assert(sizeof(SplitFinalizeNode) == 16 && sizeof(SplitFinalizeFeature) == 16);
  static_assert(sizeof(SplitNodeResult) == 40 && sizeof(SplitNodeWeights) == 16);
  results->resize(nodes.size());
  weights->resize(parents.size());
  CHECK_EQ(parents.size(), bounds.size());
  CHECK_LE(features.size(), std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(parents.size(), std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(nodes.size(), std::numeric_limits<std::uint32_t>::max());
  std::size_t previous = 0, parent_end = 0;
  for (auto const& node : nodes) {
    CHECK_EQ(node.begin, previous);
    CHECK_GE(node.end, node.begin);
    CHECK_LE(node.end, features.size());
    CHECK_EQ(node.parent_begin, parent_end);
    CHECK_GT(node.targets, 0);
    if (!vector_leaf) CHECK_EQ(node.targets, 1);
    parent_end += node.targets;
    CHECK_LE(parent_end, parents.size());
    previous = node.end;
  }
  CHECK_EQ(previous, features.size());
  CHECK_EQ(parent_end, parents.size());
  if (nodes.empty()) return;
  CHECK(workspace);
  auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
  if (!features.empty()) {
    CHECK_EQ(features.size(), scratch->candidate_count);
    CHECK_EQ(vector_leaf, scratch->vector_candidates);
    if (vector_leaf) {
      CHECK(scratch->children);
      for (auto const& node : nodes)
        for (std::uint32_t i = node.begin; i < node.end; ++i)
          CHECK_LE(std::uint64_t(features[i].child_begin) + 2 * std::uint64_t(node.targets),
                   scratch->candidate_child_count);
    }
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto candidates = Scratch(rt, &scratch->output, 0);
    auto children = Scratch(rt, &scratch->children, 0);
    auto input = Scratch(rt, &scratch->node_input, nodes.size_bytes(), nodes.data());
    auto feature = Scratch(rt, &scratch->feature_input, features.size_bytes(), features.data());
    auto parent = Scratch(rt, &scratch->node_parents, parents.size_bytes(), parents.data());
    auto bound = Scratch(rt, &scratch->node_bounds, bounds.size_bytes(), bounds.data());
    auto out = Scratch(rt, &scratch->node_results, results->size() * sizeof(SplitNodeResult));
    auto weight = Scratch(rt, &scratch->node_weights, weights->size() * sizeof(SplitNodeWeights));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    std::uint32_t multi = vector_leaf;
    [e setComputePipelineState:rt.split_finalize_pipeline];
    [e setBuffer:candidates offset:0 atIndex:0];
    [e setBuffer:feature offset:0 atIndex:1];
    [e setBuffer:input offset:0 atIndex:2];
    [e setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [e setBuffer:parent offset:0 atIndex:4];
    [e setBuffer:bound offset:0 atIndex:5];
    [e setBuffer:children offset:0 atIndex:6];
    [e setBuffer:out offset:0 atIndex:7];
    [e setBuffer:weight offset:0 atIndex:8];
    [e setBytes:&multi length:sizeof(multi) atIndex:9];
    CHECK_EQ(rt.split_finalize_pipeline.threadExecutionWidth, 32);
    CHECK_GE(rt.split_finalize_pipeline.maxTotalThreadsPerThreadgroup, 32);
    [e dispatchThreadgroups:MTLSizeMake(nodes.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &split_finalize_dispatches, &split_gpu_time_ns);
    std::memcpy(results->data(), out.contents, results->size() * sizeof(SplitNodeResult));
    std::memcpy(weights->data(), weight.contents, weights->size() * sizeof(SplitNodeWeights));
  }
}

void BuildSplitCategories(Span<float const> cuts, Span<SplitCategoryTask const> tasks,
                          std::uint32_t words_per_node, std::vector<std::uint32_t>* output,
                          SplitWorkspace* workspace) {
  static_assert(sizeof(SplitCategoryTask) == 8);
  CHECK(output);
  if (tasks.empty()) {
    output->clear();
    return;
  }
  CHECK(workspace);
  CHECK(cuts.data() || cuts.empty());
  auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
  CHECK_GT(scratch->candidate_count, 0)
      << "Category assembly requires current quantized candidates";
  CHECK_GT(words_per_node, 0);
  CHECK_LE(tasks.size(), std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(tasks.size(), std::numeric_limits<std::size_t>::max() / 4 / words_per_node);
  CHECK_LE(cuts.size(), std::numeric_limits<std::uint32_t>::max());
  // Check metadata extents against logical input sizes, not buffer capacities.
  // No category selection or bit assembly is performed on the host.
  for (auto const& request : tasks) {
    CHECK_LT(request.candidate, scratch->candidate_count);
    std::uint32_t bins, kind;
    if (scratch->vector_candidates) {
      auto task =
          static_cast<QuantizedMultiSplitTask const*>(scratch->input.contents)[request.candidate];
      bins = (task.end - task.begin) / task.targets;
      kind = task.kind;
    } else {
      auto task =
          static_cast<QuantizedSplitTask const*>(scratch->input.contents)[request.candidate];
      bins = task.end - task.begin;
      kind = task.kind;
    }
    if (kind) {
      CHECK_LE(request.cut_begin, cuts.size());
      CHECK_LE(bins, cuts.size() - request.cut_begin);
    }
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto count = tasks.size() * words_per_node;
    auto requests = Scratch(rt, &scratch->category_input, tasks.size_bytes(), tasks.data());
    auto values = Scratch(rt, &scratch->category_cuts, cuts.size_bytes(), cuts.data());
    auto bits = Scratch(rt, &scratch->category_output, count * sizeof(std::uint32_t));
    std::uint32_t zero = 0;
    auto error = Scratch(rt, &scratch->category_error, sizeof(zero), &zero);
    struct {
      std::uint32_t words, multi;
    } p{words_per_node, scratch->vector_candidates};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt.split_category_pipeline];
    [encoder setBuffer:requests offset:0 atIndex:0];
    [encoder setBuffer:scratch->input offset:0 atIndex:1];
    [encoder setBuffer:scratch->output offset:0 atIndex:2];
    [encoder setBuffer:scratch->order offset:0 atIndex:3];
    [encoder setBuffer:values offset:0 atIndex:4];
    [encoder setBuffer:bits offset:0 atIndex:5];
    [encoder setBytes:&p length:sizeof(p) atIndex:6];
    [encoder setBuffer:error offset:0 atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    CompleteOperation(command, &split_category_dispatches, &split_gpu_time_ns);
    split_category_nodes.fetch_add(tasks.size(), std::memory_order_relaxed);
    CHECK_EQ(*static_cast<std::uint32_t const*>(error.contents), 0)
        << "Invalid category, partition, sorted index or bitset extent";
    output->resize(count);
    std::memcpy(output->data(), bits.contents, count * sizeof(std::uint32_t));
  }
}

namespace {
void ValidateSplitBounds(Span<float const> lower, Span<float const> upper, std::uint32_t parent,
                         std::uint32_t left, std::uint32_t right, Span<float const> left_weight,
                         Span<float const> right_weight, bool vector_leaf) {
  CHECK_EQ(lower.size(), upper.size());
  CHECK_EQ(left_weight.size(), right_weight.size());
  CHECK(!left_weight.empty());
  CHECK_LE(left_weight.size(), std::numeric_limits<std::uint32_t>::max());
  if (!vector_leaf) CHECK_EQ(left_weight.size(), 1);
  CHECK_NE(parent, left);
  CHECK_NE(parent, right);
  CHECK_NE(left, right);
  CHECK_LE((std::uint64_t(std::max({parent, left, right})) + 1) * left_weight.size(), lower.size());
  CHECK(!MemoryRangesOverlap(lower.data(), lower.size_bytes(), upper.data(), upper.size_bytes()));
}
void UpdateBoundsBuffers(Runtime& rt, BorrowedMemory const& lo, BorrowedMemory const& hi,
                         std::uint32_t parent, std::uint32_t left, std::uint32_t right,
                         std::int32_t monotone, Span<float const> left_weight,
                         Span<float const> right_weight, bool vector_leaf) {
  // Snapshot weights before any writes, including inputs aliased to bound storage.
  auto lw = Upload(rt, left_weight.data(), left_weight.size_bytes());
  auto rw = Upload(rt, right_weight.data(), right_weight.size_bytes());
  struct {
    std::uint32_t parent, left, right, targets;
    std::int32_t monotone;
    std::uint32_t multi;
  } p{parent, left, right, static_cast<std::uint32_t>(left_weight.size()), monotone, vector_leaf};
  static_assert(sizeof(p) == 24);
  std::uint32_t zero = 0;
  auto error = Upload(rt, &zero, sizeof(zero));
  auto command = [rt.queue commandBuffer];
  CHECK(command);
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:rt.split_bounds_pipeline];
  [e setBuffer:lo.buffer offset:lo.offset atIndex:0];
  [e setBuffer:hi.buffer offset:hi.offset atIndex:1];
  [e setBuffer:lw offset:0 atIndex:2];
  [e setBuffer:rw offset:0 atIndex:3];
  [e setBytes:&p length:sizeof(p) atIndex:4];
  [e setBuffer:error offset:0 atIndex:5];
  [e dispatchThreads:MTLSizeMake(p.targets, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  CompleteOperation(command, &split_bounds_dispatches, &split_gpu_time_ns);
  CHECK_EQ(*static_cast<std::uint32_t const*>(error.contents), 0) << "Constraint midpoint is NaN";
}
class MetalSplitBoundsStorage final : public SplitBoundsStorage {
  Runtime& rt_;
  id<MTLBuffer> lower_{nil}, upper_{nil};
  std::size_t size_{0};

 public:
  explicit MetalSplitBoundsStorage(std::size_t count) : rt_{GetRuntime()} { Grow(count); }
  void Grow(std::size_t count) override {
    if (count <= size_) return;
    CHECK_LE(count, std::numeric_limits<std::uint32_t>::max());
    @autoreleasepool {
      auto lower = Allocate(rt_, count * sizeof(float));
      auto upper = Allocate(rt_, count * sizeof(float));
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      if (size_) {
        auto blit = [command blitCommandEncoder];
        CHECK(blit);
        [blit copyFromBuffer:lower_
                 sourceOffset:0
                     toBuffer:lower
            destinationOffset:0
                         size:size_ * sizeof(float)];
        [blit copyFromBuffer:upper_
                 sourceOffset:0
                     toBuffer:upper
            destinationOffset:0
                         size:size_ * sizeof(float)];
        [blit endEncoding];
      }
      struct {
        std::uint64_t begin, end;
      } range{size_, count};
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.split_bounds_init_pipeline];
      [e setBuffer:lower offset:0 atIndex:0];
      [e setBuffer:upper offset:0 atIndex:1];
      [e setBytes:&range length:sizeof(range) atIndex:2];
      [e dispatchThreads:MTLSizeMake(count - size_, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &split_bounds_init_dispatches, &split_gpu_time_ns);
      lower_ = lower;
      upper_ = upper;
      size_ = count;
    }
  }
  Span<float const> Lower() const override {
    return {static_cast<float const*>(lower_.contents), size_};
  }
  Span<float const> Upper() const override {
    return {static_cast<float const*>(upper_.contents), size_};
  }
  void Update(std::uint32_t parent, std::uint32_t left, std::uint32_t right, std::int32_t monotone,
              Span<float const> left_weight, Span<float const> right_weight,
              bool vector_leaf) override {
    ValidateSplitBounds(Lower(), Upper(), parent, left, right, left_weight, right_weight,
                        vector_leaf);
    @autoreleasepool {
      UpdateBoundsBuffers(rt_, {lower_, 0}, {upper_, 0}, parent, left, right, monotone, left_weight,
                          right_weight, vector_leaf);
    }
  }
};
}  // namespace
std::unique_ptr<SplitBoundsStorage> MakeSplitBoundsStorage(std::size_t count) {
  return std::make_unique<MetalSplitBoundsStorage>(count);
}
void UpdateSplitBounds(Span<float> lower, Span<float> upper, std::uint32_t parent,
                       std::uint32_t left, std::uint32_t right, std::int32_t monotone,
                       Span<float const> left_weight, Span<float const> right_weight,
                       bool vector_leaf) {
  ValidateSplitBounds(lower, upper, parent, left, right, left_weight, right_weight, vector_leaf);
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto lo = TryBorrowMemory(rt.device, lower.data(), lower.size_bytes(), alignof(float));
    auto hi = TryBorrowMemory(rt.device, upper.data(), upper.size_bytes(), alignof(float));
    bool copy_lo = !lo.buffer, copy_hi = !hi.buffer;
    if (copy_lo) lo = {Upload(rt, lower.data(), lower.size_bytes()), 0};
    if (copy_hi) hi = {Upload(rt, upper.data(), upper.size_bytes()), 0};
    UpdateBoundsBuffers(rt, lo, hi, parent, left, right, monotone, left_weight, right_weight,
                        vector_leaf);
    if (copy_lo) std::memcpy(lower.data(), lo.buffer.contents, lower.size_bytes());
    if (copy_hi) std::memcpy(upper.data(), hi.buffer.contents, upper.size_bytes());
  }
}

void EvaluateSplits(Span<HistogramInput const> histograms, Span<SplitTask const> tasks,
                    SplitParameters const& parameters, std::vector<SplitResult>* results,
                    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace) {
  if (workspace) static_cast<MetalSplitWorkspace*>(workspace)->candidate_count = 0;
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
void EvaluateQuantizedMultiSplits(
    Span<HistogramInput const> histograms, Span<QuantizedMultiSplitTask const> tasks,
    Span<GradientPairInt64 const> parents, Span<SplitBounds const> bounds,
    SplitParameters const& parameters, std::vector<QuantizedMultiSplitResult>* results,
    std::vector<GradientPairInt64>* integer_children, std::vector<GradientPairPrecise>* children,
    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace) {
  if (workspace) static_cast<MetalSplitWorkspace*>(workspace)->candidate_count = 0;
  static_assert(sizeof(QuantizedMultiSplitTask) == 40 && sizeof(QuantizedMultiSplitResult) == 16);
  results->resize(tasks.size());
  auto bins = tasks.empty() ? 0 : tasks.back().end;
  bool categorical =
      std::any_of(tasks.begin(), tasks.end(), [](auto const& task) { return task.kind == 2; });
  if (sorted_categories) sorted_categories->resize(categorical ? bins : 0);
  CHECK_LE(parents.size(), std::numeric_limits<std::uint32_t>::max() / 2);
  children->resize(parents.size() * 2);
  integer_children->resize(parents.size() * 2);
  if (tasks.empty()) return;
  CHECK(workspace);
  CHECK_EQ(parents.size(), bounds.size());
  CHECK_LE(tasks.size(), std::numeric_limits<std::uint32_t>::max());
  std::uint32_t previous = 0;
  for (auto const& task : tasks) {
    CHECK_GT(task.targets, 0);
    CHECK_EQ(task.targets, histograms.size());
    CHECK_EQ(task.begin, previous);
    CHECK_GE(task.end, task.begin);
    CHECK_EQ((task.end - task.begin) % task.targets, 0);
    CHECK_LE((task.end - task.begin) / task.targets, std::numeric_limits<std::int32_t>::max());
    CHECK_LE(std::uint64_t(task.parent_begin) + task.targets, parents.size());
    CHECK_LE(task.kind, 2);
    CHECK_LE(task.one_pass, 2);
    previous = task.end;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto scales = Scratch(rt, &scratch->scales, histograms.size() * sizeof(GradientScale));
    auto blit = [command blitCommandEncoder];
    CHECK(blit);
    for (std::size_t t = 0; t < histograms.size(); ++t) {
      CHECK(histograms[t].source);
      auto storage = static_cast<MetalHistogramBuffer const*>(histograms[t].source);
      CHECK(storage->integers);
      CHECK(storage->scale);
      [blit copyFromBuffer:storage->scale
               sourceOffset:0
                   toBuffer:scales
          destinationOffset:t * sizeof(GradientScale)
                       size:sizeof(GradientScale)];
    }
    [blit endEncoding];
    auto hist = GatherHistograms(rt, histograms, bins, command, scratch, true);
    auto input = Scratch(rt, &scratch->input, tasks.size_bytes(), tasks.data());
    auto parent = Scratch(rt, &scratch->parent, parents.size_bytes(), parents.data());
    auto bound = Scratch(rt, &scratch->bounds, bounds.size_bytes(), bounds.data());
    auto output = Scratch(rt, &scratch->output, tasks.size() * sizeof(QuantizedMultiSplitResult));
    auto child = Scratch(rt, &scratch->children, children->size() * sizeof(GradientPairPrecise));
    auto exact = Scratch(rt, &scratch->integer_children,
                         integer_children->size() * sizeof(GradientPairInt64));
    auto order = Scratch(rt, &scratch->order, (categorical ? bins : 0) * sizeof(std::uint32_t));
    auto scores = Scratch(rt, &scratch->weights, (categorical ? bins : 0) * sizeof(std::uint64_t));
    auto scans = Scratch(rt, &scratch->scans, bins * 2 * sizeof(GradientPairInt64));
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt.quantized_multi_split_pipeline];
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
    [encoder setBuffer:scales offset:0 atIndex:10];
    [encoder setBuffer:exact offset:0 atIndex:11];
    CHECK_EQ(rt.quantized_multi_split_pipeline.threadExecutionWidth, 32);
    CHECK_GE(rt.quantized_multi_split_pipeline.maxTotalThreadsPerThreadgroup, 32);
    [encoder dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    std::memcpy(results->data(), output.contents, tasks.size() * sizeof(QuantizedMultiSplitResult));
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto const& task = tasks[i];
      if (results->at(i).bin >= 0) {
        std::memcpy(children->data() + 2 * task.parent_begin,
                    static_cast<GradientPairPrecise const*>(child.contents) + 2 * task.parent_begin,
                    2 * task.targets * sizeof(GradientPairPrecise));
        std::memcpy(integer_children->data() + 2 * task.parent_begin,
                    static_cast<GradientPairInt64 const*>(exact.contents) + 2 * task.parent_begin,
                    2 * task.targets * sizeof(GradientPairInt64));
      }
      if (sorted_categories && task.kind == 2)
        std::memcpy(sorted_categories->data() + task.begin,
                    static_cast<std::uint32_t const*>(order.contents) + task.begin,
                    ((task.end - task.begin) / task.targets) * sizeof(std::uint32_t));
    }
    split_dispatches.fetch_add(1, std::memory_order_relaxed);
    resident_split_dispatches.fetch_add(1, std::memory_order_relaxed);
    cuda_integer_multi_split_dispatches.fetch_add(1, std::memory_order_relaxed);
    scratch->candidate_count = tasks.size();
    scratch->vector_candidates = true;
    scratch->candidate_child_count = children->size();
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
  if (workspace) static_cast<MetalSplitWorkspace*>(workspace)->candidate_count = 0;
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

namespace {
struct CudaSketchParams {
  std::uint64_t size, columns, budget, begin, base_row;
  std::uint32_t weighted, mode;
};
static_assert(sizeof(CudaSketchParams) == 48);
void CSDispatch(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline,
                std::size_t count, std::initializer_list<id<MTLBuffer>> buffers,
                CudaSketchParams const* params = nullptr) {
  if (!count) return;
  CHECK_LE(count, std::numeric_limits<std::uint32_t>::max());
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:pipeline];
  std::size_t index = 0;
  for (auto buffer : buffers) [e setBuffer:buffer offset:0 atIndex:index++];
  if (params) [e setBytes:params length:sizeof(*params) atIndex:index];
  [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
}
void CSComplete(id<MTLCommandBuffer> command) {
  CompleteOperation(command, &cuda_sketch_dispatches, &cuda_sketch_gpu_time_ns);
}
id<MTLBuffer> CSEncodePrefix(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                             id<MTLBuffer> ids, std::uint32_t count) {
  auto blocks = (std::uint64_t{count} + 255) / 256;
  auto output = Allocate(rt, std::size_t{count} * sizeof(float));
  auto totals = Allocate(rt, blocks * sizeof(float));
  auto total_ids = Allocate(rt, blocks * sizeof(std::uint32_t));
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:rt.cs_prefix_pipeline];
  [e setBuffer:input offset:0 atIndex:0];
  [e setBuffer:ids offset:0 atIndex:1];
  [e setBuffer:output offset:0 atIndex:2];
  [e setBuffer:totals offset:0 atIndex:3];
  [e setBuffer:total_ids offset:0 atIndex:4];
  [e setBytes:&count length:sizeof(count) atIndex:5];
  [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  if (blocks > 1) {
    auto upper = CSEncodePrefix(rt, command, totals, total_ids, blocks);
    CSDispatch(command, rt.cs_prefix_add_pipeline, count, {output, ids, upper, total_ids});
  }
  return output;
}
struct CSDeviceSummary {
  id<MTLBuffer> entries, ptr;
  std::size_t size;
};
class MetalCudaSketch final : public CudaSketch {
  Runtime& rt_;
  bst_feature_t columns_;
  bst_bin_t max_bin_;
  bst_idx_t rows_seen_{0};
  std::size_t weight_count_;
  bool has_cat_;
  id<MTLBuffer> types_, weights_;
  CSDeviceSummary resident_;

  CudaSketchParams Params(std::size_t size) const {
    return {size, columns_, 0, 0, 0, weight_count_ != 0, 0};
  }
  std::size_t Total(id<MTLBuffer> ptr) const {
    return static_cast<std::uint64_t const*>(ptr.contents)[columns_];
  }
  CSDeviceSummary Unique(CSDeviceSummary input, bool categories_only,
                         id<MTLBuffer> selected = nil) {
    if (!input.size) return input;
    auto p = Params(input.size);
    p.mode = categories_only ? 1 : (selected ? 2 : 0);
    auto flags = Allocate(rt_, input.size * sizeof(std::uint64_t));
    auto command = [rt_.queue commandBuffer];
    CHECK(command);
    CSDispatch(command, rt_.cs_flags_pipeline, input.size,
               {input.entries, input.ptr, types_, selected ? selected : input.entries, flags}, &p);
    auto prefix = EncodePrefixU64(rt_, command, flags, input.size);
    CSDeviceSummary output{Allocate(rt_, input.size * sizeof(WQSummary<>::Entry)),
                           Allocate(rt_, (std::size_t{columns_} + 1) * sizeof(std::uint64_t)), 0};
    CSDispatch(command, rt_.cs_compact_pipeline, std::max(input.size, std::size_t{columns_} + 1),
               {input.entries, input.ptr, prefix, output.entries, output.ptr}, &p);
    CSComplete(command);
    output.size = Total(output.ptr);
    return output;
  }
  CSDeviceSummary Prune(CSDeviceSummary input, std::size_t budget, bool raw) {
    auto p = Params(input.size);
    p.budget = budget;
    auto counts = Allocate(rt_, (std::size_t{columns_} + 1) * sizeof(std::uint64_t));
    auto command = [rt_.queue commandBuffer];
    CHECK(command);
    CSDispatch(command, rt_.cs_cut_counts_pipeline, std::size_t{columns_} + 1,
               {input.ptr, types_, counts}, &p);
    auto ptr = EncodePrefixU64(rt_, command, counts, std::size_t{columns_} + 1);
    CSComplete(command);
    auto count = Total(ptr);
    if (!raw && count == input.size) return input;
    CSDeviceSummary output{Allocate(rt_, count * sizeof(WQSummary<>::Entry)), ptr, count};
    auto selected = Allocate(rt_, count * sizeof(std::uint64_t));
    command = [rt_.queue commandBuffer];
    CHECK(command);
    CSDispatch(command, rt_.cs_prune_pipeline, count,
               {input.entries, input.ptr, ptr, types_, output.entries, selected}, &p);
    if (raw) CSDispatch(command, rt_.cs_scan_input_pipeline, columns_, {output.entries, ptr});
    CSComplete(command);
    // Fuse index-unique/gather/value-unique while retaining both predicates.
    // Value subtraction alone does not remove repeated indices with infinities.
    return Unique(output, false, raw ? nil : selected);
  }
  void Merge(CSDeviceSummary input) {
    if (!input.size) return;
    if (resident_.size) {
      auto p = Params(resident_.size + input.size);
      CSDeviceSummary output{Allocate(rt_, p.size * sizeof(WQSummary<>::Entry)),
                             Allocate(rt_, (std::size_t{columns_} + 1) * sizeof(std::uint64_t)),
                             p.size};
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      CSDispatch(command, rt_.cs_merge_ptr_pipeline, std::size_t{columns_} + 1,
                 {resident_.ptr, input.ptr, output.ptr});
      CSDispatch(
          command, rt_.cs_merge_pipeline, p.size,
          {resident_.entries, resident_.ptr, input.entries, input.ptr, output.ptr, output.entries},
          &p);
      CSComplete(command);
      resident_ = Unique(output, false);
    } else {
      resident_ = Unique(input, false);
    }
    auto command = [rt_.queue commandBuffer];
    CHECK(command);
    CSDispatch(command, rt_.cs_fix_pipeline, columns_, {resident_.entries, resident_.ptr});
    CSComplete(command);
  }

 public:
  MetalCudaSketch(bst_feature_t columns, bst_bin_t max_bin, Span<FeatureType const> types,
                  Span<float const> weights)
      : rt_{GetRuntime()},
        columns_{columns},
        max_bin_{max_bin},
        weight_count_{weights.size()},
        has_cat_{std::find(types.begin(), types.end(), FeatureType::kCategorical) != types.end()} {
    CHECK_GE(max_bin, 2);
    CHECK_LT(columns, std::numeric_limits<std::uint32_t>::max());
    CHECK(types.empty() || types.size() == columns);
    std::vector<FeatureType> numerical(types.empty() ? columns : 0, FeatureType::kNumerical);
    types_ = types.empty() ? Upload(rt_, numerical.data(), numerical.size())
                           : Upload(rt_, types.data(), types.size_bytes());
    weights_ = Upload(rt_, weights.data(), weights.size_bytes());
    std::vector<std::uint64_t> ptr(std::size_t{columns} + 1, 0);
    resident_ = {Allocate(rt_, 0), Upload(rt_, ptr.data(), ptr.size() * sizeof(std::uint64_t)), 0};
  }
  void Push(Span<::xgboost::Entry const> entries, Span<bst_idx_t const> row_ptr, std::size_t begin,
            bst_idx_t base_rowid, bst_idx_t rows_in_window) override {
    if (entries.empty()) return;
    CHECK_GT(columns_, 0);
    CHECK_LE(entries.size(), 1ULL << 26);  // The CUDA frontend's entry window.
    CHECK_GE(row_ptr.size(), 2);
    CHECK_LE(begin, row_ptr.back());
    CHECK_LE(entries.size(), row_ptr.back() - begin);
    CHECK_LE(row_ptr.front(), begin);
    CHECK_GT(rows_in_window, 0);
    if (weight_count_) CHECK_LE(base_rowid + row_ptr.size() - 1, weight_count_);
    CHECK_LE(rows_in_window, std::numeric_limits<bst_idx_t>::max() - rows_seen_);
    rows_seen_ += rows_in_window;
    @autoreleasepool {
      auto p = Params(entries.size());
      p.begin = begin;
      p.base_row = base_rowid;
      p.budget = row_ptr.size() - 1;
      auto source = Upload(rt_, entries.data(), entries.size_bytes());
      auto rows = Upload(rt_, row_ptr.data(), row_ptr.size_bytes());
      auto records = Allocate(rt_, entries.size() * 2 * sizeof(std::uint32_t));
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      CSDispatch(command, rt_.cs_sort_keys_pipeline, entries.size(), {source, source, records}, &p);
      SortShape shape{entries.size(), 0, 0, 1,
                      static_cast<std::uint32_t>((entries.size() + 255) / 256)};
      records = EncodeRadixRecords(rt_, command, records, shape);
      auto by_feature = Allocate(rt_, entries.size() * 2 * sizeof(std::uint32_t));
      p.mode = 1;
      CSDispatch(command, rt_.cs_sort_keys_pipeline, entries.size(), {source, records, by_feature},
                 &p);
      records = EncodeRadixRecords(rt_, command, by_feature, shape);
      CSDeviceSummary raw{Allocate(rt_, entries.size() * sizeof(WQSummary<>::Entry)),
                          Allocate(rt_, (std::size_t{columns_} + 1) * sizeof(std::uint64_t)),
                          entries.size()};
      auto weights = Allocate(rt_, entries.size() * sizeof(float));
      auto ids = Allocate(rt_, entries.size() * sizeof(std::uint32_t));
      CSDispatch(command, rt_.cs_bounds_pipeline, std::size_t{columns_} + 1, {records, raw.ptr},
                 &p);
      CSDispatch(command, rt_.cs_gather_pipeline, entries.size(),
                 {source, records, rows, weights_, raw.entries, weights, ids}, &p);
      if (weight_count_) {
        auto scanned = CSEncodePrefix(rt_, command, weights, ids, entries.size());
        CSDispatch(command, rt_.cs_attach_pipeline, entries.size(), {raw.entries, scanned});
      }
      CSComplete(command);
      if (has_cat_) raw = Unique(raw, true);
      auto ranked = Allocate(rt_, raw.size * sizeof(WQSummary<>::Entry));
      command = [rt_.queue commandBuffer];
      CHECK(command);
      CSDispatch(command, rt_.cs_raw_ranks_pipeline, raw.size, {raw.entries, raw.ptr, ranked}, &p);
      CSComplete(command);
      raw.entries = ranked;
      auto n = std::max<bst_idx_t>(1, rows_in_window);
      auto batch = Prune(raw, std::min<std::size_t>(SketchSummaryBudget(max_bin_, n), n), true);
      Merge(batch);
      resident_ = Prune(resident_,
                        SketchSummaryBudget(max_bin_, std::max<bst_idx_t>(1, rows_seen_)), false);
      cuda_sketch_batches.fetch_add(1, std::memory_order_relaxed);
      cuda_sketch_input_entries.fetch_add(entries.size(), std::memory_order_relaxed);
    }
  }
  CudaSketchSummary Read() const override {
    CudaSketchSummary out;
    out.ptr.resize(std::size_t{columns_} + 1);
    out.entries.resize(resident_.size);
    std::memcpy(out.ptr.data(), resident_.ptr.contents, out.ptr.size() * sizeof(std::uint64_t));
    if (!out.entries.empty())
      std::memcpy(out.entries.data(), resident_.entries.contents,
                  out.entries.size() * sizeof(WQSummary<>::Entry));
    return out;
  }
};
}  // namespace
std::unique_ptr<CudaSketch> MakeCudaSketch(bst_feature_t columns, bst_bin_t max_bin,
                                           Span<FeatureType const> types,
                                           Span<float const> row_weights) {
  @autoreleasepool {
    return std::make_unique<MetalCudaSketch>(columns, max_bin, types, row_weights);
  }
}

}  // namespace xgboost::common::metal
