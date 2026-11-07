/** Copyright 2026, XGBoost Contributors */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "../collective/aggregator.h"
#include "../collective/allreduce_v_metal.h"
#include "../data/adapter.h"
#include "../data/columnar.h"
#include "../data/gradient_index.h"
#include "../data/metainfo.h"
#include "compressed_iterator.h"
#include "io.h"
#include "metal_adapter.h"
#include "metal_auc.h"
#include "metal_category.h"
#include "metal_collective.h"
#include "metal_vector.h"
#include "metal_constraints.h"
#include "metal_cuda_sketch.h"
#include "metal_data.h"
#include "metal_ellpack.h"
#include "metal_gradient.h"
#include "metal_histogram.h"
#include "metal_leaf.h"
#include "metal_linalg.h"
#include "metal_linear.h"
#include "metal_memory.h"
#include "metal_metadata.h"
#include "metal_model.h"
#include "metal_ops.h"
#include "metal_output.h"
#include "metal_prediction_model.h"
#include "metal_quantile.h"
#include "metal_ranking.h"
#include "metal_reduction.h"
#include "metal_sampling.h"
#include "metal_shader.h"
#include "metal_shap.h"
#include "metal_sketch.h"
#include "metal_snappy.h"
#include "metal_split.h"
#include "metal_survival.h"
#include "metal_tensor.h"
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
std::atomic<std::uint64_t> split_weight_store_dispatches{0}, split_weight_readback_bytes{0},
    model_host_copy_bytes{0}, model_snapshot_bytes{0};
std::atomic<std::uint64_t> prediction_shared_input_bytes{0};
std::atomic<std::uint64_t> split_category_store_dispatches{0}, split_category_readback_bytes{0},
    partition_category_shared_input_bytes{0}, partition_category_host_copy_bytes{0};
std::atomic<std::uint64_t> gradient_sum_shared_output_bytes{0}, gradient_sum_readback_bytes{0}, split_sum_store_dispatches{0}, split_sum_gather_dispatches{0}, split_sum_apply_dispatches{0}, split_sum_shared_input_bytes{0}, split_sum_host_input_bytes{0}, split_sum_readback_bytes{0};
std::atomic<std::uint64_t> leaf_sum_init_dispatches{0}, leaf_sum_dispatches{0},
    leaf_weight_dispatches{0}, leaf_fitting_gpu_time_ns{0};
std::atomic<std::uint64_t> linear_gradient_dispatches{0}, linear_residual_dispatches{0},
    linear_gpu_time_ns{0};
std::atomic<std::uint64_t> shap_dispatches{0}, shap_interaction_dispatches{0}, shap_gpu_time_ns{0};
std::atomic<std::uint64_t> base_score_dispatches{0}, base_score_gpu_time_ns{0};
std::atomic<std::uint64_t> column_init_dispatches{0}, column_sampling_dispatches{0},
    column_sampling_gpu_time_ns{0};
std::atomic<std::uint64_t> feature_shared_input_bytes{0}, feature_shared_output_bytes{0},
    feature_host_input_bytes{0}, feature_readback_bytes{0};
std::atomic<std::uint64_t> feature_task_dispatches{0}, feature_task_candidates{0};
std::atomic<std::uint64_t> split_bounds_gather_dispatches{0}, split_bounds_gather_values{0},
    split_bounds_shared_input_bytes{0}, split_bounds_host_copy_bytes{0};
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
std::atomic<std::uint64_t> ranking_plan_dispatches{0}, ranking_plan_reuses{0},
    ranking_plan_gpu_time_ns{0};
std::atomic<std::uint64_t> ranking_metric_dispatches{0}, ranking_metric_gpu_time_ns{0};
std::atomic<std::uint64_t> ranking_roc_auc_dispatches{0}, ranking_roc_auc_gpu_time_ns{0};
std::atomic<std::uint64_t> ranking_pr_auc_dispatches{0}, ranking_pr_auc_gpu_time_ns{0};
std::atomic<std::uint64_t> ranking_gradient_dispatches{0}, ranking_gradient_gpu_time_ns{0},
    ranking_pairs{0}, ranking_pair_batches{0}, ranking_bias_dispatches{0};
std::atomic<std::uint64_t> ranking_label_dispatches{0}, ranking_label_gpu_time_ns{0},
    ranking_cache_dispatches{0}, ranking_cache_gpu_time_ns{0};
std::atomic<std::uint64_t> metric_shared_input_bytes{0}, metric_host_input_bytes{0};
std::atomic<std::uint64_t> metric_shared_output_bytes{0}, metric_host_output_bytes{0},
    metric_copy_dispatches{0}, metric_copy_bytes{0}, auc_final_readback_bytes{0};
std::atomic<std::uint64_t> float_sum_dispatches{0}, float_sum_gpu_time_ns{0};
std::atomic<std::uint64_t> row_workspace_allocations{0}, row_workspace_reuses{0};
std::atomic<std::uint64_t> index_gpu_time_ns{0};
std::atomic<std::uint64_t> binning_dispatches{0}, binning_gpu_time_ns{0};
std::atomic<std::uint64_t> mean_dispatches{0}, mean_gpu_time_ns{0};
std::atomic<std::uint64_t> small_histogram_dispatches{0}, vector_arithmetic_dispatches{0},
    float_mean_dispatches{0};
std::atomic<std::uint64_t> prediction_model_pack_dispatches{0},
    prediction_model_shared_input_bytes{0}, prediction_model_staging_bytes{0};
std::atomic<std::uint64_t> category_state_shared_input_bytes{0},
    category_state_shared_output_bytes{0}, category_state_staging_bytes{0};
std::atomic<std::uint64_t> category_sort_dispatches{0}, category_recode_dispatches{0},
    category_gpu_time_ns{0};
std::atomic<std::uint64_t> input_prepare_dispatches{0}, input_recode_dispatches{0},
    input_prepare_gpu_time_ns{0}, input_debug_readback_bytes{0};
std::atomic<std::uint64_t> ellpack_pack_dispatches{0}, ellpack_copy_dispatches{0},
    ellpack_restore_dispatches{0}, ellpack_cache_dispatches{0}, ellpack_gpu_time_ns{0},
    ellpack_debug_readback_bytes{0}, ellpack_cached_host_bytes{0}, ellpack_cached_device_bytes{0};
std::atomic<std::uint64_t> ellpack_adapter_dispatches{0}, ellpack_adapter_values{0},
    ellpack_pack_input_bytes{0};
std::atomic<std::uint64_t> ellpack_histogram_dispatches{0}, ellpack_partition_dispatches{0},
    ellpack_input_dispatches{0};
std::atomic<std::uint64_t> ellpack_io_read_pages{0}, ellpack_io_write_pages{0},
    ellpack_io_read_bytes{0}, ellpack_io_write_bytes{0}, ellpack_io_dispatches{0};
std::atomic<std::uint64_t> snappy_compress_dispatches{0}, snappy_decompress_dispatches{0},
    snappy_copy_dispatches{0}, snappy_input_bytes{0}, snappy_compressed_bytes{0},
    snappy_decoded_bytes{0}, snappy_host_readback_bytes{0}, snappy_gpu_time_ns{0};
std::atomic<std::uint64_t> tensor_imports{0}, tensor_copy_dispatches{0}, tensor_copied_bytes{0},
    tensor_adapter_windows{0}, tensor_prediction_windows{0}, tensor_gpu_time_ns{0},
    input_staging_bytes{0};
std::atomic<std::uint64_t> metadata_copy_dispatches{0}, metadata_convert_dispatches{0},
    metadata_validate_dispatches{0}, metadata_qid_dispatches{0}, metadata_values{0},
    metadata_shared_output_bytes{0}, metadata_host_copy_bytes{0}, metadata_gpu_time_ns{0};
std::atomic<std::uint64_t> gradient_import_dispatches{0}, gradient_import_pairs{0},
    gradient_import_shared_output_bytes{0}, gradient_import_host_copy_bytes{0},
    gradient_import_gpu_time_ns{0};
std::atomic<std::uint64_t> tensor_csr_dispatches{0}, tensor_csr_values{0},
    tensor_csr_shared_output_bytes{0}, tensor_csr_host_copy_bytes{0}, tensor_csr_gpu_time_ns{0},
    csr_shared_input_bytes{0}, csr_input_copy_bytes{0};
std::atomic<std::uint64_t> columnar_imports{0}, columnar_pack_bytes{0}, columnar_gpu_time_ns{0},
    category_import_dispatches{0}, category_import_shared_bytes{0}, category_import_copy_bytes{0};
std::atomic<std::uint64_t> prediction_shared_output_bytes{0},
    prediction_host_input_bytes{0},
    prediction_host_output_bytes{0},
    prediction_device_copy_bytes{0},
    output_snapshot_dispatches{0},
    output_snapshot_shared_input_bytes{0},
    output_snapshot_host_copy_bytes{0},
    output_snapshot_bytes{0},
    output_gpu_time_ns{0},
    output_host_readback_bytes{0},
    output_dlpack_exports{0},
    output_live_buffers{0},
    output_live_exports{0};
std::atomic<std::uint64_t> initialization_shared_input_bytes{0}, initialization_shared_output_bytes{0},
    initialization_host_input_bytes{0}, initialization_host_output_bytes{0},
    initialization_copy_dispatches{0}, initialization_copy_bytes{0}, initialization_fill_dispatches{0};
std::atomic<std::uint64_t> vector_allocations{0}, vector_live_buffers{0}, vector_capacity_bytes{0},
    vector_copy_dispatches{0}, vector_copy_bytes{0}, vector_fill_dispatches{0}, vector_fill_bytes{0},
    vector_upload_bytes{0}, vector_readback_bytes{0}, vector_host_copy_bytes{0}, vector_gpu_time_ns{0},
    metadata_gather_dispatches{0}, metadata_gather_values{0}, metadata_equal_dispatches{0};
std::atomic<std::uint64_t> ellpack_csr_dispatches{0}, ellpack_csr_values{0}, ellpack_csr_input_bytes{0};
std::atomic<std::uint64_t> ellpack_view_pages{0}, ellpack_count_dispatches{0},
    ellpack_host_materializations{0}, ellpack_host_readback_bytes{0};
std::atomic<std::uint64_t> adapter_prepare_dispatches{0}, adapter_index_dispatches{0},
    adapter_gpu_time_ns{0}, adapter_debug_readback_bytes{0};
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
std::atomic<std::uint64_t> metric_reduction_passes{0}, metric_final_readback_bytes{0},
    metric_label_dispatches{0};
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
  id<MTLComputePipelineState> quantized_split_pipeline;
  id<MTLComputePipelineState> quantized_multi_split_pipeline;
  id<MTLComputePipelineState> feature_lengths_pipeline, feature_tasks_pipeline,
      feature_winners_pipeline;
  id<MTLComputePipelineState> split_finalize_pipeline, split_bounds_pipeline,
      split_bounds_init_pipeline, split_bounds_gather_pipeline;
  id<MTLComputePipelineState> model_scale_pipeline, split_weight_store_pipeline;
  id<MTLComputePipelineState> split_category_pipeline;
  id<MTLComputePipelineState> interaction_restore_pipeline, interaction_split_pipeline,
      interaction_input_pipeline, interaction_filter_pipeline, interaction_scatter_pipeline;
  id<MTLComputePipelineState> dequantize_hist_pipeline, subtract_hist_pipeline,
      gather_hist_pipeline;
  id<MTLComputePipelineState> store_integer_hist_pipeline, subtract_integer_hist_pipeline,
      sum_integer_hist_pipeline, integer_statistics_view_pipeline;
  id<MTLComputePipelineState> residual_reduce_pipeline, residual_finish_pipeline, adaptive_pipeline;
  id<MTLComputePipelineState> quantile_metric_pipeline, binning_pipeline, metric_sum_pipeline;
  id<MTLComputePipelineState> mean_weight_pipeline, mean_pipeline, mean_result_pipeline;
  id<MTLComputePipelineState> vector_arithmetic_pipeline, float_mean_pipeline,
      small_hist_init_pipeline, small_hist_flags_pipeline, small_hist_boundaries_pipeline,
      small_hist_sum_pipeline;
  id<MTLComputePipelineState> sort_init_pipeline, radix_rank_pipeline, radix_scatter_pipeline,
      prefix_u32_pipeline, prefix_add_pipeline, quantile_cdf_pipeline, quantile_select_pipeline;
  id<MTLComputePipelineState> roc_sort_init_pipeline, roc_counts_pipeline, roc_area_pipeline,
      pr_area_pipeline, pr_totals_pipeline, auc_result_pipeline, auc_normalize_pipeline,
      prefix_double_pair_pipeline, prefix_double_pair_add_pipeline;
  id<MTLComputePipelineState> group_sort_small_pipeline, group_sort_medium_pipeline,
      group_sort_key_pipeline, group_sort_export_pipeline, ranking_score_pipeline,
      ranking_map_labels_pipeline;
  id<MTLComputePipelineState> ranking_labels_pipeline, ndcg_discount_pipeline, ndcg_ideal_pipeline;
  id<MTLComputePipelineState> ranking_plan_init_pipeline, ranking_plan_finish_pipeline;
  id<MTLComputePipelineState> ranking_map_stats_pipeline, ranking_pairs_pipeline,
      ranking_normalize_pipeline, ranking_gather_pipeline, ranking_mean_labels_pipeline,
      ranking_mean_sample_pipeline, ranking_mean_links_pipeline, ranking_mean_gather_pipeline,
      ranking_bias_sum_pipeline, ranking_bias_finish_pipeline, ranking_bias_update_pipeline,
      ranking_bias_min_pipeline, ranking_cost_rounding_pipeline;
  id<MTLComputePipelineState> aft_gradient_pipeline, aft_metric_pipeline, aft_transform_pipeline;
  id<MTLComputePipelineState> cs_has_categories_pipeline, cs_sort_keys_pipeline;
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
  id<MTLComputePipelineState> residual_stats_pipeline, residual_apply_pipeline,
      float32_sum_pipeline;
  id<MTLComputePipelineState> category_iota_pipeline, category_merge_pipeline,
      category_extract_pipeline, category_validate_pipeline, category_recode_pipeline;
  id<MTLComputePipelineState> input_prepare_pipeline;
  id<MTLComputePipelineState> gradient_import_pipeline;
  id<MTLComputePipelineState> tensor_gather_bytes_pipeline;
  id<MTLComputePipelineState> metadata_convert_pipeline, metadata_validate_pipeline,
      metadata_qid_mark_pipeline, metadata_qid_starts_pipeline, metadata_qid_counts_pipeline;
  id<MTLComputePipelineState> initialization_store_pipeline, vector_center_pipeline;
  id<MTLComputePipelineState> vector_fill_pipeline, vector_equal_pipeline, metadata_gather_pipeline;
  id<MTLComputePipelineState> collective_reduce_pipeline;
  id<MTLComputePipelineState> ellpack_symbols_pipeline, ellpack_pack_pipeline,
      ellpack_copy_pipeline, ellpack_count_pipeline, ellpack_unpack_pipeline;
  id<MTLComputePipelineState> ellpack_adapter_pipeline, ellpack_null_pipeline, ellpack_csr_pipeline;
  id<MTLComputePipelineState> snappy_compress_pipeline, snappy_decompress_pipeline;
  id<MTLComputePipelineState> ellpack_histogram_pipeline, ellpack_shared_histogram_pipeline,
      ellpack_tiled_histogram_pipeline, ellpack_sparse_histogram_pipeline,
      ellpack_partition_pipeline, ellpack_input_pipeline;
  id<MTLComputePipelineState> adapter_count_pipeline, adapter_scatter_pipeline,
      adapter_offsets_pipeline;
  id<MTLComputePipelineState> multiclass_objective_pipeline, multiclass_transform_pipeline;
  id<MTLComputePipelineState> quantize_pipeline, objective_pipeline, transform_pipeline,
      prediction_pipeline, prediction_model_pipeline, leaf_pipeline, metric_pipeline;
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
      multiclass_objective_pipeline = make_pipeline(@"xgb_multiclass_objective");
      multiclass_transform_pipeline = make_pipeline(@"xgb_multiclass_transform");
      objective_pipeline = make_pipeline(@"xgb_objective");
      transform_pipeline = make_pipeline(@"xgb_transform");
      prediction_pipeline = make_pipeline(@"xgb_predict");
      prediction_model_pipeline = make_pipeline(@"xgb_pack_prediction_model");
      leaf_pipeline = make_pipeline(@"xgb_add_leaf");
      metric_pipeline = make_pipeline(@"xgb_metric");
      metric_sum_pipeline = make_pipeline(@"xgb_metric_sum");
      row_iota_pipeline = make_pipeline(@"xgb_row_iota");
      partition_classify_pipeline = make_pipeline(@"xgb_partition_classify");
      partition_prefix_pipeline = make_pipeline(@"xgb_partition_prefix");
      partition_prefix_serial_pipeline = make_pipeline(@"xgb_partition_prefix_serial");
      partition_scatter_pipeline = make_pipeline(@"xgb_partition_scatter");
      leaf_positions_pipeline = make_pipeline(@"xgb_leaf_positions");
      transpose_index_pipeline = make_pipeline(@"xgb_transpose_index");
      quantized_split_pipeline = make_pipeline(@"xgb_evaluate_quantized_splits");
      quantized_multi_split_pipeline = make_pipeline(@"xgb_evaluate_quantized_multi_splits");
      feature_lengths_pipeline = make_pipeline(@"xgb_feature_split_lengths");
      feature_tasks_pipeline = make_pipeline(@"xgb_feature_split_tasks");
      feature_winners_pipeline = make_pipeline(@"xgb_feature_split_winners");
      split_finalize_pipeline = make_pipeline(@"xgb_finalize_split_nodes");
      split_weight_store_pipeline = make_pipeline(@"xgb_store_split_weights");
      split_bounds_pipeline = make_pipeline(@"xgb_update_split_bounds");
      split_bounds_init_pipeline = make_pipeline(@"xgb_init_split_bounds");
      split_bounds_gather_pipeline = make_pipeline(@"xgb_gather_split_bounds");
      model_scale_pipeline = make_pipeline(@"xgb_scale_model_weights");
      split_category_pipeline = make_pipeline(@"xgb_build_split_categories");
      interaction_restore_pipeline = make_pipeline(@"xgb_interaction_restore");
      interaction_split_pipeline = make_pipeline(@"xgb_interaction_split");
      interaction_input_pipeline = make_pipeline(@"xgb_interaction_input");
      interaction_filter_pipeline = make_pipeline(@"xgb_interaction_filter");
      interaction_scatter_pipeline = make_pipeline(@"xgb_interaction_scatter");
      dequantize_hist_pipeline = make_pipeline(@"xgb_histogram_dequantize");
      subtract_hist_pipeline = make_pipeline(@"xgb_histogram_subtract");
      gather_hist_pipeline = make_pipeline(@"xgb_histogram_gather");
      store_integer_hist_pipeline = make_pipeline(@"xgb_histogram_store_integer");
      subtract_integer_hist_pipeline = make_pipeline(@"xgb_histogram_subtract_integer");
      sum_integer_hist_pipeline = make_pipeline(@"xgb_histogram_sum_integer");
      integer_statistics_view_pipeline = make_pipeline(@"xgb_integer_statistics_view");
      residual_stats_pipeline = make_pipeline(@"xgb_residual_stats_reduce");
      residual_apply_pipeline = make_pipeline(@"xgb_residual_scale_apply");
      float32_sum_pipeline = make_pipeline(@"xgb_float32_sum");
      category_iota_pipeline = make_pipeline(@"xgb_category_iota");
      category_merge_pipeline = make_pipeline(@"xgb_category_merge");
      category_extract_pipeline = make_pipeline(@"xgb_category_extract");
      category_validate_pipeline = make_pipeline(@"xgb_category_validate");
      category_recode_pipeline = make_pipeline(@"xgb_category_recode");
      input_prepare_pipeline = make_pipeline(@"xgb_prepare_input");
      gradient_import_pipeline = make_pipeline(@"xgb_import_gradient_pairs");
      tensor_gather_bytes_pipeline = make_pipeline(@"xgb_tensor_gather_bytes");
      metadata_convert_pipeline = make_pipeline(@"xgb_metadata_convert");
      initialization_store_pipeline = make_pipeline(@"xgb_initialization_store");
      vector_center_pipeline = make_pipeline(@"xgb_vector_center");
      vector_fill_pipeline = make_pipeline(@"xgb_vector_fill");
      vector_equal_pipeline = make_pipeline(@"xgb_vector_equal");
      collective_reduce_pipeline = make_pipeline(@"xgb_collective_reduce");
      metadata_gather_pipeline = make_pipeline(@"xgb_metadata_gather");
      metadata_validate_pipeline = make_pipeline(@"xgb_metadata_validate");
      metadata_qid_mark_pipeline = make_pipeline(@"xgb_metadata_qid_mark");
      metadata_qid_starts_pipeline = make_pipeline(@"xgb_metadata_qid_starts");
      metadata_qid_counts_pipeline = make_pipeline(@"xgb_metadata_qid_counts");
      ellpack_symbols_pipeline = make_pipeline(@"xgb_ellpack_symbols");
      ellpack_pack_pipeline = make_pipeline(@"xgb_ellpack_pack");
      ellpack_copy_pipeline = make_pipeline(@"xgb_ellpack_copy");
      ellpack_count_pipeline = make_pipeline(@"xgb_ellpack_count");
      ellpack_unpack_pipeline = make_pipeline(@"xgb_ellpack_unpack");
      ellpack_adapter_pipeline = make_pipeline(@"xgb_ellpack_adapter");
      ellpack_csr_pipeline = make_pipeline(@"xgb_ellpack_csr");
      ellpack_null_pipeline = make_pipeline(@"xgb_ellpack_null");
      snappy_compress_pipeline = make_pipeline(@"xgb_snappy_compress");
      snappy_decompress_pipeline = make_pipeline(@"xgb_snappy_decompress");
      ellpack_histogram_pipeline = make_pipeline(@"xgb_histogram_ellpack");
      ellpack_shared_histogram_pipeline = make_pipeline(@"xgb_histogram_dense_shared_ellpack");
      ellpack_tiled_histogram_pipeline = make_pipeline(@"xgb_histogram_dense_tiled_ellpack");
      ellpack_sparse_histogram_pipeline = make_pipeline(@"xgb_histogram_sparse_shared_ellpack");
      ellpack_partition_pipeline = make_pipeline(@"xgb_partition_classify_ellpack");
      ellpack_input_pipeline = make_pipeline(@"xgb_prepare_ellpack_input");
      adapter_count_pipeline = make_pipeline(@"xgb_adapter_count");
      adapter_scatter_pipeline = make_pipeline(@"xgb_adapter_scatter");
      adapter_offsets_pipeline = make_pipeline(@"xgb_adapter_offsets");
      residual_reduce_pipeline = make_pipeline(@"xgb_residual_scale_reduce");
      residual_finish_pipeline = make_pipeline(@"xgb_residual_scale_finish");
      adaptive_pipeline = make_pipeline(@"xgb_adaptive_objective");
      quantile_metric_pipeline = make_pipeline(@"xgb_quantile_metric");
      uniform_sampling_pipeline = make_pipeline(@"xgb_uniform_sample");
      cs_has_categories_pipeline = make_pipeline(@"xgb_cs_has_categories");
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
      vector_arithmetic_pipeline = make_pipeline(@"xgb_vector_arithmetic");
      float_mean_pipeline = make_pipeline(@"xgb_float32_mean");
      small_hist_init_pipeline = make_pipeline(@"xgb_small_hist_init");
      small_hist_flags_pipeline = make_pipeline(@"xgb_small_hist_flags");
      small_hist_boundaries_pipeline = make_pipeline(@"xgb_small_hist_boundaries");
      small_hist_sum_pipeline = make_pipeline(@"xgb_small_hist_sum");
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
      ranking_plan_init_pipeline = make_pipeline(@"xgb_ranking_plan_init");
      ranking_plan_finish_pipeline = make_pipeline(@"xgb_ranking_plan_finish");
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

BorrowedMemory BindGradientStateInput(Runtime const& rt, void const* data, std::size_t bytes,
                                      std::size_t alignment, bool* shared = nullptr) {
  auto input = TryBorrowMemory(rt.device, data, bytes, alignment);
  if (shared) *shared = input.buffer != nil;
  if (input.buffer) {
    gradient_state_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    input = {Upload(rt, data, bytes), 0};
    gradient_state_staging_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}

// A synchronous view: callers keep their sparse-page allocations alive until
// all dependent commands complete and the enclosing autorelease pool drains.
BorrowedMemory BindCSRInput(Runtime const& rt, void const* data, std::size_t bytes) {
  auto view = TryBorrowMemory(rt.device, data, bytes, 8);
  if (view.buffer) {
    csr_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    view = {Upload(rt, data, bytes), 0};
    csr_input_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return view;
}

void EncodeIntegerStatistics(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> integers,
                             id<MTLBuffer> output, id<MTLBuffer> scales, std::size_t count,
                             std::size_t targets, std::size_t input_offset = 0,
                             std::size_t output_offset = 0) {
  CHECK_GT(targets, 0);
  CHECK_LE(count, std::numeric_limits<std::uint32_t>::max());
  if (!count) return;
  auto e = [command computeCommandEncoder];
  CHECK(e);
  std::uint64_t shape[]{count, targets};
  [e setComputePipelineState:rt.integer_statistics_view_pipeline];
  [e setBuffer:integers offset:input_offset atIndex:0];
  [e setBuffer:output offset:output_offset atIndex:1];
  [e setBuffer:scales offset:0 atIndex:2];
  [e setBytes:shape length:sizeof(shape) atIndex:3];
  [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
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
  id<MTLBuffer> node_input, feature_input, node_parents, node_precise_parents;
  id<MTLBuffer> node_results, node_weights;
  id<MTLBuffer> gathered_bounds;
  id<MTLBuffer> category_input, category_output, category_error;
  id<MTLBuffer> feature_ids, feature_lengths, feature_segments, feature_offsets,
      feature_error, feature_selected;
  std::size_t candidate_count{0};
  std::size_t candidate_child_count{0};
  struct SumShape { std::uint32_t parent_begin, targets; };
  std::vector<SumShape> quantized_multi_shapes;
  std::vector<bool> quantized_multi_valid;
  bool vector_candidates{false};
  std::vector<SplitFinalizeNode> finalized_nodes;
  std::size_t category_count{0};
  std::uint32_t category_words{0};
  std::vector<BufferHolder> metadata;
  void ResetCandidates() {
    candidate_count = candidate_child_count = 0;
    vector_candidates = false;
    quantized_multi_shapes.clear();
    quantized_multi_valid.clear();
    finalized_nodes.clear();
    category_count = category_words = 0;
  }
};
id<MTLBuffer> Scratch(Runtime const& rt, id<MTLBuffer> __strong* buffer, std::size_t bytes,
                      void const* data = nullptr) {
  EnsureBuffer(rt, buffer, bytes);
  if (data && bytes) std::memcpy((*buffer).contents, data, bytes);
  return *buffer;
}
BorrowedMemory BindSplitSums(Runtime const& rt, id<MTLBuffer> __strong* fallback,
                              void const* data, std::size_t bytes) {
  auto input = TryBorrowMemory(rt.device, data, bytes, 8);
  if (input.buffer) {
    split_sum_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    input = {Scratch(rt, fallback, bytes, data), 0};
    split_sum_host_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}
BorrowedMemory BindSplitBounds(Runtime const& rt, void const* data, std::size_t bytes) {
  auto input = TryBorrowMemory(rt.device, data, bytes, 4);
  if (input.buffer) {
    split_bounds_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    input = {Upload(rt, data, bytes), 0};
    split_bounds_host_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
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

struct CompressedParams {
  std::uint64_t split;
  std::uint32_t stride, bits, symbols, dense, features, padding;
};
static_assert(sizeof(CompressedParams) == 32);
class Page final : public QuantizedPage {
  id<MTLBuffer> partition_index_;
  std::once_flag partition_index_once_;

 public:
  id<MTLBuffer> index, row_ptr, offsets, cuts;
  id<MTLBuffer> tail;
  CompressedParams compressed{};
  std::size_t compressed_bytes{0};
  bool IsCompressed() const { return compressed.bits != 0; }
  id<MTLBuffer> PartitionIndex(Runtime const& rt, std::size_t rows) {
    // Large, narrow dense pages repeatedly gather one feature per row during
    // partitioning. A column cache improves locality without changing the
    // canonical compressed page used by histograms, prediction and exports.
    // Bound the extra allocation; wide/small/split-storage pages stay compressed.
    if (!IsCompressed() || !compressed.dense || compressed.bits != 8 || compressed.symbols != 256 ||
        compressed.stride != compressed.features || compressed.features < 4 ||
        compressed.features > 64 || rows < (std::size_t{1} << 20) ||
        (compressed.split != 0 && compressed.split < compressed_bytes)) return nil;
    auto budget = std::min<std::size_t>(std::size_t{512} << 20,
                                       rt.device.recommendedMaxWorkingSetSize / 32);
    if (rows > budget / compressed.features) return nil;
    std::call_once(partition_index_once_, [&] {
      auto bytes = rows * compressed.features;
      auto source = compressed.split == 0 ? tail : index;
      CHECK_LE(bytes + 8, source.length);
      auto output = Allocate(rt, bytes);
      struct {
        std::uint64_t rows;
        std::uint32_t features, width;
      } p{rows, compressed.features, 1};
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      auto encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:rt.transpose_index_pipeline];
      [encoder setBuffer:source offset:8 atIndex:0];
      [encoder setBuffer:output offset:0 atIndex:1];
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
      partition_index_ = output;
    });
    return partition_index_;
  }
  Page(Runtime const& rt, HistogramCuts const& values, EllpackInfo info, id<MTLBuffer> head,
       id<MTLBuffer> tail, std::uint64_t split, std::size_t bytes)
      : index{head}, tail{tail}, compressed{split,
                                            info.row_stride,
                                            info.bits,
                                            info.symbols,
                                            std::uint32_t(info.dense),
                                            info.features,
                                            0}, compressed_bytes{bytes} {
    // Compressed consumers do not need restored indices, row pointers or a transpose.
    row_ptr = Allocate(rt, 0);
    offsets = Upload(rt, values.Ptrs().data(), values.Ptrs().size() * 4);
    cuts = Upload(rt, values.Values().data(), values.Values().size() * 4);
  }
  Page(Runtime const& rt, GHistIndexMatrix const& gidx) {
    gidx.RequireHost();
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
  void AllReduce(Context const* ctx) override {
    CHECK(ctx);
    CHECK(quantizer) << "Metal leaf collectives require integer statistics.";
    if (!collective::GlobalCommGroup()->IsDistributed() || !leaves) return;
    @autoreleasepool {
      auto& rt = GetRuntime();
      auto count = leaves * targets;
      auto view = linalg::MakeVec(static_cast<std::int64_t*>(integers.contents), count * 2,
                                  DeviceOrd::MPS());
      collective::SafeColl(collective::GlobalSum(ctx, view));
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      EncodeIntegerStatistics(rt, command, integers, buffer, scales, count, targets);
      CompleteOperation(command, nullptr, &leaf_fitting_gpu_time_ns);
    }
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

struct LeafSamplingParams {
  // 0 retains scalar Hessian validity; 1/2/3 replay none/uniform/gradient sampling.
  std::uint32_t method{0}, seed{0};
  float probability{1.f};
  std::uint32_t padding{0};
};
static_assert(sizeof(LeafSamplingParams) == 16);
void BindLeafSampling(id<MTLComputeCommandEncoder> encoder, QuantizedGradients const* sampling,
                      id<MTLBuffer> fallback);

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
    bool full_coverage = true;
    std::size_t covered = 0;
    for (std::size_t node = 0; node < splits.size(); ++node) {
      auto& split = splits[node];
      CHECK_LE(split.cat_begin, categories.size());
      CHECK_LE(split.cat_size, categories.size() - split.cat_begin);
      CHECK_LE(split.end, size_);
      CHECK_LE(split.begin, split.end);
      full_coverage = full_coverage && split.begin == covered;
      covered = split.end;
      split.block_begin = blocks.size();
      for (auto begin = split.begin; begin < split.end; begin += 256)
        blocks.push_back(
            {begin, std::min(begin + 256, split.end), static_cast<std::uint32_t>(node), 0});
      split.block_end = blocks.size();
      largest_block_count =
          std::max(largest_block_count, std::size_t{split.block_end - split.block_begin});
    }
    full_coverage = full_coverage && covered == size_;
    bool serial_prefix = largest_block_count <= 16;
    std::vector<std::uint64_t> result(splits.size(), 0);
    if (blocks.empty()) return result;
    @autoreleasepool {
      auto page = GetPage(rt_, gidx);
      auto column_index = page->PartitionIndex(rt_, size_);
      bool compressed = page->IsCompressed() && !column_index;
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      this->EnsureReady(command);
      Workspace(&saved_rows_, size_ * sizeof(bst_idx_t));
      Workspace(&ranks_, size_ * sizeof(std::uint32_t));
      auto split_buffer =
          Workspace(&splits_, splits.size() * sizeof(PartitionSplit), splits.data());
      auto block_buffer = Workspace(&blocks_, blocks.size() * sizeof(Block), blocks.data());
      auto cats = TryBorrowMemory(rt_.device, categories.data(), categories.size_bytes(), 4);
      if (cats.buffer) {
        partition_category_shared_input_bytes.fetch_add(categories.size_bytes(),
                                                         std::memory_order_relaxed);
      } else {
        cats = {Workspace(&categories_, categories.size_bytes(), categories.data()), 0};
        partition_category_host_copy_bytes.fetch_add(categories.size_bytes(),
                                                      std::memory_order_relaxed);
      }
      auto counts = Workspace(&counts_, blocks.size() * sizeof(std::uint32_t));
      auto prefix = Workspace(&prefix_, blocks.size() * sizeof(std::uint32_t));
      auto left = Workspace(&left_, splits.size() * sizeof(std::uint64_t));
      Params p{size_,
               base_,
               column_index ? 1u : static_cast<std::uint32_t>(gidx.index.GetBinTypeSize()),
               static_cast<std::uint32_t>(column_index || gidx.IsDense()),
               static_cast<std::uint32_t>(splits.size()),
               static_cast<std::uint32_t>(blocks.size())};
      auto encoder = [command computeCommandEncoder];
      CHECK(encoder);
      [encoder setComputePipelineState:compressed ? rt_.ellpack_partition_pipeline
                                                 : rt_.partition_classify_pipeline];
      [encoder setBuffer:column_index ? column_index : page->index offset:0 atIndex:0];
      [encoder setBuffer:page->row_ptr offset:0 atIndex:1];
      [encoder setBuffer:page->offsets offset:0 atIndex:2];
      [encoder setBuffer:page->cuts offset:0 atIndex:3];
      [encoder setBuffer:rows_ offset:0 atIndex:4];
      [encoder setBuffer:split_buffer offset:0 atIndex:5];
      [encoder setBuffer:block_buffer offset:0 atIndex:6];
      [encoder setBuffer:cats.buffer offset:cats.offset atIndex:7];
      [encoder setBuffer:ranks_ offset:0 atIndex:8];
      [encoder setBuffer:counts offset:0 atIndex:9];
      // A complete ordered cover can scatter into the spare buffer and swap
      // ownership. Partial or unordered segments retain the copy/in-place path
      // so rows outside the selected ranges are preserved.
      [encoder setBuffer:(full_coverage ? rows_ : saved_rows_) offset:0 atIndex:10];
      [encoder setBytes:&p length:sizeof(p) atIndex:11];
      if (compressed) {
        [encoder setBuffer:page->tail offset:0 atIndex:12];
        [encoder setBytes:&page->compressed length:sizeof(page->compressed) atIndex:13];
      }
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
      [encoder setBuffer:(full_coverage ? saved_rows_ : rows_) offset:0 atIndex:0];
      [encoder setBuffer:(full_coverage ? rows_ : saved_rows_) offset:0 atIndex:1];
      [encoder setBuffer:split_buffer offset:0 atIndex:2];
      [encoder setBuffer:block_buffer offset:0 atIndex:3];
      [encoder setBuffer:ranks_ offset:0 atIndex:4];
      [encoder setBuffer:prefix offset:0 atIndex:5];
      [encoder setBuffer:left offset:0 atIndex:6];
      [encoder dispatchThreadgroups:MTLSizeMake(blocks.size(), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      CompleteTraining(command);
      if (full_coverage) std::swap(rows_, saved_rows_);
      this->MarkReady();
      std::memcpy(result.data(), left.contents, result.size() * sizeof(std::uint64_t));
      partition_dispatches.fetch_add(1, std::memory_order_relaxed);
      if (page->IsCompressed())
        ellpack_partition_dispatches.fetch_add(1, std::memory_order_relaxed);
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
                     Span<bst_node_t> output, QuantizedGradients const* sampling) override {
    if (!size_) return;
    CHECK_GE(output.size(), base_ + size_);
    CHECK_GE(gradients.Shape(0), base_ + size_);
    if (sampling) CHECK_EQ(sampling->Rows(), gradients.Shape(0));
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
      BindLeafSampling(encoder, sampling, leaf_buffer);
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
  std::pair<BorrowedMemory, std::size_t> EncodeBounds(Span<GradientPair const> gradients,
      id<MTLCommandBuffer> command, bool quantize) {
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
    return {input, shared_bytes};
  }
  void EncodeScale(std::uint64_t count, id<MTLCommandBuffer> command) {
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt_.gradient_scale_pipeline];
    [encoder setBuffer:gradient_bounds_ offset:0 atIndex:0];
    [encoder setBuffer:gradient_scale_ offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
  }
  void EncodeQuantize(BorrowedMemory input, std::uint64_t count, id<MTLCommandBuffer> command) {
    auto encoder = [command computeCommandEncoder];
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
  void ReadBounds(double* out) const {
    std::memcpy(out, gradient_bounds_.contents, 4 * sizeof(double));
  }
  void SetGlobalScale(double const* bounds, std::uint64_t rows) {
    // CUDA quantiser.cu::BuildQuantiserFromPair runs on HOST after GlobalSum.
    // Each factor is a power of two; Metal stores its reciprocal's exponent.
    for (unsigned i = 0; i < 4; ++i)
      CHECK(std::isfinite(bounds[i])) << "Metal quantization requires finite gradients and Hessians.";
    auto exponent = [&](unsigned component) {
      auto magnitude = std::max(bounds[component], bounds[component + 2]);
      auto delta = magnitude / (1.0 - 2.0 * static_cast<double>(rows) *
                                         std::numeric_limits<double>::epsilon());
      CHECK(std::isfinite(delta)) << "Metal quantizer rounding factor is not finite.";
      int exp = 0;
      std::frexp(delta, &exp);
      return 62 - exp;
    };
    GradientScale scale{exponent(0), exponent(1), 1, 0};
    std::memcpy(gradient_scale_.contents, &scale, sizeof(scale));
  }
};

std::vector<std::unique_ptr<MetalQuantizedColumn>> MakeGradientColumns(
    Runtime& rt, linalg::MatrixView<GradientPair const> gradients, bool quantize,
    Context const* ctx) {
  std::vector<std::unique_ptr<MetalQuantizedColumn>> columns;
  auto distributed = collective::GlobalCommGroup()->IsDistributed();
  CHECK(ctx || !distributed) << "Distributed Metal quantization requires a Context.";
  CheckGradientView(gradients, false);
  auto rows = gradients.Shape(0);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_GT(gradients.Shape(1), 0);
  CHECK_LE(gradients.Shape(1), std::numeric_limits<std::uint32_t>::max());
  // Quantization is per target. A GPU layout copy handles C order, padding and
  // broadcasts; its temporary raw floats never own the persistent tree state.
  linalg::Matrix<GradientPair> contiguous;
  if (gradients.Stride(0) != 1) {
    contiguous =
        linalg::Matrix<GradientPair>{gradients.Shape(), DeviceOrd::MPS(), linalg::Order::kF};
    CopyGradients(gradients, contiguous.View(DeviceOrd::MPS()));
    gradients = std::as_const(contiguous).View(DeviceOrd::MPS());
  }
  // Release command-held borrowed buffers before the owning layout tensor dies.
  @autoreleasepool {
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    std::size_t shared_bytes = 0;
    std::vector<BorrowedMemory> inputs;
    for (std::size_t t = 0; t < gradients.Shape(1); ++t) {
      auto column = std::make_unique<MetalQuantizedColumn>(rt);
      auto [input, shared] = column->EncodeBounds(
          gradients.Slice(linalg::All(), t).Values().subspan(0, rows), command, quantize);
      inputs.push_back(input);
      shared_bytes += shared;
      if (!distributed) {
        column->EncodeScale(rows, command);
        if (quantize) column->EncodeQuantize(input, rows, command);
      }
      columns.push_back(std::move(column));
    }
    auto complete = [&] {
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      if (!quantize && command.GPUEndTime > command.GPUStartTime)
        leaf_fitting_gpu_time_ns.fetch_add(
            static_cast<std::uint64_t>((command.GPUEndTime - command.GPUStartTime) * 1e9),
            std::memory_order_relaxed);
    };
    complete();
    if (distributed) {
      std::vector<double> bounds(4 * columns.size());
      for (std::size_t t = 0; t < columns.size(); ++t) columns[t]->ReadBounds(bounds.data() + 4 * t);
      // These small control reductions intentionally use HOST storage, as CUDA.
      std::uint64_t global_rows = rows;
      auto rc = collective::GlobalSum(ctx, linalg::MakeVec(bounds.data(), bounds.size())) << [&] {
        return collective::GlobalSum(ctx, linalg::MakeVec(&global_rows, 1));
      };
      collective::SafeColl(rc);
      for (std::size_t t = 0; t < columns.size(); ++t)
        columns[t]->SetGlobalScale(bounds.data() + 4 * t, global_rows);
      if (quantize) {
        command = [rt.queue commandBuffer];
        CHECK(command);
        for (std::size_t t = 0; t < columns.size(); ++t)
          columns[t]->EncodeQuantize(inputs[t], rows, command);
        complete();
      }
    }
    for (auto const& column : columns) {
      CHECK(static_cast<GradientScale const*>(column->gradient_scale_.contents)->valid)
          << "Metal quantization requires finite gradients and Hessians.";
      column->ReleaseScratch();
    }
    gradient_bound_dispatches.fetch_add(columns.size(), std::memory_order_relaxed);
    if (quantize) quantization_dispatches.fetch_add(columns.size(), std::memory_order_relaxed);
    (quantize ? histogram_shared_gradient_bytes : leaf_shared_gradient_bytes)
        .fetch_add(shared_bytes, std::memory_order_relaxed);
  }
  return columns;
}
class MetalGradientQuantizer final : public GradientQuantizer {
  std::size_t rows_;

 public:
  std::vector<std::unique_ptr<MetalQuantizedColumn>> columns;
  explicit MetalGradientQuantizer(linalg::MatrixView<GradientPair const> original, Context const* ctx)
      : rows_{original.Shape(0)}, columns{MakeGradientColumns(GetRuntime(), original, false, ctx)} {}
  std::size_t Rows() const override { return rows_; }
  std::size_t Targets() const override { return columns.size(); }
};

MetalLeafSums::MetalLeafSums(std::size_t n_leaves, std::size_t n_targets,
                             std::shared_ptr<GradientQuantizer const> original_quantizer)
    : quantizer{std::move(original_quantizer)}, leaves{n_leaves}, targets{n_targets} {
  CHECK_GT(targets, 0);
  CHECK_LE(targets, std::numeric_limits<std::uint32_t>::max());
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
  friend void BindLeafSampling(id<MTLComputeCommandEncoder>, QuantizedGradients const*,
                               id<MTLBuffer>);
  Runtime& rt_;
  std::size_t rows_;
  float probability_{1.f};
  std::uint32_t seed_{0};
  bool sampled_{false}, gradient_sampling_{false};
  id<MTLBuffer> norms_, sampling_state_;

 public:
  std::vector<std::unique_ptr<MetalQuantizedColumn>> columns;
  explicit MetalQuantizedGradients(linalg::MatrixView<GradientPair const> gradients, Context const* ctx)
      : rt_{GetRuntime()}, rows_{gradients.Shape(0)} {
    columns = MakeGradientColumns(rt_, gradients, true, ctx);
  }
  std::size_t Rows() const override { return rows_; }
  std::size_t Targets() const override { return columns.size(); }
  void Sum(linalg::VectorView<GradientPairPrecise> output,
           Span<GradientPairInt64> integers) const override {
    SumImpl(nullptr, DeviceOrd::CPU(), output, integers);
  }
  void SumRoot(Context const* ctx, DeviceOrd device, linalg::VectorView<GradientPairPrecise> output,
               Span<GradientPairInt64> integers) const override {
    CHECK(ctx);
    CHECK(device.IsCPU() || device.IsMPS()) << "Invalid Metal root reduction device.";
    SumImpl(ctx, device, output, integers);
  }
  void SumImpl(Context const* ctx, DeviceOrd device, linalg::VectorView<GradientPairPrecise> output,
               Span<GradientPairInt64> integers) const {
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
      CompleteOperation(command, nullptr, &gradient_sum_gpu_time_ns);
      bool distributed = ctx && collective::GlobalCommGroup()->IsDistributed();
      if (distributed && device.IsCPU()) {
        CHECK(output.Device().IsCPU());
        // CUDA's scalar root explicitly returns integer pairs to host before
        // GlobalSum/ToFloatingPoint. Do not globally add its rounded double view.
        std::vector<GradientPairInt64> host(columns.size());
        std::memcpy(host.data(), exact.contents, host.size() * sizeof(GradientPairInt64));
        auto view = linalg::MakeVec(reinterpret_cast<std::int64_t*>(host.data()), host.size() * 2);
        collective::SafeColl(collective::GlobalSum(ctx, view));
        for (std::size_t t = 0; t < columns.size(); ++t) {
          auto scale = static_cast<GradientScale const*>(columns[t]->gradient_scale_.contents);
          output(t) = {std::ldexp(static_cast<double>(host[t].GetQuantisedGrad()), -scale->grad_exponent),
                       std::ldexp(static_cast<double>(host[t].GetQuantisedHess()), -scale->hess_exponent)};
        }
        if (!integers.empty()) std::copy(host.begin(), host.end(), integers.begin());
        gradient_sum_readback_bytes.fetch_add(host.size() * sizeof(GradientPairInt64),
                                               std::memory_order_relaxed);
      } else {
        if (distributed) {
          auto view = linalg::MakeVec(static_cast<std::int64_t*>(exact.contents), columns.size() * 2,
                                      DeviceOrd::MPS());
          collective::SafeColl(collective::GlobalSum(ctx, view));
          command = [rt_.queue commandBuffer];
          CHECK(command);
          for (std::size_t t = 0; t < columns.size(); ++t)
            EncodeIntegerStatistics(rt_, command, exact, result, columns[t]->gradient_scale_, 1, 1,
                                    t * sizeof(GradientPairInt64), t * sizeof(GradientPairPrecise));
          CompleteOperation(command, nullptr, &gradient_sum_gpu_time_ns);
        }
        // Publish resident roots with GPU copies, including strided/offset views.
        // CPU diagnostics retain the explicit host-publication behavior.
        auto out = output.Device().IsMPS()
                       ? TryBorrowMemory(rt_.device, output.Values().data(),
                                         output.Values().size_bytes(), 8)
                       : BorrowedMemory{};
        CHECK(!output.Device().IsMPS() || out.buffer);
        auto ints = integers.empty() ? BorrowedMemory{}
                                     : TryBorrowMemory(rt_.device, integers.data(),
                                                       integers.size_bytes(), 8);
        if (out.buffer || ints.buffer) {
          command = [rt_.queue commandBuffer];
          auto e = [command blitCommandEncoder];
          CHECK(e);
          if (out.buffer) {
            for (std::size_t t = 0; t < columns.size(); ++t)
              [e copyFromBuffer:result sourceOffset:t * sizeof(GradientPairPrecise)
                      toBuffer:out.buffer
             destinationOffset:out.offset + t * output.Stride(0) * sizeof(GradientPairPrecise)
                          size:sizeof(GradientPairPrecise)];
            gradient_sum_shared_output_bytes.fetch_add(output.Size() * sizeof(GradientPairPrecise),
                                                         std::memory_order_relaxed);
          }
          if (ints.buffer) {
            [e copyFromBuffer:exact sourceOffset:0 toBuffer:ints.buffer
                destinationOffset:ints.offset size:integers.size_bytes()];
            gradient_sum_shared_output_bytes.fetch_add(integers.size_bytes(),
                                                         std::memory_order_relaxed);
          }
          [e endEncoding];
          CompleteOperation(command, nullptr, &gradient_sum_gpu_time_ns);
        }
        if (!out.buffer) {
          auto values = static_cast<GradientPairPrecise const*>(result.contents);
          for (std::size_t t = 0; t < columns.size(); ++t) output(t) = values[t];
          gradient_sum_readback_bytes.fetch_add(output.Size() * sizeof(GradientPairPrecise),
                                                 std::memory_order_relaxed);
        }
        if (!integers.empty() && !ints.buffer) {
          std::memcpy(integers.data(), exact.contents, integers.size_bytes());
          gradient_sum_readback_bytes.fetch_add(integers.size_bytes(), std::memory_order_relaxed);
        }
      }
      gradient_sum_dispatches.fetch_add(1, std::memory_order_relaxed);
      quantized_gradient_sum_dispatches.fetch_add(1, std::memory_order_relaxed);
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
      // Each row owns disjoint output elements. Bind resident floats directly;
      // unshareable output retains the previous staging/padding behavior.
      auto span = values.Values();
      bool shared = false;
      auto data = BindGradientStateInput(rt_, span.data(), span.size_bytes(), 8, &shared);
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
      [e setBuffer:data.buffer offset:data.offset atIndex:0];
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
      if (!shared) std::memcpy(span.data(), data.buffer.contents, span.size_bytes());
      (shared ? gradient_state_shared_output_bytes : gradient_state_staging_bytes)
          .fetch_add(span.size_bytes(), std::memory_order_relaxed);
    }
  }
};

void BindLeafSampling(id<MTLComputeCommandEncoder> encoder, QuantizedGradients const* sampling,
                      id<MTLBuffer> fallback) {
  LeafSamplingParams parameters;
  // All slots are bound for Metal validation, including unused scalar/uniform
  // inputs. No row payload or MVS threshold is read by the host here.
  auto norms = fallback, state = fallback;
  if (sampling) {
    auto source = static_cast<MetalQuantizedGradients const*>(sampling);
    parameters.method = source->gradient_sampling_ ? 3 : source->probability_ < 1.f ? 2 : 1;
    parameters.seed = source->seed_;
    parameters.probability = source->probability_;
    if (source->gradient_sampling_) {
      norms = source->norms_;
      state = source->sampling_state_;
      CHECK(norms && state);
    }
  }
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
  [encoder setBuffer:norms offset:0 atIndex:6];
  [encoder setBuffer:state offset:0 atIndex:7];
}

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
      CHECK(storage);
      auto shared = static_cast<MetalHistogramBuffer*>(storage);
      shared->UseQuantization(quantized.gradient_scale_);
      std::vector<std::uint64_t> output_offsets;
      bool direct = page_idx == 0;
      for (auto out : output) {
        auto offset = out.data() - storage->Data().data();
        if (!output_offsets.empty())
          direct = direct && offset == output_offsets.back() + bins;
        output_offsets.push_back(offset);
      }
      // Page zero replaces these slots. Contiguous destinations can receive
      // integer accumulation directly, without a second histogram allocation.
      // Later pages and discontiguous destinations retain the staging path.
      std::size_t histogram_offset = direct ? output_offsets.front() * sizeof(QuantizedPair) : 0;
      if (!direct) EnsureBuffer(rt_, &histogram_buffer_, bytes);
      auto histogram = direct ? shared->integers : histogram_buffer_;
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
      [clear fillBuffer:histogram range:NSMakeRange(histogram_offset, bytes) value:0];
      [clear endEncoding];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      CHECK(encoder) << "Metal compute encoder allocation failed.";
      std::size_t max_feature_bins = 0;
      std::size_t min_feature_bins = std::numeric_limits<std::size_t>::max();
      auto const& cuts = gidx.cut.Ptrs();
      for (std::size_t f = 1; f < cuts.size(); ++f) {
        max_feature_bins = std::max<std::size_t>(max_feature_bins, cuts[f] - cuts[f - 1]);
        min_feature_bins = std::min<std::size_t>(min_feature_bins, cuts[f] - cuts[f - 1]);
      }
      auto shared_bytes = max_feature_bins * sizeof(QuantizedPair);
      // Wide compressed pages have enough feature work to amortize private
      // histograms even when fewer rows remain in the smaller child nodes.
      bool use_shared = gidx.IsDense() &&
                        (total_rows >= 16384 || (page->IsCompressed() &&
                         params.n_features >= 1024 && total_rows >= 4096)) &&
                        shared_bytes <= rt_.device.maxThreadgroupMemoryLength &&
                        params.n_features <= 65535;
      // Keep the register-reduction path for binary features and the existing
      // single-feature path when four histograms would exceed shared memory.
      bool use_tiled = use_shared && page->IsCompressed() && params.n_features >= 4 &&
                       min_feature_bins > 2 &&
                       shared_bytes <= rt_.device.maxThreadgroupMemoryLength / 4;
      auto features_per_group = use_tiled ? 4u : 1u;
      shared_bytes *= features_per_group;
      // Global-bin ELLPACK rows are sorted. A shared bin range amortizes global
      // atomics for large, moderately missing pages. Keep highly sparse, wide
      // and small populations on the existing path: repeated range searches
      // cost more there. These bounds cover the measured 128-column workload.
      constexpr std::size_t kSparseBins = 2048;
      auto const& compressed = page->compressed;
      bool use_sparse = page->IsCompressed() && !compressed.dense &&
                        compressed.stride < params.n_features &&
                        params.n_features >= 128 && params.n_features <= 256 &&
                        compressed.stride >= (params.n_features * 3 + 3) / 4 &&
                        total_rows >= 65536 && bins >= 16384 && bins <= 65536 &&
                        kSparseBins * sizeof(QuantizedPair) <= rt_.device.maxThreadgroupMemoryLength;
      auto pipeline = page->IsCompressed() ? (use_shared ? rt_.ellpack_shared_histogram_pipeline
                                                         : rt_.ellpack_histogram_pipeline)
                                           : (use_shared ? rt_.shared_pipeline : rt_.pipeline);
      if (use_tiled) pipeline = rt_.ellpack_tiled_histogram_pipeline;
      auto groups = (params.n_features + features_per_group - 1) / features_per_group;
      if (use_sparse) {
        use_shared = true;
        shared_bytes = kSparseBins * sizeof(QuantizedPair);
        params.padding = kSparseBins;
        params.rows_per_lane = 32;
        groups = (bins + kSparseBins - 1) / kSparseBins;
        pipeline = rt_.ellpack_sparse_histogram_pipeline;
      }
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:page->index offset:0 atIndex:0];
      [encoder setBuffer:page->row_ptr offset:0 atIndex:1];
      [encoder setBuffer:page->offsets offset:0 atIndex:2];
      [encoder setBuffer:quantized.gradients_ offset:0 atIndex:3];
      [encoder setBuffer:row_buffer offset:0 atIndex:4];
      [encoder setBuffer:histogram offset:histogram_offset atIndex:5];
      [encoder setBytes:&params length:sizeof(params) atIndex:6];
      auto threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
      [encoder setBuffer:node_buffer offset:0 atIndex:7];
      [encoder setBuffer:row_begin_buffer offset:0 atIndex:8];
      if (page->IsCompressed()) {
        auto slot = use_shared ? 10u : 9u;
        [encoder setBuffer:page->tail offset:0 atIndex:slot];
        [encoder setBytes:&page->compressed length:sizeof(page->compressed) atIndex:slot + 1];
      }
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
        [encoder dispatchThreadgroups:MTLSizeMake(block_ptr.back(), groups, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      } else if (total_rows) {
        [encoder dispatchThreads:MTLSizeMake(total_rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      }
      [encoder endEncoding];
      if (direct) {
        EncodeIntegerStatistics(rt_, command, shared->integers, shared->buffer,
                                quantized.gradient_scale_, nodes.size() * bins, 1,
                                histogram_offset, histogram_offset);
      } else {
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
      }
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
      if (page->IsCompressed() && total_rows)
        ellpack_histogram_dispatches.fetch_add(1, std::memory_order_relaxed);
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
    linalg::MatrixView<GradientPair const> original, Context const* ctx) {
  @autoreleasepool {
    return std::make_shared<MetalGradientQuantizer>(original, ctx);
  }
}

std::shared_ptr<QuantizedGradients> MakeQuantizedGradients(
    linalg::MatrixView<GradientPair const> gradients, Context const* ctx) {
  @autoreleasepool {
    return std::make_shared<MetalQuantizedGradients>(gradients, ctx);
  }
}

std::unique_ptr<SplitWorkspace> MakeSplitWorkspace() {
  return std::make_unique<MetalSplitWorkspace>();
}
std::unique_ptr<HistogramBuffer> MakeHistogramBuffer() {
  return std::make_unique<MetalHistogramBuffer>();
}
void AllreduceHistograms(Context const* ctx, HistogramBuffer* storage, std::size_t offset,
                         std::size_t bins) {
  CHECK(ctx && storage);
  CHECK_LE(offset, storage->Data().size());
  CHECK_LE(bins, storage->Data().size() - offset);
  if (!bins) return;
  auto shared = static_cast<MetalHistogramBuffer*>(storage);
  CHECK(shared->integers && shared->scale) << "Metal histogram collectives require integer statistics.";
  if (!collective::GlobalCommGroup()->IsDistributed()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto first = static_cast<std::int64_t*>(shared->integers.contents) + offset * 2;
    auto view = linalg::MakeVec(first, bins * 2, DeviceOrd::MPS());
    collective::SafeColl(collective::GlobalSum(ctx, view));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    EncodeIntegerStatistics(rt, command, shared->integers, shared->buffer, shared->scale, bins, 1,
                            offset * sizeof(GradientPairInt64), offset * sizeof(GradientPairPrecise));
    CompleteOperation(command, &histogram_conversion_dispatches, &gpu_time_ns);
  }
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
          metric_reduction_passes.load(std::memory_order_relaxed),
          metric_final_readback_bytes.load(std::memory_order_relaxed),
          metric_label_dispatches.load(std::memory_order_relaxed),
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
          ranking_plan_dispatches.load(std::memory_order_relaxed),
          ranking_plan_reuses.load(std::memory_order_relaxed),
          ranking_plan_gpu_time_ns.load(std::memory_order_relaxed),
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
          cuda_sketch_gpu_time_ns.load(std::memory_order_relaxed),
          small_histogram_dispatches.load(std::memory_order_relaxed),
          vector_arithmetic_dispatches.load(std::memory_order_relaxed),
          float_mean_dispatches.load(std::memory_order_relaxed),
          category_sort_dispatches.load(std::memory_order_relaxed),
          category_recode_dispatches.load(std::memory_order_relaxed),
          category_gpu_time_ns.load(std::memory_order_relaxed),
          input_prepare_dispatches.load(std::memory_order_relaxed),
          input_recode_dispatches.load(std::memory_order_relaxed),
          input_prepare_gpu_time_ns.load(std::memory_order_relaxed),
          input_debug_readback_bytes.load(std::memory_order_relaxed),
          adapter_prepare_dispatches.load(std::memory_order_relaxed),
          adapter_index_dispatches.load(std::memory_order_relaxed),
          adapter_gpu_time_ns.load(std::memory_order_relaxed),
          adapter_debug_readback_bytes.load(std::memory_order_relaxed),
          ellpack_pack_dispatches.load(std::memory_order_relaxed),
          ellpack_copy_dispatches.load(std::memory_order_relaxed),
          ellpack_restore_dispatches.load(std::memory_order_relaxed),
          ellpack_cache_dispatches.load(std::memory_order_relaxed),
          ellpack_gpu_time_ns.load(std::memory_order_relaxed),
          ellpack_debug_readback_bytes.load(std::memory_order_relaxed),
          ellpack_cached_host_bytes.load(std::memory_order_relaxed),
          ellpack_cached_device_bytes.load(std::memory_order_relaxed),
          ellpack_adapter_dispatches.load(std::memory_order_relaxed),
          ellpack_adapter_values.load(std::memory_order_relaxed),
          ellpack_pack_input_bytes.load(std::memory_order_relaxed),
          ellpack_histogram_dispatches.load(std::memory_order_relaxed),
          ellpack_partition_dispatches.load(std::memory_order_relaxed),
          ellpack_input_dispatches.load(std::memory_order_relaxed),
          ellpack_view_pages.load(std::memory_order_relaxed),
          ellpack_count_dispatches.load(std::memory_order_relaxed),
          ellpack_host_materializations.load(std::memory_order_relaxed),
          ellpack_host_readback_bytes.load(std::memory_order_relaxed),
          ellpack_csr_dispatches.load(std::memory_order_relaxed),
          ellpack_csr_values.load(std::memory_order_relaxed),
          ellpack_csr_input_bytes.load(std::memory_order_relaxed),
          ellpack_io_read_pages.load(std::memory_order_relaxed),
          ellpack_io_write_pages.load(std::memory_order_relaxed),
          ellpack_io_read_bytes.load(std::memory_order_relaxed),
          ellpack_io_write_bytes.load(std::memory_order_relaxed),
          ellpack_io_dispatches.load(std::memory_order_relaxed),
          snappy_compress_dispatches.load(std::memory_order_relaxed),
          snappy_decompress_dispatches.load(std::memory_order_relaxed),
          snappy_copy_dispatches.load(std::memory_order_relaxed),
          snappy_input_bytes.load(std::memory_order_relaxed),
          snappy_compressed_bytes.load(std::memory_order_relaxed),
          snappy_decoded_bytes.load(std::memory_order_relaxed),
          snappy_host_readback_bytes.load(std::memory_order_relaxed),
          snappy_gpu_time_ns.load(std::memory_order_relaxed),
          tensor_imports.load(std::memory_order_relaxed),
          tensor_copy_dispatches.load(std::memory_order_relaxed),
          tensor_copied_bytes.load(std::memory_order_relaxed),
          tensor_adapter_windows.load(std::memory_order_relaxed),
          tensor_prediction_windows.load(std::memory_order_relaxed),
          tensor_gpu_time_ns.load(std::memory_order_relaxed),
          input_staging_bytes.load(std::memory_order_relaxed),
          metadata_copy_dispatches.load(std::memory_order_relaxed),
          metadata_convert_dispatches.load(std::memory_order_relaxed),
          metadata_validate_dispatches.load(std::memory_order_relaxed),
          metadata_qid_dispatches.load(std::memory_order_relaxed),
          metadata_values.load(std::memory_order_relaxed),
          metadata_shared_output_bytes.load(std::memory_order_relaxed),
          metadata_host_copy_bytes.load(std::memory_order_relaxed),
          metadata_gpu_time_ns.load(std::memory_order_relaxed),
          gradient_import_dispatches.load(std::memory_order_relaxed),
          gradient_import_pairs.load(std::memory_order_relaxed),
          gradient_import_shared_output_bytes.load(std::memory_order_relaxed),
          gradient_import_host_copy_bytes.load(std::memory_order_relaxed),
          gradient_import_gpu_time_ns.load(std::memory_order_relaxed),
          tensor_csr_dispatches.load(std::memory_order_relaxed),
          tensor_csr_values.load(std::memory_order_relaxed),
          tensor_csr_shared_output_bytes.load(std::memory_order_relaxed),
          tensor_csr_host_copy_bytes.load(std::memory_order_relaxed),
          tensor_csr_gpu_time_ns.load(std::memory_order_relaxed),
          csr_shared_input_bytes.load(std::memory_order_relaxed),
          csr_input_copy_bytes.load(std::memory_order_relaxed),
          columnar_imports.load(std::memory_order_relaxed),
          columnar_pack_bytes.load(std::memory_order_relaxed),
          columnar_gpu_time_ns.load(std::memory_order_relaxed),
          category_import_dispatches.load(std::memory_order_relaxed),
          category_import_shared_bytes.load(std::memory_order_relaxed),
          category_import_copy_bytes.load(std::memory_order_relaxed),
          prediction_shared_output_bytes.load(std::memory_order_relaxed),
          prediction_host_input_bytes.load(std::memory_order_relaxed),
          prediction_host_output_bytes.load(std::memory_order_relaxed),
          prediction_device_copy_bytes.load(std::memory_order_relaxed),
          output_snapshot_dispatches.load(std::memory_order_relaxed),
          output_snapshot_shared_input_bytes.load(std::memory_order_relaxed),
          output_snapshot_host_copy_bytes.load(std::memory_order_relaxed),
          output_snapshot_bytes.load(std::memory_order_relaxed),
          output_gpu_time_ns.load(std::memory_order_relaxed),
          output_host_readback_bytes.load(std::memory_order_relaxed),
          output_dlpack_exports.load(std::memory_order_relaxed),
          output_live_buffers.load(std::memory_order_relaxed),
          output_live_exports.load(std::memory_order_relaxed),
          vector_allocations.load(std::memory_order_relaxed),
          vector_live_buffers.load(std::memory_order_relaxed),
          vector_capacity_bytes.load(std::memory_order_relaxed),
          vector_copy_dispatches.load(std::memory_order_relaxed),
          vector_copy_bytes.load(std::memory_order_relaxed),
          vector_fill_dispatches.load(std::memory_order_relaxed),
          vector_fill_bytes.load(std::memory_order_relaxed),
          vector_upload_bytes.load(std::memory_order_relaxed),
          vector_readback_bytes.load(std::memory_order_relaxed),
          vector_host_copy_bytes.load(std::memory_order_relaxed),
          vector_gpu_time_ns.load(std::memory_order_relaxed),
          metadata_gather_dispatches.load(std::memory_order_relaxed),
          metadata_gather_values.load(std::memory_order_relaxed),
          metadata_equal_dispatches.load(std::memory_order_relaxed),
          initialization_shared_input_bytes.load(std::memory_order_relaxed),
          initialization_shared_output_bytes.load(std::memory_order_relaxed),
          initialization_host_input_bytes.load(std::memory_order_relaxed),
          initialization_host_output_bytes.load(std::memory_order_relaxed),
          initialization_copy_dispatches.load(std::memory_order_relaxed),
          initialization_copy_bytes.load(std::memory_order_relaxed),
          initialization_fill_dispatches.load(std::memory_order_relaxed),
          metric_shared_input_bytes.load(std::memory_order_relaxed),
          metric_host_input_bytes.load(std::memory_order_relaxed),
          metric_shared_output_bytes.load(std::memory_order_relaxed),
          metric_host_output_bytes.load(std::memory_order_relaxed),
          metric_copy_dispatches.load(std::memory_order_relaxed),
          metric_copy_bytes.load(std::memory_order_relaxed),
          auc_final_readback_bytes.load(std::memory_order_relaxed),
          category_state_shared_input_bytes.load(std::memory_order_relaxed),
          category_state_shared_output_bytes.load(std::memory_order_relaxed),
          category_state_staging_bytes.load(std::memory_order_relaxed),
          prediction_model_pack_dispatches.load(std::memory_order_relaxed),
          prediction_model_shared_input_bytes.load(std::memory_order_relaxed),
          prediction_model_staging_bytes.load(std::memory_order_relaxed),
          split_weight_store_dispatches.load(std::memory_order_relaxed),
          split_weight_readback_bytes.load(std::memory_order_relaxed),
          model_host_copy_bytes.load(std::memory_order_relaxed),
          model_snapshot_bytes.load(std::memory_order_relaxed),
          prediction_shared_input_bytes.load(std::memory_order_relaxed),
          split_category_store_dispatches.load(std::memory_order_relaxed),
          split_category_readback_bytes.load(std::memory_order_relaxed),
          partition_category_shared_input_bytes.load(std::memory_order_relaxed),
          partition_category_host_copy_bytes.load(std::memory_order_relaxed),
          gradient_sum_shared_output_bytes.load(std::memory_order_relaxed),
          gradient_sum_readback_bytes.load(std::memory_order_relaxed),
          split_sum_store_dispatches.load(std::memory_order_relaxed),
          split_sum_gather_dispatches.load(std::memory_order_relaxed),
          split_sum_apply_dispatches.load(std::memory_order_relaxed),
          split_sum_shared_input_bytes.load(std::memory_order_relaxed),
          split_sum_host_input_bytes.load(std::memory_order_relaxed),
          split_sum_readback_bytes.load(std::memory_order_relaxed),
          feature_shared_input_bytes.load(std::memory_order_relaxed),
          feature_shared_output_bytes.load(std::memory_order_relaxed),
          feature_host_input_bytes.load(std::memory_order_relaxed),
          feature_readback_bytes.load(std::memory_order_relaxed),
          feature_task_dispatches.load(std::memory_order_relaxed),
          feature_task_candidates.load(std::memory_order_relaxed),
          split_bounds_gather_dispatches.load(std::memory_order_relaxed),
          split_bounds_gather_values.load(std::memory_order_relaxed),
          split_bounds_shared_input_bytes.load(std::memory_order_relaxed),
          split_bounds_host_copy_bytes.load(std::memory_order_relaxed)};
}

namespace {
void CompleteOperation(id<MTLCommandBuffer> command, std::atomic<std::uint64_t>* counter,
                       std::atomic<std::uint64_t>* duration = &objective_gpu_time_ns) {
  [command commit];
  [command waitUntilCompleted];
  CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
  if (counter) counter->fetch_add(1, std::memory_order_relaxed);
  auto elapsed = command.GPUEndTime - command.GPUStartTime;
  if (elapsed > 0)
    duration->fetch_add(static_cast<std::uint64_t>(elapsed * 1e9), std::memory_order_relaxed);
}
}  // namespace
namespace {
class MetalVectorBuffer final : public VectorBuffer {
  id<MTLBuffer> buffer_{nil};
  std::size_t size_{0}, capacity_{0};

  void Blit(id<MTLBuffer> source, std::size_t source_offset, id<MTLBuffer> target,
            std::size_t target_offset, std::size_t bytes) const {
    if (!bytes) return;
    auto command = [GetRuntime().queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    [blit copyFromBuffer:source sourceOffset:source_offset toBuffer:target
      destinationOffset:target_offset size:bytes];
    [blit endEncoding];
    CompleteOperation(command, &vector_copy_dispatches, &vector_gpu_time_ns);
    vector_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }

 public:
  ~MetalVectorBuffer() override {
    if (buffer_) {
      vector_live_buffers.fetch_sub(1, std::memory_order_relaxed);
      vector_capacity_bytes.fetch_sub(capacity_, std::memory_order_relaxed);
    }
  }
  id<MTLBuffer> Buffer() const { return buffer_; }
  void* Data() const override { return buffer_ ? buffer_.contents : nullptr; }
  std::size_t SizeBytes() const override { return size_; }
  void Resize(std::size_t bytes) override {
    if (bytes > capacity_) {
      @autoreleasepool {
        auto& rt = GetRuntime();
        CHECK_LE(bytes, rt.device.maxBufferLength);
        auto capacity = std::max<std::size_t>(16, bytes);
        auto replacement = [rt.device newBufferWithLength:capacity
                                                 options:MTLResourceStorageModeShared];
        CHECK(replacement);
        Blit(buffer_, 0, replacement, 0, size_);
        if (!buffer_) vector_live_buffers.fetch_add(1, std::memory_order_relaxed);
        vector_allocations.fetch_add(1, std::memory_order_relaxed);
        vector_capacity_bytes.fetch_add(capacity - capacity_, std::memory_order_relaxed);
        buffer_ = replacement;
        capacity_ = capacity;
      }
    }
    size_ = bytes;
  }
  void Fill(Span<std::byte const> pattern, std::size_t offset, std::size_t bytes) override {
    CHECK(!pattern.empty() && pattern.size() <= 4096);
    CHECK_LE(offset, size_);
    CHECK_LE(bytes, size_ - offset);
    CHECK_EQ(bytes % pattern.size(), 0);
    if (!bytes) return;
    @autoreleasepool {
      auto& rt = GetRuntime();
      auto command = [rt.queue commandBuffer];
      auto e = [command computeCommandEncoder];
      struct { std::uint64_t offset, bytes, width; } p{offset, bytes, pattern.size()};
      [e setComputePipelineState:rt.vector_fill_pipeline];
      [e setBuffer:buffer_ offset:0 atIndex:0];
      [e setBytes:pattern.data() length:pattern.size() atIndex:1];
      [e setBytes:&p length:sizeof(p) atIndex:2];
      // Bounded grid, with a strided byte loop for buffers above 2^32 bytes.
      auto threads = std::min<std::size_t>(bytes, 1 << 20);
      [e dispatchThreads:MTLSizeMake(threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &vector_fill_dispatches, &vector_gpu_time_ns);
      vector_fill_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
  }
  void Copy(VectorBuffer const& source, std::size_t source_offset, std::size_t offset,
            std::size_t bytes) override {
    auto const* input = dynamic_cast<MetalVectorBuffer const*>(&source);
    CHECK(input);
    CHECK_LE(source_offset, input->size_);
    CHECK_LE(bytes, input->size_ - source_offset);
    CHECK_LE(offset, size_);
    CHECK_LE(bytes, size_ - offset);
    if (!bytes || (input == this && source_offset == offset)) return;
    @autoreleasepool {
      if (input == this && (source_offset > offset ? source_offset - offset : offset - source_offset) < bytes) {
        auto temporary = Allocate(GetRuntime(), bytes);
        Blit(input->buffer_, source_offset, temporary, 0, bytes);
        Blit(temporary, 0, buffer_, offset, bytes);
      } else {
        Blit(input->buffer_, source_offset, buffer_, offset, bytes);
      }
    }
  }
  void Import(void const* source, std::size_t offset, std::size_t bytes) override {
    CHECK_LE(offset, size_);
    CHECK_LE(bytes, size_ - offset);
    CHECK(source || !bytes);
    if (!bytes) return;
    @autoreleasepool {
      auto& rt = GetRuntime();
      auto in = TryBorrowMemory(rt.device, source, bytes, 1);
      if (!in.buffer || MemoryRangesOverlap(source, bytes, Data(), size_)) {
        in = {Upload(rt, source, bytes), 0};
        vector_host_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
      }
      Blit(in.buffer, in.offset, buffer_, offset, bytes);
      vector_upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
  }
  void Readback(void* destination, std::size_t bytes) const override {
    CHECK_EQ(bytes, size_);
    CHECK(destination || !bytes);
    if (!bytes) return;
    @autoreleasepool {
      auto& rt = GetRuntime();
      auto out = TryBorrowMemory(rt.device, destination, bytes, 1);
      bool shared = out.buffer != nil && !MemoryRangesOverlap(destination, bytes, Data(), size_);
      if (!shared) out = {Allocate(rt, bytes), 0};
      Blit(buffer_, 0, out.buffer, out.offset, bytes);
      if (!shared) {
        std::memcpy(destination, out.buffer.contents, bytes);
        vector_host_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
      }
      vector_readback_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
  }
};

BorrowedMemory VectorInput(Runtime const& rt, void const* data, std::size_t bytes,
                           std::size_t alignment) {
  auto input = TryBorrowMemory(rt.device, data, bytes, alignment);
  if (!input.buffer) {
    input = {Upload(rt, data, bytes), 0};
    vector_host_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}
}  // namespace

std::unique_ptr<VectorBuffer> MakeVectorBuffer() { return std::make_unique<MetalVectorBuffer>(); }

namespace {
std::atomic<std::uint64_t> collective_reductions{0}, collective_copies{0},
    collective_input{0}, collective_output{0}, collective_staging{0}, collective_time{0};

BorrowedMemory CollectiveInput(Runtime const& rt, Span<std::int8_t const> input) {
  auto view = TryBorrowMemory(rt.device, input.data(), input.size(), 1);
  if (view.buffer) {
    collective_input.fetch_add(input.size(), std::memory_order_relaxed);
  } else {
    view = {Upload(rt, input.data(), input.size()), 0};
    collective_staging.fetch_add(input.size(), std::memory_order_relaxed);
  }
  return view;
}
}  // namespace

CollectiveStatistics GetCollectiveStatistics() {
  return {collective_reductions.load(), collective_copies.load(), collective_input.load(),
          collective_output.load(), collective_staging.load(), collective_time.load()};
}

void CollectiveCopy(Span<std::int8_t const> input, Span<std::int8_t> output) {
  CHECK_EQ(input.size(), output.size());
  if (input.empty() || input.data() == output.data()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto in = CollectiveInput(rt, input);
    auto out = TryBorrowMemory(rt.device, output.data(), output.size(), 1);
    bool shared = out.buffer != nil;
    if (!shared) out = {Allocate(rt, output.size()), 0};
    auto command = [rt.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    // Snapshot overlapping views on the GPU before writing the destination.
    id<MTLBuffer> temporary = nil;
    if (MemoryRangesOverlap(input.data(), input.size(), output.data(), output.size())) {
      temporary = Allocate(rt, input.size());
      [blit copyFromBuffer:in.buffer sourceOffset:in.offset toBuffer:temporary
        destinationOffset:0 size:input.size()];
      in = {temporary, 0};
    }
    [blit copyFromBuffer:in.buffer sourceOffset:in.offset toBuffer:out.buffer
      destinationOffset:out.offset size:input.size()];
    [blit endEncoding];
    CompleteOperation(command, &collective_copies, &collective_time);
    if (shared) {
      collective_output.fetch_add(output.size(), std::memory_order_relaxed);
    } else {
      std::memcpy(output.data(), out.buffer.contents, output.size());
      collective_staging.fetch_add(output.size(), std::memory_order_relaxed);
    }
  }
}

void CollectiveReduce(Span<std::int8_t const> input, Span<std::int8_t> output,
                      ArrayInterfaceHandler::Type type, collective::Op op,
                      std::size_t ranks, bool initialize) {
  CHECK_GT(ranks, 0);
  CHECK_EQ(input.size() / ranks, output.size());
  CHECK_EQ(input.size() % ranks, 0);
  auto operation = static_cast<std::uint32_t>(op);
  CHECK_LE(operation, 5);
  std::uint32_t width = 0;
  using H = ArrayInterfaceHandler;
  if (operation >= 3) {
    width = 1;
    type = H::kU1;
  } else {
    switch (type) {
      case H::kF2: width = 2; break;
      case H::kF4: case H::kI4: case H::kU4: width = 4; break;
      case H::kF8: case H::kI8: case H::kU8: width = 8; break;
      case H::kI1: case H::kU1: width = 1; break;
      default: LOG(FATAL) << "Unsupported Metal collective numeric type.";
    }
  }
  CHECK_EQ(output.size() % width, 0);
  CHECK(!MemoryRangesOverlap(input.data(), input.size(), output.data(), output.size()));
  if (output.empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto in = CollectiveInput(rt, input);
    auto out = TryBorrowMemory(rt.device, output.data(), output.size(), 1);
    bool shared = out.buffer != nil;
    if (!shared) {
      out = {initialize ? Allocate(rt, output.size()) : Upload(rt, output.data(), output.size()), 0};
      if (!initialize) collective_staging.fetch_add(output.size(), std::memory_order_relaxed);
    }
    struct {
      std::uint64_t count, ranks;
      std::uint32_t type, operation, width, initialize;
    } p{output.size() / width, ranks, static_cast<std::uint32_t>(type), operation, width,
        static_cast<std::uint32_t>(initialize)};
    auto command = [rt.queue commandBuffer];
    auto e = [command computeCommandEncoder];
    [e setComputePipelineState:rt.collective_reduce_pipeline];
    [e setBuffer:in.buffer offset:in.offset atIndex:0];
    [e setBuffer:out.buffer offset:out.offset atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(std::min<std::size_t>(p.count, 1 << 20), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &collective_reductions, &collective_time);
    if (shared) {
      collective_output.fetch_add(output.size(), std::memory_order_relaxed);
    } else {
      std::memcpy(output.data(), out.buffer.contents, output.size());
      collective_staging.fetch_add(output.size(), std::memory_order_relaxed);
    }
  }
}

bool EqualBytes(Span<std::byte const> lhs, Span<std::byte const> rhs) {
  if (lhs.size() != rhs.size()) return false;
  if (lhs.empty()) return true;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto left = VectorInput(rt, lhs.data(), lhs.size(), 1);
    auto right = VectorInput(rt, rhs.data(), rhs.size(), 1);
    std::uint32_t zero = 0;
    auto status = Upload(rt, &zero, 4);
    auto command = [rt.queue commandBuffer];
    auto e = [command computeCommandEncoder];
    std::uint64_t n = lhs.size();
    [e setComputePipelineState:rt.vector_equal_pipeline];
    [e setBuffer:left.buffer offset:left.offset atIndex:0];
    [e setBuffer:right.buffer offset:right.offset atIndex:1];
    [e setBuffer:status offset:0 atIndex:2];
    [e setBytes:&n length:sizeof(n) atIndex:3];
    [e dispatchThreads:MTLSizeMake(std::min<std::size_t>(n, 1 << 20), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &metadata_equal_dispatches, &vector_gpu_time_ns);
    return *static_cast<std::uint32_t const*>(status.contents) == 0;
  }
}

void GatherRows(Span<float const> input, std::size_t rows, std::size_t columns,
                Span<std::uint64_t const> indices, Span<float> output) {
  CHECK(columns && rows <= std::numeric_limits<std::size_t>::max() / columns);
  CHECK_EQ(input.size(), rows * columns);
  CHECK_LE(indices.size(), std::numeric_limits<std::size_t>::max() / columns);
  CHECK_EQ(output.size(), indices.size() * columns);
  if (output.empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto in = VectorInput(rt, input.data(), input.size_bytes(), 4);
    auto ids = VectorInput(rt, indices.data(), indices.size_bytes(), 8);
    auto out = TryBorrowMemory(rt.device, output.data(), output.size_bytes(), 4);
    bool shared = out.buffer != nil;
    if (!shared) out = {Allocate(rt, output.size_bytes()), 0};
    std::uint32_t zero = 0;
    auto status = Upload(rt, &zero, 4);
    auto command = [rt.queue commandBuffer];
    auto snapshot = [&](BorrowedMemory source, std::size_t bytes) {
      auto buffer = Allocate(rt, bytes);
      auto blit = [command blitCommandEncoder];
      [blit copyFromBuffer:source.buffer sourceOffset:source.offset toBuffer:buffer
        destinationOffset:0 size:bytes];
      [blit endEncoding];
      vector_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
      vector_copy_dispatches.fetch_add(1, std::memory_order_relaxed);
      return BorrowedMemory{buffer, 0};
    };
    if (MemoryRangesOverlap(input.data(), input.size_bytes(), output.data(), output.size_bytes()))
      in = snapshot(in, input.size_bytes());
    if (MemoryRangesOverlap(indices.data(), indices.size_bytes(), output.data(), output.size_bytes()))
      ids = snapshot(ids, indices.size_bytes());
    auto e = [command computeCommandEncoder];
    struct { std::uint64_t rows, columns, count; } p{rows, columns, output.size()};
    [e setComputePipelineState:rt.metadata_gather_pipeline];
    [e setBuffer:in.buffer offset:in.offset atIndex:0];
    [e setBuffer:ids.buffer offset:ids.offset atIndex:1];
    [e setBuffer:out.buffer offset:out.offset atIndex:2];
    [e setBuffer:status offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(std::min<std::size_t>(output.size(), 1 << 20), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &metadata_gather_dispatches, &vector_gpu_time_ns);
    CHECK_EQ(*static_cast<std::uint32_t const*>(status.contents), 0) << "Metadata row index out of range.";
    if (!shared) {
      std::memcpy(output.data(), out.buffer.contents, output.size_bytes());
      vector_host_copy_bytes.fetch_add(output.size_bytes(), std::memory_order_relaxed);
    }
    metadata_gather_values.fetch_add(output.size(), std::memory_order_relaxed);
  }
}

namespace {
BorrowedMemory PredictionInput(Runtime const& rt, void const* data, std::size_t bytes) {
  auto input = TryBorrowMemory(rt.device, data, bytes, 4);
  if (input.buffer) {
    prediction_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    input = {Upload(rt, data, bytes), 0};
    prediction_host_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}
BorrowedMemory PredictionSnapshot(Runtime const& rt, id<MTLCommandBuffer> command,
                                     BorrowedMemory input, std::size_t bytes) {
  if (!bytes) return input;
  auto snapshot = Allocate(rt, bytes);
  auto blit = [command blitCommandEncoder];
  CHECK(blit);
  [blit copyFromBuffer:input.buffer sourceOffset:input.offset toBuffer:snapshot
    destinationOffset:0 size:bytes];
  [blit endEncoding];
  prediction_device_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
  return {snapshot, 0};
}
struct PredictionOutput {
  Span<float> host;
  BorrowedMemory memory;
  bool shared;
  PredictionOutput(Runtime const& rt, Span<float> values, bool initialize) : host{values} {
    memory = TryBorrowMemory(rt.device, values.data(), values.size_bytes(), alignof(float));
    shared = memory.buffer != nil;
    if (shared) {
      prediction_shared_output_bytes.fetch_add(values.size_bytes(), std::memory_order_relaxed);
    } else if (initialize) {
      memory = {Upload(rt, values.data(), values.size_bytes()), 0};
      prediction_host_input_bytes.fetch_add(values.size_bytes(), std::memory_order_relaxed);
    } else {
      memory = {Allocate(rt, values.size_bytes()), 0};
    }
  }
  void Publish(std::size_t count) {
    CHECK_LE(count, host.size());
    if (!shared && count) {
      std::memcpy(host.data(), memory.buffer.contents, count * sizeof(float));
      prediction_host_output_bytes.fetch_add(count * sizeof(float), std::memory_order_relaxed);
    }
  }
};

class MetalOutput final : public Output {
  id<MTLBuffer> buffer_;
  std::vector<std::int64_t> shape_, strides_;
  std::size_t size_{1};

 public:
  explicit MetalOutput(Span<bst_ulong const> shape) {
    CHECK_LE(shape.size(), std::numeric_limits<std::int32_t>::max());
    shape_.resize(shape.size());
    strides_.resize(shape.size());
    std::uint64_t stride{1};
    for (std::size_t i = shape.size(); i != 0; --i) {
      CHECK_LE(shape[i - 1], std::numeric_limits<std::int64_t>::max());
      CHECK_LE(stride, std::numeric_limits<std::int64_t>::max());
      shape_[i - 1] = shape[i - 1];
      strides_[i - 1] = stride;
      CHECK(!shape[i - 1] || stride <= std::numeric_limits<std::int64_t>::max() / shape[i - 1]);
      stride *= shape[i - 1];
    }
    size_ = stride;
    auto& rt = GetRuntime();
    CHECK_LE(size_, rt.device.maxBufferLength / sizeof(float));
    buffer_ = [rt.device newBufferWithLength:std::max<std::size_t>(16, size_ * sizeof(float))
                                    options:MTLResourceStorageModePrivate];
    CHECK(buffer_);
    output_live_buffers.fetch_add(1, std::memory_order_relaxed);
  }
  ~MetalOutput() override { output_live_buffers.fetch_sub(1, std::memory_order_relaxed); }
  std::size_t Size() const override { return size_; }
  DLTensor View() const override {
    return {(__bridge void*)buffer_, {kDLMetal, 0}, static_cast<std::int32_t>(shape_.size()),
            {kDLFloat, 32, 1}, const_cast<std::int64_t*>(shape_.data()),
            const_cast<std::int64_t*>(strides_.data()), 0};
  }
  void CopyFrom(BorrowedMemory source) {
    if (!size_) return;
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    [blit copyFromBuffer:source.buffer sourceOffset:source.offset toBuffer:buffer_
      destinationOffset:0 size:size_ * sizeof(float)];
    [blit endEncoding];
    CompleteOperation(command, &output_snapshot_dispatches, &output_gpu_time_ns);
    output_snapshot_bytes.fetch_add(size_ * sizeof(float), std::memory_order_relaxed);
  }
  void Readback(Span<float> output) const override {
    CHECK_EQ(output.size(), size_);
    if (!size_) return;
    @autoreleasepool {
      auto& rt = GetRuntime();
      auto target = TryBorrowMemory(rt.device, output.data(), output.size_bytes(), alignof(float));
      bool shared = target.buffer != nil;
      if (!shared) target = {Allocate(rt, output.size_bytes()), 0};
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      [blit copyFromBuffer:buffer_ sourceOffset:0 toBuffer:target.buffer
        destinationOffset:target.offset size:output.size_bytes()];
      [blit endEncoding];
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      if (!shared) std::memcpy(output.data(), target.buffer.contents, output.size_bytes());
      output_host_readback_bytes.fetch_add(output.size_bytes(), std::memory_order_relaxed);
    }
  }
  std::shared_ptr<Output> Clone() const override {
    @autoreleasepool {
      std::vector<bst_ulong> shape(shape_.begin(), shape_.end());
      auto result = std::make_shared<MetalOutput>(Span<bst_ulong const>{shape});
      result->CopyFrom({buffer_, 0});
      return result;
    }
  }
};

template <typename Managed>
struct OutputExport {
  Managed tensor{};
  std::shared_ptr<Output> owner;
  explicit OutputExport(std::shared_ptr<Output> value, bool copied) : owner{std::move(value)} {
    tensor.dl_tensor = owner->View();
    tensor.manager_ctx = this;
    tensor.deleter = [](Managed* ptr) { delete static_cast<OutputExport*>(ptr->manager_ctx); };
    if constexpr (std::is_same_v<Managed, DLManagedTensorVersioned>) {
      tensor.version = {1, 0};
      tensor.flags = copied ? DLPACK_FLAG_BITMASK_IS_COPIED : 0;
    }
    output_live_exports.fetch_add(1, std::memory_order_relaxed);
    output_dlpack_exports.fetch_add(1, std::memory_order_relaxed);
  }
  ~OutputExport() { output_live_exports.fetch_sub(1, std::memory_order_relaxed); }
};

// The destructor runs under CPython's GIL and never calls back into Python code.
// It therefore survives Python module/global teardown and unconsumed capsules.
// The Metal library has no link-time dependency on a particular Python version.
struct CapsuleAPI {
  int (*valid)(void*, char const*);
  void* (*pointer)(void*, char const*);
};
CapsuleAPI const& PythonCapsuleAPI() {
  static CapsuleAPI const api{
      reinterpret_cast<decltype(CapsuleAPI::valid)>(dlsym(RTLD_DEFAULT, "PyCapsule_IsValid")),
      reinterpret_cast<decltype(CapsuleAPI::pointer)>(dlsym(RTLD_DEFAULT, "PyCapsule_GetPointer"))};
  return api;
}
void DropOutputCapsule(void* capsule) noexcept {
  auto const& api = PythonCapsuleAPI();
  if (api.valid(capsule, "dltensor")) {
    auto ptr = static_cast<DLManagedTensor*>(api.pointer(capsule, "dltensor"));
    if (ptr && ptr->deleter) ptr->deleter(ptr);
  } else if (api.valid(capsule, "dltensor_versioned")) {
    auto ptr = static_cast<DLManagedTensorVersioned*>(api.pointer(capsule, "dltensor_versioned"));
    if (ptr && ptr->deleter) ptr->deleter(ptr);
  }
  // used_dltensor[_versioned] is owned by the consumer, not the capsule.
}
}  // namespace

std::shared_ptr<Output> MakeOutput(float const* values, Span<bst_ulong const> shape) {
  @autoreleasepool {
    auto result = std::make_shared<MetalOutput>(shape);
    auto bytes = result->Size() * sizeof(float);
    CHECK(values || !bytes);
    if (bytes) {
      auto& rt = GetRuntime();
      auto source = TryBorrowMemory(rt.device, values, bytes, alignof(float));
      if (source.buffer) {
        output_snapshot_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
      } else {
        source = {Upload(rt, values, bytes), 0};
        output_snapshot_host_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
      }
      result->CopyFrom(source);
    }
    return result;
  }
}
void* ExportOutput(std::shared_ptr<Output> output, bool versioned, bool copy) {
  CHECK(output);
  if (copy) output = output->Clone();
  if (versioned) return &(new OutputExport<DLManagedTensorVersioned>(std::move(output), copy))->tensor;
  return &(new OutputExport<DLManagedTensor>(std::move(output), copy))->tensor;
}
void* OutputCapsuleDestructor() {
  auto const& api = PythonCapsuleAPI();
  CHECK(api.valid && api.pointer) << "CPython capsule API is not available in this process.";
  return reinterpret_cast<void*>(&DropOutputCapsule);
}

namespace {
struct CategoryColumn {
  std::uint64_t values{0}, offsets{0};
  std::uint32_t begin{0}, size{0}, type{0}, empty{0};
};
static_assert(sizeof(CategoryColumn) == 32);
struct CategoryParams {
  std::uint32_t count, features, width, padding;
};
struct CategoryError {
  void Error(std::string&& message) const { LOG(FATAL) << message; }
};
using CategoryPolicy = enc::Policy<CategoryError>;
BorrowedMemory BindCategoryInput(Runtime const& rt, void const* data, std::size_t bytes,
                                  std::size_t alignment) {
  auto input = TryBorrowMemory(rt.device, data, bytes, alignment);
  if (input.buffer) {
    category_state_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    input = {Upload(rt, data, bytes), 0};
    category_state_staging_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}
void PublishCategoryOutput(Runtime const& rt, id<MTLBuffer> source, Span<std::int32_t> output) {
  if (output.empty()) return;
  auto destination = TryBorrowMemory(rt.device, output.data(), output.size_bytes(), 4);
  if (destination.buffer) {
    auto command = [rt.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    CHECK(blit);
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination.buffer
      destinationOffset:destination.offset size:output.size_bytes()];
    [blit endEncoding];
    CompleteOperation(command, nullptr, &category_gpu_time_ns);
    category_state_shared_output_bytes.fetch_add(output.size_bytes(), std::memory_order_relaxed);
  } else {
    std::memcpy(output.data(), source.contents, output.size_bytes());
    category_state_staging_bytes.fetch_add(output.size_bytes(), std::memory_order_relaxed);
  }
}
class MetalCategoryDictionary final : public CategoryDictionary {
  std::vector<CategoryColumn> columns_;
  std::vector<std::pair<std::size_t, std::size_t>> string_sizes_;
  std::vector<std::int32_t> segments_;
  std::vector<enc::HostCatIndexView> views_;
  std::unique_ptr<VectorBuffer> data_{MakeVectorBuffer()}, metadata_{MakeVectorBuffer()};

  std::vector<enc::HostCatIndexView> Views(void const* data) const {
    auto bytes = static_cast<std::uint8_t const*>(data);
    std::vector<enc::HostCatIndexView> views;
    auto address = [&](std::size_t offset) { return bytes ? bytes + offset : nullptr; };
    for (std::size_t f = 0; f < columns_.size(); ++f) {
      auto const& col = columns_[f];
      switch (col.type) {
        case 0:
          views.emplace_back(enc::CatStrArrayView{
              {reinterpret_cast<std::int32_t const*>(address(col.offsets)), string_sizes_[f].first},
              {reinterpret_cast<enc::CatCharT const*>(address(col.values)), string_sizes_[f].second}});
          break;
#define CATEGORY_VIEW(Code, T) \
        case Code: views.emplace_back(Span<T const>{reinterpret_cast<T const*>(address(col.values)), col.size}); break;
        CATEGORY_VIEW(1, std::uint8_t)
        CATEGORY_VIEW(2, std::int8_t)
        CATEGORY_VIEW(3, std::uint16_t)
        CATEGORY_VIEW(4, std::int16_t)
        CATEGORY_VIEW(5, std::uint32_t)
        CATEGORY_VIEW(6, std::int32_t)
        CATEGORY_VIEW(7, std::uint64_t)
        CATEGORY_VIEW(8, std::int64_t)
#undef CATEGORY_VIEW
        default: LOG(FATAL) << "Invalid Metal category type.";
      }
    }
    return views;
  }

 public:
  explicit MetalCategoryDictionary(enc::HostColumnsView view) {
    struct Part { void const* source; std::size_t bytes, offset; };
    std::vector<Part> parts;
    std::size_t size = 0;
    auto append = [&](void const* source, std::size_t bytes) {
      auto offset = (size + 7) & ~std::size_t{7};
      CHECK_LE(bytes, GetRuntime().device.maxBufferLength);
      CHECK_LE(offset, GetRuntime().device.maxBufferLength - bytes);
      parts.push_back({source, bytes, offset});
      size = offset + bytes;
      return offset;
    };
    if (view.Size() == 0 && view.feature_segments.empty()) {
      CHECK_EQ(view.n_total_cats, 0);
      segments_ = {0};
    } else {
      CHECK_EQ(view.feature_segments.size(), view.Size() + 1);
      CHECK_EQ(view.feature_segments.front(), 0);
      CHECK_EQ(view.feature_segments.back(), view.n_total_cats);
      segments_.assign(view.feature_segments.begin(), view.feature_segments.end());
    }
    CHECK_LE(view.Size(), std::numeric_limits<std::uint32_t>::max());
    columns_.resize(view.Size());
    string_sizes_.resize(view.Size());
    for (std::size_t f = 0; f < view.Size(); ++f) {
      auto& out = columns_[f];
      CHECK_GE(segments_[f + 1], segments_[f]);
      out.begin = segments_[f];
      std::visit([&](auto const& col) {
        CHECK_EQ(col.size(), segments_[f + 1] - segments_[f]);
        out.size = col.size();
        out.empty = col.empty();
        if constexpr (std::is_same_v<std::decay_t<decltype(col)>, enc::CatStrArrayView>) {
          string_sizes_[f] = {col.offsets.size(), col.values.size()};
          out.offsets = append(col.offsets.data(), col.offsets.size_bytes());
          out.values = append(col.values.data(), col.values.size_bytes());
        } else {
          using T = std::remove_const_t<typename std::decay_t<decltype(col)>::value_type>;
          static_assert(std::is_integral_v<T>);
          out.type = (sizeof(T) == 1 ? 1 : sizeof(T) == 2 ? 3 : sizeof(T) == 4 ? 5 : 7) +
                     std::is_signed_v<T>;
          out.values = append(col.data(), col.size_bytes());
        }
      }, view.columns[f]);
    }
    data_->Resize(size);
    for (auto const& part : parts) data_->Import(part.source, part.offset, part.bytes);
    metadata_->Resize(columns_.size() * sizeof(CategoryColumn));
    metadata_->Import(columns_.data(), 0, metadata_->SizeBytes());
    views_ = Views(data_->Data());
  }
  explicit MetalCategoryDictionary(MetalCategoryDictionary const& that)
      : columns_{that.columns_}, string_sizes_{that.string_sizes_}, segments_{that.segments_} {
    data_->Resize(that.data_->SizeBytes());
    data_->Copy(*that.data_, 0, 0, that.data_->SizeBytes());
    metadata_->Resize(that.metadata_->SizeBytes());
    metadata_->Copy(*that.metadata_, 0, 0, that.metadata_->SizeBytes());
    views_ = Views(data_->Data());
  }
  enc::HostColumnsView View() const override { return {views_, segments_, segments_.back()}; }
  std::unique_ptr<CategoryDictionary> Clone() const override {
    return std::make_unique<MetalCategoryDictionary>(*this);
  }
  void CopyToHost(::xgboost::cpu_impl::CatContainerImpl* output) const override {
    std::vector<std::uint8_t> data(data_->SizeBytes());
    data_->Readback(data.data(), data.size());
    auto views = Views(data.data());
    output->columns.clear();
    for (auto const& col : views) {
      std::visit([&](auto const& values) {
        if constexpr (std::is_same_v<std::decay_t<decltype(values)>, enc::CatStrArrayView>) {
          output->columns.emplace_back(::xgboost::cpu_impl::CatStrArray{
              {values.offsets.begin(), values.offsets.end()}, {values.values.begin(), values.values.end()}});
        } else {
          using T = std::remove_const_t<typename std::decay_t<decltype(values)>::value_type>;
          output->columns.emplace_back(std::vector<T>{values.begin(), values.end()});
        }
      }, col);
    }
    output->Finalize();
  }
  void Sort(Span<std::int32_t> sorted) const override;
  void Recode(Span<std::int32_t const> sorted, CategoryDictionary const& incoming,
              Span<std::int32_t> mapping) const override;
};
}  // namespace
void MetalCategoryDictionary::Sort(Span<std::int32_t> sorted) const {
  auto columns = View();
  CHECK_GE(columns.n_total_cats, 0);
  CHECK_EQ(sorted.size(), columns.n_total_cats)
      << "`sorted_idx` should have the same size as `n_total_cats`.";
  if (sorted.empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto metadata = BindCategoryInput(rt, metadata_->Data(), metadata_->SizeBytes(), 8);
    auto data = BindCategoryInput(rt, data_->Data(), data_->SizeBytes(), 8);
    auto keys = Allocate(rt, sorted.size() * 8), scratch = Allocate(rt, sorted.size() * 8);
    auto result = Allocate(rt, sorted.size_bytes());
    CategoryParams p{static_cast<std::uint32_t>(sorted.size()),
                     static_cast<std::uint32_t>(columns.Size()), 1, 0};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.category_iota_pipeline];
    [e setBuffer:metadata.buffer offset:metadata.offset atIndex:0];
    [e setBuffer:keys offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(p.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    for (; p.width < p.count; p.width *= 2) {
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.category_merge_pipeline];
      [e setBuffer:data.buffer offset:data.offset atIndex:0];
      [e setBuffer:metadata.buffer offset:metadata.offset atIndex:1];
      [e setBuffer:keys offset:0 atIndex:2];
      [e setBuffer:scratch offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreads:MTLSizeMake(p.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      std::swap(keys, scratch);
    }
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.category_extract_pipeline];
    [e setBuffer:keys offset:0 atIndex:0];
    [e setBuffer:result offset:0 atIndex:1];
    [e setBytes:&p.count length:sizeof(p.count) atIndex:2];
    [e dispatchThreads:MTLSizeMake(p.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &category_sort_dispatches, &category_gpu_time_ns);
    PublishCategoryOutput(rt, result, sorted);
  }
}

void MetalCategoryDictionary::Recode(Span<std::int32_t const> sorted,
                                       CategoryDictionary const& input,
                                       Span<std::int32_t> mapping) const {
  auto original = View();
  auto const& next = dynamic_cast<MetalCategoryDictionary const&>(input);
  auto incoming = next.View();
  enc::detail::BasicChecks(CategoryPolicy{}, original, sorted, incoming, mapping);
  CHECK_GE(original.n_total_cats, 0);
  CHECK_GE(incoming.n_total_cats, 0);
  if (original.Empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto om = BindCategoryInput(rt, metadata_->Data(), metadata_->SizeBytes(), 8);
    auto od = BindCategoryInput(rt, data_->Data(), data_->SizeBytes(), 8);
    auto nm = BindCategoryInput(rt, next.metadata_->Data(), next.metadata_->SizeBytes(), 8);
    auto nd = BindCategoryInput(rt, next.data_->Data(), next.data_->SizeBytes(), 8);
    auto order = BindCategoryInput(rt, sorted.data(), sorted.size_bytes(), 4);
    auto result = Allocate(rt, mapping.size_bytes());
    CategoryParams p{static_cast<std::uint32_t>(mapping.size()),
                     static_cast<std::uint32_t>(incoming.Size()), 0, 0};
    std::uint32_t status[2]{0, p.count};
    auto error = Upload(rt, status, sizeof(status));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.category_validate_pipeline];
    [e setBuffer:om.buffer offset:om.offset atIndex:0];
    [e setBuffer:nm.buffer offset:nm.offset atIndex:1];
    [e setBuffer:error offset:0 atIndex:2];
    [e setBytes:&p.features length:sizeof(p.features) atIndex:3];
    [e dispatchThreads:MTLSizeMake(p.features, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    if (p.count) {
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.category_recode_pipeline];
      [e setBuffer:od.buffer offset:od.offset atIndex:0];
      [e setBuffer:om.buffer offset:om.offset atIndex:1];
      [e setBuffer:order.buffer offset:order.offset atIndex:2];
      [e setBuffer:nd.buffer offset:nd.offset atIndex:3];
      [e setBuffer:nm.buffer offset:nm.offset atIndex:4];
      [e setBuffer:result offset:0 atIndex:5];
      [e setBuffer:error offset:0 atIndex:6];
      [e setBytes:&p length:sizeof(p) atIndex:7];
      [e dispatchThreads:MTLSizeMake(p.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    CompleteOperation(command, &category_recode_dispatches, &category_gpu_time_ns);
    std::memcpy(status, error.contents, sizeof(status));
    CHECK_EQ(status[0], 0) << "Invalid new DataFrame. "
                              "The data type doesn't match the one used in the training dataset. "
                              "Both should be either numeric or categorical. "
                              "For a categorical feature, the index type must match between the "
                              "training and test set.";
    if (status[1] < p.count) {
      auto it = std::upper_bound(incoming.feature_segments.begin(), incoming.feature_segments.end(),
                                 status[1]);
      auto f = std::distance(incoming.feature_segments.begin(), it) - 1;
      auto index = status[1] - incoming.feature_segments[f];
      std::stringstream name;
      std::visit(
          [&](auto const& col) {
            if constexpr (std::is_same_v<std::decay_t<decltype(col)>, enc::CatStrArrayView>) {
              for (auto c : col.values.subspan(col.offsets[index],
                                               col.offsets[index + 1] - col.offsets[index]))
                name.put(c);
            } else {
              name << col[index];
            }
          },
          incoming.columns[f]);
      enc::detail::ReportMissing(CategoryPolicy{}, name.str(), f);
    }
    PublishCategoryOutput(rt, result, mapping);
  }
}
std::unique_ptr<CategoryDictionary> MakeCategoryDictionary(enc::HostColumnsView columns) {
  @autoreleasepool { return std::make_unique<MetalCategoryDictionary>(columns); }
}
void SortCategories(enc::HostColumnsView columns, Span<std::int32_t> sorted) {
  CHECK_GE(columns.n_total_cats, 0);
  CHECK_EQ(sorted.size(), columns.n_total_cats);
  if (sorted.empty()) return;
  MakeCategoryDictionary(columns)->Sort(sorted);
}
void RecodeCategories(enc::HostColumnsView original, Span<std::int32_t const> sorted,
                      enc::HostColumnsView incoming, Span<std::int32_t> mapping) {
  enc::detail::BasicChecks(CategoryPolicy{}, original, sorted, incoming, mapping);
  CHECK_GE(original.n_total_cats, 0);
  CHECK_GE(incoming.n_total_cats, 0);
  if (original.Empty()) return;
  auto orig = MakeCategoryDictionary(original), next = MakeCategoryDictionary(incoming);
  orig->Recode(sorted, *next, mapping);
}

namespace {
class MetalModelSnapshot final : public ModelSnapshot {
 public:
  id<MTLBuffer> buffer;
  explicit MetalModelSnapshot(id<MTLBuffer> value) : buffer{value} {}
  void const* Data() const override { return buffer.contents; }
};
id<MTLBuffer> SnapshotModelWords(void const* source, std::size_t words) {
  CHECK(source || !words);
  auto& rt = GetRuntime();
  CHECK_LE(words, rt.device.maxBufferLength / 4);
  @autoreleasepool {
    auto bytes = words * 4;
    auto output = Allocate(rt, bytes);
    if (bytes) {
      auto input = TryBorrowMemory(rt.device, source, bytes, 4);
      if (!input.buffer) {
        input = {Upload(rt, source, bytes), 0};
        model_host_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
      }
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      [blit copyFromBuffer:input.buffer sourceOffset:input.offset toBuffer:output
        destinationOffset:0 size:bytes];
      [blit endEncoding];
      CompleteOperation(command, nullptr, &model_state_gpu_time_ns);
      model_snapshot_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return output;
  }
}
}  // namespace
std::unique_ptr<ModelSnapshot> SnapshotModelBuffer(void const* source, std::size_t words) {
  return std::make_unique<MetalModelSnapshot>(SnapshotModelWords(source, words));
}

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
  std::vector<std::unique_ptr<ModelSnapshot>> snapshots;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto bytes = words * sizeof(std::uint32_t);
    auto out = TryBorrowMemory(rt.device, output, bytes, alignof(std::uint32_t));
    bool copy_out = !out.buffer;
    if (copy_out) {
      out = {Upload(rt, output, bytes), 0};
      model_host_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    std::vector<BorrowedMemory> sources;
    sources.reserve(copies.size());
    // Retain every source, including snapshots of aliases, until completion.
    for (auto const& copy : copies) {
      auto size = copy.count * sizeof(std::uint32_t);
      if (!size) {
        sources.push_back({});
        continue;
      }
      auto source = copy.source;
      if (MemoryRangesOverlap(source, size, output, bytes)) {
        snapshots.push_back(SnapshotModelBuffer(source, copy.count));
        source = snapshots.back()->Data();
      }
      auto in = TryBorrowMemory(rt.device, source, size, alignof(std::uint32_t));
      if (!in.buffer) {
        in = {Upload(rt, source, size), 0};
        model_host_copy_bytes.fetch_add(size, std::memory_order_relaxed);
      }
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
    if (copy_out) {
      std::memcpy(output, out.buffer.contents, bytes);
      model_host_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
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
    if (copy_out) {
      out = {Upload(rt, weights.data(), weights.size_bytes()), 0};
      model_host_copy_bytes.fetch_add(weights.size_bytes(), std::memory_order_relaxed);
    }
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
    if (copy_out) {
      std::memcpy(weights.data(), out.buffer.contents, weights.size_bytes());
      model_host_copy_bytes.fetch_add(weights.size_bytes(), std::memory_order_relaxed);
    }
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
// Objective payloads may be owned Metal spans or host callers of the native API.
// Keep guarded host compatibility and account every fallback explicitly.
BorrowedMemory ObjectiveInput(Runtime const& rt, void const* data, std::size_t bytes,
                                std::size_t alignment = 4) {
  auto input = TryBorrowMemory(rt.device, data, bytes, alignment);
  if (input.buffer) {
    objective_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    input = {Upload(rt, data, bytes), 0};
    objective_staging_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}
struct ObjectiveOutput {
  void* destination;
  std::size_t bytes;
  BorrowedMemory memory;
  bool shared;
  ObjectiveOutput(Runtime const& rt, void* data, std::size_t size, bool disjoint,
                    id<MTLBuffer> fallback = nil)
      : destination{data}, bytes{size},
        memory{disjoint ? TryBorrowMemory(rt.device, data, size, 8) : BorrowedMemory{}},
        shared{memory.buffer != nil} {
    if (shared) objective_shared_output_bytes.fetch_add(bytes, std::memory_order_relaxed);
    else {
      memory = {fallback ? fallback : Allocate(rt, bytes), 0};
      objective_staging_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
  }
  // Late publication from independent scratch preserves aliased input reads.
  // Unshareable output already uses this scratch as its host publication source.
  void Store(id<MTLCommandBuffer> command, id<MTLBuffer> source) {
    if (!bytes) return;
    if (!shared) {
      memory = {source, 0};
      return;
    }
    auto e = [command blitCommandEncoder];
    CHECK(e);
    [e copyFromBuffer:source sourceOffset:0 toBuffer:memory.buffer
        destinationOffset:memory.offset size:bytes];
    [e endEncoding];
  }
  void Publish() const {
    if (!shared && bytes) std::memcpy(destination, memory.buffer.contents, bytes);
  }
};
template <typename T, typename... Inputs>
bool ObjectiveOutputDisjoint(Span<T> output, Inputs... inputs) {
  return (!MemoryRangesOverlap(output.data(), output.size_bytes(), inputs.data(),
                               inputs.size_bytes()) && ...);
}
// Bias kernels update their input buffers. Snapshot on GPU so caller storage
// remains unchanged until all input reads and gradient calculations finish.
id<MTLBuffer> ObjectiveSnapshot(Runtime const& rt, id<MTLCommandBuffer> command,
                                 void const* data, std::size_t bytes) {
  auto input = ObjectiveInput(rt, data, bytes, alignof(double));
  auto result = Allocate(rt, bytes);
  if (bytes) {
    auto e = [command blitCommandEncoder];
    CHECK(e);
    [e copyFromBuffer:input.buffer sourceOffset:input.offset toBuffer:result
        destinationOffset:0 size:bytes];
    [e endEncoding];
  }
  return result;
}
// Guarded synchronous bindings for metric and ranking-cache payloads.
BorrowedMemory MetricInput(Runtime const& rt, void const* data, std::size_t bytes,
                             std::size_t alignment = 4) {
  auto input = TryBorrowMemory(rt.device, data, bytes, alignment);
  if (input.buffer) metric_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  else {
    input = {Upload(rt, data, bytes), 0};
    metric_host_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}
// Compute into scratch before publication so output may alias any input.
BorrowedMemory PublishMetric(Runtime const& rt, id<MTLCommandBuffer> command,
                           id<MTLBuffer> result, void* output, std::size_t bytes) {
  if (!bytes) return {};
  auto destination = TryBorrowMemory(rt.device, output, bytes, alignof(double));
  if (destination.buffer) {
    auto e = [command blitCommandEncoder];
    CHECK(e);
    [e copyFromBuffer:result sourceOffset:0 toBuffer:destination.buffer
        destinationOffset:destination.offset size:bytes];
    [e endEncoding];
    metric_shared_output_bytes.fetch_add(bytes, std::memory_order_relaxed);
    metric_copy_dispatches.fetch_add(1, std::memory_order_relaxed);
    metric_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return destination;
}
void FinishMetricPublication(BorrowedMemory destination, id<MTLBuffer> result,
                              void* output, std::size_t bytes) {
  if (!destination.buffer && bytes) {
    std::memcpy(output, result.contents, bytes);
    metric_host_output_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
}
BorrowedMemory InitializationInput(Runtime const& rt, void const* data, std::size_t bytes,
                                   std::size_t alignment = 4) {
  auto input = TryBorrowMemory(rt.device, data, bytes, alignment);
  if (input.buffer) initialization_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  else {
    input = {Upload(rt, data, bytes), 0};
    initialization_host_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}
id<MTLBuffer> InitializationSnapshot(Runtime const& rt, id<MTLCommandBuffer> command,
                                     BorrowedMemory input, std::size_t bytes) {
  auto result = Allocate(rt, bytes);
  if (bytes) {
    auto e = [command blitCommandEncoder];
    [e copyFromBuffer:input.buffer sourceOffset:input.offset toBuffer:result destinationOffset:0 size:bytes];
    [e endEncoding];
    initialization_copy_dispatches.fetch_add(1, std::memory_order_relaxed);
    initialization_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return result;
}
// Sort helpers consume offset-zero buffers. Preserve device payloads with a blit
// when a borrowed view starts inside a VM page, instead of uploading via the CPU.
id<MTLBuffer> InitializationPacked(Runtime const& rt, id<MTLCommandBuffer> command,
                                   void const* data, std::size_t bytes) {
  auto input = InitializationInput(rt, data, bytes);
  return input.offset ? InitializationSnapshot(rt, command, input, bytes) : input.buffer;
}
template <typename T>
class InitializationOutput {
  linalg::VectorView<T> view_;
  bool shared_;
 public:
  BorrowedMemory memory;
  InitializationOutput(Runtime const& rt, linalg::VectorView<T> view, bool initialize = false)
      : view_{view} {
    CHECK(view.Size() <= 1 || view.Stride(0) > 0);
    if (view.Size()) {
      CHECK(!view.Values().empty());
      CHECK_LE(view.Size()-1, (view.Values().size()-1) / std::max<std::size_t>(view.Stride(0), 1));
    }
    auto bytes = view.Values().size_bytes();
    memory = TryBorrowMemory(rt.device, view.Values().data(), bytes, alignof(T));
    shared_ = memory.buffer != nil;
    if (shared_) initialization_shared_output_bytes.fetch_add(bytes, std::memory_order_relaxed);
    else if (initialize || view.Size() != view.Values().size() || view.Stride(0) != 1) {
      memory = {Upload(rt, view.Values().data(), bytes), 0};
      initialization_host_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else memory = {Allocate(rt, bytes), 0};
  }
  void Store(Runtime const& rt, id<MTLCommandBuffer> command, BorrowedMemory source,
              std::size_t source_stride = sizeof(T)) {
    if (!view_.Size()) return;
    CHECK_LE(view_.Size(), std::numeric_limits<std::size_t>::max()/sizeof(T));
    struct { std::uint64_t count, width, source_stride, target_stride; }
      p{view_.Size(), sizeof(T), source_stride, view_.Stride(0)*sizeof(T)};
    auto e = [command computeCommandEncoder];
    [e setComputePipelineState:rt.initialization_store_pipeline];
    [e setBuffer:source.buffer offset:source.offset atIndex:0];
    [e setBuffer:memory.buffer offset:memory.offset atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(std::min<std::size_t>(view_.Size()*sizeof(T), 1<<20),1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
    if (source_stride) {
      initialization_copy_dispatches.fetch_add(1, std::memory_order_relaxed);
      initialization_copy_bytes.fetch_add(view_.Size()*sizeof(T), std::memory_order_relaxed);
    } else initialization_fill_dispatches.fetch_add(1, std::memory_order_relaxed);
  }
  void Publish() const {
    if (!shared_ && !view_.Values().empty()) {
      std::memcpy(view_.Values().data(), memory.buffer.contents, view_.Values().size_bytes());
      initialization_host_output_bytes.fetch_add(view_.Values().size_bytes(), std::memory_order_relaxed);
    }
  }
};
template <typename T>
void CopyInitializationResult(Runtime const& rt, id<MTLBuffer> result, linalg::VectorView<T> view) {
  if (!view.Size()) return;
  InitializationOutput<T> out{rt, view};
  auto command = [rt.queue commandBuffer];
  out.Store(rt, command, {result,0});
  CompleteOperation(command, nullptr, &initialization_gpu_time_ns);
  out.Publish();
}
template <typename T>
void FillInitialization(linalg::VectorView<T> view, T value) {
  if (!view.Size()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto pattern = Upload(rt, &value, sizeof(T));  // Scalar fill argument, not a payload array.
    InitializationOutput<T> out{rt, view};
    auto command = [rt.queue commandBuffer];
    out.Store(rt, command, {pattern,0}, 0);
    CompleteOperation(command, nullptr, &initialization_gpu_time_ns);
    out.Publish();
  }
}
id<MTLBuffer> EncodeGradientSum(Runtime& rt, linalg::MatrixView<GradientPair const> gradients,
                                id<MTLCommandBuffer> command) {
  auto rows = gradients.Shape(0), targets = gradients.Shape(1);
  CHECK_GT(rows, 0);
  CHECK_GT(targets, 0);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(targets, std::numeric_limits<std::uint32_t>::max());
  auto values = gradients.Values();
  auto input = InitializationInput(rt, values.data(), values.size_bytes(), 8);
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
  [e setBuffer:input.buffer offset:input.offset atIndex:0];
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
    FillInitialization(output, GradientPairPrecise{});
    return;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto result = EncodeGradientSum(rt, gradients, command);
    CompleteGradientSum(command);
    CopyInitializationResult(rt, result, output);
  }
}
void FitStump(linalg::MatrixView<GradientPair const> gradients, linalg::VectorView<float> output,
              Context const* ctx) {
  CHECK_EQ(gradients.Shape(1), output.Size());
  auto distributed = collective::GlobalCommGroup()->IsDistributed();
  CHECK(ctx || !distributed) << "Distributed Metal stump fitting requires a Context.";
  if (!output.Size()) return;
  if (!gradients.Size() && !distributed) {
    FillInitialization(output, 0.0f);
    return;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    id<MTLBuffer> sums;
    if (gradients.Size()) {
      sums = EncodeGradientSum(rt, gradients, command);
    } else {
      // CUDA initializes per-target GPU sums to zero even for empty workers.
      sums = Allocate(rt, output.Size() * sizeof(GradientPairPrecise));
      auto clear = [command blitCommandEncoder];
      CHECK(clear);
      [clear fillBuffer:sums range:NSMakeRange(0, output.Size() * sizeof(GradientPairPrecise)) value:0];
      [clear endEncoding];
    }
    if (distributed) {
      CompleteGradientSum(command);
      auto view = linalg::MakeVec(static_cast<double*>(sums.contents), output.Size() * 2,
                                  DeviceOrd::MPS());
      collective::SafeColl(collective::GlobalSum(ctx, view));
      command = [rt.queue commandBuffer];
      CHECK(command);
    }
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
    CopyInitializationResult(rt, result, output);
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
    auto input = InitializationInput(rt, values.data(), values.size_bytes(), alignof(float));
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
    [e setBuffer:input.buffer offset:input.offset atIndex:0];
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

double SumFloat32(Span<float const> values) {
  if (values.empty()) return 0.;
  // Match dh::Reduce's outer batch boundary and FP32 host aggregation.
  constexpr std::size_t kLimit = std::numeric_limits<std::int32_t>::max() / 2;
  float aggregate = 0.f;
  @autoreleasepool {
    auto& rt = GetRuntime();
    for (std::size_t offset = 0; offset < values.size(); offset += kLimit) {
      std::uint64_t n = std::min(kLimit, values.size() - offset);
      auto input = InitializationInput(rt, values.data() + offset, n * sizeof(float));
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      do {
        auto blocks = (n + 255) / 256;
        auto output = Allocate(rt, blocks * sizeof(float));
        auto e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt.float32_sum_pipeline];
        [e setBuffer:input.buffer offset:input.offset atIndex:0];
        [e setBuffer:output offset:0 atIndex:1];
        [e setBytes:&n length:sizeof(n) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        input = {output, 0};
        n = blocks;
      } while (n > 1);
      CompleteOperation(command, &float_sum_dispatches, &float_sum_gpu_time_ns);
      aggregate += *static_cast<float const*>(input.buffer.contents);
    }
  }
  return aggregate;
}

void VectorArithmetic(VectorOp op, linalg::VectorView<float> values, double parameter) {
  CHECK_LE(static_cast<std::uint32_t>(op), 2);
  CHECK_LE(values.Size(), std::numeric_limits<std::uint32_t>::max());
  CHECK(values.Size() <= 1 || values.Stride(0) > 0);
  if (!values.Size()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    InitializationOutput<float> buffer{rt, values, true};
    struct {
      std::uint64_t count, stride;
      double parameter;
      std::uint32_t op, padding;
    } p{values.Size(), values.Stride(0), parameter, static_cast<std::uint32_t>(op), 0};
    static_assert(sizeof(p) == 32);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.vector_arithmetic_pipeline];
    [e setBuffer:buffer.memory.buffer offset:buffer.memory.offset atIndex:0];
    [e setBytes:&p length:sizeof(p) atIndex:1];
    [e dispatchThreads:MTLSizeMake(values.Size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &vector_arithmetic_dispatches, &initialization_gpu_time_ns);
    buffer.Publish();
  }
}

void Center(linalg::VectorView<float> values, linalg::VectorView<float const> mean) {
  CHECK_EQ(mean.Size(), 1);
  CHECK_LE(values.Size(), std::numeric_limits<std::uint32_t>::max());
  if (!values.Size()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    auto scalar = InitializationInput(rt, mean.Values().data(), sizeof(float));
    if (MemoryRangesOverlap(mean.Values().data(), sizeof(float), values.Values().data(), values.Values().size_bytes()))
      scalar = {InitializationSnapshot(rt, command, scalar, sizeof(float)), 0};
    InitializationOutput<float> output{rt, values, true};
    auto e = [command computeCommandEncoder];
    std::uint64_t stride = values.Stride(0);
    [e setComputePipelineState:rt.vector_center_pipeline];
    [e setBuffer:output.memory.buffer offset:output.memory.offset atIndex:0];
    [e setBuffer:scalar.buffer offset:scalar.offset atIndex:1];
    [e setBytes:&stride length:sizeof(stride) atIndex:2];
    [e dispatchThreads:MTLSizeMake(values.Size(),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
    CompleteOperation(command, &vector_arithmetic_dispatches, &initialization_gpu_time_ns);
    output.Publish();
  }
}

void Mean(MeanKind kind, linalg::MatrixView<float const> values, Span<float const> weights,
          linalg::VectorView<float> output, Context const* ctx) {
  bool weighted = kind == MeanKind::kWeightedSample;
  bool distributed = kind != MeanKind::kFloat32 && collective::GlobalCommGroup()->IsDistributed();
  CHECK(ctx || !distributed) << "Distributed Metal sample means require a Context.";
  auto rows = values.Shape(0), targets = values.Shape(1);
  CHECK_EQ(output.Size(), std::max<std::size_t>(targets, 1));
  if (weighted) {
    CHECK_EQ(weights.size(), rows);
    if (!distributed) CHECK_GT(rows, 0) << "weights must contain at least one non-zero value.";
  }
  if (!distributed && (!rows || (!targets && !weighted))) {
    FillInitialization(output, 0.0f);
    return;
  }
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(targets, std::numeric_limits<std::uint32_t>::max());
  double global_denominator = 0;
  if (distributed) {
    // stats.cu deliberately reduces this scalar on HOST, after a GPU local
    // weight sum when weighted. Empty workers must participate as well.
    if (weighted) {
      global_denominator = Sum(weights);
      collective::SafeColl(collective::GlobalSum(ctx, linalg::MakeVec(&global_denominator, 1)));
      CHECK_GT(global_denominator, 0.0) << "weights must contain at least one non-zero value.";
    } else {
      std::uint64_t global_rows = rows;
      collective::SafeColl(collective::GlobalSum(ctx, linalg::MakeVec(&global_rows, 1)));
      global_denominator = static_cast<double>(global_rows);
    }
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto input = InitializationInput(rt, values.Values().data(), values.Values().size_bytes());
    auto weight = InitializationInput(rt, weights.data(), weights.size_bytes());
    struct Shape {
      std::uint64_t rows, stride0, stride1;
      std::uint32_t targets, blocks;
    } shape{rows, values.Stride(0), values.Stride(1), static_cast<std::uint32_t>(targets),
            static_cast<std::uint32_t>((rows + 255) / 256)};
    static_assert(sizeof(Shape) == 32);
    if (kind == MeanKind::kFloat32) {
      auto output_buffer = Allocate(rt, targets * sizeof(float));
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.float_mean_pipeline];
      [e setBuffer:input.buffer offset:input.offset atIndex:0];
      [e setBuffer:output_buffer offset:0 atIndex:1];
      [e setBytes:&shape length:sizeof(shape) atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(targets, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &mean_dispatches, &mean_gpu_time_ns);
      CopyInitializationResult(rt, output_buffer, output);
      float_mean_dispatches.fetch_add(1, std::memory_order_relaxed);
      return;
    }
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
    if (distributed) {
      GradientPairPrecise host_denominator{global_denominator, 0.0};
      std::memcpy(denominator.contents, &host_denominator, sizeof(host_denominator));
    }
    if (weighted && !distributed) {
      auto partial = Allocate(rt, shape.blocks * sizeof(GradientPairPrecise));
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.mean_weight_pipeline];
      [e setBuffer:weight.buffer offset:weight.offset atIndex:0];
      [e setBuffer:partial offset:0 atIndex:1];
      [e setBytes:&shape length:sizeof(shape) atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(shape.blocks, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto weight_shape = shape;
      weight_shape.targets = 1;
      denominator = finish(partial, weight_shape);
    }
    auto result = Allocate(rt, output.Size() * sizeof(float));
    if (!targets && distributed) {
      auto clear = [command blitCommandEncoder];
      CHECK(clear);
      [clear fillBuffer:result range:NSMakeRange(0, sizeof(float)) value:0];
      [clear endEncoding];
    }
    if (targets) {
      auto partial = Allocate(rt, shape.blocks * targets * sizeof(GradientPairPrecise));
      struct {
        Shape shape;
        std::uint32_t weighted, float32_terms, global_denominator, padding;
      } p{shape, static_cast<std::uint32_t>(weighted), 0,
          static_cast<std::uint32_t>(distributed), 0};
      static_assert(sizeof(p) == 48);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.mean_pipeline];
      [e setBuffer:input.buffer offset:input.offset atIndex:0];
      [e setBuffer:weight.buffer offset:weight.offset atIndex:1];
      [e setBuffer:denominator offset:0 atIndex:2];
      [e setBuffer:partial offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      if (shape.blocks)
        [e dispatchThreadgroups:MTLSizeMake(shape.blocks, targets, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto sums = finish(partial, shape);
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
    if (distributed) {
      // CUDA casts each local binary64 mean to FP32 before its GPU GlobalSum.
      auto view = linalg::MakeVec(static_cast<float*>(result.contents), output.Size(), DeviceOrd::MPS());
      collective::SafeColl(collective::GlobalSum(ctx, view));
    }
    if (targets || distributed) CopyInitializationResult(rt, result, output);
    else FillInitialization(output, 0.0f);
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
BorrowedMemory BindFeatureInput(Runtime const& rt, void const* data, std::size_t bytes,
                               std::size_t alignment = 4) {
  auto input = TryBorrowMemory(rt.device, data, bytes, alignment);
  if (input.buffer) {
    feature_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    input = {Upload(rt, data, bytes), 0};
    feature_host_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}
void PublishFeatureOutput(Runtime const& rt, BorrowedMemory source, void* output,
                           std::size_t bytes, std::atomic<std::uint64_t>* duration) {
  if (!bytes) return;
  auto destination = TryBorrowMemory(rt.device, output, bytes, 4);
  if (destination.buffer) {
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto blit = [command blitCommandEncoder];
    CHECK(blit);
    [blit copyFromBuffer:source.buffer sourceOffset:source.offset toBuffer:destination.buffer
      destinationOffset:destination.offset size:bytes];
    [blit endEncoding];
    CompleteOperation(command, nullptr, duration);
    feature_shared_output_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    std::memcpy(output, static_cast<std::byte const*>(source.buffer.contents) + source.offset, bytes);
    feature_readback_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
}
class MetalInteractionConstraints final : public InteractionConstraints {
  Runtime& rt_;
  bool enabled_;
  std::uint32_t features_, words_;
  std::size_t capacity_{0};
  std::vector<std::uint32_t> group_counts_;
  id<MTLBuffer> group_ptr_, group_features_, feature_ptr_, feature_groups_;
  id<MTLBuffer> nodes_{nil}, feature_mask_{nil}, input_mask_{nil}, flags_{nil},
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
  std::size_t Filter(Span<bst_feature_t const> input, bst_node_t node, bool all) {
    CHECK_GE(node, 0);
    CHECK_LE(input.size(), std::numeric_limits<std::uint32_t>::max());
    if (!features_) return 0;
    @autoreleasepool {
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      auto grew = Grow(command, std::size_t(node) + 1);
      auto mask = Scratch(rt_, &input_mask_, words_ * 4ul);
      auto source = BindFeatureInput(rt_, input.data(), input.size_bytes());
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
        [e setBuffer:source.buffer offset:source.offset atIndex:0];
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
      return size;
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
  HostDeviceVector<bst_feature_t> QueryDevice(Span<bst_feature_t const> input,
                                              bst_node_t node) override {
    // CUDA preserves the original list, including order/duplicates, at root or
    // when disabled. QueryNode deliberately does not use this root bypass.
    @autoreleasepool {
      BorrowedMemory source;
      std::size_t size;
      if (!enabled_ || node == 0) {
        size = input.size();
        source = BindFeatureInput(rt_, input.data(), input.size_bytes());
      } else {
        size = Filter(input, node, false);
        source = {output_, 0};
      }
      HostDeviceVector<bst_feature_t> result(size, 0, DeviceOrd::MPS());
      PublishFeatureOutput(rt_, source, result.DevicePointer(), result.SizeBytes(),
                           &interaction_gpu_time_ns);
      return result;
    }
  }
  HostDeviceVector<bst_feature_t> QueryNodeDevice(bst_node_t node) override {
    @autoreleasepool {
      auto size = enabled_ ? Filter({}, node, true) : 0;
      HostDeviceVector<bst_feature_t> result(size, 0, DeviceOrd::MPS());
      PublishFeatureOutput(rt_, {output_, 0}, result.DevicePointer(), result.SizeBytes(),
                           &interaction_gpu_time_ns);
      return result;
    }
  }
  std::vector<bst_feature_t> Query(Span<bst_feature_t const> input, bst_node_t node) override {
    auto result = QueryDevice(input, node);
    feature_readback_bytes.fetch_add(result.SizeBytes(), std::memory_order_relaxed);
    return result.ConstHostVector();
  }
  std::vector<bst_feature_t> QueryNode(bst_node_t node) override {
    auto result = QueryNodeDevice(node);
    feature_readback_bytes.fetch_add(result.SizeBytes(), std::memory_order_relaxed);
    return result.ConstHostVector();
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
struct RadixWorkspace {
  id<MTLBuffer> alternate{nil}, ranks{nil}, counts{nil};
};
id<MTLBuffer> EncodeRadixRecords(Runtime& rt, id<MTLCommandBuffer> command,
                                  id<MTLBuffer> source, SortShape shape,
                                  std::uint32_t key_bits = 32,
                                  RadixWorkspace* workspace = nullptr) {
  auto size = shape.rows * shape.targets;
  CHECK_LE(size, std::numeric_limits<std::uint32_t>::max());
  RadixWorkspace local;
  auto& scratch = workspace ? *workspace : local;
  EnsureBuffer(rt, &scratch.alternate, source.length);
  CHECK(scratch.alternate != source);
  EnsureBuffer(rt, &scratch.ranks, size * sizeof(std::uint32_t));
  auto count_size = std::uint64_t{shape.targets} * 16 * shape.blocks;
  CHECK_LE(count_size, std::numeric_limits<std::uint32_t>::max());
  EnsureBuffer(rt, &scratch.counts, count_size * sizeof(std::uint32_t));
  auto target = scratch.alternate, ranks = scratch.ranks, counts = scratch.counts;
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
  scratch.alternate = target;
  return source;
}
id<MTLBuffer> EncodeColumnSort(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                               SortShape shape, bool descending = false,
                               std::size_t input_offset = 0) {
  auto size = shape.rows * shape.targets;
  CHECK_LE(size, std::numeric_limits<std::uint32_t>::max());
  auto source = Allocate(rt, size * 2 * sizeof(std::uint32_t));
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:descending ? rt.roc_sort_init_pipeline : rt.sort_init_pipeline];
  [e setBuffer:input offset:input_offset atIndex:0];
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

void SmallHistogram(linalg::MatrixView<float const> indices, OptionalWeights const& weights,
                    linalg::VectorView<float> bins) {
  CHECK_EQ(indices.Shape(1), 1);
  auto rows = indices.Shape(0);
  CHECK_LT(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LT(bins.Size(), std::numeric_limits<std::uint32_t>::max());
  CHECK(weights.Empty() || weights.Size() == rows);
  CHECK(bins.Size() <= 1 || bins.Stride(0) > 0);
  if (!bins.Size()) {
    CHECK_EQ(rows, 0);
    return;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto input = InitializationPacked(rt, command, indices.Values().data(), indices.Values().size_bytes());
    auto weight = InitializationInput(rt, weights.Data(), weights.Size() * sizeof(float));
    auto initial = InitializationInput(rt, bins.Values().data(), bins.Values().size_bytes());
    auto output = InitializationSnapshot(rt, command, initial, bins.Values().size_bytes());
    auto bounds = Allocate(rt, (bins.Size() + 1) * sizeof(std::uint32_t));
    auto error = Allocate(rt, sizeof(std::uint32_t));
    struct {
      std::uint64_t rows, bins, output_stride;
      std::uint32_t weighted;
      float default_weight;
    } p{rows, bins.Size(), bins.Stride(0), !weights.Empty(), weights.dft};
    static_assert(sizeof(p) == 32);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.small_hist_init_pipeline];
    [e setBuffer:bounds offset:0 atIndex:0];
    [e setBuffer:error offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(bins.Size() + 1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    id<MTLBuffer> sorted = Allocate(rt, sizeof(std::uint64_t));
    if (rows) {
      sorted = EncodeColumnSort(rt, command, input,
                                SortShape{rows, indices.Stride(0), indices.Stride(1), 1,
                                          static_cast<std::uint32_t>((rows + 255) / 256)});
      auto flags = Allocate(rt, rows * sizeof(std::uint32_t));
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.small_hist_flags_pipeline];
      [e setBuffer:sorted offset:0 atIndex:0];
      [e setBuffer:flags offset:0 atIndex:1];
      [e setBuffer:error offset:0 atIndex:2];
      [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto prefix = EncodePrefixSum(rt, command, flags, rows);
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.small_hist_boundaries_pipeline];
      [e setBuffer:flags offset:0 atIndex:0];
      [e setBuffer:prefix offset:0 atIndex:1];
      [e setBuffer:bounds offset:0 atIndex:2];
      [e setBuffer:error offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.small_hist_sum_pipeline];
    [e setBuffer:sorted offset:0 atIndex:0];
    [e setBuffer:bounds offset:0 atIndex:1];
    [e setBuffer:weight.buffer offset:weight.offset atIndex:2];
    [e setBuffer:output offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(bins.Size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &small_histogram_dispatches, &initialization_gpu_time_ns);
    CHECK_EQ(*static_cast<std::uint32_t const*>(error.contents), 0)
        << "SmallHistogram requires finite labels and no more distinct labels than bins.";
    CopyInitializationResult(rt, output, linalg::MakeVec(bins.Values().data(), bins.Values().size()));
    if (rows) radix_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
  }
}

namespace {
class MetalRankingPlan final : public RankingPlan {
  std::uint32_t groups_, pairs_, maximum_{0};
  bool mean_;
  id<MTLBuffer> offsets_, tasks_;

 public:
  MetalRankingPlan(Span<bst_group_t const> groups, std::uint32_t pairs, bool mean)
      : pairs_{pairs}, mean_{mean} {
    static_assert(sizeof(bst_group_t) == 4 && sizeof(std::size_t) == 8);
    static_assert(sizeof(RankingPairTask) == 32);
    CHECK_GE(groups.size(), 1);
    CHECK_EQ(groups.front(), 0);
    CHECK_LE(groups.size() - 1, std::numeric_limits<std::uint32_t>::max());
    groups_ = groups.size() - 1;
    @autoreleasepool {
      auto& rt = GetRuntime();
      offsets_ = Upload(rt, groups.data(), groups.size_bytes());
      tasks_ = Allocate(rt, groups.size() * sizeof(RankingPairTask));
      auto pair_counts = Allocate(rt, groups.size() * sizeof(std::uint64_t));
      auto tile_counts = Allocate(rt, groups.size() * sizeof(std::uint64_t));
      std::uint32_t initial[2]{0, 0};
      auto state = Upload(rt, initial, sizeof(initial));
      struct {
        std::uint32_t groups, pairs, mean, padding;
      } p{groups_, pairs_, static_cast<std::uint32_t>(mean_), 0};
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.ranking_plan_init_pipeline];
      [e setBuffer:offsets_ offset:0 atIndex:0];
      [e setBuffer:tasks_ offset:0 atIndex:1];
      [e setBuffer:pair_counts offset:0 atIndex:2];
      [e setBuffer:tile_counts offset:0 atIndex:3];
      [e setBuffer:state offset:0 atIndex:4];
      [e setBytes:&p length:sizeof(p) atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake((groups.size() + 255) / 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      if (pairs_) {
        auto pair_prefix = EncodePrefixU64(rt, command, pair_counts, groups.size());
        auto tile_prefix = EncodePrefixU64(rt, command, tile_counts, groups.size());
        e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt.ranking_plan_finish_pipeline];
        [e setBuffer:tasks_ offset:0 atIndex:0];
        [e setBuffer:pair_prefix offset:0 atIndex:1];
        [e setBuffer:tile_prefix offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(groups.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      auto result = static_cast<std::uint32_t const*>(state.contents);
      CHECK_EQ(result[1], 0) << "Ranking group offsets must be nondecreasing.";
      maximum_ = result[0];
      ranking_plan_dispatches.fetch_add(1, std::memory_order_relaxed);
      auto elapsed = command.GPUEndTime - command.GPUStartTime;
      if (elapsed > 0)
        ranking_plan_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                           std::memory_order_relaxed);
    }
  }
  Span<bst_group_t const> Groups() const override {
    return {static_cast<bst_group_t const*>(offsets_.contents), std::size_t{groups_} + 1};
  }
  Span<RankingPairTask const> Tasks() const override {
    return {static_cast<RankingPairTask const*>(tasks_.contents), std::size_t{groups_} + 1};
  }
  std::uint32_t MaxGroupSize() const override { return maximum_; }
  std::uint32_t NumPair() const override { return pairs_; }
  bool IsMean() const override { return mean_; }
  id<MTLBuffer> OffsetsBuffer() const { return offsets_; }
  id<MTLBuffer> TasksBuffer() const { return tasks_; }
};
std::shared_ptr<MetalRankingPlan const> CheckGroupSort(linalg::VectorView<float const> values,
                                                       Span<bst_group_t const> groups,
                                                       std::shared_ptr<RankingPlan const> plan = {},
                                                       std::uint32_t pairs = 0, bool mean = false) {
  CHECK_GE(groups.size(), 1);
  CHECK_EQ(groups.front(), 0);
  CHECK_EQ(groups.back(), values.Size());
  CHECK_LE(values.Size(), std::numeric_limits<std::uint32_t>::max());
  if (plan) {
    auto saved = plan->Groups();
    CHECK_EQ(saved.size(), groups.size());
    // Validate reuse against immutable owned offsets, including changes to an
    // input array at the same address. This is validation, not count generation.
    CHECK_EQ(std::memcmp(saved.data(), groups.data(), groups.size_bytes()), 0)
        << "Ranking metadata belongs to different query groups.";
    if (pairs) {
      CHECK_EQ(plan->NumPair(), pairs);
      CHECK_EQ(plan->IsMean(), mean);
    }
    ranking_plan_reuses.fetch_add(1, std::memory_order_relaxed);
  } else {
    plan = MakeRankingPlan(groups, pairs, mean);
  }
  auto result = std::dynamic_pointer_cast<MetalRankingPlan const>(plan);
  CHECK(result);
  return result;
}
struct GroupOrder {
  id<MTLBuffer> records;
  bool radix;
};
GroupOrder EncodeGroupSort(Runtime& rt, id<MTLCommandBuffer> command, id<MTLBuffer> input,
                           id<MTLBuffer> offsets, std::uint64_t rows, std::uint64_t stride,
                           std::uint32_t n_groups, std::uint32_t max_group_size,
                           std::size_t input_offset = 0) {
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
    [e setBuffer:input offset:input_offset atIndex:0];
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
    [e setBuffer:input offset:input_offset atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBuffer:records offset:0 atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:shared_count * 2 * sizeof(std::uint32_t) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(p.groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
  } else {
    SortShape shape{p.rows, p.stride, 1, 1, static_cast<std::uint32_t>((p.rows + 255) / 256)};
    records = EncodeColumnSort(rt, command, input, shape, true, input_offset);
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
std::shared_ptr<RankingPlan const> MakeRankingPlan(Span<bst_group_t const> groups,
                                                  std::uint32_t pairs, bool mean) {
  return std::make_shared<MetalRankingPlan>(groups, pairs, mean);
}
void SortGroups(linalg::VectorView<float const> values, Span<bst_group_t const> groups,
                Span<std::size_t> relative_indices, std::shared_ptr<RankingPlan const> plan) {
  auto metadata = CheckGroupSort(values, groups, std::move(plan));
  auto max_group_size = metadata->MaxGroupSize();
  CHECK_EQ(relative_indices.size(), values.Size());
  if (!values.Size()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = MetricInput(rt, values.Values().data(), values.Values().size_bytes());
    auto offsets = metadata->OffsetsBuffer();
    auto output = Allocate(rt, relative_indices.size_bytes());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto order = EncodeGroupSort(rt, command, input.buffer, offsets, values.Size(), values.Stride(0),
                                 groups.size() - 1, max_group_size, input.offset);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.group_sort_export_pipeline];
    [e setBuffer:order.records offset:0 atIndex:0];
    [e setBuffer:offsets offset:0 atIndex:1];
    [e setBuffer:output offset:0 atIndex:2];
    [e dispatchThreads:MTLSizeMake(values.Size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    auto destination = PublishMetric(rt, command, output, relative_indices.data(),
                                     relative_indices.size_bytes());
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    FinishMetricPublication(destination, output, relative_indices.data(),
                            relative_indices.size_bytes());
    RecordGroupSort(order.radix, command.GPUEndTime - command.GPUStartTime);
  }
}

RankingAUCSums RankingROCAUC(linalg::VectorView<float const> predictions,
                             linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                             Span<float const> weights, Span<RankingAUCSums> individual) {
  auto metadata = CheckGroupSort(labels, groups);
  auto max_group_size = metadata->MaxGroupSize();
  auto n_groups = groups.size() - 1;
  CHECK_EQ(predictions.Size(), labels.Size());
  CHECK(weights.empty() || weights.size() == n_groups);
  CHECK(individual.empty() || individual.size() == n_groups);
  if (!n_groups) return {0, 0};
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = MetricInput(rt, predictions.Values().data(), predictions.Values().size_bytes());
    auto response = MetricInput(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = metadata->OffsetsBuffer();
    auto weight = MetricInput(rt, weights.data(), weights.size_bytes());
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
      order = EncodeGroupSort(rt, command, response.buffer, offsets, labels.Size(), labels.Stride(0),
                              n_groups, max_group_size, response.offset);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.ranking_roc_init_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:input.buffer offset:input.offset atIndex:1];
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
    [e setBuffer:weight.buffer offset:weight.offset atIndex:2];
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
    auc_final_readback_bytes.fetch_add(sizeof(RankingAUCSums), std::memory_order_relaxed);
    return *static_cast<RankingAUCSums const*>(output.contents);
  }
}

std::uint32_t CheckRankingLabels(linalg::VectorView<float const> labels, RankingLabelKind kind) {
  CHECK(kind == RankingLabelKind::kBinary || kind == RankingLabelKind::kNDCG);
  CHECK_LE(labels.Size(), std::numeric_limits<std::uint32_t>::max());
  if (!labels.Size()) return 0;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = MetricInput(rt, labels.Values().data(), labels.Values().size_bytes());
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
    [e setBuffer:input.buffer offset:input.offset atIndex:0];
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
              Span<double> inverse_idcg, std::shared_ptr<RankingPlan const> plan) {
  auto metadata = CheckGroupSort(labels, groups, std::move(plan));
  auto max_group_size = metadata->MaxGroupSize();
  auto n_groups = groups.size() - 1;
  CHECK_EQ(discounts.size(), max_group_size);
  CHECK_EQ(inverse_idcg.size(), n_groups);
  if (!n_groups) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = MetricInput(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = metadata->OffsetsBuffer();
    auto discount = Allocate(rt, discounts.size_bytes());
    auto inverse = Allocate(rt, inverse_idcg.size_bytes());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    GroupOrder order{Allocate(rt, 16), false};
    if (labels.Size()) {
      order = EncodeGroupSort(rt, command, input.buffer, offsets, labels.Size(), labels.Stride(0),
                              n_groups, max_group_size, input.offset);
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
    [e setBuffer:input.buffer offset:input.offset atIndex:1];
    [e setBuffer:offsets offset:0 atIndex:2];
    [e setBuffer:discount offset:0 atIndex:3];
    [e setBuffer:inverse offset:0 atIndex:4];
    [e setBytes:&p length:sizeof(p) atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake((n_groups + 7) / 8, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    // Overlapping outputs retain the legacy discount-then-inverse publication
    // order, even if only one of the two destinations can be shared with Metal.
    bool overlap = MemoryRangesOverlap(discounts.data(), discounts.size_bytes(),
                                       inverse_idcg.data(), inverse_idcg.size_bytes());
    auto discount_out = overlap ? BorrowedMemory{} :
        PublishMetric(rt, command, discount, discounts.data(), discounts.size_bytes());
    auto inverse_out = overlap ? BorrowedMemory{} :
        PublishMetric(rt, command, inverse, inverse_idcg.data(), inverse_idcg.size_bytes());
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    FinishMetricPublication(discount_out, discount, discounts.data(), discounts.size_bytes());
    FinishMetricPublication(inverse_out, inverse, inverse_idcg.data(), inverse_idcg.size_bytes());
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
                     RankingGradientTrace diagnostics, std::shared_ptr<RankingPlan const> plan) {
  auto values = linalg::MakeTensorView(DeviceOrd::CPU(), predictions, predictions.size());
  CHECK_GE(pairs, 1);
  auto metadata = CheckGroupSort(values, groups, std::move(plan), pairs, mean);
  auto max_group_size = metadata->MaxGroupSize();
  auto n_groups = groups.size() - 1;
  CHECK_EQ(bias.positive.size(), bias.negative.size());
  CHECK_LE(bias.positive.size(), std::numeric_limits<std::uint32_t>::max());
  auto positions = static_cast<std::uint32_t>(bias.positive.size());
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
  auto tasks = metadata->Tasks();
  auto total_pairs = tasks.back().pair_begin;
  // GPU-generated counts/prefixes remain resident. Read them only to allocate
  // bounded pair storage and schedule batches; a single larger query is allowed
  // up to the device buffer limit.
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
    auto input = ObjectiveInput(rt, predictions.data(), predictions.size_bytes());
    auto response = ObjectiveInput(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = metadata->OffsetsBuffer();
    auto task_buffer = metadata->TasksBuffer();
    auto weight = ObjectiveInput(rt, weights.data(), weights.size_bytes());
    auto discount = ObjectiveInput(rt, discounts.data(), discounts.size_bytes(), alignof(double));
    auto inverse = ObjectiveInput(rt, inverse_idcg.data(), inverse_idcg.size_bytes(), alignof(double));
    auto pair_buffer = Allocate(rt, max_pairs * sizeof(GradientPair));
    auto costs = Allocate(rt, positions ? max_pairs * 8 : 8);
    auto minimum_bias = Allocate(rt, 8);
    auto cost_rounding = Allocate(rt, 8);
    auto bias_pair = Allocate(rt, positions ? max_pairs * 16 : 16);
    auto bias_row = Allocate(rt, positions ? predictions.size() * 16 : 16);
    auto bias_segments = (n_groups + 127) / 128;
    auto bias_partial = Allocate(rt, positions ? positions * bias_segments * 16ull : 16);
    auto bias_sum = Allocate(rt, positions ? positions * 16ull : 16);
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
    auto bias_positive = ObjectiveSnapshot(rt, command, bias.positive.data(), bias.positive.size_bytes());
    auto bias_negative = ObjectiveSnapshot(rt, command, bias.negative.data(), bias.negative.size_bytes());
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
    auto order = EncodeGroupSort(rt, command, input.buffer, offsets, predictions.size(), 1, n_groups, max_group_size, input.offset);
    GroupOrder label_order{};
    if (mean) {
      auto ranked_labels = Allocate(rt, predictions.size_bytes());
      auto e = encoder("mean_labels");
      CHECK(e);
      [e setComputePipelineState:rt.ranking_mean_labels_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:response.buffer offset:response.offset atIndex:1];
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
      [e setBuffer:response.buffer offset:response.offset atIndex:1];
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
            [e setBuffer:response.buffer offset:response.offset atIndex:2];
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
          [e setBuffer:input.buffer offset:input.offset atIndex:1];
          [e setBuffer:response.buffer offset:response.offset atIndex:2];
          [e setBuffer:task_buffer offset:0 atIndex:3];
          [e setBuffer:discount.buffer offset:discount.offset atIndex:4];
          [e setBuffer:inverse.buffer offset:inverse.offset atIndex:5];
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
          [e setBuffer:weight.buffer offset:weight.offset atIndex:6];
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
          [e setBuffer:weight.buffer offset:weight.offset atIndex:4];
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
    // Preserve legacy gradient -> diagnostics -> positive -> negative ordering
    // for overlapping output/diagnostic spans, including mixed sharing eligibility.
    bool disjoint =
        ObjectiveOutputDisjoint(output, bias.positive, bias.negative, diagnostics.groups,
                                diagnostics.global_cost_rounding, diagnostics.map_prefix, mean_pair_trace) &&
        ObjectiveOutputDisjoint(bias.positive, bias.negative, diagnostics.groups,
                                diagnostics.global_cost_rounding, diagnostics.map_prefix, mean_pair_trace) &&
        ObjectiveOutputDisjoint(bias.negative, diagnostics.groups,
                                diagnostics.global_cost_rounding, diagnostics.map_prefix, mean_pair_trace);
    ObjectiveOutput gradient_out{rt, output.data(), output.size_bytes(), disjoint, result};
    ObjectiveOutput positive_out{rt, bias.positive.data(), bias.positive.size_bytes(), disjoint, bias_positive};
    ObjectiveOutput negative_out{rt, bias.negative.data(), bias.negative.size_bytes(), disjoint, bias_negative};
    gradient_out.Store(command, result);
    positive_out.Store(command, bias_positive);
    negative_out.Store(command, bias_negative);
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
    gradient_out.Publish();
    if (!diagnostics.groups.empty())
      std::memcpy(diagnostics.groups.data(), norms.contents, diagnostics.groups.size_bytes());
    if (!diagnostics.global_cost_rounding.empty())
      std::memcpy(diagnostics.global_cost_rounding.data(), cost_rounding.contents, sizeof(double));
    if (!diagnostics.map_prefix.empty())
      std::memcpy(diagnostics.map_prefix.data(), map_stats.contents,
                  diagnostics.map_prefix.size_bytes());
    if (trace) std::memcpy(mean_pair_trace.data(), trace.contents, mean_pair_trace.size_bytes());
    if (positions) {
      positive_out.Publish();
      negative_out.Publish();
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
                         Span<double const> inverse_idcg, std::shared_ptr<RankingPlan const> plan) {
  auto metadata = CheckGroupSort(predictions, groups, std::move(plan));
  auto max_group_size = metadata->MaxGroupSize();
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
    auto input = MetricInput(rt, predictions.Values().data(), predictions.Values().size_bytes());
    auto response = MetricInput(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = metadata->OffsetsBuffer();
    auto weight = MetricInput(rt, weights.data(), weights.size_bytes());
    auto discount = MetricInput(rt, discounts.data(), discounts.size_bytes(), alignof(double));
    auto inverse = MetricInput(rt, inverse_idcg.data(), inverse_idcg.size_bytes(), alignof(double));
    auto partial = Allocate(rt, n_groups * 2 * sizeof(double));
    auto output = Allocate(rt, 2 * sizeof(double));
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    GroupOrder order{Allocate(rt, 16), false};
    if (predictions.Size()) {
      order = EncodeGroupSort(rt, command, input.buffer, offsets, predictions.Size(),
                              predictions.Stride(0), n_groups, max_group_size, input.offset);
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
      [e setBuffer:response.buffer offset:response.offset atIndex:1];
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
    [e setBuffer:response.buffer offset:response.offset atIndex:1];
    [e setBuffer:offsets offset:0 atIndex:2];
    [e setBuffer:weight.buffer offset:weight.offset atIndex:3];
    [e setBuffer:discount.buffer offset:discount.offset atIndex:4];
    [e setBuffer:inverse.buffer offset:inverse.offset atIndex:5];
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
    metric_final_readback_bytes.fetch_add(sizeof(RankingSums), std::memory_order_relaxed);
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
  if (rows) CHECK_EQ(labels.Shape(1), multiclass ? 1 : targets);
  CHECK_EQ(output_count, targets);
  CHECK(weights.empty() || weights.size() == rows);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(targets, std::numeric_limits<std::uint32_t>::max());
  if (!targets) return;
  CHECK(output);
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = MetricInput(rt, predictions.Values().data(), predictions.Values().size_bytes());
    auto response = MetricInput(rt, labels.Values().data(), labels.Values().size_bytes());
    auto weight = MetricInput(rt, weights.data(), weights.size_bytes());
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
      auto sorted = EncodeColumnSort(rt, command, input.buffer, shape, true, input.offset);
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
      [e setBuffer:response.buffer offset:response.offset atIndex:1];
      [e setBuffer:weight.buffer offset:weight.offset atIndex:2];
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
        [e setBuffer:response.buffer offset:response.offset atIndex:1];
        [e setBuffer:weight.buffer offset:weight.offset atIndex:2];
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
    auto destination = PublishMetric(rt, command, result, output, targets * 3 * sizeof(double));
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
    FinishMetricPublication(destination, result, output, targets * 3 * sizeof(double));
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
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = MetricInput(rt, summaries.data(), summaries.size_bytes(), alignof(double));
    auto output = Allocate(rt, sizeof(double));
    std::uint64_t targets = summaries.size() / 3;
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.auc_normalize_pipeline];
    [e setBuffer:input.buffer offset:input.offset atIndex:0];
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
    auc_final_readback_bytes.fetch_add(sizeof(double), std::memory_order_relaxed);
    return *static_cast<double const*>(output.contents);
  }
}

RankingAUCSums RankingPRAUC(linalg::VectorView<float const> predictions,
                            linalg::VectorView<float const> labels, Span<bst_group_t const> groups,
                            Span<float const> weights, Span<RankingAUCSums> individual) {
  auto metadata = CheckGroupSort(predictions, groups);
  auto max_group_size = metadata->MaxGroupSize();
  auto n_groups = groups.size() - 1;
  CHECK_EQ(predictions.Size(), labels.Size());
  CHECK(weights.empty() || weights.size() == n_groups);
  CHECK(individual.empty() || individual.size() == n_groups);
  if (!n_groups) return {0, 0};
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto input = MetricInput(rt, predictions.Values().data(), predictions.Values().size_bytes());
    auto response = MetricInput(rt, labels.Values().data(), labels.Values().size_bytes());
    auto offsets = metadata->OffsetsBuffer();
    auto weight = MetricInput(rt, weights.data(), weights.size_bytes());
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
    [total_encoder setBuffer:response.buffer offset:response.offset atIndex:0];
    [total_encoder setBuffer:weight.buffer offset:weight.offset atIndex:1];
    [total_encoder setBuffer:offsets offset:0 atIndex:2];
    [total_encoder setBuffer:totals offset:0 atIndex:3];
    [total_encoder setBytes:&p length:sizeof(p) atIndex:4];
    [total_encoder dispatchThreadgroups:MTLSizeMake((n_groups + 7) / 8, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [total_encoder endEncoding];
    GroupOrder order{Allocate(rt, 16), false};
    id<MTLBuffer> prefix = Allocate(rt, 16);
    if (p.rows) {
      order = EncodeGroupSort(rt, command, input.buffer, offsets, p.rows, predictions.Stride(0), n_groups,
                              max_group_size, input.offset);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.ranking_pr_counts_pipeline];
      [e setBuffer:order.records offset:0 atIndex:0];
      [e setBuffer:input.buffer offset:input.offset atIndex:1];
      [e setBuffer:response.buffer offset:response.offset atIndex:2];
      [e setBuffer:weight.buffer offset:weight.offset atIndex:3];
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
    auc_final_readback_bytes.fetch_add(sizeof(RankingAUCSums), std::memory_order_relaxed);
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
  CHECK_LE(targets, std::numeric_limits<std::uint32_t>::max());
  CHECK(weights.empty() || weights.size() == rows);
  for (auto const& request : requests) {
    CHECK_LT(request.target, targets);
    CHECK_GE(request.alpha, 0);
    CHECK_LE(request.alpha, 1);
  }
  if (!rows) {
    FillInitialization(linalg::MakeVec(output.data(), output.size()), std::numeric_limits<float>::quiet_NaN());
    return;
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    SortShape shape{rows, values.Stride(0), values.Stride(1), static_cast<std::uint32_t>(targets),
                    static_cast<std::uint32_t>((rows + 255) / 256)};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto input = InitializationPacked(rt, command, values.Values().data(), values.Values().size_bytes());
    auto sorted = EncodeColumnSort(rt, command, input, shape);
    auto cdf = Allocate(rt, weights.empty() ? 4 : rows * targets * sizeof(std::uint32_t));
    if (!weights.empty()) {
      auto weight = InitializationInput(rt, weights.data(), weights.size_bytes());
      auto e = [command computeCommandEncoder];
      CHECK(e);
      auto n = static_cast<std::uint32_t>(rows);
      [e setComputePipelineState:rt.quantile_cdf_pipeline];
      [e setBuffer:sorted offset:0 atIndex:0];
      [e setBuffer:weight.buffer offset:weight.offset atIndex:1];
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
    CopyInitializationResult(rt, result, linalg::MakeVec(output.data(), output.size()));
    radix_sort_dispatches.fetch_add(1, std::memory_order_relaxed);
    quantile_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      quantile_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                     std::memory_order_relaxed);
  }
}

namespace {
void BinEntriesBuffer(id<MTLBuffer> input, std::size_t count, Span<FeatureType const> feature_types,
                      GHistIndexMatrix* output, std::size_t offset) {
  CHECK(output);
  CHECK_LE(offset, output->index.Size());
  CHECK_LE(count, output->index.Size() - offset);
  if (!count) return;
  static_assert(sizeof(Entry) == 8 && offsetof(Entry, fvalue) == 4);
  CHECK_LE(count, std::numeric_limits<std::uint32_t>::max());
  auto const& cuts = output->cut;
  CHECK(!cuts.Ptrs().empty());
  auto features = cuts.Ptrs().size() - 1;
  CHECK(feature_types.empty() || feature_types.size() == features);
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto ptrs = Upload(rt, cuts.Ptrs().data(), cuts.Ptrs().size() * sizeof(std::uint32_t));
    auto values = Upload(rt, cuts.Values().data(), cuts.Values().size() * sizeof(float));
    std::vector<std::uint8_t> is_cat(features, 0);
    for (std::size_t f = 0; f < feature_types.size(); ++f)
      is_cat[f] = feature_types[f] == FeatureType::kCategorical;
    auto categories = Upload(rt, is_cat.data(), is_cat.size());
    auto bytes = count * output->index.GetBinTypeSize();
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
    } p{count, static_cast<std::uint32_t>(features),
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
    [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    auto flags = *static_cast<std::uint32_t const*>(status.contents);
    CHECK(!(flags & 1)) << error::InfInData();
    CHECK_EQ(flags, 0) << "Metal bin assignment found a feature without cut points.";
    std::memcpy(output->index.data<std::uint8_t>() + offset * output->index.GetBinTypeSize(),
                bins.contents, bytes);
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
}  // namespace
void BinEntries(Span<Entry const> entries, Span<FeatureType const> types,
                GHistIndexMatrix* output) {
  CHECK_EQ(entries.size(), output->index.Size());
  if (entries.empty()) return;
  @autoreleasepool {
    BinEntriesBuffer(Upload(GetRuntime(), entries.data(), entries.size_bytes()), entries.size(),
                     types, output, 0);
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

void MakeSketchWeights(std::size_t rows, Span<float const> hessians, Span<float const> weights,
                        Span<bst_group_t const> groups, Span<float> result) {
  CHECK_EQ(result.size(), rows);
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
  if (!rows) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto h = BindGradientStateInput(rt, hessians.data(), hessians.size_bytes(), 4);
    auto w = BindGradientStateInput(rt, weights.data(), weights.size_bytes(), 4);
    auto g = BindGradientStateInput(rt, groups.data(), groups.size_bytes(), 4);
    auto destination = TryBorrowMemory(rt.device, result.data(), result.size_bytes(), 4);
    bool alias =
        MemoryRangesOverlap(hessians.data(), hessians.size_bytes(), result.data(), result.size_bytes()) ||
        MemoryRangesOverlap(weights.data(), weights.size_bytes(), result.data(), result.size_bytes()) ||
        MemoryRangesOverlap(groups.data(), groups.size_bytes(), result.data(), result.size_bytes());
    auto out = destination.buffer && !alias ? destination
                                           : BorrowedMemory{Allocate(rt, result.size_bytes()), 0};
    struct {
      std::uint32_t hessian, weighted, groups, padding;
    } p{!hessians.empty(), !weights.empty(), static_cast<std::uint32_t>(groups.size()), 0};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.sketch_weight_pipeline];
    [e setBuffer:h.buffer offset:h.offset atIndex:0];
    [e setBuffer:w.buffer offset:w.offset atIndex:1];
    [e setBuffer:g.buffer offset:g.offset atIndex:2];
    [e setBuffer:out.buffer offset:out.offset atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    if (destination.buffer && alias) {
      auto blit = [command blitCommandEncoder];
      CHECK(blit);
      [blit copyFromBuffer:out.buffer sourceOffset:out.offset toBuffer:destination.buffer
        destinationOffset:destination.offset size:result.size_bytes()];
      [blit endEncoding];
    }
    CompleteOperation(command, &sketch_weight_dispatches, &sketch_ingest_gpu_time_ns);
    if (!destination.buffer) std::memcpy(result.data(), out.buffer.contents, result.size_bytes());
    (destination.buffer ? gradient_state_shared_output_bytes : gradient_state_staging_bytes)
        .fetch_add(result.size_bytes(), std::memory_order_relaxed);
  }
}

std::vector<float> MakeSketchWeights(std::size_t rows, Span<float const> hessians,
                                     Span<float const> weights, Span<bst_group_t const> groups) {
  std::vector<float> result(rows);
  MakeSketchWeights(rows, hessians, weights, groups, result);
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
    PublishFeatureOutput(rt, {output, 0}, features.data(), features.size_bytes(),
                         &column_sampling_gpu_time_ns);
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
    auto input = BindFeatureInput(rt, features.data(), features.size_bytes());
    auto output = Allocate(rt, selected.size_bytes());
    auto weight = BindFeatureInput(rt, weights.data(), weights.size_bytes());
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
      // Publish only after validation. Scratch results make input/output aliases
      // safe without staging feature IDs or weights through CPU memory.
      PublishFeatureOutput(rt, {output, 0}, selected.data(), selected.size_bytes(),
                           &column_sampling_gpu_time_ns);
      PublishFeatureOutput(rt, {random, 0}, uniforms.data(), uniforms.size_bytes(),
                           &column_sampling_gpu_time_ns);
      PublishFeatureOutput(rt, {scores, 0}, keys.data(), keys.size_bytes(),
                           &column_sampling_gpu_time_ns);
      PublishFeatureOutput(rt, {indices, 0}, permutation.data(), permutation.size_bytes(),
                           &column_sampling_gpu_time_ns);
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
      [e setBuffer:input.buffer offset:input.offset atIndex:0];
      [e setBuffer:weight.buffer offset:weight.offset atIndex:1];
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
    [e setBuffer:input.buffer offset:input.offset atIndex:0];
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

namespace {
struct MetricReduction {
  id<MTLBuffer> result;
  std::uint64_t passes;
};
MetricReduction EncodeMetricReduction(Runtime& rt, id<MTLCommandBuffer> command,
                                      id<MTLBuffer> partial, std::uint64_t count) {
  CHECK_GT(count, 0);
  CHECK_EQ(rt.metric_sum_pipeline.threadExecutionWidth, 32);
  std::uint64_t passes = 0;
  while (count > 1) {
    auto next_count = (count + 255) / 256;
    auto next = Allocate(rt, next_count * 2 * sizeof(double));
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.metric_sum_pipeline];
    [e setBuffer:partial offset:0 atIndex:0];
    [e setBuffer:next offset:0 atIndex:1];
    [e setBytes:&count length:sizeof(count) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(next_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    partial = next;
    count = next_count;
    ++passes;
  }
  return {partial, passes};
}
std::array<double, 2> ReadMetricReduction(MetricReduction const& reduction) {
  std::array<double, 2> result;
  std::memcpy(result.data(), reduction.result.contents, sizeof(result));
  metric_reduction_passes.fetch_add(reduction.passes, std::memory_order_relaxed);
  metric_final_readback_bytes.fetch_add(sizeof(result), std::memory_order_relaxed);
  return result;
}
}  // namespace

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
    auto pred = MetricInput(rt, predictions.data(), predictions.size_bytes());
    auto label = MetricInput(rt, labels.data(), labels.size_bytes());
    auto weight = MetricInput(rt, weights.data(), weights.size_bytes());
    auto levels = MetricInput(rt, alpha.data(), alpha.size_bytes());
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
    [e setBuffer:pred.buffer offset:pred.offset atIndex:0];
    [e setBuffer:label.buffer offset:label.offset atIndex:1];
    [e setBuffer:weight.buffer offset:weight.offset atIndex:2];
    [e setBuffer:levels.buffer offset:levels.offset atIndex:3];
    [e setBuffer:partial offset:0 atIndex:4];
    [e setBytes:&p length:sizeof(p) atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    auto reduced = EncodeMetricReduction(rt, command, partial, blocks);
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      metric_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                   std::memory_order_relaxed);
    return ReadMetricReduction(reduced);
  }
}

namespace {
struct SurvivalCommand {
  id<MTLCommandBuffer> command;
  id<MTLBuffer> result;
  std::size_t blocks;
};
SurvivalCommand EncodeSurvival(bool metric, std::uint32_t distribution, float scale, bool accuracy,
                               Span<float const> predictions, Span<float const> lower,
                               Span<float const> upper, Span<float const> weights,
                               BorrowedMemory gradient_output = {}) {
  CHECK_LE(distribution, 2);
  CHECK_EQ(predictions.size(), lower.size());
  CHECK_EQ(predictions.size(), upper.size());
  CHECK(weights.empty() || weights.size() == predictions.size());
  CHECK_LE(predictions.size(), std::numeric_limits<std::uint32_t>::max());
  auto& rt = GetRuntime();
  auto input = [&](Span<float const> values) {
    return metric ? MetricInput(rt, values.data(), values.size_bytes())
                  : ObjectiveInput(rt, values.data(), values.size_bytes());
  };
  auto pred = input(predictions), lo = input(lower), hi = input(upper), w = input(weights);
  auto blocks = (predictions.size() + 255) / 256;
  auto result = metric ? Allocate(rt, blocks * 2 * sizeof(double)) : gradient_output.buffer;
  struct {
    std::uint64_t rows;
    std::uint32_t distribution, weighted, accuracy;
    float scale;
  } p{predictions.size(), distribution, !weights.empty(), accuracy, scale};
  static_assert(sizeof(p) == 24);
  auto command = [rt.queue commandBuffer];
  CHECK(command);
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:metric ? rt.aft_metric_pipeline : rt.aft_gradient_pipeline];
  [e setBuffer:pred.buffer offset:pred.offset atIndex:0];
  [e setBuffer:lo.buffer offset:lo.offset atIndex:1];
  [e setBuffer:hi.buffer offset:hi.offset atIndex:2];
  [e setBuffer:w.buffer offset:w.offset atIndex:3];
  [e setBuffer:result offset:(metric ? 0 : gradient_output.offset) atIndex:4];
  [e setBytes:&p length:sizeof(p) atIndex:5];
  [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  return {command, result, blocks};
}
}  // namespace
void SurvivalGradients(std::uint32_t distribution, float scale, Span<float const> predictions,
                       Span<float const> lower, Span<float const> upper, Span<float const> weights,
                       Span<GradientPair> output) {
  CHECK_EQ(predictions.size(), output.size());
  if (predictions.empty()) return;
  @autoreleasepool {
    auto& rt = GetRuntime();
    ObjectiveOutput out{rt, output.data(), output.size_bytes(),
                         ObjectiveOutputDisjoint(output, predictions, lower, upper, weights)};
    auto op = EncodeSurvival(false, distribution, scale, false, predictions, lower, upper, weights,
                              out.memory);
    CompleteOperation(op.command, &objective_dispatches);
    out.Publish();
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
    auto reduced = EncodeMetricReduction(GetRuntime(), op.command, op.result, op.blocks);
    [op.command commit];
    [op.command waitUntilCompleted];
    CHECK(op.command.status == MTLCommandBufferStatusCompleted) << ErrorText(op.command.error);
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    survival_metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = op.command.GPUEndTime - op.command.GPUStartTime;
    if (elapsed > 0)
      metric_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                   std::memory_order_relaxed);
    return ReadMetricReduction(reduced);
  }
}
void SurvivalTransform(bool to_margin, Span<float> values) {
  if (values.empty()) return;
  CHECK_LE(values.size(), std::numeric_limits<std::uint32_t>::max());
  @autoreleasepool {
    auto& rt = GetRuntime();
    PredictionOutput output{rt, values, true};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    std::uint32_t kind = to_margin;
    [e setComputePipelineState:rt.aft_transform_pipeline];
    [e setBuffer:output.memory.buffer offset:output.memory.offset atIndex:0];
    [e setBuffer:output.memory.buffer offset:output.memory.offset atIndex:1];
    [e setBytes:&kind length:sizeof(kind) atIndex:2];
    [e dispatchThreads:MTLSizeMake(values.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &transform_dispatches);
    output.Publish(values.size());
  }
}

void AdaptiveGradients(Context const* ctx, AdaptiveObjective objective,
                       Span<float const> predictions, Span<float const> labels,
                       Span<float const> weights, std::uint32_t targets, Span<float const> alpha,
                       Span<GradientPair> output) {
  CHECK_GT(targets, 0);
  CHECK_EQ(predictions.size() % targets, 0);
  CHECK_EQ(predictions.size(), output.size());
  CHECK_LE(predictions.size(), std::numeric_limits<std::uint32_t>::max());
  auto rows = predictions.size() / targets;

  CHECK(weights.empty() || weights.size() == rows);
  CHECK(labels.size() == rows || labels.size() == predictions.size());
  CHECK(ctx);
  if (objective == AdaptiveObjective::kQuantile) CHECK_EQ(alpha.size(), targets);
  struct {
    std::uint64_t rows;
    std::uint32_t targets, label_targets, weighted, kind, blocks, padding;
  } p{rows,
      targets,
      static_cast<std::uint32_t>(rows ? labels.size() / rows : 1),
      !weights.empty(),
      static_cast<std::uint32_t>(objective),
      static_cast<std::uint32_t>((rows + 255) / 256),
      0};
  @autoreleasepool {
    auto& rt = GetRuntime();
    CHECK_EQ(rt.residual_reduce_pipeline.threadExecutionWidth, 32);
    auto pred = ObjectiveInput(rt, predictions.data(), predictions.size_bytes());
    auto label = ObjectiveInput(rt, labels.data(), labels.size_bytes());
    auto weight = ObjectiveInput(rt, weights.data(), weights.size_bytes());
    auto levels = ObjectiveInput(rt, alpha.data(), alpha.size_bytes());
    std::vector<double> statistics(targets + 1, 0.);
    if (rows) {
      auto partial = Allocate(rt, std::size_t(p.blocks) * targets * 2 * sizeof(std::uint64_t));
      auto stats = Allocate(rt, statistics.size() * sizeof(double));
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.residual_reduce_pipeline];
      [e setBuffer:pred.buffer offset:pred.offset atIndex:0];
      [e setBuffer:label.buffer offset:label.offset atIndex:1];
      [e setBuffer:weight.buffer offset:weight.offset atIndex:2];
      [e setBuffer:partial offset:0 atIndex:3];
      [e setBytes:&p length:sizeof(p) atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake(p.blocks, targets, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      while (p.blocks > 1) {
        auto blocks = (p.blocks + 255) / 256;
        auto next = Allocate(rt, std::size_t(blocks) * targets * 2 * sizeof(std::uint64_t));
        e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt.residual_stats_pipeline];
        [e setBuffer:partial offset:0 atIndex:0];
        [e setBuffer:next offset:0 atIndex:1];
        [e setBytes:&p length:sizeof(p) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(blocks, targets, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        partial = next;
        p.blocks = blocks;
      }
      e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.residual_finish_pipeline];
      [e setBuffer:partial offset:0 atIndex:0];
      [e setBuffer:stats offset:0 atIndex:1];
      [e setBytes:&p length:sizeof(p) atIndex:2];
      [e dispatchThreads:MTLSizeMake(targets, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &residual_scale_dispatches);
      std::memcpy(statistics.data(), stats.contents, statistics.size() * sizeof(double));
    }
    // Upstream CUDA also communicates these small statistics through a CPU
    // context. Empty workers must participate before returning.
    auto cpu_ctx = ctx->MakeCPU();
    collective::SafeColl(
        collective::GlobalSum(&cpu_ctx, linalg::MakeVec(statistics.data(), statistics.size())));
    if (!rows) return;
    auto stats = Upload(rt, statistics.data(), statistics.size() * sizeof(double));
    auto scale = Allocate(rt, targets * sizeof(float));
    ObjectiveOutput out{rt, output.data(), output.size_bytes(),
                         ObjectiveOutputDisjoint(output, predictions, labels, weights, alpha)};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.residual_apply_pipeline];
    [e setBuffer:stats offset:0 atIndex:0];
    [e setBuffer:scale offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreads:MTLSizeMake(targets, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [e endEncoding];
    e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.adaptive_pipeline];
    [e setBuffer:pred.buffer offset:pred.offset atIndex:0];
    [e setBuffer:label.buffer offset:label.offset atIndex:1];
    [e setBuffer:weight.buffer offset:weight.offset atIndex:2];
    [e setBuffer:scale offset:0 atIndex:3];
    [e setBuffer:levels.buffer offset:levels.offset atIndex:4];
    [e setBuffer:out.memory.buffer offset:out.memory.offset atIndex:5];
    [e setBytes:&p length:sizeof(p) atIndex:6];
    [e dispatchThreads:MTLSizeMake(predictions.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &objective_dispatches);
    out.Publish();
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
  if (objective == Objective::kSoftmax) CHECK_EQ(labels.size(), rows);
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
    bool multiclass = objective == Objective::kSoftmax;
    [encoder setComputePipelineState:multiclass ? rt.multiclass_objective_pipeline
                                                : rt.objective_pipeline];
    [encoder setBuffer:pred.buffer offset:pred.offset atIndex:0];
    [encoder setBuffer:label.buffer offset:label.offset atIndex:1];
    [encoder setBuffer:weight.buffer offset:weight.offset atIndex:2];
    [encoder setBuffer:levels.buffer offset:levels.offset atIndex:3];
    [encoder setBuffer:out.buffer offset:out.offset atIndex:4];
    [encoder setBytes:&p length:sizeof(p) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(multiclass ? rows : predictions.size(), 1, 1)
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
    auto input = ObjectiveInput(rt, labels.Values().data(), labels.Values().size_bytes());
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
    [e setBuffer:input.buffer offset:input.offset atIndex:0];
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
    auto input = ObjectiveInput(rt, values.data(), values.size_bytes());
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
    [e setBuffer:input.buffer offset:input.offset atIndex:0];
    [e setBuffer:output offset:0 atIndex:1];
    [e setBuffer:invalid offset:0 atIndex:2];
    [e setBytes:&kind length:sizeof(kind) atIndex:3];
    [e dispatchThreads:MTLSizeMake(values.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &base_score_dispatches, &base_score_gpu_time_ns);
    if (*static_cast<std::uint32_t const*>(invalid.contents)) return false;
    // Commit the result only after validation succeeds. A GPU blit preserves
    // the no-write-on-invalid contract without reading payloads back to the CPU.
    auto target = TryBorrowMemory(rt.device, values.data(), values.size_bytes(), 4);
    if (target.buffer) {
      auto publish = [rt.queue commandBuffer];
      auto blit = [publish blitCommandEncoder];
      [blit copyFromBuffer:output sourceOffset:0 toBuffer:target.buffer
        destinationOffset:target.offset size:values.size_bytes()];
      [blit endEncoding];
      CompleteOperation(publish, nullptr, &base_score_gpu_time_ns);
      objective_shared_output_bytes.fetch_add(values.size_bytes(), std::memory_order_relaxed);
    } else {
      std::memcpy(values.data(), output.contents, values.size_bytes());
      objective_staging_bytes.fetch_add(values.size_bytes(), std::memory_order_relaxed);
    }
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
    auto label = InitializationInput(rt, labels.data(), labels.size_bytes());
    auto weight = InitializationInput(rt, weights.data(), weights.size_bytes());
    auto levels = InitializationInput(rt, alpha.data(), alpha.size_bytes());
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
    [e setBuffer:label.buffer offset:label.offset atIndex:0];
    [e setBuffer:weight.buffer offset:weight.offset atIndex:1];
    [e setBuffer:levels.buffer offset:levels.offset atIndex:2];
    [e setBuffer:out offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(output.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &expectile_init_dispatches, &initialization_gpu_time_ns);
    CopyInitializationResult(rt, out, linalg::MakeVec(output.data(), output.size()));
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
    auto input = InitializationInput(rt, targets.data(), targets.size_bytes());
    auto out = Allocate(rt, output.size_bytes());
    auto n = static_cast<std::uint32_t>(targets.size());
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.broadcast_targets_pipeline];
    [e setBuffer:input.buffer offset:input.offset atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    [e setBytes:&n length:sizeof(n) atIndex:2];
    [e dispatchThreads:MTLSizeMake(output.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &prediction_init_dispatches, &initialization_gpu_time_ns);
    CopyInitializationResult(rt, out, linalg::MakeVec(output.data(), output.size()));
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
    PredictionOutput out{rt, values, true};
    auto in = out.memory;
    // Argmax compacts rows into the prefix and would race with later rows if
    // input and output aliased. Snapshot on GPU before dispatching that kernel.
    if (transform == Transform::kArgmax) {
      in = {Allocate(rt, values.size_bytes()), 0};
      auto blit = [command blitCommandEncoder];
      [blit copyFromBuffer:out.memory.buffer sourceOffset:out.memory.offset toBuffer:in.buffer
        destinationOffset:0 size:values.size_bytes()];
      [blit endEncoding];
      prediction_device_copy_bytes.fetch_add(values.size_bytes(), std::memory_order_relaxed);
    }
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    auto kind = static_cast<std::uint32_t>(transform);
    auto n = kind >= 4 ? values.size() / targets : values.size();
    struct {
      std::uint64_t n;
      std::uint32_t targets, kind;
    } p{n, targets, kind};
    bool multiclass = transform == Transform::kSoftmax || transform == Transform::kArgmax;
    [encoder setComputePipelineState:multiclass ? rt.multiclass_transform_pipeline
                                                : rt.transform_pipeline];
    [encoder setBuffer:in.buffer offset:in.offset atIndex:0];
    [encoder setBuffer:out.memory.buffer offset:out.memory.offset atIndex:1];
    [encoder setBytes:&p length:sizeof(p) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    CompleteOperation(command, &transform_dispatches);
    auto count = transform == Transform::kArgmax ? n : values.size();
    out.Publish(count);
  }
}

namespace {
class MetalInputBatch final : public InputBatch {
 public:
  id<MTLBuffer> values;
  std::size_t rows;
  std::uint32_t features;
  MetalInputBatch(id<MTLBuffer> values, std::size_t rows, std::uint32_t features)
      : values{values}, rows{rows}, features{features} {}
  std::size_t Rows() const override { return rows; }
  std::uint32_t Features() const override { return features; }
  void Readback(Span<float> output) const override {
    CHECK_EQ(output.size(), rows * features);
    if (!output.empty()) std::memcpy(output.data(), values.contents, output.size_bytes());
    input_debug_readback_bytes.fetch_add(output.size_bytes(), std::memory_order_relaxed);
  }
};
struct InputArray {
  std::uint64_t offset{0}, mask{0};
  std::int64_t row_stride{0}, column_stride{0};
  std::uint32_t type{0}, mask_bit{0}, masked{0}, padding{0};
};
struct InputParams {
  std::uint64_t rows, first_entry{0}, entries{0};
  std::uint32_t features, kind, dense{0}, width{0}, missing{0x7fc00000u}, mapped{0}, input_features,
      padding{0};
};
static_assert(sizeof(InputArray) == 48 && sizeof(InputParams) == 56);
class MetalFeatureInput : public Tensor {
 public:
  virtual id<MTLBuffer> Buffer() const = 0;
  virtual std::uint32_t Kind() const = 0;
  virtual std::vector<InputArray> Arrays(std::size_t begin) const = 0;
};
class MetalTensor final : public MetalFeatureInput {
 public:
  id<MTLBuffer> buffer;
  InputArray array;
  bst_idx_t rows;
  bst_feature_t columns;
  explicit MetalTensor(StringView descriptor, bool allow_tensor_shape = false) {
    @autoreleasepool {
      auto json = Json::Load(descriptor);
      CHECK_EQ(get<Integer const>(json["device_type"]), 8);
      CHECK_EQ(get<Integer const>(json["device_id"]), 0);
      auto const& shape = get<Array const>(json["shape"]);
      auto const& strides = get<Array const>(json["strides"]);
      CHECK((allow_tensor_shape && shape.size() < 2) || shape.size() == 2)
          << "Metal feature input must be two-dimensional; other tensors may also be scalar/vector.";
      CHECK_EQ(strides.size(), shape.size());
      auto r = shape.empty() ? std::int64_t{1} : get<Integer const>(shape[0]);
      auto c = shape.size() < 2 ? std::int64_t{1} : get<Integer const>(shape[1]);
      CHECK_GE(r, 0);
      CHECK_LE(r, std::numeric_limits<std::uint32_t>::max());
      CHECK_GE(c, 0);
      CHECK_LE(c, std::numeric_limits<bst_feature_t>::max());
      rows = r;
      columns = c;
      auto code = get<Integer const>(json["dtype_code"]);
      auto bits = get<Integer const>(json["dtype_bits"]);
      CHECK_EQ(get<Integer const>(json["dtype_lanes"]), 1);
      if (code == 2 && (bits == 16 || bits == 32 || bits == 64)) {
        array.type = bits == 16 ? 0 : bits == 32 ? 1 : 2;
      } else if ((code == 0 || code == 1 || code == 6) &&
                 (bits == 8 || bits == 16 || bits == 32 || bits == 64)) {
        CHECK(code != 6 || bits == 8);
        array.type = (code == 0 ? 4 : 8) + (bits == 8 ? 0 : bits == 16 ? 1 : bits == 32 ? 2 : 3);
      } else {
        LOG(FATAL) << "Unsupported Metal tensor dtype (expected float16/32/64, integer or bool).";
      }
      using Wide = __int128;
      auto item = bits / 8;
      Wide rs = shape.empty() ? item : Wide(get<Integer const>(strides[0])) * item;
      Wide cs = shape.size() < 2 ? item : Wide(get<Integer const>(strides[1])) * item;
      CHECK(rs >= std::numeric_limits<std::int64_t>::min() && rs <= std::numeric_limits<std::int64_t>::max());
      CHECK(cs >= std::numeric_limits<std::int64_t>::min() && cs <= std::numeric_limits<std::int64_t>::max());
      array.row_stride = rs;
      array.column_stride = cs;
      auto offset = get<Integer const>(json["byte_offset"]);
      auto handle = get<Integer const>(json["buffer"]);
      CHECK_GE(offset, 0);
      CHECK_GE(handle, 0);
      CHECK_EQ(offset % item, 0);
      auto& rt = GetRuntime();
      if (!rows || !columns) {
        buffer = Allocate(rt, 0);
      } else {
        CHECK_NE(handle, 0) << "Nonempty Metal tensor has a null buffer.";
        // The caller supplies a valid producer-owned MTLBuffer. As with CUDA
        // array pointers, arbitrary fabricated handles cannot be introspected.
        id<MTLBuffer> source = (__bridge id<MTLBuffer>)(reinterpret_cast<void*>(handle));
        CHECK_EQ(source.device.registryID, rt.device.registryID);
        Wide dr = Wide(rows - 1) * rs, dc = Wide(columns - 1) * cs;
        Wide low = Wide(offset) + std::min<Wide>(0, dr) + std::min<Wide>(0, dc);
        Wide high = Wide(offset) + std::max<Wide>(0, dr) + std::max<Wide>(0, dc) + item;
        CHECK(low >= 0 && high <= Wide(source.length)) << "Metal tensor view exceeds its buffer.";
        auto bytes = std::size_t(high - low);
        CHECK_LE(bytes, rt.device.maxBufferLength);
        buffer = [rt.device newBufferWithLength:std::max<std::size_t>(16, bytes)
                                        options:MTLResourceStorageModePrivate];
        CHECK(buffer);
        array.offset = std::uint64_t(Wide(offset) - low);
        auto command = [rt.queue commandBuffer];
        auto blit = [command blitCommandEncoder];
        [blit copyFromBuffer:source sourceOffset:std::size_t(low) toBuffer:buffer
          destinationOffset:0 size:bytes];
        [blit endEncoding];
        CompleteOperation(command, &tensor_copy_dispatches, &tensor_gpu_time_ns);
        tensor_copied_bytes.fetch_add(bytes, std::memory_order_relaxed);
      }
      tensor_imports.fetch_add(1, std::memory_order_relaxed);
    }
  }
  bst_idx_t NumRows() const override { return rows; }
  bst_feature_t NumColumns() const override { return columns; }
  id<MTLBuffer> Buffer() const override { return buffer; }
  std::uint32_t Kind() const override { return 1; }
  std::vector<InputArray> Arrays(std::size_t begin) const override { return {Window(begin)}; }
  InputArray Window(std::size_t begin) const {
    auto out = array;
    // Import validated the complete signed-stride range against the buffer.
    out.offset = std::uint64_t(__int128(array.offset) + __int128(begin) * array.row_stride);
    return out;
  }
};
// Column descriptors are metadata; only GPU commands access their payloads.
std::unique_ptr<MetalTensor> NativeVector(Json const& json) {
  CHECK_EQ(get<Array const>(json["shape"]).size(), 1) << "Metal columns must be vectors.";
  return std::make_unique<MetalTensor>(Json::Dump(json), true);
}
template <typename T>
Span<T const> NativeNames(MetalTensor const& tensor,
                           std::vector<std::unique_ptr<VectorBuffer>>* owners) {
  auto& rt = GetRuntime();
  CHECK_LE(tensor.rows, rt.device.maxBufferLength / sizeof(T));
  if (!tensor.rows) return {};
  auto owner = std::make_unique<MetalVectorBuffer>();
  owner->Resize(tensor.rows * sizeof(T));
  auto output = Span<T const>{static_cast<T const*>(owner->Data()), tensor.rows};
  auto out = owner->Buffer();
  std::uint32_t p[]{static_cast<std::uint32_t>(output.size()), sizeof(T)};
  auto command = [rt.queue commandBuffer];
  auto e = [command computeCommandEncoder];
  [e setComputePipelineState:rt.tensor_gather_bytes_pipeline];
  [e setBuffer:tensor.buffer offset:0 atIndex:0];
  [e setBuffer:out offset:0 atIndex:1];
  [e setBytes:&tensor.array length:sizeof(tensor.array) atIndex:2];
  [e setBytes:p length:sizeof(p) atIndex:3];
  [e dispatchThreads:MTLSizeMake(output.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  CompleteOperation(command, &category_import_dispatches, &columnar_gpu_time_ns);
  category_import_shared_bytes.fetch_add(output.size_bytes(), std::memory_order_relaxed);
  owners->push_back(std::move(owner));
  return output;
}
enc::HostCatIndexView NativeCategory(Json const& names,
                                   std::vector<std::unique_ptr<VectorBuffer>>* owners) {
  auto const& obj = get<Object const>(names);
  if (obj.find("offsets") != obj.cend()) {
    auto offsets = NativeVector(names["offsets"]), values = NativeVector(names["values"]);
    CHECK_EQ(offsets->array.type, 6) << "String category offsets must be int32.";
    CHECK_EQ(values->array.type, 4) << "String category values must be int8.";
    CHECK_GT(offsets->rows, 1) << "Categorical feature must have at least one category.";
    CHECK_LE(offsets->rows - 1, std::numeric_limits<std::int32_t>::max());
    enc::CatStrArrayView out{NativeNames<std::int32_t>(*offsets, owners),
                             NativeNames<std::int8_t>(*values, owners)};
    CHECK_GE(out.offsets.front(), 0) << "Invalid negative string offset from category index.";
    CHECK(std::is_sorted(out.offsets.begin(), out.offsets.end())) << "Invalid string offsets.";
    CHECK_LE(out.offsets.back(), out.values.size()) << "String category offset exceeds values.";
    return out;
  }
  auto tensor = NativeVector(names);
  auto code = get<Integer const>(names["dtype_code"]);
  CHECK(code == 0 || code == 1) << error::NoFloatCat();
  data::CheckNonEmptyCategory(tensor->rows);
  CHECK_LE(tensor->rows, std::numeric_limits<std::int32_t>::max());
  switch (tensor->array.type) {
    case 4: return NativeNames<std::int8_t>(*tensor, owners);
    case 5: return NativeNames<std::int16_t>(*tensor, owners);
    case 6: return NativeNames<std::int32_t>(*tensor, owners);
    case 7: return NativeNames<std::int64_t>(*tensor, owners);
    case 8: return NativeNames<std::uint8_t>(*tensor, owners);
    case 9: return NativeNames<std::uint16_t>(*tensor, owners);
    case 10: return NativeNames<std::uint32_t>(*tensor, owners);
    case 11: return NativeNames<std::uint64_t>(*tensor, owners);
    default: LOG(FATAL) << "Unsupported category index type.";
  }
  return {};
}
class MetalColumnar final : public MetalFeatureInput {
  id<MTLBuffer> buffer_;
  bst_idx_t rows_{0};
  std::vector<InputArray> arrays_;
  std::unique_ptr<CategoryDictionary> categories_;
  std::vector<std::int32_t> segments_{0};
  std::shared_ptr<CatContainer> reference_;

 public:
  explicit MetalColumnar(Json const& json) {
    @autoreleasepool {
      auto const& columns = get<Array const>(json["columns"]);
      CHECK_GT(columns.size(), 0) << "The number of columns must not equal to 0.";
      CHECK_LE(columns.size(), std::numeric_limits<bst_feature_t>::max());
      std::vector<std::unique_ptr<MetalTensor>> sources;
      std::vector<std::unique_ptr<VectorBuffer>> names;
      std::vector<enc::HostCatIndexView> categories;
      std::vector<std::size_t> destinations;
      std::size_t length{0};
      auto& rt = GetRuntime();
      auto append = [&](std::unique_ptr<MetalTensor> source) {
        auto offset = (length + 7) & ~std::size_t{7};
        CHECK_LE(source->buffer.length, rt.device.maxBufferLength);
        CHECK_LE(offset, rt.device.maxBufferLength - source->buffer.length);
        length = offset + source->buffer.length;
        destinations.push_back(offset);
        sources.push_back(std::move(source));
        return offset;
      };
      for (auto const& col : columns) {
        auto values = NativeVector(col["values"]);
        if (arrays_.empty()) rows_ = values->rows;
        CHECK_EQ(values->rows, rows_) << "All columns should have the same number of rows.";
        auto a = values->array;
        a.offset += append(std::move(values));
        auto const& obj = get<Object const>(col);
        if (obj.find("mask") != obj.cend()) {
          auto mask = NativeVector(col["mask"]);
          CHECK_EQ(mask->array.type, 8) << "Packed validity mask must use uint8 storage.";
          CHECK_EQ(get<Integer const>(col["mask"]["dtype_code"]), 1);
          CHECK_EQ(mask->array.row_stride, 1) << "Packed validity mask must be contiguous.";
          auto bit = get<Integer const>(col["mask_bit"]);
          CHECK_GE(bit, 0);
          CHECK_LT(bit, 8);
          CHECK_GE(mask->rows, (rows_ + bit + 7) / 8) << "Packed validity mask is too short.";
          a.masked = 1;
          a.mask_bit = bit;
          a.mask = mask->array.offset;
          a.mask += append(std::move(mask));
        }
        arrays_.push_back(a);
        if (obj.find("categories") != obj.cend()) {
          categories.push_back(NativeCategory(col["categories"], &names));
        } else {
          categories.emplace_back();
        }
        auto count = std::visit([](auto const& c) { return c.size(); }, categories.back());
        segments_.push_back(data::AddCatCount(count, segments_.back()));
      }
      categories_ = MakeCategoryDictionary({categories, segments_, segments_.back()});
      auto const& obj = get<Object const>(json);
      if (obj.find("ref_categories") != obj.cend()) {
        auto handle = get<Integer const>(json["ref_categories"]);
        CHECK_GT(handle, 0);
        auto cats = reinterpret_cast<CatContainer const*>(handle);
        Context ctx;
        ctx.Init({{"device", "mps"}});
        reference_ = std::make_shared<CatContainer>(&ctx, cats->MetalView().View(), true);
        if (reference_->HasCategorical()) {
          CHECK_EQ(reference_->NumFeatures(), columns.size())
              << "Invalid reference categories, different number of columns";
        }
      }
      buffer_ = [rt.device newBufferWithLength:std::max<std::size_t>(16, length)
                                       options:MTLResourceStorageModePrivate];
      CHECK(buffer_);
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      for (std::size_t i = 0; i < sources.size(); ++i) {
        [blit copyFromBuffer:sources[i]->buffer sourceOffset:0 toBuffer:buffer_
          destinationOffset:destinations[i] size:sources[i]->buffer.length];
      }
      [blit endEncoding];
      CompleteOperation(command, &columnar_imports, &columnar_gpu_time_ns);
      columnar_pack_bytes.fetch_add(length, std::memory_order_relaxed);
    }
  }
  bst_idx_t NumRows() const override { return rows_; }
  bst_feature_t NumColumns() const override { return arrays_.size(); }
  id<MTLBuffer> Buffer() const override { return buffer_; }
  std::uint32_t Kind() const override { return 2; }
  std::vector<InputArray> Arrays(std::size_t begin) const override {
    auto result = arrays_;
    for (auto& a : result) {
      a.offset = std::uint64_t(__int128(a.offset) + __int128(begin) * a.row_stride);
      if (a.masked) {
        auto bit = begin + a.mask_bit;
        a.mask += bit / 8;
        a.mask_bit = bit % 8;
      }
    }
    return result;
  }
  enc::HostColumnsView Cats() const override {
    return categories_->View();
  }
  enc::HostColumnsView RefCats() const override {
    return reference_ ? reference_->MetalView().View() : enc::HostColumnsView{};
  }
};
struct PackedInput {
  std::vector<std::uint8_t> bytes;
  id<MTLBuffer> native{nil};
  std::vector<InputArray> arrays;
  InputParams params;
  PackedInput(std::size_t rows, std::uint32_t features, std::uint32_t kind, enc::MappingView map)
      : params{rows, 0, 0, features, kind, 0, 0, 0x7fc00000u, !map.mapping.empty(), features, 0} {
    CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
    CHECK(!rows || features <= std::numeric_limits<std::size_t>::max() / rows / 4);
  }
  std::uint64_t Append(void const* ptr, std::size_t n) {
    auto offset = (bytes.size() + 7) & ~std::size_t{7};
    CHECK_LE(n, GetRuntime().device.maxBufferLength);
    CHECK_LE(offset, GetRuntime().device.maxBufferLength - n);
    // Only alignment padding needs zero initialization; append the owned payload
    // directly instead of zero-filling it before overwriting it with a copy.
    bytes.resize(offset);
    if (n) {
      auto first = static_cast<std::uint8_t const*>(ptr);
      bytes.insert(bytes.end(), first, first + n);
    }
    input_staging_bytes.fetch_add(n, std::memory_order_relaxed);
    return offset;
  }
  InputArray Raw(void const* ptr, std::size_t bytes_count, std::uint32_t type,
                 std::int64_t stride) {
    return {Append(ptr, bytes_count), 0, stride, 0, type, 0, 0, 0};
  }
  template <int D, bool M>
  InputArray Window(ArrayInterface<D, M> const& array, std::size_t begin, std::size_t rows,
                    std::size_t columns = 1) {
    auto type = static_cast<unsigned>(array.type);
    CHECK_LT(type, 12);
    CHECK(type != 0 && type != 3)
        << "Metal prediction input requires float32/float64 or an integer array.";
    auto size = array.ElementSize();
    InputArray out;
    out.type = type;
    // ArrayInterface stores element strides unsigned. Reconstruct signed byte
    // strides so reversed views remain bounded windows instead of huge copies.
    out.row_stride = static_cast<std::int64_t>(array.strides[0] * size);
    if constexpr (D == 2) out.column_stride = static_cast<std::int64_t>(array.strides[1] * size);
    if (rows && columns) {
      using Wide = __int128;
      Wide start = Wide(begin) * out.row_stride;
      Wide r = Wide(rows - 1) * out.row_stride, c = Wide(columns - 1) * out.column_stride;
      Wide low = std::min<Wide>(0, r) + std::min<Wide>(0, c),
           high = std::max<Wide>(0, r) + std::max<Wide>(0, c);
      Wide length = high - low + size,
           address = Wide(reinterpret_cast<std::uintptr_t>(array.data)) + start + low;
      CHECK(address > 0);
      CHECK(address + length <= Wide(std::numeric_limits<std::uintptr_t>::max()));
      CHECK(length <= Wide(GetRuntime().device.maxBufferLength));
      out.offset = Append(reinterpret_cast<void const*>(static_cast<std::uintptr_t>(address)),
                          std::size_t(length)) -
                   std::int64_t(low);
    }
    if (array.valid.Data() && rows) {
      out.masked = 1;
      out.mask_bit = begin % 8;
      out.mask = Append(array.valid.Data() + begin / 8, (out.mask_bit + rows + 7) / 8);
    }
    return out;
  }
  std::unique_ptr<AdapterInput> FinishAdapter(enc::MappingView map, bst_idx_t base_row,
                                              bst_idx_t sketch_rows);
  std::unique_ptr<InputBatch> Finish(enc::MappingView map, Span<std::uint32_t const> cuts = {},
                                     Span<float const> cut_values = {},
                                     Span<FeatureType const> types = {},
                                     Page const* compressed = nullptr) {
    auto& rt = GetRuntime();
    @autoreleasepool {
      auto output = Allocate(rt, params.rows * params.features * 4);
      auto result = std::make_unique<MetalInputBatch>(output, params.rows, params.features);
      if (!params.rows || !params.features) return result;
      CHECK_LE(params.input_features, params.features);
      if (params.mapped) CHECK_EQ(map.offsets.size(), std::size_t(params.features) + 1);
      CHECK(types.empty() || types.size() >= params.input_features);
      auto raw = compressed ? compressed->index : native ? native : Upload(rt, bytes.data(), bytes.size());
      auto descriptors = compressed ? compressed->tail
                                    : Upload(rt, arrays.data(), arrays.size() * sizeof(InputArray));
      auto ptrs = compressed ? compressed->offsets : Upload(rt, cuts.data(), cuts.size_bytes());
      auto values =
          compressed ? compressed->cuts : Upload(rt, cut_values.data(), cut_values.size_bytes());
      std::vector<FeatureType> numerical(types.empty() ? params.input_features : 0,
                                         FeatureType::kNumerical);
      if (types.empty()) types = numerical;
      auto ft = Upload(rt, types.data(), types.size_bytes());
      auto segments = BindCategoryInput(rt, map.offsets.data(), map.offsets.size_bytes(), 4),
           mapping = BindCategoryInput(rt, map.mapping.data(), map.mapping.size_bytes(), 4);
      std::uint32_t zero = 0;
      auto error = Upload(rt, &zero, 4);
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:compressed ? rt.ellpack_input_pipeline
                                            : rt.input_prepare_pipeline];
      [e setBuffer:raw offset:0 atIndex:0];
      [e setBuffer:descriptors offset:0 atIndex:1];
      [e setBuffer:ptrs offset:0 atIndex:2];
      [e setBuffer:values offset:0 atIndex:3];
      [e setBuffer:ft offset:0 atIndex:4];
      [e setBuffer:segments.buffer offset:segments.offset atIndex:5];
      [e setBuffer:mapping.buffer offset:mapping.offset atIndex:6];
      [e setBuffer:output offset:0 atIndex:7];
      [e setBuffer:error offset:0 atIndex:8];
      [e setBytes:&params length:sizeof(params) atIndex:9];
      if (compressed)
        [e setBytes:&compressed->compressed length:sizeof(compressed->compressed) atIndex:10];
      [e dispatchThreads:MTLSizeMake(params.rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &input_prepare_dispatches, &input_prepare_gpu_time_ns);
      if (compressed) ellpack_input_dispatches.fetch_add(1, std::memory_order_relaxed);
      if (params.mapped) input_recode_dispatches.fetch_add(1, std::memory_order_relaxed);
      CHECK_EQ(*static_cast<std::uint32_t const*>(error.contents), 0)
          << "Invalid sparse prediction row or feature index.";
      return result;
    }
  }
};
void InputBounds(std::size_t total, std::size_t begin, std::size_t rows) {
  CHECK_LE(begin, total);
  CHECK_LE(rows, total - begin);
}
}  // namespace
std::unique_ptr<InputBatch> PrepareInput(HostSparsePageView page, std::size_t begin,
                                         std::size_t rows, std::uint32_t features,
                                         enc::MappingView map) {
  InputBounds(page.Size(), begin, rows);
  PackedInput p{rows, features, 0, map};
  if (!rows) return p.Finish(map);
  auto first = page.offset[begin], last = page.offset[begin + rows];
  CHECK_LE(first, last);
  CHECK_LE(last, page.data.size());
  p.params.first_entry = first;
  p.params.entries = last - first;
  p.arrays.push_back(p.Raw(page.offset.data() + begin, (rows + 1) * 8, 11, 8));
  auto offset = p.Append(page.data.data() + first, (last - first) * sizeof(Entry));
  static_assert(sizeof(Entry) == 8 && sizeof(bst_idx_t) == 8);
  p.arrays.push_back({offset, 0, 8, 0, 10, 0, 0, 0});
  p.arrays.push_back({offset + 4, 0, 8, 0, 1, 0, 0, 0});
  return p.Finish(map);
}
std::unique_ptr<InputBatch> PrepareInput(GHistIndexMatrix const& page,
                                         Span<FeatureType const> types, std::size_t begin,
                                         std::size_t rows, std::uint32_t features,
                                         enc::MappingView map) {
  InputBounds(page.Size(), begin, rows);
  PackedInput p{rows, features, 4, map};
  CHECK_LE(page.Features(), features);
  p.params.input_features = page.Features();
  p.params.dense = page.IsDense();
  if (!rows) return p.Finish(map);
  if (auto cached = std::atomic_load(&page.metal_page)) {
    auto device = std::static_pointer_cast<Page>(cached);
    if (device->IsCompressed()) {
      p.params.first_entry = begin;
      return p.Finish(map, {}, {}, types, device.get());
    }
  }
  page.RequireHost();
  auto first = page.row_ptr[begin], last = page.row_ptr[begin + rows];
  CHECK_LE(first, last);
  CHECK_LE(last, page.index.Size());
  p.params.first_entry = first;
  p.params.entries = last - first;
  p.arrays.push_back(p.Raw(page.row_ptr.data() + begin, (rows + 1) * 8, 11, 8));
  auto width = static_cast<unsigned>(page.index.GetBinTypeSize());
  p.arrays.push_back(p.Raw(page.index.data<std::uint8_t>() + first * width, (last - first) * width,
                           width == 1   ? 8
                           : width == 2 ? 9
                                        : 10,
                           width));
  return p.Finish(map, page.cut.Ptrs(), page.cut.Values(), types);
}
std::unique_ptr<InputBatch> PrepareInput(data::DenseAdapter const& adapter, float missing,
                                         std::size_t begin, std::size_t rows,
                                         std::uint32_t features, enc::MappingView map) {
  InputBounds(adapter.NumRows(), begin, rows);
  PackedInput p{rows, features, 1, map};
  auto columns = adapter.NumColumns();
  CHECK_LE(columns, features);
  p.params.input_features = columns;
  std::memcpy(&p.params.missing, &missing, 4);
  auto ptr = adapter.Value().Values();
  if (rows && columns) ptr += begin * columns;
  auto array = p.Raw(ptr, rows * columns * 4, 1, columns * 4);
  array.column_stride = 4;
  p.arrays.push_back(array);
  return p.Finish(map);
}
std::unique_ptr<InputBatch> PrepareInput(data::ArrayAdapter const& adapter, float missing,
                                         std::size_t begin, std::size_t rows,
                                         std::uint32_t features, enc::MappingView map) {
  InputBounds(adapter.NumRows(), begin, rows);
  PackedInput p{rows, features, 1, map};
  CHECK_LE(adapter.NumColumns(), features);
  p.params.input_features = adapter.NumColumns();
  std::memcpy(&p.params.missing, &missing, 4);
  p.arrays.push_back(p.Window(adapter.Array(), begin, rows, adapter.NumColumns()));
  return p.Finish(map);
}
std::unique_ptr<InputBatch> PrepareInput(data::ColumnarAdapter const& adapter, float missing,
                                         std::size_t begin, std::size_t rows,
                                         std::uint32_t features, enc::MappingView map) {
  InputBounds(adapter.NumRows(), begin, rows);
  PackedInput p{rows, features, 2, map};
  CHECK_LE(adapter.NumColumns(), features);
  p.params.input_features = adapter.NumColumns();
  std::memcpy(&p.params.missing, &missing, 4);
  for (auto const& column : adapter.Columns()) p.arrays.push_back(p.Window(column, begin, rows));
  return p.Finish(map);
}
std::unique_ptr<InputBatch> PrepareInput(data::CSRArrayAdapter const& adapter, float missing,
                                         std::size_t begin, std::size_t rows,
                                         std::uint32_t features, enc::MappingView map) {
  InputBounds(adapter.NumRows(), begin, rows);
  PackedInput p{rows, features, 3, map};
  CHECK_LE(adapter.NumColumns(), features);
  std::memcpy(&p.params.missing, &missing, 4);
  if (!rows) return p.Finish(map);
  auto first = TypedIndex<std::uint64_t, 1>{adapter.Indptr()}(begin);
  auto last = TypedIndex<std::uint64_t, 1>{adapter.Indptr()}(begin + rows);
  CHECK_LE(first, last);
  CHECK_LE(last, adapter.Indices().n);
  CHECK_LE(last, adapter.Values().n);
  p.params.first_entry = first;
  p.params.entries = last - first;
  p.arrays.push_back(p.Window(adapter.Indptr(), begin, rows + 1));
  p.arrays.push_back(p.Window(adapter.Indices(), first, last - first));
  p.arrays.push_back(p.Window(adapter.Values(), first, last - first));
  return p.Finish(map);
}

namespace {
class MetalAdapterInput final : public AdapterInput {
 public:
  id<MTLBuffer> entries, row_ptr;
  bst_idx_t rows, base_row, nnz, stride, sketch_rows;
  bst_idx_t raw_begin, raw_slots;
  std::uint32_t features;
  bool partial_first, has_inf;
  bst_idx_t Rows() const override { return rows; }
  bst_idx_t BaseRow() const override { return base_row; }
  bst_idx_t NonMissing() const override { return nnz; }
  bst_idx_t RowStride() const override { return stride; }
  bst_idx_t SketchRows() const override { return sketch_rows; }
  void Readback(SparsePage* output) const override {
    output->base_rowid = base_row;
    auto& ptr = output->offset.HostVector();
    ptr.resize(rows + 1);
    auto& data = output->data.HostVector();
    data.resize(nnz);
    std::memcpy(ptr.data(), row_ptr.contents, ptr.size() * sizeof(bst_idx_t));
    if (nnz) std::memcpy(data.data(), entries.contents, nnz * sizeof(Entry));
    adapter_debug_readback_bytes.fetch_add(ptr.size() * 8 + nnz * 8, std::memory_order_relaxed);
  }
  void WriteIndex(Span<FeatureType const> types, GHistIndexMatrix* output, bst_idx_t row_base,
                  bst_idx_t entry_offset) const override {
    CHECK(!has_inf) << error::InfInData();
    CHECK_LE(row_base + base_row + rows + 1, output->row_ptr.size());
    CHECK_LE(entry_offset + nnz, output->index.Size());
    BinEntriesBuffer(entries, nnz, types, output, entry_offset);
    auto& rt = GetRuntime();
    @autoreleasepool {
      auto ptr = Allocate(rt, (rows + 1) * 8);
      auto command = [rt.queue commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      struct {
        std::uint64_t rows, offset;
      } p{rows + 1, entry_offset};
      [e setComputePipelineState:rt.adapter_offsets_pipeline];
      [e setBuffer:row_ptr offset:0 atIndex:0];
      [e setBuffer:ptr offset:0 atIndex:1];
      [e setBytes:&p length:sizeof(p) atIndex:2];
      [e dispatchThreads:MTLSizeMake(rows + 1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &adapter_index_dispatches, &adapter_gpu_time_ns);
      auto skip = std::size_t(partial_first);
      std::memcpy(output->row_ptr.data() + row_base + base_row + skip,
                  static_cast<std::uint64_t const*>(ptr.contents) + skip, (rows + 1 - skip) * 8);
    }
  }
};
std::unique_ptr<AdapterInput> PackedInput::FinishAdapter(enc::MappingView map, bst_idx_t base_row,
                                                         bst_idx_t sketch_rows) {
  auto& rt = GetRuntime();
  auto result = std::make_unique<MetalAdapterInput>();
  result->rows = params.rows;
  result->base_row = base_row;
  result->nnz = 0;
  result->stride = 0;
  result->sketch_rows = sketch_rows;
  result->features = params.features;
  result->raw_begin = base_row * params.features + params.first_entry;
  result->raw_slots = params.entries;
  result->partial_first = params.first_entry != 0;
  result->has_inf = false;
  if (params.mapped) CHECK_EQ(map.offsets.size(), std::size_t(params.features) + 1);
  @autoreleasepool {
    if (!params.rows) {
      std::uint64_t zero = 0;
      result->row_ptr = Upload(rt, &zero, 8);
      result->entries = Allocate(rt, 0);
      return result;
    }
    auto raw = native ? native : Upload(rt, bytes.data(), bytes.size());
    auto descriptors = Upload(rt, arrays.data(), arrays.size() * sizeof(InputArray));
    auto segments = BindCategoryInput(rt, map.offsets.data(), map.offsets.size_bytes(), 4);
    auto mapping = BindCategoryInput(rt, map.mapping.data(), map.mapping.size_bytes(), 4);
    auto counts = Allocate(rt, (params.rows + 1) * 8);
    std::uint32_t zeros[2]{};
    auto status = Upload(rt, zeros, sizeof(zeros));
    auto dispatch = [&](id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> row_ptr, id<MTLBuffer> entries) {
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:pipeline];
      [e setBuffer:raw offset:0 atIndex:0];
      [e setBuffer:descriptors offset:0 atIndex:1];
      [e setBuffer:segments.buffer offset:segments.offset atIndex:2];
      [e setBuffer:mapping.buffer offset:mapping.offset atIndex:3];
      [e setBuffer:row_ptr offset:0 atIndex:4];
      [e setBuffer:entries offset:0 atIndex:5];
      [e setBuffer:status offset:0 atIndex:6];
      [e setBytes:&params length:sizeof(params) atIndex:7];
      [e dispatchThreads:MTLSizeMake(params.rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    };
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    dispatch(command, rt.adapter_count_pipeline, counts, counts);
    result->row_ptr = EncodePrefixU64(rt, command, counts, params.rows + 1);
    CompleteOperation(command, &adapter_prepare_dispatches, &adapter_gpu_time_ns);
    auto flags = static_cast<std::uint32_t const*>(status.contents);
    result->has_inf = flags[0] != 0;
    result->stride = flags[1];
    result->nnz = static_cast<std::uint64_t const*>(result->row_ptr.contents)[params.rows];
    CHECK_LE(result->nnz, params.entries);
    result->entries = Allocate(rt, result->nnz * 8);
    if (result->nnz) {
      command = [rt.queue commandBuffer];
      CHECK(command);
      dispatch(command, rt.adapter_scatter_pipeline, result->row_ptr, result->entries);
      CompleteOperation(command, &adapter_prepare_dispatches, &adapter_gpu_time_ns);
    }
    return result;
  }
}
template <class Adapter>
std::unique_ptr<AdapterInput> PrepareAdapterImpl(Adapter const& adapter, float missing,
                                                 bst_idx_t begin, bst_idx_t count,
                                                 enc::MappingView map) {
  auto cols = adapter.NumColumns(), total_rows = adapter.NumRows();
  CHECK_GT(cols, 0);
  CHECK_LE(cols, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(total_rows, std::numeric_limits<std::size_t>::max() / cols);
  InputBounds(total_rows * cols, begin, count);
  CHECK_LE(count, 1ULL << 26);
  auto base = begin / cols, first = begin % cols,
       rows = count ? (first + count + cols - 1) / cols : 0;
  PackedInput p{rows, static_cast<std::uint32_t>(cols),
                std::is_same_v<Adapter, data::ArrayAdapter> ? 1u : 2u, map};
  p.params.first_entry = first;
  p.params.entries = count;
  std::memcpy(&p.params.missing, &missing, 4);
  if constexpr (std::is_same_v<Adapter, data::ArrayAdapter>) {
    p.arrays.push_back(p.Window(adapter.Array(), base, rows, cols));
  } else {
    for (auto const& column : adapter.Columns()) p.arrays.push_back(p.Window(column, base, rows));
  }
  return p.FinishAdapter(map, base, count ? (count + cols - 1) / cols : 0);
}
}  // namespace
std::shared_ptr<Tensor> ImportTensor(StringView descriptor) {
  auto json = Json::Load(descriptor);
  auto const& obj = get<Object const>(json);
  if (obj.find("columns") != obj.cend()) return std::make_shared<MetalColumnar>(json);
  return std::make_shared<MetalTensor>(descriptor);
}
void CopyTensorToSparsePage(Tensor const& source, float missing, SparsePage* output) {
  auto const& input = dynamic_cast<MetalFeatureInput const&>(source);
  HostDeviceVector<std::int32_t> mapping;
  enc::MappingView map;
  if (input.HasRefCategorical()) {
    Context ctx;
    ctx.Init({{"device", "mps"}});
    CatContainer cats{&ctx, input.RefCats(), true};
    cats.Sort(&ctx);
    auto encoded = ::xgboost::cpu_impl::MakeCatAccessor(&ctx, input.Cats(), &cats);
    map = std::get<0>(encoded).enc;
    mapping = std::move(std::get<1>(encoded));
  }
  PackedInput p{input.NumRows(), input.NumColumns(), input.Kind(), map};
  p.native = input.Buffer();
  p.arrays = input.Arrays(0);
  p.params.entries = input.NumRows() * input.NumColumns();
  std::memcpy(&p.params.missing, &missing, 4);
  auto prepared = p.FinishAdapter(map, 0, input.NumRows());
  auto const& packed = static_cast<MetalAdapterInput const&>(*prepared);
  CHECK(!packed.has_inf) << error::InfInData();
  // CUDA reads the last row offset to size the entry allocation. All entry
  // conversion/filtering and prefix work above remains on GPU.
  output->offset.SetDevice(DeviceOrd::MPS());
  output->data.SetDevice(DeviceOrd::MPS());
  output->offset.Resize(input.NumRows() + 1);
  output->data.Resize(packed.nnz);
  auto rows = output->offset.DeviceSpan();
  auto entries = output->data.DeviceSpan();
  static_assert(sizeof(Entry) == 8 && sizeof(bst_idx_t) == 8);
  auto& rt = GetRuntime();
  @autoreleasepool {
    auto row_view = TryBorrowMemory(rt.device, rows.data(), rows.size_bytes(), 8);
    auto data_view = TryBorrowMemory(rt.device, entries.data(), entries.size_bytes(), 8);
    auto command = [rt.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    CHECK(blit);
    if (row_view.buffer)
      [blit copyFromBuffer:packed.row_ptr sourceOffset:0 toBuffer:row_view.buffer
        destinationOffset:row_view.offset size:rows.size_bytes()];
    if (data_view.buffer)
      [blit copyFromBuffer:packed.entries sourceOffset:0 toBuffer:data_view.buffer
        destinationOffset:data_view.offset size:entries.size_bytes()];
    [blit endEncoding];
    CompleteOperation(command, &tensor_csr_dispatches, &tensor_csr_gpu_time_ns);
    if (!row_view.buffer) std::memcpy(rows.data(), packed.row_ptr.contents, rows.size_bytes());
    if (!data_view.buffer && !entries.empty())
      std::memcpy(entries.data(), packed.entries.contents, entries.size_bytes());
    tensor_csr_shared_output_bytes.fetch_add((row_view.buffer ? rows.size_bytes() : 0) +
                                             (data_view.buffer ? entries.size_bytes() : 0),
                                             std::memory_order_relaxed);
    tensor_csr_host_copy_bytes.fetch_add((row_view.buffer ? 0 : rows.size_bytes()) +
                                         (data_view.buffer ? 0 : entries.size_bytes()),
                                         std::memory_order_relaxed);
    tensor_csr_values.fetch_add(packed.nnz, std::memory_order_relaxed);
  }
  // CopyToSparsePage publishes row offsets on the host for sizing/window
  // selection. Entry payloads remain device-authoritative until CPU access.
  CHECK_EQ(output->offset.HostVector().back(), packed.nnz);
}
void CopyNativeGradients(StringView grad, StringView hess, bst_idx_t rows,
                         linalg::Matrix<GradientPair>* output) {
  // Both snapshots validate devices, dtypes, extents and signed strides before
  // resizing output, also protecting producers that alias its previous storage.
  MetalTensor g{grad, true}, h{hess, true};
  StringView msg{"Mismatched shape between the gradient and hessian."};
  CHECK_EQ(g.rows, h.rows) << msg;
  CHECK_EQ(g.columns, h.columns) << msg;
  CHECK_EQ(g.rows, rows) << "Mismatched size between the gradient and training data.";
  auto count = rows * g.columns;
  auto& rt = GetRuntime();
  CHECK_LT(count, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(count, rt.device.maxBufferLength / sizeof(GradientPair));
  // CUDA selects the input device independently of the learner's device. Keep
  // the packed pairs device-authoritative until a consumer explicitly reads
  // them on the CPU, including empty and reused destinations.
  output->SetDevice(DeviceOrd::MPS());
  output->Reshape(rows, g.columns);
  auto destination = output->Data()->DeviceSpan();
  if (!count) return;
  static_assert(sizeof(GradientPair) == 8);
  @autoreleasepool {
    auto out = TryBorrowMemory(rt.device, destination.data(), destination.size_bytes(), 8);
    bool shared = out.buffer != nil;
    if (!shared) out = {Allocate(rt, destination.size_bytes()), 0};
    struct { std::uint64_t count; std::uint32_t columns, padding; } p{count, g.columns, 0};
    static_assert(sizeof(p) == 16);
    auto command = [rt.queue commandBuffer];
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.gradient_import_pipeline];
    [e setBuffer:g.buffer offset:0 atIndex:0];
    [e setBuffer:h.buffer offset:0 atIndex:1];
    [e setBuffer:out.buffer offset:out.offset atIndex:2];
    [e setBytes:&g.array length:sizeof(g.array) atIndex:3];
    [e setBytes:&h.array length:sizeof(h.array) atIndex:4];
    [e setBytes:&p length:sizeof(p) atIndex:5];
    [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &gradient_import_dispatches, &gradient_import_gpu_time_ns);
    if (!shared) std::memcpy(destination.data(), out.buffer.contents, destination.size_bytes());
    (shared ? gradient_import_shared_output_bytes : gradient_import_host_copy_bytes)
        .fetch_add(destination.size_bytes(), std::memory_order_relaxed);
    gradient_import_pairs.fetch_add(count, std::memory_order_relaxed);
  }
}
void SliceMetadata(MetaInfo const& info, Span<bst_idx_t const> indices, MetaInfo* output) {
  auto matrix = [&](linalg::Matrix<float> const& source, linalg::Matrix<float>* target) {
    if (!source.Size()) return;
    source.SetDevice(DeviceOrd::MPS());
    target->SetDevice(DeviceOrd::MPS());
    target->Reshape(indices.size(), source.Shape(1));
    GatherRows(source.Data()->ConstDeviceSpan(), source.Shape(0), source.Shape(1), indices,
               target->Data()->DeviceSpan());
  };
  auto vector = [&](HostDeviceVector<float> const& source, HostDeviceVector<float>* target) {
    if (source.Empty()) return;
    source.SetDevice(DeviceOrd::MPS());
    target->SetDevice(DeviceOrd::MPS());
    target->Resize(indices.size());
    GatherRows(source.ConstDeviceSpan(), source.Size(), 1, indices, target->DeviceSpan());
  };
  matrix(info.labels, &output->labels);
  matrix(info.base_margin_, &output->base_margin_);
  vector(info.labels_lower_bound_, &output->labels_lower_bound_);
  vector(info.labels_upper_bound_, &output->labels_upper_bound_);
  vector(info.weights_, &output->weights_);
}

void SetMetadata(MetaInfo* info, StringView field, StringView descriptor) {
  auto kind = static_cast<std::uint32_t>(data::MapMetaField(field, true));
  MetalTensor input{descriptor, true};
  if (kind != 0 && kind != 2) {
    CHECK(input.rows == 1 || input.columns == 1) << "Metadata field requires a vector.";
  }
  if (kind == 6) {
    // Match CUDA CopyGroupInfoImpl (including its binary16 conversion path).
    CHECK(input.array.type != 1 && input.array.type != 2) << "Expected integer for group info.";
  }
  auto count = input.rows * input.columns;
  auto& rt = GetRuntime();
  CHECK_LT(count, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(count + 1, rt.device.maxBufferLength / sizeof(std::uint32_t));
  struct { std::uint64_t count; std::uint32_t columns, kind; } p{count, input.columns, kind};
  static_assert(sizeof(p) == 16);
  auto encode = [&](id<MTLCommandBuffer> command, BorrowedMemory output, id<MTLBuffer> status) {
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.metadata_convert_pipeline];
    [e setBuffer:input.buffer offset:0 atIndex:0];
    [e setBuffer:output.buffer offset:output.offset atIndex:1];
    [e setBuffer:status offset:0 atIndex:2];
    [e setBytes:&input.array length:sizeof(input.array) atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
  };
  metadata_values.fetch_add(count, std::memory_order_relaxed);
  if (kind < 6) {
    Span<float> output;
    if (kind == 0 || kind == 2) {
      auto target = kind == 0 ? &info->labels : &info->base_margin_;
      // CUDA preserves array shape here; only the CPU setter calls ReshapeInfo.
      target->SetDevice(DeviceOrd::MPS());
      target->Reshape(input.rows, input.columns);
      output = target->Data()->DeviceSpan();
    } else {
      auto target = kind == 1 ? &info->weights_ : kind == 3 ? &info->labels_lower_bound_
                    : kind == 4 ? &info->labels_upper_bound_ : &info->feature_weights;
      target->SetDevice(DeviceOrd::MPS());
      target->Resize(count);
      output = target->DeviceSpan();
    }
    if (!count) return;
    @autoreleasepool {
      auto out = TryBorrowMemory(rt.device, output.data(), output.size_bytes(), alignof(float));
      bool shared = out.buffer != nil;
      if (!shared) out = {Allocate(rt, output.size_bytes()), 0};
      std::uint32_t zero = 0;
      auto status = Upload(rt, &zero, sizeof(zero));
      auto command = [rt.queue commandBuffer];
      bool copy = input.array.type == 1 &&
          (input.rows <= 1 || input.array.row_stride == std::int64_t(input.columns) * 4) &&
          (input.columns <= 1 || input.array.column_stride == 4);
      if (copy) {
        auto e = [command blitCommandEncoder];
        CHECK(e);
        [e copyFromBuffer:input.buffer sourceOffset:input.array.offset toBuffer:out.buffer
           destinationOffset:out.offset size:output.size_bytes()];
        [e endEncoding];
      } else {
        encode(command, out, status);
      }
      bool validate = kind == 0 || kind == 1 || kind == 5;
      if (validate) {
        auto e = [command computeCommandEncoder];
        CHECK(e);
        [e setComputePipelineState:rt.metadata_validate_pipeline];
        [e setBuffer:out.buffer offset:out.offset atIndex:0];
        [e setBuffer:status offset:0 atIndex:1];
        [e setBytes:&p length:sizeof(p) atIndex:2];
        [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
      }
      CompleteOperation(command, copy ? &metadata_copy_dispatches : &metadata_convert_dispatches,
                        &metadata_gpu_time_ns);
      if (validate) metadata_validate_dispatches.fetch_add(1, std::memory_order_relaxed);
      if (!shared) std::memcpy(output.data(), out.buffer.contents, output.size_bytes());
      (shared ? metadata_shared_output_bytes : metadata_host_copy_bytes)
          .fetch_add(output.size_bytes(), std::memory_order_relaxed);
      CHECK_EQ(*static_cast<std::uint32_t const*>(status.contents), 0)
          << (kind == 0 ? "Label contains NaN, infinity or a value too large."
              : kind == 1 ? "Weights must be positive values."
                          : "Feature weight must be greater than 0.");
    }
    return;
  }
  if (!count) {
    info->group_ptr_.assign(1, 0);
    return;
  }
  @autoreleasepool {
    auto n = static_cast<std::uint32_t>(count);
    std::uint32_t zeros[3]{};
    auto status = Upload(rt, zeros, sizeof(zeros));
    auto ids = Allocate(rt, count * 4);
    auto command = [rt.queue commandBuffer];
    encode(command, {ids, 0}, status);
    id<MTLBuffer> starts{nil};
    if (kind == 7) {
      auto flags = Allocate(rt, count * 4);
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:rt.metadata_qid_mark_pipeline];
      [e setBuffer:ids offset:0 atIndex:0];
      [e setBuffer:flags offset:0 atIndex:1];
      [e setBuffer:status offset:0 atIndex:2];
      [e setBytes:&n length:sizeof(n) atIndex:3];
      [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      auto prefix = EncodePrefixSum(rt, command, flags, n);
      starts = Allocate(rt, (count + 1) * 4);
      e = [command computeCommandEncoder];
      [e setComputePipelineState:rt.metadata_qid_starts_pipeline];
      [e setBuffer:flags offset:0 atIndex:0];
      [e setBuffer:prefix offset:0 atIndex:1];
      [e setBuffer:starts offset:0 atIndex:2];
      [e setBuffer:status offset:0 atIndex:3];
      [e setBytes:&n length:sizeof(n) atIndex:4];
      [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    CompleteOperation(command, &metadata_convert_dispatches, &metadata_gpu_time_ns);
    auto checked = static_cast<std::uint32_t const*>(status.contents);
    CHECK_EQ(checked[0], 0) << "Metal group/qid floating conversion requires a finite, uint32-representable truncated value.";
    if (kind == 6) {
      // CUDA also keeps group_ptr_ on host and performs this prefix sum there.
      info->group_ptr_.resize(count + 1);
      info->group_ptr_[0] = 0;
      std::memcpy(info->group_ptr_.data() + 1, ids.contents, count * 4);
      metadata_host_copy_bytes.fetch_add(count * 4, std::memory_order_relaxed);
      std::partial_sum(info->group_ptr_.begin(), info->group_ptr_.end(), info->group_ptr_.begin());
    } else {
      metadata_qid_dispatches.fetch_add(1, std::memory_order_relaxed);
      CHECK_EQ(checked[1], 0) << "`qid` must be sorted in increasing order along with data.";
      std::uint32_t runs = checked[2];
      CHECK_GT(runs, 0);
      CHECK_LE(runs, count);
      auto counts = Allocate(rt, (std::size_t(runs) + 1) * 4);
      command = [rt.queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:rt.metadata_qid_counts_pipeline];
      [e setBuffer:starts offset:0 atIndex:0];
      [e setBuffer:counts offset:0 atIndex:1];
      [e setBytes:&runs length:sizeof(runs) atIndex:2];
      [e dispatchThreads:MTLSizeMake(std::size_t(runs) + 1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
      // CUDA RLE counts are cumulatively scanned on GPU before host publication.
      auto prefix = EncodePrefixSum(rt, command, counts, runs + 1);
      CompleteOperation(command, &metadata_qid_dispatches, &metadata_gpu_time_ns);
      info->group_ptr_.resize(std::size_t(runs) + 1);
      std::memcpy(info->group_ptr_.data(), prefix.contents, info->group_ptr_.size() * 4);
      metadata_host_copy_bytes.fetch_add(info->group_ptr_.size() * 4, std::memory_order_relaxed);
    }
    data::ValidateQueryGroup(info->group_ptr_);
  }
}
std::unique_ptr<AdapterInput> PrepareAdapter(Tensor const& source, float missing,
                                            bst_idx_t begin, bst_idx_t count,
                                            enc::MappingView mapping) {
  auto const& input = dynamic_cast<MetalFeatureInput const&>(source);
  auto cols = input.NumColumns();
  InputBounds(input.NumRows() * cols, begin, count);
  CHECK_LE(count, 1ULL << 26);
  auto base = cols ? begin / cols : 0, first = cols ? begin % cols : 0;
  auto rows = cols ? (count ? (first + count + cols - 1) / cols : 0) : input.NumRows();
  PackedInput p{rows, cols, input.Kind(), mapping};
  p.native = input.Buffer();
  p.arrays = input.Arrays(base);
  p.params.first_entry = first;
  p.params.entries = count;
  std::memcpy(&p.params.missing, &missing, 4);
  auto result = p.FinishAdapter(mapping, base, count ? (count + cols - 1) / cols : 0);
  tensor_adapter_windows.fetch_add(1, std::memory_order_relaxed);
  return result;
}
std::unique_ptr<InputBatch> PrepareInput(Tensor const& source, float missing, std::size_t begin,
                                        std::size_t rows, std::uint32_t features,
                                        enc::MappingView mapping) {
  auto const& input = dynamic_cast<MetalFeatureInput const&>(source);
  InputBounds(input.NumRows(), begin, rows);
  CHECK_LE(input.NumColumns(), features);
  PackedInput p{rows, features, input.Kind(), mapping};
  p.params.input_features = input.NumColumns();
  p.native = input.Buffer();
  p.arrays = input.Arrays(begin);
  std::memcpy(&p.params.missing, &missing, 4);
  auto result = p.Finish(mapping);
  tensor_prediction_windows.fetch_add(1, std::memory_order_relaxed);
  return result;
}
std::unique_ptr<AdapterInput> PrepareAdapter(data::ArrayAdapter const& adapter, float missing,
                                             bst_idx_t begin, bst_idx_t count,
                                             enc::MappingView mapping) {
  return PrepareAdapterImpl(adapter, missing, begin, count, mapping);
}
std::unique_ptr<AdapterInput> PrepareAdapter(data::ColumnarAdapter const& adapter, float missing,
                                             bst_idx_t begin, bst_idx_t count,
                                             enc::MappingView mapping) {
  return PrepareAdapterImpl(adapter, missing, begin, count, mapping);
}

namespace {
struct EllpackParams {
  std::uint64_t rows, split, row_offset;
  std::uint32_t features, stride, symbols, bits, dense, width;
};
static_assert(sizeof(EllpackParams) == 48);
static_assert(common::detail::kPadding == 8, "Metal bitstream padding must match CUDA.");
id<MTLBuffer> EllpackPrivate(Runtime const& rt, std::size_t bytes) {
  auto length = std::max<std::size_t>(bytes, 16);
  CHECK_LE(length, rt.device.maxBufferLength) << "Metal compressed page exceeds the buffer limit.";
  auto buffer = [rt.device newBufferWithLength:length options:MTLResourceStorageModePrivate];
  CHECK(buffer) << "Metal private cache allocation failed.";
  return buffer;
}
struct SnappyWindow {
  std::uint64_t input;
  std::uint32_t bytes, padding;
};
struct SnappyChunk {
  std::uint64_t source, destination;
  std::uint32_t bytes, decoded;
};
static_assert(sizeof(SnappyWindow) == 16 && sizeof(SnappyChunk) == 24);
struct MetalSnappy {
  id<MTLBuffer> data;
  std::vector<SnappyChunk> chunks;
  std::size_t bytes{0}, decoded{0};
};
// A block is processed at a time, bounding transient hash/output workspace even
// for multi-gigabyte cache pages. The actual codec reads/writes device buffers.
MetalSnappy CompressSnappyBuffer(Runtime& rt, id<MTLBuffer> input, std::size_t offset,
                                std::size_t bytes, std::size_t chunk_bytes) {
  CHECK_GT(chunk_bytes, 0);
  CHECK_LE(chunk_bytes, 1 << 21);
  CHECK_LE(offset, input.length);
  CHECK_LE(bytes, input.length - offset);
  constexpr std::size_t kWindow = 32768, kCapacity = 2 * kWindow + 16, kHash = 4096;
  MetalSnappy result;
  result.decoded = bytes;
  std::vector<id<MTLBuffer>> blocks;
  for (std::size_t start = 0; start < bytes;) {
    @autoreleasepool {
      auto size = std::min(chunk_bytes, bytes - start);
      std::vector<SnappyWindow> windows;
      for (std::size_t begin = 0; begin < size; begin += kWindow)
        windows.push_back({offset + start + begin,
                           std::uint32_t(std::min(kWindow, size - begin)), 0});
      std::uint32_t count = windows.size();
      auto descriptors = Upload(rt, windows.data(), windows.size() * sizeof(SnappyWindow));
      auto hashes = EllpackPrivate(rt, count * kHash * 4);
      auto workspace = EllpackPrivate(rt, count * kCapacity);
      auto lengths = Allocate(rt, count * 4);
      auto command = [rt.queue commandBuffer];
      auto e = [command computeCommandEncoder];
      [e setComputePipelineState:rt.snappy_compress_pipeline];
      [e setBuffer:input offset:0 atIndex:0];
      [e setBuffer:descriptors offset:0 atIndex:1];
      [e setBuffer:hashes offset:0 atIndex:2];
      [e setBuffer:workspace offset:0 atIndex:3];
      [e setBuffer:lengths offset:0 atIndex:4];
      [e setBytes:&count length:sizeof(count) atIndex:5];
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [e endEncoding];
      CompleteOperation(command, &snappy_compress_dispatches, &snappy_gpu_time_ns);
      auto sizes = static_cast<std::uint32_t const*>(lengths.contents);
      std::size_t header = 1;
      for (auto value = size; value >= 128; value >>= 7) ++header;
      std::size_t compressed = header;
      for (std::uint32_t i = 0; i < count; ++i) {
        CHECK_LE(sizes[i], kCapacity);
        compressed += sizes[i];
      }
      auto block = Allocate(rt, compressed);
      // CPU metadata/framing only. All literal and backreference payloads are GPU
      // generated, and coalesced by blit without reading them back to the CPU.
      auto dest = static_cast<std::uint8_t*>(block.contents);
      auto remaining = size;
      for (std::size_t i = 0; i < header; ++i) {
        dest[i] = (remaining & 127) | (remaining >= 128 ? 128 : 0);
        remaining >>= 7;
      }
      command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      std::size_t position = header;
      for (std::uint32_t i = 0; i < count; ++i) {
        [blit copyFromBuffer:workspace sourceOffset:i * kCapacity toBuffer:block
          destinationOffset:position size:sizes[i]];
        position += sizes[i];
      }
      [blit endEncoding];
      CompleteOperation(command, &snappy_copy_dispatches, &snappy_gpu_time_ns);
      result.chunks.push_back({result.bytes, start, std::uint32_t(compressed), std::uint32_t(size)});
      CHECK_LE(compressed, rt.device.maxBufferLength - result.bytes);
      result.bytes += compressed;
      blocks.push_back(block);
      start += size;
    }
  }
  result.data = Allocate(rt, result.bytes);
  if (!blocks.empty()) {
    auto command = [rt.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    for (std::size_t i = 0; i < blocks.size(); ++i)
      [blit copyFromBuffer:blocks[i] sourceOffset:0 toBuffer:result.data
        destinationOffset:result.chunks[i].source size:result.chunks[i].bytes];
    [blit endEncoding];
    CompleteOperation(command, &snappy_copy_dispatches, &snappy_gpu_time_ns);
  }
  snappy_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  snappy_compressed_bytes.fetch_add(result.bytes, std::memory_order_relaxed);
  return result;
}
void DecompressSnappyBuffer(Runtime& rt, MetalSnappy const& input, id<MTLBuffer> output,
                            std::size_t offset) {
  CHECK_LE(offset, output.length);
  CHECK_LE(input.decoded, output.length - offset);
  if (input.chunks.empty()) return;
  auto chunks = input.chunks;
  for (auto& c : chunks) c.destination += offset;
  auto descriptors = Upload(rt, chunks.data(), chunks.size() * sizeof(SnappyChunk));
  auto statuses = Allocate(rt, chunks.size() * 4);
  auto command = [rt.queue commandBuffer];
  auto e = [command computeCommandEncoder];
  [e setComputePipelineState:rt.snappy_decompress_pipeline];
  [e setBuffer:input.data offset:0 atIndex:0];
  [e setBuffer:descriptors offset:0 atIndex:1];
  [e setBuffer:output offset:0 atIndex:2];
  [e setBuffer:statuses offset:0 atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(chunks.size(), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [e endEncoding];
  CompleteOperation(command, &snappy_decompress_dispatches, &snappy_gpu_time_ns);
  auto status = static_cast<std::uint32_t const*>(statuses.contents);
  for (std::size_t i = 0; i < chunks.size(); ++i)
    CHECK_EQ(status[i], 0) << "Invalid Metal Snappy block " << i;
  snappy_decoded_bytes.fetch_add(input.decoded, std::memory_order_relaxed);
}
}  // namespace
SnappyBlocks CompressSnappy(Span<std::uint8_t const> input, std::size_t chunk_bytes) {
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto source = Upload(rt, input.data(), input.size());
    auto compressed = CompressSnappyBuffer(rt, source, 0, input.size(), chunk_bytes);
    SnappyBlocks result;
    auto ptr = static_cast<std::uint8_t const*>(compressed.data.contents);
    result.data.assign(ptr, ptr + compressed.bytes);
    for (auto c : compressed.chunks) {
      result.offsets.push_back(c.source + c.bytes);
      result.decoded_sizes.push_back(c.decoded);
    }
    snappy_host_readback_bytes.fetch_add(compressed.bytes, std::memory_order_relaxed);
    return result;
  }
}
std::vector<std::uint8_t> DecompressSnappy(SnappyBlocks const& input) {
  @autoreleasepool {
    CHECK_EQ(input.offsets.size(), input.decoded_sizes.size() + 1);
    CHECK_EQ(input.offsets.front(), 0);
    CHECK_EQ(input.offsets.back(), input.data.size());
    auto& rt = GetRuntime();
    MetalSnappy compressed;
    compressed.bytes = input.data.size();
    for (std::size_t i = 0; i < input.decoded_sizes.size(); ++i) {
      CHECK_LE(input.offsets[i], input.offsets[i + 1]);
      auto bytes = input.offsets[i + 1] - input.offsets[i];
      CHECK_LE(bytes, std::numeric_limits<std::uint32_t>::max());
      CHECK_LE(input.decoded_sizes[i], rt.device.maxBufferLength - compressed.decoded);
      compressed.chunks.push_back({input.offsets[i], compressed.decoded, std::uint32_t(bytes),
                                  input.decoded_sizes[i]});
      compressed.decoded += input.decoded_sizes[i];
    }
    compressed.data = Upload(rt, input.data.data(), input.data.size());
    // Guard bytes make the diagnostic boundary useful for malformed-stream tests.
    CHECK_LE(compressed.decoded, rt.device.maxBufferLength - 32);
    auto output = Allocate(rt, compressed.decoded + 32);
    std::memset(output.contents, 0xa5, output.length);
    auto ptr = static_cast<std::uint8_t const*>(output.contents);
    auto check_guards = [&] {
      for (std::size_t i = 0; i < 16; ++i) {
        CHECK_EQ(ptr[i], 0xa5) << "Snappy destination guard overwritten";
        CHECK_EQ(ptr[16 + compressed.decoded + i], 0xa5) << "Snappy destination guard overwritten";
      }
    };
    try {
      DecompressSnappyBuffer(rt, compressed, output, 16);
    } catch (...) {
      check_guards();
      throw;
    }
    check_guards();
    snappy_host_readback_bytes.fetch_add(compressed.decoded, std::memory_order_relaxed);
    return {ptr + 16, ptr + 16 + compressed.decoded};
  }
}
namespace {
// The row prefix stays on GPU. Callers may read its final scalar for validation
// or copy it only at the explicit host-materialization boundary.
id<MTLBuffer> EllpackRowPtr(Runtime& rt, EllpackParams const& p, id<MTLBuffer> head,
                           id<MTLBuffer> tail, std::atomic<std::uint64_t>* counter) {
  auto counts = Allocate(rt, (p.rows + 1) * 8);
  auto command = [rt.queue commandBuffer];
  auto e = [command computeCommandEncoder];
  [e setComputePipelineState:rt.ellpack_count_pipeline];
  [e setBuffer:head offset:0 atIndex:0];
  [e setBuffer:tail offset:0 atIndex:1];
  [e setBuffer:counts offset:0 atIndex:2];
  [e setBytes:&p length:sizeof(p) atIndex:3];
  [e dispatchThreads:MTLSizeMake(std::max<std::uint64_t>(1, p.rows), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  auto rows = EncodePrefixU64(rt, command, counts, p.rows + 1);
  CompleteOperation(command, counter, &ellpack_gpu_time_ns);
  return rows;
}
class MetalCompressedPage final : public CompressedPage {
  EllpackInfo info_;
  bst_idx_t rows_, base_, written_{0};
  std::size_t bytes_, host_bytes_{0};
  std::size_t plain_host_bytes_{0};
  std::unique_ptr<MetalSnappy> snappy_;
  bool committed_{false};
  id<MTLBuffer> buffer_, head_, tail_;
  bool adapter_active_{false};
  bst_idx_t adapter_slots_{0};
  id<MTLBuffer> adapter_counts_, adapter_ptrs_, adapter_cuts_, adapter_types_;
  EllpackParams Params() const {
    return {rows_,
            committed_ ? host_bytes_ : 0,
            0,
            info_.features,
            info_.row_stride,
            info_.symbols,
            info_.bits,
            std::uint32_t(info_.dense),
            4};
  }
  id<MTLBuffer> Head() const { return committed_ ? head_ : buffer_; }
  id<MTLBuffer> Tail() const { return committed_ ? tail_ : buffer_; }
  struct ReadView {
    id<MTLBuffer> head, tail;
    EllpackParams params;
  };
  ReadView Read(bool prefetch_copy = false) const {
    ReadView view{Head(), Tail(), Params()};
    // CUDA forces prefetch_copy when any host segment is Snappy-compressed.
    if (snappy_ || (prefetch_copy && view.params.split)) {
      auto& rt = GetRuntime();
      auto full = EllpackPrivate(rt, bytes_);
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      auto prefix = snappy_ ? plain_host_bytes_ : view.params.split;
      if (prefix)
        [blit copyFromBuffer:view.head sourceOffset:0 toBuffer:full destinationOffset:0 size:prefix];
      if (bytes_ > view.params.split)
        [blit copyFromBuffer:view.tail sourceOffset:0 toBuffer:full
          destinationOffset:view.params.split size:bytes_ - view.params.split];
      [blit endEncoding];
      CompleteOperation(command, &ellpack_cache_dispatches, &ellpack_gpu_time_ns);
      if (snappy_) DecompressSnappyBuffer(rt, *snappy_, full, prefix);
      view.head = view.tail = full;
      view.params.split = 0;
    }
    return view;
  }

 public:
  MetalCompressedPage(EllpackInfo info, bst_idx_t rows, bst_idx_t base, std::size_t bytes)
      : info_{info}, rows_{rows}, base_{base}, bytes_{bytes ? bytes : EllpackBytes(rows, info)} {
    @autoreleasepool {
      CHECK_GE(bytes_, EllpackBytes(rows, info));
      CHECK_LE(rows_, std::numeric_limits<std::uint32_t>::max());
      auto& rt = GetRuntime();
      buffer_ = EllpackPrivate(rt, bytes_);
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      [blit fillBuffer:buffer_ range:NSMakeRange(0, buffer_.length) value:0];
      [blit endEncoding];
      CompleteOperation(command, &ellpack_cache_dispatches, &ellpack_gpu_time_ns);
    }
  }
  EllpackInfo Info() const override { return info_; }
  bst_idx_t Rows() const override { return rows_; }
  bst_idx_t BaseRow() const override { return base_; }
  std::size_t SizeBytes() const override { return bytes_; }
  std::size_t HostBytes() const override { return host_bytes_; }
  std::size_t HostStorageBytes() const override {
    return snappy_ ? plain_host_bytes_ + snappy_->bytes : host_bytes_;
  }
  std::size_t DeviceBytes() const override { return bytes_ - host_bytes_; }
  void Pack(GHistIndexMatrix const& input) override {
    @autoreleasepool {
      input.RequireHost();
      CHECK(!committed_);
      CHECK(!adapter_active_);
      CHECK_EQ(written_, 0);
      CHECK_EQ(input.Size(), rows_);
      CHECK_EQ(input.Features(), info_.features);
      CHECK_EQ(input.IsDense(), info_.dense);
      auto& rt = GetRuntime();
      auto p = Params();
      p.width = input.index.GetBinTypeSize();
      if (rows_ && info_.row_stride) {
        auto index = Upload(rt, input.index.data<std::uint8_t>(), input.index.Size() * p.width);
        auto rows = Upload(rt, input.row_ptr.data(), input.row_ptr.size() * sizeof(std::size_t));
        auto cuts = Upload(rt, input.cut.Ptrs().data(), input.cut.Ptrs().size() * 4);
        auto command = [rt.queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:rt.ellpack_pack_pipeline];
        [e setBuffer:index offset:0 atIndex:0];
        [e setBuffer:rows offset:0 atIndex:1];
        [e setBuffer:cuts offset:0 atIndex:2];
        [e setBuffer:buffer_ offset:0 atIndex:3];
        [e setBytes:&p length:sizeof(p) atIndex:4];
        [e dispatchThreads:MTLSizeMake(info_.row_stride, rows_, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [e endEncoding];
        CompleteOperation(command, &ellpack_pack_dispatches, &ellpack_gpu_time_ns);
        ellpack_pack_input_bytes.fetch_add(input.index.Size() * p.width + input.row_ptr.size() * 8,
                                           std::memory_order_relaxed);
      }
      written_ = rows_;
    }
  }
  void PushCSR(SparsePage const& input, HistogramCuts const& cuts,
               Span<FeatureType const> types, bst_idx_t window_rows) override {
    @autoreleasepool {
      CHECK(!committed_ && !adapter_active_);
      CHECK_EQ(input.base_rowid, base_ + written_);
      CHECK_LE(input.Size(), rows_ - written_);
      CHECK_EQ(cuts.NumFeatures(), info_.features);
      CHECK(types.empty() || types.size() == info_.features);
      auto const& offsets = input.offset.ConstHostVector();
      auto values = ConstDeviceSpan(input.data);
      CHECK_EQ(offsets.size(), input.Size() + 1);
      CHECK_LE(offsets.front(), offsets.back());
      CHECK_LE(offsets.back(), values.size());
      if (!info_.row_stride) {
        CHECK_EQ(offsets.front(), offsets.back());
        written_ += input.Size();
        return;
      }
      auto& rt = GetRuntime();
      // CUDA uses TotalMemory/(16*stride*sizeof(Entry)); keep the same bounded
      // row-window policy with Apple's working-set recommendation and buffer cap.
      auto max_rows = std::max<std::uint64_t>(1, CacheWorkingSetBytes() / 16 /
                                                   (info_.row_stride * sizeof(Entry)));
      max_rows = std::min<std::uint64_t>(max_rows, rt.device.maxBufferLength /
                                                    (info_.row_stride * sizeof(Entry)));
      max_rows = std::min<std::uint64_t>(max_rows, rt.device.maxBufferLength / 8 - 1);
      CHECK_GT(max_rows, 0);
      if (window_rows) max_rows = std::min<std::uint64_t>(max_rows, window_rows);
      auto ptrs = Upload(rt, cuts.Ptrs().data(), cuts.Ptrs().size() * 4);
      auto cut_values = Upload(rt, cuts.Values().data(), cuts.Values().size() * 4);
      std::vector<std::uint8_t> categories(info_.features, 0);
      for (std::size_t f = 0; f < types.size(); ++f)
        categories[f] = types[f] == FeatureType::kCategorical;
      auto categorical = Upload(rt, categories.data(), categories.size());
      for (bst_idx_t first = 0; first < input.Size(); first += max_rows) {
        auto count = std::min<bst_idx_t>(max_rows, input.Size() - first);
        auto begin = offsets[first], end = offsets[first + count];
        CHECK_LE(begin, end);
        CHECK_LE(end, values.size());
        CHECK_LE(end - begin, count * info_.row_stride);
        auto row_ptr = BindCSRInput(rt, offsets.data() + first, (count + 1) * 8);
        auto entries = BindCSRInput(rt, values.empty() ? nullptr : values.data() + begin,
                               (end - begin) * sizeof(Entry));
        std::uint32_t zero = 0;
        auto status = Upload(rt, &zero, 4);
        struct { EllpackParams shape; std::uint64_t entries; } params{Params(), end - begin};
        static_assert(sizeof(params) == 56);
        params.shape.rows = count;
        params.shape.row_offset = written_ + first;
        auto command = [rt.queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:rt.ellpack_csr_pipeline];
        [e setBuffer:row_ptr.buffer offset:row_ptr.offset atIndex:0];
        [e setBuffer:entries.buffer offset:entries.offset atIndex:1];
        [e setBuffer:ptrs offset:0 atIndex:2];
        [e setBuffer:cut_values offset:0 atIndex:3];
        [e setBuffer:categorical offset:0 atIndex:4];
        [e setBuffer:buffer_ offset:0 atIndex:5];
        [e setBuffer:status offset:0 atIndex:6];
        [e setBytes:&params length:sizeof(params) atIndex:7];
        [e dispatchThreads:MTLSizeMake(info_.row_stride, count, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [e endEncoding];
        CompleteOperation(command, &ellpack_csr_dispatches, &ellpack_gpu_time_ns);
        CHECK_EQ(*static_cast<std::uint32_t const*>(status.contents), 0)
            << "Metal CSR compression found invalid rows, features or cut points.";
        ellpack_csr_values.fetch_add(end - begin, std::memory_order_relaxed);
        ellpack_csr_input_bytes.fetch_add((count + 1) * 8 + (end - begin) * sizeof(Entry),
                                          std::memory_order_relaxed);
      }
      written_ += input.Size();
    }
  }
  void BeginAdapter(HistogramCuts const& cuts, Span<FeatureType const> types) override {
    @autoreleasepool {
      CHECK(!committed_ && !adapter_active_);
      CHECK_EQ(written_, 0);
      CHECK_EQ(adapter_slots_, 0);
      CHECK_EQ(cuts.NumFeatures(), info_.features);
      CHECK(types.empty() || types.size() == info_.features);
      auto& rt = GetRuntime();
      adapter_ptrs_ = Upload(rt, cuts.Ptrs().data(), cuts.Ptrs().size() * 4);
      adapter_cuts_ = Upload(rt, cuts.Values().data(), cuts.Values().size() * 4);
      std::vector<std::uint8_t> categorical(info_.features, 0);
      for (std::size_t f = 0; f < types.size(); ++f)
        categorical[f] = types[f] == FeatureType::kCategorical;
      adapter_types_ = Upload(rt, categorical.data(), categorical.size());
      // Sparse row cursors remain private across partial-row adapter windows.
      auto sparse = !info_.dense && info_.row_stride && info_.row_stride != info_.features;
      adapter_counts_ = EllpackPrivate(rt, sparse ? rows_ * 8 : 0);
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      [blit fillBuffer:adapter_counts_ range:NSMakeRange(0, adapter_counts_.length) value:0];
      [blit endEncoding];
      CompleteOperation(command, &ellpack_cache_dispatches, &ellpack_gpu_time_ns);
      adapter_active_ = true;
    }
  }
  void PushAdapter(AdapterInput const& source) override {
    @autoreleasepool {
      CHECK(adapter_active_ && !committed_);
      auto& input = dynamic_cast<MetalAdapterInput const&>(source);
      CHECK(!input.has_inf) << error::InfInData();
      CHECK_EQ(input.features, info_.features);
      CHECK_EQ(input.raw_begin, adapter_slots_)
          << "Adapter windows must be contiguous and ordered.";
      InputBounds(rows_ * info_.features, input.raw_begin, input.raw_slots);
      CHECK_LE(input.stride, info_.row_stride)
          << "Non-missing row width changed between iterations.";
      if (input.rows) {
        auto& rt = GetRuntime();
        auto p = Params();
        p.rows = input.rows;
        p.row_offset = input.base_row;
        struct {
          EllpackParams shape;
          std::uint64_t first, slots;
        } params{p, input.raw_begin % info_.features, input.raw_slots};
        static_assert(sizeof(params) == 64);
        std::uint32_t zero = 0;
        auto status = Upload(rt, &zero, 4);
        auto command = [rt.queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:rt.ellpack_adapter_pipeline];
        [e setBuffer:input.entries offset:0 atIndex:0];
        [e setBuffer:input.row_ptr offset:0 atIndex:1];
        [e setBuffer:adapter_ptrs_ offset:0 atIndex:2];
        [e setBuffer:adapter_cuts_ offset:0 atIndex:3];
        [e setBuffer:adapter_types_ offset:0 atIndex:4];
        [e setBuffer:adapter_counts_ offset:0 atIndex:5];
        [e setBuffer:buffer_ offset:0 atIndex:6];
        [e setBuffer:status offset:0 atIndex:7];
        [e setBytes:&params length:sizeof(params) atIndex:8];
        [e dispatchThreads:MTLSizeMake(input.rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        CompleteOperation(command, &ellpack_adapter_dispatches, &ellpack_gpu_time_ns);
        CHECK_EQ(*static_cast<std::uint32_t const*>(status.contents), 0)
            << "Metal compressed bin assignment found a feature without cut points or an invalid row width.";
        ellpack_adapter_values.fetch_add(input.nnz, std::memory_order_relaxed);
      }
      adapter_slots_ += input.raw_slots;
    }
  }
  void EndAdapter() override {
    @autoreleasepool {
      CHECK(adapter_active_ && !committed_);
      CHECK_EQ(adapter_slots_, rows_ * info_.features) << "Incomplete adapter input.";
      if (!info_.dense && info_.row_stride != info_.features && rows_ && info_.row_stride) {
        auto& rt = GetRuntime();
        auto p = Params();
        auto command = [rt.queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:rt.ellpack_null_pipeline];
        [e setBuffer:adapter_counts_ offset:0 atIndex:0];
        [e setBuffer:buffer_ offset:0 atIndex:1];
        [e setBytes:&p length:sizeof(p) atIndex:2];
        [e dispatchThreads:MTLSizeMake(info_.row_stride, rows_, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [e endEncoding];
        CompleteOperation(command, &ellpack_adapter_dispatches, &ellpack_gpu_time_ns);
      }
      written_ = rows_;
      adapter_active_ = false;
      adapter_counts_ = adapter_ptrs_ = adapter_cuts_ = adapter_types_ = nil;
    }
  }
  void Copy(CompressedPage const& source, bst_idx_t row_offset) override {
    @autoreleasepool {
      CHECK_NE(this, &source);
      CHECK(!committed_);
      CHECK(!adapter_active_);
      CHECK_EQ(written_, row_offset);
      auto& input = dynamic_cast<MetalCompressedPage const&>(source);
      CHECK(!input.adapter_active_);
      CHECK_EQ(input.written_, input.rows_);
      CHECK_EQ(info_.features, input.info_.features);
      CHECK_EQ(info_.row_stride, input.info_.row_stride);
      CHECK_EQ(info_.symbols, input.info_.symbols);
      CHECK_EQ(info_.dense, input.info_.dense);
      CHECK_LE(row_offset, rows_);
      CHECK_LE(input.rows_, rows_ - row_offset);
      auto& rt = GetRuntime();
      auto source_view = input.Read();
      auto p = source_view.params;
      p.row_offset = row_offset;
      if (input.rows_ && info_.row_stride) {
        auto command = [rt.queue commandBuffer];
        auto e = [command computeCommandEncoder];
        [e setComputePipelineState:rt.ellpack_copy_pipeline];
        [e setBuffer:source_view.head offset:0 atIndex:0];
        [e setBuffer:source_view.tail offset:0 atIndex:1];
        [e setBuffer:buffer_ offset:0 atIndex:2];
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e dispatchThreads:MTLSizeMake(info_.row_stride, input.rows_, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [e endEncoding];
        CompleteOperation(command, &ellpack_copy_dispatches, &ellpack_gpu_time_ns);
      }
      written_ += input.rows_;
    }
  }
  void Commit(double host_ratio, float snappy_ratio, bool allow_decomp_fallback) override {
    @autoreleasepool {
      CHECK(!committed_);
      CHECK(!adapter_active_);
      CHECK_EQ(written_, rows_);
      CHECK_GE(host_ratio, 0.0);
      CHECK_LE(host_ratio, 1.0);
      host_bytes_ = host_ratio == 0   ? 0
                    : host_ratio == 1 ? bytes_
                                      : std::max(std::size_t(bytes_ * host_ratio), std::size_t{1});
      auto& rt = GetRuntime();
      std::size_t compressed = 0;
      // IsDenseCompressed in CUDA includes feature-relative sparse pages.
      if (allow_decomp_fallback && info_.row_stride != info_.features) {
        auto ratio = std::isnan(snappy_ratio) ? 0.4f : snappy_ratio;
        CHECK_GE(ratio, 0.0f);
        CHECK_LE(ratio, 1.0f);
        compressed = std::min(host_bytes_, std::size_t(host_bytes_ * ratio));
      }
      plain_host_bytes_ = host_bytes_ - compressed;
      if (compressed)
        snappy_ = std::make_unique<MetalSnappy>(
            CompressSnappyBuffer(rt, buffer_, plain_host_bytes_, compressed, 1 << 21));
      head_ = Allocate(rt, plain_host_bytes_);
      if (!host_bytes_) {
        tail_ = buffer_;  // The complete cache page stays private to the GPU.
      } else {
        tail_ = EllpackPrivate(rt, bytes_ - host_bytes_);
        auto command = [rt.queue commandBuffer];
        auto blit = [command blitCommandEncoder];
        if (plain_host_bytes_)
          [blit copyFromBuffer:buffer_ sourceOffset:0 toBuffer:head_
            destinationOffset:0 size:plain_host_bytes_];
        if (bytes_ > host_bytes_) {
          [blit copyFromBuffer:buffer_
                   sourceOffset:host_bytes_
                       toBuffer:tail_
              destinationOffset:0
                           size:bytes_ - host_bytes_];
        }
        [blit endEncoding];
        CompleteOperation(command, &ellpack_cache_dispatches, &ellpack_gpu_time_ns);
      }
      buffer_ = nil;
      committed_ = true;
      ellpack_cached_host_bytes.fetch_add(host_bytes_, std::memory_order_relaxed);
      ellpack_cached_device_bytes.fetch_add(bytes_ - host_bytes_, std::memory_order_relaxed);
    }
  }
  void View(HistogramCuts const& cuts, bst_bin_t max_bin, bool prefetch_copy,
            GHistIndexMatrix* output) const override {
    @autoreleasepool {
      CHECK(output->row_ptr.empty());
      CHECK(!output->metal_page);
      CHECK_EQ(cuts.NumFeatures(), info_.features);
      CHECK(!adapter_active_);
      CHECK_EQ(written_, rows_);
      // Per-bin hits cannot exceed the row count. Resource limits are tighter on
      // supported Apple GPUs, but make the 32-bit counter contract explicit.
      CHECK_LE(rows_, std::numeric_limits<std::uint32_t>::max());
      auto& rt = GetRuntime();
      auto view = Read(prefetch_copy);
      output->cut = cuts;
      output->base_rowid = base_;
      output->max_numeric_bins_per_feat = max_bin;
      output->SetDense(info_.dense);
      output->SetMetalRows(rows_);
      output->metal_page = std::make_shared<Page>(rt, cuts, info_, view.head, view.tail,
                                                view.params.split, bytes_);
      ellpack_view_pages.fetch_add(1, std::memory_order_relaxed);
    }
  }
  bst_idx_t NonMissing() const override {
    @autoreleasepool {
      CHECK(!adapter_active_);
      CHECK_EQ(written_, rows_);
      auto view = Read();
      auto rows = EllpackRowPtr(GetRuntime(), view.params, view.head, view.tail,
                                &ellpack_count_dispatches);
      return static_cast<std::uint64_t const*>(rows.contents)[rows_];
    }
  }
  void Restore(Context const* ctx, HistogramCuts const& cuts, bst_bin_t max_bin,
               double sparse_thresh, bool prefetch_copy, GHistIndexMatrix* output) const override {
    this->View(cuts, max_bin, prefetch_copy, output);
    MaterializeHost(ctx, sparse_thresh, output);
  }
  std::vector<std::uint8_t> Readback() const override {
    @autoreleasepool {
      auto& rt = GetRuntime();
      auto view = Read();
      auto result = Allocate(rt, bytes_);
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      auto split = view.params.split;
      if (split)
        [blit copyFromBuffer:view.head sourceOffset:0 toBuffer:result destinationOffset:0 size:split];
      if (bytes_ > split)
        [blit copyFromBuffer:view.tail
                 sourceOffset:0
                     toBuffer:result
            destinationOffset:split
                         size:bytes_ - split];
      [blit endEncoding];
      CompleteOperation(command, &ellpack_cache_dispatches, &ellpack_gpu_time_ns);
      ellpack_debug_readback_bytes.fetch_add(bytes_, std::memory_order_relaxed);
      auto ptr = static_cast<std::uint8_t const*>(result.contents);
      return {ptr, ptr + bytes_};
    }
  }
};
}  // namespace
std::size_t WriteCompressedPage(GHistIndexMatrix const& input, AlignedFileWriteStream* output) {
  CHECK(output);
  auto page = std::dynamic_pointer_cast<Page>(std::atomic_load(&input.metal_page));
  CHECK(page && page->IsCompressed());
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto const& p = page->compressed;
    auto bytes = page->compressed_bytes;
    CHECK_LE(p.split, bytes);
    auto buffer = Allocate(rt, bytes);
    auto command = [rt.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    if (p.split)
      [blit copyFromBuffer:page->index sourceOffset:0 toBuffer:buffer destinationOffset:0 size:p.split];
    if (bytes > p.split)
      [blit copyFromBuffer:page->tail sourceOffset:0 toBuffer:buffer destinationOffset:p.split
                     size:bytes - p.split];
    [blit endEncoding];
    CompleteOperation(command, &ellpack_io_dispatches, &ellpack_gpu_time_ns);
    bst_idx_t rows = input.Size(), stride = p.stride, base = input.base_rowid, symbols = p.symbols;
    bool dense = p.dense;
    std::uint64_t count = bytes;
    auto written = output->Write(rows);
    written += output->Write(dense);
    written += output->Write(stride);
    written += output->Write(count);
    written += output->Write(buffer.contents, bytes);
    written += output->Write(base);
    written += output->Write(symbols);
    ellpack_io_write_pages.fetch_add(1, std::memory_order_relaxed);
    ellpack_io_write_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return written;
  }
}
bool ReadCompressedPage(AlignedResourceReadStream* input, HistogramCuts const& cuts,
                        bst_bin_t max_bin, bool prefetch_copy, GHistIndexMatrix* output) {
  CHECK(input && output);
  CHECK(output->row_ptr.empty() && !output->metal_page);
  bst_idx_t rows = 0, stride = 0, base = 0, symbols = 0;
  std::uint8_t dense = 0;
  std::uint64_t bytes = 0;
  // Read bool as a byte so an invalid representation never becomes a C++ bool.
  if (!input->Read(&rows) || !input->Read(&dense) || !input->Read(&stride) || !input->Read(&bytes))
    return false;
  CHECK_LE(dense, 1);
  CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
  CHECK_LE(stride, cuts.NumFeatures());
  auto& rt = GetRuntime();
  CHECK_LE(bytes, rt.device.maxBufferLength);
  auto [data, available] = input->Consume(bytes);
  if (available != bytes || !input->Read(&base) || !input->Read(&symbols)) return false;
  CHECK_LE(base, std::numeric_limits<bst_idx_t>::max() - rows);
  auto info = MakeEllpackInfo(cuts, stride, dense != 0);
  CHECK_EQ(stride, info.row_stride);
  CHECK_EQ(symbols, info.symbols);
  CHECK_GE(bytes, EllpackBytes(rows, info));
  @autoreleasepool {
    // The buffer owns its bytes, independently of the mapped reader's lifetime.
    auto shared = Upload(rt, data, bytes);
    auto buffer = shared;
    if (prefetch_copy) {
      buffer = EllpackPrivate(rt, bytes);
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      [blit copyFromBuffer:shared sourceOffset:0 toBuffer:buffer destinationOffset:0 size:bytes];
      [blit endEncoding];
      CompleteOperation(command, &ellpack_io_dispatches, &ellpack_gpu_time_ns);
    }
    auto view = std::make_shared<Page>(rt, cuts, info, buffer, buffer, 0, bytes);
    output->cut = cuts;
    output->base_rowid = base;
    output->max_numeric_bins_per_feat = max_bin;
    output->SetDense(dense != 0);
    output->SetMetalRows(rows);
    output->metal_page = std::move(view);
    ellpack_view_pages.fetch_add(1, std::memory_order_relaxed);
    ellpack_io_read_pages.fetch_add(1, std::memory_order_relaxed);
    ellpack_io_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return true;
}
bst_idx_t CompressedNonMissing(GHistIndexMatrix const& input) {
  @autoreleasepool {
    auto page = std::dynamic_pointer_cast<Page>(std::atomic_load(&input.metal_page));
    CHECK(page && page->IsCompressed());
    auto const& c = page->compressed;
    EllpackParams p{input.Size(), c.split, 0, c.features, c.stride, c.symbols, c.bits, c.dense, 4};
    auto rows = EllpackRowPtr(GetRuntime(), p, page->index, page->tail, &ellpack_count_dispatches);
    return static_cast<std::uint64_t const*>(rows.contents)[input.Size()];
  }
}
void MaterializeHost(Context const* ctx, double sparse_thresh, GHistIndexMatrix* output) {
  if (output->HostReadable()) return;
  @autoreleasepool {
    auto cached = std::atomic_load(&output->metal_page);
    auto page = std::dynamic_pointer_cast<Page>(cached);
    CHECK(page && page->IsCompressed());
    auto& rt = GetRuntime();
    auto rows = output->Size();
    auto const& cuts = output->cut;
    auto const& c = page->compressed;
    EllpackParams p{rows, c.split, 0, c.features, c.stride, c.symbols, c.bits, c.dense, 4};
    auto row_ptr = EllpackRowPtr(rt, p, page->index, page->tail, &ellpack_restore_dispatches);
    auto nnz = static_cast<std::uint64_t const*>(row_ptr.contents)[rows];
    output->row_ptr = common::MakeFixedVecWithMalloc(rows + 1, std::size_t{0});
    std::memcpy(output->row_ptr.data(), row_ptr.contents, (rows + 1) * 8);
    output->hit_count = common::MakeFixedVecWithMalloc(cuts.TotalBins(), std::size_t{0});
    output->ResizeIndex(ctx, nnz, output->IsDense());
    if (output->IsDense()) output->index.SetBinOffset(output->cut.Ptrs());
    p.width = output->index.GetBinTypeSize();
    auto index = Allocate(rt, nnz * p.width);
    auto hits = Allocate(rt, cuts.TotalBins() * 4);
    auto command = [rt.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    [blit fillBuffer:hits range:NSMakeRange(0, hits.length) value:0];
    [blit endEncoding];
    auto e = [command computeCommandEncoder];
    [e setComputePipelineState:rt.ellpack_unpack_pipeline];
    [e setBuffer:page->index offset:0 atIndex:0];
    [e setBuffer:page->tail offset:0 atIndex:1];
    [e setBuffer:row_ptr offset:0 atIndex:2];
    [e setBuffer:page->offsets offset:0 atIndex:3];
    [e setBuffer:index offset:0 atIndex:4];
    [e setBuffer:hits offset:0 atIndex:5];
    [e setBytes:&p length:sizeof(p) atIndex:6];
    [e dispatchThreads:MTLSizeMake(std::max<bst_idx_t>(1, rows), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &ellpack_restore_dispatches, &ellpack_gpu_time_ns);
    if (nnz) std::memcpy(output->index.data<std::uint8_t>(), index.contents, nnz * p.width);
    for (std::size_t i = 0; i < output->hit_count.size(); ++i)
      output->hit_count[i] = static_cast<std::uint32_t const*>(hits.contents)[i];
    if (!std::isnan(sparse_thresh)) output->InitColumns(ctx, sparse_thresh);
    // ResizeIndex invalidates a stale device cache. These bins came from this
    // exact compressed view, so retain it rather than re-uploading host data.
    std::atomic_store(&output->metal_page, std::move(cached));
    ellpack_host_materializations.fetch_add(1, std::memory_order_relaxed);
    ellpack_host_readback_bytes.fetch_add((rows + 1) * 8 + nnz * p.width + cuts.TotalBins() * 4,
                                         std::memory_order_relaxed);
  }
}
EllpackInfo MakeEllpackInfo(HistogramCuts const& cuts, bst_idx_t row_stride, bool dense) {
  @autoreleasepool {
    CHECK_LE(row_stride, cuts.NumFeatures());
    CHECK_LT(cuts.TotalBins(), std::numeric_limits<std::uint32_t>::max());
    std::uint32_t symbols = cuts.TotalBins() + 1;
    if (symbols == 1) return {cuts.NumFeatures(), 0, symbols, 1, dense};
    auto& rt = GetRuntime();
    std::uint32_t zero = 0, features = cuts.NumFeatures();
    auto ptrs = Upload(rt, cuts.Ptrs().data(), cuts.Ptrs().size() * 4);
    auto max_bins = Upload(rt, &zero, 4);
    auto command = [rt.queue commandBuffer];
    auto e = [command computeCommandEncoder];
    [e setComputePipelineState:rt.ellpack_symbols_pipeline];
    [e setBuffer:ptrs offset:0 atIndex:0];
    [e setBuffer:max_bins offset:0 atIndex:1];
    [e setBytes:&features length:sizeof(features) atIndex:2];
    [e dispatchThreads:MTLSizeMake(features, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &ellpack_pack_dispatches, &ellpack_gpu_time_ns);
    auto maximum = *static_cast<std::uint32_t const*>(max_bins.contents);
    if (dense) {
      row_stride = features;
      symbols = maximum;
    } else if (row_stride == features)
      symbols = maximum + 1;
    return {features, static_cast<std::uint32_t>(row_stride), symbols,
            common::detail::SymbolBits(symbols), dense};
  }
}
std::size_t EllpackBytes(bst_idx_t rows, EllpackInfo info) {
  CHECK_GT(info.symbols, 0);
  CHECK_EQ(info.bits, common::detail::SymbolBits(info.symbols));
  if (info.row_stride)
    CHECK_LE(rows, std::numeric_limits<std::size_t>::max() / info.row_stride / info.bits);
  return common::CompressedBufferWriter::CalculateBufferSize(rows * info.row_stride, info.symbols);
}
std::uint64_t CacheWorkingSetBytes() { return GetRuntime().device.recommendedMaxWorkingSetSize; }
std::shared_ptr<CompressedPage> MakeCompressedPage(EllpackInfo info, bst_idx_t rows,
                                                   bst_idx_t base_row, std::size_t bytes) {
  return std::make_shared<MetalCompressedPage>(info, rows, base_row, bytes);
}
EllpackCachePlan PlanEllpackCache(EllpackInfo info, Span<bst_idx_t const> batch_rows,
                                  bool validation, double ratio, std::int64_t minimum,
                                  std::uint64_t device_bytes) {
  CHECK(!batch_rows.empty());
  if (!std::isnan(ratio)) {
    CHECK_GE(ratio, 0.0);
    CHECK_LE(ratio, 1.0);
  }
  std::size_t total = 0;
  for (auto rows : batch_rows) {
    auto bytes = EllpackBytes(rows, info);
    CHECK_LE(bytes, std::numeric_limits<std::size_t>::max() - total);
    total += bytes;
  }
  if (minimum == -1) minimum = device_bytes / 16;
  auto available = device_bytes / 2;
  if (std::isnan(ratio)) {
    if (validation)
      ratio = 1.0;
    else if (total <= available)
      ratio = 0.0;
    else
      ratio = double(total - available * 0.85) / double(total);
    // Apple unified memory has no NVIDIA C2C link-count branch.
  }
  EllpackCachePlan plan{ratio, minimum, {}, {}, {}};
  for (auto rows : batch_rows) {
    auto bytes = EllpackBytes(rows, info);
    if (plan.bytes.empty() || static_cast<std::int64_t>(plan.bytes.back()) >= minimum) {
      plan.bytes.push_back(bytes);
      plan.rows.push_back(rows);
    } else {
      plan.bytes.back() += bytes;
      plan.rows.back() += rows;
    }
    plan.mapping.push_back(plan.bytes.size() - 1);
  }
  if (plan.rows.size() == 1) plan.host_ratio = 0.0;
  return plan;
}

namespace {
struct PredictionModelParams {
  std::uint64_t leaf_stride_row{0}, leaf_stride_target{0};
  std::uint32_t nodes{0}, categories{0}, leaves{0}, targets{0};
  std::uint32_t root{0}, category_begin{0}, leaf_begin{0}, tree{0};
  std::uint32_t multi{0}, has_categories{0};
  float weight{1};
  std::uint32_t padding{0};
};

BorrowedMemory PredictionModelInput(Runtime const& rt, void const* data, std::size_t bytes,
                                    std::size_t alignment = 4) {
  auto input = TryBorrowMemory(rt.device, data, bytes, alignment);
  if (input.buffer) {
    prediction_model_shared_input_bytes.fetch_add(bytes, std::memory_order_relaxed);
  } else {
    input = {Upload(rt, data, bytes), 0};
    prediction_model_staging_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  return input;
}

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
  MetalPrediction(Span<PredictionTreeView const> views, Span<bst_target_t const> groups,
                  OptionalWeights weights) : rt_{GetRuntime()}, queue_{rt_.queue} {
    static_assert(sizeof(RegTree::Node) == 20 && std::is_standard_layout_v<RegTree::Node> &&
                  std::is_trivially_copyable_v<RegTree::Node>);
    static_assert(sizeof(RegTree::CategoricalSplitMatrix::Segment) == 16 &&
                  sizeof(FeatureType) == 1 && sizeof(bst_target_t) == 4);
    static_assert(sizeof(PredictionModelParams) == 64);
    CHECK_EQ(views.size(), groups.size());
    CHECK(weights.Empty() || weights.Size() == views.size());
    auto checked = [](std::size_t n) {
      CHECK_LE(n, std::numeric_limits<std::uint32_t>::max());
      return static_cast<std::uint32_t>(n);
    };
    n_trees_ = checked(views.size());
    std::size_t nodes = 0, categories = 0, leaves = 0;
    std::vector<PredictionModelParams> params;
    for (std::size_t t = 0; t < views.size(); ++t) {
      std::visit([&](auto const& view) {
        PredictionModelParams p;
        p.nodes = checked(view.Size());
        CHECK_GT(p.nodes, 0);
        p.targets = checked(view.NumTargets());
        CHECK_GT(p.targets, 0);
        auto const& cats = view.GetCategoriesMatrix();
        p.categories = checked(cats.categories.size());
        p.has_categories = !cats.categories.empty();
        if (p.has_categories) {
          CHECK_EQ(cats.split_type.size(), p.nodes);
          CHECK_EQ(cats.node_ptr.size(), p.nodes);
        }
        p.leaves = p.nodes;
        if constexpr (std::is_same_v<std::decay_t<decltype(view)>, tree::MultiTargetTreeView>) {
          p.multi = 1;
          p.leaves = checked(view.leaf_weights.Size());
          p.leaf_stride_row = view.leaf_weights.Stride(0);
          p.leaf_stride_target = view.leaf_weights.Stride(1);
        }
        p.root = checked(nodes);
        p.category_begin = checked(categories);
        p.leaf_begin = checked(leaves);
        p.tree = t;
        p.weight = weights[t];
        nodes += p.nodes;
        categories += p.categories;
        leaves += p.leaves;
        checked(nodes);
        checked(categories);
        checked(leaves);
        params.push_back(p);
      }, views[t]);
    }
    @autoreleasepool {
      nodes_ = Allocate(rt_, nodes * sizeof(PredictionNode));
      trees_ = Allocate(rt_, views.size() * sizeof(PredictionTree));
      categories_ = Allocate(rt_, categories * sizeof(std::uint32_t));
      leaves_ = Allocate(rt_, leaves * sizeof(float));
      if (views.empty()) return;
      auto tree_groups = PredictionModelInput(rt_, groups.data(), groups.size_bytes());
      auto command = [queue_ commandBuffer];
      CHECK(command);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt_.prediction_model_pipeline];
      [e setBuffer:tree_groups.buffer offset:tree_groups.offset atIndex:10];
      [e setBuffer:nodes_ offset:0 atIndex:11];
      [e setBuffer:trees_ offset:0 atIndex:12];
      [e setBuffer:categories_ offset:0 atIndex:13];
      [e setBuffer:leaves_ offset:0 atIndex:14];
      auto empty = Allocate(rt_, sizeof(RegTree::Node));
      for (std::size_t t = 0; t < views.size(); ++t) {
        // Unused arguments must still have valid bindings for Metal validation.
        for (unsigned i = 0; i < 10; ++i) [e setBuffer:empty offset:0 atIndex:i];
        auto bind = [&](unsigned index, void const* data, std::size_t bytes,
                        std::size_t alignment = 4) {
          auto input = PredictionModelInput(rt_, data, bytes, alignment);
          [e setBuffer:input.buffer offset:input.offset atIndex:index];
        };
        auto const& p = params[t];
        std::visit([&](auto const& view) {
          if constexpr (std::is_same_v<std::decay_t<decltype(view)>, tree::ScalarTreeView>) {
            bind(0, view.nodes, p.nodes * sizeof(RegTree::Node));
          } else {
            bind(1, view.left, p.nodes * sizeof(bst_node_t));
            bind(2, view.right, p.nodes * sizeof(bst_node_t));
            bind(3, view.split_index, p.nodes * sizeof(bst_feature_t));
            bind(4, view.default_left, p.nodes, 1);
            bind(5, view.split_conds, p.nodes * sizeof(float));
            auto data = view.leaf_weights.Values();
            bind(6, data.data(), data.size_bytes());
          }
          if (p.has_categories) {
            auto const& cats = view.GetCategoriesMatrix();
            bind(7, cats.split_type.data(), cats.split_type.size_bytes(), 1);
            bind(8, cats.categories.data(), cats.categories.size_bytes());
            bind(9, cats.node_ptr.data(), cats.node_ptr.size_bytes(), 8);
          }
        }, views[t]);
        [e setBytes:&p length:sizeof(p) atIndex:15];
        auto count = std::max({p.nodes, p.categories, p.multi ? p.leaves : 0u});
        [e dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [e endEncoding];
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      prediction_model_pack_dispatches.fetch_add(views.size(), std::memory_order_relaxed);
    }
  }
  void Predict(InputBatch const& data, std::uint32_t groups, Span<float> predictions,
               bool leaf) override {
    auto const* input = dynamic_cast<MetalInputBatch const*>(&data);
    CHECK(input);
    Run({input->values, 0}, data.Rows(), data.Features(), groups, predictions, leaf);
  }
  void Predict(Span<float const> data, std::size_t rows, std::uint32_t features,
               std::uint32_t groups, Span<float> predictions, bool leaf) override {
    CHECK_EQ(data.size(), rows * features);
    @autoreleasepool {
      Run(PredictionInput(rt_, data.data(), data.size_bytes()), rows, features, groups, predictions,
          leaf, MemoryRangesOverlap(data.data(), data.size_bytes(), predictions.data(), predictions.size_bytes()));
    }
  }
  void Run(BorrowedMemory input, std::size_t rows, std::uint32_t features, std::uint32_t groups,
           Span<float> predictions, bool leaf, bool snapshot = false) {
    CHECK_EQ(predictions.size(), rows * groups);
    CHECK_LE(rows, std::numeric_limits<std::uint32_t>::max());
    if (!rows || !n_trees_) return;
    @autoreleasepool {
      PredictionOutput output{rt_, predictions, true};
      auto command = [queue_ commandBuffer];
      CHECK(command);
      if (snapshot) input = PredictionSnapshot(rt_, command, input, rows * features * sizeof(float));
      auto encoder = [command computeCommandEncoder];
      CHECK(encoder);
      struct {
        std::uint64_t rows;
        std::uint32_t features, groups, trees, padding;
      } p{rows, features, groups, n_trees_, static_cast<std::uint32_t>(leaf)};
      [encoder setComputePipelineState:rt_.prediction_pipeline];
      [encoder setBuffer:input.buffer offset:input.offset atIndex:0];
      [encoder setBuffer:nodes_ offset:0 atIndex:1];
      [encoder setBuffer:trees_ offset:0 atIndex:2];
      [encoder setBuffer:categories_ offset:0 atIndex:3];
      [encoder setBuffer:leaves_ offset:0 atIndex:4];
      [encoder setBuffer:output.memory.buffer offset:output.memory.offset atIndex:5];
      [encoder setBytes:&p length:sizeof(p) atIndex:6];
      [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
      output.Publish(predictions.size());
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
std::unique_ptr<Prediction> MakePrediction(Span<PredictionTreeView const> trees,
                                         Span<bst_target_t const> groups,
                                         OptionalWeights weights) {
  return std::make_unique<MetalPrediction>(trees, groups, weights);
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
  static constexpr std::size_t kFrameBytes = 84;

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
  void Predict(InputBatch const& data, Span<float const> base_margin, Span<float> output) override {
    auto const* input = dynamic_cast<MetalInputBatch const*>(&data);
    CHECK(input);
    CHECK_EQ(data.Features(), params_.features);
    Run({input->values, 0}, data.Rows(), base_margin, output);
  }
  void Predict(Span<float const> data, std::size_t rows, Span<float const> base_margin,
               Span<float> output) override {
    CHECK_EQ(data.size(), rows * params_.features);
    @autoreleasepool {
      Run(PredictionInput(rt_, data.data(), data.size_bytes()), rows, base_margin, output,
          MemoryRangesOverlap(data.data(), data.size_bytes(), output.data(), output.size_bytes()));
    }
  }
  void Run(BorrowedMemory input, std::size_t rows, Span<float const> base_margin,
           Span<float> output, bool snapshot = false) {
    CHECK_LE(rows, batch_rows_);
    CHECK_EQ(output.size_bytes(), rows * row_output_);
    CHECK(base_margin.empty() || base_margin.size() == rows * params_.groups);
    if (!rows) return;
    @autoreleasepool {
      PredictionOutput result{rt_, output, false};
      BorrowedMemory bases{bases_, 0};
      if (!base_margin.empty()) {
        bases = PredictionInput(rt_, base_margin.data(), base_margin.size_bytes());
      }
      auto frames = Allocate(rt_, rows * row_stack_);
      auto temporary =
          Allocate(rt_, rows * params_.groups * (std::size_t(params_.features) + 1) * 4);
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      if (snapshot) input = PredictionSnapshot(rt_, command, input, rows * params_.features * sizeof(float));
      if (MemoryRangesOverlap(base_margin.data(), base_margin.size_bytes(), output.data(),
                              output.size_bytes())) {
        bases = PredictionSnapshot(rt_, command, bases, base_margin.size_bytes());
      }
      auto e = [command computeCommandEncoder];
      CHECK(e);
      auto p = params_;
      p.margin = !base_margin.empty();
      [e setComputePipelineState:rt_.shap_pipeline];
      [e setBuffer:input.buffer offset:input.offset atIndex:0];
      [e setBuffer:nodes_ offset:0 atIndex:1];
      [e setBuffer:trees_ offset:0 atIndex:2];
      [e setBuffer:categories_ offset:0 atIndex:3];
      [e setBuffer:leaves_ offset:0 atIndex:4];
      [e setBuffer:roots_ offset:0 atIndex:5];
      [e setBuffer:bases.buffer offset:bases.offset atIndex:6];
      [e setBuffer:result.memory.buffer offset:result.memory.offset atIndex:7];
      [e setBuffer:frames offset:0 atIndex:8];
      [e setBuffer:temporary offset:0 atIndex:9];
      [e setBytes:&p length:sizeof(p) atIndex:10];
      [e dispatchThreads:MTLSizeMake(rows * params_.groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [e endEncoding];
      CompleteOperation(command,
                        params_.interactions ? &shap_interaction_dispatches : &shap_dispatches,
                        &shap_gpu_time_ns);
      result.Publish(output.size());
    }
  }
};
}  // namespace
std::unique_ptr<Shap> MakeShap(ShapModel const& model, std::uint32_t features, std::uint32_t groups,
                               bool interactions) {
  return std::make_unique<MetalShap>(model, features, groups, interactions);
}

void AddLeafValues(Span<std::int32_t const> ids, Span<float const> weights, std::uint32_t targets,
                   std::size_t stride, Span<float> output,
                   Span<std::int32_t const> leaf_mapping) {
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
    auto input = PredictionInput(rt, ids.data(), ids.size_bytes());
    auto leaves = PredictionInput(rt, weights.data(), weights.size_bytes());
    auto mapping = leaf_mapping.empty()
                       ? input
                       : PredictionInput(rt, leaf_mapping.data(), leaf_mapping.size_bytes());
    PredictionOutput out{rt, output, true};
    if (MemoryRangesOverlap(ids.data(), ids.size_bytes(), output.data(), output.size_bytes()))
      input = PredictionSnapshot(rt, command, input, ids.size_bytes());
    if (MemoryRangesOverlap(weights.data(), weights.size_bytes(), output.data(), output.size_bytes()))
      leaves = PredictionSnapshot(rt, command, leaves, weights.size_bytes());
    if (MemoryRangesOverlap(leaf_mapping.data(), leaf_mapping.size_bytes(), output.data(),
                            output.size_bytes()))
      mapping = PredictionSnapshot(rt, command, mapping, leaf_mapping.size_bytes());
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    struct {
      std::uint64_t rows, stride;
      std::uint32_t targets, mapped;
    } p{ids.size(), stride, targets, static_cast<std::uint32_t>(!leaf_mapping.empty())};
    [encoder setComputePipelineState:rt.leaf_pipeline];
    [encoder setBuffer:input.buffer offset:input.offset atIndex:0];
    [encoder setBuffer:leaves.buffer offset:leaves.offset atIndex:1];
    [encoder setBuffer:out.memory.buffer offset:out.memory.offset atIndex:2];
    [encoder setBytes:&p length:sizeof(p) atIndex:3];
    [encoder setBuffer:mapping.buffer offset:mapping.offset atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(ids.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    out.Publish(output.size());
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
  if (multi) {
    CHECK_LE(targets, std::numeric_limits<std::int32_t>::max());
  } else {
    CHECK_EQ(labels.size() % targets, 0);
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto queue = rt.queue;
    CHECK(queue);
    auto command = [queue commandBuffer];
    CHECK(command);
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    auto pred = MetricInput(rt, predictions.data(), predictions.size_bytes());
    auto label = MetricInput(rt, labels.data(), labels.size_bytes());
    auto weight = MetricInput(rt, weights.data(), weights.size_bytes());
    auto groups = (labels.size() + 255) / 256;
    auto partial = Allocate(rt, groups * 2 * sizeof(double));
    std::uint32_t zero = 0;
    auto invalid = Upload(rt, &zero, sizeof(zero));
    struct {
      std::uint64_t n;
      std::uint32_t targets, kind, weighted, padding;
      float parameter, reserved;
    } p{labels.size(), targets, kind, !weights.empty(), 0, parameter, 0};
    [encoder setComputePipelineState:rt.metric_pipeline];
    [encoder setBuffer:pred.buffer offset:pred.offset atIndex:0];
    [encoder setBuffer:label.buffer offset:label.offset atIndex:1];
    [encoder setBuffer:weight.buffer offset:weight.offset atIndex:2];
    [encoder setBuffer:partial offset:0 atIndex:3];
    [encoder setBytes:&p length:sizeof(p) atIndex:4];
    [encoder setBuffer:invalid offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    auto reduced = EncodeMetricReduction(rt, command, partial, groups);
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted) << ErrorText(command.error);
    if (multi) metric_label_dispatches.fetch_add(1, std::memory_order_relaxed);
    CHECK_EQ(*static_cast<std::uint32_t const*>(invalid.contents), 0)
        << "MultiClassEvaluation: label must be in [0, num_class) after integer conversion.";
    metric_dispatches.fetch_add(1, std::memory_order_relaxed);
    auto elapsed = command.GPUEndTime - command.GPUStartTime;
    if (elapsed > 0)
      metric_gpu_time_ns.fetch_add(static_cast<std::uint64_t>(elapsed * 1e9),
                                   std::memory_order_relaxed);
    return ReadMetricReduction(reduced);
  }
}

// The resident mode is private to EvaluateFeatureSplits, whose GPU builder
// validates IDs/cuts and creates the ranges. Diagnostic wrappers retain full
// host descriptor validation; resident mode never dereferences task payloads.
static void RecordQuantizedSplitDispatch(bool vector_leaf) {
  split_dispatches.fetch_add(1, std::memory_order_relaxed);
  resident_split_dispatches.fetch_add(1, std::memory_order_relaxed);
  (vector_leaf ? cuda_integer_multi_split_dispatches : cuda_integer_split_dispatches)
      .fetch_add(1, std::memory_order_relaxed);
}
static void EvaluateQuantizedSplitsImpl(
    Span<HistogramInput const> histograms, Span<QuantizedSplitTask const> tasks,
    SplitParameters const& parameters, std::vector<QuantizedSplitResult>* results,
    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace,
    std::optional<std::size_t> resident_bins, id<MTLCommandBuffer> pending = nil) {
  CHECK(!pending || resident_bins.has_value());
  if (resident_bins) CHECK(!results && !sorted_categories);
  if (workspace) static_cast<MetalSplitWorkspace*>(workspace)->ResetCandidates();
  static_assert(sizeof(QuantizedSplitTask) == 48 && sizeof(QuantizedSplitResult) == 80);
  static_assert(sizeof(GradientPairInt64) == 16);
  if (results) results->resize(tasks.size());
  auto bins = resident_bins ? *resident_bins : (tasks.empty() ? 0 : tasks.back().end);
  bool categorical = resident_bins.has_value() ||
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
  if (!resident_bins) {
    for (auto const& task : tasks) {
      CHECK_EQ(task.begin, previous);
      CHECK_GE(task.end, task.begin);
      CHECK_LE(task.end - task.begin, std::numeric_limits<std::int32_t>::max());
      CHECK_LE(task.kind, 2);
      previous = task.end;
    }
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
    auto command = pending ? pending : [rt.queue commandBuffer];
    CHECK(command);
    auto hist = resident_bins ? scratch->hist
                              : GatherHistograms(rt, histograms, bins, command, scratch, true);
    auto input = Scratch(rt, &scratch->input, tasks.size_bytes(), resident_bins ? nullptr : tasks.data());
    auto output = Scratch(rt, &scratch->output, tasks.size() * sizeof(QuantizedSplitResult));
    auto order = Scratch(rt, &scratch->order, (categorical ? bins : 0) * sizeof(std::uint32_t));
    // Numeric screening reuses this buffer for upper gain bounds.
    auto weights = Scratch(rt, &scratch->weights, bins * sizeof(float));
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
    if (!pending) {
      CompleteOperation(command, nullptr, &split_gpu_time_ns);
      RecordQuantizedSplitDispatch(false);
    }
    if (results) std::memcpy(results->data(), output.contents, tasks.size() * sizeof(QuantizedSplitResult));
    if (sorted_categories) for (auto const& task : tasks)
      if (task.kind == 2) {
        std::memcpy(sorted_categories->data() + task.begin,
                    static_cast<std::uint32_t const*>(order.contents) + task.begin,
                    (task.end - task.begin) * sizeof(std::uint32_t));
      }
    scratch->candidate_count = tasks.size();
    scratch->vector_candidates = false;
  }
}

void EvaluateQuantizedSplits(Span<HistogramInput const> histograms,
                             Span<QuantizedSplitTask const> tasks,
                             SplitParameters const& parameters,
                             std::vector<QuantizedSplitResult>* results,
                             std::vector<std::uint32_t>* sorted_categories,
                             SplitWorkspace* workspace) {
  CHECK(results);
  EvaluateQuantizedSplitsImpl(histograms, tasks, parameters, results, sorted_categories,
                              workspace, std::nullopt);
}

static void ReadFinalizedSplitNodes(MetalSplitWorkspace const& scratch,
                                     std::vector<SplitNodeResult>* results,
                                     std::vector<SplitNodeWeights>* weights) {
  std::memcpy(results->data(), scratch.node_results.contents,
              results->size() * sizeof(SplitNodeResult));
  if (weights) {
    auto bytes = weights->size() * sizeof(SplitNodeWeights);
    std::memcpy(weights->data(), scratch.node_weights.contents, bytes);
    split_weight_readback_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
}
static void FinalizeSplitNodesImpl(
    Span<SplitFinalizeNode const> nodes, Span<SplitFinalizeFeature const> features,
    Span<GradientPairPrecise const> parents, Span<SplitBounds const> bounds,
    SplitParameters const& parameters, bool vector_leaf, std::vector<SplitNodeResult>* results,
    std::vector<SplitNodeWeights>* weights, SplitWorkspace* workspace, bool resident_features,
    id<MTLCommandBuffer> pending = nil) {
  CHECK(!pending || resident_features);
  static_assert(sizeof(SplitFinalizeNode) == 16 && sizeof(SplitFinalizeFeature) == 16);
  static_assert(sizeof(SplitNodeResult) == 40 && sizeof(SplitNodeWeights) == 16);
  results->resize(nodes.size());
  if (weights) weights->resize(parents.size());
  if (workspace) static_cast<MetalSplitWorkspace*>(workspace)->finalized_nodes.clear();
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
    if (vector_leaf && !resident_features) {
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
    auto feature = Scratch(rt, &scratch->feature_input, features.size_bytes(),
                            resident_features ? nullptr : features.data());
    // Integer parents may still be queued for task construction. Precise
    // finalizer parents need separate staging when neither input is borrowable.
    auto parent = BindSplitSums(rt, &scratch->node_precise_parents, parents.data(), parents.size_bytes());
    auto bound = BindSplitBounds(rt, bounds.data(), bounds.size_bytes());
    auto out = Scratch(rt, &scratch->node_results, results->size() * sizeof(SplitNodeResult));
    auto weight = Scratch(rt, &scratch->node_weights, parents.size() * sizeof(SplitNodeWeights));
    auto command = pending ? pending : [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    std::uint32_t multi = vector_leaf;
    [e setComputePipelineState:rt.split_finalize_pipeline];
    [e setBuffer:candidates offset:0 atIndex:0];
    [e setBuffer:feature offset:0 atIndex:1];
    [e setBuffer:input offset:0 atIndex:2];
    [e setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [e setBuffer:parent.buffer offset:parent.offset atIndex:4];
    [e setBuffer:bound.buffer offset:bound.offset atIndex:5];
    [e setBuffer:children offset:0 atIndex:6];
    [e setBuffer:out offset:0 atIndex:7];
    [e setBuffer:weight offset:0 atIndex:8];
    [e setBytes:&multi length:sizeof(multi) atIndex:9];
    CHECK_EQ(rt.split_finalize_pipeline.threadExecutionWidth, 32);
    CHECK_GE(rt.split_finalize_pipeline.maxTotalThreadsPerThreadgroup, 32);
    [e dispatchThreadgroups:MTLSizeMake(nodes.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    if (!pending) {
      CompleteOperation(command, &split_finalize_dispatches, &split_gpu_time_ns);
      ReadFinalizedSplitNodes(*scratch, results, weights);
    }
    scratch->finalized_nodes.assign(nodes.begin(), nodes.end());
  }
}

void FinalizeSplitNodes(Span<SplitFinalizeNode const> nodes,
                        Span<SplitFinalizeFeature const> features,
                        Span<GradientPairPrecise const> parents, Span<SplitBounds const> bounds,
                        SplitParameters const& parameters, bool vector_leaf,
                        std::vector<SplitNodeResult>* results,
                        std::vector<SplitNodeWeights>* weights, SplitWorkspace* workspace) {
  FinalizeSplitNodesImpl(nodes, features, parents, bounds, parameters, vector_leaf, results,
                         weights, workspace, false);
}

namespace {
class MetalSplitSumStorage final : public SplitSumStorage {
  std::shared_ptr<QuantizedGradients const> gradients_;
  std::size_t targets_;
  std::unique_ptr<VectorBuffer> nodes_{MakeVectorBuffer()}, splits_{MakeVectorBuffer()},
      gathered_{MakeVectorBuffer()}, precise_{MakeVectorBuffer()};
  std::vector<bool> node_ready_, split_ready_;
  id<MTLBuffer> scales_;

  std::size_t Bytes() const { return targets_ * sizeof(GradientPairInt64); }
  BorrowedMemory Bind(VectorBuffer const& buffer) const {
    auto view = TryBorrowMemory(GetRuntime().device, buffer.Data(), buffer.SizeBytes(), 8);
    CHECK(view.buffer) << "Cannot bind owned Metal sum storage.";
    return view;
  }
  void CheckCapacity(std::size_t size, std::size_t fields = 1) const {
    CHECK_LE(size, GetRuntime().device.maxBufferLength / Bytes() / fields);
  }

 public:
  explicit MetalSplitSumStorage(std::shared_ptr<QuantizedGradients const> gradients)
      : gradients_{std::move(gradients)}, targets_{gradients_ ? gradients_->Targets() : 0} {
    CHECK_GT(targets_, 0);
    auto source = dynamic_cast<MetalQuantizedGradients const*>(gradients_.get());
    CHECK(source);
    auto& rt = GetRuntime();
    @autoreleasepool {
      scales_ = Allocate(rt, targets_ * sizeof(GradientScale));
      auto command = [rt.queue commandBuffer];
      auto e = [command blitCommandEncoder];
      CHECK(e);
      for (std::size_t t = 0; t < targets_; ++t)
        [e copyFromBuffer:source->columns[t]->gradient_scale_ sourceOffset:0 toBuffer:scales_
            destinationOffset:t * sizeof(GradientScale) size:sizeof(GradientScale)];
      [e endEncoding];
      CompleteOperation(command, nullptr, &split_gpu_time_ns);
    }
  }
  void InitRoot(Span<GradientPairInt64 const> root) override {
    CHECK(node_ready_.empty()) << "Root sums have already been initialized.";
    CHECK_EQ(root.size(), targets_);
    auto& rt = GetRuntime();
    @autoreleasepool {
      auto input = TryBorrowMemory(rt.device, root.data(), root.size_bytes(), 8);
      CHECK(input.buffer) << "Root sums must be resident on Metal.";
      nodes_->Resize(Bytes());
      auto out = Bind(*nodes_);
      auto command = [rt.queue commandBuffer];
      auto e = [command blitCommandEncoder];
      CHECK(e);
      [e copyFromBuffer:input.buffer sourceOffset:input.offset toBuffer:out.buffer
          destinationOffset:out.offset size:Bytes()];
      [e endEncoding];
      CompleteOperation(command, &split_sum_store_dispatches, &split_gpu_time_ns);
    }
    node_ready_.push_back(true);
  }
  void Store(SplitWorkspace const& workspace, Span<SplitSumCopy const> copies) override {
    auto const& source = static_cast<MetalSplitWorkspace const&>(workspace);
    auto size = split_ready_.size();
    std::vector<std::uint32_t> ids;
    for (auto const& copy : copies) {
      CHECK(source.vector_candidates);
      CHECK_EQ(source.quantized_multi_shapes.size(), source.candidate_count);
      CHECK_LT(copy.candidate, source.quantized_multi_shapes.size());
      CHECK(source.quantized_multi_valid.at(copy.candidate));
      auto const& task = source.quantized_multi_shapes[copy.candidate];
      CHECK_EQ(task.targets, targets_);
      CHECK_LE(2 * (std::uint64_t{task.parent_begin} + targets_), source.candidate_child_count);
      CHECK_LT(copy.node, node_ready_.size());
      CHECK(node_ready_[copy.node]);
      size = std::max(size, std::size_t(copy.node) + 1);
      ids.push_back(copy.node);
    }
    std::sort(ids.begin(), ids.end());
    CHECK(std::adjacent_find(ids.begin(), ids.end()) == ids.end()) << "Duplicate sum destinations.";
    CheckCapacity(size, 2);
    if (copies.empty()) return;
    splits_->Resize(size * 2 * Bytes());
    @autoreleasepool {
      auto out = Bind(*splits_);
      auto command = [GetRuntime().queue commandBuffer];
      auto e = [command blitCommandEncoder];
      CHECK(e);
      for (auto const& copy : copies) {
        auto begin = source.quantized_multi_shapes[copy.candidate].parent_begin;
        [e copyFromBuffer:source.integer_children sourceOffset:2 * std::size_t(begin) * sizeof(GradientPairInt64)
                toBuffer:out.buffer destinationOffset:out.offset + std::size_t(copy.node) * 2 * Bytes()
                    size:2 * Bytes()];
      }
      [e endEncoding];
      CompleteOperation(command, &split_sum_store_dispatches, &split_gpu_time_ns);
    }
    split_ready_.resize(size, false);
    for (auto id : ids) split_ready_[id] = true;
  }
  void Apply(std::uint32_t parent, std::uint32_t left, std::uint32_t right) override {
    CHECK_LT(parent, split_ready_.size());
    CHECK(split_ready_[parent]);
    CHECK_GT(left, parent);
    CHECK_GT(right, parent);
    CHECK_NE(left, right);
    CHECK(left >= node_ready_.size() || !node_ready_[left]);
    CHECK(right >= node_ready_.size() || !node_ready_[right]);
    auto size = std::max(node_ready_.size(), std::size_t(std::max(left, right)) + 1);
    CheckCapacity(size);
    nodes_->Resize(size * Bytes());
    @autoreleasepool {
      auto input = Bind(*splits_), out = Bind(*nodes_);
      auto command = [GetRuntime().queue commandBuffer];
      auto e = [command blitCommandEncoder];
      CHECK(e);
      [e copyFromBuffer:input.buffer sourceOffset:input.offset + std::size_t(parent) * 2 * Bytes()
              toBuffer:out.buffer destinationOffset:out.offset + std::size_t(left) * Bytes() size:Bytes()];
      [e copyFromBuffer:input.buffer sourceOffset:input.offset + (std::size_t(parent) * 2 + 1) * Bytes()
              toBuffer:out.buffer destinationOffset:out.offset + std::size_t(right) * Bytes() size:Bytes()];
      [e endEncoding];
      CompleteOperation(command, &split_sum_apply_dispatches, &split_gpu_time_ns);
    }
    node_ready_.resize(size, false);
    node_ready_[left] = node_ready_[right] = true;
  }
  SplitSumViews Gather(Span<std::uint32_t const> nodes) override {
    CheckCapacity(nodes.size());
    CHECK_LE(nodes.size(), std::numeric_limits<std::uint32_t>::max() / targets_);
    for (auto node : nodes) {
      CHECK_LT(node, node_ready_.size());
      CHECK(node_ready_[node]);
    }
    auto count = nodes.size() * targets_;
    gathered_->Resize(count * sizeof(GradientPairInt64));
    precise_->Resize(count * sizeof(GradientPairPrecise));
    if (count) {
      auto& rt = GetRuntime();
      @autoreleasepool {
        auto input = Bind(*nodes_), out = Bind(*gathered_), fp = Bind(*precise_);
        auto command = [rt.queue commandBuffer];
        auto e = [command blitCommandEncoder];
        CHECK(e);
        for (std::size_t i = 0; i < nodes.size(); ++i)
          [e copyFromBuffer:input.buffer sourceOffset:input.offset + std::size_t(nodes[i]) * Bytes()
                  toBuffer:out.buffer destinationOffset:out.offset + i * Bytes() size:Bytes()];
        [e endEncoding];
        EncodeIntegerStatistics(rt, command, out.buffer, fp.buffer, scales_, count, targets_,
                                out.offset, fp.offset);
        CompleteOperation(command, &split_sum_gather_dispatches, &split_gpu_time_ns);
      }
    }
    return {{static_cast<GradientPairInt64 const*>(gathered_->Data()), count},
            {static_cast<GradientPairPrecise const*>(precise_->Data()), count}};
  }
};

class MetalSplitWeightStorage final : public SplitWeightStorage {
  std::uint32_t targets_;
  std::unique_ptr<VectorBuffer> data_{MakeVectorBuffer()};
  std::vector<bool> ready_;

  Span<float const> View(std::uint32_t node, std::size_t field) const {
    CHECK_LT(node, ready_.size());
    CHECK(ready_[node]) << "No finalized weights for this node.";
    return {static_cast<float const*>(data_->Data()) + (std::size_t(node) * 3 + field) * targets_,
            targets_};
  }

 public:
  explicit MetalSplitWeightStorage(std::uint32_t targets) : targets_{targets} {
    CHECK_GT(targets, 0);
  }
  Span<float const> Base(std::uint32_t node) const override { return View(node, 0); }
  Span<float const> Left(std::uint32_t node) const override { return View(node, 1); }
  Span<float const> Right(std::uint32_t node) const override { return View(node, 2); }
  void Store(SplitWorkspace const& workspace, Span<SplitWeightCopy const> copies) override {
    static_assert(sizeof(SplitWeightCopy) == 8);
    auto const& source = static_cast<MetalSplitWorkspace const&>(workspace);
    auto& rt = GetRuntime();
    CHECK_LE(copies.size(), std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> ids;
    auto size = ready_.size();
    for (auto const& copy : copies) {
      CHECK_LT(copy.source, source.finalized_nodes.size());
      CHECK_EQ(source.finalized_nodes[copy.source].targets, targets_);
      size = std::max(size, std::size_t(copy.node) + 1);
      ids.push_back(copy.node);
    }
    std::sort(ids.begin(), ids.end());
    CHECK(std::adjacent_find(ids.begin(), ids.end()) == ids.end()) << "Duplicate weight destinations";
    CHECK_LE(size, rt.device.maxBufferLength / (std::uint64_t{targets_} * 3 * sizeof(float)));
    if (copies.empty()) return;
    data_->Resize(size * targets_ * 3 * sizeof(float));
    @autoreleasepool {
      auto output = TryBorrowMemory(rt.device, data_->Data(), data_->SizeBytes(), 4);
      CHECK(output.buffer) << "Cannot bind owned Metal weight storage.";
      auto requests = Upload(rt, copies.data(), copies.size_bytes());
      auto command = [rt.queue commandBuffer];
      auto encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:rt.split_weight_store_pipeline];
      [encoder setBuffer:source.node_weights offset:0 atIndex:0];
      [encoder setBuffer:source.node_input offset:0 atIndex:1];
      [encoder setBuffer:requests offset:0 atIndex:2];
      [encoder setBuffer:output.buffer offset:output.offset atIndex:3];
      [encoder setBytes:&targets_ length:sizeof(targets_) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(targets_, copies.size(), 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [encoder endEncoding];
      CompleteOperation(command, &split_weight_store_dispatches, &split_gpu_time_ns);
    }
    ready_.resize(size, false);
    for (auto id : ids) ready_[id] = true;
  }
};
}  // namespace
std::unique_ptr<SplitSumStorage> MakeSplitSumStorage(
    std::shared_ptr<QuantizedGradients const> gradients) {
  return std::make_unique<MetalSplitSumStorage>(std::move(gradients));
}
std::unique_ptr<SplitWeightStorage> MakeSplitWeightStorage(std::uint32_t targets) {
  return std::make_unique<MetalSplitWeightStorage>(targets);
}

void BuildSplitCategories(Span<float const> cuts, Span<SplitCategoryTask const> tasks,
                          std::uint32_t words_per_node, std::vector<std::uint32_t>* output,
                          SplitWorkspace* workspace) {
  static_assert(sizeof(SplitCategoryTask) == 8);
  if (workspace) {
    auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
    scratch->category_count = 0;
    scratch->category_words = 0;
  }
  if (tasks.empty()) {
    if (output) output->clear();
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
    auto values = BindFeatureInput(rt, cuts.data(), cuts.size_bytes());
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
    [encoder setBuffer:values.buffer offset:values.offset atIndex:4];
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
    if (output) {
      output->resize(count);
      std::memcpy(output->data(), bits.contents, count * sizeof(std::uint32_t));
      split_category_readback_bytes.fetch_add(count * sizeof(std::uint32_t),
                                               std::memory_order_relaxed);
    }
    scratch->category_count = tasks.size();
    scratch->category_words = words_per_node;
  }
}

namespace {
class MetalSplitCategoryStorage final : public SplitCategoryStorage {
  std::uint32_t words_;
  std::unique_ptr<VectorBuffer> data_{MakeVectorBuffer()};
  std::vector<bool> ready_;

 public:
  explicit MetalSplitCategoryStorage(std::uint32_t words) : words_{words} { CHECK_GT(words, 0); }
  Span<std::uint32_t const> Get(std::uint32_t node) const override {
    CHECK_LT(node, ready_.size());
    CHECK(ready_[node]) << "No selected categories for this node.";
    return {static_cast<std::uint32_t const*>(data_->Data()) + std::size_t(node) * words_, words_};
  }
  void Store(SplitWorkspace const& workspace, Span<SplitCategoryCopy const> copies) override {
    if (copies.empty()) return;
    auto const& source = static_cast<MetalSplitWorkspace const&>(workspace);
    CHECK_EQ(words_, source.category_words);
    std::vector<std::uint32_t> ids;
    auto size = ready_.size();
    for (auto const& copy : copies) {
      CHECK_LT(copy.source, source.category_count);
      size = std::max(size, std::size_t(copy.node) + 1);
      ids.push_back(copy.node);
    }
    std::sort(ids.begin(), ids.end());
    CHECK(std::adjacent_find(ids.begin(), ids.end()) == ids.end())
        << "Duplicate category destinations";
    auto& rt = GetRuntime();
    auto bytes_per_node = std::size_t{words_} * sizeof(std::uint32_t);
    CHECK_LE(size, rt.device.maxBufferLength / bytes_per_node);
    data_->Resize(size * bytes_per_node);
    @autoreleasepool {
      auto out = TryBorrowMemory(rt.device, data_->Data(), data_->SizeBytes(), 4);
      CHECK(out.buffer) << "Cannot bind owned Metal category storage.";
      auto command = [rt.queue commandBuffer];
      auto blit = [command blitCommandEncoder];
      for (auto const& copy : copies) {
        [blit copyFromBuffer:source.category_output sourceOffset:copy.source * bytes_per_node
                   toBuffer:out.buffer destinationOffset:out.offset + copy.node * bytes_per_node
                       size:bytes_per_node];
      }
      [blit endEncoding];
      CompleteOperation(command, &split_category_store_dispatches, &split_gpu_time_ns);
    }
    ready_.resize(size, false);
    for (auto node : ids) ready_[node] = true;
  }
};
}  // namespace
std::unique_ptr<SplitCategoryStorage> MakeSplitCategoryStorage(std::uint32_t words_per_node) {
  return std::make_unique<MetalSplitCategoryStorage>(words_per_node);
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
  auto lw = SnapshotModelWords(left_weight.data(), left_weight.size());
  auto rw = SnapshotModelWords(right_weight.data(), right_weight.size());
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
Span<SplitBounds const> GatherSplitBounds(Span<std::uint32_t const> nodes,
                                         std::uint32_t targets, Span<float const> lower,
                                         Span<float const> upper, SplitWorkspace* workspace) {
  static_assert(sizeof(SplitBounds) == 8);
  CHECK(workspace);
  CHECK_GT(targets, 0);
  CHECK_EQ(lower.size(), upper.size());
  CHECK_EQ(lower.size() % targets, 0);
  CHECK_LE(nodes.size(), std::numeric_limits<std::uint32_t>::max() / targets);
  if (!lower.empty()) {
    for (auto node : nodes) CHECK_LT(node, lower.size() / targets);
  }
  auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
  auto count = nodes.size() * targets;
  if (!count) {
    scratch->gathered_bounds = nil;
    return {};
  }
  id<MTLBuffer> output;
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto ids = Upload(rt, nodes.data(), nodes.size_bytes());
    auto lo = BindSplitBounds(rt, lower.data(), lower.size_bytes());
    auto hi = BindSplitBounds(rt, upper.data(), upper.size_bytes());
    // Publish a separate snapshot only after completion. Sources may alias the
    // previous gather, and later bound updates/growth must not change the result.
    output = Allocate(rt, count * sizeof(SplitBounds));
    std::uint32_t p[]{targets, static_cast<std::uint32_t>(!lower.empty())};
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto e = [command computeCommandEncoder];
    CHECK(e);
    [e setComputePipelineState:rt.split_bounds_gather_pipeline];
    [e setBuffer:ids offset:0 atIndex:0];
    [e setBuffer:lo.buffer offset:lo.offset atIndex:1];
    [e setBuffer:hi.buffer offset:hi.offset atIndex:2];
    [e setBuffer:output offset:0 atIndex:3];
    [e setBytes:p length:sizeof(p) atIndex:4];
    [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, &split_bounds_gather_dispatches, &split_gpu_time_ns);
    split_bounds_gather_values.fetch_add(count, std::memory_order_relaxed);
  }
  // Release borrowed aliases/command references before replacing their owner.
  scratch->gathered_bounds = output;
  return {static_cast<SplitBounds const*>(output.contents), count};
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

static void EvaluateQuantizedMultiSplitsImpl(
    Span<HistogramInput const> histograms, Span<QuantizedMultiSplitTask const> tasks,
    Span<GradientPairInt64 const> parents, Span<SplitBounds const> bounds,
    SplitParameters const& parameters, std::vector<QuantizedMultiSplitResult>* results,
    std::vector<GradientPairInt64>* integer_children, std::vector<GradientPairPrecise>* children,
    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace,
    std::optional<std::size_t> resident_bins, id<MTLCommandBuffer> pending = nil) {
  CHECK(!pending || resident_bins.has_value());
  if (resident_bins) CHECK(!results && !sorted_categories && !integer_children && !children);
  if (workspace) static_cast<MetalSplitWorkspace*>(workspace)->ResetCandidates();
  static_assert(sizeof(QuantizedMultiSplitTask) == 40 && sizeof(QuantizedMultiSplitResult) == 16);
  if (results) results->resize(tasks.size());
  auto bins = resident_bins ? *resident_bins : (tasks.empty() ? 0 : tasks.back().end);
  bool categorical = resident_bins.has_value() ||
      std::any_of(tasks.begin(), tasks.end(), [](auto const& task) { return task.kind == 2; });
  if (sorted_categories) sorted_categories->resize(categorical ? bins : 0);
  CHECK_LE(parents.size(), std::numeric_limits<std::uint32_t>::max() / 2);
  if (children) children->resize(parents.size() * 2);
  if (integer_children) integer_children->resize(parents.size() * 2);
  if (tasks.empty()) return;
  CHECK(workspace);
  CHECK_EQ(parents.size(), bounds.size());
  CHECK_LE(tasks.size(), std::numeric_limits<std::uint32_t>::max());
  std::uint32_t previous = 0;
  if (!resident_bins) {
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
  }
  @autoreleasepool {
    auto& rt = GetRuntime();
    auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
    auto command = pending ? pending : [rt.queue commandBuffer];
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
    auto hist = resident_bins ? scratch->hist
                              : GatherHistograms(rt, histograms, bins, command, scratch, true);
    auto input = Scratch(rt, &scratch->input, tasks.size_bytes(), resident_bins ? nullptr : tasks.data());
    // Resident parents are produced earlier in this command. Bind the owned
    // buffer directly, without a host staging copy before that producer runs.
    BorrowedMemory parent;
    if (resident_bins) {
      parent = {scratch->parent, 0};
      split_sum_shared_input_bytes.fetch_add(parents.size_bytes(), std::memory_order_relaxed);
    } else {
      parent = BindSplitSums(rt, &scratch->parent, parents.data(), parents.size_bytes());
    }
    auto bound = Scratch(rt, &scratch->bounds, bounds.size_bytes(), resident_bins ? nullptr : bounds.data());
    auto output = Scratch(rt, &scratch->output, tasks.size() * sizeof(QuantizedMultiSplitResult));
    auto child = Scratch(rt, &scratch->children, parents.size() * 2 * sizeof(GradientPairPrecise));
    auto exact = Scratch(rt, &scratch->integer_children,
                         parents.size() * 2 * sizeof(GradientPairInt64));
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
    [encoder setBuffer:parent.buffer offset:parent.offset atIndex:7];
    [encoder setBuffer:bound offset:0 atIndex:8];
    [encoder setBuffer:child offset:0 atIndex:9];
    [encoder setBuffer:scales offset:0 atIndex:10];
    [encoder setBuffer:exact offset:0 atIndex:11];
    CHECK_EQ(rt.quantized_multi_split_pipeline.threadExecutionWidth, 32);
    CHECK_GE(rt.quantized_multi_split_pipeline.maxTotalThreadsPerThreadgroup, 32);
    [encoder dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    if (!pending) {
      CompleteOperation(command, nullptr, &split_gpu_time_ns);
      RecordQuantizedSplitDispatch(true);
    }
    if (results) std::memcpy(results->data(), output.contents, tasks.size() * sizeof(QuantizedMultiSplitResult));
    if (results) for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto const& task = tasks[i];
      if (results->at(i).bin >= 0) {
        if (children) {
          auto bytes = 2 * task.targets * sizeof(GradientPairPrecise);
          std::memcpy(children->data() + 2 * task.parent_begin,
                      static_cast<GradientPairPrecise const*>(child.contents) + 2 * task.parent_begin,
                      bytes);
          split_sum_readback_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }
        if (integer_children) {
          auto bytes = 2 * task.targets * sizeof(GradientPairInt64);
          std::memcpy(integer_children->data() + 2 * task.parent_begin,
                      static_cast<GradientPairInt64 const*>(exact.contents) + 2 * task.parent_begin,
                      bytes);
          split_sum_readback_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }
      }
      if (sorted_categories && task.kind == 2)
        std::memcpy(sorted_categories->data() + task.begin,
                    static_cast<std::uint32_t const*>(order.contents) + task.begin,
                    ((task.end - task.begin) / task.targets) * sizeof(std::uint32_t));
    }
    scratch->candidate_count = tasks.size();
    scratch->vector_candidates = true;
    scratch->candidate_child_count = parents.size() * 2;
    if (resident_bins) {
      auto targets = static_cast<std::uint32_t>(histograms.size());
      for (std::uint32_t i = 0; i < tasks.size(); ++i)
        scratch->quantized_multi_shapes.push_back({i * targets, targets});
      scratch->quantized_multi_valid.resize(tasks.size(), false);
    } else {
      for (auto const& task : tasks)
        scratch->quantized_multi_shapes.push_back({task.parent_begin, task.targets});
      for (auto const& result : *results) scratch->quantized_multi_valid.push_back(result.bin >= 0);
    }
  }
}

void EvaluateQuantizedMultiSplits(
    Span<HistogramInput const> histograms, Span<QuantizedMultiSplitTask const> tasks,
    Span<GradientPairInt64 const> parents, Span<SplitBounds const> bounds,
    SplitParameters const& parameters, std::vector<QuantizedMultiSplitResult>* results,
    std::vector<GradientPairInt64>* integer_children, std::vector<GradientPairPrecise>* children,
    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace) {
  CHECK(results);
  EvaluateQuantizedMultiSplitsImpl(histograms, tasks, parents, bounds, parameters, results,
                                   integer_children, children, sorted_categories, workspace,
                                   std::nullopt);
}

void EvaluateFeatureSplits(
    Span<HistogramInput const> histograms, Span<Span<bst_feature_t const> const> features,
    Span<std::uint32_t const> cut_ptrs, Span<float const> cut_values, Span<FeatureType const> types,
    Span<std::int32_t const> monotone, std::uint32_t onehot, std::uint32_t category_limit,
    Span<GradientPairInt64 const> parents, Span<GradientPairPrecise const> precise_parents,
    Span<SplitBounds const> bounds, SplitParameters const& parameters, bool vector_leaf,
    std::vector<SplitNodeResult>* results, std::vector<FeatureSplitResult>* selected,
    std::vector<SplitNodeWeights>* weights, SplitWorkspace* workspace) {
  static_assert(sizeof(FeatureSplitResult) == 112 && offsetof(FeatureSplitResult, split_value) == 12);
  CHECK(workspace && results && selected);
  auto scratch = static_cast<MetalSplitWorkspace*>(workspace);
  scratch->ResetCandidates();
  CHECK_GT(histograms.size(), 0);
  CHECK_LE(histograms.size(), std::numeric_limits<std::uint32_t>::max());
  auto targets = static_cast<std::uint32_t>(histograms.size());
  if (!vector_leaf) CHECK_EQ(targets, 1);
  CHECK_LE(features.size(), std::numeric_limits<std::uint32_t>::max() / targets);
  CHECK_EQ(parents.size(), features.size() * targets);
  CHECK_EQ(precise_parents.size(), parents.size());
  CHECK_EQ(bounds.size(), parents.size());
  CHECK_GE(cut_ptrs.size(), 1);
  CHECK_LE(cut_ptrs.size() - 1, std::numeric_limits<std::uint32_t>::max());
  auto columns = static_cast<std::uint32_t>(cut_ptrs.size() - 1);
  CHECK(types.empty() || types.size() == columns);
  CHECK(monotone.empty() || monotone.size() == columns);
  std::vector<SplitFinalizeNode> nodes;
  std::size_t count = 0;
  for (auto ids : features) {
    CHECK_LE(ids.size(), std::numeric_limits<std::uint32_t>::max() - count);
    nodes.push_back({static_cast<std::uint32_t>(count),
                     static_cast<std::uint32_t>(count + ids.size()),
                     static_cast<std::uint32_t>(nodes.size() * targets), targets});
    count += ids.size();
  }
  CHECK_LE(count, std::numeric_limits<std::uint32_t>::max() / targets / 2);
  std::vector<std::uint64_t> offsets;
  std::uint32_t histogram_bins = 0;
  for (auto const& input : histograms) {
    CHECK_EQ(input.segments.size(), nodes.size());
    CHECK(input.source);
    auto storage = static_cast<MetalHistogramBuffer const*>(input.source);
    CHECK(storage->integers && storage->scale);
    for (auto const& segment : input.segments) {
      CHECK_LE(segment.source, input.source->Data().size());
      CHECK_LE(segment.bins, input.source->Data().size() - segment.source);
      if (offsets.empty()) histogram_bins = segment.bins;
      CHECK_EQ(segment.bins, histogram_bins);
      offsets.push_back(segment.source);
    }
  }
  selected->resize(nodes.size());
  if (nodes.empty()) {
    results->clear();
    if (weights) weights->clear();
    return;
  }
  CHECK_EQ(cut_values.size(), histogram_bins);
  @autoreleasepool {
    auto& rt = GetRuntime();
    struct FeatureParams {
      std::uint32_t begin, count, targets, node, features, onehot, limit, flags;
      std::uint32_t bins, nodes, candidates, padding;
    };
    static_assert(sizeof(FeatureParams) == 48);
    auto params = [&](std::size_t node) {
      return FeatureParams{nodes[node].begin, nodes[node].end - nodes[node].begin,
                            targets, static_cast<std::uint32_t>(node), columns, onehot,
                            category_limit, static_cast<std::uint32_t>(!types.empty()) |
                                (static_cast<std::uint32_t>(!monotone.empty()) << 1) |
                                (static_cast<std::uint32_t>(vector_leaf) << 2),
                            histogram_bins, static_cast<std::uint32_t>(nodes.size()),
                            static_cast<std::uint32_t>(count), 0};
    };
    auto cuts = BindFeatureInput(rt, cut_ptrs.data(), cut_ptrs.size_bytes());
    auto values = BindFeatureInput(rt, cut_values.data(), cut_values.size_bytes());
    auto type = BindFeatureInput(rt, types.data(), types.size_bytes());
    auto constraints = BindFeatureInput(rt, monotone.data(), monotone.size_bytes());
    auto ids = Scratch(rt, &scratch->feature_ids, count * sizeof(bst_feature_t));
    auto lengths = Scratch(rt, &scratch->feature_lengths, count * sizeof(std::uint64_t));
    std::uint32_t zero = 0;
    auto error = Scratch(rt, &scratch->feature_error, sizeof(zero), &zero);
    auto command = [rt.queue commandBuffer];
    CHECK(command);
    auto blit = [command blitCommandEncoder];
    CHECK(blit);
    for (std::size_t node = 0; node < nodes.size(); ++node) {
      if (features[node].empty()) continue;
      auto input = BindFeatureInput(rt, features[node].data(), features[node].size_bytes());
      [blit copyFromBuffer:input.buffer sourceOffset:input.offset toBuffer:ids
        destinationOffset:std::size_t(nodes[node].begin) * sizeof(bst_feature_t)
                     size:features[node].size_bytes()];
    }
    [blit endEncoding];
    auto lengths_encoder = [command computeCommandEncoder];
    CHECK(lengths_encoder);
    [lengths_encoder setComputePipelineState:rt.feature_lengths_pipeline];
    [lengths_encoder setBuffer:ids offset:0 atIndex:0];
    [lengths_encoder setBuffer:cuts.buffer offset:cuts.offset atIndex:1];
    [lengths_encoder setBuffer:type.buffer offset:type.offset atIndex:2];
    [lengths_encoder setBuffer:lengths offset:0 atIndex:3];
    [lengths_encoder setBuffer:error offset:0 atIndex:4];
    for (std::size_t node = 0; node < nodes.size(); ++node) {
      auto p = params(node);
      if (!p.count) continue;
      [lengths_encoder setBytes:&p length:sizeof(p) atIndex:5];
      [lengths_encoder dispatchThreads:MTLSizeMake(p.count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [lengths_encoder endEncoding];
    auto prefix = count ? EncodePrefixU64(rt, command, lengths, count) : lengths;
    CompleteOperation(command, nullptr, &split_gpu_time_ns);
    CHECK_EQ(*static_cast<std::uint32_t const*>(error.contents), 0)
        << "Invalid resident split feature, cut range or feature type";
    // Only the scalar allocation length is read back, never feature IDs or tasks.
    auto bins = count ? static_cast<std::uint64_t const*>(prefix.contents)[count - 1] : 0;
    CHECK_LE(bins, std::numeric_limits<std::uint32_t>::max());
    auto task_bytes = vector_leaf ? sizeof(QuantizedMultiSplitTask) : sizeof(QuantizedSplitTask);
    auto tasks = Scratch(rt, &scratch->input, count * task_bytes);
    auto final_features = Scratch(rt, &scratch->feature_input, count * sizeof(SplitFinalizeFeature));
    auto segments = Scratch(rt, &scratch->feature_segments, count * targets * sizeof(HistogramGather));
    auto hist = Scratch(rt, &scratch->hist, bins * sizeof(GradientPairInt64));
    auto source_offsets = Scratch(rt, &scratch->feature_offsets, offsets.size() * sizeof(std::uint64_t),
                                   offsets.data());
    // Gathered node bounds and exact sums remain resident while the GPU repeats
    // them by selected feature. The finalizer consumes the same bound snapshot.
    auto node_bounds = BindSplitBounds(rt, bounds.data(), bounds.size_bytes());
    auto node_parents = BindSplitSums(rt, &scratch->node_parents, parents.data(), parents.size_bytes());
    auto feature_parents = Scratch(rt, &scratch->parent, vector_leaf ? count * targets * sizeof(GradientPairInt64) : 0);
    auto feature_bounds = Scratch(rt, &scratch->bounds, vector_leaf ? count * targets * sizeof(SplitBounds) : 0);
    command = [rt.queue commandBuffer];
    CHECK(command);
    // Everything after the allocation-length readback remains on the GPU until
    // winner extraction. Encoders retain their buffers through completion; the
    // enclosing synchronous call retains borrowed input allocations as before.
    auto tasks_encoder = [command computeCommandEncoder];
    CHECK(tasks_encoder);
    [tasks_encoder setComputePipelineState:rt.feature_tasks_pipeline];
    [tasks_encoder setBuffer:ids offset:0 atIndex:0];
    [tasks_encoder setBuffer:cuts.buffer offset:cuts.offset atIndex:1];
    [tasks_encoder setBuffer:type.buffer offset:type.offset atIndex:2];
    [tasks_encoder setBuffer:constraints.buffer offset:constraints.offset atIndex:3];
    [tasks_encoder setBuffer:prefix offset:0 atIndex:4];
    [tasks_encoder setBuffer:source_offsets offset:0 atIndex:5];
    [tasks_encoder setBuffer:node_parents.buffer offset:node_parents.offset atIndex:6];
    [tasks_encoder setBuffer:node_bounds.buffer offset:node_bounds.offset atIndex:7];
    [tasks_encoder setBuffer:tasks offset:0 atIndex:8];
    [tasks_encoder setBuffer:final_features offset:0 atIndex:9];
    [tasks_encoder setBuffer:segments offset:0 atIndex:10];
    [tasks_encoder setBuffer:feature_parents offset:0 atIndex:11];
    [tasks_encoder setBuffer:feature_bounds offset:0 atIndex:12];
    for (std::size_t node = 0; node < nodes.size(); ++node) {
      auto p = params(node);
      if (!p.count) continue;
      [tasks_encoder setBytes:&p length:sizeof(p) atIndex:13];
      [tasks_encoder dispatchThreads:MTLSizeMake(p.count, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [tasks_encoder endEncoding];
    for (std::size_t t = 0; count && t < targets; ++t) {
      auto storage = static_cast<MetalHistogramBuffer const*>(histograms[t].source);
      auto e = [command computeCommandEncoder];
      CHECK(e);
      [e setComputePipelineState:rt.gather_hist_pipeline];
      [e setBuffer:storage->integers offset:0 atIndex:0];
      [e setBuffer:hist offset:0 atIndex:1];
      [e setBuffer:segments offset:t * count * sizeof(HistogramGather) atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    }
    if (vector_leaf) {
      EvaluateQuantizedMultiSplitsImpl(
          histograms, {static_cast<QuantizedMultiSplitTask const*>(tasks.contents), count},
          {static_cast<GradientPairInt64 const*>(feature_parents.contents), count * targets},
          {static_cast<SplitBounds const*>(feature_bounds.contents), count * targets}, parameters,
          nullptr, nullptr, nullptr, nullptr, workspace, bins, command);
    } else {
      EvaluateQuantizedSplitsImpl(histograms,
          {static_cast<QuantizedSplitTask const*>(tasks.contents), count}, parameters, nullptr,
          nullptr, workspace, bins, command);
    }
    FinalizeSplitNodesImpl(nodes,
        {static_cast<SplitFinalizeFeature const*>(final_features.contents), count}, precise_parents,
        bounds, parameters, vector_leaf, results, weights, workspace, true, command);
    auto out = Scratch(rt, &scratch->feature_selected, selected->size() * sizeof(FeatureSplitResult));
    auto e = [command computeCommandEncoder];
    CHECK(e);
    std::uint32_t multi = vector_leaf;
    [e setComputePipelineState:rt.feature_winners_pipeline];
    [e setBuffer:scratch->node_results offset:0 atIndex:0];
    [e setBuffer:final_features offset:0 atIndex:1];
    [e setBuffer:cuts.buffer offset:cuts.offset atIndex:2];
    [e setBuffer:tasks offset:0 atIndex:3];
    [e setBuffer:scratch->output offset:0 atIndex:4];
    [e setBuffer:out offset:0 atIndex:5];
    [e setBytes:&multi length:sizeof(multi) atIndex:6];
    [e setBuffer:values.buffer offset:values.offset atIndex:7];
    [e dispatchThreads:MTLSizeMake(nodes.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    CompleteOperation(command, nullptr, &split_gpu_time_ns);
    if (count) {
      feature_task_dispatches.fetch_add(1, std::memory_order_relaxed);
      RecordQuantizedSplitDispatch(vector_leaf);
    }
    feature_task_candidates.fetch_add(count, std::memory_order_relaxed);
    split_finalize_dispatches.fetch_add(1, std::memory_order_relaxed);
    ReadFinalizedSplitNodes(*scratch, results, weights);
    std::memcpy(selected->data(), out.contents, selected->size() * sizeof(FeatureSplitResult));
    if (vector_leaf) for (auto const& node : *results)
      if (node.winner != std::numeric_limits<std::uint32_t>::max())
        scratch->quantized_multi_valid.at(node.winner) = true;
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
                CudaSketchParams const* params = nullptr,
                std::initializer_list<std::size_t> offsets = {}) {
  if (!count) return;
  CHECK_LE(count, std::numeric_limits<std::uint32_t>::max());
  auto e = [command computeCommandEncoder];
  CHECK(e);
  [e setComputePipelineState:pipeline];
  std::size_t index = 0;
  CHECK(offsets.size() == 0 || offsets.size() == buffers.size());
  for (auto buffer : buffers) {
    [e setBuffer:buffer offset:offsets.size() ? offsets.begin()[index] : 0 atIndex:index];
    ++index;
  }
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
        weight_count_{weights.size()} {
    CHECK_GE(max_bin, 2);
    CHECK_LT(columns, std::numeric_limits<std::uint32_t>::max());
    CHECK(types.empty() || types.size() == columns);
    static_assert(sizeof(FeatureType) == 1 && static_cast<int>(FeatureType::kNumerical) == 0 &&
                  static_cast<int>(FeatureType::kCategorical) == 1);
    types_ = Allocate(rt_, columns);
    this->SetWeights(weights);
    auto ptr_bytes = (std::size_t{columns} + 1) * sizeof(std::uint64_t);
    resident_ = {Allocate(rt_, 0), Allocate(rt_, ptr_bytes), 0};
    auto categorical = Allocate(rt_, sizeof(std::uint32_t));
    auto source = BindFeatureInput(rt_, types.data(), types.size_bytes(), 1);
    auto command = [rt_.queue commandBuffer];
    CHECK(command);
    auto blit = [command blitCommandEncoder];
    CHECK(blit);
    // CUDA initializes device pointer arrays and keeps an owned feature-type
    // snapshot. Its categorical any_of returns only a host scheduling flag.
    [blit fillBuffer:resident_.ptr range:NSMakeRange(0, ptr_bytes) value:0];
    [blit fillBuffer:categorical range:NSMakeRange(0, sizeof(std::uint32_t)) value:0];
    if (!types.empty()) {
      [blit copyFromBuffer:source.buffer sourceOffset:source.offset toBuffer:types_
        destinationOffset:0 size:types.size_bytes()];
    } else if (columns) {
      [blit fillBuffer:types_ range:NSMakeRange(0, columns) value:0];
    }
    [blit endEncoding];
    CSDispatch(command, rt_.cs_has_categories_pipeline, types.size(), {types_, categorical});
    CSComplete(command);
    has_cat_ = *static_cast<std::uint32_t const*>(categorical.contents) != 0;
  }
  void SetWeights(Span<float const> weights) override {
    weight_count_ = weights.size();
    @autoreleasepool {
      auto source = BindGradientStateInput(rt_, weights.data(), weights.size_bytes(), 4);
      auto snapshot = Allocate(rt_, weights.size_bytes());
      if (!weights.empty()) {
        auto command = [rt_.queue commandBuffer];
        auto blit = [command blitCommandEncoder];
        CHECK(blit);
        [blit copyFromBuffer:source.buffer sourceOffset:source.offset toBuffer:snapshot
          destinationOffset:0 size:weights.size_bytes()];
        [blit endEncoding];
        CompleteOperation(command, nullptr, &cuda_sketch_gpu_time_ns);
      }
      weights_ = snapshot;
    }
  }
  void Push(AdapterInput const& input) override {
    auto const* gpu = dynamic_cast<MetalAdapterInput const*>(&input);
    CHECK(gpu);
    PushBuffers(gpu->entries, gpu->row_ptr, gpu->nnz, 0, gpu->base_row, gpu->rows,
                gpu->sketch_rows);
  }
  void Push(Span<::xgboost::Entry const> entries, Span<bst_idx_t const> row_ptr, std::size_t begin,
            bst_idx_t base_rowid, bst_idx_t rows_in_window) override {
    if (entries.empty() && !rows_in_window) return;
    CHECK_GE(row_ptr.size(), 2);
    CHECK_LE(begin, row_ptr.back());
    CHECK_LE(entries.size(), row_ptr.back() - begin);
    CHECK_LE(row_ptr.front(), begin);
    @autoreleasepool {
      auto source = BindCSRInput(rt_, entries.data(), entries.size_bytes());
      auto rows = BindCSRInput(rt_, row_ptr.data(), row_ptr.size_bytes());
      PushBuffers(source.buffer, rows.buffer, entries.size(), begin,
                  base_rowid, row_ptr.size() - 1, rows_in_window, source.offset, rows.offset);
    }
  }
  void PushBuffers(id<MTLBuffer> source, id<MTLBuffer> rows, std::size_t count, std::size_t begin,
                   bst_idx_t base_rowid, bst_idx_t row_count, bst_idx_t rows_in_window,
                   std::size_t source_offset = 0, std::size_t row_offset = 0) {
    CHECK_GT(columns_, 0);
    CHECK_LE(count, 1ULL << 26);
    if (weight_count_) CHECK_LE(base_rowid + row_count, weight_count_);
    CHECK_LE(rows_in_window, std::numeric_limits<bst_idx_t>::max() - rows_seen_);
    rows_seen_ += rows_in_window;
    // CUDA updates rows_seen before the empty-summary return.
    if (!count) return;
    @autoreleasepool {
      auto p = Params(count);
      p.begin = begin;
      p.base_row = base_rowid;
      p.budget = row_count;
      auto records = Allocate(rt_, count * 2 * sizeof(std::uint32_t));
      auto command = [rt_.queue commandBuffer];
      CHECK(command);
      CSDispatch(command, rt_.cs_sort_keys_pipeline, count, {source, source, records}, &p,
                   {source_offset, source_offset, 0});
      SortShape shape{count, 0, 0, 1, static_cast<std::uint32_t>((count + 255) / 256)};
      // Value and feature sorts execute serially in one command. Reuse their
      // ping-pong records and rank/count scratch within this input window.
      RadixWorkspace sort;
      records = EncodeRadixRecords(rt_, command, records, shape, 32, &sort);
      p.mode = 1;
      // Each key depends only on its own previous record, allowing in-place
      // key replacement without changing the stable value/tie order.
      CSDispatch(command, rt_.cs_sort_keys_pipeline, count, {source, records, records}, &p,
                   {source_offset, 0, 0});
      // Feature IDs lie in [0, columns_). Higher zero-key radix passes are
      // stable identity operations, so omit them without changing value/tie order.
      std::uint32_t feature_bits = 0;
      for (auto key = columns_ - 1; key; key >>= 1) ++feature_bits;
      if (feature_bits)
        records = EncodeRadixRecords(rt_, command, records, shape, feature_bits, &sort);
      CSDeviceSummary raw{Allocate(rt_, count * sizeof(WQSummary<>::Entry)),
                          Allocate(rt_, (std::size_t{columns_} + 1) * sizeof(std::uint64_t)),
                          count};
      // Only weighted sketches consume the segmented-scan inputs. Keep valid
      // placeholder bindings for unweighted gathers without allocating or
      // writing two arrays as large as the input window.
      auto weights = Allocate(rt_, weight_count_ ? count * sizeof(float) : 0);
      auto ids = Allocate(rt_, weight_count_ ? count * sizeof(std::uint32_t) : 0);
      CSDispatch(command, rt_.cs_bounds_pipeline, std::size_t{columns_} + 1, {records, raw.ptr},
                 &p);
      CSDispatch(command, rt_.cs_gather_pipeline, count,
                 {source, records, rows, weights_, raw.entries, weights, ids}, &p,
                 {source_offset, 0, row_offset, 0, 0, 0, 0});
      if (weight_count_) {
        auto scanned = CSEncodePrefix(rt_, command, weights, ids, count);
        CSDispatch(command, rt_.cs_attach_pipeline, count, {raw.entries, scanned});
      }
      CSComplete(command);
      if (has_cat_) raw = Unique(raw, true);
      // Unweighted ranks depend only on the entry's own value and position.
      // Weighted ranks read the previous entry's prefix sum, so those must
      // continue to use a separate destination.
      auto ranked = weight_count_ ? Allocate(rt_, raw.size * sizeof(WQSummary<>::Entry))
                                  : raw.entries;
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
      cuda_sketch_input_entries.fetch_add(count, std::memory_order_relaxed);
    }
  }
  void AllReduce(Context const* ctx) override {
    auto const& group = *collective::GlobalCommGroup();
    // CUDA SketchContainer returns immediately even for an attached world=1.
    if (!group.IsDistributed() || group.World() == 1) return;
    CHECK(ctx);
    // quantile.cu::SketchContainer::AllReduce intentionally exchanges these
    // control scalars on the host. Summary arithmetic and payload copies stay GPU.
    std::uint64_t rows = rows_seen_;
    collective::SafeColl(collective::Allreduce(ctx, &rows, collective::Op::kSum));
    auto budget = SketchSummaryBudget(max_bin_, rows);
    @autoreleasepool {
      resident_ = Prune(resident_, budget, false);
      std::uint64_t columns = std::uint64_t{columns_} + 1;
      collective::SafeColl(collective::Allreduce(ctx, &columns, collective::Op::kMax));
      CHECK_EQ(columns, std::size_t{columns_} + 1) << "Number of columns differs across workers";
      auto columns_bytes = columns * sizeof(std::uint64_t);
      using Entry = WQSummary<>::Entry;
      auto entries_offset = ((columns_bytes + alignof(Entry) - 1) / alignof(Entry)) * alignof(Entry);
      auto payload_bytes = [&](std::size_t entries) {
        CHECK_LE(entries, (std::numeric_limits<std::size_t>::max() - entries_offset) / sizeof(Entry));
        return entries_offset + entries * sizeof(Entry);
      };
      auto span = [](void* ptr, std::size_t size) {
        return Span<std::int8_t>{static_cast<std::int8_t*>(ptr), size};
      };
      auto pack = [&](VectorBuffer* out) {
        out->Resize(payload_bytes(resident_.size));
        auto bytes = span(out->Data(), out->SizeBytes());
        CollectiveCopy(span(resident_.ptr.contents, columns_bytes), bytes.first(columns_bytes));
        CollectiveCopy(span(resident_.entries.contents, resident_.size * sizeof(Entry)),
                       bytes.subspan(entries_offset));
      };
      auto parse = [&](VectorBuffer const& in) {
        CHECK_GE(in.SizeBytes(), entries_offset);
        auto entries_bytes = in.SizeBytes() - entries_offset;
        CHECK_EQ(entries_bytes % sizeof(Entry), 0) << "Invalid sketch payload size.";
        // Metal kernels bind float4 entries, requiring 16-byte alignment. CUDA's
        // serialized offset is only alignof(Entry), so load into aligned owned
        // buffers with GPU copies rather than casting an unaligned payload.
        CSDeviceSummary result{Allocate(rt_, entries_bytes), Allocate(rt_, columns_bytes),
                               entries_bytes / sizeof(Entry)};
        auto bytes = span(in.Data(), in.SizeBytes());
        CollectiveCopy(bytes.first(columns_bytes), span(result.ptr.contents, columns_bytes));
        CollectiveCopy(bytes.subspan(entries_offset), span(result.entries.contents, entries_bytes));
        CHECK_EQ(Total(result.ptr), result.size);
        return result;
      };
      auto payload = MakeVectorBuffer();
      collective::metal_impl::AllreduceVScratch scratch;
      CHECK(!columns_ || budget <= std::numeric_limits<std::size_t>::max() / columns_);
      auto reserve = payload_bytes(budget * columns_);
      payload->Resize(reserve);
      payload->Resize(0);
      scratch.Reserve(reserve);
      pack(payload.get());
      collective::SafeColl(collective::metal_impl::AllreduceV(
          ctx, &payload, &scratch,
          [&](VectorBuffer const&, VectorBuffer const& rhs, VectorBuffer* next) {
            Merge(parse(rhs));
            resident_ = Prune(resident_, budget, false);
            pack(next);
          }));
      resident_ = parse(*payload);
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
