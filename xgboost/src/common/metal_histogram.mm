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
#include "metal_histogram.h"
#include "metal_ops.h"
#include "metal_predictor.h"
#include "metal_shader.h"
#include "xgboost/logging.h"

namespace xgboost::common::metal {
namespace {
std::atomic<std::uint64_t> dispatches{0};
std::atomic<std::uint64_t> processed_rows{0};
std::atomic<std::uint64_t> shared_dispatches{0};
std::atomic<std::uint64_t> gpu_time_ns{0};
std::atomic<std::uint64_t> quantization_dispatches{0};
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
  std::uint64_t n_rows, base_rowid;
  std::uint32_t n_bins, n_features, bin_bytes, dense, n_nodes, padding;
};
static_assert(sizeof(Params) == 40);
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
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
  id<MTLComputePipelineState> shared_pipeline;
  id<MTLComputePipelineState> quantize_pipeline, objective_pipeline, transform_pipeline,
      prediction_pipeline, leaf_pipeline, metric_pipeline;

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
      objective_pipeline = make_pipeline(@"xgb_objective");
      transform_pipeline = make_pipeline(@"xgb_transform");
      prediction_pipeline = make_pipeline(@"xgb_predict");
      leaf_pipeline = make_pipeline(@"xgb_add_leaf");
      metric_pipeline = make_pipeline(@"xgb_metric");
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
  id<MTLBuffer> gradients_, raw_gradients_, histogram_buffer_, row_buffer_, node_buffer_;
  bool gradients_ready_{false};
  std::vector<Page> pages_;
  double inverse_scale_[2]{1.0, 1.0};

  void UploadGradients(Span<GradientPair const> gradients, id<MTLCommandBuffer> command) {
    // Use independent power-of-two scales. Bound every possible subset by
    // n_rows * max(abs(value)), with three signed bits of accumulation headroom.
    // CPU FP64 is intentional: Metal has no native FP64 arithmetic.
    CHECK_LE(gradients.size(), std::numeric_limits<std::uint32_t>::max());
    double maxima[2]{0.0, 0.0};
    for (auto const& g : gradients) {
      CHECK(std::isfinite(g.GetGrad()) && std::isfinite(g.GetHess()))
          << "Metal histograms require finite gradients and Hessians.";
      maxima[0] = std::max(maxima[0], std::abs(static_cast<double>(g.GetGrad())));
      maxima[1] = std::max(maxima[1], std::abs(static_cast<double>(g.GetHess())));
    }
    int exponents[2];
    for (int i = 0; i < 2; ++i) {
      double bound = maxima[i] * static_cast<double>(gradients.size());
      int exponent = bound > 0.0 ? 60 - static_cast<int>(std::ceil(std::log2(bound))) : 0;
      exponents[i] = exponent;
      inverse_scale_[i] = std::ldexp(1.0, -exponent);
    }
    if (!gradients_ || gradients_.length < gradients.size() * sizeof(QuantizedPair))
      gradients_ = Allocate(rt_, gradients.size() * sizeof(QuantizedPair));
    static_assert(sizeof(GradientPair) == 8);
    if (!raw_gradients_ || raw_gradients_.length < gradients.size() * sizeof(GradientPair))
      raw_gradients_ = Allocate(rt_, gradients.size() * sizeof(GradientPair));
    std::memcpy(raw_gradients_.contents, gradients.data(), gradients.size() * sizeof(GradientPair));
    auto raw = raw_gradients_;
    struct {
      std::uint64_t n;
      std::int32_t grad, hess;
    } p{gradients.size(), exponents[0], exponents[1]};
    auto encoder = [command computeCommandEncoder];
    CHECK(encoder);
    [encoder setComputePipelineState:rt_.quantize_pipeline];
    [encoder setBuffer:raw offset:0 atIndex:0];
    [encoder setBuffer:gradients_ offset:0 atIndex:1];
    [encoder setBytes:&p length:sizeof(p) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(gradients.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
  }

 public:
  MetalHistogram() : rt_{GetRuntime()} {
    queue_ = rt_.queue;
    CHECK(queue_) << "Metal command queue creation failed.";
  }

  void Reset() override {
    gradients_ready_ = false;
    pages_.clear();
  }

  void Build(std::size_t page_idx, GHistIndexMatrix const& gidx, RowSetCollection const& rows,
             std::vector<bst_node_t> const& nodes, Span<GradientPair const> gradients,
             std::vector<GHistRow> const& output) override {
    @autoreleasepool {
      CHECK_EQ(nodes.size(), output.size());
      CHECK_LE(nodes.size(), std::numeric_limits<std::uint32_t>::max());
      auto staging_start = Clock::now();
      if (pages_.size() <= page_idx) pages_.resize(page_idx + 1);
      auto& page = pages_[page_idx];
      if (!page.index) {
        page.index = Upload(rt_, gidx.index.data<std::uint8_t>(),
                            gidx.index.Size() * gidx.index.GetBinTypeSize());
        page.row_ptr = Upload(rt_, gidx.row_ptr.data(), gidx.row_ptr.size() * sizeof(std::size_t));
        page.offsets =
            Upload(rt_, gidx.cut.Ptrs().data(), gidx.cut.Ptrs().size() * sizeof(std::uint32_t));
      }
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
      auto row_bytes = total_rows * sizeof(bst_idx_t);
      if (!row_buffer_ || row_buffer_.length < row_bytes) row_buffer_ = Allocate(rt_, row_bytes);
      auto row_buffer = row_buffer_;
      for (std::size_t node = 0; node < nodes.size(); ++node) {
        auto size = rows[nodes[node]].Size();
        if (size)
          std::memcpy(static_cast<bst_idx_t*>(row_buffer.contents) + node_ptr[node],
                      rows[nodes[node]].begin(), size * sizeof(bst_idx_t));
      }
      auto node_bytes = node_ptr.size() * sizeof(std::uint64_t);
      if (!node_buffer_ || node_buffer_.length < node_bytes)
        node_buffer_ = Allocate(rt_, node_bytes);
      std::memcpy(node_buffer_.contents, node_ptr.data(), node_bytes);
      auto node_buffer = node_buffer_;
      Params params{total_rows,
                    gidx.base_rowid,
                    static_cast<std::uint32_t>(bins),
                    static_cast<std::uint32_t>(gidx.cut.Ptrs().size() - 1),
                    static_cast<std::uint32_t>(gidx.index.GetBinTypeSize()),
                    static_cast<std::uint32_t>(gidx.IsDense()),
                    static_cast<std::uint32_t>(nodes.size()),
                    0};
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      CHECK(command) << "Metal command buffer allocation failed.";
      bool quantized = !gradients_ready_;
      if (quantized) this->UploadGradients(gradients, command);
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
                        params.n_features <= 65535 && nodes.size() <= 65535;
      auto pipeline = use_shared ? rt_.shared_pipeline : rt_.pipeline;
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:page.index offset:0 atIndex:0];
      [encoder setBuffer:page.row_ptr offset:0 atIndex:1];
      [encoder setBuffer:page.offsets offset:0 atIndex:2];
      [encoder setBuffer:gradients_ offset:0 atIndex:3];
      [encoder setBuffer:row_buffer offset:0 atIndex:4];
      [encoder setBuffer:histogram offset:0 atIndex:5];
      [encoder setBytes:&params length:sizeof(params) atIndex:6];
      auto threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
      [encoder setBuffer:node_buffer offset:0 atIndex:7];
      if (use_shared) {
        [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
        auto groups = (max_node_rows + threads * 8 - 1) / (threads * 8);
        [encoder dispatchThreadgroups:MTLSizeMake(groups, params.n_features, nodes.size())
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      } else {
        [encoder dispatchThreads:MTLSizeMake(total_rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      }
      [encoder endEncoding];
      staging_time_ns.fetch_add(Elapsed(staging_start), std::memory_order_relaxed);
      auto wait_start = Clock::now();
      [command commit];
      [command waitUntilCompleted];
      CHECK(command.status == MTLCommandBufferStatusCompleted)
          << "Metal histogram execution failed: " << ErrorText(command.error);
      wait_time_ns.fetch_add(Elapsed(wait_start), std::memory_order_relaxed);
      auto readback_start = Clock::now();
      if (quantized) {
        quantization_dispatches.fetch_add(1, std::memory_order_relaxed);
        gradients_ready_ = true;
      }
      auto const* values = static_cast<QuantizedPair const*>(histogram.contents);
      for (std::size_t node = 0; node < nodes.size(); ++node) {
        for (std::size_t bin = 0; bin < bins; ++bin) {
          auto const& v = values[node * bins + bin];
          output[node][bin] +=
              GradientPairPrecise{v.grad * inverse_scale_[0], v.hess * inverse_scale_[1]};
        }
      }
      readback_time_ns.fetch_add(Elapsed(readback_start), std::memory_order_relaxed);
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
          metric_gpu_time_ns.load(std::memory_order_relaxed)};
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
    auto pred = Upload(rt, predictions.data(), predictions.size_bytes());
    auto label = Upload(rt, labels.data(), labels.size_bytes());
    auto weight = Upload(rt, weights.data(), weights.size_bytes());
    auto levels = Upload(rt, alpha.data(), alpha.size_bytes());
    auto out = Allocate(rt, output.size_bytes());
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
    [encoder setBuffer:pred offset:0 atIndex:0];
    [encoder setBuffer:label offset:0 atIndex:1];
    [encoder setBuffer:weight offset:0 atIndex:2];
    [encoder setBuffer:levels offset:0 atIndex:3];
    [encoder setBuffer:out offset:0 atIndex:4];
    [encoder setBytes:&p length:sizeof(p) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(predictions.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    CompleteOperation(command, &objective_dispatches);
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
}  // namespace xgboost::common::metal
