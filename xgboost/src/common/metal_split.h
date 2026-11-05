/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstdint>
#include <vector>

#include "metal_storage.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/span.h"
namespace xgboost::common::metal {
class QuantizedGradients;
class SplitWorkspace {
 public:
  virtual ~SplitWorkspace() = default;
};
std::unique_ptr<SplitWorkspace> MakeSplitWorkspace();

// CUDA scalar split semantics: integer prefixes/missing/children, exclusive
// numeric scans, strict missing-direction comparison (ties go right).
struct QuantizedSplitTask {
  std::uint32_t begin, end, kind, limit;
  GradientPairInt64 parent;
  float lower, upper;
  std::int32_t monotone;
  std::uint32_t padding{0};
};
struct QuantizedSplitResult {
  GradientPairInt64 left, right;
  // Raw child gain: choose the winning feature BEFORE subtracting parent gain.
  float gain;
  // Numeric: exclusive prefix position, including position zero (-infinity).
  // Categorical: category index or sorted partition boundary.
  std::int32_t bin;
  std::uint32_t default_left, partition;
  GradientPairPrecise left_sum, right_sum;
};

struct SplitTask {
  std::uint32_t begin, end, kind, limit;
  double parent_grad, parent_hess;
  float root_gain, lower, upper;
  std::int32_t monotone;
};
struct SplitParameters {
  float reg_lambda, reg_alpha, max_delta_step, min_child_weight;
};
// Quantized evaluators accept nullptr for sorted_categories to retain category
// order only on the GPU (BuildSplitCategories consumes that resident order).
void EvaluateQuantizedSplits(Span<HistogramInput const> histograms,
                             Span<QuantizedSplitTask const> tasks,
                             SplitParameters const& parameters,
                             std::vector<QuantizedSplitResult>* results,
                             std::vector<std::uint32_t>* sorted_categories,
                             SplitWorkspace* workspace);
struct SplitResult {
  double left_grad, left_hess, right_grad, right_hess;
  float gain;
  std::int32_t bin;
  std::uint32_t default_left, partition;
};
void EvaluateSplits(Span<HistogramInput const> histograms, Span<SplitTask const> tasks,
                    SplitParameters const& parameters, std::vector<SplitResult>* results,
                    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace);

// Histograms are packed bin-major, target-minor. Parent sums and bounds are
// packed by task, followed by target. Only the winning child sums are returned.
struct MultiSplitTask {
  std::uint32_t begin, end, kind, limit, parent_begin, targets;
  double root_gain;
  std::int32_t monotone;
  std::uint32_t constrained;
};
struct SplitBounds {
  float lower, upper;
};
// Gather node-major bounds on GPU. Node IDs are host scheduling metadata; lower
// and upper are resident payloads (empty means unconstrained +/-infinity).
// The result is an owned snapshot in workspace, valid until its next gather or
// destruction. Evaluating/finalizing splits does not invalidate the snapshot.
Span<SplitBounds const> GatherSplitBounds(Span<std::uint32_t const> nodes,
                                         std::uint32_t targets, Span<float const> lower,
                                         Span<float const> upper, SplitWorkspace* workspace);
// GPU-owned monotonic bounds. Grow preserves existing values and initializes
// only the new tail to +/-FLT_MAX, as CUDA HostDeviceVector::Resize does.
// Shared read views are valid until the next growth or destruction; operations
// complete before returning, including updates visible through existing views.
class SplitBoundsStorage {
 public:
  virtual ~SplitBoundsStorage() = default;
  virtual void Grow(std::size_t count) = 0;
  virtual Span<float const> Lower() const = 0;
  virtual Span<float const> Upper() const = 0;
  virtual void Update(std::uint32_t parent, std::uint32_t left, std::uint32_t right,
                      std::int32_t monotone, Span<float const> left_weight,
                      Span<float const> right_weight, bool vector_leaf) = 0;
};
std::unique_ptr<SplitBoundsStorage> MakeSplitBoundsStorage(std::size_t count);
// Vector CUDA evaluation selects each 32-bin chunk in binary64, narrows the
// winner to FP32, then compares chunk winners. Root gain is subtracted only
// after the winning feature has been chosen. one_pass is a CUDA test hook:
// 0 = both numeric directions, 1 = forward only, 2 = backward only.
struct QuantizedMultiSplitTask {
  std::uint32_t begin, end, kind, limit, parent_begin, targets;
  std::int32_t monotone;
  std::uint32_t constrained, one_pass{0}, padding{0};
};
struct QuantizedMultiSplitResult {
  float gain;
  std::int32_t bin;
  std::uint32_t default_left, partition;
};
struct SplitSumCopy {
  std::uint32_t candidate, node;
};
struct SplitSumViews {
  Span<GradientPairInt64 const> integers;
  Span<GradientPairPrecise const> precise;
};
// Exact node and selected-child payloads survive queued candidates and workspace
// reuse. Only descriptors live on the host. Gather derives binary64 views on GPU
// using the original quantizers; integers are never reconstructed from floats.
class SplitSumStorage {
 public:
  virtual ~SplitSumStorage() = default;
  // One root per storage instance; input must be resident on Metal.
  virtual void InitRoot(Span<GradientPairInt64 const> root) = 0;
  virtual void Store(SplitWorkspace const& workspace, Span<SplitSumCopy const> copies) = 0;
  virtual void Apply(std::uint32_t parent, std::uint32_t left, std::uint32_t right) = 0;
  // Views remain valid until the next Gather or destruction. Duplicate/reordered
  // node IDs are allowed; all requested nodes must have been initialized.
  virtual SplitSumViews Gather(Span<std::uint32_t const> nodes) = 0;
};
std::unique_ptr<SplitSumStorage> MakeSplitSumStorage(
    std::shared_ptr<QuantizedGradients const> gradients);
// Null child destinations retain both exact and precise payloads on the GPU.
void EvaluateQuantizedMultiSplits(
    Span<HistogramInput const> histograms, Span<QuantizedMultiSplitTask const> tasks,
    Span<GradientPairInt64 const> parents, Span<SplitBounds const> bounds,
    SplitParameters const& parameters, std::vector<QuantizedMultiSplitResult>* results,
    std::vector<GradientPairInt64>* integer_children, std::vector<GradientPairPrecise>* children,
    std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace);

void EvaluateMultiSplits(Span<HistogramInput const> histograms, Span<MultiSplitTask const> tasks,
                         Span<GradientPairPrecise const> parents, Span<SplitBounds const> bounds,
                         SplitParameters const& parameters, std::vector<SplitResult>* results,
                         std::vector<GradientPairPrecise>* children,
                         std::vector<std::uint32_t>* sorted_categories, SplitWorkspace* workspace);

// Finalize the resident output of EvaluateQuantized[Multi]Splits. Candidate
// ranges and parent ranges are contiguous in node order. Empty ranges also
// support root-weight calculation before any split evaluation.
struct SplitFinalizeNode {
  std::uint32_t begin, end, parent_begin, targets;
};
struct SplitFinalizeFeature {
  std::uint32_t feature;
  std::int32_t monotone;
  std::uint32_t child_begin, padding{0};
};
struct SplitNodeResult {
  std::uint32_t winner;
  float loss_chg;
  double parent_gain, parent_hess, left_hess, right_hess;
};
struct SplitNodeWeights {
  float base, left, right, padding{0};
};
// Persistent CUDA-style Base/Left/Right weight views, indexed by node ID. Store
// gathers the latest finalized GPU results; views survive until the next Store
// that grows storage, or destruction. No weight payload is read on the CPU.
struct SplitWeightCopy {
  std::uint32_t source, node;
};
class SplitWeightStorage {
 public:
  virtual ~SplitWeightStorage() = default;
  virtual void Store(SplitWorkspace const& workspace, Span<SplitWeightCopy const> copies) = 0;
  virtual Span<float const> Base(std::uint32_t node) const = 0;
  virtual Span<float const> Left(std::uint32_t node) const = 0;
  virtual Span<float const> Right(std::uint32_t node) const = 0;
};
std::unique_ptr<SplitWeightStorage> MakeSplitWeightStorage(std::uint32_t targets);
// A null weights destination retains finalized weights only in the workspace.
void FinalizeSplitNodes(Span<SplitFinalizeNode const> nodes,
                        Span<SplitFinalizeFeature const> features,
                        Span<GradientPairPrecise const> parents, Span<SplitBounds const> bounds,
                        SplitParameters const& parameters, bool vector_leaf,
                        std::vector<SplitNodeResult>* results,
                        std::vector<SplitNodeWeights>* weights, SplitWorkspace* workspace);
struct FeatureSplitResult {
  std::uint32_t feature, kind, cut_begin;
  float split_value;
  QuantizedSplitResult scalar;
  QuantizedMultiSplitResult vector;
};
// Resident feature IDs drive GPU histogram gathering and task construction.
// Each histogram input has one full-node segment per feature set (source and
// bins are used; output/stride are assigned on GPU). Parents/bounds are in node
// order. Cut values remain resident through threshold/category lookup. Only
// selected per-node control/scalar results (including split_value) are published.
// Vector child sums, categories and weights remain in the workspace for their consumers.
void EvaluateFeatureSplits(
    Span<HistogramInput const> histograms, Span<Span<bst_feature_t const> const> features,
    Span<std::uint32_t const> cut_ptrs, Span<float const> cut_values, Span<FeatureType const> types,
    Span<std::int32_t const> monotone, std::uint32_t onehot, std::uint32_t category_limit,
    Span<GradientPairInt64 const> parents, Span<GradientPairPrecise const> precise_parents,
    Span<SplitBounds const> bounds, SplitParameters const& parameters, bool vector_leaf,
    std::vector<SplitNodeResult>* nodes, std::vector<FeatureSplitResult>* selected,
    std::vector<SplitNodeWeights>* weights, SplitWorkspace* workspace);
void UpdateSplitBounds(Span<float> lower, Span<float> upper, std::uint32_t parent,
                       std::uint32_t left, std::uint32_t right, std::int32_t monotone,
                       Span<float const> left_weight, Span<float const> right_weight,
                       bool vector_leaf);

struct SplitCategoryTask {
  std::uint32_t candidate, cut_begin;
};
struct SplitCategoryCopy {
  std::uint32_t source, node;
};
// Node-indexed selected bitsets, retained across candidate-workspace reuse.
// Reacquire views after a Store that grows storage.
class SplitCategoryStorage {
 public:
  virtual ~SplitCategoryStorage() = default;
  virtual void Store(SplitWorkspace const& workspace, Span<SplitCategoryCopy const> copies) = 0;
  virtual Span<std::uint32_t const> Get(std::uint32_t node) const = 0;
};
std::unique_ptr<SplitCategoryStorage> MakeSplitCategoryStorage(std::uint32_t words_per_node);
// Build one zero-initialized LBitField32 per requested candidate from the latest
// successful quantized evaluation. Reads resident results and sorted bin indices.
// Numeric and invalid candidates produce empty bitsets; tasks can be reordered.
// A null output retains the payload only in the workspace for device storage.
void BuildSplitCategories(Span<float const> cuts, Span<SplitCategoryTask const> tasks,
                          std::uint32_t words_per_node, std::vector<std::uint32_t>* output,
                          SplitWorkspace* workspace);
}  // namespace xgboost::common::metal
