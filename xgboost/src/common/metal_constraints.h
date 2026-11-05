/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <memory>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// CUDA constraints.cu semantics: masks accumulate the union of groups containing
// each split feature. Nonroot queries emit ascending, unique feature IDs.
class InteractionConstraints {
 public:
  virtual ~InteractionConstraints() = default;
  virtual void Reset() = 0;
  virtual void Split(bst_node_t node, bst_feature_t feature, bst_node_t left, bst_node_t right) = 0;
  // Independent GPU-owned snapshots survive later queries, growth and reset.
  // Only the result count returns to the CPU. Root/disabled queries preserve
  // input order and duplicates; QueryNodeDevice always queries the node mask.
  virtual HostDeviceVector<bst_feature_t> QueryDevice(Span<bst_feature_t const> features,
                                                     bst_node_t node) = 0;
  virtual HostDeviceVector<bst_feature_t> QueryNodeDevice(bst_node_t node) = 0;
  // Host diagnostic wrappers; production GPU callers use the resident methods.
  virtual std::vector<bst_feature_t> Query(Span<bst_feature_t const> features, bst_node_t node) = 0;
  virtual std::vector<bst_feature_t> QueryNode(bst_node_t node) = 0;
};
// Nodes are allocated lazily with zero masks; growth and reset execute on GPU.
// An empty group list with enabled=true differs from disabled constraints.
std::unique_ptr<InteractionConstraints> MakeInteractionConstraints(
    bst_feature_t features, std::vector<std::vector<bst_feature_t>> const& groups,
    bool enabled = true);
}  // namespace xgboost::common::metal
