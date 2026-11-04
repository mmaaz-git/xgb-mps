/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <memory>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
// CUDA constraints.cu semantics: masks accumulate the union of groups containing
// each split feature. Nonroot queries emit ascending, unique feature IDs.
class InteractionConstraints {
 public:
  virtual ~InteractionConstraints() = default;
  virtual void Reset() = 0;
  virtual void Split(bst_node_t node, bst_feature_t feature, bst_node_t left, bst_node_t right) = 0;
  virtual std::vector<bst_feature_t> Query(Span<bst_feature_t const> features, bst_node_t node) = 0;
  virtual std::vector<bst_feature_t> QueryNode(bst_node_t node) = 0;
};
// Nodes are allocated lazily with zero masks; growth and reset execute on GPU.
// An empty group list with enabled=true differs from disabled constraints.
std::unique_ptr<InteractionConstraints> MakeInteractionConstraints(
    bst_feature_t features, std::vector<std::vector<bst_feature_t>> const& groups,
    bool enabled = true);
}  // namespace xgboost::common::metal
