/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <variant>

#include "../tree/tree_view.h"
#include "metal_predictor.h"
#include "optional_weight.h"

namespace xgboost::common::metal {
using PredictionTreeView = std::variant<tree::ScalarTreeView, tree::MultiTargetTreeView>;

// Views select the RegTree device mirrors under GBTreeModel::Mutex(), as in CUDA.
// This synchronous call snapshots their payloads on GPU; only per-tree sizes and
// weights are assembled on the host. The returned object owns its packed arrays.
std::unique_ptr<Prediction> MakePrediction(Span<PredictionTreeView const> trees,
                                           Span<bst_target_t const> groups,
                                           OptionalWeights weights);
}  // namespace xgboost::common::metal
