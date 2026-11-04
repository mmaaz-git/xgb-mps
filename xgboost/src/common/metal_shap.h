/** Copyright 2026, XGBoost Contributors */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "metal_predictor.h"
#include "xgboost/span.h"

namespace xgboost::common::metal {
struct ShapNode {
  PredictionNode node;
  float left_weight{0}, right_weight{0};
};
struct ShapTree {
  std::uint32_t root, group, target;
  float weight;
};
struct ShapModel {
  std::vector<ShapNode> nodes;
  std::vector<ShapTree> trees;
  std::vector<std::uint32_t> categories;
  std::vector<float> leaves, root_means, base_scores;
  float quadrature_nodes[8], quadrature_weights[8];
  std::uint32_t depth{1};
};
// Traversal stacks have a runtime depth, not a fixed maximum. Input, output and
// per-row workspaces are bounded by MaxBatchRows; uploaded model storage is reused.
class Shap {
 public:
  virtual ~Shap() = default;
  virtual std::size_t MaxBatchRows() const = 0;
  virtual void Predict(Span<float const> data, std::size_t rows, Span<float const> base_margin,
                       Span<float> output) = 0;
};
std::unique_ptr<Shap> MakeShap(ShapModel const& model, std::uint32_t features, std::uint32_t groups,
                               bool interactions);
}  // namespace xgboost::common::metal
