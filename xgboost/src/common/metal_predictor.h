/** Copyright 2026, XGBoost Contributors */
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "xgboost/span.h"
namespace xgboost::common::metal {
struct PredictionNode {
  std::int32_t left{-1}, right{-1}, missing{-1};
  std::uint32_t feature{0};
  float split{0};
  std::uint32_t category_begin{0}, category_size{0}, leaf_begin{0}, categorical{0}, padding{0};
};
struct PredictionTree {
  std::uint32_t root, targets, group;
  float weight;
};
struct PredictionModel {
  std::vector<PredictionNode> nodes;
  std::vector<PredictionTree> trees;
  std::vector<std::uint32_t> categories;
  std::vector<float> leaves;
};
class Prediction {
 public:
  virtual ~Prediction() = default;
  virtual void Predict(Span<float const> data, std::size_t rows, std::uint32_t features,
                       std::uint32_t groups, Span<float> predictions, bool leaf = false) = 0;
};
void AddLeafValues(Span<std::int32_t const> ids, Span<float const> weights, std::uint32_t targets,
                   std::size_t stride, Span<float> output);
std::unique_ptr<Prediction> MakePrediction(PredictionModel const& model);
}  // namespace xgboost::common::metal
