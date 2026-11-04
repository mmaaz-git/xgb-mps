// Independent one-split expectations, guards, aliasing and bounded batches.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_shap.h"
#include "../xgboost/src/predictor/interpretability/quadrature.h"

namespace metal = xgboost::common::metal;
std::size_t cases = 0, checked = 0;
float max_error = 0;
void Check(bool ok, char const* text) {
  if (!ok) throw std::runtime_error(text);
}
void Close(float actual, float expected) {
  auto error = std::abs(actual - expected);
  max_error = std::max(max_error, error);
  Check(std::isfinite(actual) && error <= 2e-6f * (1.f + std::abs(expected)),
        "One-feature Shapley value differs from independent expectation");
  ++checked;
}
metal::ShapModel Model(unsigned groups, float scale) {
  metal::ShapModel m;
  m.depth = 2;
  m.nodes.resize(3);
  auto& split = m.nodes[0];
  split.node.left = split.node.missing = 1;
  split.node.right = 2;
  split.node.feature = 0;
  split.node.split = 0;
  split.left_weight = .25f;
  split.right_weight = .75f;
  m.nodes[1].node.leaf_begin = 0;
  m.nodes[2].node.leaf_begin = groups;
  for (unsigned g = 0; g < groups; ++g) m.leaves.push_back(-1.5f * (g + 1));
  for (unsigned g = 0; g < groups; ++g) m.leaves.push_back(.5f * (g + 1));
  for (unsigned g = 0; g < groups; ++g) {
    m.trees.push_back({0, g, g, scale});
    m.root_means.push_back(0.f);
    m.base_scores.push_back(.25f * g);
  }
  auto rule = xgboost::interpretability::detail::GetQuadratureRule();
  std::copy_n(rule.nodes, 8, m.quadrature_nodes);
  std::copy_n(rule.weights, 8, m.quadrature_weights);
  return m;
}
void Run(unsigned features, unsigned groups, float scale, bool interactions,
         bool margin, int alias, std::size_t requested_rows) {
  auto model = Model(groups, scale);
  auto gpu = metal::MakeShap(model, features, groups, interactions);
  auto rows = std::min(requested_rows, gpu->MaxBatchRows());
  auto columns = std::size_t(features) + 1;
  auto per_group = interactions ? columns * columns : columns;
  auto count = rows * groups * per_group;
  std::vector<float> output(count + 2, -971.f);
  std::vector<float> data(rows * features, 2.f);
  std::vector<float> margins(margin ? rows * groups : 0, -.125f);
  for (std::size_t r = 0; r < rows; ++r) {
    data[r * features] = r % 3 == 0   ? -1.f
                         : r % 3 == 1 ? 1.f
                                      : std::numeric_limits<float>::quiet_NaN();
  }
  auto original = data;
  float* input = data.data();
  if (alias == 1) {
    Check(data.size() <= count, "Alias storage too small");
    std::copy(data.begin(), data.end(), output.begin() + 1);
    input = output.data() + 1;
  }
  xgboost::common::Span<float const> base_margin{margins};
  if (alias == 2) {
    std::copy(margins.begin(), margins.end(), output.begin() + 1);
    base_margin = {output.data() + 1, margins.size()};
  }
  auto before = metal::GetStatistics();
  gpu->Predict({input, data.size()}, rows, base_margin,
               {output.data() + 1, count});
  auto after = metal::GetStatistics();
  auto dispatches = interactions
                        ? after.shap_interaction_dispatches - before.shap_interaction_dispatches
                        : after.shap_dispatches - before.shap_dispatches;
  Check(dispatches == bool(rows), "Wrong SHAP dispatch count");
  Check(!rows || after.shap_gpu_time_ns > before.shap_gpu_time_ns, "Missing GPU time");
  Check(after.prediction_device_copy_bytes -
                before.prediction_device_copy_bytes ==
            (alias == 2 ? margins.size() * 4 : 0),
        "Overlapping margins require a GPU snapshot");
  Check(output.front() == -971.f && output.back() == -971.f, "Output guard overwritten");
  Check(data.empty() || !std::memcmp(data.data(), original.data(), data.size() * 4),
        "Input modified");
  for (std::size_t r = 0; r < rows; ++r) {
    auto value = original[r * features];
    auto left = std::isnan(value) || value < 0;
    for (unsigned g = 0; g < groups; ++g) {
      auto offset = (r * groups + g) * per_group + 1;
      float contribution = (left ? -1.5f : .5f) * (g + 1) * scale;
      float bias = margin ? -.125f : model.base_scores[g];
      for (std::size_t i = 0; i < per_group; ++i)
        Close(output[offset + i], i == 0 ? contribution : i == per_group - 1 ? bias : 0.f);
    }
  }
  ++cases;
  if (cases % 96 == 0) {
    std::printf("Checked %zu SHAP cases / %zu output values\n", cases, checked);
    std::fflush(stdout);
  }
}
int main() {
  try {
    for (unsigned features : {1u, 7u, 65u})
      for (unsigned groups : {1u, 3u})
        for (float scale : {0.f, .25f, -1.f})
          for (bool interactions : {false, true})
            for (bool margin : {false, true})
              for (bool alias : {false, true})
                for (std::size_t rows : {0ul, 1ul, 33ul, 4097ul})
                  Run(features, groups, scale, interactions, margin, alias, rows);
    for (bool interactions : {false, true})
      for (unsigned groups : {1u, 3u})
        for (std::size_t rows : {0ul, 1ul, 33ul, 4097ul})
          Run(7, groups, .25f, interactions, true, 2, rows);
    auto model = Model(1, 1);
    auto gpu = metal::MakeShap(model, 1, 1, false);
    std::vector<float> wrong(3), output(4);
    bool rejected = false;
    try {
      gpu->Predict(wrong, 2, {}, output);
    } catch (std::exception const&) {
      rejected = true;
    }
    Check(rejected, "Malformed input shape accepted");
    std::printf(
        "PASS %zu native SHAP cases / %zu analytic values; max absolute error %.9g; guards, "
        "aliases, empty inputs and batch bounds\n",
        cases, checked, max_error);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
