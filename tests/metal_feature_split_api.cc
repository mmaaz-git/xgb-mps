// Check GPU feature-indexed task assembly against explicit host descriptors.
// The arithmetic kernels have separate independent CUDA-source oracle tests.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_split.h"
#include "../xgboost/src/data/gradient_index.h"
#include "cuda_gradient_reference.h"

using namespace xgboost;
namespace metal = common::metal;
using common::Span;
std::size_t cases = 0, comparisons = 0, rejected = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
  ++comparisons;
}
template <typename F> void Reject(F fn) {
  try { fn(); } catch (std::exception const&) { ++rejected; return; }
  throw std::runtime_error("Invalid feature split input accepted");
}
template <typename T> void Same(T const& a, T const& b, char const* message) {
  Check(std::memcmp(&a, &b, sizeof(T)) == 0, message);
}

void Run(unsigned targets, bool vector_leaf, unsigned onehot, bool constrained) {
  Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "1"}});
  auto device = DeviceOrd::MPS();
  constexpr unsigned rows = 16, bins = 12;
  std::vector<GradientPair> raw;
  std::vector<cuda_reference::Pair> oracle_input;
  for (unsigned r = 0; r < rows; ++r) for (unsigned t = 0; t < targets; ++t) {
    float g = (float(r % 4) - 1.75f) * (t + 1), h = 1.f + (r % 3) / 4.f;
    raw.emplace_back(g, h);
    oracle_input.push_back({g, h});
  }
  cuda_reference::GradientSampling oracle(oracle_input, rows, targets, 1.f, 0);
  std::vector<GradientPairInt64> root(targets);
  for (unsigned r = 0; r < rows; ++r) for (unsigned t = 0; t < targets; ++t) {
    auto v = oracle.original[r * targets + t];
    root[t] += GradientPairInt64{v.g, v.h};
  }
  std::vector<GradientPairPrecise> root_fp;
  for (unsigned t = 0; t < targets; ++t)
    root_fp.emplace_back(std::ldexp(double(root[t].GetQuantisedGrad()), -oracle.scales[t].g),
                         std::ldexp(double(root[t].GetQuantisedHess()), -oracle.scales[t].h));
  auto state = metal::MakeQuantizedGradients(linalg::MakeTensorView(&ctx, raw, rows, targets));
  std::vector<std::uint32_t> ptr{0, 4, 4, 8, bins};
  std::vector<FeatureType> types{FeatureType::kNumerical, FeatureType::kNumerical,
                                FeatureType::kCategorical, FeatureType::kNumerical};
  std::vector<std::int32_t> monotone = constrained ? std::vector<std::int32_t>{0, 0, 0, 1}
                                                  : std::vector<std::int32_t>{};
  common::HistogramCuts cuts{4};
  cuts.cut_ptrs_.HostVector() = ptr;
  cuts.cut_values_.HostVector() = {1, 2, 3, 4, 0, 1, 2, 3, 1, 2, 3, 4};
  SparsePage page;
  page.offset.HostVector() = {0};
  for (unsigned r = 0; r < rows; ++r) {
    for (auto f : {0u, 2u, 3u}) page.data.HostVector().emplace_back(f, float(r % 4));
    page.offset.HostVector().push_back(page.data.Size());
  }
  GHistIndexMatrix index{&ctx, page, types, std::move(cuts), 4, false, .2};
  common::RowSetCollection row_set;
  row_set.Data()->resize(rows);
  std::iota(row_set.Data()->begin(), row_set.Data()->end(), 0);
  row_set.Init();
  std::vector<std::unique_ptr<metal::HistogramBuffer>> caches;
  for (unsigned t = 0; t < targets; ++t) {
    auto hist = metal::MakeHistogram();
    hist->BindQuantized(state, t);
    auto cache = metal::MakeHistogramBuffer();
    cache->Resize(bins);
    std::vector<GradientPair> column;
    for (unsigned r = 0; r < rows; ++r) column.push_back(raw[r * targets + t]);
    hist->Build(0, index, row_set, {0}, column, {cache->Data()}, cache.get());
    caches.push_back(std::move(cache));
  }
  auto workspace = metal::MakeSplitWorkspace();
  auto reference_workspace = metal::MakeSplitWorkspace();
  auto stored = metal::MakeSplitWeightStorage(targets);
  std::vector<std::pair<unsigned, std::vector<metal::SplitNodeWeights>>> expected_weights;
  metal::SplitParameters parameters{1, .25f, 0, 0};
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    std::vector<std::vector<bst_feature_t>> sets{{3, 2, 0, 2, 1}, {1}, {2, 0}, {}};
    if (repeat == 1) sets = {{1}, {}};
    if (repeat == 2) {
      sets[0].resize(257);
      for (unsigned i = 0; i < sets[0].size(); ++i) sets[0][i] = i % 4;
    }
    std::vector<HostDeviceVector<bst_feature_t>> owners;
    std::vector<Span<bst_feature_t const>> features;
    std::vector<GradientPairInt64> node_parents, feature_parents;
    std::vector<GradientPairPrecise> node_fp;
    std::vector<metal::SplitBounds> node_bounds, feature_bounds;
    std::vector<metal::HistogramInput> inputs, reference_inputs;
    for (auto const& cache : caches) {
      inputs.push_back({cache.get(), {}});
      reference_inputs.push_back({cache.get(), {}});
    }
    std::vector<metal::QuantizedSplitTask> scalar_tasks;
    std::vector<metal::QuantizedMultiSplitTask> vector_tasks;
    std::vector<metal::SplitFinalizeNode> nodes;
    std::vector<metal::SplitFinalizeFeature> final_features;
    std::vector<unsigned> kinds;
    unsigned packed_bins = 0;
    for (unsigned node = 0; node < sets.size(); ++node) {
      std::vector<bst_feature_t> padded = {0xbadcafeu};
      padded.insert(padded.end(), sets[node].begin(), sets[node].end());
      padded.push_back(0xbadcafeu);
      owners.emplace_back(padded, device);
      features.push_back(owners.back().ConstDeviceSpan().subspan(1, sets[node].size()));
      node_parents.insert(node_parents.end(), root.begin(), root.end());
      node_fp.insert(node_fp.end(), root_fp.begin(), root_fp.end());
      metal::SplitBounds bound = constrained ? metal::SplitBounds{-2.f - node, 2.f + node}
                                             : metal::SplitBounds{-INFINITY, INFINITY};
      node_bounds.insert(node_bounds.end(), targets, bound);
      auto first = static_cast<unsigned>(final_features.size());
      for (auto& input : inputs) input.segments.push_back({0, 0, bins, 1, 0});
      for (auto f : sets[node]) {
        auto n = ptr[f + 1] - ptr[f];
        unsigned kind = types[f] == FeatureType::kCategorical ? (n < onehot ? 1 : 2) : 0;
        auto slot = static_cast<unsigned>(final_features.size());
        int mono = constrained ? monotone[f] : 0;
        scalar_tasks.push_back({packed_bins, packed_bins + n, kind, 4, root[0],
                                bound.lower, bound.upper, mono});
        vector_tasks.push_back({packed_bins, packed_bins + n * targets, kind, 4,
                                slot * targets, targets, mono, unsigned(constrained)});
        final_features.push_back({f, mono, vector_leaf ? 2 * slot * targets : 0});
        kinds.push_back(kind);
        feature_parents.insert(feature_parents.end(), root.begin(), root.end());
        feature_bounds.insert(feature_bounds.end(), targets, bound);
        for (unsigned t = 0; t < targets; ++t)
          reference_inputs[t].segments.push_back({ptr[f], packed_bins + t, n, targets, 0});
        packed_bins += n * targets;
      }
      nodes.push_back({first, static_cast<unsigned>(final_features.size()), node * targets, targets});
    }
    HostDeviceVector<GradientPairInt64> resident_parents(node_parents, device);
    HostDeviceVector<GradientPairPrecise> resident_fp(node_fp, device);
    std::vector<metal::SplitNodeResult> actual, expected;
    std::vector<metal::FeatureSplitResult> selected;
    auto before = metal::GetStatistics();
    metal::EvaluateFeatureSplits(inputs, features, ptr, types, monotone, onehot, 4,
                                 resident_parents.ConstDeviceSpan(), resident_fp.ConstDeviceSpan(),
                                 node_bounds, parameters, vector_leaf, &actual, &selected, nullptr,
                                 workspace.get());
    std::vector<metal::SplitWeightCopy> copies;
    for (unsigned node = 0; node < sets.size(); ++node) copies.push_back({node, repeat * 300 + node});
    stored->Store(*workspace, copies);  // GPU consumer before inspecting any result payload.
    auto after = metal::GetStatistics();
    Check(after.feature_host_input_bytes == before.feature_host_input_bytes &&
              after.feature_readback_bytes == before.feature_readback_bytes &&
              after.vector_readback_bytes == before.vector_readback_bytes &&
              after.split_sum_readback_bytes == before.split_sum_readback_bytes &&
              after.split_weight_readback_bytes == before.split_weight_readback_bytes,
          "Feature task/consumer chain used host payload publication");
    Check(after.feature_task_candidates - before.feature_task_candidates == final_features.size(),
          "GPU did not assemble every feature candidate");
    std::vector<metal::QuantizedSplitResult> scalar_results;
    std::vector<metal::QuantizedMultiSplitResult> vector_results;
    if (vector_leaf) {
      metal::EvaluateQuantizedMultiSplits(reference_inputs, vector_tasks, feature_parents,
          feature_bounds, parameters, &vector_results, nullptr, nullptr, nullptr,
          reference_workspace.get());
    } else {
      metal::EvaluateQuantizedSplits(reference_inputs, scalar_tasks, parameters, &scalar_results,
                                     nullptr, reference_workspace.get());
    }
    std::vector<metal::SplitNodeWeights> weights;
    metal::FinalizeSplitNodes(nodes, final_features, node_fp, node_bounds, parameters, vector_leaf,
                              &expected, &weights, reference_workspace.get());
    for (unsigned node = 0; node < sets.size(); ++node) {
      Same(actual[node], expected[node], "GPU/host task assembly chose different node results");
      auto winner = expected[node].winner;
      if (winner == 0xffffffffu) {
        Check(selected[node].feature == 0xffffffffu, "Invalid winner published a feature");
      } else {
        auto f = final_features[winner].feature;
        Check(selected[node].feature == f && selected[node].kind == kinds[winner] &&
                  selected[node].cut_begin == ptr[f], "Selected GPU metadata differs");
        if (vector_leaf) Same(selected[node].vector, vector_results[winner], "Vector split differs");
        else Same(selected[node].scalar, scalar_results[winner], "Scalar split differs");
      }
      expected_weights.push_back({repeat * 300 + node,
          {weights.begin() + node * targets, weights.begin() + (node + 1) * targets}});
      auto const& host = owners[node].ConstHostVector();
      Check(host.front() == 0xbadcafeu && host.back() == 0xbadcafeu &&
                std::equal(sets[node].begin(), sets[node].end(), host.begin() + 1),
            "Feature input or offset guard changed");
    }
    if (repeat == 0) {
      auto invalid_features = features;
      HostDeviceVector<bst_feature_t> invalid(1, 4, device);
      invalid_features[0] = invalid.ConstDeviceSpan();
      auto reject = [&](auto const& ids, auto const& cuts, auto const& feature_types) {
        Reject([&] { metal::EvaluateFeatureSplits(inputs, ids, cuts, feature_types, monotone,
            onehot, 4, resident_parents.ConstDeviceSpan(), resident_fp.ConstDeviceSpan(),
            node_bounds, parameters, vector_leaf, &actual, &selected, nullptr, workspace.get()); });
      };
      reject(invalid_features, ptr, types);
      auto invalid_cuts = ptr;
      invalid_cuts.back() = bins + 1;
      reject(features, invalid_cuts, types);
      invalid_cuts = ptr;
      invalid_cuts[2] = 3;
      reject(features, invalid_cuts, types);
      auto invalid_types = types;
      invalid_types[2] = static_cast<FeatureType>(2);
      reject(features, ptr, invalid_types);
    }
    ++cases;
  }
  workspace.reset();
  for (auto const& [node, expected] : expected_weights) {
    auto base = stored->Base(node), left = stored->Left(node), right = stored->Right(node);
    for (unsigned t = 0; t < targets; ++t) {
      Same(base[t], expected[t].base, "Queued base weight changed");
      Same(left[t], expected[t].left, "Queued left weight changed");
      Same(right[t], expected[t].right, "Queued right weight changed");
    }
  }
}
int main() {
  try {
    for (auto onehot : {1u, 8u}) for (bool constrained : {false, true}) {
      Run(1, false, onehot, constrained);
      for (auto targets : {1u, 3u, 33u}) Run(targets, true, onehot, constrained);
    }
    std::printf("PASS %zu feature-task cases / %zu exact checks / %zu rejected inputs\n",
                cases, comparisons, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
