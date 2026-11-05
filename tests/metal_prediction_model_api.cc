// Device tree publication: compare GPU packing with independently walked host
// views, then exercise mutations, snapshots, serialized reload and concurrent
// immutable readers. Existing predictor tests compare these paths with CPU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_prediction_model.h"
#include "xgboost/json.h"

using namespace xgboost;
namespace metal = common::metal;
using Trees = std::vector<std::unique_ptr<RegTree>>;
std::size_t cases = 0, checks = 0;
void Check(bool ok, char const* message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
auto Views(Trees const& trees, DeviceOrd device) {
  std::vector<metal::PredictionTreeView> views;
  for (auto const& tree : trees) {
    if (tree->IsMultiTarget())
      views.emplace_back(tree::MultiTargetTreeView{device, false, tree.get()});
    else
      views.emplace_back(tree::ScalarTreeView{device, false, tree.get()});
  }
  return views;
}
auto Reference(common::Span<metal::PredictionTreeView const> views,
               common::Span<bst_target_t const> groups, common::OptionalWeights weights) {
  metal::PredictionModel packed;
  for (std::size_t t = 0; t < views.size(); ++t) {
    std::visit(
        [&](auto const& view) {
          packed.trees.push_back({static_cast<unsigned>(packed.nodes.size()), view.NumTargets(),
                                  view.NumTargets() == 1 ? groups[t] : 0, weights[t]});
          for (bst_node_t i = 0; i < view.Size(); ++i) {
            metal::PredictionNode node;
            if constexpr (std::is_same_v<std::decay_t<decltype(view)>, tree::ScalarTreeView>) {
              if (view.IsDeleted(i)) {
                packed.nodes.push_back(node);
                continue;
              }
            }
            if (view.IsLeaf(i)) {
              node.leaf_begin = packed.leaves.size();
              if constexpr (std::is_same_v<std::decay_t<decltype(view)>, tree::ScalarTreeView>) {
                packed.leaves.push_back(view.LeafValue(i));
              } else {
                for (unsigned k = 0; k < view.NumTargets(); ++k)
                  packed.leaves.push_back(view.LeafValue(i)(k));
              }
            } else {
              node.left = view.LeftChild(i);
              node.right = view.RightChild(i);
              node.missing = view.DefaultChild(i);
              node.feature = view.SplitIndex(i);
              node.split = view.SplitCond(i);
              node.categorical = view.SplitType(i) == FeatureType::kCategorical;
              if (node.categorical) {
                auto bits = view.NodeCats(i);
                node.category_begin = packed.categories.size();
                node.category_size = bits.size();
                packed.categories.insert(packed.categories.end(), bits.begin(), bits.end());
              }
            }
            packed.nodes.push_back(node);
          }
        },
        views[t]);
  }
  return metal::MakePrediction(packed);
}
Trees MakeTrees(Context const& ctx, unsigned targets, bool cat, bool missing_left) {
  Trees trees;
  std::vector<unsigned> bits{0x80000001, 0x40000000, 0x01000001};
  for (unsigned t = 0; t < 3; ++t) {
    auto tree = std::make_unique<RegTree>(targets, 3);
    if (targets == 1) {
      if (cat) {
        tree->ExpandCategorical(0, t, bits, missing_left, 0, -1.f, 3.f, 1, 8, 2, 6);
      } else {
        tree->ExpandNode(0, t, .5f, missing_left, 0, -1.f, 3.f, 1, 8, 2, 6);
      }
      tree->ExpandNode(1, (t + 1) % 3, -0.f, !missing_left, 0, .25f, -.75f, 1, 2, 1, 1);
      // Exercise deleted slots; they must never be interpreted as active splits.
      if (t == 1) tree->ChangeToLeaf(1, 7.5f);
    } else {
      std::vector<float> base(targets, 0.f), left(targets), right(targets);
      for (unsigned k = 0; k < targets; ++k) {
        left[k] = -.25f * (k + 1);
        right[k] = .5f * (k + 1);
      }
      tree->SetRoot(linalg::MakeVec(DeviceOrd::CPU(), common::Span<float const>{base}), 8);
      tree::ExpandBatch batch{1.f};
      batch.Push(0, t, .5f, missing_left, base, left, right, 1, 2, 6,
                 cat ? common::Span<unsigned const>{bits} : common::Span<unsigned const>{});
      tree->Expand(&ctx, batch);
      // Reverse leaf order deliberately: node IDs are not leaf-weight indices.
      std::vector<float> weights = right;
      weights.insert(weights.end(), left.begin(), left.end());
      tree->SetLeaves({2, 1}, weights);
    }
    trees.push_back(std::move(tree));
  }
  return trees;
}
std::vector<float> Predict(metal::Prediction* prediction, unsigned groups, bool leaf,
                           unsigned ntrees) {
  std::vector<float> data(257 * 3);
  float values[]{NAN,  -INFINITY, -1.f, -0.f, 0.f,  .25f, .5f,  1.f,        31.f,
                 32.f, 33.f,      63.f, 64.f, 71.f, 95.f, 96.f, 16777216.f, INFINITY};
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = values[(i * 7 + i / 3) % 18];
  auto columns = leaf ? ntrees : groups;
  std::vector<float> output(257 * columns, .125f);
  prediction->Predict(data, 257, 3, columns, output, leaf);
  return output;
}
void Compare(std::vector<float> const& a, std::vector<float> const& b) {
  Check(a.size() == b.size(), "Output shape");
  for (std::size_t i = 0; i < a.size(); ++i) Check(a[i] == b[i], "Packed traversal differs");
}
void Case(Context const& ctx, unsigned targets, bool cat, bool missing_left, bool weighted) {
  auto trees = MakeTrees(ctx, targets, cat, missing_left);
  unsigned groups = targets == 1 ? 3 : targets;
  HostDeviceVector<bst_target_t> tree_groups(std::vector<bst_target_t>{2, 0, 1}, DeviceOrd::MPS());
  auto groups_host = tree_groups.ConstHostVector();
  std::vector<float> weight_values{.25f, -2.f, 0.f};
  auto weights = weighted ? common::OptionalWeights{weight_values} : common::OptionalWeights{1.f};
  std::unique_ptr<metal::Prediction> old;
  std::vector<float> old_values, old_leaves;
  for (unsigned pass = 0; pass < 4; ++pass) {
    if (pass == 1) {
      // Replace values without changing node/tree count.
      if (targets == 1) {
        (*trees[0])[2].SetLeaf(15.25f);
        if (!cat) (*trees[0])[0].SetSplit(2, 31.5f, !missing_left);
      } else {
        std::vector<float> new_weights(targets * 2);
        for (std::size_t i = 0; i < new_weights.size(); ++i) new_weights[i] = i * .25f - 2.f;
        trees[0]->SetLeaves({1, 2}, new_weights);
      }
    } else if (pass == 2) {
      // Reload into the same objects, after their device mirrors were populated.
      auto replacement = MakeTrees(ctx, targets, !cat, !missing_left);
      for (std::size_t t = 0; t < trees.size(); ++t) {
        Json saved{Object{}};
        replacement[t]->SaveModel(&saved);
        trees[t]->LoadModel(saved);
      }
    }
    auto host = Views(trees, DeviceOrd::CPU());
    auto ref = Reference(host, groups_host, weights);
    auto before_import = metal::GetStatistics();
    auto views = Views(trees, DeviceOrd::MPS());
    auto device_groups = tree_groups.ConstDeviceSpan();
    auto before = metal::GetStatistics();
    if (pass == 0)
      Check(before.vector_upload_bytes > before_import.vector_upload_bytes,
            "First device publication did not create mirrors");
    if (pass == 3)
      Check(before.vector_upload_bytes == before_import.vector_upload_bytes,
            "Read-only CPU/MPS handoff reimported model");
    auto gpu = metal::MakePrediction(views, device_groups, weights);
    auto after = metal::GetStatistics();
    Check(after.prediction_model_pack_dispatches - before.prediction_model_pack_dispatches == 3,
          "Packing did not run on Metal");
    Check(after.prediction_model_shared_input_bytes > before.prediction_model_shared_input_bytes,
          "No shared model bindings");
    Check(after.prediction_model_staging_bytes == before.prediction_model_staging_bytes,
          "Resident model staged on host");
    Check(after.vector_readback_bytes == before.vector_readback_bytes &&
              after.vector_upload_bytes == before.vector_upload_bytes,
          "Packing copied payload through host");
    auto values = Predict(gpu.get(), groups, false, 3);
    auto leaves = Predict(gpu.get(), groups, true, 3);
    Compare(values, Predict(ref.get(), groups, false, 3));
    Compare(leaves, Predict(ref.get(), groups, true, 3));
    if (old) {
      Compare(old_values, Predict(old.get(), groups, false, 3));
      Compare(old_leaves, Predict(old.get(), groups, true, 3));
    } else {
      old = std::move(gpu);
      old_values = values;
      old_leaves = leaves;
    }
    // Sliced descriptors use local group indexing, matching GBTreeModelView.
    auto slice =
        metal::MakePrediction(common::Span<metal::PredictionTreeView const>{views}.subspan(1),
                              device_groups.subspan(1), common::OptionalWeights{1.f});
    auto slice_ref = Reference(common::Span<metal::PredictionTreeView const>{host}.subspan(1),
                               common::Span<bst_target_t const>{groups_host}.subspan(1),
                               common::OptionalWeights{1.f});
    Compare(Predict(slice.get(), groups, false, 2), Predict(slice_ref.get(), groups, false, 2));
    ++cases;
  }
  trees.clear();
  Compare(old_values, Predict(old.get(), groups, false, 3));
  Compare(old_leaves, Predict(old.get(), groups, true, 3));
}
void Concurrent(Context const& ctx) {
  auto trees = MakeTrees(ctx, 3, true, true);
  HostDeviceVector<bst_target_t> groups(3, 0, DeviceOrd::MPS());
  auto views = Views(trees, DeviceOrd::MPS());
  auto group_span = groups.ConstDeviceSpan();
  auto reference = metal::MakePrediction(views, group_span, common::OptionalWeights{1.f});
  auto expected = Predict(reference.get(), 3, false, 3);
  std::vector<std::future<std::vector<float>>> readers;
  for (unsigned i = 0; i < 4; ++i) {
    readers.push_back(std::async(std::launch::async, [&] {
      auto snapshot = metal::MakePrediction(views, group_span, common::OptionalWeights{1.f});
      return Predict(snapshot.get(), 3, false, 3);
    }));
  }
  for (auto& reader : readers) Compare(reader.get(), expected);
  ++cases;
}
void LargeTree() {
  Trees trees;
  auto tree = std::make_unique<RegTree>(1, 3);
  for (int i = 0; i < 511; ++i) {
    tree->ExpandNode(i, i % 3, float(i % 67), i % 2, 0, float(i) / 8, -float(i) / 4, 1, 8, 2, 6);
  }
  trees.push_back(std::move(tree));
  HostDeviceVector<bst_target_t> groups(1, 0, DeviceOrd::MPS());
  auto host = Views(trees, DeviceOrd::CPU());
  auto reference = Reference(host, std::vector<bst_target_t>{0}, common::OptionalWeights{1.f});
  auto device = Views(trees, DeviceOrd::MPS());
  auto packed =
      metal::MakePrediction(device, groups.ConstDeviceSpan(), common::OptionalWeights{1.f});
  for (bool leaf : {false, true})
    Compare(Predict(packed.get(), 1, leaf, 1), Predict(reference.get(), 1, leaf, 1));
  ++cases;
}
int main() {
  try {
    Context ctx;
    ctx.Init({{"nthread", "2"}});
    for (unsigned targets : {1, 3, 17}) {
      for (bool cat : {false, true})
        for (bool missing_left : {false, true})
          for (bool weighted : {false, true}) Case(ctx, targets, cat, missing_left, weighted);
      std::printf("Checked model publication targets=%u: %zu cases / %zu checks\n", targets, cases,
                  checks);
      std::fflush(stdout);
    }
    // Both node packing and vector-weight copying cross a 256-thread group.
    LargeTree();
    Case(ctx, 129, true, false, true);
    Concurrent(ctx);
    auto empty = metal::MakePrediction({}, {}, common::OptionalWeights{1.f});
    auto unchanged = Predict(empty.get(), 1, false, 0);
    for (auto value : unchanged) Check(value == .125f, "Empty model changed predictions");
    ++cases;
    Check(metal::GetStatistics().vector_live_buffers == 0 &&
              metal::GetStatistics().vector_capacity_bytes == 0,
          "Model mirror lifetime leak");
    std::printf("PASS %zu prediction model cases / %zu checks\n", cases, checks);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
