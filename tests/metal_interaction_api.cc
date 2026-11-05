// Independent set-based oracle for tree/constraints.cu, through the production API.
#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_constraints.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_sampling.h"
#include "../xgboost/src/tree/constraints.h"
#include "cuda_column_reference.h"

using namespace xgboost;
namespace metal = common::metal;
using Features = std::vector<bst_feature_t>;
using Groups = std::vector<Features>;
std::size_t checks = 0;
std::size_t resident_queries = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
  ++checks;
}
template <typename F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (dmlc::Error const&) {
    rejected = true;
  }
  Check(rejected, "Invalid input accepted");
}
struct Reference {
  Groups groups;
  std::map<bst_node_t, std::set<bst_feature_t>> nodes;
  void Split(bst_node_t node, bst_feature_t feature, bst_node_t left, bst_node_t right) {
    auto& parent = nodes[node];
    parent.insert(feature);
    for (auto const& group : groups) {
      if (std::find(group.begin(), group.end(), feature) != group.end())
        parent.insert(group.begin(), group.end());
    }
    for (auto child : {left, right}) nodes[child].insert(parent.begin(), parent.end());
  }
  Features Query(Features const& input, bst_node_t node) {
    if (node == 0) return input;
    std::set<bst_feature_t> result;
    for (auto feature : input)
      if (nodes[node].count(feature)) result.insert(feature);
    return {result.begin(), result.end()};
  }
  Features QueryNode(bst_node_t node) { return {nodes[node].begin(), nodes[node].end()}; }
};
struct Harness {
  std::unique_ptr<metal::InteractionConstraints> gpu;
  Reference ref;
  explicit Harness(bst_feature_t features, Groups groups) : ref{groups, {}} {
    gpu = metal::MakeInteractionConstraints(features, groups);
    // The GPU object must own its metadata after construction.
    for (auto& group : groups) std::fill(group.begin(), group.end(), features);
  }
  void Reset() {
    auto before = metal::GetStatistics();
    gpu->Reset();
    ref.nodes.clear();
    Check(metal::GetStatistics().interaction_reset_dispatches ==
              before.interaction_reset_dispatches + 1,
          "Reset did not execute on GPU");
  }
  void Split(bst_node_t node, bst_feature_t feature, bst_node_t left, bst_node_t right) {
    auto before = metal::GetStatistics();
    gpu->Split(node, feature, left, right);
    ref.Split(node, feature, left, right);
    Check(metal::GetStatistics().interaction_split_dispatches ==
              before.interaction_split_dispatches + 1,
          "Split did not execute on GPU");
  }
  void Query(Features input, bst_node_t node) {
    auto original = input;
    auto before = metal::GetStatistics();
    Check(gpu->Query(input, node) == ref.Query(input, node), "Wrong feature filter/order");
    Check(input == original, "Query modified input");
    Check(metal::GetStatistics().interaction_query_dispatches ==
              before.interaction_query_dispatches + (node != 0),
          "Wrong query dispatch count");
  }
  void QueryNode(bst_node_t node) {
    auto before = metal::GetStatistics();
    Check(gpu->QueryNode(node) == ref.QueryNode(node), "Wrong persistent node mask");
    Check(metal::GetStatistics().interaction_query_dispatches ==
              before.interaction_query_dispatches + 1,
          "Node query did not execute on GPU");
  }
};
void Overlap() {
  Harness h{4, {{0, 1}, {1, 2}}};
  h.Reset();
  h.QueryNode(0);  // Root mask starts empty, but root Query bypasses filtering.
  h.Query({3, 2, 2, 0, 1}, 0);
  h.Split(0, 0, 1, 2);
  Check(h.gpu->QueryNode(1) == Features({0, 1}), "First group incorrect");
  h.Split(1, 1, 3, 4);
  Check(h.gpu->QueryNode(3) == Features({0, 1, 2}), "CUDA cumulative union missing");
  h.Query({3, 2, 1, 2, 0}, 3);
  h.QueryNode(2);  // Splitting one sibling cannot enlarge the other.
  tree::TrainParam param;
  param.interaction_constraints = "[[0,1],[1,2]]";
  FeatureInteractionConstraintHost cpu;
  cpu.Configure(param, 4);
  cpu.Split(0, 0, 1, 2);
  cpu.Split(1, 1, 3, 4);
  Check(!cpu.Query(3, 2), "Fixture must distinguish CUDA union from CPU history filtering");
  // Existing child masks are ORed, not overwritten. Equal child IDs are supported.
  h.Split(2, 3, 3, 4);
  h.QueryNode(3);
  h.Split(4, 2, 17, 17);
  h.QueryNode(17);
  h.QueryNode(1);
  h.Reset();
  for (int node : {0, 1, 2, 3, 4, 17}) h.QueryNode(node);
  h.Split(0, 3, 1, 2);  // Unlisted feature does not inherit the previous tree.
  h.QueryNode(1);
}
void Sizes() {
  std::mt19937 rng(8821);
  for (bst_feature_t n : {1, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1025, 65537}) {
    Features all(n);
    std::iota(all.begin(), all.end(), 0);
    Groups groups{{}, {0, 0, n - 1}, {}};
    Features long_group;
    for (bst_feature_t f = 0; f < n; f += 2) long_group.push_back(f);
    groups.push_back(long_group);
    // Multiple overlapping sets stress concurrent atomic ORs and CSR deduplication.
    for (bst_feature_t f = 0; f < std::min(n, 33u); ++f) groups.push_back({0, f, f});
    Harness h{n, groups};
    for (int repeat = 0; repeat < 2; ++repeat) {
      h.Reset();
      h.QueryNode(0);
      h.Split(0, 0, 1, 2);
      h.Split(1, n - 1, 127, 128);  // Growth must preserve all prior masks.
      std::shuffle(all.begin(), all.end(), rng);
      h.Query(all, 0);
      h.Query(all, 128);
      Features sampled(all.begin(), all.begin() + (n + 1) / 2);
      sampled.insert(sampled.end(), all.begin(), all.begin() + (n + 1) / 2);
      h.Query(sampled, 127);
      h.Query({}, 127);
      for (int node : {0, 1, 2, 64, 127, 128}) h.QueryNode(node);
    }
    std::printf("Interaction feature count %u passed (reset, growth, ordering)\n", n);
    std::fflush(stdout);
  }
  Harness empty_groups{65, {}};
  empty_groups.Reset();
  empty_groups.Split(0, 32, 1, 2);
  empty_groups.Split(1, 64, 3, 4);
  empty_groups.Query({64, 31, 32, 0, 64}, 3);
  empty_groups.QueryNode(2);
  empty_groups.QueryNode(3);
}
void ErrorsAndBypasses() {
  auto before = metal::GetStatistics();
  auto disabled = metal::MakeInteractionConstraints(0, {{999}}, false);
  disabled->Reset();
  disabled->Split(-1, 999, -1, -1);
  Features input{999, 1, 999};
  Check(disabled->Query(input, -1) == input, "Disabled constraints must bypass input");
  Check(disabled->QueryNode(-1).empty(), "Disabled node query must return empty");
  auto zero = metal::MakeInteractionConstraints(0, {});
  zero->Reset();
  Check(zero->Query({}, 1).empty() && zero->QueryNode(0).empty(), "Zero-feature masks");
  auto after = metal::GetStatistics();
  Check(after.interaction_split_dispatches == before.interaction_split_dispatches &&
            after.interaction_query_dispatches == before.interaction_query_dispatches &&
            after.interaction_reset_dispatches == before.interaction_reset_dispatches &&
            after.interaction_buffer_growths == before.interaction_buffer_growths,
        "Disabled/empty operations must not dispatch");
  Reject([&] { metal::MakeInteractionConstraints(3, {{3}}); });
  Reject(
      [&] { metal::MakeInteractionConstraints(3, {{std::numeric_limits<bst_feature_t>::max()}}); });
  Reject([&] { metal::MakeInteractionConstraints(0x80000000u, {}); });
  Reject([&] { zero->Split(0, 0, 1, 2); });
  Harness h{3, {{0, 1}, {1, 2}}};
  h.Reset();
  h.Split(0, 0, 1, 2);
  Reject([&] { h.gpu->Split(-1, 0, 1, 2); });
  Reject([&] { h.gpu->Split(0, 0, -1, 2); });
  Reject([&] { h.gpu->Split(0, 0, 1, -1); });
  Reject([&] { h.gpu->Split(0, 0, 0, 2); });
  Reject([&] { h.gpu->Split(0, 0, 1, 0); });
  Reject([&] { h.gpu->Split(0, 3, 1, 2); });
  Reject([&] { h.gpu->Query({}, -1); });
  Reject([&] { h.gpu->QueryNode(-1); });
  Features invalid{0, 3};
  // Root bypass matches CUDA even for input outside the usual caller contract.
  Check(h.gpu->Query(invalid, 0) == invalid, "Root bypass altered input");
  Reject([&] { h.gpu->Query(invalid, 1); });  // GPU validates sampled feature IDs.
  h.Query({2, 1, 0, 1}, 1);
  h.QueryNode(1);
  h.Split(1, 1, 3, 4);
  h.QueryNode(3);
}

void ResidentQueries(bst_feature_t n, bool enabled) {
  Groups groups;
  if (n) {
    groups = {{0, n - 1}, {}};
    for (bst_feature_t f = 0; f < n; f += 2) groups.back().push_back(f);
  }
  auto gpu = metal::MakeInteractionConstraints(n, groups, enabled);
  Reference ref{groups, {}};
  auto device = DeviceOrd::MPS();
  Features input;
  for (bst_feature_t f = n; f > 0; --f) {
    input.push_back(f - 1);
    if (f % 3 == 0) input.push_back(f - 1);
  }
  if (!enabled) input = {999, 1, 999};  // Disabled bypass does not validate IDs.
  Features padded(input.size() + 2, 0xbadcafeu);
  std::copy(input.begin(), input.end(), padded.begin() + 1);
  HostDeviceVector<bst_feature_t> source(padded, device);
  auto source_view = source.ConstDeviceSpan().subspan(1, input.size());
  std::vector<HostDeviceVector<bst_feature_t>> snapshots, consumed;
  std::vector<Features> expected_snapshots, expected_consumed;
  auto remember = [&](HostDeviceVector<bst_feature_t> value, Features expected) {
    Check(value.DeviceCanRead() && !value.HostCanRead(), "Query did not return GPU ownership");
    auto count = (value.Size() + 1) / 2;
    consumed.emplace_back(count, 0, device);
    metal::SampleFeatures(value.ConstDeviceSpan(), {}, 731, consumed.back().DeviceSpan());
    expected_consumed.push_back(cuda_reference::SampleFeatures(expected, {}, 731, count));
    snapshots.push_back(std::move(value));
    expected_snapshots.push_back(std::move(expected));
    ++resident_queries;
  };
  auto before = metal::GetStatistics();
  gpu->Reset();
  remember(gpu->QueryDevice(source_view, 0), input);
  remember(gpu->QueryNodeDevice(0), {});
  if (!enabled) {
    remember(gpu->QueryDevice(source_view, -1), input);
    remember(gpu->QueryNodeDevice(-1), {});
  } else if (n) {
    gpu->Split(0, 0, 1, 2);
    ref.Split(0, 0, 1, 2);
    remember(gpu->QueryDevice(source_view, 1), ref.Query(input, 1));
    remember(gpu->QueryNodeDevice(2), ref.QueryNode(2));
    gpu->Split(1, n - 1, 511, 512);
    ref.Split(1, n - 1, 511, 512);
    remember(gpu->QueryDevice(source_view, 511), ref.Query(input, 511));
    // A resident result feeds another query, then a sample, with no host view.
    auto expected = ref.Query(expected_snapshots.back(), 512);
    remember(gpu->QueryDevice(snapshots.back().ConstDeviceSpan(), 512), expected);
    remember(gpu->QueryDevice({}, 512), {});
    remember(gpu->QueryNodeDevice(128), {});
    HostDeviceVector<bst_feature_t> invalid(1, n, device);
    Reject([&] { gpu->QueryDevice(invalid.ConstDeviceSpan(), 1); });
    Reject([&] { gpu->QueryDevice(source_view, -1); });
    Reject([&] { gpu->QueryNodeDevice(-1); });
  } else {
    remember(gpu->QueryDevice({}, 1), {});
  }
  gpu->Reset();
  remember(gpu->QueryNodeDevice(0), {});
  // Source can be replaced by its result; old input must live until publication.
  source = gpu->QueryDevice(source_view, 0);
  auto after = metal::GetStatistics();
  Check(after.feature_readback_bytes == before.feature_readback_bytes &&
            after.feature_host_input_bytes == before.feature_host_input_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes &&
            after.vector_host_copy_bytes == before.vector_host_copy_bytes,
        "Resident filtering staged or published feature payloads through CPU memory");
  Check(source.ConstHostVector() == input, "Replacing input with root result changed IDs");
  for (std::size_t i = 0; i < snapshots.size(); ++i) {
    Check(snapshots[i].ConstHostVector() == expected_snapshots[i],
          "Queued query snapshot changed after scratch reuse/growth/reset");
    Check(consumed[i].ConstHostVector() == expected_consumed[i],
          "Resident filter-to-sampler consumer differs from independent oracle");
  }
}

int main() {
  try {
    auto before = metal::GetStatistics();
    Overlap();
    Sizes();
    ErrorsAndBypasses();
    for (bst_feature_t n : {0, 1, 33, 257, 65537}) ResidentQueries(n, true);
    ResidentQueries(0, false);
    auto after = metal::GetStatistics();
    Check(after.interaction_buffer_growths > before.interaction_buffer_growths,
          "GPU growth was not exercised");
    Check(after.interaction_gpu_time_ns > before.interaction_gpu_time_ns, "No GPU time recorded");
    std::printf("PASS %zu interaction checks; %zu resident queries; reset=%llu split=%llu "
                "query=%llu growth=%llu\n",
                checks, resident_queries, (unsigned long long)after.interaction_reset_dispatches,
                (unsigned long long)after.interaction_split_dispatches,
                (unsigned long long)after.interaction_query_dispatches,
                (unsigned long long)after.interaction_buffer_growths);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
