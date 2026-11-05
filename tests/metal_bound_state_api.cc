// Persistent GPU bounds: CUDA's allocation/growth rules and exact scalar/vector updates.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_split.h"
#include "../xgboost/src/tree/split_evaluator.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t values = 0, updates = 0, cases = 0;
std::size_t gathers = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Same(float const* a, float const* b, std::size_t n) {
  Check(std::memcmp(a, b, n * sizeof(float)) == 0, "Bound values differ");
  values += n;
}
template <typename F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (dmlc::Error const&) {
    rejected = true;
  }
  Check(rejected, "Invalid bound operation accepted");
}
void Consume(common::Span<metal::SplitBounds const> bounds,
             std::vector<metal::SplitBounds> const& expected, unsigned targets, bool multi,
             metal::SplitWorkspace* workspace) {
  std::vector<metal::SplitFinalizeNode> nodes;
  std::vector<metal::SplitWeightCopy> copies;
  for (unsigned n = 0; n < expected.size() / targets; ++n) {
    nodes.push_back({0, 0, n * targets, targets});
    copies.push_back({n, n});
  }
  HostDeviceVector<GradientPairPrecise> parents(expected.size(), {2.0, 1.0}, DeviceOrd::MPS());
  auto stored = metal::MakeSplitWeightStorage(targets);
  std::vector<metal::SplitNodeResult> actual, oracle;
  auto before = metal::GetStatistics();
  metal::FinalizeSplitNodes(nodes, {}, parents.ConstDeviceSpan(), bounds, {1, 0, 0, 0}, multi,
                            &actual, nullptr, workspace);
  stored->Store(*workspace, copies);
  auto after = metal::GetStatistics();
  Check(after.split_bounds_host_copy_bytes == before.split_bounds_host_copy_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes &&
            after.split_weight_readback_bytes == before.split_weight_readback_bytes,
        "Gather/finalizer/weight consumer copied payloads through CPU");
  auto reference = metal::MakeSplitWorkspace();
  metal::FinalizeSplitNodes(nodes, {}, parents.ConstDeviceSpan(), expected, {1, 0, 0, 0}, multi,
                            &oracle, nullptr, reference.get());
  Check(actual.size() == oracle.size() &&
            !std::memcmp(actual.data(), oracle.data(), actual.size() * sizeof(actual[0])),
        "Gathered bounds changed split finalization");
  Check(bounds.size() == expected.size() &&
            !std::memcmp(bounds.data(), expected.data(), bounds.size_bytes()),
        "Gathered bound bits differ");
  values += bounds.size() * 2;
}

void GatherEdges() {
  auto workspace = metal::MakeSplitWorkspace();
  auto state = metal::MakeSplitBoundsStorage(6);
  float a = -.5f, b = .25f;
  state->Update(0, 1, 2, 1, {&a, 1}, {&b, 1}, false);
  std::vector<std::uint32_t> ids{2, 0, 1, 2};
  float limit = std::numeric_limits<float>::max();
  std::vector<metal::SplitBounds> expected{{-.125f, limit}, {-limit, limit},
                                           {-limit, -.125f}, {-.125f, limit}};
  auto before = metal::GetStatistics();
  auto snapshot = metal::GatherSplitBounds(ids, 1, state->Lower(), state->Upper(), workspace.get());
  state->Grow(513);
  state->Update(0, 1, 2, -1, {&b, 1}, {&b, 1}, false);
  state.reset();
  Check(metal::GetStatistics().split_bounds_host_copy_bytes == before.split_bounds_host_copy_bytes,
        "Gather snapshot used a host copy");
  Consume(snapshot, expected, 1, false, workspace.get());
  ++gathers;

  // Both source arrays alias the old gather. A new publication cannot overwrite
  // inputs while other GPU threads still read them.
  common::Span<float const> alias{reinterpret_cast<float const*>(snapshot.data()), snapshot.size() * 2};
  ids = {3, 1, 2, 0};
  std::vector<float> flat;
  for (auto bound : expected) { flat.push_back(bound.lower); flat.push_back(bound.upper); }
  expected.clear();
  for (auto id : ids) for (unsigned t = 0; t < 2; ++t)
    expected.push_back({flat[id * 2 + t], flat[id * 2 + t]});
  snapshot = metal::GatherSplitBounds(ids, 2, alias, alias, workspace.get());
  Consume(snapshot, expected, 2, true, workspace.get());
  ++gathers;

  before = metal::GetStatistics();
  auto valid = snapshot;
  Reject([&] { metal::GatherSplitBounds(ids, 0, {}, {}, workspace.get()); });
  Reject([&] { metal::GatherSplitBounds(ids, 1, {}, {}, nullptr); });
  Reject([&] { metal::GatherSplitBounds(ids, 1, {flat.data(), 1}, {}, workspace.get()); });
  Reject([&] { metal::GatherSplitBounds(ids, 3, flat, flat, workspace.get()); });
  Reject([&] { metal::GatherSplitBounds(ids, 1, {flat.data(), 1}, {flat.data(), 1}, workspace.get()); });
  Reject([&] { metal::GatherSplitBounds(ids, 0xffffffffu, {}, {}, workspace.get()); });
  Check(metal::GetStatistics().split_bounds_gather_dispatches == before.split_bounds_gather_dispatches,
        "Invalid gathers dispatched");
  Consume(valid, expected, 2, true, workspace.get());  // Rejections preserve the previous result.

  auto bytes = static_cast<std::size_t>(getpagesize());
  auto* memory = static_cast<float*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANON, -1, 0));
  Check(memory != MAP_FAILED, "mmap failed");
  std::copy(flat.begin(), flat.end(), memory);
  Check(mprotect(memory, bytes, PROT_READ) == 0, "mprotect failed");
  before = metal::GetStatistics();
  snapshot = metal::GatherSplitBounds(ids, 2, {memory, flat.size()}, {memory, flat.size()}, workspace.get());
  Check(metal::GetStatistics().split_bounds_host_copy_bytes - before.split_bounds_host_copy_bytes ==
            2 * flat.size() * sizeof(float), "Read-only diagnostic fallback accounting");
  Check(munmap(memory, bytes) == 0, "munmap failed");
  Consume(snapshot, expected, 2, true, workspace.get());
  ++gathers;
  before = metal::GetStatistics();
  Check(metal::GatherSplitBounds({}, 1, {}, {}, workspace.get()).empty(), "Empty gather output");
  Check(metal::GetStatistics().split_bounds_gather_dispatches == before.split_bounds_gather_dispatches,
        "Empty gather dispatched");
}
void Storage() {
  auto before = metal::GetStatistics();
  auto state = metal::MakeSplitBoundsStorage(0);
  Check(state->Lower().empty() && state->Upper().empty(), "Empty bound storage");
  state->Grow(0);
  Check(metal::GetStatistics().split_bounds_init_dispatches == before.split_bounds_init_dispatches,
        "Empty storage dispatched");
  std::vector<float> lo, hi;
  for (std::size_t n : {1, 3, 31, 32, 33, 255, 256, 257, 65537}) {
    before = metal::GetStatistics();
    state->Grow(n);
    lo.resize(n, -std::numeric_limits<float>::max());
    hi.resize(n, std::numeric_limits<float>::max());
    Check(state->Lower().size() == n && state->Upper().size() == n, "Wrong bound size");
    Same(state->Lower().data(), lo.data(), n);
    Same(state->Upper().data(), hi.data(), n);
    Check(metal::GetStatistics().split_bounds_init_dispatches ==
              before.split_bounds_init_dispatches + 1,
          "Missing GPU initialization/growth");
    auto lower = state->Lower(), upper = state->Upper();
    state->Grow(n);
    state->Grow(n - 1);
    Check(state->Lower().data() == lower.data() && state->Upper().data() == upper.data(),
          "No-op growth invalidated views");
    Check(metal::GetStatistics().split_bounds_init_dispatches ==
              before.split_bounds_init_dispatches + 1,
          "No-op growth dispatched");
    if (n >= 3) {
      float a = -.5f, b = .25f;
      state->Update(0, 1, 2, 1, {&a, 1}, {&b, 1}, false);
      hi[1] = lo[2] = -.125f;
      lo[1] = lo[0];
      hi[2] = hi[0];
      Same(lower.data(), lo.data(), n);  // Existing views observe GPU updates.
      Same(upper.data(), hi.data(), n);
      // Inputs can alias the buffer being updated; they must be snapshotted.
      state->Update(0, 1, 2, -1, lower.subspan(2, 1), upper.subspan(1, 1), false);
      lo[1] = hi[2] = -.125f;
      hi[1] = hi[0];
      lo[2] = lo[0];
      Same(lower.data(), lo.data(), n);
      Same(upper.data(), hi.data(), n);
      updates += 2;
    }
    ++cases;
  }
  before = metal::GetStatistics();
  Reject([&] { state->Grow(std::size_t{1} << 32); });
  float a = 0, b = 1;
  Reject([&] { state->Update(0, 0, 2, 1, {&a, 1}, {&b, 1}, false); });
  Reject([&] { state->Update(0, 1, 1, 1, {&a, 1}, {&b, 1}, false); });
  Reject([&] { state->Update(0, 1, 65537, 1, {&a, 1}, {&b, 1}, false); });
  Reject([&] { state->Update(0, 1, 2, 1, {}, {&b, 1}, false); });
  Check(
      metal::GetStatistics().split_bounds_init_dispatches == before.split_bounds_init_dispatches &&
          metal::GetStatistics().split_bounds_dispatches == before.split_bounds_dispatches,
      "Invalid bounds dispatched");
  a = std::numeric_limits<float>::quiet_NaN();
  Reject([&] { state->Update(0, 1, 2, 1, {&a, 1}, {&b, 1}, false); });
  Same(state->Lower().data(), lo.data(), lo.size());
  Same(state->Upper().data(), hi.data(), hi.size());
}
void Evaluators() {
  for (bool multi : {false, true}) {
    for (bst_target_t targets : {1, 2, 33, 129}) {
      if (!multi && targets != 1) continue;
      for (int monotone : {-1, 0, 1}) {
        tree::TrainParam param;
        // Keep constraints enabled even when updating an unconstrained feature.
        param.monotone_constraints = {monotone, 1};
        tree::TreeEvaluator cpu{param, 2, DeviceOrd::CPU(), targets};
        auto before = metal::GetStatistics();
        tree::TreeEvaluator gpu{param, 2, DeviceOrd::CPU(), targets, true};
        auto workspace = metal::MakeSplitWorkspace();
        Check(metal::GetStatistics().split_bounds_init_dispatches ==
                  before.split_bounds_init_dispatches + 1,
              "Evaluator did not initialize on GPU");
        std::size_t n = 256 * targets;
        auto compare = [&] {
          auto c = cpu.GetEvaluator(), g = gpu.GetEvaluator();
          Same(c.lower, g.lower, n);
          Same(c.upper, g.upper, n);
        };
        compare();
        for (auto ids :
             {std::vector<int>{0, 1, 2}, {1, 127, 128}, {127, 255, 256}, {255, 1023, 1024}}) {
          std::vector<float> lw(targets), rw(targets);
          for (std::size_t t = 0; t < targets; ++t) {
            // Include exact subnormals, signed zeros and ordinary positive/negative weights.
            lw[t] = std::ldexp(float(int(t % 5) - 2), t % 2 ? 0 : -145);
            rw[t] = std::ldexp(float(int(t % 7) - 3), t % 2 ? 0 : -145);
          }
          lw[0] = -0.f;
          HostDeviceVector<float> resident_left(lw, DeviceOrd::MPS()),
              resident_right(rw, DeviceOrd::MPS());
          before = metal::GetStatistics();
          if (multi) {
            auto left = linalg::MakeVec(lw.data(), lw.size());
            auto right = linalg::MakeVec(rw.data(), rw.size());
            cpu.AddSplit(ids[0], ids[1], ids[2], 0, left, right);
            gpu.AddSplit(ids[0], ids[1], ids[2], 0,
                          linalg::MakeVec(DeviceOrd::MPS(), resident_left.ConstDeviceSpan()),
                          linalg::MakeVec(DeviceOrd::MPS(), resident_right.ConstDeviceSpan()));
            auto after = metal::GetStatistics();
            Check(after.vector_readback_bytes == before.vector_readback_bytes &&
                      after.vector_upload_bytes == before.vector_upload_bytes &&
                      after.model_host_copy_bytes == before.model_host_copy_bytes,
                  "Bound consumer copied vector weights through host memory");
            Check(!resident_left.HostCanRead() && !resident_right.HostCanRead(),
                  "Bound consumer requested a weight host mirror");
          } else {
            cpu.AddSplit(ids[0], ids[1], ids[2], 0, lw[0], rw[0]);
            gpu.AddSplit(ids[0], ids[1], ids[2], 0, lw[0], rw[0]);
          }
          auto required = std::size_t(ids[2] * 2 + 1) * targets;
          Check(metal::GetStatistics().split_bounds_init_dispatches ==
                    before.split_bounds_init_dispatches + (required > n),
                "Wrong growth dispatch");
          Check(
              metal::GetStatistics().split_bounds_dispatches == before.split_bounds_dispatches + 1,
              "Missing resident bound update");
          n = std::max(n, required);
          std::vector<std::uint32_t> selected{unsigned(ids[2]), unsigned(ids[1]), 0, unsigned(ids[2])};
          before = metal::GetStatistics();
          auto gathered = gpu.GatherBounds(selected, workspace.get());
          Check(metal::GetStatistics().split_bounds_gather_values - before.split_bounds_gather_values ==
                    selected.size() * targets &&
                    metal::GetStatistics().split_bounds_host_copy_bytes == before.split_bounds_host_copy_bytes,
                "Evaluator bounds were not gathered on GPU");
          std::vector<metal::SplitBounds> expected;
          auto c = cpu.GetEvaluator();
          for (auto node : selected) for (unsigned t = 0; t < targets; ++t)
            expected.push_back({c.lower[node * targets + t], c.upper[node * targets + t]});
          Consume(gathered, expected, targets, multi, workspace.get());
          ++gathers;
          compare();
          ++updates;
        }
        ++cases;
      }
    }
  }
  for (auto constraints : {std::vector<int>{}, {0, 0}}) {
    tree::TrainParam param;
    param.monotone_constraints = constraints;
    auto before = metal::GetStatistics();
    tree::TreeEvaluator gpu{param, 2, DeviceOrd::CPU(), 1, true};
    Check(!gpu.GetEvaluator().has_constraint, "Zero constraints enabled bounds");
    gpu.AddSplit(0, 1, 2, 0, -.5f, .5f);
    Check(metal::GetStatistics().split_bounds_init_dispatches ==
                  before.split_bounds_init_dispatches &&
              metal::GetStatistics().split_bounds_dispatches == before.split_bounds_dispatches,
          "Unconstrained evaluator dispatched bounds");
    auto workspace = metal::MakeSplitWorkspace();
    std::vector<std::uint32_t> ids{512, 0, 512};
    auto snapshot = gpu.GatherBounds(ids, workspace.get());
    Consume(snapshot, std::vector<metal::SplitBounds>(ids.size(), {-INFINITY, INFINITY}),
            1, false, workspace.get());
    ++gathers;
  }
}
int main() {
  try {
    Storage();
    Evaluators();
    GatherEdges();
    std::printf("PASS %zu bound-storage cases / %zu updates / %zu exact values\n", cases, updates,
                values);
    std::printf("PASS %zu GPU bound gather/consumer chains, snapshots, aliases and invalid inputs\n", gathers);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
