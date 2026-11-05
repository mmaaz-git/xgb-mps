// Exact queued node sums against independent CUDA-source integer/split oracles.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_split.h"
#include "../xgboost/src/data/gradient_index.h"
#include "cuda_gradient_reference.h"
#include "cuda_multi_split_reference.h"

using namespace xgboost;
namespace metal = common::metal;
namespace ref = cuda_multi_split_reference;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
template <typename Fn> void Reject(Fn fn) {
  try { fn(); } catch (std::exception const&) { ++rejected; return; }
  throw std::runtime_error("Invalid sum storage request accepted");
}
void Same(GradientPairInt64 value, ref::Pair expected) {
  Check(value.GetQuantisedGrad() == expected.g && value.GetQuantisedHess() == expected.h,
        "Exact stored node sum differs from independent integer oracle");
  fields += 2;
}
void Same(GradientPairPrecise value, ref::Floating expected) {
  Check(std::memcmp(&expected.g, &value, sizeof(value)) == 0,
        "GPU dequantized node sum differs from independent binary64 oracle");
  fields += 2;
}
void Run(unsigned targets, bool cancellation) {
  Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "1"}});
  constexpr unsigned rows = 16;
  std::vector<GradientPair> raw;
  std::vector<cuda_reference::Pair> input;
  for (unsigned r = 0; r < rows; ++r)
    for (unsigned t = 0; t < targets; ++t) {
      float g = (float(r) - 7.25f) * (t + 1);
      if (cancellation) g = std::ldexp(r == 0 ? 0x1p54f : r == rows - 1 ? -0x1p54f : r == 1 ? 2.f : 1.f, t % 3);
      raw.emplace_back(g, 1.f);
      input.push_back({g, 1.f});
    }
  cuda_reference::GradientSampling oracle(input, rows, targets, 1.f, 0);
  std::vector<ref::Scale> scales;
  for (auto s : oracle.scales) scales.push_back({s.g, s.h});
  auto state = metal::MakeQuantizedGradients(linalg::MakeTensorView(&ctx, raw, rows, targets));
  auto storage = metal::MakeSplitSumStorage(state);
  auto workspace = metal::MakeSplitWorkspace();
  std::map<unsigned, std::vector<ref::Pair>> expected;
  expected[0].resize(targets);
  for (unsigned r = 0; r < rows; ++r)
    for (unsigned t = 0; t < targets; ++t) {
      auto p = oracle.original[r * targets + t];
      expected[0][t] = ref::Add(expected[0][t], {p.g, p.h});
    }

  SparsePage page;
  page.offset.HostVector() = {0};
  common::HistogramCuts cuts{1};
  cuts.cut_ptrs_.HostVector() = {0, rows};
  for (unsigned r = 0; r < rows; ++r) {
    page.data.HostVector().emplace_back(0, float(r));
    page.offset.HostVector().push_back(r + 1);
    cuts.cut_values_.HostVector().push_back(float(r + 1));
  }
  Check(page.Size() == rows, "Sparse-page fixture row count");
  GHistIndexMatrix index{&ctx, page, {}, std::move(cuts), 16, true, .2};
  std::vector<std::unique_ptr<metal::Histogram>> histograms;
  std::vector<std::unique_ptr<metal::HistogramBuffer>> caches;
  std::vector<metal::HistogramInput> inputs;
  for (unsigned t = 0; t < targets; ++t) {
    auto hist = metal::MakeHistogram();
    hist->BindQuantized(state, t);
    auto cache = metal::MakeHistogramBuffer();
    cache->Resize(rows);
    inputs.push_back({cache.get(), {}});
    for (unsigned k = 0; k < 3; ++k)
      inputs.back().segments.push_back({0, k * rows * targets + t, rows, targets, 0});
    histograms.push_back(std::move(hist));
    caches.push_back(std::move(cache));
  }
  HostDeviceVector<GradientPairPrecise> root_fp(2 * targets + 2, {13., 17.}, DeviceOrd::MPS());
  HostDeviceVector<GradientPairInt64> root_int(targets + 2, {19, 23}, DeviceOrd::MPS());
  auto fp = root_fp.DeviceSpan();
  auto exact = root_int.DeviceSpan();
  linalg::VectorView<GradientPairPrecise> root_view{
      fp.subspan(1, 2 * targets), {std::size_t{targets}}, {std::size_t{2}}, DeviceOrd::MPS()};
  auto before = metal::GetStatistics();
  state->SumRoot(&ctx, DeviceOrd::MPS(), root_view, exact.subspan(1, targets));
  storage->InitRoot(exact.subspan(1, targets));
  Reject([&] { storage->InitRoot(exact.subspan(1, targets)); });
  Reject([&] { storage->Gather(std::vector<unsigned>{1}); });
  Reject([&] { storage->Apply(0, 1, 2); });

  std::vector<metal::QuantizedMultiSplitTask> tasks;
  for (unsigned k = 0; k < 3; ++k)
    tasks.push_back({k * rows * targets, (k + 1) * rows * targets, k, rows, k * targets, targets, 0, 0});
  std::vector<metal::SplitBounds> bounds(3 * targets, {-INFINITY, INFINITY});
  std::vector<ref::Bounds> ref_bounds(3 * targets, {-INFINITY, INFINITY});
  std::vector<metal::QuantizedMultiSplitResult> actual;
  std::vector<std::vector<ref::Pair>> child_sums;
  std::vector<ref::Result> reference;
  auto evaluate = [&](unsigned node, std::vector<bst_idx_t> const& members) {
    common::RowSetCollection selected;
    *selected.Data() = members;
    selected.Init();
    for (unsigned t = 0; t < targets; ++t) {
      std::vector<GradientPair> column;
      for (unsigned r = 0; r < rows; ++r) column.push_back(raw[r * targets + t]);
      histograms[t]->Build(0, index, selected, {0}, column, {caches[t]->Data()}, caches[t].get());
    }
    std::vector<ref::Pair> histogram(3 * rows * targets), parents;
    for (unsigned k = 0; k < 3; ++k) {
      parents.insert(parents.end(), expected.at(node).begin(), expected.at(node).end());
      for (auto r : members)
        for (unsigned t = 0; t < targets; ++t) {
          auto p = oracle.original[r * targets + t];
          histogram[k * rows * targets + r * targets + t] = {p.g, p.h};
        }
    }
    auto gathered = storage->Gather(std::vector<unsigned>{node, node, node});
    metal::EvaluateQuantizedMultiSplits(inputs, tasks, gathered.integers, bounds, {1, 0, 0, 0},
                                       &actual, nullptr, nullptr, nullptr, workspace.get());
    child_sums.assign(3, {});
    reference.clear();
    for (unsigned k = 0; k < 3; ++k) {
      ref::Task task{};
      static_assert(sizeof(task) == sizeof(tasks[k]));
      std::memcpy(&task, &tasks[k], sizeof(task));
      std::vector<unsigned> sorted;
      auto result = ref::Evaluate(histogram, parents, scales, ref_bounds, task, {1, 0, 0, 0},
                                  &sorted, &child_sums[k]);
      if (std::memcmp(&actual[k], &result, sizeof(result))) {
        std::fprintf(stderr, "targets=%u cancellation=%d node=%u kind=%u gain=%a/%a bin=%d/%d dir=%u/%u partition=%u/%u\n",
                     targets, cancellation, node, k, actual[k].gain, result.gain,
                     actual[k].bin, result.bin, actual[k].default_left, result.missing,
                     actual[k].partition, result.partition);
        for (unsigned t = 0; t < targets; ++t) {
          auto p = gathered.integers[t];
          std::fprintf(stderr, "parent[%u]=%lld,%lld expected=%lld,%lld scale=%d,%d\n", t,
                       (long long)p.GetQuantisedGrad(), (long long)p.GetQuantisedHess(),
                       (long long)parents[t].g, (long long)parents[t].h, scales[t].ge, scales[t].he);
        }
        for (unsigned r = 0; r < rows; ++r) {
          auto p = caches[0]->Data()[r];
          auto e = ref::Dequantize(histogram[r * targets], scales[0]);
          std::fprintf(stderr, "hist[%u]=%a,%a expected=%a,%a\n", r, p.GetGrad(), p.GetHess(), e.g, e.h);
        }
        Check(false, "Selected split scalar differs");
      }
      Check(result.bin >= 0, "Fixture did not produce valid children");
      reference.push_back(result);
    }
  };
  auto save_expected = [&](unsigned candidate, unsigned left, unsigned right) {
    auto const& v = child_sums[candidate];
    expected[left] = {v.begin(), v.begin() + targets};
    expected[right] = {v.begin() + targets, v.end()};
  };
  std::vector<bst_idx_t> all(rows), left_rows, right_rows;
  std::iota(all.begin(), all.end(), 0);
  evaluate(0, all);
  // Overwrite the root's selection, keeping only the last accepted candidate.
  storage->Store(*workspace, std::vector<metal::SplitSumCopy>{{2, 0}});
  storage->Store(*workspace, std::vector<metal::SplitSumCopy>{{0, 0}});
  save_expected(0, 5, 17);
  auto split = reference[0];
  for (auto r : all) {
    bool left = split.missing ? int(r) < split.bin : int(r) <= split.bin;
    (left ? left_rows : right_rows).push_back(r);
  }
  storage->Apply(0, 5, 17);
  if (!cancellation) {
  evaluate(5, left_rows);
  storage->Store(*workspace, std::vector<metal::SplitSumCopy>{{2, 5}});
  save_expected(2, 257, 258);
  evaluate(17, right_rows);  // Workspace reuse must not corrupt queued node5.
  storage->Store(*workspace, std::vector<metal::SplitSumCopy>{{1, 17}});
  save_expected(1, 300, 301);
  Reject([&] { storage->Store(*workspace, std::vector<metal::SplitSumCopy>{{3, 17}}); });
  Reject([&] { storage->Store(*workspace, std::vector<metal::SplitSumCopy>{{0, 17}, {1, 17}}); });
  Reject([&] { storage->Store(*workspace, std::vector<metal::SplitSumCopy>{{0, 1}}); });
  storage->Apply(5, 257, 258);
  storage->Apply(17, 300, 301);
  Reject([&] { storage->Apply(5, 257, 400); });
  Reject([&] { storage->Apply(5, 400, 400); });
  }
  storage->Store(*workspace, {});
  metal::EvaluateQuantizedMultiSplits({}, {}, {}, {}, {1, 0, 0, 0}, &actual, nullptr, nullptr,
                                     nullptr, workspace.get());
  Reject([&] { storage->Store(*workspace, std::vector<metal::SplitSumCopy>{{0, 17}}); });

  std::vector<unsigned> ids = cancellation ? std::vector<unsigned>{17, 0, 5, 17}
                                            : std::vector<unsigned>{301, 0, 257, 17, 5, 258, 300, 257};
  auto gathered = storage->Gather(ids);
  // A real GPU consumer precedes the first CPU inspection of all node payloads.
  std::vector<metal::SplitFinalizeNode> nodes;
  std::vector<metal::SplitBounds> final_bounds(ids.size() * targets, {-INFINITY, INFINITY});
  for (unsigned i = 0; i < ids.size(); ++i) nodes.push_back({0, 0, i * targets, targets});
  std::vector<metal::SplitNodeResult> finalized;
  metal::FinalizeSplitNodes(nodes, {}, gathered.precise, final_bounds, {1, 0, 0, 0}, true,
                            &finalized, nullptr, workspace.get());
  auto after = metal::GetStatistics();
  Check(after.gradient_sum_readback_bytes == before.gradient_sum_readback_bytes &&
            after.split_sum_readback_bytes == before.split_sum_readback_bytes &&
            after.split_sum_host_input_bytes == before.split_sum_host_input_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes &&
            after.vector_upload_bytes == before.vector_upload_bytes,
        "Exact sum chain published/staged a payload on the CPU");
  Check(after.gradient_sum_shared_output_bytes - before.gradient_sum_shared_output_bytes ==
            targets * (sizeof(GradientPairInt64) + sizeof(GradientPairPrecise)), "Root publication bytes");
  Check(after.split_sum_store_dispatches - before.split_sum_store_dispatches == (cancellation ? 3 : 5) &&
            after.split_sum_apply_dispatches - before.split_sum_apply_dispatches == (cancellation ? 1 : 3) &&
            after.split_sum_gather_dispatches - before.split_sum_gather_dispatches == (cancellation ? 2 : 4),
        "Sum storage/gather/apply dispatch counts");
  Check(!root_int.HostCanRead() && !root_fp.HostCanRead(), "Root acquired a host mirror");
  bool lost_bits = false;
  for (unsigned i = 0; i < ids.size(); ++i) {
    double hess = 0;
    for (unsigned t = 0; t < targets; ++t) {
      auto p = expected.at(ids[i])[t];
      Same(gathered.integers[i * targets + t], p);
      Same(gathered.precise[i * targets + t], ref::Dequantize(p, scales[t]));
      lost_bits |= p.g != static_cast<std::int64_t>(double(p.g));
      hess += ref::Dequantize(p, scales[t]).h;
    }
    Check(finalized[i].parent_hess == hess, "GPU finalizer did not consume gathered parent");
  }
  if (cancellation) Check(lost_bits, "Fixture failed to distinguish integers from rounded doubles");
  auto rf = root_fp.ConstHostSpan();
  auto ri = root_int.ConstHostSpan();
  for (unsigned t = 0; t < targets; ++t) {
    Same(ri[t + 1], expected.at(0)[t]);
    Same(rf[1 + 2 * t], ref::Dequantize(expected.at(0)[t], scales[t]));
    Check(rf[2 + 2 * t] == GradientPairPrecise{13, 17}, "Strided root guard changed");
  }
  Check(ri.front() == GradientPairInt64{19, 23} && ri.back() == GradientPairInt64{19, 23} &&
            rf.front() == GradientPairPrecise{13, 17} && rf.back() == GradientPairPrecise{13, 17},
        "Root boundary guard changed");
  Check(storage->Gather({}).integers.empty(), "Empty gather has payload");
  ++cases;
  std::printf("Checked targets=%u cancellation=%d (%zu cases, %zu exact fields)\n",
              targets, cancellation, cases, fields);
  std::fflush(stdout);
}
int main() {
  try {
    Reject([] { metal::MakeSplitSumStorage(nullptr); });
    for (unsigned targets : {1u, 3u, 33u, 129u})
      for (bool cancellation : {false, true}) Run(targets, cancellation);
    std::printf("PASS %zu resident sum chains, %zu exact fields, %zu rejected calls\n", cases, fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
