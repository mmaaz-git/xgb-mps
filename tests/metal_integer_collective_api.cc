// Global integer tree statistics against independent CUDA-derived arithmetic.
#include <execinfo.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/collective/broadcast.h"
#include "../xgboost/src/common/metal_collective.h"
#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_leaf.h"
#include "../xgboost/src/common/metal_reduction.h"
#include "../xgboost/src/common/metal_split.h"
#include "../xgboost/src/common/metal_training.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/data/gradient_index.h"
#include "../xgboost/src/tree/param.h"
#include "../xgboost/src/tree/tree_view.h"
#include "cuda_gradient_reference.h"
#include "xgboost/c_api.h"
#include "xgboost/gradient.h"
#include "xgboost/json.h"
#include "xgboost/task.h"
#include "xgboost/tree_updater.h"

using namespace xgboost;
namespace metal = common::metal;
namespace ref = cuda_reference;
std::string current;
char const* phase = "initialization";
std::size_t cases = 0, fields = 0, rejected = 0, cpu_fields = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Api(int code) {
  if (code) throw std::runtime_error(XGBGetLastError());
}
template <typename T>
void Same(T a, T b) {
  ++fields;
  if (std::memcmp(&a, &b, sizeof(T))) {
    std::fprintf(stderr, "actual %a expected %a\n", double(a), double(b));
    Check(false, "Exact field mismatch");
  }
}
template <typename Fn>
void Reject(Fn&& fn) {
  try {
    fn();
  } catch (std::exception const&) {
    ++rejected;
    return;
  }
  Check(false, "Invalid call accepted");
}
struct Sum {
  std::uint64_t g{0}, h{0};
  void Add(ref::IntegerPair value) {
    g += value.g;
    h += value.h;
  }
  GradientPairInt64 Integer() const {
    std::int64_t sg, sh;
    std::memcpy(&sg, &g, 8);
    std::memcpy(&sh, &h, 8);
    return {sg, sh};
  }
  GradientPairPrecise Floating(ref::Scale scale) const {
    auto v = Integer();
    return {std::ldexp(double(v.GetQuantisedGrad()), -scale.g),
            std::ldexp(double(v.GetQuantisedHess()), -scale.h)};
  }
};
void SamePair(GradientPairPrecise a, GradientPairPrecise b) {
  Same(a.GetGrad(), b.GetGrad());
  Same(a.GetHess(), b.GetHess());
}
std::size_t Rows(int rank, unsigned layout) {
  if (layout == 3 || (layout == 1 && rank == 0) || (layout == 2 && rank != 0)) return 0;
  return layout == 4 ? 1 : 129 + rank * 17;
}
struct Reference {
  std::size_t targets;
  std::vector<ref::Scale> scales;
  std::vector<std::vector<ref::Pair>> original, replay;
  std::vector<std::vector<ref::IntegerPair>> integers;
  Reference(int world, unsigned layout, unsigned mode, unsigned sampling, std::size_t t)
      : targets{t}, scales(t), original(world), replay(world), integers(world) {
    std::size_t rows = 0;
    std::vector<double> bounds(4 * t);
    for (int rank = 0; rank < world; ++rank) {
      auto n = Rows(rank, layout);
      rows += n;
      original[rank].resize(n * t);
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t col = 0; col < t; ++col) {
          float g = std::ldexp(float(int(row % 13) - 6), rank + int(col % 3) - 4);
          float h = std::ldexp(float(1 + row % 7), int(col % 3) - 3);
          if (mode == 1) {
            g = std::ldexp(float(int(row % 7) - 3), rank == 0 ? 100 : -80);
            h = row == 0 ? 0x1p100f : std::numeric_limits<float>::denorm_min();
          }
          if (mode == 2) g = h = 0;
          if (mode == 3) {
            // A local binary64 view loses the two small terms beside 2^54.
            // Integer global reduction must recover them after rank 1 cancels
            // rank 0's large term. Rows share one histogram bin and one leaf.
            g = rank == 0 && (row == 9 || row == 17) ? std::ldexp(1.f, col % 3) : 0.f;
            if (row == 1 && rank < 2) g = std::ldexp(rank == 0 ? 1.f : -1.f, 54 + col % 3);
            h = 1.f;
          }
          auto& v = original[rank][row * t + col];
          v = {g, h};
          for (unsigned k = 0; k < 2; ++k) {
            double value = k ? h : g;
            bounds[col * 4 + k + (value < 0 ? 2 : 0)] += std::abs(value);
          }
        }
    }
    for (std::size_t col = 0; col < t; ++col)
      scales[col] = {ref::Exponent(bounds[4 * col], bounds[4 * col + 2], rows),
                     ref::Exponent(bounds[4 * col + 1], bounds[4 * col + 3], rows)};
    for (int rank = 0; rank < world; ++rank) {
      auto n = Rows(rank, layout);
      ref::GradientSampling oracle(original[rank], n, t, sampling == 2 ? .7f : 1.f, 79, scales);
      Check(!oracle.invalid_rescale, "Invalid sampling reference fixture");
      integers[rank] = oracle.sampled;
      replay[rank] = original[rank];
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t col = 0; col < t; ++col) {
          auto at = row * t + col;
          if (sampling == 1 && !ref::Selected(79, row, .7f))
            integers[rank][at] = {}, replay[rank][at] = {};
          if (sampling == 2) replay[rank][at] = oracle.Replay(row, original[rank][at]);
        }
    }
  }
  ref::IntegerPair ValueInteger(int rank, std::size_t row, std::size_t col) const {
    auto value = replay[rank][row * targets + col];
    auto scale = scales[col];
    double g = std::ldexp(double(value.g), scale.g), h = std::ldexp(double(value.h), scale.h);
    Check(g >= -0x1p63 && g < 0x1p63 && h >= -0x1p63 && h < 0x1p63, "Invalid value conversion");
    ref::IntegerPair out{static_cast<std::int64_t>(g), static_cast<std::int64_t>(h)};
    if (value.h > 0 && !out.h) out.h = 1;
    return out;
  }
};
void Roots(Context const& ctx, int rank, int world, Reference const& reference,
           metal::QuantizedGradients const& state) {
  auto targets = reference.targets;
  std::vector<GradientPairPrecise> result(targets * 2 + 2, {13, 17});
  std::vector<GradientPairInt64> integers(targets + 2, {19, 23});
  linalg::VectorView<GradientPairPrecise> out{
      {result.data() + 1, targets * 2}, {targets}, {std::size_t{2}}, DeviceOrd::CPU()};
  for (auto device : {DeviceOrd::CPU(), DeviceOrd::MPS()}) {
    auto before = metal::GetCollectiveStatistics();
    state.SumRoot(&ctx, device, out, {integers.data() + 1, targets});
    for (std::size_t t = 0; t < targets; ++t) {
      Sum expected;
      for (auto const& rows : reference.integers)
        for (std::size_t row = 0; row < rows.size() / targets; ++row)
          expected.Add(rows[row * targets + t]);
      Same(integers[t + 1].GetQuantisedGrad(), expected.Integer().GetQuantisedGrad());
      Same(integers[t + 1].GetQuantisedHess(), expected.Integer().GetQuantisedHess());
      SamePair(out(t), expected.Floating(reference.scales[t]));
      SamePair(result[t * 2 + 2], {13, 17});
    }
    Check(integers.front() == GradientPairInt64{19, 23} &&
              integers.back() == GradientPairInt64{19, 23},
          "Root guards changed");
    auto after = metal::GetCollectiveStatistics();
    Check((after.reduce_dispatches > before.reduce_dispatches) == (world > 1 && device.IsMPS()),
          "Wrong root collective placement");
    Check(after.staging_bytes == before.staging_bytes, "Root integers staged");
  }
  // Resident global roots must feed a GPU finalizer before any host inspection.
  HostDeviceVector<GradientPairInt64> resident_exact(targets + 2, {19, 23}, DeviceOrd::MPS());
  HostDeviceVector<GradientPairPrecise> resident_fp(2 * targets + 2, {13, 17}, DeviceOrd::MPS());
  auto ints = resident_exact.DeviceSpan();
  auto fp = resident_fp.DeviceSpan();
  linalg::VectorView<GradientPairPrecise> view{
      fp.subspan(1, 2 * targets), {targets}, {std::size_t{2}}, DeviceOrd::MPS()};
  auto before = metal::GetStatistics();
  auto coll_before = metal::GetCollectiveStatistics();
  state.SumRoot(&ctx, DeviceOrd::MPS(), view, ints.subspan(1, targets));
  // Finalize one target per descriptor directly from the offset GPU span;
  // alternating descriptors read the intentionally initialized guard values.
  std::vector<metal::SplitFinalizeNode> nodes;
  for (std::uint32_t t = 0; t < 2 * targets; ++t) nodes.push_back({0, 0, t, 1});
  std::vector<metal::SplitBounds> bounds(2 * targets, {-INFINITY, INFINITY});
  std::vector<metal::SplitNodeResult> finalized;
  auto workspace = metal::MakeSplitWorkspace();
  metal::FinalizeSplitNodes(nodes, {}, fp.subspan(1, 2 * targets), bounds, {1, 0, 0, 0}, true,
                            &finalized, nullptr, workspace.get());
  auto after = metal::GetStatistics();
  auto coll_after = metal::GetCollectiveStatistics();
  Check(after.gradient_sum_readback_bytes == before.gradient_sum_readback_bytes &&
            after.split_sum_host_input_bytes == before.split_sum_host_input_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes &&
            after.vector_upload_bytes == before.vector_upload_bytes,
        "Resident collective root published through CPU");
  Check((coll_after.reduce_dispatches > coll_before.reduce_dispatches) == (world > 1),
        "Resident root collective placement");
  Check(!resident_exact.HostCanRead() && !resident_fp.HostCanRead(), "Root acquired host mirror");
  auto hi = resident_exact.ConstDeviceSpan();
  auto hf = resident_fp.ConstDeviceSpan();
  for (std::size_t t = 0; t < targets; ++t) {
    Sum expected;
    for (auto const& rows : reference.integers)
      for (std::size_t row = 0; row < rows.size() / targets; ++row)
        expected.Add(rows[row * targets + t]);
    Same(hi[t + 1].GetQuantisedGrad(), expected.Integer().GetQuantisedGrad());
    Same(hi[t + 1].GetQuantisedHess(), expected.Integer().GetQuantisedHess());
    SamePair(hf[1 + 2 * t], expected.Floating(reference.scales[t]));
    Same(finalized[2 * t].parent_hess, expected.Floating(reference.scales[t]).GetHess());
    SamePair(hf[2 + 2 * t], {13, 17});
  }
  Check(hi.front() == GradientPairInt64{19, 23} && hi.back() == GradientPairInt64{19, 23} &&
            hf.front() == GradientPairPrecise{13, 17} && hf.back() == GradientPairPrecise{13, 17},
        "Resident root guards changed");
  state.Sum(out);  // Global roots must not mutate the per-row local state.
  for (std::size_t t = 0; t < targets; ++t) {
    Sum expected;
    for (std::size_t row = 0; row < state.Rows(); ++row)
      expected.Add(reference.integers[rank][row * targets + t]);
    SamePair(out(t), expected.Floating(reference.scales[t]));
  }
  Reject([&] { state.SumRoot(nullptr, DeviceOrd::MPS(), out, {}); });
  Reject([&] { state.SumRoot(&ctx, DeviceOrd::CUDA(0), out, {}); });
}
void Histograms(Context const& ctx, int rank, int world, unsigned layout,
                Reference const& reference,
                std::shared_ptr<metal::QuantizedGradients const> state) {
  constexpr std::size_t bins = 8, root = 8, left = 16, right = 24;
  auto n = Rows(rank, layout), target = reference.targets - 1;
  auto histogram = metal::MakeHistogram();
  histogram->BindQuantized(state, target);
  auto cache = metal::MakeHistogramBuffer();
  cache->Resize(40);
  auto data = cache->Data();
  std::fill(data.begin(), data.end(), GradientPairPrecise{997, 991});
  std::vector<GradientPair> unused(n, {777, 888});
  auto cpu = ctx.MakeCPU();
  for (unsigned page = 0; page < 3; ++page) {
    auto begin = n * page / 3, end = n * (page + 1) / 3;
    SparsePage sparse;
    sparse.base_rowid = begin;
    sparse.offset.HostVector() = {0};
    for (auto row = begin; row < end; ++row) {
      if (row % 11) sparse.data.HostVector().push_back({0, float(row % bins)});
      sparse.offset.HostVector().push_back(sparse.data.Size());
    }
    common::HistogramCuts cuts{1};
    cuts.cut_ptrs_.HostVector() = {0, bins};
    for (std::size_t bin = 0; bin < bins; ++bin)
      cuts.cut_values_.HostVector().push_back(float(bin) + .5f);
    GHistIndexMatrix gidx{&cpu, sparse, {}, std::move(cuts), 16, false, .2};
    for (unsigned child = 0; child < 2; ++child) {
      common::RowSetCollection rows;
      for (auto row = begin; row < end; ++row)
        if (!child || row % 3) rows.Data()->push_back(row);
      rows.Init();
      auto offset = child ? left : root;
      histogram->Build(page, gidx, rows, {0}, unused, {cache->Data().subspan(offset, bins)},
                       cache.get());
    }
  }
  histogram.reset();
  state.reset();  // Cache retains its own scales and integer payload.
  auto before = metal::GetCollectiveStatistics();
  metal::AllreduceHistograms(&ctx, cache.get(), root, 2 * bins);
  auto after = metal::GetCollectiveStatistics();
  Check((after.reduce_dispatches > before.reduce_dispatches) == (world > 1),
        "Missing histogram GPU reduction");
  Check(after.staging_bytes == before.staging_bytes, "Histogram staging");
  std::vector<Sum> expected(3 * bins);
  for (int peer = 0; peer < world; ++peer)
    for (std::size_t row = 0; row < Rows(peer, layout); ++row) {
      if (!(row % 11)) continue;
      auto v = reference.integers[peer][row * reference.targets + target];
      expected[row % bins].Add(v);
      expected[(row % 3 ? bins : 2 * bins) + row % bins].Add(v);
    }
  for (unsigned growth = 0; growth < 2; ++growth) {
    if (growth) cache->Resize(80);
    metal::HistogramSubtraction task{root, left, right, bins};
    metal::SubtractHistograms(cache.get(), {&task, 1});
    for (std::size_t i = 0; i < expected.size(); ++i)
      SamePair(cache->Data()[root + i], expected[i].Floating(reference.scales[target]));
    for (std::size_t i = 0; i < root; ++i) SamePair(cache->Data()[i], {997, 991});
    for (std::size_t i = 32; i < 40; ++i) SamePair(cache->Data()[i], {997, 991});
    Sum total;
    for (std::size_t bin = 0; bin < bins; ++bin) {
      total.g += expected[bin].g;
      total.h += expected[bin].h;
    }
    SamePair(metal::SumHistogram(cache.get(), root, bins),
             total.Floating(reference.scales[target]));
  }
  metal::AllreduceHistograms(&ctx, cache.get(), cache->Data().size(), 0);
  Reject([&] { metal::AllreduceHistograms(&ctx, cache.get(), cache->Data().size(), 1); });
}
void Leaves(Context const& ctx, int rank, int world, unsigned layout, Reference const& reference,
            linalg::MatrixView<GradientPair const> original) {
  auto n = Rows(rank, layout), targets = reference.targets;
  auto quantizer = metal::MakeGradientQuantizer(original, &ctx);
  auto sums = metal::MakeLeafSums(4, targets, quantizer);
  quantizer.reset();
  linalg::Matrix<GradientPair> replay({n, targets}, DeviceOrd::CPU());
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t t = 0; t < targets; ++t) {
      auto v = reference.replay[rank][r * targets + t];
      replay.HostView()(r, t) = {v.g, v.h};
    }
  auto values = std::as_const(replay).View(DeviceOrd::MPS());
  for (unsigned page = 0; page < 3; ++page) {
    auto begin = n * page / 3, end = n * (page + 1) / 3;
    auto rows = metal::MakeDeviceRows();
    rows->Reset(end - begin, begin);
    std::vector<metal::LeafSumSegment> segments;
    for (unsigned leaf = 0; leaf < 3; ++leaf) {
      auto lo = std::max(begin, n * leaf / 3), hi = std::min(end, n * (leaf + 1) / 3);
      if (hi > lo) segments.push_back({lo - begin, hi - begin, leaf, 0});
    }
    rows->AddLeafGradients(segments, values, sums.get());
  }
  auto before = metal::GetCollectiveStatistics();
  sums->AllReduce(&ctx);
  auto after = metal::GetCollectiveStatistics();
  Check((after.reduce_dispatches > before.reduce_dispatches) == (world > 1),
        "Missing leaf GPU reduction");
  Check(after.staging_bytes == before.staging_bytes, "Leaf staging");
  std::vector<Sum> expected(4 * targets);
  for (int peer = 0; peer < world; ++peer)
    for (unsigned leaf = 0; leaf < 3; ++leaf) {
      auto count = Rows(peer, layout);
      for (std::size_t row = count * leaf / 3; row < count * (leaf + 1) / 3; ++row)
        for (std::size_t t = 0; t < targets; ++t)
          expected[leaf * targets + t].Add(reference.ValueInteger(peer, row, t));
    }
  for (std::size_t i = 0; i < expected.size(); ++i)
    SamePair(sums->Data()[i], expected[i].Floating(reference.scales[i % targets]));
  tree::TrainParam p;
  p.reg_lambda = 1.25f;
  p.reg_alpha = .25f;
  p.max_delta_step = .5f;
  std::vector<float> weights(4 * targets + 2, 913);
  sums->Weights({p.reg_lambda, p.reg_alpha, p.max_delta_step, p.min_child_weight}, {}, .3f,
                {weights.data() + 1, 4 * targets});
  for (std::size_t i = 0; i < expected.size(); ++i) {
    float wanted = tree::CalcWeight(p, expected[i].Floating(reference.scales[i % targets])) * .3f;
    Same(weights[i + 1], wanted);
  }
  Same(weights.front(), 913.f);
  Same(weights.back(), 913.f);
  auto empty = metal::MakeLeafSums(0, targets, metal::MakeGradientQuantizer(original, &ctx));
  empty->AllReduce(&ctx);
  auto legacy = metal::MakeLeafSums(1, targets);
  Reject([&] { legacy->AllReduce(&ctx); });
}

bst_node_t Walk(RegTree const& tree, float const* row, unsigned targets) {
  bst_node_t node = 0;
  if (targets == 1) {
    auto view = tree.HostScView();
    while (!view.IsLeaf(node)) {
      auto x = row[view.SplitIndex(node)];
      node = std::isnan(x)              ? view.DefaultChild(node)
             : x < view.SplitCond(node) ? view.LeftChild(node)
                                        : view.RightChild(node);
    }
  } else {
    auto view = tree.HostMtView();
    while (!view.IsLeaf(node)) {
      auto x = row[view.SplitIndex(node)];
      node = std::isnan(x)              ? view.DefaultChild(node)
             : x < view.SplitCond(node) ? view.LeftChild(node)
                                        : view.RightChild(node);
    }
  }
  return node;
}
void SameModel(Context const& ctx, RegTree const& tree) {
  Json model{Object{}};
  tree.SaveModel(&model);
  auto local = Json::Dump(model), root = local;
  std::uint64_t size = local.size();
  collective::SafeColl(collective::Broadcast(&ctx, linalg::MakeVec(&size, 1), 0));
  root.resize(size);
  collective::SafeColl(collective::Broadcast(&ctx, linalg::MakeVec(root.data(), root.size()), 0));
  Check(local == root, "Distributed trees differ across workers");
}
void Updater(int rank, int world, unsigned kind, unsigned layout, unsigned sampling) {
  bool approx = kind == 1, reduced = kind == 3;
  unsigned targets = kind >= 2 ? 3 : 1;
  constexpr std::size_t columns = 3;
  auto rows = Rows(rank, layout);
  current = "updater/world" + std::to_string(world) + "/rank" + std::to_string(rank) + "/kind" +
            std::to_string(kind) + "/layout" + std::to_string(layout) + "/sampling" +
            std::to_string(sampling);
  auto make_row = [](std::size_t r, float* x) {
    x[0] = float(r % 8);
    x[1] = r % 11 == 0 ? NAN : float((r / 8) % 8);
    x[2] = r % 13 == 0 ? NAN : float((r * 3) % 8);
  };
  std::vector<float> data(rows * columns);
  for (std::size_t r = 0; r < rows; ++r) make_row(r, data.data() + r * columns);
  data::DenseAdapter adapter{data.data(), rows, columns};
  std::unique_ptr<DMatrix> matrix{DMatrix::Create(&adapter, NAN, 2)};
  // Separate caches prevent CPU validation from reusing Metal quantile pages.
  std::unique_ptr<DMatrix> cpu_matrix;
  if (!sampling) cpu_matrix.reset(DMatrix::Create(&adapter, NAN, 2));
  Context ctx, cpu;
  ctx.Init({{"device", "mps"}, {"nthread", "2"}});
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  ObjInfo task{ObjInfo::kRegression};
  Args args{{"max_depth", "3"},
            {"max_bin", "16"},
            {"reg_lambda", "1.25"},
            {"reg_alpha", ".25"},
            {"subsample", sampling ? ".7" : "1"},
            {"sampling_method", sampling == 2 ? "gradient_based" : "uniform"}};
  tree::TrainParam param;
  param.Init(args);
  auto name = approx ? "grow_histmaker" : "grow_quantile_histmaker";
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create(name, &ctx, &task)};
  updater->Configure(args);
  std::unique_ptr<TreeUpdater> cpu_updater;
  if (!sampling) {
    cpu_updater.reset(TreeUpdater::Create(name, &cpu, &task));
    cpu_updater->Configure(args);
  }
  std::vector<HostDeviceVector<bst_node_t>> positions(1), cpu_positions(1);
  for (unsigned round = 0; round < 2; ++round) {
    auto fill = [&](GradientContainer* gradients, bool gpu) {
      auto seed = [&](linalg::Matrix<GradientPair>* tensor, unsigned n_targets) {
        tensor->Reshape(rows, n_targets);
        auto out = tensor->HostView();
        for (std::size_t r = 0; r < rows; ++r)
          for (unsigned t = 0; t < n_targets; ++t)
            out(r, t) = {float(r % 8 < 4 ? -2 : 3) * float(t + 1) + float(round) * .25f,
                         1.f + float((r + round) % 3) * .25f};
        if (gpu) tensor->View(DeviceOrd::MPS());
      };
      seed(&gradients->gpair, reduced ? 1 : targets);
      if (reduced) seed(&gradients->value_gpair, targets);
    };
    GradientContainer gradients;
    fill(&gradients, true);
    RegTree tree{targets, columns};
    ctx.Rng().seed(79 + round);
    auto before = metal::GetCollectiveStatistics();
    updater->Update(&param, &gradients, matrix.get(), positions, {&tree});
    auto after = metal::GetCollectiveStatistics();
    Check((after.reduce_dispatches > before.reduce_dispatches) == (world > 1),
          "Updater missed Metal collectives");
    SameModel(ctx, tree);
    Check(positions[0].Device().IsMPS(), "Leaf positions lost device ownership");
    auto indices = positions[0].ConstDeviceSpan();
    Check(indices.size() == rows, "Position count differs");
    for (std::size_t r = 0; r < rows; ++r) {
      auto leaf = Walk(tree, data.data() + r * columns, targets);
      Check((indices[r] < 0 ? ~indices[r] : indices[r]) == leaf, "Leaf assignment differs");
      if (!sampling) Check(indices[r] >= 0, "Unsampled row excluded");
    }
    if (!sampling) {
      GradientContainer cpu_gradients;
      fill(&cpu_gradients, false);
      RegTree expected{targets, columns};
      cpu.Rng().seed(79 + round);
      cpu_updater->Update(&param, &cpu_gradients, cpu_matrix.get(), cpu_positions, {&expected});
      for (std::size_t r = 0; r < 64; ++r) {
        float x[columns];
        make_row(r, x);
        auto actual_leaf = Walk(tree, x, targets), expected_leaf = Walk(expected, x, targets);
        for (unsigned t = 0; t < targets; ++t) {
          float actual = targets == 1 ? tree[actual_leaf].LeafValue()
                                      : tree.HostMtView().LeafValue(actual_leaf)(t);
          float wanted = targets == 1 ? expected[expected_leaf].LeafValue()
                                      : expected.HostMtView().LeafValue(expected_leaf)(t);
          ++cpu_fields;
          if (!(std::abs(actual - wanted) <= 2e-6f * std::max(1.f, std::abs(wanted)))) {
            std::fprintf(stderr, "round%u row%zu target%u actual %a expected %a\n", round, r, t,
                         double(actual), double(wanted));
            Check(false, "CPU updater prediction differs");
          }
        }
      }
    }
  }
}
int main(int argc, char** argv) {
  std::set_terminate([] {
    std::fprintf(stderr, "TERMINATE %s / %s\n", current.c_str(), phase);
    void* frames[32];
    auto count = backtrace(frames, 32);
    backtrace_symbols_fd(frames, count, 2);
    std::abort();
  });
  try {
    Check(argc == 2, "Communicator JSON required");
    Api(XGCommunicatorInit(argv[1]));
    int rank = XGCommunicatorGetRank(), world = XGCommunicatorGetWorldSize();
    Context ctx;
    ctx.Init(Args{{"device", "mps"}, {"nthread", "2"}});
    for (unsigned mode = 0; mode < 4; ++mode) {
      for (unsigned layout = 0; layout < 5; ++layout)
        for (std::size_t targets : {1, 3, 17})
          for (unsigned sampling = 0; sampling < (mode == 0 ? 3u : 2u); ++sampling) {
            current = "world" + std::to_string(world) + "/rank" + std::to_string(rank) + "/mode" +
                      std::to_string(mode) + "/layout" + std::to_string(layout) + "/targets" +
                      std::to_string(targets) + "/sampling" + std::to_string(sampling);
            phase = "reference";
            Reference reference(world, layout, mode, sampling, targets);
            auto n = Rows(rank, layout);
            linalg::Matrix<GradientPair> original({n, targets}, DeviceOrd::CPU());
            for (std::size_t r = 0; r < n; ++r)
              for (std::size_t t = 0; t < targets; ++t) {
                auto v = reference.original[rank][r * targets + t];
                original.HostView()(r, t) = {v.g, v.h};
              }
            auto view = std::as_const(original).View(DeviceOrd::MPS());
            auto before = metal::GetStatistics();
            phase = "quantizer";
            auto state = metal::MakeQuantizedGradients(view, &ctx);
            if (sampling == 1) state->SampleUniform(.7f, 79);
            if (sampling == 2) state->SampleGradient(.7f, 79);
            phase = "roots";
            Roots(ctx, rank, world, reference, *state);
            phase = "histograms";
            Histograms(ctx, rank, world, layout, reference, std::move(state));
            phase = "leaves";
            Leaves(ctx, rank, world, layout, reference, view);
            Check(metal::GetStatistics().vector_readback_bytes == before.vector_readback_bytes,
                  "Resident payload mirrored");
            ++cases;
          }
      std::printf("PASS world%d rank%d mode%d: %zu cases / %zu fields\n", world, rank, mode, cases,
                  fields);
      std::fflush(stdout);
    }
    phase = "production updater";
    std::size_t updater_cases = 0;
    for (unsigned kind = 0; kind < 4; ++kind) {
      for (unsigned layout = 0; layout < 3; ++layout)
        for (unsigned sampling = 0; sampling < 3; ++sampling) {
          Updater(rank, world, kind, layout, sampling);
          ++updater_cases;
        }
      std::printf("PASS world%d rank%d: %zu production updater cases\n", world, rank,
                  updater_cases);
      std::fflush(stdout);
    }
    Check(metal::GetStatistics().vector_live_buffers == 0, "Leaked vectors");
    Api(XGCommunicatorFinalize());
    std::printf("PASS world%d rank%d: %zu integer collective cases / %zu fields / %zu rejected\n",
                world, rank, cases, fields, rejected);
    std::printf("PASS world%d rank%d: %zu updater cases / %zu CPU prediction checks\n", world, rank,
                updater_cases, cpu_fields);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), e.what());
    return 1;
  }
}
