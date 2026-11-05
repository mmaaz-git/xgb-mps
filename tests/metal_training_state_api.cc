// Exercise production updater ownership and guarded sampling/sketch bindings.
// Arithmetic/source equivalence is also covered by the independent CUDA oracles.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_cuda_sketch.h"
#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_sketch.h"
#include "../xgboost/src/common/metal_training.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/tree/param.h"
#include "../xgboost/src/tree/hist/histogram.h"
#include "../xgboost/src/tree/tree_view.h"
#include "xgboost/gradient.h"
#include "xgboost/task.h"
#include "xgboost/tree_updater.h"
#include "cuda_gradient_reference.h"

using namespace xgboost;
namespace metal = common::metal;
namespace xgboost::gbm {
void CopyGradient(Context const*, linalg::Matrix<GradientPair> const*, bst_group_t,
                  linalg::Matrix<GradientPair>*);
}
std::string current;
std::size_t cases = 0, checks = 0;
void Check(bool ok, char const* message) {
  ++checks;
  if (!ok) throw std::runtime_error(current + ": " + message);
}
template <typename T>
void Owned(HostDeviceVector<T> const& values) {
  Check(values.Device().IsMPS() && values.DeviceCanRead() && !values.HostCanRead(),
        "Training payload acquired a host mirror");
}
void NoTransfers(metal::Statistics const& before) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes == before.vector_readback_bytes, "Vector readback");
  Check(after.vector_upload_bytes == before.vector_upload_bytes, "Vector upload");
  Check(after.gradient_state_staging_bytes == before.gradient_state_staging_bytes,
        "Gradient/sketch payload staging");
}
template <typename T>
void Same(common::Span<T const> a, common::Span<T const> b) {
  Check(a.size() == b.size(), "Size differs");
  for (std::size_t i = 0; i < a.size(); ++i)
    Check(std::memcmp(&a[i], &b[i], sizeof(T)) == 0, "Payload bits differ");
}

void GroupCopies(Context const& ctx) {
  current = "GBTree group selection";
  linalg::Matrix<GradientPair> source{{257, 3}, DeviceOrd::MPS()};
  std::vector<GradientPair> values(257 * 3);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = {float(i) - 93.f, float(i % 7)};
  metal::CopyGradients(linalg::MakeTensorView(DeviceOrd::CPU(), common::Span{values}, 257, 3),
                       source.View(DeviceOrd::MPS()));
  linalg::Matrix<GradientPair> result;
  for (int round = 0; round < 3; ++round) {
    for (bst_group_t group = 0; group < 3; ++group) {
      auto before = metal::GetStatistics();
      gbm::CopyGradient(&ctx, &source, group, &result);
      NoTransfers(before);
      Owned(*source.Data());
      Owned(*result.Data());
      auto out = result.Data()->ConstDeviceSpan();
      for (std::size_t r = 0; r < 257; ++r)
        Check(out[r] == values[r * 3 + group], "Group selection differs");
      ++cases;
    }
  }
}

void Replay(std::size_t rows, unsigned layout, bool gradient) {
  current = "sampling replay/" + std::to_string(rows) + "/" + std::to_string(layout) + "/" +
            std::to_string(gradient);
  std::size_t targets = 3, a = layout == 1 ? 1 : layout == 2 ? 7 : 3;
  std::size_t b = layout == 1 ? rows + 3 : layout == 2 ? 2 : 1;
  auto size = rows ? (rows - 1) * a + (targets - 1) * b + 1 : 0;
  std::vector<GradientPair> original(size + 8, {73.f, -19.f});
  linalg::MatrixView<GradientPair> host{
      {original.data() + 4, size}, {rows, targets}, {a, b}, DeviceOrd::CPU()};
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t t = 0; t < targets; ++t)
      host(r, t) = {float(int(r % 23) - 11) * (t + 1) / 8.f, 1.f + float(r % 7) / 4.f};
  HostDeviceVector<GradientPair> storage{original, DeviceOrd::MPS()};
  auto resident = storage.DeviceSpan();
  linalg::MatrixView<GradientPair> view{
      resident.subspan(4, size), {rows, targets}, {a, b}, DeviceOrd::MPS()};
  auto before = metal::GetStatistics();
  auto state = metal::MakeQuantizedGradients(view);
  if (gradient)
    state->SampleGradient(.6f, 31);
  else
    state->SampleUniform(.6f, 31);
  NoTransfers(before);  // Includes the hidden C-to-F quantizer scratch.
  for (int repeat = 0; repeat < 2; ++repeat) {
    // Compare with the same production arithmetic through the staged fallback.
    // The independent source oracles validate the arithmetic separately.
    std::vector<float> fallback((size + 8) * 2 + 1);
    auto* unaligned = reinterpret_cast<GradientPair*>(fallback.data() + 1);
    std::memcpy(unaligned, original.data(), original.size() * sizeof(GradientPair));
    linalg::MatrixView<GradientPair> unshared{
        {unaligned + 4, size}, {rows, targets}, {a, b}, DeviceOrd::CPU()};
    auto staging = metal::GetStatistics();
    state->ApplySampling(unshared);
    Check(metal::GetStatistics().gradient_state_staging_bytes -
                  staging.gradient_state_staging_bytes ==
              2 * size * sizeof(GradientPair),
          "Replay fallback byte count");
    // A GPU write replaces the resident input before every replay.
    storage.Copy(HostDeviceVector<GradientPair>{original, DeviceOrd::MPS()});
    view = {storage.DeviceSpan().subspan(4, size), {rows, targets}, {a, b}, DeviceOrd::MPS()};
    before = metal::GetStatistics();
    state->ApplySampling(view);
    NoTransfers(before);
    Owned(storage);
    Check(metal::GetStatistics().gradient_state_shared_output_bytes -
                  before.gradient_state_shared_output_bytes ==
              size * sizeof(GradientPair),
          "Resident replay not bound directly");
    Same<GradientPair>(storage.ConstDeviceSpan(), {unaligned, original.size()});
  }
  ++cases;
}

void Weights(std::size_t rows, bool grouped, unsigned alias) {
  current = "sketch weights/" + std::to_string(rows) + "/" + std::to_string(grouped) + "/" +
            std::to_string(alias);
  std::vector<float> h(rows + 8, 19.f), w(rows + 8, 31.f), expected(rows + 8, 47.f);
  std::vector<bst_group_t> groups{0, 0, unsigned(rows / 2), unsigned(rows)};
  auto nw = grouped ? 3 : rows;
  for (std::size_t i = 0; i < rows; ++i) h[i + 4] = float(i % 11) / 4.f;
  for (std::size_t i = 0; i < nw; ++i) w[i + 4] = float(i % 7) / 2.f;
  if (alias == 1) expected = h;
  if (alias == 2) expected = w;
  for (std::size_t i = 0; i < rows; ++i)
    expected[i + 4] = h[i + 4] * w[4 + (grouped ? (i < rows / 2 ? 1 : 2) : i)];
  HostDeviceVector<float> hs{h, DeviceOrd::MPS()}, ws{w, DeviceOrd::MPS()};
  HostDeviceVector<float> os{std::vector<float>(rows + 8, 47.f), DeviceOrd::MPS()};
  HostDeviceVector<bst_group_t> gs{groups, DeviceOrd::MPS()};
  auto* destination = alias == 1 ? &hs : alias == 2 ? &ws : &os;
  auto output = destination->DeviceSpan().subspan(4, rows);
  auto before = metal::GetStatistics();
  metal::MakeSketchWeights(
      rows, hs.ConstDeviceSpan().subspan(4, rows), ws.ConstDeviceSpan().subspan(4, nw),
      grouped ? gs.ConstDeviceSpan() : common::Span<bst_group_t const>{}, output);
  NoTransfers(before);
  Owned(*destination);
  Same<float>(destination->ConstDeviceSpan(), expected);
  if (alias != 1) Same<float>(hs.ConstDeviceSpan(), h);
  if (alias != 2) Same<float>(ws.ConstDeviceSpan(), w);
  // The sketch must own a snapshot even if the resident weights are overwritten
  // and their allocation is destroyed before input windows are consumed.
  before = metal::GetStatistics();
  auto sketch = metal::MakeCudaSketch(1, 16, {}, output);
  NoTransfers(before);
  auto replacement = metal::MakeCudaSketch(1, 16, {}, {});
  before = metal::GetStatistics();
  replacement->SetWeights(output);
  NoTransfers(before);
  auto oracle = metal::MakeCudaSketch(1, 16, {}, {expected.data() + 4, rows});
  destination->Fill(-999.f);
  destination->DeviceSpan();
  *destination = HostDeviceVector<float>{};
  std::vector<Entry> entries;
  std::vector<bst_idx_t> ptr{0};
  for (std::size_t i = 0; i < rows; ++i) {
    entries.emplace_back(0, float(i % 19));
    ptr.push_back(entries.size());
  }
  sketch->Push(entries, ptr, 0, 0, rows);
  replacement->Push(entries, ptr, 0, 0, rows);
  oracle->Push(entries, ptr, 0, 0, rows);
  auto result = sketch->Read(), reference = oracle->Read();
  Same<std::uint64_t>(result.ptr, reference.ptr);
  Same<common::WQSummary<>::Entry>(result.entries, reference.entries);
  result = replacement->Read();
  Same<std::uint64_t>(result.ptr, reference.ptr);
  Same<common::WQSummary<>::Entry>(result.entries, reference.entries);
  ++cases;
}

void WeightFallback() {
  current = "read-only weights and unaligned output";
  auto page = std::size_t(getpagesize());
  auto* memory = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  Check(memory != MAP_FAILED, "mmap failed");
  std::vector<float> expected{0.f, -0.f, .25f, 1.f, 8.f};
  std::memcpy(memory, expected.data(), expected.size() * sizeof(float));
  Check(mprotect(memory, page, PROT_READ) == 0, "mprotect failed");
  std::vector<unsigned char> raw(expected.size() * sizeof(float) + 8, 0xa5);
  auto before = metal::GetStatistics();
  metal::MakeSketchWeights(expected.size(), {},
                           {static_cast<float const*>(memory), expected.size()}, {},
                           {reinterpret_cast<float*>(raw.data() + 2), expected.size()});
  auto after = metal::GetStatistics();
  Check(after.gradient_state_shared_input_bytes == before.gradient_state_shared_input_bytes &&
            after.gradient_state_shared_output_bytes == before.gradient_state_shared_output_bytes,
        "Unshareable weight view incorrectly borrowed");
  Check(after.gradient_state_staging_bytes - before.gradient_state_staging_bytes ==
            2 * expected.size() * sizeof(float),
        "Weight fallback byte count");
  Check(!std::memcmp(raw.data() + 2, expected.data(), expected.size() * sizeof(float)),
        "Fallback weight bits differ");
  for (std::size_t i = 0; i < raw.size(); ++i)
    if (i < 2 || i >= 2 + expected.size() * sizeof(float))
      Check(raw[i] == 0xa5, "Guard overwritten");
  Check(!munmap(memory, page), "munmap failed");
  ++cases;
}

void VectorLeafSampling() {
  constexpr std::uint32_t seed = 71;
  for (std::size_t rows : {0u, 1u, 257u})
    for (std::size_t targets : {1u, 3u})
      for (unsigned method = 0; method < 3; ++method)
        for (float fraction : {0.f, .7f, 1.f}) {
          if (!method && fraction != 1.f) continue;
          current = "vector leaf sampling/" + std::to_string(rows) + "/" +
                    std::to_string(targets) + "/" + std::to_string(method) + "/" +
                    std::to_string(fraction);
          std::vector<cuda_reference::Pair> source(rows * targets);
          std::vector<GradientPair> pairs(source.size());
          for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t t = 0; t < targets; ++t) {
              // Every Hessian is zero, with some all-zero gradient rows too.
              float g = r % 3 ? float(int(r % 17) - 8) * float(t + 1) / 8 : 0.f;
              source[r * targets + t] = {g, 0};
              pairs[r * targets + t] = {g, 0};
            }
          cuda_reference::GradientSampling reference(source, rows, targets, fraction, seed);
          HostDeviceVector<GradientPair> gradients(pairs, DeviceOrd::MPS());
          auto view = linalg::MakeTensorView(DeviceOrd::MPS(), gradients.DeviceSpan(), rows, targets);
          auto state = metal::MakeQuantizedGradients(view);
          if (method == 1) state->SampleUniform(fraction, seed);
          if (method == 2) state->SampleGradient(fraction, seed);
          state->ApplySampling(view);
          // Sampling decisions must survive float replay/reuse, exactly as
          // CUDA's retained seed/norm/threshold state does.
          gradients.Fill(GradientPair{});
          HostDeviceVector<bst_node_t> positions(rows + 2, -91, DeviceOrd::MPS());
          auto output = positions.DeviceSpan().subspan(1, rows);
          auto indices = metal::MakeDeviceRows();
          auto before = metal::GetStatistics();
          std::vector<bst_node_t> expected(rows, -91);
          for (unsigned page = 0; page < 3; ++page) {
            auto base = page == 2 ? rows / 2 : 0;
            auto count = page == 0 ? 0 : page == 1 ? rows / 2 : rows - base;
            bst_node_t node = page == 1 ? 3 : 7;
            indices->Reset(count, base);
            std::vector<metal::LeafRange> leaves;
            if (count) leaves.push_back({0, count, node, 0});
            indices->LeafPositions(leaves, view, output, state.get());
            for (std::size_t r = base; r < base + count; ++r) {
              float probability = method == 2 && fraction < 1.f
                                      ? cuda_reference::Probability(reference.norms[r], reference.threshold)
                                      : fraction;
              expected[r] = cuda_reference::Selected(seed, r, probability) ? node : ~node;
            }
          }
          NoTransfers(before);
          auto after = metal::GetStatistics();
          Check(after.leaf_partition_dispatches - before.leaf_partition_dispatches ==
                    (rows / 2 != 0) + (rows != 0), "Leaf finalization dispatch count");
          Check(after.leaf_shared_output_bytes - before.leaf_shared_output_bytes == rows * sizeof(bst_node_t),
                "Leaf positions did not stay in GPU output storage");
          Owned(positions);
          auto actual = positions.ConstHostSpan();
          Check(actual.front() == -91 && actual.back() == -91, "Position guard overwritten");
          for (std::size_t r = 0; r < rows; ++r)
            Check(actual[r + 1] == expected[r], "CUDA sampler membership differs");
          ++cases;
        }
}

void CategoricalUpdater(unsigned targets, bool onehot, bool lossguide) {
  current = "resident category updater/" + std::to_string(targets) + "/" +
            std::to_string(onehot) + "/" + std::to_string(lossguide);
  constexpr std::size_t rows = 257, columns = 2;
  std::vector<float> data(rows * columns);
  for (std::size_t r = 0; r < rows; ++r) {
    data[2 * r] = r % 11 ? float(r % 9) : NAN;
    data[2 * r + 1] = float(r / 9 % 7);
  }
  data::DenseAdapter adapter{data.data(), rows, columns};
  std::unique_ptr<DMatrix> matrix{DMatrix::Create(&adapter, NAN, 2)};
  matrix->Info().feature_types.HostVector() = {FeatureType::kCategorical, FeatureType::kNumerical};
  Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "2"}});
  ObjInfo task{ObjInfo::kRegression};
  tree::TrainParam param;
  Args args{{"max_depth", "4"}, {"max_bin", "17"}, {"max_cached_hist_node", "1"},
            {"max_cat_to_onehot", onehot ? "16" : "2"},
            {"grow_policy", lossguide ? "lossguide" : "depthwise"}};
  param.InitAllowUnknown(args);  // max_cached_hist_node belongs to the updater configuration.
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_quantile_histmaker", &ctx, &task)};
  updater->Configure(args);
  GradientContainer gradients;
  gradients.gpair.Reshape(rows, targets);
  auto pairs = gradients.gpair.HostView();
  for (std::size_t r = 0; r < rows; ++r)
    for (unsigned t = 0; t < targets; ++t)
      pairs(r, t) = {(float(int(r % 9) - 4) * 4.f + data[2 * r + 1]) * (t + 1), 1.f};
  gradients.gpair.View(DeviceOrd::MPS());
  RegTree model(targets, columns);
  std::vector<RegTree*> trees{&model};
  std::vector<HostDeviceVector<bst_node_t>> positions(1);
  auto before = metal::GetStatistics();
  updater->Update(&param, &gradients, matrix.get(), positions, trees);
  auto after = metal::GetStatistics();
  // Ordinary vector SetLeaves reads final base weights once, as CUDA does.
  // Any growing-tree category HostView adds bytes and violates this bound.
  Check(after.vector_readback_bytes - before.vector_readback_bytes ==
            model.NumNodes() * targets * sizeof(float),
        "Category payload was read back during production tree growth");
  Check(after.split_category_readback_bytes == before.split_category_readback_bytes &&
            after.partition_category_host_copy_bytes == before.partition_category_host_copy_bytes,
        "Production categorical payload used host publication/staging");
  Check(after.split_category_store_dispatches > before.split_category_store_dispatches &&
            after.partition_category_shared_input_bytes > before.partition_category_shared_input_bytes,
        "Production categorical storage/partition consumers missing");
  Check(after.gradient_sum_readback_bytes == before.gradient_sum_readback_bytes &&
            after.split_sum_readback_bytes == before.split_sum_readback_bytes &&
            after.split_sum_host_input_bytes == before.split_sum_host_input_bytes,
        "Production vector sums used CPU publication/staging");
  Check(after.split_sum_store_dispatches > before.split_sum_store_dispatches &&
            after.split_sum_gather_dispatches > before.split_sum_gather_dispatches &&
            after.split_sum_apply_dispatches > before.split_sum_apply_dispatches,
        "Production vector sum chain missing");
  Owned(positions[0]);
  auto view = model.HostMtView();  // First intentional CPU category inspection.
  Check(view.HasCategoricalSplit(), "Fixture did not select a category split");
  auto leaves = positions[0].ConstHostSpan();
  for (std::size_t r = 0; r < rows; ++r) {
    bst_node_t node = 0;
    while (!view.IsLeaf(node)) {
      auto value = data[r * columns + view.SplitIndex(node)];
      bool left = std::isnan(value) ? view.DefaultLeft(node)
                  : view.SplitType(node) == FeatureType::kCategorical
                      ? (view.NodeCats(node)[0] & (1u << (31u - unsigned(value)))) == 0
                      : value < view.SplitCond(node);
      node = left ? view.LeftChild(node) : view.RightChild(node);
    }
    Check(leaves[r] == node, "Resident-category partition position differs");
  }
  ++cases;
}

void Updater(bool approx, unsigned targets, bool reduced, unsigned forest, unsigned sampling,
             bool missing) {
  current = "updater/" + std::to_string(approx) + "/" + std::to_string(targets) + "/" +
            std::to_string(reduced) + "/" + std::to_string(forest) + "/" +
            std::to_string(sampling) + "/" + std::to_string(missing);
  constexpr std::size_t rows = 257, columns = 3;
  std::vector<float> data(rows * columns);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t f = 0; f < columns; ++f)
      data[r * columns + f] = missing && (r + f) % 9 == 0 ? NAN : float((r * (f + 1)) % 17);
  data::DenseAdapter adapter{data.data(), rows, columns};
  std::unique_ptr<DMatrix> matrix{DMatrix::Create(&adapter, NAN, 2)};
  Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "2"}});
  ObjInfo task{ObjInfo::kRegression};
  tree::TrainParam param;
  Args args{{"max_depth", "3"},
            {"max_bin", "16"},
            {"subsample", sampling ? ".7" : "1"},
            {"sampling_method", sampling == 2 ? "gradient_based" : "uniform"}};
  param.Init(args);
  std::unique_ptr<TreeUpdater> updater{
      TreeUpdater::Create(approx ? "grow_histmaker" : "grow_quantile_histmaker", &ctx, &task)};
  updater->Configure(args);
  auto split_targets = reduced ? 1u : targets;
  GradientContainer gradients;
  gradients.gpair.Reshape(rows, split_targets);
  if (reduced) gradients.value_gpair.Reshape(rows, targets);
  std::vector<HostDeviceVector<bst_node_t>> positions(forest);
  for (unsigned round = 0; round < 3; ++round) {
    bool gpu = round != 1 || sampling == 2;
    ctx.UpdateAllowUnknown(Args{{"device", gpu ? "mps" : "cpu"}});
    auto seed = [&](linalg::Matrix<GradientPair>* tensor) {
      auto host = tensor->HostView();
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t t = 0; t < host.Shape(1); ++t) {
          host(r, t) = {float(int(r % 17) - 8) * float(t + 1) / 8.f,
                        1.f + float((r + round) % 7) / 4.f};
          // CUDA vector positions retain selected rows even if every Hessian
          // is zero. Include zero and nonzero gradients, also for reduced trees.
          if (gpu && targets > 1 && r % 7 == 0)
            host(r, t) = {r % 2 ? host(r, t).GetGrad() : 0.f, 0.f};
        }
      if (gpu) tensor->View(DeviceOrd::MPS());
    };
    seed(&gradients.gpair);
    if (reduced) seed(&gradients.value_gpair);
    std::vector<std::unique_ptr<RegTree>> owners;
    std::vector<RegTree*> trees;
    for (unsigned t = 0; t < forest; ++t) {
      owners.push_back(std::make_unique<RegTree>(targets, columns));
      trees.push_back(owners.back().get());
    }
    auto sum_before = metal::GetStatistics();
    updater->Update(&param, &gradients, matrix.get(), positions, trees);
    if (gpu && targets > 1) {
      auto sum_after = metal::GetStatistics();
      Check(sum_after.gradient_sum_readback_bytes == sum_before.gradient_sum_readback_bytes &&
                sum_after.split_sum_readback_bytes == sum_before.split_sum_readback_bytes &&
                sum_after.split_sum_host_input_bytes == sum_before.split_sum_host_input_bytes,
            "Vector/reduced/forest sum chain used CPU publication/staging");
      Check(sum_after.split_sum_gather_dispatches > sum_before.split_sum_gather_dispatches,
            "Production node sums were not gathered on GPU");
    }
    if (gpu) {
      Owned(*gradients.gpair.Data());
      if (reduced) Owned(*gradients.value_gpair.Data());
    }
    for (unsigned t = 0; t < forest; ++t) {
      if (gpu) Owned(positions[t]);
      auto output = gpu ? positions[t].ConstDeviceSpan() : positions[t].ConstHostSpan();
      Check(output.size() == rows, "Leaf position count");
      // Independently walk the final public model on the CPU, including missing
      // directions. Negated leaf indices encode excluded rows in both backends.
      for (std::size_t r = 0; r < rows; ++r) {
        bst_node_t nid = 0;
        if (targets == 1) {
          auto& tree = *trees[t];
          while (!tree[nid].IsLeaf()) {
            auto x = data[r * columns + tree[nid].SplitIndex()];
            nid = std::isnan(x)               ? tree[nid].DefaultChild()
                  : x < tree[nid].SplitCond() ? tree[nid].LeftChild()
                                              : tree[nid].RightChild();
          }
        } else {
          auto tree = trees[t]->HostMtView();
          while (!tree.IsLeaf(nid)) {
            auto x = data[r * columns + tree.SplitIndex(nid)];
            nid = std::isnan(x)             ? tree.DefaultChild(nid)
                  : x < tree.SplitCond(nid) ? tree.LeftChild(nid)
                                            : tree.RightChild(nid);
          }
        }
        Check((output[r] < 0 ? ~output[r] : output[r]) == nid, "Leaf assignment differs");
        if (!sampling) Check(output[r] >= 0, "Unsampled row excluded");
      }
    }
    ++cases;
  }
  std::printf("Checked %s (%zu cases / %zu checks)\n", current.c_str(), cases, checks);
  std::fflush(stdout);
}

void HistogramScheduling() {
  current = "vector scalar Hessian scheduling";
  Context ctx;
  ctx.Init({{"device", "cpu"}, {"nthread", "1"}});
  RegTree model(3, 1);
  std::vector<float> weights(3, 0.f);
  model.SetRoot(linalg::MakeVec(weights), 2.f);
  tree::ExpandBatch batch{1.f};
  batch.Push(0, 0, .5f, false, weights, weights, weights, 1.f, 1., 1., {});
  model.Expand(&ctx, batch);
  for (auto hessians : {std::pair{1., 3.}, std::pair{3., 1.}, std::pair{2., 2.}}) {
    std::vector<tree::MultiExpandEntry> host{{0, 0}}, device{{0, 0}};
    host[0].split.left_sum = {{0., hessians.first}, {0., 0.}, {0., 0.}};
    host[0].split.right_sum = {{0., hessians.second}, {0., 0.}, {0., 0.}};
    device[0].child_hessians = hessians;  // Target arrays remain empty.
    std::vector<bst_node_t> host_build(1), host_sub(1), device_build(1), device_sub(1);
    tree::AssignNodes(model.HostMtView(), host, host_build, host_sub);
    tree::AssignNodes(model.HostMtView(), device, device_build, device_sub);
    Check(host_build == device_build && host_sub == device_sub,
          "Resident sums changed smaller-child histogram scheduling");
    Check(device_build[0] == (hessians.second < hessians.first ? model.RightChild(0)
                                                             : model.LeftChild(0)),
          "Wrong smaller-child/tie scheduling");
    ++cases;
  }
}

int main() {
  try {
    Context ctx;
    ctx.Init({{"device", "mps"}, {"nthread", "2"}});
    GroupCopies(ctx);
    VectorLeafSampling();
    WeightFallback();
    HistogramScheduling();
    for (auto rows : {0u, 1u, 257u, 4097u})
      for (unsigned layout = 0; layout < 3; ++layout)
        for (bool gradient : {false, true}) Replay(rows, layout, gradient);
    for (auto rows : {1u, 257u, 4097u})
      for (bool grouped : {false, true})
        for (unsigned alias = 0; alias < 3; ++alias) Weights(rows, grouped, alias);
    for (unsigned targets : {3u, 17u})
      for (bool onehot : {false, true})
        for (bool lossguide : {false, true}) CategoricalUpdater(targets, onehot, lossguide);
    for (bool approx : {false, true})
      for (unsigned targets : {1u, 3u}) {
        if (approx && targets > 1) continue;
        for (bool reduced : {false, true}) {
          if (reduced && targets == 1) continue;
          for (unsigned forest : {1u, 3u})
            for (unsigned sampling = 0; sampling < 3; ++sampling)
              for (bool missing : {false, true})
                Updater(approx, targets, reduced, forest, sampling, missing);
        }
      }
    std::printf("PASS: %zu training state cases / %zu checks\n", cases, checks);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
