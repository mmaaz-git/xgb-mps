// Predictor/learner ownership. CPU references use the same models; direct native
// binding checks isolate memory behavior from the existing arithmetic oracles.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_output.h"
#include "../xgboost/src/common/metal_predictor.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/data/proxy_dmatrix.h"
#include "../xgboost/src/gbm/gbtree_model.h"
#include "../xgboost/src/tree/tree_view.h"
#include "xgboost/learner.h"
#include "xgboost/predictor.h"

using namespace xgboost;
namespace metal = common::metal;
std::string current;
std::size_t cases = 0, checks = 0, rejected = 0;
void Check(bool ok, char const* message) {
  ++checks;
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Compare(common::Span<float const> actual, common::Span<float const> expected) {
  Check(actual.size() == expected.size(), "Output size");
  for (std::size_t i = 0; i < actual.size(); ++i)
    Check(actual[i] == expected[i] || (std::isnan(actual[i]) && std::isnan(expected[i])) ||
              (std::isfinite(actual[i]) && std::isfinite(expected[i]) &&
               std::abs(actual[i] - expected[i]) <= 2e-6f * (1.f + std::abs(expected[i]))),
          "CPU/reference value differs");
}
template <typename T>
void Resident(HostDeviceVector<T>* values) {
  values->SetDevice(DeviceOrd::MPS());
  values->DevicePointer();
}
void Owned(HostDeviceVector<float> const& values) {
  Check(values.Device().IsMPS() && values.DeviceCanWrite() && !values.HostCanRead(),
        "Prediction is not device-authoritative");
}
template <typename Fn>
void Reject(Fn fn) {
  bool caught = false;
  try {
    fn();
  } catch (dmlc::Error const&) {
    caught = true;
  }
  Check(caught, "Malformed base margin accepted");
  ++rejected;
}
void NoTransfers(metal::Statistics const& before) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes == before.vector_readback_bytes, "Host mirror");
  if (after.vector_upload_bytes != before.vector_upload_bytes) {
    std::fprintf(
        stderr, "Unexpected import: %llu bytes, model packs=%llu\n",
        static_cast<unsigned long long>(after.vector_upload_bytes - before.vector_upload_bytes),
        static_cast<unsigned long long>(after.prediction_model_pack_dispatches -
                                        before.prediction_model_pack_dispatches));
  }
  Check(after.vector_upload_bytes == before.vector_upload_bytes, "Vector import");
  Check(after.prediction_host_input_bytes == before.prediction_host_input_bytes &&
            after.prediction_host_output_bytes == before.prediction_host_output_bytes,
        "Prediction payload staging");
}
auto Data(std::vector<float> const& values, std::size_t rows) {
  data::DenseAdapter adapter(values.data(), rows, 3);
  return std::shared_ptr<DMatrix>{DMatrix::Create(&adapter, NAN, 2)};
}
auto Proxy(std::vector<float> const& values, std::size_t rows, bool native) {
  auto proxy = std::make_shared<data::DMatrixProxy>();
  std::ostringstream descriptor;
  if (native) {
    std::vector<bst_ulong> shape{rows, 3};
    auto source = metal::MakeOutput(values.data(), shape);
    descriptor << "{\"device_type\":8,\"device_id\":0,\"buffer\":"
               << reinterpret_cast<std::uintptr_t>(source->View().data)
               << ",\"byte_offset\":0,\"shape\":[" << rows
               << ",3],\"strides\":[3,1],\"dtype_code\":2,\"dtype_bits\":32,\"dtype_lanes\":1}";
    proxy->SetMetalArray(descriptor.str());  // Owns an independent snapshot.
  } else {
    descriptor << "{\"data\":[" << reinterpret_cast<std::uintptr_t>(values.data())
               << ",false],\"shape\":[" << rows << ",3],\"typestr\":\"<f4\",\"version\":3}";
    proxy->SetArray(descriptor.str());
  }
  return proxy;
}
std::unique_ptr<RegTree> Tree(Context const& cpu, unsigned targets, unsigned feature, float scale) {
  auto tree = std::make_unique<RegTree>(targets, 3);
  if (targets == 1) {
    tree->ExpandNode(0, feature, 0.f, true, 0.f, -.5f * scale, .75f * scale, 1.f, 8.f, 2.f, 6.f);
  } else {
    std::vector<float> base(targets, 0.f), left(targets), right(targets);
    for (unsigned t = 0; t < targets; ++t) {
      left[t] = -.5f * scale * (t + 1);
      right[t] = .75f * scale * (t + 1);
    }
    tree->SetRoot(linalg::MakeVec(DeviceOrd::CPU(), common::Span<float const>{base}), 8.f);
    tree::ExpandBatch batch{1.f};
    batch.Push(0, feature, 0.f, true, base, left, right, 1.f, 2., 6.);
    tree->Expand(&cpu, batch);
    tree->GetMultiTargetTree()->SetLeaves();
  }
  return tree;
}
void StateCase(Context const& gpu, std::size_t rows, unsigned groups, bool vector, bool weighted,
               bool margin) {
  current = "predictor/" + std::to_string(rows) + "/" + std::to_string(groups) + "/" +
            std::to_string(vector) + "/" + std::to_string(weighted) + "/" + std::to_string(margin);
  auto cpu = gpu.MakeCPU();
  std::vector<float> values(std::max<std::size_t>(rows * 3, 1));
  for (std::size_t i = 0; i < rows * 3; ++i) values[i] = i % 7 ? float(int(i % 11) - 5) / 8 : NAN;
  auto data = Data(values, rows);
  std::vector<float> base_values(groups);
  for (unsigned t = 0; t < groups; ++t) base_values[t] = .125f * (t + 1);
  linalg::Vector<float> base{base_values.cbegin(), base_values.cend(), {groups}, DeviceOrd::MPS()};
  std::as_const(base).HostView();  // Both read mirrors are a model-publication contract.
  LearnerModelParam param{
      3, std::move(base), groups, groups,
      vector ? MultiStrategy::kMultiOutputTree : MultiStrategy::kOneOutputPerTree};
  gbm::GBTreeModel model{&param, &cpu};
  for (unsigned round = 0; round < 2; ++round) {
    gbm::TreesOneIter trees(vector ? 1 : groups);
    for (unsigned g = 0; g < trees.size(); ++g)
      trees[g].push_back(Tree(cpu, vector ? groups : 1, round, float((round + 1) * (g + 1))));
    model.CommitModel(std::move(trees));
  }
  if (weighted)
    for (std::size_t i = 0; i < model.trees.size(); ++i)
      model.weight_drop.push_back(i % 2 ? -.25f : .5f);
  if (margin) {
    auto& m = data->Info().base_margin_;
    m.Reshape(rows, groups);
    for (std::size_t i = 0; i < m.Size(); ++i) m.Data()->HostVector()[i] = float(i % 17) / 32;
  }
  std::unique_ptr<Predictor> ref{Predictor::Create("cpu_predictor", &cpu)},
      obj{Predictor::Create("cpu_predictor", &gpu)};
  PredictionCacheEntry expected, actual;
  ref->InitOutPredictions(data->Info(), &expected.predictions, model);
  auto initial = expected.predictions.ConstHostVector();
  ref->PredictBatch(data.get(), &expected, model, 0, model.trees.size());
  Resident(data->Info().base_margin_.Data());
  // First publication creates the model's lazy device mirrors, as in CUDA.
  // These checks isolate prediction/cache storage after that one-time import.
  for (auto const& tree : model.trees) {
    if (tree->IsMultiTarget()) {
      tree::MultiTargetTreeView{DeviceOrd::MPS(), false, tree.get()};
    } else {
      tree::ScalarTreeView{DeviceOrd::MPS(), false, tree.get()};
    }
  }
  model.TreeGroups(DeviceOrd::MPS());
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    auto before = metal::GetStatistics();
    obj->InitOutPredictions(data->Info(), &actual.predictions, model);
    NoTransfers(before);
    Owned(actual.predictions);
    Compare(actual.predictions.ConstHostSpan(), initial);
    before = metal::GetStatistics();
    auto half = model.trees.size() / 2;
    obj->PredictBatch(data.get(), &actual, model, 0, half);
    obj->PredictBatch(data.get(), &actual, model, half, model.trees.size());
    NoTransfers(before);
    Owned(actual.predictions);
    Compare(actual.predictions.ConstHostSpan(), expected.predictions.ConstHostSpan());
    if (repeat)
      Check(metal::GetStatistics().vector_allocations == before.vector_allocations,
            "Repeated prediction allocation");
  }
  HostDeviceVector<float> leaves, expected_leaves;
  ref->PredictLeaf(data.get(), &expected_leaves, model, 0);
  // GBTreeModelView explicitly switches tree_info to CPU for the CPU reference,
  // also in CUDA. Restore that small group array before cache-only assertions.
  model.TreeGroups(DeviceOrd::MPS());
  auto before = metal::GetStatistics();
  obj->PredictLeaf(data.get(), &leaves, model, 0);
  NoTransfers(before);
  Owned(leaves);
  Compare(leaves.ConstHostSpan(), expected_leaves.ConstHostSpan());
  for (bool native : {false, true}) {
    auto proxy = Proxy(values, rows, native);
    if (margin) {
      proxy->Info().base_margin_.Reshape(rows, groups);
      proxy->Info().base_margin_.Data()->Copy(*data->Info().base_margin_.Data());
    }
    before = metal::GetStatistics();
    Check(obj->InplacePredict(proxy, model, NAN, &actual, 0, 0), "Inplace adapter rejected");
    NoTransfers(before);
    Owned(actual.predictions);
    Compare(actual.predictions.ConstHostSpan(), expected.predictions.ConstHostSpan());
  }
  if (rows <= 33) {
    for (bool interactions : {false, true}) {
      HostDeviceVector<float> shap, reference;
      if (interactions)
        ref->PredictInteractionContributions(data.get(), &reference, model, 0, false);
      else
        ref->PredictContribution(data.get(), &reference, model, 0, false, 0, 0);
      Resident(data->Info().base_margin_.Data());
      before = metal::GetStatistics();
      if (interactions)
        obj->PredictInteractionContributions(data.get(), &shap, model, 0, false);
      else
        obj->PredictContribution(data.get(), &shap, model, 0, false, 0, 0);
      NoTransfers(before);
      Owned(shap);
      Compare(shap.ConstHostSpan(), reference.ConstHostSpan());
    }
  }
  if (margin && rows) {
    data->Info().base_margin_.Data()->Fill(.375f);
    before = metal::GetStatistics();
    obj->InitOutPredictions(data->Info(), &actual.predictions, model);
    obj->PredictBatch(data.get(), &actual, model, 0, 0);
    NoTransfers(before);
    Owned(actual.predictions);
    ref->InitOutPredictions(data->Info(), &expected.predictions, model);
    ref->PredictBatch(data.get(), &expected, model, 0, 0);
    Compare(actual.predictions.ConstHostSpan(), expected.predictions.ConstHostSpan());
  }
  // Explicit CPU users can consume and overwrite a previously Metal-owned cache.
  ref->InitOutPredictions(data->Info(), &actual.predictions, model);
  ref->PredictBatch(data.get(), &actual, model, 0, 0);
  Check(actual.predictions.Device().IsCPU(), "CPU handoff inherited Metal placement");
  Compare(actual.predictions.ConstHostSpan(), expected.predictions.ConstHostSpan());
  obj->InitOutPredictions(data->Info(), &actual.predictions, model);
  obj->PredictBatch(data.get(), &actual, model, 0, 0);
  Owned(actual.predictions);
  Compare(actual.predictions.ConstHostSpan(), expected.predictions.ConstHostSpan());
  ++cases;
}
void InitialOnly(Context const& gpu, std::size_t rows, unsigned groups, bool scalar) {
  current = "initial/" + std::to_string(rows) + "/" + std::to_string(groups) + "/" +
            std::to_string(scalar);
  auto cpu = gpu.MakeCPU();
  linalg::Vector<float> base{{scalar ? 1 : groups}, DeviceOrd::MPS()};
  base.Data()->Fill(.25f);
  std::as_const(base).HostView();
  LearnerModelParam param{3, std::move(base), groups, groups, MultiStrategy::kOneOutputPerTree};
  gbm::GBTreeModel model{&param, &cpu};
  std::vector<float> values(std::max<std::size_t>(rows * 3, 1), 1.f);
  auto data = Data(values, rows);
  std::unique_ptr<Predictor> obj{Predictor::Create("cpu_predictor", &gpu)};
  PredictionCacheEntry output;
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    auto before = metal::GetStatistics();
    obj->InitOutPredictions(data->Info(), &output.predictions, model);
    obj->PredictBatch(data.get(), &output, model, 0, 0);
    NoTransfers(before);
    Owned(output.predictions);
    Compare(output.predictions.ConstHostSpan(), std::vector<float>(rows * groups, .25f));
  }
  HostDeviceVector<float> leaves(17, -91.f);
  obj->PredictLeaf(data.get(), &leaves, model, 0);
  Owned(leaves);
  Check(leaves.Empty(), "Zero-tree leaf shape");
  if (rows) {
    data->Info().base_margin_.Reshape(rows, groups + 1);
    Resident(data->Info().base_margin_.Data());
    auto before = metal::GetStatistics();
    Reject([&] { obj->InitOutPredictions(data->Info(), &output.predictions, model); });
    NoTransfers(before);
  }
  ++cases;
}
void LeafCase(Context const& gpu, std::size_t rows, unsigned targets, std::size_t stride) {
  current = "leaf cache/" + std::to_string(rows) + "/" + std::to_string(targets) + "/" +
            std::to_string(stride);
  auto cpu = gpu.MakeCPU();
  auto tree = Tree(cpu, targets, 0, 1.);
  std::vector<RegTree const*> trees{tree.get()};
  std::vector<HostDeviceVector<bst_node_t>> ids(1);
  ids[0].Resize(rows);
  for (std::size_t i = 0; i < rows; ++i)
    ids[0].HostVector()[i] = i % 3 == 0 ? ~1 : i % 3 == 1 ? 1 : 2;
  std::vector<float> original(rows * stride + 2, -91.f);
  for (std::size_t i = 0; i < rows; ++i)
    for (unsigned t = 0; t < targets; ++t) original[1 + i * stride + t] = .125f;
  auto expected = original;
  HostDeviceVector<float> actual{original, DeviceOrd::MPS()};
  std::size_t shape[2]{rows, targets}, strides[2]{stride, 1};
  std::unique_ptr<Predictor> ref{Predictor::Create("cpu_predictor", &cpu)},
      obj{Predictor::Create("cpu_predictor", &gpu)};
  ref->PredictFromLeafIds(ids, trees,
                          {{expected.data() + 1, rows * stride}, shape, strides, DeviceOrd::CPU()});
  Resident(&ids[0]);
  // CUDA SetLeaves publishes host leaf storage too. Publish that model once;
  // the prediction consumer must then use its compact GPU weights and mapping.
  if (targets > 1 && rows) (void)tree::MultiTargetTreeView{DeviceOrd::MPS(), false, tree.get()};
  auto before = metal::GetStatistics();
  obj->PredictFromLeafIds(
      ids, trees,
      {actual.DeviceSpan().subspan(1, rows * stride), shape, strides, DeviceOrd::MPS()});
  NoTransfers(before);
  if (targets > 1 && rows) {
    auto after = metal::GetStatistics();
    Check(after.prediction_shared_input_bytes - before.prediction_shared_input_bytes ==
              (rows + 2 * targets + 3) * sizeof(float),
          "Leaf cache did not bind compact GPU weights and node mapping");
  }
  Owned(actual);
  Check(!ids[0].HostCanRead(), "Leaf IDs read on host");
  Compare(actual.ConstHostSpan(), expected);
  if (!rows) {
    ++shape[1];
    Reject([&] {
      ref->PredictFromLeafIds(ids, trees,
                              {{expected.data(), std::size_t{0}}, shape, strides, DeviceOrd::CPU()});
    });
    Reject([&] {
      obj->PredictFromLeafIds(ids, trees,
                              {actual.DeviceSpan().subspan(1, 0), shape, strides, DeviceOrd::MPS()});
    });
  }
  ++cases;
}
void LearnerCase(Context const& gpu, char const* objective, bool vector) {
  current = std::string{"learner/"} + objective + "/" + std::to_string(vector);
  std::vector<float> values(257 * 3);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 19) - 9) / 8;
  auto data = Data(values, 257);
  unsigned targets = vector ? 3 : 1;
  data->Info().labels.Reshape(257, targets);
  for (std::size_t i = 0; i < data->Info().labels.Size(); ++i)
    data->Info().labels.Data()->HostVector()[i] =
        std::string{objective} == "binary:logistic" ? float(i % 2) : values[i % values.size()];
  std::unique_ptr<Learner> cpu{Learner::Create({data})}, actual{Learner::Create({data})};
  cpu->SetParams({{"device", "cpu"},
                  {"nthread", "2"},
                  {"objective", objective},
                  {"max_depth", "2"},
                  {"multi_strategy", vector ? "multi_output_tree" : "one_output_per_tree"}});
  for (int iteration = 0; iteration < 3; ++iteration) cpu->UpdateOneIter(iteration, data);
  Json model{Object{}};
  cpu->SaveModel(&model);
  actual->LoadModel(model);
  actual->SetParams({{"device", "mps"}, {"nthread", "2"}});
  actual->Configure();
  HostDeviceVector<float> out, expected;
  // Publish all model mirrors before measuring incremental cache operations.
  // The model publication test separately verifies the first import.
  actual->Predict(data, true, &out, 0, 0, false, false, false, false, false, false);
  // Prefix increment, repeat, backwards reset, nonzero start, full-model reset.
  std::vector<std::pair<int, int>> ranges{{0, 1}, {0, 3}, {0, 3}, {0, 1}, {1, 3}, {0, 0}};
  for (bool margin : {false, true}) {
    for (auto range : ranges) {
      cpu->Predict(data, margin, &expected, range.first, range.second, false, false, false, false,
                   false, false);
      auto before = metal::GetStatistics();
      actual->Predict(data, margin, &out, range.first, range.second, false, false, false, false,
                      false, false);
      NoTransfers(before);
      Owned(out);
      Compare(out.ConstHostSpan(), expected.ConstHostSpan());
    }
  }
  // Complete cached raw output has no objective transform to repair placement.
  for (char const* device : {"cpu", "mps", "cpu", "mps"}) {
    actual->SetParam("device", device);
    actual->Configure();
    actual->Predict(data, true, &out, 0, 0, false, false, false, false, false, false);
    if (std::string{device} == "mps")
      Owned(out);
    else
      Check(out.Device().IsCPU(), "CPU learner inherited Metal placement");
    Compare(out.ConstHostSpan(), expected.ConstHostSpan());
  }
  // Evaluation copies the cache before transforming it. A second call must not
  // read back either predictions or labels left on GPU by the first call.
  actual->EvalOneIter(0, {data}, {"test"});
  Resident(data->Info().labels.Data());
  auto before = metal::GetStatistics();
  actual->EvalOneIter(1, {data}, {"test"});
  NoTransfers(before);
  ++cases;
}
struct Mapping {
  void* data{nullptr};
  std::size_t bytes{0};
  void const* ReadOnly(void const* src, std::size_t size) {
    auto page = std::size_t(getpagesize());
    bytes = (size + 8 + page - 1) / page * page;
    data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    Check(data != MAP_FAILED, "mmap");
    auto result = static_cast<char*>(data) + 8;
    std::memcpy(result, src, size);
    Check(mprotect(data, bytes, PROT_READ) == 0, "mprotect");
    return result;
  }
  ~Mapping() {
    if (data && data != MAP_FAILED) munmap(data, bytes);
  }
};
void LeafBindings(unsigned targets, bool padded, bool mapped) {
  constexpr std::size_t rows = 33;
  auto stride = targets + (padded ? 2 : 0);
  auto count = rows * stride;
  std::vector<std::int32_t> ids(rows);
  std::vector<float> weights((mapped ? 2 : 3) * targets), initial(count, -.125f);
  std::vector<std::int32_t> leaf_mapping{0, 1, 0};
  for (std::size_t i = 0; i < rows; ++i) ids[i] = i % 3 == 0 ? ~1 : i % 3 == 1 ? 1 : 2;
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(i + 1) / 8;
  for (unsigned mode = 0; mode < (mapped ? 8u : 6u); ++mode) {
    current = "leaf bindings/" + std::to_string(targets) + "/" + std::to_string(padded) + "/" +
              std::to_string(mapped) + "/" + std::to_string(mode);
    std::vector<std::int32_t> padded_ids(rows + 2, -91);
    std::copy(ids.begin(), ids.end(), padded_ids.begin() + 1);
    HostDeviceVector<std::int32_t> ibuf{padded_ids, DeviceOrd::MPS()};
    std::vector<float> padded_weights(weights.size() + 2, -91.f);
    std::copy(weights.begin(), weights.end(), padded_weights.begin() + 1);
    HostDeviceVector<float> wbuf{padded_weights, DeviceOrd::MPS()};
    HostDeviceVector<std::int32_t> mbuf{leaf_mapping, DeviceOrd::MPS()};
    std::vector<std::uint32_t> storage(count + 8, 0xabababab);
    HostDeviceVector<std::uint32_t> output{storage, DeviceOrd::MPS()};
    auto bytes = reinterpret_cast<char*>(output.DevicePointer());
    auto out = reinterpret_cast<float*>(bytes + (mode == 5 ? 9 : 8));
    std::memcpy(out, initial.data(), count * 4);
    common::Span<std::int32_t const> in = ibuf.DeviceSpan().subspan(1, rows);
    common::Span<float const> leaf = wbuf.DeviceSpan().subspan(1, weights.size());
    common::Span<std::int32_t const> node_mapping = mapped ? mbuf.ConstDeviceSpan()
                                                         : common::Span<std::int32_t const>{};
    Mapping mapping;
    if (mode == 1)
      in = {static_cast<std::int32_t const*>(mapping.ReadOnly(ids.data(), rows * 4)), rows};
    if (mode == 2)
      leaf = {static_cast<float const*>(mapping.ReadOnly(weights.data(), weights.size() * 4)),
              weights.size()};
    if (mode == 3) {
      std::memcpy(out, ids.data(), rows * 4);
      in = {reinterpret_cast<std::int32_t const*>(out), rows};
    }
    if (mode == 4) {
      std::memcpy(out, weights.data(), weights.size() * 4);
      leaf = {out, weights.size()};
    }
    if (mode == 6)
      node_mapping = {static_cast<std::int32_t const*>(
                          mapping.ReadOnly(leaf_mapping.data(), leaf_mapping.size() * 4)),
                      leaf_mapping.size()};
    if (mode == 7) {
      std::memcpy(out, leaf_mapping.data(), leaf_mapping.size() * 4);
      node_mapping = {reinterpret_cast<std::int32_t const*>(out), leaf_mapping.size()};
    }
    std::vector<float> expected(count);
    std::memcpy(expected.data(), out, count * 4);
    for (std::size_t row = 0; row < rows; ++row) {
      auto node = ids[row] >= 0 ? ids[row] : ~ids[row];
      auto index = mapped ? leaf_mapping[node] : node;
      for (unsigned t = 0; t < targets; ++t)
        expected[row * stride + t] += weights[index * targets + t];
    }
    auto guard = std::vector<char>(bytes, bytes + storage.size() * 4);
    std::memcpy(guard.data() + (reinterpret_cast<char*>(out) - bytes), expected.data(), count * 4);
    auto before = metal::GetStatistics();
    metal::AddLeafValues(in, leaf, targets, stride, {out, count}, node_mapping);
    auto after = metal::GetStatistics();
    Check(after.prediction_host_input_bytes - before.prediction_host_input_bytes ==
              (mode == 1   ? rows * 4
               : mode == 2 ? weights.size() * 4
               : mode == 5 ? count * 4
               : mode == 6 ? leaf_mapping.size() * 4
                           : 0),
          "Leaf input fallback bytes");
    Check(after.prediction_host_output_bytes - before.prediction_host_output_bytes ==
              (mode == 5 ? count * 4 : 0),
          "Leaf output fallback bytes");
    Check(after.prediction_device_copy_bytes - before.prediction_device_copy_bytes ==
              (mode == 3   ? rows * 4
               : mode == 4 ? weights.size() * 4
               : mode == 7 ? leaf_mapping.size() * 4
                           : 0),
          "Leaf alias snapshot bytes");
    // Aliased encoded IDs can be NaN bit patterns when interpreted as the
    // initial prediction. Arithmetic preserves NaN classification, not payload.
    // Finite results, padding and guards still require exact bits.
    for (std::size_t row = 0; row < rows; ++row)
      for (unsigned t = 0; t < targets; ++t) {
        auto index = row * stride + t;
        if (!std::isnan(expected[index])) continue;
        float value;
        auto address = reinterpret_cast<char*>(out) + index * sizeof(float);
        std::memcpy(&value, address, sizeof(value));
        Check(std::isnan(value), "Leaf arithmetic lost NaN");
        std::memcpy(guard.data() + (address - bytes), address, sizeof(value));
      }
    Check(!std::memcmp(bytes, guard.data(), guard.size()), "Leaf publication/padding/guards");
    Check(!std::memcmp(ibuf.ConstHostSpan().data(), padded_ids.data(), padded_ids.size() * 4),
          "Leaf IDs changed");
    Check(
        !std::memcmp(wbuf.ConstHostSpan().data(), padded_weights.data(), padded_weights.size() * 4),
        "Leaf weights changed");
    Check(mbuf.ConstHostVector() == leaf_mapping, "Leaf mapping changed");
    ++cases;
  }
}
void Bindings(bool leaf, std::size_t rows) {
  current = "raw traversal/" + std::to_string(leaf) + "/" + std::to_string(rows);
  metal::PredictionModel model;
  model.nodes.resize(3);
  model.nodes[0].left = model.nodes[0].missing = 1;
  model.nodes[0].right = 2;
  model.nodes[1].leaf_begin = 0;
  model.nodes[2].leaf_begin = 1;
  model.leaves = {-.5f, .75f};
  model.trees = {{0, 1, 0, .5f}, {0, 1, 1, -.25f}};
  auto obj = metal::MakePrediction(model);
  std::vector<float> data(rows), original(rows * 2, .125f);
  for (std::size_t i = 0; i < rows; ++i) data[i] = i % 3 == 0 ? NAN : i % 3 == 1 ? -1.f : 1.f;
  auto expected = original;
  obj->Predict(data, rows, 1, 2, expected, leaf);
  for (unsigned mode = 0; mode < 4; ++mode) {
    std::vector<float> padded(data.size() + 2, -91.f);
    std::copy(data.begin(), data.end(), padded.begin() + 1);
    HostDeviceVector<float> input{padded, DeviceOrd::MPS()};
    std::vector<std::uint32_t> storage(original.size() + 8, 0xabababab);
    HostDeviceVector<std::uint32_t> owned{storage, DeviceOrd::MPS()};
    auto bytes = reinterpret_cast<char*>(owned.DevicePointer());
    auto output = reinterpret_cast<float*>(bytes + (mode == 3 ? 9 : 8));
    std::memcpy(output, original.data(), original.size() * 4);
    common::Span<float const> in = input.DeviceSpan().subspan(1, rows);
    Mapping mapping;
    if (mode == 1)
      in = {static_cast<float const*>(mapping.ReadOnly(data.data(), data.size() * 4)), rows};
    auto expected_out = expected;
    if (mode == 2) {
      std::memcpy(output, data.data(), data.size() * 4);
      in = {output, rows};
      // Initial predictions are also aliased, so reproduce those bits before
      // computing the disjoint baseline. Both calls read the original snapshot.
      expected_out = original;
      std::copy(data.begin(), data.end(), expected_out.begin());
      obj->Predict(data, rows, 1, 2, expected_out, leaf);
    }
    auto guard = std::vector<char>(bytes, bytes + storage.size() * 4);
    std::memcpy(guard.data() + (reinterpret_cast<char*>(output) - bytes), expected_out.data(),
                expected_out.size() * 4);
    auto before = metal::GetStatistics();
    obj->Predict(in, rows, 1, 2, {output, original.size()}, leaf);
    auto after = metal::GetStatistics();
    Check(after.prediction_host_input_bytes - before.prediction_host_input_bytes ==
              (mode == 1   ? data.size() * 4
               : mode == 3 ? original.size() * 4
                           : 0),
          "Input fallback bytes");
    Check(after.prediction_host_output_bytes - before.prediction_host_output_bytes ==
              (mode == 3 ? original.size() * 4 : 0),
          "Output fallback bytes");
    Check(after.prediction_device_copy_bytes - before.prediction_device_copy_bytes ==
              (mode == 2 ? data.size() * 4 : 0),
          "Alias snapshot bytes");
    Check(!std::memcmp(bytes, guard.data(), guard.size()), "Traversal output bits/guards");
    ++cases;
  }
}
int main() {
  try {
    Context gpu;
    gpu.Init({{"device", "mps"}, {"nthread", "2"}});
    for (auto rows : {0, 1, 33, 65537}) {
      for (unsigned groups : {1, 3})
        for (bool vector : {false, true}) {
          if (vector && groups == 1) continue;
          for (bool weighted : {false, true})
            for (bool margin : {false, true})
              StateCase(gpu, rows, groups, vector, weighted, margin);
        }
      std::printf("Checked predictor rows=%d: %zu cases / %zu checks\n", rows, cases, checks);
      std::fflush(stdout);
    }
    for (auto rows : {0, 1, 33, 65537})
      for (unsigned targets : {1, 2, 3, 17})
        for (bool padded : {false, true}) LeafCase(gpu, rows, targets, targets + (padded ? 2 : 0));
    for (auto rows : {0, 33})
      for (unsigned groups : {1, 3})
        for (bool scalar : {false, true}) InitialOnly(gpu, rows, groups, scalar);
    for (unsigned targets : {1, 3, 17})
      for (bool padded : {false, true})
        for (bool mapped : {false, true}) LeafBindings(targets, padded, mapped);
    LearnerCase(gpu, "reg:squarederror", false);
    LearnerCase(gpu, "reg:squarederror", true);
    LearnerCase(gpu, "binary:logistic", false);
    for (bool leaf : {false, true})
      for (auto rows : {1, 33, 65537}) Bindings(leaf, rows);
    Check(metal::GetStatistics().vector_live_buffers == 0 &&
              metal::GetStatistics().vector_capacity_bytes == 0,
          "Leaked vectors");
    std::printf("PASS %zu prediction state cases / %zu checks / %zu rejected calls\n", cases,
                checks, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
