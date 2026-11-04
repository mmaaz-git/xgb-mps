// LambdaRank storage/lifecycle tests. Arithmetic uses the independent CUDA-source
// oracle; direct bindings compare identical calls to isolate memory semantics.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "cuda_ranking_reference.h"
#include "xgboost/objective.h"

using namespace xgboost;
namespace metal = common::metal;
std::string current;
std::size_t cases = 0, checks = 0, rejected = 0;
void Check(bool ok, char const* message) {
  ++checks;
  if (!ok) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), message);
    throw std::runtime_error(current + ": " + message);
  }
}
void Near(double actual, double expected) {
  Check(actual == expected || (std::isnan(actual) && std::isnan(expected)) ||
            (std::isfinite(actual) && std::isfinite(expected) &&
             std::abs(actual - expected) <= 2e-6 + 2e-5 * std::abs(expected)),
        "Source reference differs");
}
template <typename Fn>
void Reject(Fn fn) {
  bool caught = false;
  try {
    fn();
  } catch (dmlc::Error const&) {
    caught = true;
  }
  Check(caught, "Invalid initialization accepted");
  ++rejected;
}
template <typename T>
void Resident(HostDeviceVector<T>* values) {
  values->SetDevice(DeviceOrd::MPS());
  values->DevicePointer();
}
void NoTransfers(metal::Statistics const& before, std::size_t imported = 0) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes == before.vector_readback_bytes, "Implicit host mirror");
  Check(after.vector_upload_bytes - before.vector_upload_bytes == imported, "Vector import");
  Check(after.objective_staging_bytes == before.objective_staging_bytes, "Objective staging");
  Check(after.metric_host_input_bytes == before.metric_host_input_bytes &&
            after.metric_host_output_bytes == before.metric_host_output_bytes &&
            after.initialization_host_input_bytes == before.initialization_host_input_bytes,
        "Cache/validator staging");
}
void Compare(linalg::Matrix<GradientPair> const& actual,
             linalg::Matrix<GradientPair> const& expected) {
  auto a = actual.HostView().Values(), e = expected.HostView().Values();
  Check(a.size() == e.size(), "Gradient shape");
  for (std::size_t i = 0; i < a.size(); ++i) {
    Near(a[i].GetGrad(), e[i].GetGrad());
    Near(a[i].GetHess(), e[i].GetHess());
  }
}
void CompareBias(Json const& actual, Json const& expected) {
  for (auto key : {"ti+", "tj-"}) {
    auto const& a = get<F32Array>(actual[key]);
    auto const& e = get<F32Array>(expected[key]);
    Check(a.size() == e.size(), "Bias shape");
    for (std::size_t i = 0; i < a.size(); ++i) Near(a[i], e[i]);
  }
}
void SetBias(cuda_reference::RankingReference* reference, Json const& saved) {
  auto const& positive = get<F32Array>(saved["ti+"]);
  auto const& negative = get<F32Array>(saved["tj-"]);
  reference->SetBias({positive.cbegin(), positive.cend()}, {negative.cbegin(), negative.cend()});
}
Args Parameters(bool mean, bool unbiased) {
  return {{"lambdarank_pair_method", mean ? "mean" : "topk"},
          {"lambdarank_num_pair_per_sample", "7"},
          {"lambdarank_unbiased", unbiased ? "true" : "false"}};
}
void Data(MetaInfo* info, HostDeviceVector<float>* pred, std::size_t rows, unsigned weights) {
  info->num_row_ = rows;
  info->labels.Reshape(rows, 1);
  info->group_ptr_ = rows > 2
                         ? std::vector<bst_group_t>{0, bst_group_t(rows / 3), bst_group_t(rows)}
                         : std::vector<bst_group_t>{0, bst_group_t(rows)};
  pred->Resize(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    info->labels.Data()->HostVector()[i] = float((i + 1) % 2);
    pred->HostVector()[i] = float((i * 7) % 23) / 32;
  }
  if (weights) {
    info->weights_.Resize(info->group_ptr_.size() - 1);
    for (std::size_t i = 0; i < info->weights_.Size(); ++i)
      info->weights_.HostVector()[i] = weights == 2 ? 0.f : 1.f + float(i) / 4;
  }
}
void Inputs(MetaInfo* info, HostDeviceVector<float>* pred) {
  Resident(pred);
  Resident(info->labels.Data());
  Resident(&info->weights_);
}
void StateCase(Context const& gpu, char const* name, bool mean, bool unbiased, std::size_t rows,
               unsigned weights) {
  current = std::string{name} + (mean ? "/mean/" : "/topk/") + std::to_string(unbiased) + "/" +
            std::to_string(rows) + "/" + std::to_string(weights);
  MetaInfo info;
  HostDeviceVector<float> pred;
  Data(&info, &pred, rows, weights);
  auto args = Parameters(mean, unbiased);
  cuda_reference::RankingReference reference{name, args, info};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, &gpu)};
  obj->Configure(args);
  linalg::Matrix<GradientPair> expected, actual;
  auto positions = unbiased ? reference.Positive().size() : 0;
  auto bias_bytes = 2 * positions * sizeof(double);
  auto maximum = rows > 2 ? rows - rows / 3 : rows;
  for (int iteration = 0; iteration < 3; ++iteration) {
    // A GPU write between calls must be visible to the next gradient operation.
    if (iteration == 2) pred.Fill(.375f);
    auto rng = gpu.Rng();
    reference.Gradient(pred, info, mean ? std::uint32_t(rng()) : 0, &expected);
    Inputs(&info, &pred);
    auto before = metal::GetStatistics();
    obj->GetGradient(pred, info, rows || !unbiased ? iteration : 0, &actual);
    NoTransfers(before);
    Check(gpu.Rng() == rng, "Source RNG consumption");
    Check(
        actual.Device().IsMPS() && actual.Data()->DeviceCanWrite() && !actual.Data()->HostCanRead(),
        "Output access state");
    Check(!pred.HostCanRead() && !info.labels.Data()->HostCanRead() && !info.weights_.HostCanRead(),
          "Input access state");
    auto after = metal::GetStatistics();
    auto cache_bytes = std::string{name} == "rank:ndcg"
                           ? (maximum + info.group_ptr_.size() - 1) * sizeof(double)
                           : 0;
    Check(after.objective_shared_input_bytes - before.objective_shared_input_bytes ==
              (rows ? rows * 8 + info.weights_.Size() * 4 + cache_bytes + bias_bytes : 0),
          "Input binding bytes");
    Check(after.objective_shared_output_bytes - before.objective_shared_output_bytes ==
              (rows ? rows * 8 + bias_bytes : 0),
          "Output binding bytes");
    if (iteration && (rows || !unbiased))
      Check(after.vector_allocations == before.vector_allocations, "Repeated allocation");
    Compare(actual, expected);
    if (unbiased) {
      Json saved{Object{}}, oracle{Object{}}, repeated{Object{}};
      reference.SaveBias(&oracle);
      before = metal::GetStatistics();
      obj->SaveConfig(&saved);
      Check(
          metal::GetStatistics().vector_readback_bytes - before.vector_readback_bytes == bias_bytes,
          "Serialization readback bytes");
      CompareBias(saved, oracle);
      before = metal::GetStatistics();
      obj->SaveConfig(&repeated);
      NoTransfers(before);
      CompareBias(repeated, saved);
    }
  }
  if (!rows && unbiased) Reject([&] { obj->GetGradient(pred, info, 1, &actual); });
  ++cases;
}
void Reload(Context const& gpu, char const* name, bool mean) {
  current = std::string{name} + (mean ? "/mean reload" : "/topk reload");
  MetaInfo info;
  HostDeviceVector<float> pred;
  Data(&info, &pred, 33, 1);
  auto args = Parameters(mean, true);
  auto make = [&] {
    auto obj = std::unique_ptr<ObjFunction>{ObjFunction::Create(name, &gpu)};
    obj->Configure(args);
    return obj;
  };
  auto original = make();
  linalg::Matrix<GradientPair> actual, expected;
  original->GetGradient(pred, info, 0, &actual);
  original->GetGradient(pred, info, 1, &actual);
  Json saved{Object{}}, actual_bias{Object{}}, expected_bias{Object{}};
  original->SaveConfig(&saved);
  auto const& positive = get<F32Array>(saved["ti+"]);
  auto const& negative = get<F32Array>(saved["tj-"]);
  Check(std::any_of(positive.cbegin(), positive.cend(), [](float v) { return v != 1.f; }) ||
            std::any_of(negative.cbegin(), negative.cend(), [](float v) { return v != 1.f; }),
        "Reload witness never learned bias");
  auto loaded = make();
  loaded->LoadConfig(saved);
  Reject([&] { loaded->GetGradient(pred, info, 5, &actual); });
  // A fresh source objective has no row-cost state and resets saved bias at iter 0.
  cuda_reference::RankingReference fresh{name, args, info};
  auto rng = gpu.Rng();
  fresh.Gradient(pred, info, mean ? std::uint32_t(rng()) : 0, &expected);
  Inputs(&info, &pred);
  auto before = metal::GetStatistics();
  loaded->GetGradient(pred, info, 0, &actual);
  NoTransfers(before);
  Compare(actual, expected);
  loaded->SaveConfig(&actual_bias);
  fresh.SaveBias(&expected_bias);
  CompareBias(actual_bias, expected_bias);
  // An initialized objective instead retains its row-cost state and imports the
  // F32 serialized bias once. Do not silently preserve its former double values.
  cuda_reference::RankingReference restored{name, args, info};
  SetBias(&restored, saved);
  original->LoadConfig(saved);
  for (int iteration = 2; iteration < 4; ++iteration) {
    rng = gpu.Rng();
    restored.Gradient(pred, info, mean ? std::uint32_t(rng()) : 0, &expected);
    Inputs(&info, &pred);
    before = metal::GetStatistics();
    original->GetGradient(pred, info, iteration, &actual);
    NoTransfers(before, iteration == 2 ? 2 * positive.size() * sizeof(double) : 0);
    Compare(actual, expected);
    original->SaveConfig(&actual_bias);
    restored.SaveBias(&expected_bias);
    CompareBias(actual_bias, expected_bias);
  }
  ++cases;
}
void Handoff(char const* name, bool start_mps, bool reset_mps) {
  current =
      std::string{name} + "/handoff/" + std::to_string(start_mps) + "/" + std::to_string(reset_mps);
  Context ctx, cpu;
  ctx.Init({{"device", start_mps ? "mps" : "cpu"}, {"nthread", "2"}});
  cpu.Init({{"nthread", "2"}});
  MetaInfo info;
  HostDeviceVector<float> pred;
  Data(&info, &pred, 33, 1);
  auto args = Parameters(false, true);
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, &ctx)},
      reference{ObjFunction::Create(name, &cpu)};
  obj->Configure(args);
  reference->Configure(args);
  linalg::Matrix<GradientPair> actual, expected;
  Json saved{Object{}}, oracle{Object{}};
  obj->GetGradient(pred, info, 0, &actual);
  obj->SaveConfig(&saved);
  reference->GetGradient(pred, info, 0, &expected);  // Initialize source row-cost state.
  reference->LoadConfig(saved);
  obj->LoadConfig(saved);  // Both consume the same F32 checkpoint.
  ctx.UpdateAllowUnknown(Args{{"device", start_mps ? "cpu" : "mps"}});
  obj->GetGradient(pred, info, 1, &actual);
  reference->GetGradient(pred, info, 1, &expected);
  Compare(actual, expected);
  obj->SaveConfig(&saved);
  reference->SaveConfig(&oracle);
  CompareBias(saved, oracle);
  // Force source reinitialization with empty saved bias, after both device states
  // have been used. Empty rows must replace, not inherit, the former row-cost state.
  saved["ti+"] = F32Array();
  saved["tj-"] = F32Array();
  obj->LoadConfig(saved);
  ctx.UpdateAllowUnknown(Args{{"device", reset_mps ? "mps" : "cpu"}});
  MetaInfo empty_info;
  HostDeviceVector<float> empty_pred;
  Data(&empty_info, &empty_pred, 0, 0);
  obj->GetGradient(empty_pred, empty_info, 0, &actual);
  Reject([&] { obj->GetGradient(empty_pred, empty_info, 1, &actual); });
  ctx.UpdateAllowUnknown(Args{{"device", reset_mps ? "cpu" : "mps"}});
  Reject([&] { obj->GetGradient(empty_pred, empty_info, 1, &actual); });
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
template <typename T>
auto Buffer(std::vector<T> const& source) {
  std::vector<T> padded(source.size() + 2, T(-91));
  std::copy(source.begin(), source.end(), padded.begin() + 1);
  return HostDeviceVector<T>{padded, DeviceOrd::MPS()};
}
void Bindings(metal::RankingLossKind kind, bool mean, std::size_t rows) {
  std::size_t shape[1]{rows}, stride[1]{2}, positions = mean ? 32 : 7;
  std::vector<bst_group_t> groups{0, bst_group_t(rows - 3), bst_group_t(rows)};
  std::array<std::vector<float>, 3> source{
      std::vector<float>(rows), std::vector<float>(2 * rows, -91), {1.f, 1.25f}};
  for (std::size_t i = 0; i < rows; ++i) {
    source[0][i] = float((i * 7) % 23) / 32;
    source[1][2 * i] = float((i + 1) % 2);
  }
  std::vector<double> discounts(rows - 3), inverse(2), initial(positions, 1.);
  metal::InitNDCG({source[1], shape, stride, DeviceOrd::CPU()}, groups, 7, true, discounts,
                  inverse);
  auto run = [&](std::array<common::Span<float const>, 3> in, common::Span<double const> d,
                 common::Span<double const> inv, common::Span<GradientPair> out,
                 common::Span<double> positive, common::Span<double> negative,
                 metal::RankingGradientTrace trace = {}) {
    metal::RankingGradient(kind, in[0], {in[1], shape, stride, DeviceOrd::MPS()}, groups, in[2], 7,
                           true, true, true, 1., d, inv, out, mean, 19891, {positive, negative, .5},
                           {}, trace);
  };
  std::vector<GradientPair> expected(rows);
  auto positive = initial, negative = initial;
  std::vector<metal::RankingGroupStatistics> statistics(2);
  double rounding;
  run({source[0], source[1], source[2]}, discounts, inverse, expected, positive, negative,
      {statistics, {&rounding, 1}, {}});
  // Each immutable slot can be read-only. Other cases alias inputs with outputs,
  // overlap output destinations or mix shareable and unaligned output ranges.
  for (int mode = -1; mode < 12; ++mode) {
    current = "bindings/" + std::to_string(int(kind)) + "/" + std::to_string(mean) + "/" +
              std::to_string(rows) + "/" + std::to_string(mode);
    std::array<HostDeviceVector<float>, 3> buffers{Buffer(source[0]), Buffer(source[1]),
                                                   Buffer(source[2])};
    std::array<common::Span<float const>, 3> input;
    for (unsigned i = 0; i < 3; ++i)
      input[i] = buffers[i].DeviceSpan().subspan(1, source[i].size());
    auto dbuf = Buffer(discounts), ibuf = Buffer(inverse), pbuf = Buffer(initial),
         nbuf = Buffer(initial);
    common::Span<double const> d = dbuf.DeviceSpan().subspan(1, discounts.size()),
                               inv = ibuf.DeviceSpan().subspan(1, inverse.size());
    auto p = pbuf.DeviceSpan().subspan(1, positions), n = nbuf.DeviceSpan().subspan(1, positions);
    Mapping mapping;
    std::size_t host_input = 0;
    if (mode >= 0 && mode < 3) {
      host_input = source[mode].size() * 4;
      input[mode] = {static_cast<float const*>(mapping.ReadOnly(source[mode].data(), host_input)),
                     source[mode].size()};
    } else if (mode == 3 || mode == 4) {
      auto& in = mode == 3 ? d : inv;
      host_input = in.size_bytes();
      in = {static_cast<double const*>(mapping.ReadOnly(in.data(), host_input)), in.size()};
    }
    // Byte-level expected writes also verify guards and legacy publication order.
    std::vector<std::uint64_t> storage(3 * rows + 2 * positions + 64, 0xababababababababull);
    HostDeviceVector<std::uint64_t> shared{storage, DeviceOrd::MPS()};
    auto bytes = reinterpret_cast<char*>(shared.DevicePointer());
    auto output = reinterpret_cast<GradientPair*>(bytes + (mode == 5 ? 4 : 8));
    char* p_dest = nullptr;
    char* n_dest = nullptr;
    char* diagnostic_dest = nullptr;
    if (mode == 6) {  // Output overlaps predictions; all input reads precede publication.
      std::memcpy(bytes + 8, source[0].data(), source[0].size() * 4);
      input[0] = {reinterpret_cast<float const*>(bytes + 8), rows};
    }
    if (mode >= 7) {
      p_dest = bytes + (mode == 7 ? rows * 8 + 24 : mode == 9 ? 12 : 8);
      n_dest = mode == 7 ? p_dest : mode == 10 ? p_dest + 8 : bytes + rows * 8 + 24;
      p = {reinterpret_cast<double*>(p_dest), positions};
      n = {reinterpret_cast<double*>(n_dest), positions};
      std::memcpy(p_dest, initial.data(), positions * 8);
      std::memcpy(n_dest, initial.data(), positions * 8);
      if (mode == 11) diagnostic_dest = bytes + 16;
    }
    auto expected_bytes = std::vector<char>(bytes, bytes + storage.size() * 8);
    std::memcpy(expected_bytes.data() + (reinterpret_cast<char*>(output) - bytes), expected.data(),
                rows * 8);
    if (diagnostic_dest)
      std::memcpy(expected_bytes.data() + (diagnostic_dest - bytes), &rounding, 8);
    if (p_dest)
      std::memcpy(expected_bytes.data() + (p_dest - bytes), positive.data(), positions * 8);
    if (n_dest)
      std::memcpy(expected_bytes.data() + (n_dest - bytes), negative.data(), positions * 8);
    metal::RankingGradientTrace trace;
    if (diagnostic_dest)
      trace.global_cost_rounding = {reinterpret_cast<double*>(diagnostic_dest), 1};
    auto before = metal::GetStatistics();
    run(input, d, inv, {output, rows}, p, n, trace);
    auto after = metal::GetStatistics();
    auto host_output = mode >= 7 ? rows * 8 + positions * 16 : mode == 5 ? rows * 8 : 0;
    Check(after.objective_staging_bytes - before.objective_staging_bytes ==
              host_input + host_output + (mode == 9 ? positions * 8 : 0),
          "Fallback bytes");
    auto input_bytes = (3 * rows + 2) * 4 + (discounts.size() + inverse.size() + 2 * positions) * 8;
    Check(after.objective_shared_input_bytes - before.objective_shared_input_bytes ==
              input_bytes - host_input - (mode == 9 ? positions * 8 : 0),
          "Shared input bytes");
    Check(after.objective_shared_output_bytes - before.objective_shared_output_bytes ==
              rows * 8 + positions * 16 - host_output,
          "Shared output bytes");
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Binding readback");
    Check(!std::memcmp(bytes, expected_bytes.data(), expected_bytes.size()), "Publication/guards");
    if (!p_dest) Check(!std::memcmp(p.data(), positive.data(), positions * 8), "Positive bias");
    if (!n_dest) Check(!std::memcmp(n.data(), negative.data(), positions * 8), "Negative bias");
    for (unsigned i = 0; i < 3; ++i) {
      auto const& values = buffers[i].ConstHostVector();
      Check(values.front() == -91 && values.back() == -91, "Input guards");
      Check(!std::memcmp(values.data() + 1, source[i].data(), source[i].size() * 4),
            "Input changed");
    }
    for (auto* values : {&dbuf, &ibuf, &pbuf, &nbuf})
      Check(values->ConstHostVector().front() == -91 && values->ConstHostVector().back() == -91,
            "Double guards");
    ++cases;
  }
}
int main() {
  try {
    Context gpu;
    gpu.Init({{"device", "mps"}, {"nthread", "2"}});
    for (auto name : {"rank:pairwise", "rank:ndcg", "rank:map"}) {
      for (bool mean : {false, true}) {
        for (bool unbiased : {false, true})
          for (auto rows : {0, 1, 33, 257})
            for (unsigned weight = 0; weight < 3; ++weight)
              StateCase(gpu, name, mean, unbiased, rows, weight);
        Reload(gpu, name, mean);
      }
      for (bool start : {false, true}) {
        Handoff(name, start, true);
        // CPU MAPStat indexes the first row even for an empty query. That
        // upstream empty-query path cannot be used as a lifecycle oracle.
        if (std::string{name} != "rank:map") Handoff(name, start, false);
      }
      std::printf("Checked %s ownership/reloads/handoffs: %zu cases / %zu checks\n", name, cases,
                  checks);
      std::fflush(stdout);
    }
    for (auto kind : {metal::RankingLossKind::kPairwise, metal::RankingLossKind::kNDCG,
                      metal::RankingLossKind::kMAP})
      for (bool mean : {false, true})
        for (auto rows : {33, 513, 8197}) Bindings(kind, mean, rows);
    Check(metal::GetStatistics().vector_live_buffers == 0 &&
              metal::GetStatistics().vector_capacity_bytes == 0,
          "Leaked vectors");
    std::printf("PASS %zu ranking gradient/bias state cases / %zu checks / %zu rejected calls\n",
                cases, checks, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
