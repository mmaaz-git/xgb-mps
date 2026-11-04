// Ranking metric/cache ownership. Existing source oracles validate arithmetic.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/ranking_utils.h"
#include "../xgboost/src/data/adapter.h"
#include "xgboost/metric.h"

using namespace xgboost;
namespace metal = common::metal;
std::string current;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* message) {
  ++fields;
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Near(double a, double b) {
  Check(a == b || (std::isnan(a) && std::isnan(b)) ||
            (std::isfinite(a) && std::isfinite(b) &&
             std::abs(a - b) <= 3e-12 * std::max(1., std::abs(b))),
        "CPU reference differs");
}
template <typename Fn>
void Reject(Fn fn) {
  bool caught = false;
  try {
    fn();
  } catch (dmlc::Error const&) {
    caught = true;
  }
  Check(caught, "Invalid call accepted");
  ++rejected;
}
void Resident(HostDeviceVector<float>* values) {
  values->SetDevice(DeviceOrd::MPS());
  values->DevicePointer();
}
void NoTransfers(metal::Statistics const& before) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes == before.vector_readback_bytes, "Host mirror");
  Check(after.vector_upload_bytes == before.vector_upload_bytes, "Vector upload");
  Check(after.metric_host_input_bytes == before.metric_host_input_bytes, "Metric input staging");
  Check(after.metric_host_output_bytes == before.metric_host_output_bytes, "Metric output staging");
  Check(after.initialization_host_input_bytes == before.initialization_host_input_bytes,
        "Cache weight staging");
}
auto Matrix(std::size_t rows) {
  std::vector<float> data(std::max<std::size_t>(rows, 1), 1.f);
  data::DenseAdapter adapter(data.data(), rows, 1);
  return std::shared_ptr<DMatrix>{DMatrix::Create(&adapter, NAN, 2)};
}
void MetricCase(Context const& gpu, char const* name, std::size_t rows, unsigned weights) {
  current = std::string{name} + "/" + std::to_string(rows) + "/" + std::to_string(weights);
  auto data = Matrix(rows);
  auto& info = data->Info();
  info.labels.Reshape(rows, 1);
  bool ndcg = std::string{name}.find("ndcg") == 0;
  if (rows > 2) info.group_ptr_ = {0, bst_group_t(rows / 3), bst_group_t(rows)};
  std::size_t groups = info.group_ptr_.empty() ? 1 : info.group_ptr_.size() - 1;
  std::size_t maximum = rows > 2 ? rows - rows / 3 : rows;
  for (std::size_t i = 0; i < rows; ++i) info.labels.Data()->HostVector()[i] = float((i + 1) % 2);
  if (weights) {
    info.weights_.Resize(groups);
    for (std::size_t i = 0; i < groups; ++i)
      info.weights_.HostVector()[i] = weights == 2 ? 0.f : 1.f + float(i) / 4;
  }
  HostDeviceVector<float> pred(rows);
  for (std::size_t i = 0; i < rows; ++i) pred.HostVector()[i] = float((i * 7) % 23) / 32;
  auto cpu = gpu.MakeCPU();
  std::unique_ptr<Metric> ref{Metric::Create(name, &cpu)}, actual{Metric::Create(name, &gpu)};
  ref->Configure({});
  actual->Configure({});
  // CUDA's empty Precision contributes zero; NDCG/MAP keep their source fallback
  // (one unless '-', including zero weights). CPU Precision divides by zero.
  double expected = rows ? ref->Evaluate(pred, data)
                    : std::string{name}.find("pre") == 0 || std::string{name}.back() == '-' ? 0.
                                                                                            : 1.;
  auto resident = [&] {
    Resident(&pred);
    Resident(info.labels.Data());
    Resident(&info.weights_);
  };
  resident();
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    auto before = metal::GetStatistics();
    Near(actual->Evaluate(pred, data), expected);
    NoTransfers(before);
    auto after = metal::GetStatistics();
    Check(after.metric_final_readback_bytes - before.metric_final_readback_bytes == 16,
          "Final scalar bytes");
    Check(after.ranking_plan_dispatches - before.ranking_plan_dispatches == !repeat,
          "Cache plan reuse");
    Check(after.ranking_cache_dispatches - before.ranking_cache_dispatches == (ndcg && !repeat),
          "NDCG cache reuse");
    Check(after.metric_shared_output_bytes - before.metric_shared_output_bytes ==
              (ndcg && !repeat ? 8 * (maximum + groups) : 0),
          "Cache publication bytes");
    Check(!pred.HostCanRead() && !info.labels.Data()->HostCanRead() && !info.weights_.HostCanRead(),
          "Input access state");
    if (repeat) Check(after.vector_allocations == before.vector_allocations, "Repeated allocation");
  }
  if (rows) {
    pred.Fill(.375f);  // GPU writes, then CPU/MPS handoff through the same metadata.
    expected = ref->Evaluate(pred, data);
    resident();
    auto before = metal::GetStatistics();
    Near(actual->Evaluate(pred, data), expected);
    NoTransfers(before);
  }
  if (ndcg) {
    // This parameter resets the ideal-gain cache; do not reuse stale discounts/results.
    ref->Configure({{"ndcg_exp_gain", "false"}});
    actual->Configure({{"ndcg_exp_gain", "false"}});
    if (rows) expected = ref->Evaluate(pred, data);
    resident();
    auto before = metal::GetStatistics();
    Near(actual->Evaluate(pred, data), expected);
    NoTransfers(before);
    Check(metal::GetStatistics().ranking_cache_dispatches == before.ranking_cache_dispatches + 1,
          "Parameter reset did not rebuild cache");
  }
  ++cases;
}
void CacheCase(Context const& gpu, std::size_t rows) {
  current = "cache/" + std::to_string(rows);
  MetaInfo info;
  info.num_row_ = rows;
  info.labels.Reshape(rows, 1);
  info.group_ptr_ = {0, bst_group_t(rows)};
  info.weights_.HostVector() = {1.25f};
  HostDeviceVector<float> pred(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    info.labels.Data()->HostVector()[i] = float(i % 4);
    pred.HostVector()[i] = float((i * 7) % 23);
  }
  ltr::LambdaRankParam param;
  param.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "7"}});
  auto cpu = gpu.MakeCPU();
  ltr::NDCGCache ref{&cpu, info, param};
  auto rank = ref.SortedIdx(&cpu, pred.ConstHostSpan());
  std::vector<std::size_t> expected(rank.begin(), rank.end());
  Resident(&pred);
  Resident(info.labels.Data());
  Resident(&info.weights_);
  auto before = metal::GetStatistics();
  ltr::NDCGCache cache{&gpu, info, param};
  auto inverse = cache.InvIDCG(&gpu);
  auto discount = cache.Discount(&gpu);
  Check(inverse.Device().IsMPS(), "Inverse cache device");
  NoTransfers(before);
  Near(inverse(0), ref.InvIDCG(&cpu)(0));
  for (std::size_t i = 0; i < discount.size(); ++i) Near(discount[i], ref.Discount(&cpu)[i]);
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    before = metal::GetStatistics();
    auto sorted = cache.SortedIdx(&gpu, pred.ConstDeviceSpan());
    NoTransfers(before);
    for (std::size_t i = 0; i < sorted.size(); ++i)
      Check(sorted[i] == expected[i], "Cached sort index");
    auto after = metal::GetStatistics();
    Check(after.metric_copy_bytes - before.metric_copy_bytes == rows * 8,
          "Index publication bytes");
    if (repeat)
      Check(after.vector_allocations == before.vector_allocations, "Index cache allocation");
  }
  // Explicit CPU use of the same cache is allowed; the next MPS call recovers
  // device state without changing the cached result or silently staying on CPU.
  auto host_rank = cache.SortedIdx(&cpu, pred.ConstHostSpan());
  for (std::size_t i = 0; i < rows; ++i) Check(host_rank[i] == expected[i], "CPU cache handoff");
  Resident(&pred);
  auto device_rank = cache.SortedIdx(&gpu, pred.ConstDeviceSpan());
  for (std::size_t i = 0; i < rows; ++i) Check(device_rank[i] == expected[i], "MPS cache handoff");
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
void ScoreBindings(metal::RankingMetricKind kind, std::size_t rows) {
  std::size_t shape[1]{rows}, stride[1]{2};
  std::vector<bst_group_t> groups{0, bst_group_t(rows - 3), bst_group_t(rows)};
  std::array<std::vector<float>, 3> source{std::vector<float>(2 * rows, -91),
                                           std::vector<float>(2 * rows, -91),
                                           std::vector<float>{1.f, 1.25f}};
  for (std::size_t i = 0; i < rows; ++i) {
    source[0][2 * i] = float((i * 7) % 23);
    source[1][2 * i] = float((i + 1) % 2);
  }
  std::vector<double> discounts(rows - 3), inverse(2);
  metal::InitNDCG({source[1], shape, stride, DeviceOrd::CPU()}, groups, 7, true, discounts,
                  inverse);
  auto run = [&](std::array<common::Span<float const>, 3> in, common::Span<double const> d,
                 common::Span<double const> inv) {
    return metal::RankingScore(kind, {in[0], shape, stride, DeviceOrd::MPS()},
                               {in[1], shape, stride, DeviceOrd::MPS()}, groups, in[2], 7, false,
                               true, d, inv);
  };
  auto expected = run({source[0], source[1], source[2]}, discounts, inverse);
  for (int fallback = -1; fallback < 5; ++fallback) {
    current = "score bindings/" + std::to_string(int(kind)) + "/" + std::to_string(rows) + "/" +
              std::to_string(fallback);
    std::array<HostDeviceVector<float>, 3> buffers{Buffer(source[0]), Buffer(source[1]),
                                                   Buffer(source[2])};
    std::array<common::Span<float const>, 3> input;
    for (unsigned i = 0; i < 3; ++i)
      input[i] = buffers[i].DeviceSpan().subspan(1, source[i].size());
    auto dbuf = Buffer(discounts), ibuf = Buffer(inverse);
    common::Span<double const> d = dbuf.DeviceSpan().subspan(1, discounts.size()),
                               inv = ibuf.DeviceSpan().subspan(1, inverse.size());
    Mapping mapping;
    std::size_t host = 0;
    if (fallback >= 0 && fallback < 3) {
      host = input[fallback].size_bytes();
      input[fallback] = {static_cast<float const*>(mapping.ReadOnly(source[fallback].data(), host)),
                         source[fallback].size()};
    }
    if (fallback == 3) {
      host = d.size_bytes();
      d = {static_cast<double const*>(mapping.ReadOnly(discounts.data(), host)), discounts.size()};
    }
    if (fallback == 4) {
      host = inv.size_bytes();
      inv = {static_cast<double const*>(mapping.ReadOnly(inverse.data(), host)), inverse.size()};
    }
    auto before = metal::GetStatistics();
    auto actual = run(input, d, inv);
    auto after = metal::GetStatistics();
    Check(!std::memcmp(&actual, &expected, sizeof(actual)), "Score binding bits");
    Check(after.metric_host_input_bytes - before.metric_host_input_bytes == host,
          "Read-only fallback bytes");
    auto total = (4 * rows + 2) * 4 + (discounts.size() + inverse.size()) * 8;
    Check(after.metric_shared_input_bytes - before.metric_shared_input_bytes == total - host,
          "Shared input bytes");
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Input readback");
    Check(after.metric_final_readback_bytes - before.metric_final_readback_bytes == 16,
          "Score final bytes");
    for (unsigned i = 0; i < 3; ++i) {
      auto const& values = buffers[i].ConstHostVector();
      Check(values.front() == -91 && values.back() == -91, "Input guards");
      Check(!std::memcmp(values.data() + 1, source[i].data(), source[i].size() * 4),
            "Input changed");
    }
    for (auto* values : {&dbuf, &ibuf})
      Check(values->ConstHostVector().front() == -91 && values->ConstHostVector().back() == -91,
            "Cache guards");
    ++cases;
  }
}
void CacheBindings(std::size_t rows) {
  std::vector<float> labels(2 * rows, -91);
  for (std::size_t i = 0; i < rows; ++i) labels[2 * i] = float(i % 4);
  std::size_t shape[1]{rows}, stride[1]{2}, max_group = rows - 3;
  std::vector<bst_group_t> groups{0, bst_group_t(max_group), bst_group_t(rows)};
  std::vector<double> discounts(max_group), inverse(2);
  metal::InitNDCG({labels, shape, stride, DeviceOrd::CPU()}, groups, 7, true, discounts, inverse);
  std::vector<std::size_t> sorted(rows);
  metal::SortGroups({labels, shape, stride, DeviceOrd::CPU()}, groups, sorted);
  for (int mode = 0; mode < 8; ++mode) {
    current = "cache bindings/" + std::to_string(rows) + "/" + std::to_string(mode);
    std::vector<float> padded(labels.size() + 4, -91);
    std::copy(labels.begin(), labels.end(), padded.begin() + 2);
    HostDeviceVector<float> input{padded, DeviceOrd::MPS()};
    common::Span<float const> label = input.DeviceSpan().subspan(2, labels.size());
    Mapping mapping;
    if (mode == 1)
      label = {static_cast<float const*>(mapping.ReadOnly(labels.data(), labels.size() * 4)),
               labels.size()};
    auto d_buffer = Buffer(std::vector<double>(max_group)),
         i_buffer = Buffer(std::vector<double>(2));
    auto d = d_buffer.DeviceSpan().subspan(1, max_group), inv = i_buffer.DeviceSpan().subspan(1, 2);
    std::vector<char> bytes(d.size_bytes() + 32, 'x'), expected_bytes = bytes;
    if (mode == 2) d = {reinterpret_cast<double*>(input.DevicePointer() + 2), max_group};
    if (mode == 3) inv = {reinterpret_cast<double*>(input.DevicePointer() + 2), 2};
    if (mode == 4 || mode == 7) d = {reinterpret_cast<double*>(bytes.data() + 1), max_group};
    if (mode == 5) inv = {reinterpret_cast<double*>(bytes.data() + 1), 2};
    if (mode == 6) {
      d = {reinterpret_cast<double*>(bytes.data() + 8), max_group};
      inv = {reinterpret_cast<double*>(bytes.data() + 16), 2};
    }
    if (mode == 7) inv = {reinterpret_cast<double*>(bytes.data() + 8), 2};
    auto before = metal::GetStatistics();
    Check(metal::CheckRankingLabels({label, shape, stride, DeviceOrd::MPS()},
                                    metal::RankingLabelKind::kNDCG) == 0,
          "Strided label validation");
    metal::InitNDCG({label, shape, stride, DeviceOrd::MPS()}, groups, 7, true, d, inv);
    auto after = metal::GetStatistics();
    auto input_bytes = 2 * labels.size() * 4, host_input = mode == 1 ? input_bytes : 0;
    auto output_bytes = d.size_bytes() + inv.size_bytes();
    auto host_output = mode >= 6   ? output_bytes
                       : mode == 4 ? d.size_bytes()
                       : mode == 5 ? inv.size_bytes()
                                   : 0;
    Check(after.metric_host_input_bytes - before.metric_host_input_bytes == host_input,
          "Cache input fallback");
    Check(after.metric_shared_input_bytes - before.metric_shared_input_bytes ==
              input_bytes - host_input,
          "Cache shared inputs");
    Check(after.metric_host_output_bytes - before.metric_host_output_bytes == host_output,
          "Cache output fallback");
    Check(after.metric_shared_output_bytes - before.metric_shared_output_bytes ==
              output_bytes - host_output,
          "Cache shared output");
    Check(after.metric_copy_bytes - before.metric_copy_bytes == output_bytes - host_output,
          "Cache copies");
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Cache input readback");
    if (mode < 6) {
      Check(!std::memcmp(d.data(), discounts.data(), d.size_bytes()), "Discount bits");
      Check(!std::memcmp(inv.data(), inverse.data(), inv.size_bytes()), "Inverse bits");
    }
    if (mode == 2) std::memcpy(padded.data() + 2, discounts.data(), d.size_bytes());
    if (mode == 3) std::memcpy(padded.data() + 2, inverse.data(), inv.size_bytes());
    if (mode == 4 || mode == 7)
      std::memcpy(expected_bytes.data() + 1, discounts.data(), d.size_bytes());
    if (mode == 5) std::memcpy(expected_bytes.data() + 1, inverse.data(), inv.size_bytes());
    if (mode == 6) {
      std::memcpy(expected_bytes.data() + 8, discounts.data(), d.size_bytes());
      std::memcpy(expected_bytes.data() + 16, inverse.data(), inv.size_bytes());
    }
    if (mode == 7) std::memcpy(expected_bytes.data() + 8, inverse.data(), inv.size_bytes());
    Check(!std::memcmp(input.ConstHostVector().data(), padded.data(), padded.size() * 4),
          "Cache input/alias guards");
    Check(bytes == expected_bytes, "Cache output overlap order/guards");
    for (auto* b : {&d_buffer, &i_buffer})
      Check(b->ConstHostVector().front() == -91 && b->ConstHostVector().back() == -91,
            "Cache output guards");
    ++cases;
  }
  for (unsigned mode = 0; mode < 3; ++mode) {
    current = "sort publication/" + std::to_string(rows) + "/" + std::to_string(mode);
    std::vector<float> padded(labels.size() + 4, -91);
    std::copy(labels.begin(), labels.end(), padded.begin() + 2);
    HostDeviceVector<float> input{padded, DeviceOrd::MPS()};
    auto result = Buffer(std::vector<std::size_t>(rows));
    auto output = result.DeviceSpan().subspan(1, rows);
    std::vector<char> bytes(rows * 8 + 2, 'x'), expected_bytes = bytes;
    if (mode == 1) output = {reinterpret_cast<std::size_t*>(input.DevicePointer() + 2), rows};
    if (mode == 2) output = {reinterpret_cast<std::size_t*>(bytes.data() + 1), rows};
    auto before = metal::GetStatistics();
    metal::SortGroups(
        {input.ConstDeviceSpan().subspan(2, labels.size()), shape, stride, DeviceOrd::MPS()},
        groups, output);
    auto after = metal::GetStatistics();
    Check(!std::memcmp(output.data(), sorted.data(), rows * 8), "Sorted index bits");
    Check(after.metric_host_output_bytes - before.metric_host_output_bytes ==
              (mode == 2 ? rows * 8 : 0),
          "Sort fallback bytes");
    Check(after.metric_shared_output_bytes - before.metric_shared_output_bytes ==
              (mode == 2 ? 0 : rows * 8),
          "Sort publication bytes");
    Check(after.metric_host_input_bytes == before.metric_host_input_bytes, "Sort input upload");
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Sort input mirror");
    if (mode == 1) std::memcpy(padded.data() + 2, sorted.data(), rows * 8);
    if (mode == 2) std::memcpy(expected_bytes.data() + 1, sorted.data(), rows * 8);
    Check(!std::memcmp(input.ConstHostVector().data(), padded.data(), padded.size() * 4),
          "Sort input/alias guards");
    Check(bytes == expected_bytes, "Sort fallback guards");
    Check(result.ConstHostVector().front() == std::size_t(-91) &&
              result.ConstHostVector().back() == std::size_t(-91),
          "Sort result guards");
    ++cases;
  }
}
void Errors(Context const& gpu) {
  for (auto name : {"pre", "ndcg", "map"}) {
    current = std::string{name} + "/GPU invalid labels";
    auto data = Matrix(33);
    auto& info = data->Info();
    info.labels.Reshape(33, 1);
    Resident(info.labels.Data());
    info.labels.Data()->Fill(-1.f);
    HostDeviceVector<float> pred(33, .5f, DeviceOrd::MPS());
    std::unique_ptr<Metric> metric{Metric::Create(name, &gpu)};
    metric->Configure({});
    auto before = metal::GetStatistics();
    Reject([&] { metric->Evaluate(pred, data); });
    NoTransfers(before);
    Check(!info.labels.Data()->HostCanRead(), "Invalid labels read on CPU");
    ++cases;
  }
}
int main() {
  try {
    Context gpu;
    gpu.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    for (auto name : {"pre", "pre@3", "ndcg", "ndcg@7-", "map", "map@3-"})
      for (auto rows : {0, 1, 33, 513})
        for (unsigned weight = 0; weight < 3; ++weight) MetricCase(gpu, name, rows, weight);
    for (auto rows : {0, 33, 513, 8197}) CacheCase(gpu, rows);
    for (auto rows : {33, 513, 8197})
      for (auto kind : {metal::RankingMetricKind::kPrecision, metal::RankingMetricKind::kNDCG,
                        metal::RankingMetricKind::kMAP})
        ScoreBindings(kind, rows);
    for (auto rows : {33, 513, 8197}) CacheBindings(rows);
    Errors(gpu);
    Check(metal::GetStatistics().vector_live_buffers == 0 &&
              metal::GetStatistics().vector_capacity_bytes == 0,
          "Leaked vectors");
    std::printf("PASS %zu ranking metric/cache state cases / %zu checks / %zu rejected calls\n",
                cases, fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
