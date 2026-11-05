// AUC storage/dispatch contract. Arithmetic is independently covered by the
// existing CUDA source, CPU pair-counting, prefix and cancellation tests.
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

#include "../xgboost/src/common/metal_auc.h"
#include "../xgboost/src/common/metal_collective.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/metric/metric_common.h"
#include "xgboost/c_api.h"

using namespace xgboost;
namespace metal = common::metal;
std::string current;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Api(int rc) {
  if (rc) throw std::runtime_error(XGBGetLastError());
}
void Near(double actual, double expected) {
  ++fields;
  if (actual == expected || (std::isnan(actual) && std::isnan(expected))) return;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > 3e-6 * std::max(1., std::abs(expected))) {
    std::fprintf(stderr, "%s: actual %.17g expected %.17g\n", current.c_str(), actual, expected);
    Check(false, "CPU reference differs");
  }
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
struct Evaluator {
  std::unique_ptr<Metric> owner;
  MetricNoCache* metric;
  Evaluator(Context const& ctx, char const* name) : owner{Metric::Create(name, &ctx)} {
    owner->Configure({});
    metric = dynamic_cast<MetricNoCache*>(owner.get());
    Check(metric, "Metric type");
  }
  double Eval(HostDeviceVector<float> const& pred, MetaInfo const& info) {
    return metric->Eval(pred, info);
  }
};
// kind: binary, multiclass, multilabel, grouped. layout: populated, rank0 empty
// (without target metadata), all empty, rank0 empty (with target metadata).
void Evaluate(Context const& gpu, char const* name, unsigned kind, std::size_t rows, unsigned mode,
              int rank, int world, unsigned layout) {
  current = std::string{name} + "/" + std::to_string(kind) + "/" + std::to_string(rows) + "/" +
            std::to_string(mode) + "/" + std::to_string(layout);
  bool multi = kind == 1 || kind == 2;
  auto local_rows = layout == 2 || ((layout == 1 || layout == 3) && rank == 0) ? 0 : rows;
  unsigned targets = multi ? 3 : 1, columns = kind == 2 ? 3 : 1;
  auto cpu = gpu.MakeCPU();
  Evaluator expected(cpu, name), actual(gpu, name);
  MetaInfo info;
  info.num_row_ = local_rows;
  info.labels.Reshape(local_rows, !local_rows && layout != 3 ? 0 : columns);
  HostDeviceVector<float> pred(local_rows * targets);
  if (kind == 3 && local_rows) info.group_ptr_ = {0, bst_group_t(local_rows)};
  if (mode) info.weights_.Resize(kind == 3 ? bool(local_rows) : local_rows);
  for (std::size_t i = 0; i < info.labels.Size(); ++i)
    info.labels.Data()->HostVector()[i] =
        kind == 1 ? float(i % 3) : float((i / columns + i % columns) % 2);
  for (std::size_t i = 0; i < pred.Size(); ++i)
    pred.HostVector()[i] = float((i * 7 + rank * 3) % 23) / 32;
  for (std::size_t i = 0; i < info.weights_.Size(); ++i)
    info.weights_.HostVector()[i] = mode == 2 ? 0.f : float(1 + i % 5) / 4;
  auto resident = [&] {
    Resident(&pred);
    Resident(info.labels.Data());
    Resident(&info.weights_);
  };
  bool globally_empty = layout == 2 || !rows || ((layout == 1 || layout == 3) && world == 1);
  bool computes = !globally_empty && (local_rows || multi);
  std::size_t summary_bytes = computes && multi ? 3 * targets * sizeof(double) : 0;
  std::size_t input_bytes =
      computes ? 4 * (pred.Size() + info.labels.Size() + info.weights_.Size()) + summary_bytes : 0;
  auto verify = [&](metal::Statistics const& before,
                    metal::CollectiveStatistics const& collective_before) {
    auto after = metal::GetStatistics();
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Unexpected host mirror");
    Check(after.vector_upload_bytes == before.vector_upload_bytes, "Unexpected vector upload");
    auto collective_after = metal::GetCollectiveStatistics();
    Check((collective_after.reduce_dispatches > collective_before.reduce_dispatches) ==
              (world > 1 && computes && multi),
          "Wrong AUC collective placement");
    Check(collective_after.staging_bytes == collective_before.staging_bytes,
          "Resident AUC collective staged its payload");
    Check(after.metric_host_input_bytes == before.metric_host_input_bytes, "Input staging");
    Check(after.metric_host_output_bytes == before.metric_host_output_bytes, "Output staging");
    Check(after.metric_shared_input_bytes - before.metric_shared_input_bytes == input_bytes,
          "Shared input bytes");
    Check(after.metric_shared_output_bytes - before.metric_shared_output_bytes ==
              (computes && kind != 3 ? 24 * targets : 0),
          "Shared output bytes");
    Check(after.auc_final_readback_bytes - before.auc_final_readback_bytes ==
              (computes ? (multi       ? 8
                           : kind == 3 ? 16
                                       : 0)
                        : 0),
          "Final scalar bytes");
    Check(after.metric_dispatches - before.metric_dispatches == (computes ? (multi ? 2 : 1) : 0),
          "Dispatch count");
    Check(!pred.HostCanRead() && !info.labels.Data()->HostCanRead() && !info.weights_.HostCanRead(),
          "Input access changed");
  };
  auto reference = expected.Eval(pred, info);
  resident();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto before = metal::GetStatistics();
    auto collective_before = metal::GetCollectiveStatistics();
    Near(actual.Eval(pred, info), reference);
    verify(before, collective_before);
  }
  if (world == 1 && local_rows) {
    pred.Fill(.375f);  // Actual GPU mutation, then explicit CPU handoff.
    reference = expected.Eval(pred, info);
    resident();
    auto before = metal::GetStatistics();
    auto collective_before = metal::GetCollectiveStatistics();
    Near(actual.Eval(pred, info), reference);
    verify(before, collective_before);
  }
  ++cases;
}
struct Mapping {
  void* data{nullptr};
  std::size_t bytes{0};
  void const* ReadOnly(void const* source, std::size_t size, std::size_t offset = 8) {
    auto page = std::size_t(getpagesize());
    bytes = (size + offset + page - 1) / page * page;
    data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    Check(data != MAP_FAILED, "mmap");
    auto result = static_cast<char*>(data) + offset;
    std::memcpy(result, source, size);
    Check(mprotect(data, bytes, PROT_READ) == 0, "mprotect");
    return result;
  }
  ~Mapping() {
    if (data && data != MAP_FAILED) munmap(data, bytes);
  }
};
// All input slots at nonzero offsets, including strided rows/columns. Grouped
// sizes exercise the small, medium and radix sort paths with the same views.
void Bindings(bool roc, unsigned kind, std::size_t rows) {
  unsigned targets = kind == 1 || kind == 2 ? 3 : 1, columns = kind == 1 ? 1 : targets;
  bool grouped = kind == 3;
  std::size_t shape[2]{rows, targets}, label_shape[2]{rows, columns};
  std::size_t stride[2]{2, rows * 2 + 3};
  std::vector<bst_group_t> groups{0, bst_group_t(rows - 3), bst_group_t(rows)};
  std::array<std::vector<float>, 3> source{
      std::vector<float>((rows - 1) * 2 + (targets - 1) * stride[1] + 1, -91),
      std::vector<float>((rows - 1) * 2 + (columns - 1) * stride[1] + 1, -91),
      std::vector<float>(grouped ? 2 : rows)};
  for (std::size_t i = 0; i < rows; ++i) {
    for (unsigned t = 0; t < targets; ++t)
      source[0][i * 2 + t * stride[1]] = float((i * 7 + t * 3) % 23) / 32;
    for (unsigned t = 0; t < columns; ++t)
      source[1][i * 2 + t * stride[1]] = kind == 1 ? float(i % 3) : float((i + t) % 2);
  }
  for (std::size_t i = 0; i < source[2].size(); ++i) source[2][i] = float(1 + i % 5) / 4;
  auto run = [&](std::array<common::Span<float const>, 3> in, common::Span<double> output) {
    linalg::MatrixView<float const> pred{in[0], shape, stride, DeviceOrd::MPS()};
    linalg::MatrixView<float const> label{in[1], label_shape, stride, DeviceOrd::MPS()};
    if (grouped) {
      auto p = pred.Slice(linalg::All(), 0), y = label.Slice(linalg::All(), 0);
      auto sums = roc ? metal::RankingROCAUC(p, y, groups, in[2])
                      : metal::RankingPRAUC(p, y, groups, in[2]);
      std::memcpy(output.data(), &sums, sizeof(sums));
    } else if (kind == 0) {
      common::Span<metal::ROCAUCSums> out{reinterpret_cast<metal::ROCAUCSums*>(output.data()),
                                          targets};
      if (roc)
        metal::ROCAUC(pred, label, in[2], false, out);
      else
        metal::PRAUC(pred, label, in[2], false, out);
    } else
      metal::MultiAUC(pred, label, in[2], kind == 1, roc, output);
  };
  std::vector<double> reference(grouped ? 2 : targets * 3);
  run({source[0], source[1], source[2]}, reference);
  auto output_bytes = reference.size() * 8;
  for (int mode = -1; mode < (grouped ? 3 : 7); ++mode) {
    // -1 shared; 0..2 read-only input; 3..5 output aliases input; 6 unaligned output fallback.
    current = "bindings/" + std::to_string(roc) + "/" + std::to_string(kind) + "/" +
              std::to_string(rows) + "/" + std::to_string(mode);
    std::array<HostDeviceVector<float>, 3> buffers;
    std::array<std::vector<float>, 3> padded;
    std::array<common::Span<float const>, 3> input;
    std::size_t input_bytes = 0;
    for (unsigned i = 0; i < 3; ++i) {
      padded[i].resize(std::max(source[i].size(), output_bytes / 4) + 4, -91);
      std::copy(source[i].begin(), source[i].end(), padded[i].begin() + 2);
      buffers[i] = HostDeviceVector<float>{padded[i], DeviceOrd::MPS()};
      input[i] = buffers[i].DeviceSpan().subspan(2, source[i].size());
      input_bytes += source[i].size() * 4;
    }
    Mapping mapping;
    std::size_t host_bytes = 0;
    if (mode >= 0 && mode < 3) {
      host_bytes = source[mode].size() * 4;
      input[mode] = {static_cast<float const*>(mapping.ReadOnly(source[mode].data(), host_bytes)),
                     source[mode].size()};
    }
    HostDeviceVector<double> result(reference.size() + 2, -91., DeviceOrd::MPS());
    common::Span<double> output = result.DeviceSpan().subspan(1, reference.size());
    std::vector<char> unaligned(output_bytes + 16, 'x');
    if (mode >= 3 && mode < 6)
      output = {reinterpret_cast<double*>(buffers[mode - 3].DevicePointer() + 2), reference.size()};
    if (mode == 6) output = {reinterpret_cast<double*>(unaligned.data() + 1), reference.size()};
    auto before = metal::GetStatistics();
    run(input, output);
    auto after = metal::GetStatistics();
    Check(!std::memcmp(output.data(), reference.data(), output_bytes), "Binding result bits");
    Check(after.metric_host_input_bytes - before.metric_host_input_bytes == host_bytes,
          "Fallback input bytes");
    Check(after.metric_shared_input_bytes - before.metric_shared_input_bytes ==
              input_bytes - host_bytes,
          "Shared input bytes");
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Input host mirror");
    Check(after.metric_host_output_bytes - before.metric_host_output_bytes ==
              (mode == 6 ? output_bytes : 0),
          "Fallback output bytes");
    Check(after.metric_shared_output_bytes - before.metric_shared_output_bytes ==
              (!grouped && mode != 6 ? output_bytes : 0),
          "Output publication bytes");
    Check(after.metric_copy_bytes - before.metric_copy_bytes ==
              (!grouped && mode != 6 ? output_bytes : 0),
          "Publication copy bytes");
    Check(after.auc_final_readback_bytes - before.auc_final_readback_bytes == (grouped ? 16 : 0),
          "Grouped scalar bytes");
    for (unsigned i = 0; i < 3; ++i) {
      if (mode == int(i + 3)) std::memcpy(padded[i].data() + 2, reference.data(), output_bytes);
      auto const& values = buffers[i].ConstHostVector();
      Check(!std::memcmp(values.data(), padded[i].data(), values.size() * 4),
            "Input/alias guards changed");
      fields += values.size();
    }
    Check(result.ConstHostVector().front() == -91 && result.ConstHostVector().back() == -91,
          "Output guards");
    Check(unaligned.front() == 'x' && unaligned.back() == 'x', "Fallback guards");
    fields += reference.size();
    ++cases;
  }
}
void NormalizeBindings() {
  std::vector<double> source{2, 1, 1, 4, 2, 3, 8, 3, 7};
  auto reference = metal::NormalizeAUC(source);
  for (bool readonly : {false, true}) {
    current = "normalization/" + std::to_string(readonly);
    HostDeviceVector<double> values(source.size() + 2, -91., DeviceOrd::MPS());
    auto input = values.DeviceSpan().subspan(1, source.size());
    std::memcpy(input.data(), source.data(), source.size() * 8);
    Mapping mapping;
    common::Span<double const> span = input;
    if (readonly)
      span = {static_cast<double const*>(mapping.ReadOnly(source.data(), source.size() * 8)),
              source.size()};
    auto before = metal::GetStatistics();
    auto actual = metal::NormalizeAUC(span);
    auto after = metal::GetStatistics();
    Check(!std::memcmp(&actual, &reference, sizeof(actual)), "Normalization bits");
    Check(after.metric_host_input_bytes - before.metric_host_input_bytes == (readonly ? 72 : 0),
          "Normalization fallback bytes");
    Check(after.metric_shared_input_bytes - before.metric_shared_input_bytes == (readonly ? 0 : 72),
          "Normalization shared bytes");
    Check(after.vector_readback_bytes == before.vector_readback_bytes, "Normalization host mirror");
    Check(after.auc_final_readback_bytes - before.auc_final_readback_bytes == 8,
          "Normalization scalar bytes");
    Check(values.ConstHostVector().front() == -91 && values.ConstHostVector().back() == -91,
          "Normalization guards");
    ++cases;
  }
}
void Errors(Context const& gpu) {
  auto cpu = gpu.MakeCPU();
  for (auto name : {"auc", "aucpr"}) {
    Evaluator ref(cpu, name), actual(gpu, name);
    MetaInfo info;
    info.num_row_ = 7;
    info.labels.Reshape(7, 1);
    info.labels.Data()->Fill(0.f);
    HostDeviceVector<float> pred(7, .5f);
    info.weights_.Resize(6, 1.f);
    current = std::string{name} + "/invalid weights";
    Reject([&] { ref.Eval(pred, info); });
    Resident(&pred);
    Resident(info.labels.Data());
    Resident(&info.weights_);
    auto before = metal::GetStatistics();
    Reject([&] { actual.Eval(pred, info); });
    Check(metal::GetStatistics().metric_dispatches == before.metric_dispatches,
          "Invalid weights dispatched");
    info.weights_.Resize(1, 1.f);
    info.group_ptr_ = {0, 7};
    info.labels.Data()->Fill(2.f);
    if (std::string{name} == "aucpr") {
      Resident(info.labels.Data());
      info.labels.Data()->Fill(2.f);
      before = metal::GetStatistics();
      Reject([&] { actual.Eval(pred, info); });
      Check(metal::GetStatistics().vector_readback_bytes == before.vector_readback_bytes,
            "Invalid label host mirror");
    }
    ++cases;
  }
}
int main(int argc, char** argv) {
  try {
    Check(argc == 2, "Pass communicator JSON");
    Api(XGCommunicatorInit(argv[1]));
    int rank = XGCommunicatorGetRank(), world = XGCommunicatorGetWorldSize();
    Context gpu;
    gpu.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    for (auto name : {"auc", "aucpr"})
      for (unsigned kind = 0; kind < 4; ++kind)
        for (unsigned mode = 0; mode < 3; ++mode)
          if (world == 1) {
            for (std::size_t rows : {0, 1, 33, 257})
              Evaluate(gpu, name, kind, rows, mode, rank, world, 0);
          } else {
            for (unsigned layout = 0; layout < 4; ++layout)
              Evaluate(gpu, name, kind, 33 + rank * 6, mode, rank, world, layout);
          }
    if (world == 1) {
      for (bool roc : {false, true}) {
        for (unsigned kind = 0; kind < 3; ++kind) Bindings(roc, kind, 257);
        for (std::size_t rows : {33, 513, 8197}) Bindings(roc, 3, rows);
      }
      NormalizeBindings();
      Errors(gpu);
    }
    Check(metal::GetStatistics().vector_live_buffers == 0 &&
              metal::GetStatistics().vector_capacity_bytes == 0,
          "Leaked vector buffers");
    Api(XGCommunicatorFinalize());
    std::printf("PASS world%d rank%d: %zu AUC state cases / %zu fields / %zu rejected calls\n",
                world, rank, cases, fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
