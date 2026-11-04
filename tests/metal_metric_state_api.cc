// Actual metric evaluators with device-authoritative inputs and CUDA's small
// host result. Independent arithmetic oracles remain in the existing suites.
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
#include "../xgboost/src/common/metal_ops.h"
#include "../xgboost/src/common/metal_survival.h"
#include "../xgboost/src/metric/metric_common.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, rejected = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Near(double actual, double expected) {
  ++fields;
  if (actual == expected || (std::isnan(actual) && std::isnan(expected))) return;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > 3e-6 * std::max(1., std::abs(expected))) {
    std::fprintf(stderr, "%s actual %.17g expected %.17g\n", current.c_str(), actual, expected);
    Check(false, "CPU reference differs");
  }
}
void Resident(HostDeviceVector<float>* values) {
  values->SetDevice(DeviceOrd::MPS());
  values->DevicePointer();
}
void NoTransfers(metal::Statistics const& before, std::uint64_t final_bytes) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes == before.vector_readback_bytes, "Input host mirror");
  Check(after.metric_host_input_bytes == before.metric_host_input_bytes, "Metric input staging");
  Check(after.metric_final_readback_bytes - before.metric_final_readback_bytes == final_bytes,
        "Wrong final scalar readback size");
}
template <typename Fn>
void Reject(Fn fn) {
  bool threw = false;
  try {
    fn();
  } catch (dmlc::Error const&) {
    threw = true;
  }
  Check(threw, "Invalid metric call accepted");
  ++rejected;
}
struct Evaluator {
  std::unique_ptr<Metric> owned;
  MetricNoCache* metric;
  Evaluator(Context const& ctx, std::string const& name, Args const& args)
      : owned{Metric::Create(name, &ctx)}, metric{dynamic_cast<MetricNoCache*>(owned.get())} {
    Check(metric, "Expected uncached metric");
    owned->Configure(args);
  }
  double Eval(HostDeviceVector<float> const& preds, MetaInfo const& info) {
    return metric->Eval(preds, info);
  }
};
void Run(Context const& gpu, std::string const& name, std::size_t rows, unsigned targets,
         unsigned mode, std::string const& distribution = "normal") {
  current = name + "/" + distribution + "/" + std::to_string(rows) + "/" + std::to_string(targets) +
            "/" + std::to_string(mode);
  bool multi = name == "merror" || name == "mlogloss";
  bool ordered = name == "quantile" || name == "expectile";
  bool survival = name == "aft-nloglik" || name == "interval-regression-accuracy";
  unsigned levels = ordered ? 3 : 1;
  Args args{{"huber_slope", "1.3"},
            {"quantile_alpha", "[0.1,0.5,0.9]"},
            {"expectile_alpha", "[0.1,0.5,0.9]"},
            {"aft_loss_distribution", distribution},
            {"aft_loss_distribution_scale", "1.3"}};
  auto cpu = gpu.MakeCPU();
  Evaluator ref(cpu, name, args), actual(gpu, name, args);
  MetaInfo info;
  info.num_row_ = rows;
  info.labels.Reshape(rows, multi || survival ? 1 : targets);
  HostDeviceVector<float> preds(rows * (survival ? 1 : targets * levels));
  for (std::size_t i = 0; i < info.labels.Size(); ++i)
    info.labels.Data()->HostVector()[i] = multi ? float(i % targets) : .125f + float(i % 7) / 8;
  for (std::size_t i = 0; i < preds.Size(); ++i)
    preds.HostVector()[i] = .0625f + float(i % 13) / 16;
  if (mode) {
    info.weights_.Resize(rows);
    for (std::size_t i = 0; i < rows; ++i)
      info.weights_.HostVector()[i] = mode == 2 ? 0.f : .25f + float(i % 5) / 4;
  }
  if (survival) {
    info.labels_lower_bound_.Resize(rows);
    info.labels_upper_bound_.Resize(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      float lo = .5f + float(i % 5) / 4;
      info.labels_lower_bound_.HostVector()[i] = i % 4 == 0 ? 0.f : lo;
      info.labels_upper_bound_.HostVector()[i] = i % 4 == 1 ? INFINITY : i % 4 == 2 ? lo : lo + 1;
    }
  }
  auto resident = [&] {
    Resident(&preds);
    Resident(info.labels.Data());
    Resident(&info.weights_);
    Resident(&info.labels_lower_bound_);
    Resident(&info.labels_upper_bound_);
  };
  bool invalid = ordered && (!rows || mode == 2);
  double expected = 0;
  if (invalid)
    Reject([&] { ref.Eval(preds, info); });
  else
    expected = ref.Eval(preds, info);
  resident();
  auto before = metal::GetStatistics();
  if (invalid)
    Reject([&] { actual.Eval(preds, info); });
  else
    Near(actual.Eval(preds, info), expected);
  NoTransfers(before, rows ? 16 : 0);
  auto after = metal::GetStatistics();
  Check(after.metric_dispatches - before.metric_dispatches == bool(rows), "GPU dispatch count");
  if (rows)
    Check(after.metric_shared_input_bytes > before.metric_shared_input_bytes, "No shared input");
  Check(!preds.HostCanRead() && !info.labels.Data()->HostCanRead() && !info.weights_.HostCanRead(),
        "Evaluator changed input access state");
  before = metal::GetStatistics();
  if (invalid)
    Reject([&] { actual.Eval(preds, info); });
  else
    Near(actual.Eval(preds, info), expected);
  NoTransfers(before, rows ? 16 : 0);
  after = metal::GetStatistics();
  Check(after.vector_upload_bytes == before.vector_upload_bytes, "Repeated input upload");
  Check(after.vector_allocations == before.vector_allocations, "Repeated vector allocation");

  // Real GPU writes invalidate the old host mirrors. A CPU handoff then reads
  // them explicitly; a subsequent Metal evaluation must see the updated data.
  if (rows) {
    preds.Fill(.375f);
    info.labels.Data()->Fill(multi ? 0.f : .625f);
    info.weights_.Fill(.75f);
    info.labels_lower_bound_.Fill(.5f);
    info.labels_upper_bound_.Fill(2.f);
    expected = ref.Eval(preds, info);
    resident();
    before = metal::GetStatistics();
    Near(actual.Eval(preds, info), expected);
    NoTransfers(before, 16);
  }
  ++cases;
}
void Errors(Context const& gpu) {
  auto cpu = gpu.MakeCPU();
  for (auto name : {"rmse", "mphe", "mlogloss", "quantile", "expectile", "aft-nloglik"}) {
    current = std::string{name} + "/invalid row weights";
    Evaluator actual(gpu, name, {{"quantile_alpha", "[0.5]"}, {"expectile_alpha", "[0.5]"}}),
        expected(cpu, name, {{"quantile_alpha", "[0.5]"}, {"expectile_alpha", "[0.5]"}});
    MetaInfo info;
    info.num_row_ = 3;
    info.labels.Reshape(3, 1);
    info.labels.Data()->Fill(0.f);
    info.labels_lower_bound_.Resize(3, .5f);
    info.labels_upper_bound_.Resize(3, 1.f);
    HostDeviceVector<float> preds(3, .5f);
    for (bool grouped : {false, true}) {
      info.weights_.Resize(grouped ? 3 : 2, 1.f);
      info.group_ptr_ = grouped ? std::vector<bst_group_t>{0, 1, 2, 3} : std::vector<bst_group_t>{};
      Reject([&] { expected.Eval(preds, info); });
      Resident(&preds);
      Resident(info.labels.Data());
      Resident(&info.weights_);
      Resident(&info.labels_lower_bound_);
      Resident(&info.labels_upper_bound_);
      auto before = metal::GetStatistics();
      Reject([&] { actual.Eval(preds, info); });
      NoTransfers(before, 0);
      Check(metal::GetStatistics().metric_dispatches == before.metric_dispatches,
            "Invalid weight shape dispatched GPU work");
      ++cases;
    }
  }
  for (auto name : {"mlogloss", "merror"}) {
    current = std::string{name} + "/GPU invalid label";
    Evaluator actual(gpu, name, {});
    MetaInfo info;
    info.num_row_ = 17;
    info.labels.Reshape(17, 1);
    info.labels.Data()->Fill(1.f);
    Resident(info.labels.Data());
    info.labels.Data()->Fill(-1.f);
    HostDeviceVector<float> preds(17 * 3, .25f, DeviceOrd::MPS());
    auto before = metal::GetStatistics();
    Reject([&] { actual.Eval(preds, info); });
    NoTransfers(before, 0);
    Check(metal::GetStatistics().metric_label_dispatches == before.metric_label_dispatches + 1,
          "Invalid label was not checked on GPU");
    Check(!info.labels.Data()->HostCanRead(), "Invalid label materialized host state");
    ++cases;
  }
}
void Bindings() {
  constexpr std::size_t rows = 257, targets = 3;
  // All four input positions get independent offset and read-only coverage.
  // Compare unchanged arithmetic across bindings; source expressions are checked
  // separately by metal_metric_api.mm and the AFT replay suites.
  for (unsigned kind = 0; kind < 7; ++kind) {
    bool multi = kind == 1, ordered = kind == 2 || kind == 3, survival = kind >= 4;
    std::vector<float> pred(rows * (survival ? 1 : targets), .5f),
        label(rows * (multi || ordered || survival ? 1 : targets), .25f), weight(rows, .75f),
        extra(ordered ? 3 : rows, .75f);
    if (multi)
      for (std::size_t i = 0; i < rows; ++i) label[i] = i % targets;
    if (ordered) extra = {.1f, .5f, .9f};
    auto run = [&](common::Span<float const> p, common::Span<float const> y,
                   common::Span<float const> w, common::Span<float const> e) {
      if (survival) return metal::SurvivalMetricSums(kind - 4, 1.3f, false, p, y, e, w);
      if (ordered) return metal::QuantileMetricSums(kind == 3, p, y, w, e, 1);
      return metal::MetricSums(multi ? "mlogloss" : "rmse", p, y, w, targets, .5f);
    };
    auto expected = run(pred, label, weight, extra);
    std::array<std::vector<float>, 4> sources{pred, label, weight, extra};
    for (int readonly = -1; readonly < 4; ++readonly) {
      if (readonly == 3 && !ordered && !survival) continue;
      current = "bindings/" + std::to_string(kind) + "/" + std::to_string(readonly);
      std::array<HostDeviceVector<float>, 4> buffers;
      std::array<common::Span<float const>, 4> inputs;
      for (std::size_t i = 0; i < 4; ++i) {
        std::vector<float> padded(sources[i].size() + 2, -91);
        std::copy(sources[i].begin(), sources[i].end(), padded.begin() + 1);
        buffers[i] = HostDeviceVector<float>{padded, DeviceOrd::MPS()};
        inputs[i] = buffers[i].ConstDeviceSpan().subspan(1, sources[i].size());
      }
      void* mapping = nullptr;
      std::size_t mapped_bytes = 0, fallback_bytes = 0;
      if (readonly >= 0) {
        fallback_bytes = sources[readonly].size() * 4;
        auto page = std::size_t(getpagesize());
        mapped_bytes = (fallback_bytes + 4 + page - 1) / page * page;
        mapping =
            mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        Check(mapping != MAP_FAILED, "mmap");
        auto data = static_cast<float*>(mapping) + 1;
        std::memcpy(data, sources[readonly].data(), fallback_bytes);
        Check(mprotect(mapping, mapped_bytes, PROT_READ) == 0, "mprotect");
        inputs[readonly] = {data, sources[readonly].size()};
      }
      auto before = metal::GetStatistics();
      auto result = run(inputs[0], inputs[1], inputs[2], inputs[3]);
      auto after = metal::GetStatistics();
      Check(!std::memcmp(result.data(), expected.data(), sizeof(result)),
            "Binding changed result bits");
      Check(after.metric_host_input_bytes - before.metric_host_input_bytes == fallback_bytes,
            "Fallback byte count");
      std::size_t input_bytes =
          (pred.size() + label.size() + weight.size() + (ordered || survival ? extra.size() : 0)) *
          4;
      Check(after.metric_shared_input_bytes - before.metric_shared_input_bytes ==
                input_bytes - fallback_bytes,
            "Shared binding byte count");
      Check(after.vector_readback_bytes == before.vector_readback_bytes, "Input host readback");
      Check(after.metric_final_readback_bytes - before.metric_final_readback_bytes == 16,
            "Final result bytes");
      for (std::size_t i = 0; i < 4; ++i) {
        auto const& values = buffers[i].ConstHostVector();
        Check(values.front() == -91 && values.back() == -91, "Input guards changed");
        Check(!std::memcmp(values.data() + 1, sources[i].data(), sources[i].size() * 4),
              "Input payload changed");
        fields += values.size();
      }
      if (mapping) Check(munmap(mapping, mapped_bytes) == 0, "munmap");
      fields += 2;
      ++cases;
    }
  }
}
int main() {
  try {
    Context gpu;
    gpu.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    for (auto name :
         {"rmse", "rmsle", "mae", "mape", "logloss", "error@0.371234567", "poisson-nloglik",
          "gamma-deviance", "gamma-nloglik", "tweedie-nloglik@1.3712345", "mphe", "mlogloss",
          "merror", "quantile", "expectile"}) {
      for (std::size_t rows : {0ul, 1ul, 33ul, 257ul, 1025ul})
        for (unsigned targets : {1u, 3u})
          for (unsigned mode : {0u, 1u, 2u}) Run(gpu, name, rows, targets, mode);
      std::printf("Checked metric storage %s\n", name);
      std::fflush(stdout);
    }
    for (auto name : {"aft-nloglik", "interval-regression-accuracy"})
      for (auto distribution : {"normal", "logistic", "extreme"})
        for (std::size_t rows : {0ul, 1ul, 33ul, 257ul, 1025ul})
          for (unsigned mode : {0u, 1u, 2u}) Run(gpu, name, rows, 1, mode, distribution);
    Errors(gpu);
    Bindings();
    auto stats = metal::GetStatistics();
    Check(!stats.vector_live_buffers && !stats.vector_capacity_bytes, "Live vector leak");
    std::printf("PASS %zu metric storage cases / %zu field checks / %zu rejected calls\n", cases,
                fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
