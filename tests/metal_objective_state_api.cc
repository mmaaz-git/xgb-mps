// Objective integration: owned GPU inputs/outputs, explicit mirrors and CPU oracles.
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

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ops.h"
#include "../xgboost/src/common/metal_survival.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, rejected = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Near(float a, float b) {
  ++fields;
  if (a == b || (std::isnan(a) && std::isnan(b))) return;
  if (!std::isfinite(a) || !std::isfinite(b) ||
      std::abs(a - b) > 3e-6f * std::max(1.f, std::abs(b))) {
    std::fprintf(stderr, "%s actual %.9g expected %.9g\n", current.c_str(), a, b);
    Check(false, "CPU oracle differs");
  }
}
template <typename Fn>
void Reject(Fn fn) {
  bool threw = false;
  try {
    fn();
  } catch (dmlc::Error const&) {
    threw = true;
  }
  Check(threw, "Invalid request accepted");
  ++rejected;
}
template <typename T>
void Resident(HostDeviceVector<T>* values) {
  values->SetDevice(DeviceOrd::MPS());
  values->DevicePointer();  // Deny host read access before measuring the operation.
}
void NoTransfers(metal::Statistics const& before) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes == before.vector_readback_bytes, "Implicit host mirror");
  Check(after.objective_staging_bytes == before.objective_staging_bytes, "Payload staging");
}
void Run(Context const& gpu, char const* name, std::size_t rows, unsigned targets, bool weighted,
         char const* distribution = "normal") {
  current = std::string{name} + "/" + distribution + "/" + std::to_string(rows) + "/" +
            std::to_string(targets) + (weighted ? "/weighted" : "/unweighted");
  bool multi = std::string{name}.find("multi:") == 0;
  bool ordered =
      std::string{name} == "reg:expectileerror" || std::string{name} == "reg:quantileerror";
  bool aft = std::string{name} == "survival:aft";
  bool binary = std::string{name}.find("logistic") != std::string::npos ||
                std::string{name} == "binary:logitraw" || std::string{name} == "binary:hinge";
  auto label_targets = multi || ordered || aft ? 1u : targets;
  auto cpu = gpu.MakeCPU();
  Args args{{"num_class", std::to_string(targets)},
            {"expectile_alpha", targets == 1 ? "[0.5]" : "[0.1,0.5,0.9]"},
            {"quantile_alpha", targets == 1 ? "[0.5]" : "[0.1,0.5,0.9]"},
            {"aft_loss_distribution", distribution},
            {"huber_slope", "1.3"},
            {"tweedie_variance_power", "1.3"},
            {"scale_pos_weight", "1.7"}};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, &gpu)},
      ref{ObjFunction::Create(name, &cpu)};
  obj->Configure(args);
  ref->Configure(args);
  MetaInfo info;
  info.num_row_ = rows;
  info.labels.Reshape(rows, label_targets);
  for (std::size_t i = 0; i < rows * label_targets; ++i)
    info.labels.Data()->HostVector()[i] = multi    ? float(i % targets)
                                          : binary ? float(i % 2)
                                                   : .5f + float(i % 7) / 4;
  if (weighted) {
    info.weights_.Resize(rows);
    for (std::size_t i = 0; i < rows; ++i) info.weights_.HostVector()[i] = float(i % 7) / 4;
  }
  if (aft) {
    info.labels_lower_bound_.Resize(rows);
    info.labels_upper_bound_.Resize(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      auto lo = .5f + float(i % 5) / 4;
      info.labels_lower_bound_.HostVector()[i] = i % 4 == 0 ? 0.f : lo;
      info.labels_upper_bound_.HostVector()[i] = i % 4 == 1 ? INFINITY : i % 4 == 2 ? lo : lo + 1;
    }
  }
  HostDeviceVector<float> pred(rows * targets);
  for (std::size_t i = 0; i < pred.Size(); ++i) pred.HostVector()[i] = -.25f + float(i % 11) / 8;
  auto raw = pred.ConstHostVector();
  linalg::Matrix<GradientPair> expected, actual;
  ref->GetGradient(pred, info, 0, &expected);
  Resident(&pred);
  Resident(info.labels.Data());
  Resident(&info.weights_);
  Resident(&info.labels_lower_bound_);
  Resident(&info.labels_upper_bound_);
  auto before = metal::GetStatistics();
  obj->GetGradient(pred, info, 0, &actual);
  NoTransfers(before);
  // CUDA multiclass deliberately returns before reshaping on an empty label array.
  if (!(multi && !rows)) {
    Check(actual.Shape(0) == rows && actual.Shape(1) == targets, "Gradient shape");
    Check(
        actual.Device().IsMPS() && actual.Data()->DeviceCanWrite() && !actual.Data()->HostCanRead(),
        "Gradient output is not device-authoritative");
  }
  auto second = metal::GetStatistics();
  auto pointer = actual.Size() ? actual.Data()->ConstDevicePointer() : nullptr;
  obj->GetGradient(pred, info, 1, &actual);
  NoTransfers(second);
  auto after = metal::GetStatistics();
  Check(after.vector_allocations == second.vector_allocations, "Repeated gradient allocation");
  Check(after.vector_upload_bytes == second.vector_upload_bytes, "Repeated input upload");
  if (actual.Size())
    Check(pointer == actual.Data()->ConstDevicePointer(), "Output buffer replaced");
  auto values = actual.HostView().Values(), oracle = expected.HostView().Values();
  Check(values.size() == oracle.size(), "Oracle shape");
  for (std::size_t i = 0; i < values.size(); ++i) {
    Near(values[i].GetGrad(), oracle[i].GetGrad());
    Near(values[i].GetHess(), oracle[i].GetHess());
  }
  auto first = actual.Data()->ConstHostVector();
  // CPU users may subsequently request the same state; then reuse it on Metal.
  ref->GetGradient(pred, info, 1, &actual);
  obj->GetGradient(pred, info, 1, &actual);
  for (std::size_t i = 0; i < actual.Size(); ++i) {
    Near(actual.HostView().Values()[i].GetGrad(), first[i].GetGrad());
    Near(actual.HostView().Values()[i].GetHess(), first[i].GetHess());
  }
  for (bool eval : {false, true}) {
    HostDeviceVector<float> a(raw), b(raw, DeviceOrd::MPS());
    auto start = metal::GetStatistics();
    if (eval)
      obj->EvalTransform(&b);
    else
      obj->PredTransform(&b);
    NoTransfers(start);
    Check(!b.HostCanRead() && b.Device().IsMPS(), "Transform materialized host state");
    if (eval)
      ref->EvalTransform(&a);
    else
      ref->PredTransform(&a);
    Check(a.Size() == b.Size(), "Transform shape");
    auto const& result = b.ConstHostVector();
    for (std::size_t i = 0; i < result.size(); ++i) Near(result[i], a.ConstHostVector()[i]);
  }
  ++cases;
}
// Direct API callers still support host views, read-only inputs and aliasing.
// Compare the same computation on disjoint storage to isolate the binding logic;
// separate source/CPU oracle suites verify the unchanged arithmetic kernels.
void Bindings(Context const& gpu) {
  for (unsigned kind = 0; kind < 3; ++kind) {
    std::size_t rows = 257, targets = kind == 2 ? 1 : 3, n = rows * targets;
    std::vector<float> pred(n), label(rows), weight(rows), alpha{.1f, .5f, .9f};
    for (std::size_t i = 0; i < n; ++i) pred[i] = -.25f + float(i % 11) / 8;
    for (std::size_t i = 0; i < rows; ++i) {
      label[i] = .5f + float(i % 5) / 4;
      weight[i] = .25f + float(i % 7) / 4;
    }
    // AFT uses label and weight arrays as its lower and upper bounds, with a
    // separate positive sample-weight vector. Ensure nonempty intervals.
    if (kind == 2)
      for (std::size_t i = 0; i < rows; ++i) weight[i] = label[i] + 1;
    std::vector<float> sample_weight(rows, 1.25f);
    auto run = [&](common::Span<float const> p, common::Span<float const> y,
                   common::Span<float const> w, common::Span<float const> a,
                   common::Span<GradientPair> out) {
      if (kind == 2)
        metal::SurvivalGradients(0, 1.f, p, y, w, sample_weight, out);
      else
        metal::AdaptiveGradients(&gpu, static_cast<metal::AdaptiveObjective>(kind), p, y, w,
                                 targets, kind ? a : common::Span<float const>{}, out);
    };
    std::vector<GradientPair> expected(n);
    run(pred, label, weight, alpha, expected);
    for (unsigned mode = 0; mode < 7; ++mode) {
      if (mode == 4 && kind != 1) continue;  // alpha only used by quantile
      current = "binding/" + std::to_string(kind) + "/" + std::to_string(mode);
      common::Span<float const> p = pred, y = label, w = weight, a = alpha;
      std::vector<GradientPair> backing(n + 4, GradientPair{-71.f, -91.f});
      common::Span<GradientPair> output{backing.data() + 1, n};
      std::vector<GradientPair> guards = backing;
      std::vector<float> misaligned(2 * n + 4, -71.f);
      void* mapping = nullptr;
      auto page = std::size_t(getpagesize());
      auto mapped_bytes = ((pred.size() * 4 + page - 1) / page) * page;
      if (mode == 6) {
        // GradientPair has float alignment on the host, but Metal float2 needs 8.
        static_assert(alignof(GradientPair) == alignof(float));
        output = {reinterpret_cast<GradientPair*>(misaligned.data() + 1), n};
        Check(reinterpret_cast<std::uintptr_t>(output.data()) % 8 == 4, "Test alignment");
      } else if (mode == 5) {
        mapping =
            mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        Check(mapping != MAP_FAILED, "mmap");
        std::memcpy(mapping, pred.data(), pred.size() * 4);
        Check(mprotect(mapping, mapped_bytes, PROT_READ) == 0, "mprotect");
        p = {static_cast<float const*>(mapping), pred.size()};
      } else if (mode) {
        auto source = mode == 1 ? p : mode == 2 ? y : mode == 3 ? w : a;
        auto aliased = reinterpret_cast<float*>(backing.data() + 2);
        std::memcpy(aliased, source.data(), source.size_bytes());
        common::Span<float const> replacement{aliased, source.size()};
        if (mode == 1)
          p = replacement;
        else if (mode == 2)
          y = replacement;
        else if (mode == 3)
          w = replacement;
        else
          a = replacement;
        guards = backing;
      }
      auto before = metal::GetStatistics();
      run(p, y, w, a, output);
      auto after = metal::GetStatistics();
      if (mode && mode != 5) {
        Check(after.objective_shared_output_bytes == before.objective_shared_output_bytes,
              "Aliased output was shared");
        Check(after.objective_staging_bytes - before.objective_staging_bytes >= output.size_bytes(),
              "Aliased output fallback not counted");
      }
      if (mode == 5)
        Check(after.objective_staging_bytes - before.objective_staging_bytes >= pred.size() * 4,
              "Read-only input fallback not counted");
      Check(!std::memcmp(output.data(), expected.data(), output.size_bytes()),
            "Binding changed result bits");
      Check(
          !std::memcmp(backing.data(), guards.data(), sizeof(GradientPair)) &&
              !std::memcmp(backing.data() + n + 1, guards.data() + n + 1, 3 * sizeof(GradientPair)),
          "Output guards");
      if (mode == 6)
        Check(misaligned.front() == -71.f && misaligned.back() == -71.f &&
                  misaligned[2 * n + 1] == -71.f,
              "Misaligned output guards");
      fields += 2 * n;
      if (mapping) Check(munmap(mapping, mapped_bytes) == 0, "munmap");
      ++cases;
    }
  }
}
void BaseMargins(Context const& gpu) {
  auto cpu = gpu.MakeCPU();
  for (auto name : {"reg:squarederror", "reg:logistic", "binary:logistic", "binary:logitraw",
                    "reg:gamma", "count:poisson", "reg:tweedie", "survival:cox", "survival:aft"}) {
    current = std::string{name} + "/base margin";
    std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, &gpu)},
        ref{ObjFunction::Create(name, &cpu)};
    obj->Configure({});
    ref->Configure({});
    for (std::size_t n : {0ul, 1ul, 257ul}) {
      linalg::Vector<float> a({n}, DeviceOrd::CPU()), b({n}, DeviceOrd::MPS());
      std::vector<float> values(n);
      for (std::size_t i = 0; i < n; ++i) values[i] = .125f + float(i % 7) / 8;
      a.Data()->Copy(values);
      b.Data()->Copy(values);
      auto before = metal::GetStatistics();
      obj->ProbToMargin(&b);
      NoTransfers(before);
      Check(!b.Data()->HostCanRead(), "Base margin materialized host mirror");
      ref->ProbToMargin(&a);
      for (std::size_t i = 0; i < n; ++i) Near(b.HostView()(i), a.HostView()(i));
      ++cases;
    }
  }
  for (auto name : {"reg:logistic", "reg:gamma"}) {
    current = std::string{name} + "/invalid base margin";
    std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, &gpu)};
    obj->Configure({});
    linalg::Vector<float> values({257}, DeviceOrd::MPS());
    values.Data()->Fill(.5f);
    auto saved = values.Data()->ConstHostVector();
    saved[129] = -1.f;
    values.Data()->Copy(saved);
    auto before = metal::GetStatistics();
    Reject([&] { obj->ProbToMargin(&values); });
    NoTransfers(before);
    Check(!values.Data()->HostCanRead(), "Invalid margin read back payload");
    Check(values.Data()->ConstHostVector() == saved, "Invalid margin partially changed values");
    ++cases;
  }
}
void InvalidLabels(Context const& gpu) {
  for (auto name : {"binary:logistic", "reg:squaredlogerror", "reg:gamma", "count:poisson",
                    "reg:tweedie", "multi:softprob"}) {
    current = std::string{name} + "/invalid label";
    bool multi = std::string{name}.find("multi:") == 0;
    std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, &gpu)};
    obj->Configure({{"num_class", "3"}});
    MetaInfo info;
    info.num_row_ = 17;
    info.labels.Reshape(17, 1);
    Resident(info.labels.Data());
    info.labels.Data()->Fill(-2.f);  // Actual GPU mutation, with a stale host mirror.
    HostDeviceVector<float> pred(17 * (multi ? 3 : 1), 0.f, DeviceOrd::MPS());
    linalg::Matrix<GradientPair> output;
    auto before = metal::GetStatistics();
    Reject([&] { obj->GetGradient(pred, info, 0, &output); });
    NoTransfers(before);
    Check(!info.labels.Data()->HostCanRead(), "Invalid label materialized host mirror");
    ++cases;
  }
}
int main() {
  try {
    Context gpu;
    gpu.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    for (auto name : {"reg:squarederror", "reg:logistic", "binary:logistic", "binary:logitraw",
                      "reg:squaredlogerror", "reg:gamma", "reg:pseudohubererror", "count:poisson",
                      "reg:tweedie", "binary:hinge", "reg:expectileerror", "reg:quantileerror",
                      "reg:absoluteerror", "multi:softprob", "multi:softmax"}) {
      for (std::size_t rows : {0ul, 1ul, 17ul, 257ul, 1025ul})
        for (unsigned targets : {1u, 3u})
          for (bool weighted : {false, true}) Run(gpu, name, rows, targets, weighted);
      std::printf("Checked resident %s\n", name);
      std::fflush(stdout);
    }
    for (auto distribution : {"normal", "logistic", "extreme"})
      for (std::size_t rows : {0ul, 1ul, 17ul, 257ul, 1025ul})
        for (bool weighted : {false, true})
          Run(gpu, "survival:aft", rows, 1, weighted, distribution);
    Bindings(gpu);
    BaseMargins(gpu);
    InvalidLabels(gpu);
    auto stats = metal::GetStatistics();
    Check(!stats.vector_live_buffers && !stats.vector_capacity_bytes, "Live buffer leak");
    std::printf("PASS %zu objective storage cases / %zu field checks / %zu rejected calls\n", cases,
                fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
