// Full binary/multi-output corpus. Included after the original-source capture
// copy in cuda_auc_contracts.cu; no expected result enters a CUDA computation.
#pragma once
#include "auc_contract_io.h"
#include "../xgboost/src/metric/metric_common.h"

namespace xgboost::metric::cuda_impl {
// Exported by the original auc.cu translation unit; not declared in auc.h.
double ScaleOutputs(Context const*, bool, common::Span<double>, common::Span<double>,
                    common::Span<double>, common::Span<double>, std::size_t);
}
namespace auc_replay {
namespace ac = auc_contracts;
using namespace xgboost;
inline std::vector<double> Tuple(std::tuple<double, double, double> const& v) {
  return {std::get<0>(v), std::get<1>(v), std::get<2>(v)};
}
inline void Same(std::vector<double> const& a, std::vector<double> const& b) {
  ac::Require(a.size() == b.size(), "AUC probe/linked shape mismatch");
  for (std::size_t i = 0; i < a.size(); ++i)
    ac::Require(ac::Bits<std::uint64_t>(a[i]) == ac::Bits<std::uint64_t>(b[i]) ||
                    (std::isnan(a[i]) && std::isnan(b[i])),
                "AUC instrumented result differs from original linked library");
}
inline std::vector<float> Pack(std::vector<float> const& data, ac::Spec s, unsigned columns) {
  auto strides = ac::Strides(s, columns);
  std::vector<float> result(std::size_t{s.rows} * columns);
  for (std::size_t r = 0; r < s.rows; ++r)
    for (unsigned c = 0; c < columns; ++c)
      result[r * columns + c] = data[r * strides[0] + c * strides[1]];
  return result;
}
inline ac::Observation Normalize(Context const& ctx, ac::Input const& in) {
  auto n = in.spec.targets;
  std::vector<double> host(4 * n);
  for (unsigned i = 0; i < n; ++i) {
    host[i] = in.summaries[3 * i];
    host[2 * n + i] = in.summaries[3 * i + 1];
    host[3 * n + i] = in.summaries[3 * i + 2];
  }
  dh::device_vector<double> buffer(host);
  auto d = dh::ToSpan(buffer);
  auc_capture::Observation capture;
  ac::Observation out;
  {
    auc_capture::Scope scope{&capture};
    out.final = {metric::auc_probe::ScaleOutputs(&ctx, false, d, d.subspan(0, n),
                       d.subspan(2 * n, n), d.subspan(3 * n, n), n)};
  }
  out.linked = {metric::cuda_impl::ScaleOutputs(&ctx, false, d, d.subspan(0, n),
                   d.subspan(2 * n, n), d.subspan(3 * n, n), n)};
  Same(out.final, out.linked);
  ac::Require(capture.normalization_present && !capture.terms_present &&
              !capture.prefixes_present && !capture.totals_present && !capture.curves_present,
              "Unexpected normalization capture stages");
  out.metadata = std::move(capture.normalization);
  auto after = auc_capture::Read(d);
  ac::Require(host.empty() || std::memcmp(host.data(), after.data(), host.size() * sizeof(double)) == 0,
              "CUDA normalization modified its input summaries");
  return out;
}
inline ac::Observation Curve(Context const& ctx, ac::Input const& in, unsigned call) {
  auto s = in.spec;
  bool roc = ac::IsROC(s, call), multi = ac::IsMulti(s);
  auto p = Pack(in.predictions, s, s.targets), y = Pack(in.labels, s, ac::LabelColumns(s));
  MetaInfo info;
  info.num_row_ = s.rows;
  info.labels.Reshape(s.rows, ac::LabelColumns(s));
  info.labels.Data()->HostVector() = y;
  info.labels.SetDevice(ctx.Device());
  info.weights_.HostVector() = in.weights;
  info.weights_.SetDevice(ctx.Device());
  HostDeviceVector<float> predictions{p, ctx.Device()};
  auto pred = predictions.ConstDeviceSpan();
  std::shared_ptr<metric::cuda_impl::DeviceAUCCache> linked_cache;
  std::shared_ptr<metric::auc_probe::DeviceAUCCache> probe_cache;
  auc_capture::Observation capture;
  capture.trace = ac::Trace(s, call);
  ac::Observation out;
  auto type = s.multiclass ? metric::MultiAUCType::kMultiClass : metric::MultiAUCType::kMultiLabel;
  if (multi) {
    {
      auc_capture::Scope scope{&capture};
      out.final = {roc ? metric::auc_probe::MultiROCAUC(&ctx, pred, info, &probe_cache, s.targets, type)
                       : metric::auc_probe::MultiPRAUC(&ctx, pred, info, &probe_cache, s.targets, type)};
    }
    out.linked = {roc ? metric::cuda_impl::MultiROCAUC(&ctx, pred, info, &linked_cache, s.targets, type)
                      : metric::cuda_impl::MultiPRAUC(&ctx, pred, info, &linked_cache, s.targets, type)};
    Same(out.final, out.linked);
    if (s.family == ac::kWide) {
      std::unique_ptr<Metric> metric{Metric::Create(roc ? "auc" : "aucpr", &ctx)};
      metric->Configure({});
      auto evaluator = dynamic_cast<MetricNoCache*>(metric.get());
      ac::Require(evaluator, "Wrong public CUDA AUC metric type");
      out.final = {evaluator->Eval(predictions, info)};
      Same(out.final, out.linked);
    }
  } else if (roc && !s.rows) {
    // BinaryAUC explicitly rejects empty labels. Observe both real guards;
    // Metal's empty raw triple is retained separately, never invented here.
    auto rejects = [&](auto fn, auto* cache) {
      try { fn(&ctx, pred, info, cache); }
      catch (dmlc::Error const& error) {
        ac::Require(std::string(error.what()).find("labels.Size()") != std::string::npos,
                    "Unexpected empty binary ROC error");
        return true;
      }
      return false;
    };
    {
      auc_capture::Scope scope{&capture};
      ac::Require(rejects(metric::auc_probe::BinaryROCAUC, &probe_cache), "Missing probe empty ROC guard");
    }
    ac::Require(rejects(metric::cuda_impl::BinaryROCAUC, &linked_cache), "Missing linked empty ROC guard");
    out.status = 1;
  } else {
    {
      auc_capture::Scope scope{&capture};
      out.curves = Tuple(roc ? metric::auc_probe::BinaryROCAUC(&ctx, pred, info, &probe_cache)
                             : metric::auc_probe::BinaryPRAUC(&ctx, pred, info, &probe_cache));
    }
    out.linked = Tuple(roc ? metric::cuda_impl::BinaryROCAUC(&ctx, pred, info, &linked_cache)
                          : metric::cuda_impl::BinaryPRAUC(&ctx, pred, info, &linked_cache));
    Same(out.curves, out.linked);
  }
  bool traced = capture.trace && s.rows &&
                (multi || (!capture.totals.empty() && capture.totals[0] > 0 && capture.totals[1] > 0));
  ac::Require(capture.normalization_present == multi && capture.curves_present == (multi && s.rows) &&
              capture.totals_present == !roc && capture.terms_present == traced &&
              capture.prefixes_present == traced, "CUDA AUC source stage presence mismatch");
  out.source_curves = std::move(capture.curves);
  out.metadata = std::move(capture.normalization);
  out.terms = std::move(capture.terms); out.prefixes = std::move(capture.prefixes);
  out.totals = std::move(capture.totals); out.order = std::move(capture.sorted_indices);
  auto same_bits = [](auto const& a, auto const& b) {
    return a.size() == b.size() && (a.empty() || !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
  };
  ac::Require(same_bits(predictions.ConstHostVector(), p) && same_bits(info.labels.Data()->ConstHostVector(), y) &&
              same_bits(info.weights_.ConstHostVector(), in.weights), "CUDA AUC modified input values");
  return out;
}
inline int Run(char const* input, char const* output) {
  auto metadata = std::string(output) + ".json";
  ac::Require(!std::filesystem::exists(metadata), "AUC metadata already exists");
  int count = 0; ::Cuda(cudaGetDeviceCount(&count));
  ac::Require(count > 0, "Actual NVIDIA GPU required");
  ::Cuda(cudaSetDevice(0));
  auto ctx = Context{}.MakeCUDA(0); ctx.Init({{"device", "cuda:0"}, {"nthread", "1"}});
  ac::InputReader reader(input); ac::Writer writer(output, ac::kCuda);
  unsigned calls = 0, guards = 0, exclusions = 0, adaptations = 0;
  for (unsigned i = 0; i < ac::kCases; ++i) {
    auto in = reader.Next();
    std::vector<ac::Observation> observations;
    if (in.spec.family == ac::kMain && in.spec.layout != 0) ++adaptations;
    for (unsigned c = 0; c < ac::Calls(in.spec); ++c) {
      ac::Observation out;
      if (in.spec.family == ac::kMalformed) { out.status = 2; ++exclusions; }
      else if (in.spec.family == ac::kNormalize) out = Normalize(ctx, in);
      else out = Curve(ctx, in, c);
      guards += out.status == 1; ++calls;
      observations.push_back(std::move(out));
    }
    writer.Add(in, observations);
    if ((i + 1) % 24 == 0 || i + 1 == ac::kCases) {
      std::printf("CUDA AUC %u/%u records; %u calls, %u source guards, %u API exclusions\n",
                  i + 1, ac::kCases, calls, guards, exclusions); std::fflush(stdout);
    }
  }
  reader.Finish(); writer.Finish();
  cudaDeviceProp gpu{}; ::Cuda(cudaGetDeviceProperties(&gpu, 0));
  int driver = 0, runtime = 0; ::Cuda(cudaDriverGetVersion(&driver)); ::Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};
  meta["suite"] = String("auc-v1"); meta["backend"] = String("cuda"); meta["gpu"] = String(gpu.name);
  meta["driver"] = Integer(driver); meta["runtime"] = Integer(runtime);
  meta["cc_major"] = Integer(gpu.major); meta["cc_minor"] = Integer(gpu.minor);
  meta["nvcc_major"] = Integer(__CUDACC_VER_MAJOR__); meta["nvcc_minor"] = Integer(__CUDACC_VER_MINOR__);
  meta["nvcc_build"] = Integer(__CUDACC_VER_BUILD__); meta["cub"] = Integer(CUB_VERSION);
  meta["thrust"] = Integer(THRUST_VERSION); meta["cases"] = Integer(ac::kCases);
  meta["calls"] = Integer(calls); meta["empty_binary_roc_guards"] = Integer(guards);
  meta["malformed_shape_exclusions"] = Integer(exclusions); meta["strided_pack_adaptations"] = Integer(adaptations);
  meta["capture_adds_synchronization"] = Boolean(true);
  meta["linked_results_checked"] = Boolean(true);
  meta["input"] = String(input); meta["output"] = String(output);
  std::string json; Json::Dump(meta, &json); std::ofstream stream(metadata); stream << json << '\n';
  ac::Require(bool(stream), "Failed AUC metadata write");
  return 0;
}
}  // namespace auc_replay
