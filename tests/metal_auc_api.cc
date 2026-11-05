// Production GPU AUC compared with upstream CPU and independent weighted pairs.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../xgboost/src/common/metal_auc.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/optional_weight.h"
#include "../xgboost/src/metric/metric_common.h"
#include "xgboost/context.h"
namespace xgboost::metric {
std::tuple<double, double, double> BinaryPRAUC(Context const*, common::Span<float const>,
                                               linalg::VectorView<float const>,
                                               common::OptionalWeights);
std::tuple<double, double, double> BinaryROCAUC(Context const*, common::Span<float const>,
                                                linalg::VectorView<float const>,
                                                common::OptionalWeights);
}  // namespace xgboost::metric
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void CudaContracts() {
  using namespace xgboost;
  size_t shape[2]{1, 1};
  std::vector<float> predictions{0}, labels{1};
  common::metal::ROCAUCSums out;
  common::metal::ROCAUC({predictions, shape, DeviceOrd::CPU()}, {labels, shape, DeviceOrd::CPU()},
                        {}, false, {&out, 1});
  Check(out.fp == 0 && out.tp == 1 && out.area == 0,
        "CUDA binary ROC retains raw positive mass for degenerate curves");
  common::metal::PRAUC({predictions, shape, DeviceOrd::CPU()}, {labels, shape, DeviceOrd::CPU()},
                       {}, false, {&out, 1});
  Check(out.fp == 0 && out.tp == 0 && out.area == 0,
        "CUDA binary PR returns zero triples for nonpositive original totals");
  shape[0] = 2;
  predictions = {0, 0};
  labels = {.25f, .25f};
  std::vector<float> weights{.7f, .7f};
  common::metal::ROCAUC({predictions, shape, DeviceOrd::CPU()}, {labels, shape, DeviceOrd::CPU()},
                        weights, false, {&out, 1});
  double fp = 2 * double(float((1.0 - double(labels[0])) * double(weights[0])));
  double tp = 2 * double(labels[0] * weights[0]);
  Check(out.fp == fp && out.tp == tp && out.area == (fp * tp) * .5,
        "CUDA curve terms must round once after binary64 negative expression");
  common::metal::PRAUC({predictions, shape, DeviceOrd::CPU()}, {labels, shape, DeviceOrd::CPU()},
                       weights, false, {&out, 1});
  Check(out.fp == 1 && out.tp == 1 && out.area == 1.0 / (fp / tp + 1),
        "CUDA tied PR must use rounded curve terms");
  std::puts("PASS four exact CUDA binary AUC degenerate/cast contracts");
}
size_t current_n, current_targets, current_target;
int current_layout, current_mode;
void Close(double actual, double expected, char const* field = "pair", double atol = 0) {
  if (actual == expected || (std::isnan(actual) && std::isnan(expected))) return;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > atol + 2e-12 * std::abs(expected)) {
    std::fprintf(
        stderr,
        "n=%zu targets=%zu layout=%d mode=%d target=%zu field=%s actual %.17g, expected %.17g\n",
        current_n, current_targets, current_layout, current_mode, current_target, field, actual,
        expected);
    throw std::runtime_error("AUC differs from its CUDA-source comparison contract");
  }
}
struct CudaCurves {
  xgboost::common::metal::ROCAUCSums roc, pr;
};
CudaCurves CudaReference(std::vector<float> const& scores, std::vector<float> const& labels,
                         std::vector<float> const& weights, bool multi) {
  if (scores.empty()) return {};
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](auto a, auto b) { return scores[a] > scores[b]; });
  double total_fp = 0, total_tp = 0;
  for (auto i : order) {
    float w = weights.empty() ? 1 : weights[i];
    total_fp += (1.f - labels[i]) * w;
    total_tp += labels[i] * w;
  }
  double fp = 0, tp = 0, prev_fp = 0, prev_tp = 0, roc = 0, pr = 0;
  for (size_t k = 0; k < order.size(); ++k) {
    auto i = order[k];
    float w = weights.empty() ? 1 : weights[i];
    fp += float((1.0 - double(labels[i])) * double(w));
    tp += labels[i] * w;
    if (k + 1 < order.size() && scores[i] == scores[order[k + 1]]) continue;
    roc += (std::abs(fp - prev_fp) * (tp + prev_tp)) * .5;
    if (total_fp <= 0 || total_tp <= 0) {
      pr = NAN;
    } else {
      double previous = prev_tp / total_tp, recall = tp / total_tp;
      double h = tp == prev_tp ? 0 : (fp - prev_fp) / (tp - prev_tp), a = h + 1;
      double b = tp == prev_tp ? 0 : (prev_fp - h * prev_tp) / total_tp;
      pr += b != 0 ? (recall - previous -
                      b / a * (std::log(a * recall + b) - std::log(a * previous + b))) /
                         a
                   : (recall - previous) / a;
    }
    prev_fp = fp;
    prev_tp = tp;
  }
  using xgboost::common::metal::ROCAUCSums;
  bool invalid = total_fp <= 0 || total_tp <= 0;
  return {{fp, tp, roc}, invalid && !multi ? ROCAUCSums{} : ROCAUCSums{1, 1, pr}};
}
// Exact nonnegative FP32 sum in units of 2^-149. Five limbs cover all finite
// FP32 inputs even at UINT32_MAX rows; Apple long double is not extra precision.
struct ExactFloatSum {
  std::array<uint64_t, 5> words{};
  void Add(float value) {
    Check(value >= 0 && std::isfinite(value), "Exact sum input domain");
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t exponent = (bits >> 23) & 255, mantissa = bits & 0x7fffff;
    if (exponent) mantissa |= 0x800000;
    uint32_t shift = exponent ? exponent - 1 : 0;
    unsigned __int128 carry = static_cast<unsigned __int128>(mantissa) << (shift % 64);
    for (size_t i = shift / 64; carry && i < words.size(); ++i) {
      carry += words[i];
      words[i] = uint64_t(carry);
      carry >>= 64;
    }
    Check(!carry, "Exact sum overflow");
  }
  double Rounded() const {
    int last = 4;
    while (last >= 0 && !words[last]) --last;
    if (last < 0) return 0;
    unsigned high = unsigned(last * 64 + 63 - __builtin_clzll(words[last]));
    unsigned shift = high > 52 ? high - 52 : 0, word = shift / 64, offset = shift % 64;
    uint64_t top = words[word] >> offset;
    if (offset && word + 1 < words.size()) top |= words[word + 1] << (64 - offset);
    top &= (uint64_t{1} << 53) - 1;
    if (shift) {
      unsigned guard = shift - 1;
      bool round = (words[guard / 64] >> (guard % 64)) & 1, sticky = false;
      for (unsigned i = 0; i < guard / 64; ++i) sticky |= words[i] != 0;
      if (guard % 64) sticky |= (words[guard / 64] & ((uint64_t{1} << (guard % 64)) - 1)) != 0;
      if (round && (sticky || (top & 1))) ++top;
    }
    return std::ldexp(double(top), int(shift) - 149);
  }
};
size_t traced_prefixes = 0, traced_terms = 0, traced_curves = 0, source_nan_curves = 0;
double max_prefix_ratio = 0, max_integral_ratio = 0;
// Validate the stages separately when the complete curve is ill-conditioned.
// Prefixes have a forward-error bound against exact integer sums. The source
// integral then receives those observed endpoints, without assuming CUB's tree.
void CheckTrace(std::vector<float> const& scores, std::vector<float> const& labels,
                std::vector<float> const& weights, xgboost::common::metal::AUCTrace const& trace,
                size_t target, double actual) {
  size_t n = scores.size();
  if (!n) return;
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](auto a, auto b) { return scores[a] > scores[b]; });
  unsigned levels = 1;
  for (size_t blocks = (n + 255) / 256; blocks > 1; blocks = (blocks + 255) / 256) ++levels;
  constexpr double u = 0x1p-53;
  auto check_sum = [&](double value, ExactFloatSum const& sum, size_t depth) {
    double exact = sum.Rounded(), gamma = depth * u / (1 - depth * u);
    double bound = (gamma + 2 * u) * exact;
    double error = std::abs(value - exact);
    if (bound) max_prefix_ratio = std::max(max_prefix_ratio, error / bound);
    Check(error <= bound, "GPU AUC prefix exceeds FP64 forward-error bound against exact sum");
  };
  ExactFloatSum fp_sum, tp_sum, fp_total;
  for (size_t k = 0; k < n; ++k) {
    auto i = order[k];
    float w = weights.empty() ? 1 : weights[i];
    float fp = float((1.0 - double(labels[i])) * double(w)), tp = labels[i] * w;
    size_t index = 2 * (target * n + k);
    Check(trace.terms[index] == double(fp) && trace.terms[index + 1] == double(tp),
          "GPU AUC term differs from CUDA FP32 cast contract");
    traced_terms += 2;
    fp_sum.Add(fp);
    tp_sum.Add(tp);
    fp_total.Add((1.f - labels[i]) * w);
    check_sum(trace.prefixes[index], fp_sum, 10 * levels);
    check_sum(trace.prefixes[index + 1], tp_sum, 10 * levels);
    traced_prefixes += 2;
  }
  double total_fp = trace.totals[2 * target], total_tp = trace.totals[2 * target + 1];
  check_sum(total_fp, fp_total, (n + 255) / 256 + 8);
  check_sum(total_tp, tp_sum, (n + 255) / 256 + 8);
  if (total_fp <= 0 || total_tp <= 0) return;
  double fp0 = 0, tp0 = 0, expected = 0, sum_abs = 0, log_bound = 0;
  auto ulp = [](double v) { return std::abs(std::nextafter(v, INFINITY) - v); };
  for (size_t k = 0; k < n; ++k) {
    if (k + 1 < n && scores[order[k]] == scores[order[k + 1]]) continue;
    size_t index = 2 * (target * n + k);
    double fp = trace.prefixes[index], tp = trace.prefixes[index + 1];
    double previous = tp0 / total_tp, recall = tp / total_tp;
    double h = tp == tp0 ? 0 : (fp - fp0) / (tp - tp0), a = h + 1;
    double b = tp == tp0 ? 0 : (fp0 - h * tp0) / total_tp;
    double term = (recall - previous) / a;
    if (b != 0) {
      double l1 = std::log(a * recall + b), l0 = std::log(a * previous + b);
      term = (recall - previous - b / a * (l1 - l0)) / a;
      // Allow four binary64 ULPs per logarithm, propagated through the exact
      // source expression. This is a tested math budget, not a libdevice claim.
      log_bound += std::abs((b / a) / a) * 4 * (ulp(l0) + ulp(l1));
    }
    expected += term;
    sum_abs += std::abs(term);
    fp0 = fp;
    tp0 = tp;
  }
  ++traced_curves;
  if (std::isnan(expected)) {
    ++source_nan_curves;
    Check(std::isnan(actual), "CUDA source formula on observed prefixes must propagate NaN");
    return;
  }
  if (!std::isfinite(expected)) {
    Check(actual == expected, "CUDA source formula infinity classification");
    return;
  }
  double depth = n + (n + 255) / 256 + 32;
  double gamma = depth * u / (1 - depth * u);
  double bound = log_bound + (gamma + 64 * u) * (sum_abs + log_bound);
  double error = std::abs(actual - expected);
  if (bound) max_integral_ratio = std::max(max_integral_ratio, error / bound);
  if (!(error <= bound)) {
    std::fprintf(stderr, "trace actual=%.17g expected=%.17g bound=%.17g\n", actual, expected,
                 bound);
    throw std::runtime_error("GPU PR integral exceeds source/log/reduction error budget");
  }
}
void CheckNormalization() {
  using namespace xgboost;
  size_t checked = 0;
  auto run = [&](std::vector<double> const& input) {
    double sum = 0, weight = 0;
    for (size_t t = 0; t < input.size() / 3; ++t) {
      if (input[3 * t] > 0) {
        sum += (input[3 * t + 2] / input[3 * t]) * input[3 * t + 1];
        weight += input[3 * t + 1];
      } else {
        sum = NAN;
      }
    }
    double expected = weight != 0 && !std::isnan(sum) ? sum / weight : NAN;
    auto snapshot = input;
    auto before = common::metal::GetStatistics();
    auto actual = common::metal::NormalizeAUC(input);
    auto after = common::metal::GetStatistics();
    Check(after.metric_dispatches == before.metric_dispatches + 1,
          "AUC normalization did not execute a GPU command");
    Check(input.empty() || std::memcmp(input.data(), snapshot.data(), input.size() * sizeof(double)) == 0,
          "AUC normalization modified its collective input");
    Close(actual, expected, "CUDA ScaleOutputs", 2e-12);
    ++checked;
  };
  for (auto input : std::vector<std::vector<double>>{{2, 3, 1},
                                                     {2, 3, 1, 4, 1, 3},
                                                     {0, 1, 1},
                                                     {-1, 2, 1},
                                                     {NAN, 1, 1},
                                                     {2, 0, 1},
                                                     {2, 1, NAN},
                                                     {2, -1, 1},
                                                     {2, 1, 1, 2, -1, 1},
                                                     {INFINITY, 1, 1},
                                                     {2, INFINITY, 1},
                                                     {1e-300, 1e-200, 5e-301}})
    run(input);
  for (size_t targets : {1, 2, 31, 257, 65535, 65536, 65537, 262145}) {
    std::vector<double> summaries(targets * 3);
    for (size_t t = 0; t < targets; ++t) {
      summaries[3 * t] = 1 + double(t % 17);
      summaries[3 * t + 1] = 1 + double(t % 5);
      summaries[3 * t + 2] = double(t % 11) / 19;
    }
    run(summaries);
  }
  for (bool multiclass : {false, true}) {
    for (bool roc : {false, true}) {
      for (size_t rows : {0, 4}) {
        size_t shape[2]{rows, 3}, label_shape[2]{rows, multiclass ? 1ul : 3ul};
        std::vector<float> scores(rows * 3), labels(rows * label_shape[1]), weights(rows, .7f);
        for (size_t i = 0; i < scores.size(); ++i) scores[i] = float(i % 5);
        for (size_t i = 0; i < labels.size(); ++i)
          labels[i] = multiclass ? float(i % 3) : float(i % 3) / 3;
        std::vector<double> output(13, -971);
        auto before = common::metal::GetStatistics();
        common::metal::MultiAUC({scores, shape, DeviceOrd::CPU()},
                                {labels, label_shape, DeviceOrd::CPU()}, weights, multiclass, roc,
                                {output.data() + 2, 9});
        Check(common::metal::GetStatistics().metric_dispatches == before.metric_dispatches + 1,
              "Per-output CUDA aggregation metadata did not run on GPU");
        Check(output.front() == -971 && output.back() == -971, "AUC aggregate output guard");
        for (size_t t = 0; t < 3; ++t) {
          std::vector<float> p(rows), y(rows);
          for (size_t r = 0; r < rows; ++r) {
            p[r] = scores[r * 3 + t];
            y[r] = multiclass ? float(labels[r] == t) : labels[r * 3 + t];
          }
          auto reference = CudaReference(p, y, weights, true);
          auto curve = roc ? reference.roc : reference.pr;
          double local = roc ? curve.fp * curve.tp : (rows ? 1.0 : 0.0);
          double weight = rows ? (roc && multiclass ? curve.tp : 1.0) : 0.0;
          Close(output[2 + 3 * t], local, "GPU local_area");
          Close(output[3 + 3 * t], weight, "GPU output_weight");
          Close(output[4 + 3 * t], curve.area, "GPU integral", 2e-12);
        }
        ++checked;
      }
    }
  }
  // CUDA ScaleOutputs returns NaN for an empty reduction and for zero local
  // areas. These are valid inputs, including beyond the former 16-bit cap.
  run({});
  run(std::vector<double>(65536 * 3));
  for (auto bad : std::vector<std::vector<double>>{{1, 2}}) {
    bool rejected = false;
    try {
      common::metal::NormalizeAUC(bad);
    } catch (std::exception const&) {
      rejected = true;
    }
    Check(rejected, "Invalid normalization shape accepted");
  }
  std::printf("PASS %zu GPU normalization/metadata cases and one malformed shape error\n", checked);
}

void WideOutputs() {
  using namespace xgboost;
  Context cpu, gpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  gpu.Init({{"device", "mps"}, {"nthread", "2"}});
  std::size_t fields = 0;
  for (std::size_t targets : {65535, 65536, 65537}) {
    current_n=2; current_targets=targets; current_layout=0; current_mode=0;
    std::size_t shape[]{2, targets};
    std::vector<float> predictions(2*targets), labels(2*targets);
    for (std::size_t t=0; t<targets; ++t) {
      predictions[t]=t%3==1 ? 1.f : 0.f;
      predictions[targets+t]=t%3==0 ? 1.f : 0.f;
      labels[t]=0.f; labels[targets+t]=1.f;
    }
    for (bool roc : {true, false}) {
      std::vector<common::metal::ROCAUCSums> curves(targets);
      auto before=common::metal::GetStatistics();
      if (roc) common::metal::ROCAUC({predictions,shape,DeviceOrd::CPU()},
          {labels,shape,DeviceOrd::CPU()}, {}, false, curves);
      else common::metal::PRAUC({predictions,shape,DeviceOrd::CPU()},
          {labels,shape,DeviceOrd::CPU()}, {}, false, curves);
      auto after=common::metal::GetStatistics();
      Check(after.radix_sort_dispatches==before.radix_sort_dispatches+1 &&
            after.metric_dispatches==before.metric_dispatches+1,"Wide AUC did not execute GPU sort/metric");
      double expected=0;
      for (std::size_t t=0; t<targets; ++t) {
        current_target=t;
        auto value=t%3==0 ? 1. : t%3==2 ? .5 : roc ? 0. : 1.-std::log(2.);
        Close(curves[t].fp,1.,"wide fp"); Close(curves[t].tp,1.,"wide tp");
        Close(curves[t].area,value,"wide analytic curve",2e-12);
        expected+=value; fields+=3;
      }
      expected/=targets;
      MetaInfo info;info.num_row_=2;info.labels.Reshape(2,targets);
      info.labels.Data()->HostVector()=labels;
      HostDeviceVector<float> pred{predictions};
      auto name=roc ? "auc" : "aucpr";
      std::unique_ptr<Metric> cpu_metric{Metric::Create(name,&cpu)}, gpu_metric{Metric::Create(name,&gpu)};
      cpu_metric->Configure({});gpu_metric->Configure({});
      auto cpu_eval=dynamic_cast<MetricNoCache*>(cpu_metric.get());
      auto gpu_eval=dynamic_cast<MetricNoCache*>(gpu_metric.get());
      Check(cpu_eval && gpu_eval,"Wrong AUC metric type");
      Close(cpu_eval->Eval(pred,info),expected,"wide CPU production metric",2e-12);
      pred.SetDevice(DeviceOrd::MPS());pred.DevicePointer();
      info.labels.SetDevice(DeviceOrd::MPS());info.labels.Data()->DevicePointer();
      info.weights_.SetDevice(DeviceOrd::MPS());info.weights_.DevicePointer();
      before=common::metal::GetStatistics();
      Close(gpu_eval->Eval(pred,info),expected,"wide MPS production metric",2e-12);
      after=common::metal::GetStatistics();
      Check(after.metric_dispatches==before.metric_dispatches+2 &&
            after.vector_readback_bytes==before.vector_readback_bytes &&
            after.vector_upload_bytes==before.vector_upload_bytes &&
            after.metric_host_input_bytes==before.metric_host_input_bytes &&
            after.auc_final_readback_bytes==before.auc_final_readback_bytes+8,
            "Wide public AUC did not preserve resident inputs/GPU normalization");
      fields+=2;
      std::printf("PASS wide %s: %zu outputs, actual %.17g\n",name,targets,expected);std::fflush(stdout);
    }
  }
  std::printf("PASS six wide ROC/PR cases / %zu analytic and public-CPU/MPS fields\n",fields);
}
long double ReferencePR(std::vector<float> const& scores, std::vector<float> const& labels,
                        std::vector<float> const& weights) {
  // Independent long-double integral of precision over recall using log1p.
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return scores[a] > scores[b]; });
  long double total = 0, negative = 0;
  for (size_t i = 0; i < scores.size(); ++i) {
    auto w = weights.empty() ? 1.0f : weights[i];
    total += labels[i] * w;
    negative += (1.0f - labels[i]) * w;
  }
  if (total <= 0 || negative <= 0) return std::numeric_limits<long double>::quiet_NaN();
  long double fp = 0, tp = 0, area = 0;
  for (size_t begin = 0; begin < order.size();) {
    size_t end = begin + 1;
    while (end < order.size() && scores[order[end]] == scores[order[begin]]) ++end;
    long double df = 0, dt = 0;
    for (size_t k = begin; k < end; ++k) {
      auto i = order[k];
      auto w = weights.empty() ? 1.0f : weights[i];
      df += float((1.0 - double(labels[i])) * double(w));
      dt += labels[i] * w;
    }
    if (dt > 0) {
      auto a = 1 + df / dt;
      if (fp + tp == 0)
        area += dt / (a * total);
      else
        area += (dt - (fp - (a - 1) * tp) / a * std::log1p(a * dt / (fp + tp))) / (a * total);
    }
    fp += df;
    tp += dt;
    begin = end;
  }
  return area;
}
int main() {
  try {
    CudaContracts();
    CheckNormalization();
    WideOutputs();
    using namespace xgboost;
    Context cpu;
    cpu.Init({{"nthread", "2"}, {"device", "cpu"}});
    std::mt19937 rng(718);
    size_t cases = 0, checked = 0, pair_cases = 0, pr_references = 0;
    double max_pr_difference = 0;
    for (size_t n : {0, 1, 2, 31, 255, 256, 257, 4097, 65537})
      for (size_t targets : {1, 3})
        for (int layout = 0; layout < 3; ++layout)
          for (int mode = 0; mode < 9; ++mode) {
            current_n = n;
            current_targets = targets;
            current_layout = layout;
            current_mode = mode;
            bool multiclass = targets == 3 && mode % 2;
            size_t columns = multiclass ? 1 : targets;
            size_t shape[2]{n, targets}, label_shape[2]{n, columns};
            size_t stride[2]{
                layout == 1 ? 1 : targets * (layout == 2 ? 2 : 1) + (layout == 2 ? 3 : 0),
                layout == 1 ? std::max(n, size_t(1)) : (layout == 2 ? 2 : 1)};
            size_t label_stride[2]{
                layout == 1 ? 1 : columns * (layout == 2 ? 2 : 1) + (layout == 2 ? 3 : 0),
                stride[1]};
            std::vector<float> predictions(n * stride[0] + targets * stride[1] + 1, -999);
            std::vector<float> labels(n * label_stride[0] + columns * label_stride[1] + 1, -999);
            std::vector<float> weights(mode >= 2 ? n : 0);
            linalg::MatrixView<float> p{predictions, shape, stride, DeviceOrd::CPU()};
            linalg::MatrixView<float> y{labels, label_shape, label_stride, DeviceOrd::CPU()};
            for (size_t row = 0; row < n; ++row) {
              for (size_t t = 0; t < targets; ++t) {
                float v = float(int(rng() % 32767) - 16383);
                if (mode == 1) v = row % 2 ? -0.0f : 0.0f;
                if (mode == 2) v = float(rng() % 7);
                if (mode == 3) v = std::ldexp(v, -149);  // nonzero subnormal scores
                if (mode == 4 && row % 37 == 0) v = row % 2 ? INFINITY : -INFINITY;
                p(row, t) = v;
              }
              for (size_t t = 0; t < columns; ++t) {
                float v = float(rng() % (multiclass ? targets : 2));
                if (mode == 5 && !multiclass) v = float(rng() % 17) / 16;
                if (multiclass && mode == 3 && row % 7 == 0)
                  v = std::numeric_limits<float>::denorm_min();
                if (mode == 6) v = 1;
                if (mode == 8) v = float(rng() % 1001) / 1000;
                y(row, t) = v;
              }
              if (!weights.empty()) {
                float w = float(rng() % 31) / 8;
                if (mode == 3) w = std::ldexp(float(rng() % 127), -149);
                if (mode == 4) w = std::ldexp(w, int(rng() % 241) - 120);
                if (mode == 7) w = 0;
                if (mode == 8) w = float(rng() % 101) / 7;
                weights[row] = w;
              }
            }
            std::vector<common::metal::ROCAUCSums> result(targets);
            auto before = common::metal::GetStatistics();
            common::metal::ROCAUC(p, y, weights, multiclass, result);
            auto after = common::metal::GetStatistics();
            Check(after.roc_auc_dispatches - before.roc_auc_dispatches == size_t(n != 0),
                  "ROC dispatch counter mismatch");
            Check(after.radix_sort_dispatches - before.radix_sort_dispatches == size_t(n != 0),
                  "ROC did not use one batched GPU sort");
            std::vector<common::metal::ROCAUCSums> pr_result(targets);
            auto pr_before = common::metal::GetStatistics();
            common::metal::AUCTrace trace;
            common::metal::PRAUC(p, y, weights, multiclass, pr_result,
                                 mode == 4 ? &trace : nullptr);
            auto pr_after = common::metal::GetStatistics();
            Check(pr_after.pr_auc_dispatches - pr_before.pr_auc_dispatches == size_t(n != 0),
                  "PR-AUC dispatch counter mismatch");
            Check(
                pr_after.radix_sort_dispatches - pr_before.radix_sort_dispatches == size_t(n != 0),
                "PR-AUC did not use one batched GPU sort");
            for (size_t t = 0; t < targets; ++t) {
              current_target = t;
              std::vector<float> scores(n), response(n);
              for (size_t row = 0; row < n; ++row) {
                scores[row] = p(row, t);
                response[row] = multiclass ? float(y(row, 0) == t) : y(row, t);
              }
              auto expected = CudaReference(scores, response, weights, multiclass || targets > 1);
              double fp = expected.roc.fp, tp = expected.roc.tp, area = expected.roc.area;
              Close(pr_result[t].fp, expected.pr.fp, "CUDA PR fp");
              Close(pr_result[t].tp, expected.pr.tp, "CUDA PR tp");
              if (mode == 4) {
                CheckTrace(scores, response, weights, trace, t, pr_result[t].area);
              } else {
                Close(pr_result[t].area, expected.pr.area, "CUDA PR area", 2e-10);
              }
              if (mode != 4 && std::isfinite(pr_result[t].area) && std::isfinite(expected.pr.area))
                max_pr_difference =
                    std::max(max_pr_difference, std::abs(pr_result[t].area - expected.pr.area));
              // The mathematical integral remains a reference for well-conditioned
              // curves. Extreme-weight source arithmetic is tested explicitly below.
              if (n <= 257 && mode != 4 && expected.pr.fp == 1) {
                Close(pr_result[t].area, double(ReferencePR(scores, response, weights)),
                      "PR independent integral", 2e-10);
                ++pr_references;
              }
              if (n && mode != 4 && std::all_of(response.begin(), response.end(), [](float v) {
                    return v == 0 || v == 1;
                  })) {
                if (fp > 0 && tp > 0) {
                  auto cpu_roc = metric::BinaryROCAUC(
                      &cpu, scores, linalg::MakeVec(response.data(), n, DeviceOrd::CPU()),
                      common::OptionalWeights{weights});
                  Close(result[t].fp, std::get<0>(cpu_roc), "compatible CPU ROC fp");
                  Close(result[t].tp, std::get<1>(cpu_roc), "compatible CPU ROC tp");
                  Close(result[t].area, std::get<2>(cpu_roc), "compatible CPU ROC area");
                }
                auto cpu_pr = metric::BinaryPRAUC(
                    &cpu, scores, linalg::MakeVec(response.data(), n, DeviceOrd::CPU()),
                    common::OptionalWeights{weights});
                if (expected.pr.fp == 1)
                  Close(pr_result[t].area, std::get<2>(cpu_pr), "compatible CPU PR", 2e-10);
              }
              Close(result[t].fp, fp, "fp");
              Close(result[t].tp, tp, "tp");
              if (mode == 4 && fp > 0 && tp > 0) {
                Close(result[t].area / (fp * tp), area / (fp * tp), "normalized area", 2e-12);
              } else {
                Close(result[t].area, area, "area");
              }
              if (n <= 257 && mode < 5) {
                long double pairs = 0;
                for (size_t i = 0; i < n; ++i)
                  if (response[i] == 1)
                    for (size_t j = 0; j < n; ++j)
                      if (response[j] == 0) {
                        long double wi = weights.empty() ? 1 : weights[i];
                        long double wj = weights.empty() ? 1 : weights[j];
                        pairs += wi * wj *
                                 (scores[i] > scores[j]    ? 1
                                  : scores[i] == scores[j] ? 0.5L
                                                           : 0);
                      }
                if (mode == 4 && fp > 0 && tp > 0)
                  Close(result[t].area / (fp * tp),
                        double(pairs / (static_cast<long double>(fp) * tp)), "normalized pair",
                        2e-12);
                else
                  Close(result[t].area, double(pairs));
                ++pair_cases;
              }
              checked += n;
            }
            ++cases;
            if (cases % 144 == 0) {
              std::printf("Checked %zu ROC cases / %zu score-label pairs\n", cases, checked);
              std::fflush(stdout);
            }
          }
    std::printf(
        "PASS %zu production ROC-AUC cases / %zu score-label pairs, %zu independent "
        "weighted-pair checks; layouts, ties, weights and three-level scans\n",
        cases, checked, pair_cases);
    std::printf(
        "PASS %zu production PR-AUC cases / %zu score-label pairs, %zu long-double "
        "integral checks; maximum CUDA-source difference %.9g\n",
        cases, checked, pr_references, max_pr_difference);
    {
      size_t shape[2]{2, 1};
      std::vector<float> scores{1, 0}, labels{0, 1}, weights{1e30f, 1};
      common::metal::ROCAUCSums result;
      common::metal::PRAUC({scores, shape, DeviceOrd::CPU()}, {labels, shape, DeviceOrd::CPU()},
                           weights, false, {&result, 1});
      Check(result.area == 1.0, "CUDA PR difference-of-logs cancellation must match source");
      auto cpu_result = metric::BinaryPRAUC(&cpu, scores,
          linalg::MakeVec(labels.data(), labels.size(), DeviceOrd::CPU()), common::OptionalWeights{weights});
      Check(std::get<2>(cpu_result) == 1.0, "Update the documented upstream cancellation example");
      std::printf("PASS extreme-weight PR integral %.17g; upstream CPU %.17g\n", result.area, std::get<2>(cpu_result));
    }
    std::printf(
        "PASS %zu exact terms / %zu bounded prefixes / %zu source-integral curves "
        "(%zu NaN); max prefix/bound %.9g, integral/bound %.9g\n",
        traced_terms, traced_prefixes, traced_curves, source_nan_curves, max_prefix_ratio,
        max_integral_ratio);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
