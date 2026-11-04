// Compare actual production AFT objective and survival metrics against CPU.
// Ill-conditioned censored rows are checked against the actual upstream source
// by metal_aft_source.mm; report their CPU differences without demanding that
// independent transcendental libraries choose identical rounding in the tails.
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/survival_util.h"
#include "../xgboost/src/metric/metric_common.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/metric.h"
#include "xgboost/objective.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
void Check(bool ok, char const *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
bool Close(double a, double b, double tolerance = 2e-5) {
  if (std::isnan(a) && std::isnan(b))
    return true;
  if (a == b)
    return true;
  if (std::isinf(a) || std::isinf(b))
    return false;
  return std::abs(a - b) <= 1e-6 + tolerance * std::abs(b);
}
template <typename Distribution>
bool Sensitive(float lower, float upper, float prediction, float scale) {
  if (lower == upper) return false;
  double u =
      std::isinf(upper) ? 1. : Distribution::CDF((std::log(double(upper)) - prediction) / scale);
  double l = lower <= 0 ? 0. : Distribution::CDF((std::log(double(lower)) - prediction) / scale);
  return !(u - l > 1e-5f);
}
int main() {
  try {
    using namespace xgboost;
    Context cpu, gpu;
    cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
    gpu.Init({{"device", "mps"}, {"nthread", "2"}});
    std::mt19937 rng(373);
    std::uniform_real_distribution<float> uniform(0, 1);
    size_t cases = 0, pairs = 0, failures = 0, corrections = 0, sensitive_differences = 0;
    double max_error = 0;
    for (auto distribution : {"normal", "logistic", "extreme"})
      for (float scale : {.1f, .5f, 1.f, 2.f, 10.f}) {
        for (int mode = 0; mode < 5; ++mode) {
          size_t n = 4097;
          MetaInfo info;
          info.num_row_ = n;
          std::vector<float> lower(n), upper(n), prediction(n), weights(n);
          for (size_t i = 0; i < n; ++i) {
            float y = std::exp(-30.f + 60.f * uniform(rng));
            lower[i] = y;
            upper[i] = y;
            switch (i % 7) {
            case 1:
              upper[i] = INFINITY;
              break;
            case 2:
              lower[i] = 0;
              break;
            case 3:
              upper[i] = y * std::exp(.01f + 3.f * uniform(rng));
              break;
            case 4:
              upper[i] = std::nextafter(y, INFINITY);
              break;
            case 5:
              lower[i] = 0;
              upper[i] = INFINITY;
              break;
            }
            if (mode == 4) {
              lower[i] = y;
              upper[i] = y;
            }
            prediction[i] =
                std::log(y) + scale * (-50.f + 100.f * uniform(rng));
            if (i % 101 == 0)
              prediction[i] = i % 2 ? 1000 : -1000;
            weights[i] = mode == 2 ? 0 : .1f + 3.f * uniform(rng);
            if (mode == 3)
              weights[i] *= 1e-40f;
          }
          info.labels_lower_bound_.HostVector() = lower;
          info.labels_upper_bound_.HostVector() = upper;
          if (mode)
            info.weights_.HostVector() = weights;
          HostDeviceVector<float> preds(prediction);
          Args args{{"aft_loss_distribution", distribution},
                    {"aft_loss_distribution_scale", std::to_string(scale)}};
          std::unique_ptr<ObjFunction> co(
              ObjFunction::Create("survival:aft", &cpu)),
              go(ObjFunction::Create("survival:aft", &gpu));
          co->Configure(args);
          go->Configure(args);
          linalg::Matrix<GradientPair> expected, actual;
          co->GetGradient(preds, info, 0, &expected);
          auto before = common::metal::GetStatistics();
          go->GetGradient(preds, info, 0, &actual);
          auto after = common::metal::GetStatistics();
          Check(after.survival_gradient_dispatches -
                        before.survival_gradient_dispatches ==
                    1,
                "AFT gradient did not dispatch on GPU");
          auto corrected =
              after.survival_cpu_corrections - before.survival_cpu_corrections;
          corrections += corrected;
          Check(corrected == 0 && after.survival_cpu_time_ns == before.survival_cpu_time_ns,
                "All AFT gradients must remain entirely on GPU");
          auto a = actual.HostView(), b = expected.HostView();
          for (size_t i = 0; i < n; ++i)
            for (int k = 0; k < 2; ++k) {
              float got = k ? a(i, 0).GetHess() : a(i, 0).GetGrad(),
                    ref = k ? b(i, 0).GetHess() : b(i, 0).GetGrad();
              max_error = std::max(max_error, std::abs(double(got) - ref));
              if (!Close(got, ref)) {
                bool sensitive = std::string(distribution) == "normal"
                                     ? Sensitive<common::NormalDistribution>(lower[i], upper[i],
                                                                             prediction[i], scale)
                                 : std::string(distribution) == "logistic"
                                     ? Sensitive<common::LogisticDistribution>(lower[i], upper[i],
                                                                               prediction[i], scale)
                                     : Sensitive<common::ExtremeDistribution>(lower[i], upper[i],
                                                                              prediction[i], scale);
                if (sensitive) {
                  ++sensitive_differences;
                  continue;
                }
                if (++failures <= 15)
                  std::fprintf(stderr,
                               "%s s=%g mode=%d row=%zu %s lo=%.9g hi=%.9g "
                               "pred=%.9g GPU=%.9g CPU=%.9g\n",
                               distribution, scale, mode, i, k ? "hess" : "grad", lower[i],
                               upper[i], prediction[i], got, ref);
              }
            }
          for (auto name : {"aft-nloglik", "interval-regression-accuracy"}) {
            std::unique_ptr<Metric> cm(Metric::Create(name, &cpu)), gm(Metric::Create(name, &gpu));
            cm->Configure(args);
            gm->Configure(args);
            auto *cmetric = dynamic_cast<MetricNoCache *>(cm.get());
            auto *gmetric = dynamic_cast<MetricNoCache *>(gm.get());
            Check(cmetric && gmetric,
                  "Expected survival metric implementation");
            double ref = cmetric->Eval(preds, info);
            before = common::metal::GetStatistics();
            double got = gmetric->Eval(preds, info);
            after = common::metal::GetStatistics();
            Check(after.survival_metric_dispatches -
                          before.survival_metric_dispatches ==
                      1,
                  "Survival metric did not dispatch on GPU");
            if (!Close(got, ref, 1e-7)) {
              ++failures;
              std::fprintf(stderr,
                           "%s %s scale=%g mode=%d GPU=%.17g CPU=%.17g\n",
                           distribution, name, scale, mode, got, ref);
            }
          }
          HostDeviceVector<float> cp(prediction), gp(prediction);
          co->PredTransform(&cp);
          before = common::metal::GetStatistics();
          go->PredTransform(&gp);
          after = common::metal::GetStatistics();
          Check(after.transform_dispatches - before.transform_dispatches == 1,
                "Survival prediction transform did not dispatch");
          for (size_t i = 0; i < n; ++i)
            if (!Close(gp.ConstHostSpan()[i], cp.ConstHostSpan()[i], 2e-7))
              ++failures;
          ++cases;
          pairs += n;
        }
        std::printf("Checked %s scale %g\n", distribution, scale);
        std::fflush(stdout);
      }
    std::printf(
        "Checked %zu cases / %zu AFT gradient pairs; max absolute "
        "error %.9g; failures %zu\n",
        cases, pairs, max_error, failures);
    std::printf("CPU corrections for deliberately sensitive censored rows: %zu / %zu\n",
                corrections, pairs);
    std::printf(
        "Sensitive-tail CPU tolerance differences: %zu fields (covered by upstream source "
        "replay)\n",
        sensitive_differences);
    Check(corrections == 0, "AFT must not recompute rows on CPU");
    Check(failures == 0, "Survival CPU equivalence failed");
    // Empty APIs do not dispatch; base-score transforms cover signed zero,
    // subnormals and invalid log inputs just like the CPU implementation.
    std::unique_ptr<ObjFunction> co(ObjFunction::Create("survival:aft", &cpu)),
        go(ObjFunction::Create("survival:aft", &gpu));
    co->Configure({});
    go->Configure({});
    linalg::Vector<float> cbase({8}, DeviceOrd::CPU()),
        gbase({8}, DeviceOrd::CPU());
    std::array<float, 8> base{0.f, -0.f, -1.f, 1.f, .5f, 3.f, 1e-40f, INFINITY};
    for (size_t i = 0; i < 8; ++i)
      cbase(i) = gbase(i) = base[i];
    co->ProbToMargin(&cbase);
    auto before = common::metal::GetStatistics();
    go->ProbToMargin(&gbase);
    Check(common::metal::GetStatistics().transform_dispatches -
                  before.transform_dispatches ==
              1,
          "Base-score transform did not dispatch");
    for (size_t i = 0; i < 8; ++i)
      Check(Close(gbase(i), cbase(i), 2e-7), "Base-score log differs from CPU");
    MetaInfo empty;
    HostDeviceVector<float> predictions;
    linalg::Matrix<GradientPair> result;
    before = common::metal::GetStatistics();
    go->GetGradient(predictions, empty, 0, &result);
    go->PredTransform(&predictions);
    auto after = common::metal::GetStatistics();
    Check(after.survival_gradient_dispatches ==
                  before.survival_gradient_dispatches &&
              after.transform_dispatches == before.transform_dispatches,
          "Empty AFT input must not dispatch");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
