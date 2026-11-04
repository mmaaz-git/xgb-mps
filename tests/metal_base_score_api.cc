// Exact CPU base-score formulas, exceptional values and all-or-nothing errors.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ops.h"
#include "../xgboost/src/objective/regression_loss.h"
#include "xgboost/objective.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t checks = 0, cases = 0;
void Check(bool ok, char const* text) {
  if (!ok) throw std::runtime_error(text);
}
float FloatFromBits(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
void Same(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return;
  if (std::memcmp(&a, &b, 4)) {
    std::printf("transform result GPU %.9g CPU %.9g\n", a, b);
    throw std::runtime_error("Base-score bits differ");
  }
}
float Reference(float x, unsigned kind) {
  if (!kind) return x;
  if (kind == 1) return obj::LogisticRegression::ProbToMargin(x);
  return std::log(x);
}
void Transforms() {
  std::mt19937 rng(6731);
  for (unsigned kind = 0; kind < 4; ++kind) {
    auto type = static_cast<metal::BaseScoreTransform>(kind);
    std::vector<float> input;
    for (std::size_t i = 0; i < 1000003; ++i) {
      uint32_t bits = rng();
      if (kind == 1)
        input.push_back(float(bits) * 0x1p-32f);
      else if (kind == 3)
        input.push_back(FloatFromBits(1 + bits % 0x7f7fffffu));
      else
        input.push_back(FloatFromBits(bits));
    }
    for (uint32_t e = 0; e < 255; ++e)
      for (uint32_t index = 0; index < 128; ++index)
        for (int delta = -2; delta <= 2; ++delta) {
          uint32_t bits = (e << 23) + (index << 16) + 0x8000 + delta;
          if (kind != 1 || bits <= 0x3f800000u) input.push_back(FloatFromBits(bits));
        }
    for (uint32_t bits :
         {0u, 1u, 2u, 0x7fffffu, 0x800000u, 0x3f7fffffu, 0x3f800000u, 0x7f7fffffu, 0x7f800000u,
          0x7f800001u, 0x7fc12345u, 0x80000000u, 0x80000001u, 0xff800000u}) {
      float x = FloatFromBits(bits);
      if (kind == 1 && !obj::LogisticRegression::CheckIntercept(x)) continue;
      if (kind == 3 && !obj::GammaDeviance::CheckIntercept(x)) continue;
      input.push_back(x);
    }
    for (std::size_t begin = 0; begin < input.size(); begin += 262144) {
      std::size_t n = std::min<std::size_t>(262144, input.size() - begin);
      std::vector<float> values(n + 2, -971);
      std::copy_n(input.data() + begin, n, values.data() + 1);
      auto before = metal::GetStatistics();
      Check(metal::BaseScoreToMargin(type, {values.data() + 1, n}), "Valid base scores rejected");
      auto after = metal::GetStatistics();
      Check(after.base_score_dispatches - before.base_score_dispatches == bool(kind),
            "Wrong base-score dispatch count");
      Check(!kind || after.base_score_gpu_time_ns > before.base_score_gpu_time_ns,
            "Missing GPU duration");
      for (std::size_t i = 0; i < n; ++i) {
        Same(values[i + 1], Reference(input[begin + i], kind));
        if (!kind)
          Check(!std::memcmp(&values[i + 1], &input[begin + i], 4), "Identity altered NaN bits");
        ++checks;
      }
      Same(values.front(), -971);
      Same(values.back(), -971);
      ++cases;
    }
    auto before = metal::GetStatistics();
    Check(metal::BaseScoreToMargin(type, {}), "Empty transformation failed");
    Check(metal::GetStatistics().base_score_dispatches == before.base_score_dispatches,
          "Empty dispatch");
    std::printf("Checked base-score kind %u / %zu exact results\n", kind, checks);
    std::fflush(stdout);
  }
  for (auto type :
       {metal::BaseScoreTransform::kLogistic, metal::BaseScoreTransform::kPositiveLogarithm}) {
    std::vector<float> invalid =
        type == metal::BaseScoreTransform::kLogistic
            ? std::vector<float>{-FloatFromBits(1), std::nextafter(1.f, 2.f), INFINITY, -INFINITY,
                                 NAN}
            : std::vector<float>{0, -0.f, -FloatFromBits(1), -INFINITY, NAN};
    for (auto bad : invalid)
      for (std::size_t pos : {0u, 17u, 256u}) {
        std::vector<float> values(259, .25f);
        values[pos + 1] = bad;
        auto original = values;
        auto before = metal::GetStatistics();
        Check(!metal::BaseScoreToMargin(type, {values.data() + 1, 257}),
              "Invalid base score accepted");
        Check(metal::GetStatistics().base_score_dispatches == before.base_score_dispatches + 1,
              "Missing error-path dispatch");
        Check(!std::memcmp(values.data(), original.data(), values.size() * 4),
              "Failed validation changed input");
        ++cases;
      }
  }
}
void Objectives() {
  Context cpu, gpu;
  cpu.UpdateAllowUnknown(Args{{"device", "cpu"}, {"nthread", "2"}});
  gpu.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
  for (auto name : {"reg:squarederror", "binary:logitraw", "reg:logistic", "binary:logistic",
                    "reg:gamma", "count:poisson", "reg:tweedie", "survival:cox"}) {
    std::unique_ptr<ObjFunction> expected{ObjFunction::Create(name, &cpu)},
        actual{ObjFunction::Create(name, &gpu)};
    expected->Configure({});
    actual->Configure({});
    bool identity =
        std::string(name) == "reg:squarederror" || std::string(name) == "binary:logitraw";
    bool logistic = std::string(name) == "reg:logistic" || std::string(name) == "binary:logistic";
    for (std::size_t n : {1u, 3u, 257u}) {
      linalg::Vector<float> a({n}, DeviceOrd::CPU()), b({n}, DeviceOrd::CPU());
      for (std::size_t i = 0; i < n; ++i) a.HostView()(i) = b.HostView()(i) = float(i % 7 + 1) / 8;
      auto before = metal::GetStatistics();
      expected->ProbToMargin(&a);
      actual->ProbToMargin(&b);
      Check(
          metal::GetStatistics().base_score_dispatches - before.base_score_dispatches == !identity,
          "Production objective bypassed Metal conversion");
      for (std::size_t i = 0; i < n; ++i) Same(a.HostView()(i), b.HostView()(i));
      if (logistic || std::string(name) == "reg:gamma") {
        a.HostView()(n - 1) = b.HostView()(n - 1) = -.25f;
        bool fail_cpu = false, fail_gpu = false;
        try {
          expected->ProbToMargin(&a);
        } catch (std::exception const&) {
          fail_cpu = true;
        }
        try {
          actual->ProbToMargin(&b);
        } catch (std::exception const&) {
          fail_gpu = true;
        }
        Check(fail_cpu && fail_gpu, "Objective validation mismatch");
        for (std::size_t i = 0; i < n; ++i) Same(a.HostView()(i), b.HostView()(i));
      }
      ++cases;
    }
  }
}
int main() {
  try {
    Transforms();
    Objectives();
    std::printf(
        "PASS %zu base-score cases / %zu exact transformed values, guards and atomic error "
        "behavior\n",
        cases, checks);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
