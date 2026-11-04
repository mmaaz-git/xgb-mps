// Exercise the production C++ statistics API, including empty and strided
// inputs.
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/numeric.h"
#include "../xgboost/src/common/optional_weight.h"
#include "../xgboost/src/common/stats.h"
#include "xgboost/context.h"
#include "xgboost/linalg.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

void Check(bool ok, char const *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Same(float actual, float expected) {
  Check((std::isnan(actual) && std::isnan(expected)) ||
            std::memcmp(&actual, &expected, sizeof(float)) == 0,
        "Median API differs from CPU");
}
void Close(float actual, float expected) {
  Check(std::abs(actual - expected) <= 1e-6f + 2e-5f * std::abs(expected),
        "Mean API differs from CPU beyond rounding tolerance");
}
void CheckReductions(xgboost::Context const *cpu, xgboost::Context const *gpu) {
  using namespace xgboost;
  std::mt19937 random(615);
  size_t cases = 0, values_checked = 0;
  for (size_t rows : {0, 1, 2, 31, 256, 257, 4097, 65537, 1000003}) {
    for (int mode = 0; mode < 7; ++mode) {
      HostDeviceVector<float> input(rows);
      auto &values = input.HostVector();
      double sum = 0, correction = 0, absolute = 0;
      for (size_t i = 0; i < rows; ++i) {
        float value = float(int(random() % 2047) - 1023);
        if (mode == 1)
          value = std::ldexp(value, -148);
        if (mode == 2)
          value = std::ldexp(value, int(random() % 231) - 120);
        if (mode == 3)
          value = i % 2 ? -0.0f : 0.0f;
        if (mode == 4 && i == 0)
          value = INFINITY;
        if (mode == 5 && i < 2)
          value = i ? -INFINITY : INFINITY;
        if (mode == 6 && i == rows / 2)
          value = NAN;
        values[i] = value;
        double next = sum + value;
        correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value
                                                       : (value - next) + sum;
        sum = next;
        absolute += std::abs(double(value));
      }
      auto compare = [&](double actual, double expected, bool float32 = false) {
        if ((std::isnan(actual) && std::isnan(expected)) || actual == expected)
          return;
        Check(std::isfinite(actual) && std::isfinite(expected),
              "Nonfinite sum differs from CPU");
        if (mode != 2)
          Check(false, "Exactly representable sum differs from CPU");
        // Wide exponent ranges change summation order. Bound the error relative
        // to input magnitudes, and check a compensated reference independently.
        double steps = 0;
        for (std::size_t n = rows; n > 1; n = (n + 255) / 256) steps += 8;
        double eps = float32 ? std::numeric_limits<float>::epsilon()
                             : std::numeric_limits<double>::epsilon();
        auto tolerance = float32 ? steps * eps / (1 - steps * eps) * absolute : 32 * eps * absolute;
        Check(std::abs(actual - expected) <= tolerance,
              "Reduction differs beyond rounding bound");
      };
      double expected = common::Reduce(cpu, input);
      auto before = common::metal::GetStatistics();
      double actual = common::Reduce(gpu, input);
      auto after = common::metal::GetStatistics();
      Check(after.float_sum_dispatches - before.float_sum_dispatches ==
                uint64_t(rows != 0),
            "Generic reduction did not dispatch on GPU");
      // CUDA numeric::Reduce uses plus<float> and widens only its final result.
      Check(actual == double(float(actual)) || std::isnan(actual), "Expected FP32 result");
      compare(actual, expected, true);
      if (mode < 4) compare(actual, sum + correction, true);
      common::OptionalWeights weights{input.ConstHostSpan()};
      expected = common::SumOptionalWeights(cpu, weights, rows);
      before = common::metal::GetStatistics();
      actual = common::SumOptionalWeights(gpu, weights, rows);
      after = common::metal::GetStatistics();
      Check(after.float_sum_dispatches - before.float_sum_dispatches ==
                uint64_t(rows != 0),
            "Optional-weight reduction did not dispatch on GPU");
      compare(actual, expected);
      values_checked += 2 * rows;
      ++cases;
    }
  }
  HostDeviceVector<float> witness(2);
  witness.HostVector() = {1e8f, 1.f};
  Check(common::Reduce(gpu, witness) == 1e8, "CUDA float sum must round 1e8+1");
  Check(common::SumOptionalWeights(gpu, common::OptionalWeights{witness.ConstHostSpan()}, 2) ==
            100000001.,
        "Optional weights must retain their binary64 sum");
  witness.HostVector() = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
  Check(std::isinf(common::Reduce(gpu, witness)), "CUDA float reduction must overflow here");
  Check(std::isfinite(
            common::SumOptionalWeights(gpu, common::OptionalWeights{witness.ConstHostSpan()}, 2)),
        "Optional-weight binary64 sum must stay finite");
  auto before = common::metal::GetStatistics();
  Check(common::SumOptionalWeights(gpu, common::OptionalWeights{3.5f}, 11) ==
            38.5,
        "Scalar optional weight must use its default multiplier");
  Check(common::metal::GetStatistics().float_sum_dispatches ==
            before.float_sum_dispatches,
        "Scalar optional weight should not dispatch an empty array");
  printf("PASS %zu generic/optional-weight reduction cases / %zu values, "
         "including nonfinite and empty inputs\n",
         cases, values_checked);
}

int main() {
  try {
    using namespace xgboost;
    Context cpu, gpu;
    cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
    gpu.Init({{"device", "mps"}, {"nthread", "2"}});
    std::mt19937 rng(910);
    size_t cases = 0;
    for (size_t rows : {0, 1, 2, 31, 256, 257, 4097})
      for (size_t targets : {1, 3})
        for (int mode = 0; mode < 4; ++mode) {
          linalg::Matrix<float> values({rows, targets}, DeviceOrd::CPU());
          auto data = values.HostView();
          for (size_t r = 0; r < rows; ++r)
            for (size_t t = 0; t < targets; ++t)
              data(r, t) = float(int(rng() % 257) - 128) / 7;
          HostDeviceVector<float> weights;
          if (mode) {
            auto &w = weights.HostVector();
            w.resize(rows);
            for (auto &v : w) {
              v = mode == 2 ? 0 : float(rng() % 31) / 3;
              if (mode == 3)
                v *= 1e-40f;
            }
            if (rows && mode != 2)
              w[0] = mode == 3 ? 1e-40f : 1;
          }
          linalg::Vector<float> expected, actual;
          common::Median(&cpu, values, weights, &expected);
          auto before = common::metal::GetStatistics();
          common::Median(&gpu, values, weights, &actual);
          auto after = common::metal::GetStatistics();
          Check(after.quantile_dispatches - before.quantile_dispatches ==
                    uint64_t(rows != 0),
                "Median did not dispatch on Metal");
          for (size_t t = 0; t < targets; ++t)
            Same(actual(t), expected(t));

          common::SampleMean(&cpu, values, &expected);
          before = common::metal::GetStatistics();
          common::SampleMean(&gpu, values, &actual);
          after = common::metal::GetStatistics();
          Check(after.mean_dispatches - before.mean_dispatches ==
                    uint64_t(rows != 0),
                "SampleMean did not dispatch on Metal");
          for (size_t t = 0; t < targets; ++t)
            Close(actual(t), expected(t));

          // A column slice is strided whenever there are multiple targets.
          auto vector = values.HostView().Slice(linalg::All(), 0);
          common::Mean(&cpu, vector, &expected);
          before = common::metal::GetStatistics();
          common::Mean(&gpu, vector, &actual);
          after = common::metal::GetStatistics();
          Check(after.mean_dispatches - before.mean_dispatches ==
                    uint64_t(rows != 0),
                "Strided scalar Mean did not dispatch on Metal");
          Close(actual(0), expected(0));

          if (mode) {
            if (!rows || mode == 2) {
              int rejected = 0;
              for (auto ctx : {&cpu, &gpu}) {
                try {
                  common::WeightedSampleMean(ctx, values, weights, &actual);
                } catch (std::exception const &) {
                  ++rejected;
                }
              }
              Check(rejected == 2, "Zero-total-weight API error not preserved");
            } else {
              common::WeightedSampleMean(&cpu, values, weights, &expected);
              before = common::metal::GetStatistics();
              common::WeightedSampleMean(&gpu, values, weights, &actual);
              after = common::metal::GetStatistics();
              Check(after.mean_dispatches - before.mean_dispatches == 1,
                    "WeightedSampleMean did not dispatch on Metal");
              for (size_t t = 0; t < targets; ++t)
                Close(actual(t), expected(t));
            }
          }
          ++cases;
        }
    printf("PASS %zu production median/mean API cases, including empty and "
           "strided inputs\n",
           cases);
    CheckReductions(&cpu, &gpu);
    return 0;
  } catch (std::exception const &e) {
    fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
