// Exercise the production C++ statistics API, including empty and strided
// inputs.
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/stats.h"
#include "xgboost/context.h"
#include "xgboost/linalg.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    return 0;
  } catch (std::exception const &e) {
    fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
