// Compare actual Metal sampling against upstream CPU threshold/norm routines
// and the platform's standard-library random stream. Includes in-place/replay,
// strided storage, padding guards, underflow, overflow and large scans.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_sampling.h"
#include "../xgboost/src/tree/hist/sampler.h"
#include "xgboost/context.h"
#include "xgboost/linalg.h"

using namespace xgboost;
using xgboost::common::Span;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}
void Same(float actual, float expected, char const* field, size_t row) {
  if (Bits(actual) == Bits(expected) || (std::isnan(actual) && std::isnan(expected))) return;
  std::fprintf(stderr, "%s row %zu: actual %.9g (0x%08x), expected %.9g (0x%08x)\n", field, row,
               actual, Bits(actual), expected, Bits(expected));
  throw std::runtime_error("Metal sampling differs from CPU");
}

struct Storage {
  size_t rows, targets, stride0, stride1;
  std::vector<GradientPair> data;
  Storage(size_t n, size_t t, int layout) : rows{n}, targets{t} {
    stride0 = layout == 1 ? 1 : (layout == 2 ? t * 2 + 3 : t);
    stride1 = layout == 1 ? n + 3 : (layout == 2 ? 2 : 1);
    auto extent = n ? (n - 1) * stride0 + (t - 1) * stride1 + 1 : 0;
    data.resize(extent + 8, GradientPair{981, -237});
  }
  auto View() {
    return linalg::MatrixView<GradientPair>{Span{data.data() + 4, data.size() - 8},
                                            {rows, targets},
                                            {stride0, stride1},
                                            DeviceOrd::CPU()};
  }
};

size_t cases = 0, checked_rows = 0, checked_fields = 0;
std::string current_case;
void Case(size_t rows, size_t targets, int layout, int mode, size_t budget, uint64_t seed,
          uint32_t threads, bool replay) {
  current_case = "rows=" + std::to_string(rows) + " targets=" + std::to_string(targets) +
                 " layout=" + std::to_string(layout) + " mode=" + std::to_string(mode) +
                 " budget=" + std::to_string(budget) + " seed=" + std::to_string(seed) +
                 " threads=" + std::to_string(threads) + " replay=" + std::to_string(replay);
  Context cpu;
  cpu.Init({{"device", "cpu"}, {"nthread", std::to_string(threads)}});
  Storage source(rows, targets, layout), output(rows, replay ? 3 : targets, layout);
  auto split = source.View(), values = output.View();
  std::mt19937 random(1021 + mode);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t t = 0; t < targets; ++t) {
      float g = float(int(random() % 4095) - 2047) / 113;
      float h = float(random() % 4095) / 207;
      if (mode == 1) g = h = 0;
      if (mode == 2) {
        g = -2;
        h = 3;
      }
      if (mode == 3) {
        g = std::ldexp(g, -70);
        h = std::ldexp(h, -70);
      }
      if (mode == 4) {
        g = std::ldexp(g, int(random() % 111) - 80);
        h = std::ldexp(h, int(random() % 111) - 80);
      }
      if (mode == 5) {
        g = std::ldexp(g, 80);
        h = std::ldexp(h, 80);
      }
      if (mode == 6) {
        g = r % 2 ? -0.f : 0.f;
        h = t % 2 ? -0.f : 0.f;
      }
      if (mode == 7) {
        g = r % 2 ? std::ldexp(g, -70) : std::ldexp(g, -25);
        h = std::ldexp(h, int(random() % 130) - 75);
      }
      if (mode == 8) {
        auto make = [&] {
          uint32_t bits = ((random() % 150 + 45) << 23) | (random() & 0x7fffff);
          float v;
          std::memcpy(&v, &bits, 4);
          return v;
        };
        g = make();
        h = make();
      }
      split(r, t) = GradientPair{g, h};
    }
    for (size_t t = 0; t < values.Shape(1); ++t) {
      values(r, t) = replay ? GradientPair{float(int(random() % 2047) - 1023) / 17,
                                           float(random() % 1023) / 71}
                            : split(r, t);
    }
  }
  Storage expected = output;
  auto reference = expected.View();
  std::vector<float> sorted, norms, cdf(rows);
  float threshold = NAN;
  bool active = rows && budget < rows;
  if (active && budget) {
    norms = tree::cpu_impl::CalcRegAbsGrad(&cpu, split, &sorted);
    std::partial_sum(sorted.begin(), sorted.end() - 1, cdf.begin());
    threshold = tree::cpu_impl::CalculateThreshold(Span{sorted}, Span{cdf}, rows, budget);
  }
  if (active) {
    size_t chunk = rows / threads;
    for (uint32_t thread = 0; thread < threads; ++thread) {
      size_t begin = thread * chunk, end = thread == threads - 1 ? rows : begin + chunk;
      auto displaced = tree::RandomReplace::SimpleSkip(begin, seed, tree::RandomReplace::kBase,
                                                       tree::RandomReplace::kMod);
      tree::RandomReplace::EngineT engine(displaced);
      std::uniform_real_distribution<float> draw;
      for (size_t r = begin; r < end; ++r) {
        float p = budget ? tree::SamplingProbability(threshold, norms[r]) : 0;
        float random_value = draw(engine);
        bool selected = p >= 1 || (p > 0 && random_value <= p);
        for (size_t t = 0; t < reference.Shape(1); ++t)
          reference(r, t) = selected ? tree::RescaleGrad(p, reference(r, t)) : GradientPair{};
      }
    }
  }
  std::vector<float> actual_norms(active && budget ? rows : 0);
  float actual_threshold = NAN;
  auto before = common::metal::GetStatistics();
  // Non-replay calls deliberately exercise the production in-place alias.
  common::metal::GradientSample(replay ? split : values, values, budget, seed, threads,
                                Span{actual_norms}, &actual_threshold);
  auto after = common::metal::GetStatistics();
  Check(
      after.gradient_sampling_dispatches - before.gradient_sampling_dispatches == uint64_t(active),
      "Gradient sampling did not execute on GPU");
  if (active && budget) {
    for (size_t r = 0; r < rows; ++r) Same(actual_norms[r], norms[r], "norm", r);
    Same(actual_threshold, threshold, "threshold", 0);
  }
  for (size_t i = 0; i < output.data.size(); ++i) {
    Same(output.data[i].GetGrad(), expected.data[i].GetGrad(), "gradient/guard", i);
    Same(output.data[i].GetHess(), expected.data[i].GetHess(), "hessian/guard", i);
  }
  checked_rows += rows;
  checked_fields += 2 * output.data.size();
  ++cases;
  if (cases % 50 == 0 || rows > 4097) {
    std::printf("Checked %zu cases / %zu rows / %zu gradient and guard fields\n", cases,
                checked_rows, checked_fields);
    std::fflush(stdout);
  }
}

void CheckSamplerAPI() {
  for (int threads : {1, 3})
    for (float fraction : {0.f, 0.6f, 1.f}) {
      Context cpu, gpu;
      cpu.Init({{"device", "cpu"}, {"nthread", std::to_string(threads)}});
      gpu.Init({{"device", "mps"}, {"nthread", std::to_string(threads)}});
      cpu.Rng().seed(701);
      gpu.Rng().seed(701);
      tree::TrainParam params;
      params.Init(
          Args{{"sampling_method", "gradient_based"}, {"subsample", std::to_string(fraction)}});
      tree::cpu_impl::Sampler a{params}, b{params};
      linalg::Matrix<GradientPair> original({1031, 1}, DeviceOrd::CPU()),
          expected({1031, 1}, DeviceOrd::CPU()), actual({1031, 1}, DeviceOrd::CPU());
      for (size_t i = 0; i < 1031; ++i) {
        auto value = GradientPair{float(int(i % 31) - 15) / 7, float(i % 17) / 3};
        original.HostView()(i, 0) = expected.HostView()(i, 0) = actual.HostView()(i, 0) = value;
      }
      a.Sample(&cpu, expected.HostView());
      b.Sample(&gpu, actual.HostView());
      Check(std::memcmp(expected.HostView().Values().data(), actual.HostView().Values().data(),
                        1031 * sizeof(GradientPair)) == 0,
            "Sampler integration differs");
      linalg::Matrix<GradientPair> ve({1031, 3}, DeviceOrd::CPU()), va({1031, 3}, DeviceOrd::CPU());
      for (size_t i = 0; i < 1031; ++i)
        for (size_t t = 0; t < 3; ++t)
          ve.HostView()(i, t) = va.HostView()(i, t) =
              GradientPair{float(i % 21) - 10, float(t + 1)};
      a.ApplySampling(&cpu, original.HostView(), &ve);
      b.ApplySampling(&gpu, original.HostView(), &va);
      Check(std::memcmp(ve.HostView().Values().data(), va.HostView().Values().data(),
                        1031 * 3 * sizeof(GradientPair)) == 0,
            "Reduced-gradient replay differs");
      Check(cpu.Rng()() == gpu.Rng()(), "Host random-engine state differs");
    }
  std::puts("PASS 6 production Sampler calls and reduced-gradient replay pairs");
}

int main() {
  try {
    uint64_t seeds[] = {0, 1, 9939, uint64_t{1} << 63, (uint64_t{1} << 63) - 1, UINT64_MAX};
    for (size_t n : {0, 1, 2, 31, 257, 4097})
      for (size_t targets : {1, 3, 17})
        for (int layout = 0; layout < 3; ++layout)
          for (int mode = 0; mode < 9; ++mode) {
            size_t budgets[] = {0, size_t(n * 0.37f), n ? n - 1 : 0, n};
            auto id = cases;
            Case(n, targets, layout, mode, budgets[id % 4], seeds[id % 6],
                 id % 3 == 0 ? 1 : (id % 3 == 1 ? 3 : 64), bool(id % 2));
          }
    for (size_t n : {65537, 1000003})
      for (size_t t : {1, 3})
        for (int mode : {0, 3, 4})
          Case(n, t, t == 1 ? 0 : 1, mode, size_t(n * 0.4f), 9939, 3, true);
    CheckSamplerAPI();
    std::printf("PASS %zu exact sampling cases / %zu rows / %zu gradient and guard fields\n", cases,
                checked_rows, checked_fields);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL after %zu cases: %s (%s)\n", cases, e.what(), current_case.c_str());
    return 1;
  }
}
