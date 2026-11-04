// Production Metal linear kernels against independent CPU reductions/residuals.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_linear.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, sums = 0, residuals = 0, source_pairs = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void SameDouble(double a, double b) {
  if (std::isnan(a) && std::isnan(b)) return;
  if (a != b) {
    std::printf("GPU sum %.17g CPU %.17g\n", a, b);
    Check(false, "Reduction differs");
  }
  ++sums;
}
void SameFloat(float a, float b) {
  ++residuals;
  if (std::isnan(a) && std::isnan(b)) return;
  if (std::memcmp(&a, &b, 4)) {
    std::printf("GPU residual %.9g CPU %.9g\n", a, b);
    Check(false, "Residual bits differ");
  }
}
void Run(std::size_t rows, unsigned groups, unsigned mode) {
  current = "rows=" + std::to_string(rows) + " groups=" + std::to_string(groups) +
            " mode=" + std::to_string(mode);
  std::vector<bst_idx_t> offsets{0};
  std::vector<Entry> entries;
  for (unsigned f = 0; f < 5; ++f) {
    for (std::size_t r = 0; r < rows; ++r) {
      if (f == 0 || (f == 1 && r % 2) || (f == 3 && r % 97)) continue;
      float value = float(int((r * 13 + f * 3) % 63) - 31) / 8;
      entries.emplace_back(r, value);
      if (f == 2 && r % 7 == 0) entries.emplace_back(r, -.5f);
    }
    offsets.push_back(entries.size());
  }
  std::vector<GradientPair> input(rows * groups + 2, GradientPair{-917, 918});
  std::mt19937 rng(137);
  for (std::size_t r = 0; r < rows; ++r) {
    for (unsigned g = 0; g < groups; ++g) {
      float grad = float(int(rng() % 1023) - 511) / 16;
      float hess = float(rng() % 127) / 8;
      if (r % 17 == 0) hess = -1;
      if (mode == 1) {
        grad = std::ldexp(grad, -140);
        hess = std::ldexp(hess, -140);
      }
      if (mode == 2) {
        grad = std::ldexp(grad, 100);
        hess = std::ldexp(hess, 100);
      }
      if (mode == 3) {
        float special[] = {0.f, -0.f, 1e-40f, -1e-40f, INFINITY, -INFINITY, NAN};
        grad = special[(r + g) % 7];
        hess = special[(r + g + 2) % 7];
      }
      input[1 + r * groups + g] = GradientPair{grad, hess};
    }
  }
  source_pairs += rows * groups;
  auto saved = input;
  auto saved_entries = entries;
  std::vector<GradientPair> reference(input.begin() + 1, input.end() - 1);
  auto state =
      metal::MakeLinearState(rows, groups, offsets, entries, {input.data() + 1, reference.size()});
  for (float delta : {0.f, .125f, -.5f}) {
    for (int feature = -1; feature < 5; ++feature) {
      for (unsigned group = 0; group < groups; ++group) {
        std::size_t begin = feature < 0 ? 0 : offsets[feature];
        std::size_t count = feature < 0 ? rows : offsets[feature + 1] - begin;
        double expected_g = 0, expected_h = 0;
        for (std::size_t i = 0; i < count; ++i) {
          auto row = feature < 0 ? i : entries[begin + i].index;
          auto pair = reference[row * groups + group];
          if (feature < 0 ? !(pair.GetHess() >= 0) : pair.GetHess() < 0) continue;
          float g = pair.GetGrad(), h = pair.GetHess();
          if (feature >= 0) {
            float value = entries[begin + i].fvalue;
            g = g * value;
            h = (h * value) * value;
          }
          expected_g += g;
          expected_h += h;
        }
        auto before = metal::GetStatistics();
        auto actual = state->Gradient(feature, group);
        auto after = metal::GetStatistics();
        Check(after.linear_gradient_dispatches - before.linear_gradient_dispatches == bool(count),
              "Gradient dispatch count differs");
        SameDouble(actual.GetGrad(), expected_g);
        SameDouble(actual.GetHess(), expected_h);
        if (delta != 0) {
          for (std::size_t i = 0; i < count; ++i) {
            auto row = feature < 0 ? i : entries[begin + i].index;
            auto& pair = reference[row * groups + group];
            if (pair.GetHess() < 0) continue;
            float product =
                feature < 0 ? pair.GetHess() : pair.GetHess() * entries[begin + i].fvalue;
            pair += GradientPair{product * delta, 0};
          }
        }
        state->UpdateResidual(feature, group, delta);
        auto updated = metal::GetStatistics();
        Check(updated.linear_residual_dispatches - after.linear_residual_dispatches ==
                  bool(count && delta != 0),
              "Residual dispatch count differs");
      }
      std::vector<GradientPair> output(reference.size() + 2, GradientPair{-999, 998});
      state->CopyGradients({output.data() + 1, reference.size()});
      for (std::size_t i = 0; i < reference.size(); ++i) {
        SameFloat(output[i + 1].GetGrad(), reference[i].GetGrad());
        SameFloat(output[i + 1].GetHess(), reference[i].GetHess());
      }
      Check(output.front() == GradientPair{-999, 998} && output.back() == GradientPair{-999, 998},
            "Output guards changed");
    }
  }
  Check(!std::memcmp(saved.data(), input.data(), input.size() * sizeof(GradientPair)),
        "Source changed");
  Check(entries.empty() ||
            !std::memcmp(saved_entries.data(), entries.data(), entries.size() * sizeof(Entry)),
        "CSC entries changed");
  ++cases;
  if (cases % 12 == 0 || rows > 1000000) {
    std::printf("Checked %zu linear cases / %zu source pairs / %zu residual fields\n", cases,
                source_pairs, residuals);
    std::fflush(stdout);
  }
}
void Rejections() {
  std::vector<bst_idx_t> offsets{0, 1};
  std::vector<Entry> entries{{0, 1}};
  std::vector<GradientPair> gradients{{1, 1}};
  auto state = metal::MakeLinearState(1, 1, offsets, entries, gradients);
  for (int feature : {-2, 1}) {
    bool failed = false;
    try {
      state->Gradient(feature, 0);
    } catch (std::exception const&) {
      failed = true;
    }
    Check(failed, "Invalid feature accepted");
  }
  entries[0].index = 1;
  bool failed = false;
  try {
    metal::MakeLinearState(1, 1, offsets, entries, gradients);
  } catch (std::exception const&) {
    failed = true;
  }
  Check(failed, "Invalid row index accepted");
}
int main() {
  try {
    for (std::size_t rows : {0ul, 1ul, 31ul, 257ul, 4097ul, 65539ul})
      for (unsigned groups : {1u, 3u})
        for (unsigned mode = 0; mode < 4; ++mode) Run(rows, groups, mode);
    Run(1000003, 3, 0);
    Rejections();
    std::printf(
        "PASS %zu linear cases / %zu source gradient pairs / %zu exact sum fields / %zu exact "
        "residual fields; duplicates, guards, exceptional values and errors\n",
        cases, source_pairs, sums, residuals);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
