// Production kernels against CUDA-source terms/residuals and FP32 error bounds.
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
std::size_t cases = 0, sums = 0, exact_sums = 0, residuals = 0, source_pairs = 0;
double max_error_fraction = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
// An independent high-precision sum of the FP32 source terms. Exact comparisons
// remain required whenever all possible partial sums fit the dyadic FP32 lattice.
// Other cases use the forward-error bound gamma_d * sum(abs(terms)), with d the
// maximum number of FP32 additions along a path in the balanced reduction.
struct ReferenceSum {
  double value{0}, magnitude{0};
  int quantum{10000};
  void Add(float term) {
    value += term;
    magnitude += std::abs(double(term));
    std::uint32_t bits;
    std::memcpy(&bits, &term, 4);
    bits &= 0x7fffffffu;
    if (bits && bits < 0x7f800000u) {
      auto exponent = bits >> 23;
      auto mantissa = (bits & 0x7fffffu) | (exponent ? 0x800000u : 0);
      int power = exponent ? int(exponent) - 127 - 23 : -149;
      quantum = std::min(quantum, power + __builtin_ctz(mantissa));
    }
  }
  void Compare(float actual, std::size_t count) const {
    ++sums;
    if (!std::isfinite(value)) {
      Check(std::isnan(value) ? std::isnan(actual) : actual == value, "Nonfinite sum differs");
      ++exact_sums;
      return;
    }
    auto exact = magnitude == 0 || std::ldexp(magnitude, -quantum) <= 0x1p24;
    if (exact) {
      Check(actual == value, "Exactly representable FP32 sum differs");
      ++exact_sums;
      return;
    }
    unsigned depth = 0;
    do {
      depth += 8;
      count = (count + 255) / 256;
    } while (count > 1);
    double unit = 0x1p-24, gamma = depth * unit / (1 - depth * unit);
    double bound = gamma * magnitude, error = std::abs(double(actual) - value);
    max_error_fraction = std::max(max_error_fraction, error / bound);
    if (!(error <= bound)) {
      std::printf("FP32 sum %.9g high-precision %.17g error %.17g bound %.17g\n", actual, value,
                  error, bound);
      Check(false, "FP32 reduction exceeded its rounding-error bound");
    }
  }
};
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
        ReferenceSum expected_g, expected_h;
        for (std::size_t i = 0; i < count; ++i) {
          auto row = feature < 0 ? i : entries[begin + i].index;
          auto pair = reference[row * groups + group];
          float g = pair.GetGrad(), h = pair.GetHess();
          if (feature >= 0) {
            float value = entries[begin + i].fvalue;
            g = g * value;
            h = (h * value) * value;
          }
          expected_g.Add(g);
          expected_h.Add(h);
        }
        auto before = metal::GetStatistics();
        auto actual = state->Gradient(feature, group);
        auto after = metal::GetStatistics();
        Check(after.linear_gradient_dispatches - before.linear_gradient_dispatches == bool(count),
              "Gradient dispatch count differs");
        expected_g.Compare(actual.GetGrad(), count);
        expected_h.Compare(actual.GetHess(), count);
        if (feature >= 0 || delta != 0) {
          for (std::size_t i = 0; i < count; ++i) {
            auto row = feature < 0 ? i : entries[begin + i].index;
            auto& pair = reference[row * groups + group];
            float product = pair.GetHess() * delta;
            if (feature >= 0) product = product * entries[begin + i].fvalue;
            pair += GradientPair{product, 0};
          }
        }
        state->UpdateResidual(feature, group, delta);
        auto updated = metal::GetStatistics();
        Check(updated.linear_residual_dispatches - after.linear_residual_dispatches ==
                  bool(count && (feature >= 0 || delta != 0)),
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
void CudaDifferences() {
  current = "CUDA source differences";
  std::vector<bst_idx_t> offsets{0, 1};
  std::vector<Entry> entries{{0, 3}};
  std::vector<GradientPair> gradients{{7, -2}};
  auto state = metal::MakeLinearState(1, 1, offsets, entries, gradients);
  auto bias = state->Gradient(-1, 0), feature = state->Gradient(0, 0);
  Check(bias.GetGrad() == 7 && bias.GetHess() == -2 && feature.GetGrad() == 21 &&
            feature.GetHess() == -18,
        "CUDA reductions must include negative Hessians");
  state->UpdateResidual(-1, 0, .5f);
  state->CopyGradients(gradients);
  Check(gradients[0] == GradientPair(6, -2), "CUDA bias residual must include negative Hessians");
  state->UpdateResidual(0, 0, .5f);
  state->CopyGradients(gradients);
  Check(gradients[0] == GradientPair(3, -2),
        "CUDA feature residual must include negative Hessians");
  auto from_bits = [](std::uint32_t bits) {
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
  };
  float h = from_bits(0x401758e3), step = from_bits(0x3fd4c3f0), x = from_bits(0x3f807d7e);
  float cuda = (h * step) * x, cpu = (h * x) * step;
  Check(cuda != cpu, "Residual-order fixture must distinguish CPU from CUDA");
  entries[0].fvalue = x;
  gradients[0] = GradientPair(0, h);
  state = metal::MakeLinearState(1, 1, offsets, entries, gradients);
  state->UpdateResidual(0, 0, step);
  state->CopyGradients(gradients);
  SameFloat(gradients[0].GetGrad(), cuda);
  // CUDA returns an FP32 pair, including overflow; the former binary64 sum did not.
  gradients.assign(2, GradientPair(std::numeric_limits<float>::max(), 1));
  auto overflow = metal::MakeLinearState(2, 1, {offsets.data(), 1}, {}, gradients);
  Check(std::isinf(overflow->Gradient(-1, 0).GetGrad()), "Reduction must retain FP32 overflow");
  gradients.resize(1);
  // A zero feature step still performs arithmetic in CUDA; a zero bias step skips it.
  entries[0].fvalue = INFINITY;
  gradients[0] = GradientPair(3, -0.f);
  state = metal::MakeLinearState(1, 1, offsets, entries, gradients);
  state->UpdateResidual(-1, 0, 0.f);
  state->CopyGradients(gradients);
  Check(gradients[0].GetGrad() == 3 && std::signbit(gradients[0].GetHess()),
        "Zero bias must bypass");
  auto before = metal::GetStatistics();
  state->UpdateResidual(0, 0, 0.f);
  state->CopyGradients(gradients);
  Check(std::isnan(gradients[0].GetGrad()) && !std::signbit(gradients[0].GetHess()),
        "Zero feature step must retain CUDA exceptional arithmetic");
  Check(metal::GetStatistics().linear_residual_dispatches == before.linear_residual_dispatches + 1,
        "Zero feature step did not dispatch");
}
int main(int argc, char**) {
  try {
    CudaDifferences();
    if (argc > 1) return 0;
    for (std::size_t rows : {0ul, 1ul, 31ul, 257ul, 4097ul, 65539ul})
      for (unsigned groups : {1u, 3u})
        for (unsigned mode = 0; mode < 4; ++mode) Run(rows, groups, mode);
    Run(1000003, 3, 0);
    Rejections();
    std::printf(
        "PASS %zu linear cases / %zu source pairs / %zu sum fields (%zu exact) / %zu exact "
        "residual fields; max FP32 forward-error fraction %.9g\n",
        cases, source_pairs, sums, exact_sums, residuals, max_error_fraction);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
