// Production initialization APIs versus CPU FP32 expressions and bitwise copies.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ops.h"

using namespace xgboost;
using xgboost::common::Span;
namespace metal = xgboost::common::metal;
std::size_t cases = 0, pairs = 0, copies = 0;
void Check(bool ok, char const* text) {
  if (!ok) throw std::runtime_error(text);
}
void Same(float actual, float expected) {
  if (std::isnan(actual) && std::isnan(expected)) return;
  if (std::memcmp(&actual, &expected, 4)) {
    std::printf("case %zu actual %.9g expected %.9g\n", cases, actual, expected);
    throw std::runtime_error("FP32 bits differ");
  }
}
void Case(std::size_t rows, std::size_t targets, int mode) {
  float tiny = std::numeric_limits<float>::denorm_min();
  std::vector<float> labels(rows), weights(rows), alpha(targets);
  std::mt19937 rng(291 + mode);
  float mean = mode == 1 ? tiny * 3 : mode == 2 ? 0x1p80f : mode == 5 ? NAN : .25f;
  for (std::size_t i = 0; i < rows; ++i) {
    labels[i] = float(int(rng() % 997) - 498) / 83;
    weights[i] = i % 11 ? float(rng() % 31) / 13 : 0;
    if (mode == 1) labels[i] = tiny * float(int(i % 17) - 8);
    if (mode == 2) {
      labels[i] = std::ldexp(labels[i], 80);
      weights[i] = std::ldexp(weights[i], 80);
    }
    if (mode == 3) labels[i] = mean;
    if (mode == 4) weights[i] = tiny * float(i % 31);
    if (mode == 5) labels[i] = i % 3 == 0 ? NAN : i % 3 == 1 ? INFINITY : -INFINITY;
  }
  std::vector<float> levels{0, tiny, 1e-30f, .1f, .5f, .9f, std::nextafter(1.f, 0.f), 1.f};
  for (std::size_t j = 0; j < targets; ++j) alpha[j] = levels[(j + mode) % levels.size()];
  auto saved_labels = labels, saved_weights = weights, saved_alpha = alpha;
  std::vector<GradientPair> output(rows * targets + 2, GradientPair{-917, 919});
  Span<float const> w = mode ? Span<float const>{weights} : Span<float const>{};
  auto before = metal::GetStatistics();
  metal::InitExpectileGradients(mean, labels, w, alpha, {output.data() + 1, rows * targets});
  auto after = metal::GetStatistics();
  Check(after.expectile_init_dispatches - before.expectile_init_dispatches == bool(rows),
        "Wrong expectile initialization dispatch count");
  Check(!rows || after.initialization_gpu_time_ns > before.initialization_gpu_time_ns,
        "Missing initialization GPU duration");
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < targets; ++j) {
      float diff = mean - labels[i];
      float scale = diff >= 0 ? 1 - alpha[j] : alpha[j];
      float weight = mode ? weights[i] : 1;
      float g = (scale * diff) * weight, h = scale * weight;
      auto actual = output[i * targets + j + 1];
      Same(actual.GetGrad(), g);
      Same(actual.GetHess(), h);
      ++pairs;
    }
  Same(output.front().GetGrad(), -917);
  Same(output.front().GetHess(), 919);
  Same(output.back().GetGrad(), -917);
  Same(output.back().GetHess(), 919);
  if (rows) {
    Check(!std::memcmp(labels.data(), saved_labels.data(), rows * 4), "Labels changed");
    Check(!std::memcmp(weights.data(), saved_weights.data(), rows * 4), "Weights changed");
  }
  Check(!std::memcmp(alpha.data(), saved_alpha.data(), targets * 4), "Levels changed");

  // Copies must preserve even NaN payloads, signed zeros and subnormal bits.
  std::vector<float> out(rows * targets + 2, -913);
  for (std::size_t j = 0; j < targets; ++j) {
    uint32_t bits = j % 3 ? uint32_t(rng()) : 0x80000000u;
    std::memcpy(&alpha[j], &bits, 4);
  }
  before = metal::GetStatistics();
  metal::BroadcastTargets(alpha, {out.data() + 1, rows * targets});
  after = metal::GetStatistics();
  Check(after.prediction_init_dispatches - before.prediction_init_dispatches == bool(rows),
        "Wrong broadcast dispatch count");
  for (std::size_t i = 0; i < rows * targets; ++i) {
    Check(!std::memcmp(&out[i + 1], &alpha[i % targets], 4), "Broadcast changed source bits");
    ++copies;
  }
  Same(out.front(), -913);
  Same(out.back(), -913);
  if (rows) {
    // Source aliases the first output row: input is snapshotted before execution.
    metal::BroadcastTargets({out.data() + 1, targets}, {out.data() + 1, rows * targets});
    for (std::size_t i = 0; i < rows * targets; ++i) {
      Check(!std::memcmp(&out[i + 1], &alpha[i % targets], 4), "Aliased broadcast differs");
      ++copies;
    }
    Same(out.front(), -913);
    Same(out.back(), -913);
  }
  if (++cases % 18 == 0) {
    std::printf("Checked %zu initialization cases / %zu gradient pairs / %zu copied floats\n",
                cases, pairs, copies);
    std::fflush(stdout);
  }
}
int main() {
  try {
    for (std::size_t n : {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, 4097u})
      for (std::size_t targets : {1u, 3u, 37u})
        for (int mode = 0; mode < 6; ++mode) Case(n, targets, mode);
    Case(1000003, 3, 0);
    Case(1000003, 3, 4);
    std::printf(
        "PASS %zu initialization cases / %zu exact gradient pairs / %zu bitwise copied floats\n",
        cases, pairs, copies);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
