// CUDA-derived scalar quantizer/Philox oracles, run locally without CUDA hardware.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_reduction.h"
#include "../xgboost/src/data/gradient_index.h"
#include "../xgboost/src/tree/hist/sampler.h"
#include "cuda_philox_reference.h"
using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, checked = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Same(GradientPairPrecise actual, GradientPairPrecise expected) {
  if (actual.GetGrad() != expected.GetGrad() || actual.GetHess() != expected.GetHess()) {
    std::fprintf(stderr, "actual %.17g %.17g; expected %.17g %.17g\n", actual.GetGrad(),
                 actual.GetHess(), expected.GetGrad(), expected.GetHess());
    throw std::runtime_error("Integer root reduction differs from CUDA-formula oracle");
  }
}
int Exponent(linalg::MatrixView<GradientPair const> input, std::size_t t, bool hessian) {
  double positive = 0, negative = 0;
  for (std::size_t r = 0; r < input.Shape(0); ++r) {
    double value = hessian ? input(r, t).GetHess() : input(r, t).GetGrad();
    if (value >= 0)
      positive += value;
    else
      negative -= value;
  }
  double delta = std::max(positive, negative) /
                 (1. - 2. * input.Shape(0) * std::numeric_limits<double>::epsilon());
  int exp = 0;
  std::frexp(delta, &exp);
  return 62 - exp;
}
GradientPairPrecise Oracle(linalg::MatrixView<GradientPair const> input, std::size_t target,
                           float probability, std::uint32_t seed,
                           GradientPairInt64* integers = nullptr) {
  auto ge = Exponent(input, target, false), he = Exponent(input, target, true);
  std::int64_t gs = 0, hs = 0;
  for (std::size_t r = 0; r < input.Shape(0); ++r) {
    if (!cuda_reference::Selected(seed, r, probability)) continue;
    auto value = input(r, target);
    auto g = static_cast<std::int64_t>(std::ldexp(double(value.GetGrad()), ge));
    auto h = static_cast<std::int64_t>(std::ldexp(double(value.GetHess()), he));
    if (value.GetHess() > 0 && !h) h = 1;
    gs += g;
    hs += h;
  }
  if (integers) *integers = {gs, hs};
  return {std::ldexp(double(gs), -ge), std::ldexp(double(hs), -he)};
}
void Run(std::size_t rows, std::size_t targets, unsigned layout, float p, std::uint32_t seed) {
  current = "rows=" + std::to_string(rows) + " targets=" + std::to_string(targets) +
            " layout=" + std::to_string(layout) + " p=" + std::to_string(p) +
            " seed=" + std::to_string(seed);
  std::size_t a = targets + 2, b = 1;
  if (layout == 1) {
    a = 1;
    b = rows + 3;
  }
  if (layout == 2) {
    a = 2 * targets + 1;
    b = 2;
  }
  if (layout == 3) {
    a = 0;
    b = 1;
  }
  std::size_t size = rows ? (rows - 1) * a + (targets - 1) * b + 1 : 0;
  std::vector<GradientPair> source(size + 8, {71, -19});
  linalg::MatrixView<GradientPair> input{
      {source.data() + 4, size}, {rows, targets}, {a, b}, DeviceOrd::CPU()};
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t t = 0; t < targets; ++t) {
      int exponent = t == 0 ? -80 : t == 1 ? 0 : 100;
      float g = std::ldexp(float(int(r % 11) - 5), exponent - 3);
      float h = std::ldexp(1.f, int(r % 3) - 1);
      if (r == 0) g = std::ldexp(1.f, exponent + 54);
      if (r == 3) g = -std::ldexp(1.f, exponent + 54);
      // Never overflow FP32 for the high-exponent target.
      if (!std::isfinite(g)) g = std::ldexp(1.f, exponent) * (r == 0 ? 1.f : -1.f);
      if (r == 0) h = 0x1p65f;
      if (r == 1) h = std::numeric_limits<float>::denorm_min();
      input(r, t) = {g, h};
    }
  auto saved = source;
  std::vector<GradientPairPrecise> expected(targets), result(targets * 2 + 4, {13, 17});
  std::vector<GradientPairInt64> expected_integer(targets), exact(targets + 2, {19, 23});
  for (std::size_t t = 0; t < targets; ++t)
    expected[t] = Oracle(input, t, p, seed, &expected_integer[t]);
  auto before = metal::GetStatistics();
  auto state = metal::MakeQuantizedGradients(input);
  Check(state->Rows() == rows && state->Targets() == targets, "State shape changed");
  state->SampleUniform(p, seed);
  linalg::VectorView<GradientPairPrecise> out{
      {result.data() + 2, targets * 2}, {targets}, {std::size_t{2}}, DeviceOrd::CPU()};
  state->Sum(out, {exact.data() + 1, targets});
  for (std::size_t t = 0; t < targets; ++t) Same(out(t), expected[t]);
  Check(std::equal(expected_integer.begin(), expected_integer.end(), exact.begin() + 1),
        "Exact root integers differ from CUDA-formula sums");
  Check(exact.front() == GradientPairInt64{19, 23} && exact.back() == GradientPairInt64{19, 23},
        "Exact root output guards changed");
  Same(result.front(), {13, 17});
  Same(result.back(), {13, 17});
  for (std::size_t t = 0; t < targets; ++t) Same(result[3 + 2 * t], {13, 17});
  Check(!std::memcmp(source.data(), saved.data(), source.size() * sizeof(GradientPair)),
        "Quantization or integer sampling mutated source storage");
  // Replay has a different target count/layout. It must use the same ROW mask,
  // preserve arbitrary input bits for selected rows, and leave padding alone.
  auto replay_targets = targets + 2;
  auto replay_size = rows ? (rows - 1) * (replay_targets + 2) + replay_targets : 0;
  std::vector<GradientPair> replay(replay_size + 8, {23, -41});
  linalg::MatrixView<GradientPair> rv{{replay.data() + 4, replay_size},
                                      {rows, replay_targets},
                                      {replay_targets + 2, std::size_t{1}},
                                      DeviceOrd::CPU()};
  auto replay_expected = replay;
  for (std::size_t r = 0; r < rows; ++r)
    if (!cuda_reference::Selected(seed, r, p))
      for (std::size_t t = 0; t < replay_targets; ++t)
        replay_expected[4 + r * (replay_targets + 2) + t] = GradientPair{};
  state->ApplySampling(rv);
  state->ApplySampling(rv);
  Check(!std::memcmp(replay.data(), replay_expected.data(), replay.size() * sizeof(GradientPair)),
        "Uniform replay/guards differ from CUDA Philox row decisions");
  // Destroy the original gradients. Integer state remains authoritative.
  std::fill(source.begin(), source.end(), GradientPair{123, 456});
  state->Sum(out);
  for (std::size_t t = 0; t < targets; ++t) Same(out(t), expected[t]);
  auto after = metal::GetStatistics();
  Check(after.quantization_dispatches - before.quantization_dispatches == targets,
        "Shared state re-quantized sampled floats");
  Check(after.quantized_gradient_sum_dispatches - before.quantized_gradient_sum_dispatches == 2,
        "Integer roots not dispatched");
  ++cases;
  checked += rows * targets;
}
void SharedHistogram() {
  Context ctx;
  ctx.Init(Args{{"device", "mps"}, {"nthread", "1"}});
  SparsePage page;
  page.offset.HostVector() = {0, 1, 2, 3, 4};
  page.data.HostVector() = {{0, 0.f}, {0, 0.f}, {0, 0.f}, {0, 0.f}};
  common::HistogramCuts cuts{1};
  cuts.cut_ptrs_.HostVector() = {0, 1};
  cuts.cut_values_.HostVector() = {1.f};
  GHistIndexMatrix gidx{&ctx, page, {}, std::move(cuts), 2, true, .2};
  std::vector<GradientPair> raw{{0x1p60f, 0}, {64, 1}, {64, 1}, {-0x1p60f, 0}};
  auto input = linalg::MakeTensorView(&ctx, raw, 4, 1);
  auto state = metal::MakeQuantizedGradients(input);
  auto histogram = metal::MakeHistogram();
  histogram->BindQuantized(state, 0);
  auto cache = metal::MakeHistogramBuffer();
  cache->Resize(1);
  common::RowSetCollection rows;
  *rows.Data() = {0, 1, 2, 3};
  rows.Init();
  auto before = metal::GetStatistics();
  std::fill(raw.begin(), raw.end(), GradientPair{777, 888});
  state.reset();  // The histogram retains ownership across its caller's lifetime.
  histogram->Build(0, gidx, rows, {0}, raw, {cache->Data()}, cache.get());
  Same(cache->Data()[0], {128, 2});
  Same(metal::SumHistogram(cache.get(), 0, 1), {128, 2});
  Check(metal::GetStatistics().quantization_dispatches == before.quantization_dispatches,
        "Bound histogram re-quantized caller's float data");
}
void SamplerLifecycle() {
  for (std::size_t rows : {0, 1, 257})
    for (float probability : {0.f, .7f, 1.f}) {
      Context ctx;
      ctx.Init(Args{{"device", "mps"}, {"nthread", "3"}});
      ctx.Rng().seed(79);
      RandomEngine expected_rng{79};
      tree::TrainParam param;
      param.subsample = probability;
      param.sampling_method = tree::TrainParam::kUniform;
      tree::cpu_impl::Sampler sampler{param};
      for (unsigned tree = 0; tree < 3; ++tree) {
        auto seed = probability < 1 ? expected_rng() : 0;
        auto gradients = linalg::Empty<GradientPair>(&ctx, rows, 1);
        auto values = gradients.HostView().Values();
        std::fill(values.begin(), values.end(), GradientPair{3, 1});
        sampler.Sample(&ctx, gradients.HostView());
        Check(sampler.Quantized() != nullptr, "Missing per-tree state");
        GradientPairPrecise sum;
        sampler.Quantized()->Sum(linalg::MakeVec(&sum, 1));
        double selected = 0;
        for (std::size_t r = 0; r < rows; ++r)
          selected += cuda_reference::Selected(seed, r, probability);
        Same(sum, {selected * 3, selected});
      }
      Check(ctx.Rng() == expected_rng, "Wrong seed consumption for forest/empty/no-op sampling");
    }
}
int main() {
  try {
    for (std::size_t rows : {0, 1, 255, 256, 257, 1027, 65537})
      for (std::size_t targets : {1, 3})
        for (unsigned layout = 0; layout < 4; ++layout)
          for (float p : {0.f, .25f, .7f, 1.f}) Run(rows, targets, layout, p, 0xabc123u);
    for (std::uint32_t seed : {0u, 1u, 0xffffffffu}) Run(1027, 3, 0, .5f, seed);
    // <= boundary: a row whose draw equals the subsampling rate is retained.
    Run(257, 3, 0, cuda_reference::Weight(17, 73), 17);
    SharedHistogram();
    SamplerLifecycle();
    std::printf(
        "PASS: %zu shared-state cases / %zu pairs, exact CUDA-formula sums, replay, ownership, RNG "
        "lifecycle\n",
        cases, checked);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s: %s\n", current.c_str(), e.what());
    return 1;
  }
}
