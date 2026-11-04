// Production shared-state sampler checked against an independent CUDA-formula oracle.
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/tree/hist/sampler.h"
#include "cuda_gradient_reference.h"
using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, pairs = 0, invalid_cases = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Same(float a, float b, char const* field, std::size_t row) {
  if (!std::memcmp(&a, &b, 4) || (std::isnan(a) && std::isnan(b))) return;
  std::fprintf(stderr, "%s row %zu: %a != %a\n", field, row, a, b);
  throw std::runtime_error("CUDA-formula mismatch");
}
void Roots(std::shared_ptr<metal::QuantizedGradients> state,
           cuda_reference::GradientSampling const& ref) {
  std::vector<GradientPairPrecise> sums(ref.targets);
  state->Sum(linalg::MakeVec(sums.data(), sums.size()));
  for (std::size_t t = 0; t < ref.targets; ++t) {
    auto expected = ref.Sum(t);
    if (sums[t].GetGrad() != expected.first || sums[t].GetHess() != expected.second) {
      std::fprintf(stderr, "target %zu root (%a,%a) != (%a,%a)\n", t, sums[t].GetGrad(),
                   sums[t].GetHess(), expected.first, expected.second);
      throw std::runtime_error("Quantized root mismatch");
    }
  }
}
void Run(std::size_t n, std::size_t t, int layout, int mode, float fraction, std::uint32_t seed) {
  current = "rows=" + std::to_string(n) + " targets=" + std::to_string(t) +
            " layout=" + std::to_string(layout) + " mode=" + std::to_string(mode) +
            " fraction=" + std::to_string(fraction);
  std::size_t s0 = layout == 1 ? 1 : layout == 2 ? 2 * t + 1 : t + 2;
  std::size_t s1 = layout == 1 ? n + 3 : layout == 2 ? 2 : 1;
  if (layout == 3) s0 = 0;
  std::size_t extent = n ? (n - 1) * s0 + (t - 1) * s1 + 1 : 0;
  std::vector<GradientPair> raw(extent + 8, {37, -19});
  linalg::MatrixView<GradientPair> view{
      {raw.data() + 4, extent}, {n, t}, {s0, s1}, DeviceOrd::CPU()};
  std::mt19937 rng(1027);
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t c = 0; c < t; ++c) {
      float g = float(int(rng() % 4095) - 2047) / 113.f, h = float(rng() % 4095) / 207.f;
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
        g = std::ldexp(g, int(rng() % 110) - 80);
        h = std::ldexp(h, int(rng() % 110) - 80);
      }
      if (mode == 5) {
        g = std::ldexp(g, 54);
        h = std::ldexp(h, 54);
      }
      if (mode == 6) {
        g = std::numeric_limits<float>::denorm_min();
        h = g;
      }
      if (mode == 7) {
        g = r < 2 ? std::ldexp(r ? -1.f : 1.f, 60) : float(int(r % 7) - 3);
        h = r ? 0x1p-50f : 0x1p65f;
      }
      view(r, c) = {g, h};
    }
  std::vector<cuda_reference::Pair> source(n * t);
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t c = 0; c < t; ++c)
      source[r * t + c] = {view(r, c).GetGrad(), view(r, c).GetHess()};
  cuda_reference::GradientSampling ref(source, n, t, fraction, seed);
  auto original = raw;
  auto before = metal::GetStatistics();
  auto state = metal::MakeQuantizedGradients(view);
  std::vector<float> norms(n, -1);
  float threshold = -1;
  bool overflow = false;
  try {
    state->SampleGradient(fraction, seed, norms, &threshold);
  } catch (dmlc::Error const& e) {
    if (std::string(e.what()).find("rescaling exceeds int64 range") == std::string::npos) throw;
    overflow = true;
  }
  Check(overflow == ref.invalid_rescale, "Rescaling domain differs from host oracle");
  if (overflow) {
    ++invalid_cases;
    return;
  }
  if (fraction < 1) {
    Same(threshold, ref.threshold, "threshold", 0);
    for (std::size_t r = 0; r < n; ++r) Same(norms[r], ref.norms[r], "norm", r);
  }
  Roots(state, ref);
  Check(!std::memcmp(raw.data(), original.data(), raw.size() * sizeof(GradientPair)),
        "Source mutated");
  // Replay onto a different target count and a padded view, using saved split norms.
  std::size_t rt = t == 1 ? 3 : 1;
  std::vector<GradientPair> replay(n * (rt + 2) + 8, {23, -41}), expected = replay;
  linalg::MatrixView<GradientPair> rv{
      {replay.data() + 4, n * (rt + 2)}, {n, rt}, {rt + 2, std::size_t{1}}, DeviceOrd::CPU()};
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t c = 0; c < rt; ++c) {
      auto value = ref.Replay(r, {23, -41});
      expected[4 + r * (rt + 2) + c] = {value.g, value.h};
    }
  state->ApplySampling(rv);
  Check(!std::memcmp(replay.data(), expected.data(), replay.size() * sizeof(GradientPair)),
        "Float replay or guards differ");
  auto after = metal::GetStatistics();
  Check(after.quantization_dispatches - before.quantization_dispatches == t,
        "Requantized after sampling");
  Check(after.cuda_gradient_sampling_dispatches - before.cuda_gradient_sampling_dispatches ==
            std::uint64_t(n && fraction < 1),
        "Missing CUDA sampler dispatch");
  std::fill(raw.begin(), raw.end(), GradientPair{919, 313});
  Roots(state, ref);
  ++cases;
  pairs += n * t;
  if (cases % 25 == 0 || n > 4097) {
    std::printf("Checked %zu cases / %zu pairs\n", cases, pairs);
    std::fflush(stdout);
  }
}
void Lifecycle() {
  for (std::size_t n : {0, 1, 257})
    for (float fraction : {0.f, .6f, 1.f})
      for (int threads : {1, 3}) {
        Context ctx;
        ctx.Init({{"device", "mps"}, {"nthread", std::to_string(threads)}});
        ctx.Rng().seed(701);
        RandomEngine rng(701);
        tree::TrainParam param;
        param.subsample = fraction;
        param.sampling_method = tree::TrainParam::kGradientBased;
        tree::cpu_impl::Sampler sampler(param);
        for (int tree = 0; tree < 3; ++tree) {
          linalg::Matrix<GradientPair> actual({n, std::size_t{1}}, DeviceOrd::CPU());
          std::vector<cuda_reference::Pair> source(n);
          for (std::size_t r = 0; r < n; ++r) {
            source[r] = {float(int(r % 31) - 15) / 7, float(r % 17) / 3};
            actual.HostView()(r, 0) = {source[r].g, source[r].h};
          }
          auto seed = fraction < 1 ? rng() : 0;
          cuda_reference::GradientSampling ref(source, n, 1, fraction, seed);
          sampler.Sample(&ctx, actual.HostView());
          Roots(sampler.Quantized(), ref);
          for (std::size_t r = 0; r < n; ++r) {
            auto e = ref.Replay(r, source[r]);
            Same(actual.HostView()(r, 0).GetGrad(), e.g, "integrated g", r);
            Same(actual.HostView()(r, 0).GetHess(), e.h, "integrated h", r);
          }
          linalg::Matrix<GradientPair> values({n, std::size_t{3}}, DeviceOrd::CPU());
          auto storage = values.HostView().Values();
          std::fill(storage.begin(), storage.end(), GradientPair{23, 41});
          // Original split values intentionally unavailable: replay must use saved norms.
          sampler.ApplySampling(&ctx, actual.HostView(), &values);
          for (std::size_t r = 0; r < n; ++r)
            for (std::size_t t = 0; t < 3; ++t) {
              auto e = ref.Replay(r, {23, 41});
              Same(values.HostView()(r, t).GetGrad(), e.g, "reduced g", r);
              Same(values.HostView()(r, t).GetHess(), e.h, "reduced h", r);
            }
        }
        Check(ctx.Rng()() == rng(), "Seed consumption differs");
      }
  std::puts("PASS 54 production Sampler calls, integer roots, replay and seed lifetime");
}
void InvalidNorms() {
  for (float p : {0.f, .5f}) {
    std::vector<GradientPair> source(4, {0x1p100f, 1});
    auto state = metal::MakeQuantizedGradients(
        linalg::MakeTensorView(DeviceOrd::CPU(), common::Span{source}, 4, 1));
    bool failed = false;
    try {
      state->SampleGradient(p, 1);
    } catch (dmlc::Error const& e) {
      failed = std::string(e.what()).find("finite FP32 norms") != std::string::npos;
    }
    Check(failed == (p > 0), "Nonfinite norm conversion domain not checked");
  }
}
int main() {
  try {
    for (std::size_t n : {0, 1, 2, 255, 256, 257, 1027})
      for (std::size_t t : {1, 3})
        for (int mode = 0; mode < 8; ++mode) {
          float p[]{0, .001f, .37f, .999f, 1};
          Run(n, t, cases % 4, mode, p[cases % 5], std::uint32_t(cases * 7919));
        }
    for (std::size_t t : {16, 17, 31, 32, 33, 511, 512, 513, 1027})
      for (int mode : {0, 3, 4}) Run(33, t, mode % 3, mode, .6f, 793);
    for (std::size_t n : {65537, 1000003})
      for (std::size_t t : {1, 3}) Run(n, t, t == 1 ? 0 : 1, 0, .4f, 9939);
    Lifecycle();
    InvalidNorms();
    std::printf(
        "PASS %zu CUDA-formula gradient sampling cases / %zu pairs / %zu rejected undefined "
        "rescalings\n",
        cases, pairs, invalid_cases);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), e.what());
    return 1;
  }
}
