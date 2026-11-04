// Production column sampling compared with upstream CPU and libc++ draws.
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
#include "../xgboost/src/common/random.h"
#include "xgboost/context.h"

using namespace xgboost;
using common::Span;
std::string current;
std::size_t cases = 0, checked_features = 0, checked_uniforms = 0;
std::uint32_t max_key_ulps = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
std::uint32_t Bits(float x) {
  std::uint32_t bits;
  std::memcpy(&bits, &x, 4);
  return bits;
}

void Case(std::uint32_t n, std::uint32_t count, std::uint32_t seed, int mode) {
  current = "n=" + std::to_string(n) + " count=" + std::to_string(count) +
            " seed=" + std::to_string(seed) + " mode=" + std::to_string(mode);
  std::vector<bst_feature_t> features(n);
  for (std::uint32_t i = 0; i < n; ++i) features[i] = i * 3 + 1;
  // Sampling accepts unordered feature IDs; final order must still be numerical.
  std::reverse(features.begin(), features.end());
  if (mode == 0 && n > 2) {
    features[0] = 0xffffffffu;
    features[1] = 0x80000001u;
  }
  std::vector<float> weights(mode ? 3 * n + 1 : 0, 1);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    if (mode == 2) weights[i] = float(1 + i % 19) / 17;
    if (mode == 3) weights[i] = std::ldexp(float(1 + i % 13), int(i % 251) - 125);
    if (mode == 4) weights[i] = i % 2 ? 0 : std::numeric_limits<float>::denorm_min();
    if (mode == 5) weights[i] = INFINITY;  // All keys tie at negative zero.
  }
  if (mode == 6) {
    // Every CPU score is exactly -1 unless the clamped floor applies. A single
    // logf ULP can reorder these deliberately tied feature priorities.
    RandomEngine draws(seed);
    std::uniform_real_distribution<float> uniform;
    for (std::uint32_t i = 0; i < n; ++i) weights[features[i]] = -std::log(uniform(draws));
  }
  std::vector<float> cpu_weights(n), gpu_uniforms(mode ? n : 0), gpu_keys(mode ? n : 0);
  std::vector<bst_feature_t> expected;
  RandomEngine rng(seed);
  Context ctx;
  ctx.Init({{"device", "cpu"}, {"nthread", "2"}});
  if (mode) {
    for (std::size_t i = 0; i < n; ++i) cpu_weights[i] = weights[features[i]];
    expected = common::WeightedSamplingWithoutReplacement(&ctx, &rng, features, cpu_weights, count);
  } else {
    expected = features;
    std::shuffle(expected.begin(), expected.end(), rng);
    expected.resize(count);
  }
  std::sort(expected.begin(), expected.end());
  std::vector<bst_feature_t> output(count + 4, 0xbadcafeu);
  auto saved_features = features;
  auto saved_weights = weights;
  auto before = common::metal::GetStatistics();
  common::metal::SampleFeatures(features, weights, seed, {output.data() + 2, count}, gpu_uniforms,
                                gpu_keys);
  auto after = common::metal::GetStatistics();
  Check(after.column_sampling_dispatches == before.column_sampling_dispatches + (count != 0),
        "Missing column sampling GPU dispatch");
  Check(features == saved_features && weights == saved_weights, "Inputs changed");
  Check(output[0] == 0xbadcafeu && output[1] == 0xbadcafeu && output[count + 2] == 0xbadcafeu &&
            output[count + 3] == 0xbadcafeu,
        "Output guards changed");
  Check(std::equal(expected.begin(), expected.end(), output.begin() + 2),
        "Feature selection differs from production CPU");
  if (mode && count) {
    RandomEngine reference(seed);
    std::uniform_real_distribution<float> uniform;
    for (std::size_t i = 0; i < n; ++i) {
      float u = uniform(reference);
      Check(Bits(u) == Bits(gpu_uniforms[i]), "GPU random draw differs from libc++");
      float key = std::log(u) / std::max(cpu_weights[i], kRtEps);
      auto a = Bits(key), b = Bits(gpu_keys[i]);
      auto ulps = a > b ? a - b : b - a;
      max_key_ulps = std::max(max_key_ulps, ulps);
      Check(ulps == 0, "Weighted key differs from CPU rounding");
      ++checked_uniforms;
    }
  }
  ++cases;
  checked_features += count;
  if (cases % 30 == 0) {
    std::printf("Checked %zu column cases / %zu selected IDs / %zu exact random draws\n", cases,
                checked_features, checked_uniforms);
    std::fflush(stdout);
  }
}

void Hierarchy(std::uint32_t n, float tree, float level, float node, bool weighted) {
  current = "hierarchy=" + std::to_string(n);
  Context cpu, gpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  gpu.Init({{"device", "mps"}, {"nthread", "2"}});
  cpu.Rng().seed(8721);
  gpu.Rng().seed(8721);
  HostDeviceVector<float> weights;
  if (weighted) {
    weights.Resize(n);
    for (std::size_t i = 0; i < n; ++i) weights.HostVector()[i] = float(1 + i % 7) / 3;
  }
  common::ColumnSampler a, b;
  auto before = common::metal::GetStatistics();
  for (int t = 0; t < 2; ++t) {
    a.Init(&cpu, n, weights, node, level, tree);
    b.Init(&gpu, n, weights, node, level, tree);
    for (auto depth : {0, 1, 1, 2, 2, 2, 0}) {
      auto expected = a.GetFeatureSet(&cpu, depth);
      auto actual = b.GetFeatureSet(&gpu, depth);
      Check(expected->ConstHostVector() == actual->ConstHostVector(),
            "Production ColumnSampler tree/level/node selection differs");
      checked_features += actual->Size();
    }
  }
  auto after = common::metal::GetStatistics();
  Check(after.column_init_dispatches == before.column_init_dispatches + 2, "Missing GPU iota");
  auto per_tree = int(tree != 1) + 3 * int(level != 1) + 7 * int(node != 1);
  Check(after.column_sampling_dispatches == before.column_sampling_dispatches + 2 * per_tree,
        "Wrong sampling/cached-level dispatch count");
  Check(cpu.Rng()() == gpu.Rng()(), "Parent RNG consumption changed");
  ++cases;
}

int main() {
  try {
    for (std::uint32_t n : {1u, 2u, 3u, 31u, 32u, 33u, 255u, 256u, 257u, 623u, 624u, 625u, 4095u,
                            4096u, 4097u, 65537u}) {
      for (int mode = 0; mode < 7; ++mode) {
        Case(n, 1, 0, mode);
        Case(n, std::max(1u, n / 3), 7, mode);
        Case(n, n, 0xffffffffu, mode);
      }
    }
    for (std::uint32_t seed = 1; seed <= 48; ++seed) {
      Case(1031, 517, seed, 0);
      Case(1031, 517, seed, 2);
    }
    for (int mode : {0, 2, 3, 6}) Case(1000003, 500003, 9852, mode);
    for (std::uint32_t n : {3u, 257u, 4097u})
      for (bool weighted : {false, true})
        for (auto rates : {std::vector<float>{1, 1, 1},
                           {0.7f, 1, 1},
                           {1, 0.7f, 1},
                           {1, 1, 0.7f},
                           {0.9f, 0.7f, 0.5f}})
          Hierarchy(n, rates[0], rates[1], rates[2], weighted);
    Case(0, 0, 7, 0);
    std::vector<bst_feature_t> init(65539, 0xbadcafeu);
    common::metal::InitFeatureSet({init.data() + 1, 65537});
    for (std::size_t i = 0; i < 65537; ++i) Check(init[i + 1] == i, "GPU iota differs");
    Check(init.front() == 0xbadcafeu && init.back() == 0xbadcafeu, "Iota guards changed");
    std::vector<bst_feature_t> invalid{0, 8}, result(1);
    std::vector<float> weight(2, 1);
    bool rejected = false;
    try {
      common::metal::SampleFeatures(invalid, weight, 0, result);
    } catch (std::exception const&) {
      rejected = true;
    }
    Check(rejected, "Out-of-bounds feature weight index accepted");
    std::printf(
        "PASS %zu column cases / %zu selected IDs / %zu exact random draws; "
        "maximum weighted-key difference %u FP32 ULP; hierarchy, RNG, guards and bounds pass\n",
        cases, checked_features, checked_uniforms, max_key_ulps);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), error.what());
    return 1;
  }
}
