// Production column sampling compared with CUDA-source Philox and Thrust shuffle.
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
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_sampling.h"
#include "../xgboost/src/common/random.h"
#include "cuda_column_reference.h"
#include "xgboost/context.h"

using namespace xgboost;
using common::Span;
std::string current;
std::size_t cases = 0, checked_features = 0, checked_uniforms = 0;
std::size_t checked_permutations = 0;
std::uint32_t max_key_ulps = 0;
std::size_t resident_chains = 0;
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
    // Every source-reference score is exactly -1 unless the clamped floor applies. A single
    // logf ULP can reorder these deliberately tied feature priorities.
    for (std::uint32_t i = 0; i < n; ++i)
      weights[features[i]] = -std::log(cuda_reference::Weight(seed, i));
  }
  std::vector<float> cpu_weights(n), gpu_uniforms(mode ? n : 0), gpu_keys(mode ? n : 0);
  for (std::size_t i = 0; mode && i < n; ++i) cpu_weights[i] = weights[features[i]];
  auto expected = cuda_reference::SampleFeatures(features, weights, seed, count);
  std::vector<bst_feature_t> permutation(mode ? 0 : n + 2, 0xbadcafeu);
  Span<bst_feature_t> permutation_view =
      mode ? Span<bst_feature_t>{} : Span<bst_feature_t>{permutation.data() + 1, n};
  std::vector<bst_feature_t> output(count + 4, 0xbadcafeu);
  auto saved_features = features;
  auto saved_weights = weights;
  auto before = common::metal::GetStatistics();
  common::metal::SampleFeatures(features, weights, seed, {output.data() + 2, count}, gpu_uniforms,
                                gpu_keys, permutation_view);
  auto after = common::metal::GetStatistics();
  Check(after.column_sampling_dispatches == before.column_sampling_dispatches + (count != 0),
        "Missing column sampling GPU dispatch");
  Check(features == saved_features && weights == saved_weights, "Inputs changed");
  Check(output[0] == 0xbadcafeu && output[1] == 0xbadcafeu && output[count + 2] == 0xbadcafeu &&
            output[count + 3] == 0xbadcafeu,
        "Output guards changed");
  Check(std::equal(expected.begin(), expected.end(), output.begin() + 2),
        "Feature selection differs from CUDA-source reference");
  if (!mode) {
    Check(permutation.front() == 0xbadcafeu && permutation.back() == 0xbadcafeu,
          "Permutation guard changed");
    if (count) {
      auto expected_permutation = cuda_reference::Shuffle(n, seed);
      Check(std::equal(expected_permutation.begin(), expected_permutation.end(),
                       permutation_view.begin()),
            "Full GPU shuffle permutation differs from Thrust source");
      checked_permutations += n;
    }
  }
  if (mode && count) {
    for (std::size_t i = 0; i < n; ++i) {
      float u = cuda_reference::Weight(seed, i);
      Check(Bits(u) == Bits(gpu_uniforms[i]), "GPU random draw differs from CUDA Philox");
      float key = std::log(u) / std::max(cpu_weights[i], kRtEps);
      auto a = Bits(key), b = Bits(gpu_keys[i]);
      auto ulps = a > b ? a - b : b - a;
      max_key_ulps = std::max(max_key_ulps, ulps);
      Check(ulps == 0, "Weighted key differs from CUDA-source/Apple-log rounding");
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
  current = "hierarchy=" + std::to_string(n) + " weighted=" + std::to_string(weighted) +
            " rates=" + std::to_string(tree) + "/" + std::to_string(level) + "/" +
            std::to_string(node);
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
  RandomEngine reference_rng(8721);
  auto sample = [&](std::vector<bst_feature_t> const& features, float rate) {
    if (rate == 1.f) return features;
    auto count = std::max(1, static_cast<int>(rate * features.size()));
    return cuda_reference::SampleFeatures(features, weights.ConstHostVector(), reference_rng(),
                                          count);
  };
  auto before = common::metal::GetStatistics();
  for (int t = 0; t < 2; ++t) {
    a.Init(&cpu, n, weights, node, level, tree);
    b.Init(&gpu, n, weights, node, level, tree);
    std::vector<bst_feature_t> all(n);
    std::iota(all.begin(), all.end(), 0);
    auto tree_features = sample(all, tree);
    std::unordered_map<int, std::vector<bst_feature_t>> levels;
    for (auto depth : {0, 1, 1, 2, 2, 2, 0}) {
      auto cpu_features = a.GetFeatureSet(&cpu, depth);
      if (!levels.count(depth)) levels.emplace(depth, sample(tree_features, level));
      auto expected = sample(levels.at(depth), node);
      auto actual = b.GetFeatureSet(&gpu, depth);
      if (expected != actual->ConstHostVector()) {
        std::fprintf(stderr, "tree=%d depth=%d expected:", t, depth);
        for (auto f : expected) std::fprintf(stderr, " %u", f);
        std::fprintf(stderr, " actual:");
        for (auto f : actual->ConstHostVector()) std::fprintf(stderr, " %u", f);
        std::fprintf(stderr, "\n");
        Check(false, "Production ColumnSampler tree/level/node selection differs");
      }
      if (tree == 1 && level == 1 && node == 1)
        Check(cpu_features->ConstHostVector() == actual->ConstHostVector(),
              "Unsampled CPU features differ");
      checked_features += actual->Size();
    }
  }
  auto after = common::metal::GetStatistics();
  Check(after.column_init_dispatches == before.column_init_dispatches + 2, "Missing GPU iota");
  auto per_tree = int(tree != 1) + 3 * int(level != 1) + 7 * int(node != 1);
  Check(after.column_sampling_dispatches == before.column_sampling_dispatches + 2 * per_tree,
        "Wrong sampling/cached-level dispatch count");
  auto next = gpu.Rng()();
  Check(cpu.Rng()() == next && reference_rng() == next, "Parent RNG consumption changed");
  ++cases;
}

void ResidentHierarchy(std::uint32_t n, float tree, float level, float node, bool weighted) {
  current = "resident hierarchy=" + std::to_string(n);
  Context gpu;
  gpu.Init({{"device", "mps"}, {"nthread", "2"}});
  gpu.Rng().seed(8721);
  RandomEngine reference_rng(8721);
  std::vector<float> host_weights(weighted ? n : 0);
  for (std::size_t i = 0; i < host_weights.size(); ++i) host_weights[i] = float(1 + i % 7) / 3;
  HostDeviceVector<float> weights(host_weights, DeviceOrd::MPS());
  common::ColumnSampler sampler;
  auto sample = [&](std::vector<bst_feature_t> const& features, float rate) {
    if (rate == 1.f) return features;
    return cuda_reference::SampleFeatures(features, host_weights, reference_rng(),
                                          std::max(1, int(rate * features.size())));
  };
  std::vector<HostDeviceVector<bst_feature_t>> consumed;
  std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> snapshots;
  std::vector<std::vector<bst_feature_t>> expected_snapshots, expected_consumed;
  auto before = common::metal::GetStatistics();
  for (int t = 0; t < 2; ++t) {
    sampler.Init(&gpu, n, weights, node, level, tree);
    std::vector<bst_feature_t> all(n);
    std::iota(all.begin(), all.end(), 0);
    auto tree_features = sample(all, tree);
    std::unordered_map<int, std::vector<bst_feature_t>> levels;
    for (auto depth : {0, 1, 1, 2, 2, 2, 0}) {
      if (!levels.count(depth)) levels.emplace(depth, sample(tree_features, level));
      auto expected = sample(levels.at(depth), node);
      auto actual = sampler.GetFeatureSet(&gpu, depth);
      Check(actual->DeviceCanRead() && !actual->HostCanRead(), "Feature cache left GPU ownership");
      auto count = std::max<std::size_t>(1, actual->Size() / 2);
      consumed.emplace_back(count, 0, DeviceOrd::MPS());
      // Consume before any CPU inspection, including previously cached levels.
      common::metal::SampleFeatures(actual->ConstDeviceSpan(), weights.ConstDeviceSpan(), 123,
                                    consumed.back().DeviceSpan());
      expected_consumed.push_back(cuda_reference::SampleFeatures(expected, host_weights, 123, count));
      // The second Init is allowed to reuse the old tree allocation. Save only
      // independently produced node/level snapshots across that boundary.
      if (node != 1.f || level != 1.f) {
        snapshots.push_back(actual);
        expected_snapshots.push_back(expected);
      }
    }
  }
  auto after = common::metal::GetStatistics();
  Check(after.feature_host_input_bytes == before.feature_host_input_bytes &&
            after.feature_readback_bytes == before.feature_readback_bytes &&
            after.vector_host_copy_bytes == before.vector_host_copy_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes,
        "Resident tree/level/node sampling staged or read feature/weight payloads");
  Check(after.feature_shared_input_bytes > before.feature_shared_input_bytes &&
            after.feature_shared_output_bytes > before.feature_shared_output_bytes,
        "Missing resident feature bindings/publication");
  Check(gpu.Rng()() == reference_rng(), "Resident hierarchy consumed different RNG draws");
  for (std::size_t i = 0; i < snapshots.size(); ++i)
    Check(snapshots[i]->ConstHostVector() == expected_snapshots[i], "Queued feature snapshot changed");
  for (std::size_t i = 0; i < consumed.size(); ++i)
    Check(consumed[i].ConstHostVector() == expected_consumed[i], "GPU sampling consumer differs");
  ++resident_chains;
}

void ResidentAliases(bool weighted) {
  current = weighted ? "resident weighted offsets/aliases" : "resident shuffle offsets/aliases";
  constexpr std::size_t n = 257, count = 129;
  constexpr bst_feature_t guard = 0xbadcafeu;
  auto device = DeviceOrd::MPS();
  HostDeviceVector<bst_feature_t> ids(n + 4, guard, device), consumed(31, 0, device);
  std::vector<float> host_weights(weighted ? n : 0);
  for (std::size_t i = 0; i < host_weights.size(); ++i) host_weights[i] = float(1 + i % 7) / 3;
  std::vector<float> padded_weights(n + 2, -99);
  if (weighted) std::copy(host_weights.begin(), host_weights.end(), padded_weights.begin() + 1);
  HostDeviceVector<float> weights(padded_weights, device), uniforms(n + 2, -99, device);
  HostDeviceVector<bst_feature_t> permutation(n + 2, guard, device);
  auto features = ids.DeviceSpan().subspan(2, n);
  auto w = weighted ? weights.DeviceSpan().subspan(1, n) : Span<float>{};
  auto u = weighted ? uniforms.DeviceSpan().subspan(1, n) : Span<float>{};
  auto perm = weighted ? Span<bst_feature_t>{} : permutation.DeviceSpan().subspan(1, n);
  auto before = common::metal::GetStatistics();
  common::metal::InitFeatureSet(features);
  // The feature destination overlaps its input; key diagnostics overwrite the
  // source weights only after all GPU readers finish.
  common::metal::SampleFeatures(features, w, 77, features.subspan(3, count), u, w, perm);
  common::metal::SampleFeatures(features.subspan(3, count), {}, 111, consumed.DeviceSpan());
  auto after = common::metal::GetStatistics();
  Check(after.feature_host_input_bytes == before.feature_host_input_bytes &&
            after.feature_readback_bytes == before.feature_readback_bytes &&
            after.vector_readback_bytes == before.vector_readback_bytes,
        "Offset/alias sampling crossed CPU payload boundary");
  std::vector<bst_feature_t> original(n), expected;
  std::iota(original.begin(), original.end(), 0);
  expected = cuda_reference::SampleFeatures(original, host_weights, 77, count);
  Check(consumed.ConstHostVector() == cuda_reference::SampleFeatures(expected, {}, 111, 31),
        "Aliased resident sample failed subsequent GPU consumer");
  auto const& actual = ids.ConstHostVector();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    auto want = (i < 2 || i >= n + 2) ? guard
                : (i >= 5 && i < 5 + count) ? expected[i - 5] : bst_feature_t(i - 2);
    Check(actual[i] == want, "Aliased sampling modified a guard or selected wrong feature");
  }
  if (weighted) {
    auto const& uw = uniforms.ConstHostVector();
    auto const& kw = weights.ConstHostVector();
    Check(uw.front() == -99 && uw.back() == -99 && kw.front() == -99 && kw.back() == -99,
          "Resident RNG diagnostic guards changed");
    for (std::size_t i = 0; i < n; ++i) {
      auto draw = cuda_reference::Weight(77, i);
      Check(Bits(uw[i + 1]) == Bits(draw) &&
                Bits(kw[i + 1]) == Bits(std::log(draw) / host_weights[i]),
            "Aliased GPU weight/key/uniform diagnostics differ");
    }
  } else {
    auto const& p = permutation.ConstHostVector();
    auto expected_permutation = cuda_reference::Shuffle(n, 77);
    Check(p.front() == guard && p.back() == guard &&
              std::equal(expected_permutation.begin(), expected_permutation.end(), p.begin() + 1),
          "Resident permutation/guards differ");
  }
  ++resident_chains;
}

void ReadOnlyInputsAndRejectedPublication() {
  current = "read-only input fallback and rejected GPU publication";
  auto page = static_cast<std::size_t>(getpagesize());
  void* allocation = mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  Check(allocation != MAP_FAILED, "Cannot allocate read-only fixture");
  struct Cleanup {
    void* pointer;
    std::size_t bytes;
    ~Cleanup() { munmap(pointer, bytes); }
  } cleanup{allocation, page * 2};
  auto features = static_cast<bst_feature_t*>(allocation);
  auto weights = reinterpret_cast<float*>(static_cast<char*>(allocation) + page);
  std::vector<bst_feature_t> ids{0, 2, 4, 6};
  std::vector<float> w{1, 2, 3, 4, 5, 6, 7};
  std::copy(ids.begin(), ids.end(), features);
  std::copy(w.begin(), w.end(), weights);
  Check(mprotect(allocation, page * 2, PROT_READ) == 0, "Cannot protect fixture");
  auto device = DeviceOrd::MPS();
  HostDeviceVector<bst_feature_t> output(2, 0xbadcafeu, device);
  auto before = common::metal::GetStatistics();
  common::metal::SampleFeatures({features, 4}, {weights, 7}, 8, output.DeviceSpan());
  auto after = common::metal::GetStatistics();
  Check(after.feature_host_input_bytes == before.feature_host_input_bytes + 11 * 4 &&
            after.feature_shared_output_bytes == before.feature_shared_output_bytes + 2 * 4 &&
            after.feature_readback_bytes == before.feature_readback_bytes,
        "Read-only payload fallback/publication was not counted exactly");
  Check(output.ConstHostVector() == cuda_reference::SampleFeatures(ids, w, 8, 2),
        "Read-only input fallback selected different features");
  output.DeviceSpan();
  output.Fill(0xbadcafeu);
  HostDeviceVector<float> u(4, -99, device), keys(4, -99, device);
  before = common::metal::GetStatistics();
  bool rejected = false;
  try {
    common::metal::SampleFeatures({features, 4}, {weights, 4}, 8, output.DeviceSpan(),
                                  u.DeviceSpan(), keys.DeviceSpan());
  } catch (std::exception const&) {
    rejected = true;
  }
  after = common::metal::GetStatistics();
  Check(rejected && after.feature_shared_output_bytes == before.feature_shared_output_bytes &&
            after.column_sampling_dispatches == before.column_sampling_dispatches,
        "Invalid input published partially computed sampling payloads");
  Check(output.ConstHostVector() == std::vector<bst_feature_t>(2, 0xbadcafeu) &&
            u.ConstHostVector() == std::vector<float>(4, -99) &&
            keys.ConstHostVector() == std::vector<float>(4, -99),
        "Rejected sample overwrote destination or diagnostics");
  Context ctx;
  ctx.Init({{"device", "mps"}});
  HostDeviceVector<float> empty_weights;
  common::ColumnSampler sampler;
  before = common::metal::GetStatistics();
  sampler.Init(&ctx, 0, empty_weights, 1, 1, 1);
  auto empty = sampler.GetFeatureSet(&ctx, 0);
  common::metal::SampleFeatures(empty->ConstDeviceSpan(), {}, 8, empty->DeviceSpan());
  after = common::metal::GetStatistics();
  Check(empty->Empty() && empty->Device().IsMPS() &&
            after.column_init_dispatches == before.column_init_dispatches &&
            after.column_sampling_dispatches == before.column_sampling_dispatches,
        "Empty sampler did not retain an empty Metal result without dispatch");
  ++resident_chains;
}

int main() {
  try {
    {
      current = "CUDA Thrust shuffle source contract";
      std::vector<bst_feature_t> features(17), selected(7);
      std::iota(features.begin(), features.end(), 0);
      common::metal::SampleFeatures(features, {}, 7, selected);
      Check(selected == cuda_reference::SampleFeatures(features, {}, 7, 7),
            "Column sampler does not use CUDA's Thrust shuffle");
    }
    for (std::uint32_t n : {1u, 2u, 3u, 31u, 32u, 33u, 255u, 256u, 257u, 623u, 624u, 625u, 2047u,
                            2048u, 2049u, 4095u, 4096u, 4097u, 65537u}) {
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
    for (std::uint32_t seed : {2147483646u, 2147483647u, 2147483648u, 0xfffffffeu})
      for (std::uint32_t n : {1u, 255u, 256u, 257u, 2049u}) Case(n, std::max(1u, n / 2), seed, 0);
    for (int mode : {0, 2, 3, 6}) Case(1000003, 500003, 9852, mode);
    for (std::uint32_t n : {3u, 257u, 4097u})
      for (bool weighted : {false, true})
        for (auto rates : {std::vector<float>{1, 1, 1},
                           {0.7f, 1, 1},
                           {1, 0.7f, 1},
                           {1, 1, 0.7f},
                           {0.9f, 0.7f, 0.5f}})
          {
            Hierarchy(n, rates[0], rates[1], rates[2], weighted);
            ResidentHierarchy(n, rates[0], rates[1], rates[2], weighted);
          }
    ResidentAliases(false);
    ResidentAliases(true);
    ReadOnlyInputsAndRejectedPublication();
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
        "PASS %zu column cases / %zu selected IDs / %zu exact random draws / %zu permutation "
        "indices; "
        "maximum weighted-key difference %u FP32 ULP; %zu resident chains; hierarchy, RNG, "
        "guards and bounds pass\n",
        cases, checked_features, checked_uniforms, checked_permutations, max_key_ulps, resident_chains);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), error.what());
    return 1;
  }
}
