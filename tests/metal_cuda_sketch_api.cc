// Production Metal raw-ingestion/resident sketch against independent CUDA formulas.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_cuda_sketch.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "cuda_sketch_reference.h"

using namespace xgboost;
namespace ref = cuda_sketch_reference;
namespace metal = xgboost::common::metal;
std::size_t cases = 0, fields = 0, input_entries = 0, pushes = 0;
std::string current;
void Check(bool ok, char const* reason) {
  if (!ok) throw std::runtime_error(current + ": " + reason);
}
void Same(float a, float b) {
  ++fields;
  if (std::isnan(a) && std::isnan(b)) return;
  if (std::memcmp(&a, &b, sizeof(a))) {
    std::fprintf(stderr, "%s: Metal %.9g oracle %.9g\n", current.c_str(), a, b);
    Check(false, "Float bits differ");
  }
}
void Verify(metal::CudaSketchSummary const& actual, ref::Sketch const& expected) {
  std::size_t offset = 0;
  Check(actual.ptr.size() == expected.columns.size() + 1, "Column count");
  for (std::size_t f = 0; f < expected.columns.size(); ++f) {
    Check(actual.ptr[f] == offset, "Column pointer");
    for (auto e : expected.columns[f]) {
      Check(offset < actual.entries.size(), "Missing entry");
      auto a = actual.entries[offset++];
      Same(a.rmin, e.rmin);
      Same(a.rmax, e.rmax);
      Same(a.wmin, e.wmin);
      Same(a.value, e.value);
    }
  }
  Check(actual.ptr.back() == offset && actual.entries.size() == offset, "Final count");
}
void Case(std::size_t n, unsigned mode, std::size_t window, int bins) {
  current = "rows=" + std::to_string(n) + " mode=" + std::to_string(mode) +
            " window=" + std::to_string(window) + " bins=" + std::to_string(bins);
  constexpr std::size_t columns = 7, base = 5;
  std::vector<FeatureType> types(columns, FeatureType::kNumerical);
  if (mode % 2) types[2] = types[4] = FeatureType::kCategorical;
  bool weighted = mode >= 2;
  std::vector<float> weights(weighted ? n + base : 0);
  std::vector<bst_idx_t> ptr{0};
  std::vector<::xgboost::Entry> entries;
  std::vector<ref::Raw> raw;
  std::mt19937 rng(793);
  for (std::size_t r = 0; r < n; ++r) {
    float w = mode == 4 || mode == 9 ? 0.f : float(rng() % 8) / 4;
    if (mode == 11) w = -0.f;
    if (mode == 6) w = std::ldexp(w, -140);
    if (mode == 7) w = std::ldexp(w, 110);
    if (weighted) weights[r + base] = w;
    for (std::size_t f = 0; f < columns; ++f) {
      // Empty columns and empty rows, plus sparse and reversed feature order.
      if (f == 3 || r % 23 == 0 || (r + f) % 5 == 0) continue;
      float v = float(int(rng() % 1009) - 504) / 8;
      if (mode >= 8 && mode <= 10 && f == 0) v = mode == 10 ? INFINITY : -INFINITY;
      if (f == 5) v = (int(r % 7) - 3) * std::numeric_limits<float>::denorm_min();
      if (f == 6) v = r % 2 ? -0.f : 0.f;
      if (types[f] == FeatureType::kCategorical) v = float(rng() % 83) * 2;
      entries.emplace_back(f, v);
      raw.push_back({std::uint32_t(f), v, w});
    }
    ptr.push_back(entries.size());
  }
  auto original = entries;
  auto original_ptr = ptr;
  auto original_weights = weights;
  std::vector<bool> cats;
  for (auto t : types) cats.push_back(t == FeatureType::kCategorical);
  ref::Sketch oracle(columns, bins, cats);
  auto before = metal::GetStatistics();
  auto gpu = metal::MakeCudaSketch(columns, bins, types, weights);
  Verify(gpu->Read(), oracle);
  std::size_t count = 0;
  for (std::size_t begin = 0; begin < entries.size(); begin += window) {
    auto end = std::min(entries.size(), begin + window);
    auto first = std::upper_bound(ptr.begin(), ptr.end(), begin) - ptr.begin() - 1;
    auto last = std::lower_bound(ptr.begin(), ptr.end(), end) - ptr.begin();
    auto rows = std::max<std::size_t>(1, last - first);
    gpu->Push({entries.data() + begin, end - begin}, ptr, begin, base, rows);
    oracle.Push({raw.begin() + begin, raw.begin() + end}, rows, weighted);
    Verify(gpu->Read(), oracle);
    ++count;
    ++pushes;
  }
  gpu->Push({}, {}, 0, 0, 0);
  Verify(gpu->Read(), oracle);
  auto after = metal::GetStatistics();
  Check(after.cuda_sketch_batches - before.cuda_sketch_batches == count, "Batch dispatch missing");
  Check(after.cuda_sketch_input_entries - before.cuda_sketch_input_entries == entries.size(),
        "Raw count");
  Check(entries.size() == original.size() &&
            (entries.empty() ||
             !std::memcmp(entries.data(), original.data(), entries.size() * sizeof(entries[0]))),
        "Source changed");
  Check(ptr == original_ptr && weights == original_weights, "Metadata changed");
  input_entries += entries.size();
  ++cases;
  std::printf("PASS %s: %zu pushes, %zu total fields\n", current.c_str(), count, fields);
  std::fflush(stdout);
}
void OneRowDuplicateFeatures() {
  current = "one-row duplicate CSR feature: keep first endpoint when budget=1";
  std::vector<::xgboost::Entry> entries{{0, 7.f}, {0, -3.f}, {0, 2.f}};
  std::vector<bst_idx_t> rows{0, 3};
  for (bool weighted : {false, true}) {
    std::vector<float> weights(weighted ? 1 : 0, .5f);
    auto gpu = metal::MakeCudaSketch(2, 2, {}, weights);
    gpu->Push(entries, rows, 0, 0, 1);
    ref::Sketch oracle(2, 2, {false, false});
    oracle.Push({{0, 7.f, .5f}, {0, -3.f, .5f}, {0, 2.f, .5f}}, 1, weighted);
    Verify(gpu->Read(), oracle);
    ++cases;
    ++pushes;
    input_entries += entries.size();
  }
}
int main() {
  try {
    OneRowDuplicateFeatures();
    for (unsigned mode = 0; mode < 8; ++mode) {
      for (std::size_t n : {0u, 1u, 17u, 257u, 8193u}) {
        Case(n, mode, 1 << 26, 17);
      }
      Case(257, mode, 71, 2);
      Case(1025, mode, 1021, 7);
    }
    for (unsigned mode = 8; mode <= 11; ++mode) Case(257, mode, 71, 2);
    Case(262147, 2, 1 << 26, 33);
    Case(65537, 3, 65537, 33);
    std::printf("PASS: %zu cases, %zu pushes, %zu source entries, %zu exact fields\n", cases,
                pushes, input_entries, fields);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
