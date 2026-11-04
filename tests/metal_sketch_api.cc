// Production GPU sketch finalization versus upstream CPU merge/prune and
// independently queried merge ranks. Includes complete streamed CPU sketches.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_sketch.h"

using namespace xgboost;
using namespace xgboost::common;
using SketchEntry = WQSummary<>::Entry;
std::size_t columns_checked = 0, entries_checked = 0, fields_checked = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Same(Span<SketchEntry const> a, Span<SketchEntry const> b) {
  Check(a.size() == b.size(), "Summary size differs");
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(SketchEntry))) {
      std::fprintf(stderr, "%s entry %zu: GPU=(%.9g %.9g %.9g %.9g) CPU=(%.9g %.9g %.9g %.9g)\n",
                   current.c_str(), i, a[i].rmin, a[i].rmax, a[i].wmin, a[i].value, b[i].rmin,
                   b[i].rmax, b[i].wmin, b[i].value);
      throw std::runtime_error("Summary bits differ");
    }
  }
  fields_checked += a.size() * 4;
}
std::vector<SketchEntry> IndependentMerge(Span<SketchEntry const> a, Span<SketchEntry const> b) {
  if (a.empty()) return {b.begin(), b.end()};
  if (b.empty()) return {a.begin(), a.end()};
  std::vector<float> values;
  for (auto const& e : a) values.push_back(e.value);
  for (auto const& e : b) values.push_back(e.value);
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  auto rank = [](Span<SketchEntry const> entries, float value) {
    auto found = std::lower_bound(entries.begin(), entries.end(), value,
                                  [](SketchEntry const& e, float x) { return e.value < x; });
    if (found != entries.end() && found->value == value) return *found;
    float lower = found == entries.begin() ? 0 : (found - 1)->RMinNext();
    float upper = found == entries.end() ? entries.back().rmax : found->RMaxPrev();
    return SketchEntry{lower, upper, 0, value};
  };
  std::vector<SketchEntry> output;
  float rmin = 0, rmax = 0;
  for (float value : values) {
    auto aa = rank(a, value), bb = rank(b, value);
    SketchEntry e{aa.rmin + bb.rmin, aa.rmax + bb.rmax, aa.wmin + bb.wmin, value};
    rmin = std::max(rmin, e.rmin);
    e.rmin = rmin;
    rmax = std::max({rmax, e.rmax, e.rmin + e.wmin});
    e.rmax = rmax;
    output.push_back(e);
  }
  return output;
}
struct Owned {
  std::vector<std::vector<SketchEntry>> storage;
  std::vector<WQSummary<>> levels;
  std::size_t intermediate, final;
  void Views() {
    levels.clear();
    for (auto& s : storage)
      levels.emplace_back(Span<SketchEntry>{s.data() + 1, s.size() - 2}, s.size() - 2);
  }
};
std::vector<SketchEntry> Generate(std::size_t n, int level, int mode) {
  std::vector<SketchEntry> out(n + 2, SketchEntry{-971, 973, -977, 983});
  std::mt19937 rng(1679 + level * 31 + mode);
  float rank = 0;
  for (std::size_t i = 0; i < n; ++i) {
    float weight = float(1 + rng() % 13);
    if (mode == 1) weight /= 19;
    if (mode == 2) weight = std::ldexp(weight, -140);
    if (mode == 3) weight = std::ldexp(weight, int(rng() % 101) - 50);
    if (mode == 4) weight = std::ldexp(weight, 90);
    if (mode == 5) weight = 0;
    if (mode == 9) weight = std::ldexp(weight, 120);
    float value = float(i * (mode == 6 ? 8 : 2) + level % 5);
    if (mode == 7) value = std::ldexp(value - 16384.f, -140);
    out[i + 1] = SketchEntry{rank, rank + weight, weight, value};
    rank += weight;
    if (mode == 8) {  // Slightly inconsistent ranks exercise the correction pass.
      out[i + 1].rmax -= float(i % 3);
      out[i + 1].rmin -= float(i % 7);
    }
  }
  return out;
}
std::uint64_t Fingerprint(std::vector<SketchEntry> const& data) {
  std::uint64_t hash = 1469598103934665603ull;
  auto bytes = reinterpret_cast<unsigned char const*>(data.data());
  for (std::size_t i = 0; i < data.size() * sizeof(SketchEntry); ++i)
    hash = (hash ^ bytes[i]) * 1099511628211ull;
  return hash;
}
void Batch(std::vector<Owned>* batch) {
  std::vector<metal::SketchFinalization> inputs;
  for (auto& column : *batch) {
    column.Views();
    inputs.push_back({column.levels, column.intermediate, column.final});
  }
  std::vector<std::vector<std::uint64_t>> fingerprints;
  for (auto const& column : *batch) {
    fingerprints.emplace_back();
    for (auto const& level : column.storage) fingerprints.back().push_back(Fingerprint(level));
  }
  auto before = metal::GetStatistics();
  auto actual = metal::FinalizeSketches(inputs);
  auto after = metal::GetStatistics();
  Check(after.sketch_finalize_dispatches == before.sketch_finalize_dispatches + !inputs.empty(),
        "Missing GPU sketch dispatch");
  Check(after.sketch_finalize_columns == before.sketch_finalize_columns + inputs.size(),
        "Wrong GPU column count");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    for (std::size_t l = 0; l < (*batch)[i].storage.size(); ++l)
      Check(Fingerprint((*batch)[i].storage[l]) == fingerprints[i][l], "Source sketch changed");
    auto const& input = inputs[i];
    current = "column=" + std::to_string(columns_checked) +
              " intermediate=" + std::to_string(input.intermediate_budget) +
              " final=" + std::to_string(input.final_budget);
    WQSummaryContainer expected;
    for (auto const& level : input.levels) {
      auto independent = IndependentMerge(expected.Entries(), level.Entries());
      expected.Reserve(expected.Size() + level.Size());
      expected.SetCombine(level);
      Same(expected.Entries(), independent);
      expected.SetPrune(input.intermediate_budget);
      entries_checked += level.Size();
    }
    expected.SetPrune(input.final_budget);
    Same(actual[i].Entries(), expected.Entries());
    auto cuts_a = actual[i].QueryCutValues(17), cuts_b = expected.QueryCutValues(17);
    Check(cuts_a.size() == cuts_b.size() &&
              std::memcmp(cuts_a.data(), cuts_b.data(), cuts_a.size() * 4) == 0,
          "Final cuts differ");
    SketchEntry guard{-971, 973, -977, 983};
    for (auto const& s : (*batch)[i].storage)
      Check(std::memcmp(&guard, &s.front(), sizeof(guard)) == 0 &&
                std::memcmp(&guard, &s.back(), sizeof(guard)) == 0,
            "Input guards changed");
    ++columns_checked;
  }
  std::printf("Checked %zu sketch columns / %zu source entries / %zu exact fields\n",
              columns_checked, entries_checked, fields_checked);
  std::fflush(stdout);
  batch->clear();
}
void Streamed() {
  std::vector<WQuantileSketch> sketches;
  std::vector<metal::SketchFinalization> inputs;
  std::vector<WQSummaryContainer> expected;
  for (std::size_t n : {0u, 1u, 31u, 513u, 4097u, 65537u, 1000003u}) {
    for (int mode = 0; mode < 4; ++mode) {
      auto budget = SketchSummaryBudget(33, n);
      WQuantileSketch cpu(n, SketchEpsilon(33, n)), gpu(n, SketchEpsilon(33, n));
      std::mt19937 rng(981 + mode);
      for (std::size_t i = 0; i < n; ++i) {
        float value = float(int(rng() % 997) - 498) / 83;
        float weight = mode ? float(rng() % 31) / 17 : 1;
        if (mode == 2) weight = std::ldexp(weight, -135);
        if (mode == 3) value = float((i / 9) % 3);
        cpu.Push(value, weight);
        gpu.Push(value, weight);
      }
      expected.push_back(cpu.GetSummary(budget));
      gpu.PrepareSummary();
      sketches.push_back(std::move(gpu));
    }
  }
  for (auto const& sketch : sketches) {
    auto budget = SketchSummaryBudget(33, sketch.NumElements());
    // The CPU reference used total rows as the budget input above; nonzero-count
    // differences change only tiny cases here. Match its actual requested budget.
    auto index = inputs.size();
    std::size_t n = std::vector<std::size_t>{0, 1, 31, 513, 4097, 65537, 1000003}[index / 4];
    budget = SketchSummaryBudget(33, n);
    inputs.push_back({sketch.SummaryLevels(), std::max(budget, sketch.SummaryLimit()), budget});
  }
  auto actual = metal::FinalizeSketches(inputs);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    current = "streamed=" + std::to_string(i);
    Same(actual[i].Entries(), expected[i].Entries());
    ++columns_checked;
  }
  std::puts(
      "PASS 28 streamed sketches through million-row ingestion, including duplicate/zero/tiny "
      "weights");
}
int main() {
  try {
    std::vector<Owned> batch;
    for (std::size_t n : {0u, 1u, 2u, 3u, 7u, 31u, 32u, 33u, 255u, 256u, 257u, 1025u, 4097u})
      for (int mode = 0; mode < 10; ++mode)
        for (std::size_t budget : {0u, 1u, 2u, 3u, 17u, 128u, 4096u}) {
          Owned column;
          column.intermediate = budget;
          column.final = budget ? (budget + 1) / 2 : 0;
          for (int level = 0; level < 5; ++level)
            column.storage.push_back(Generate(level % 3 == 0 ? 0 : n + level % 2, level, mode));
          batch.push_back(std::move(column));
          if (batch.size() == 64) Batch(&batch);
        }
    Batch(&batch);
    for (std::size_t budget : {1u, 17u, 257u, 65537u}) {
      Owned column;
      column.intermediate = budget;
      column.final = budget;
      for (int level = 0; level < 3; ++level) column.storage.push_back(Generate(65537, level, 1));
      batch.push_back(std::move(column));
    }
    Batch(&batch);
    Streamed();
    std::vector<metal::SketchFinalization> empty;
    Check(metal::FinalizeSketches(empty).empty(), "Empty batch result");
    std::printf("PASS %zu native sketch columns / %zu source entries / %zu exact fields\n",
                columns_checked, entries_checked, fields_checked);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), error.what());
    return 1;
  }
}
