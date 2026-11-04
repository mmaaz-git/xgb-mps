// Actual Metal sketch ingestion/compaction against upstream CPU arithmetic.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/hist_util.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_sketch.h"

using namespace xgboost;
using namespace xgboost::common;
using SummaryEntry = WQSummary<>::Entry;
std::size_t weight_cases = 0, sorted_cases = 0, stream_cases = 0, fields = 0, source_entries = 0;
std::string current;
void Check(bool value, char const* message) {
  if (!value) throw std::runtime_error(current + ": " + message);
}
void Same(float actual, float expected) {
  ++fields;
  if (std::isnan(actual) && std::isnan(expected)) return;
  if (std::memcmp(&actual, &expected, sizeof(float))) {
    std::fprintf(stderr, "%s GPU %.9g CPU %.9g\n", current.c_str(), actual, expected);
    Check(false, "Float bits differ");
  }
}
void Same(Span<SummaryEntry const> actual, Span<SummaryEntry const> expected) {
  Check(actual.size() == expected.size(), "Summary sizes differ");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    Same(actual[i].rmin, expected[i].rmin);
    Same(actual[i].rmax, expected[i].rmax);
    Same(actual[i].wmin, expected[i].wmin);
    Same(actual[i].value, expected[i].value);
  }
}
template <typename T>
bool Unchanged(std::vector<T> const& a, std::vector<T> const& b) {
  return a.size() == b.size() &&
         (a.empty() || !std::memcmp(a.data(), b.data(), a.size() * sizeof(T)));
}
float TestValue(std::size_t i) {
  float exceptional[] = {0.f, -0.f, 1e-40f, -1e-40f, 1e30f, -1e30f, INFINITY, -INFINITY, NAN};
  return i % 19 < 9 ? exceptional[i % 19] : float(int(i % 173) - 70) / 19;
}
void Weights() {
  for (std::size_t n : {0u, 1u, 31u, 257u, 4097u, 1000003u}) {
    for (bool grouped : {false, true}) {
      for (bool hessian : {false, true}) {
        for (bool weighted : {false, true}) {
          current = "weights n=" + std::to_string(n);
          std::vector<bst_group_t> groups;
          if (grouped) groups = {0, 0, unsigned(n / 3), unsigned(n / 3), unsigned(n)};
          std::vector<float> h(hessian ? n : 0), w(weighted ? (grouped ? 4 : n) : 0);
          for (std::size_t i = 0; i < h.size(); ++i) h[i] = TestValue(i);
          for (std::size_t i = 0; i < w.size(); ++i) w[i] = TestValue(i + 7);
          auto saved_h = h, saved_w = w;
          auto before = metal::GetStatistics();
          auto actual = metal::MakeSketchWeights(n, h, w, groups);
          auto after = metal::GetStatistics();
          Check(after.sketch_weight_dispatches - before.sketch_weight_dispatches == bool(n),
                "Weight dispatch missing");
          Check(actual.size() == n, "Weight size differs");
          for (std::size_t i = 0; i < n; ++i) {
            auto group =
                grouped ? std::upper_bound(groups.begin(), groups.end(), i) - groups.begin() - 1
                        : i;
            float weight = w.empty() ? 1.f : w[group];
            Same(actual[i], h.empty() ? weight : h[i] * weight);
          }
          Check(Unchanged(h, saved_h) && Unchanged(w, saved_w), "Weight source changed");
          ++weight_cases;
        }
      }
    }
    std::printf("Checked %zu weight cases / %zu fields\n", weight_cases, fields);
    std::fflush(stdout);
  }
}
void Sorted(std::size_t n, unsigned mode) {
  std::vector<float> weights(n);
  std::vector<::xgboost::Entry> entries;
  std::mt19937 rng(793 + mode);
  for (std::size_t i = 0; i < n; ++i) {
    float w = float(rng() % 397) / 23;
    if (i % 13 == 0 || mode == 1) w = 0;
    if (mode == 2) w = std::ldexp(w, -140);
    if (mode == 3) w = std::ldexp(w, 120);
    if (mode == 4) w = std::ldexp(w, int(i % 80) - 40);
    if (mode == 6) w = i % 2 ? -w : w;
    if (mode == 5) w = i % 83 == 0 ? NAN : (i % 17 == 0 ? INFINITY : w);
    weights[i] = w;
    float v = mode == 4 ? std::ldexp(float(int(i % 67) - 33), -145) : float(i % 397) / 31;
    if (mode == 4 && i % 67 == 33 && i % 2) v = -0.f;
    entries.emplace_back(i, v);
  }
  std::stable_sort(entries.begin(), entries.end(),
                   [](auto a, auto b) { return a.fvalue < b.fvalue; });
  auto saved_entries = entries;
  auto saved_weights = weights;
  std::vector<metal::SortedSketchColumn> inputs;
  for (std::size_t budget : {1u, 2u, 7u, 33u, 511u}) inputs.push_back({entries, budget});
  if (n > 1000000) inputs.resize(1);
  auto before = metal::GetStatistics();
  auto actual = metal::MakeSortedSketches(inputs, weights);
  auto after = metal::GetStatistics();
  Check(after.sketch_sorted_dispatches == before.sketch_sorted_dispatches + 1,
        "Sorted dispatch missing");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    current = "sorted n=" + std::to_string(n) + " mode=" + std::to_string(mode) +
              " budget=" + std::to_string(inputs[i].budget);
    WQSummaryContainer cpu;
    cpu.Reserve(inputs[i].budget + 1);
    cpu.SetPruneSorted(entries, weights, inputs[i].budget);
    Same(actual[i].summary.Entries(), cpu.Entries());
    auto count = std::count_if(entries.begin(), entries.end(),
                               [&](auto e) { return weights[e.index] != 0; });
    Check(actual[i].num_elements == std::size_t(count), "Nonzero source count differs");
    ++sorted_cases;
    source_entries += entries.size();
  }
  Check(Unchanged(entries, saved_entries) && Unchanged(weights, saved_weights),
        "Sorted source changed");
}
void Streams() {
  for (std::size_t n : {0u, 1u, 31u, 4097u, 65539u, 1000003u}) {
    for (int mode = 0; mode < 4; ++mode) {
      current = "stream n=" + std::to_string(n) + " mode=" + std::to_string(mode);
      WQuantileSketch cpu(n, SketchEpsilon(17, n)), gpu(n, SketchEpsilon(17, n), true);
      std::mt19937 rng(117);
      auto before = metal::GetStatistics();
      for (std::size_t i = 0; i < n; ++i) {
        float value = mode == 3 ? float((i / 11) % 3) : float(int(rng() % 997) - 498) / 31;
        float weight = float(rng() % 373) / 71;
        if (mode == 1) weight = std::ldexp(weight, -140);
        if (mode == 2) weight = 0;
        cpu.Push(value, weight);
        gpu.Push(value, weight);
      }
      // Flush twice to exercise empty incoming summaries against resident levels.
      for (int flush = 0; flush < 2; ++flush) {
        cpu.PrepareSummary();
        gpu.PrepareSummary();
        auto a = gpu.SummaryLevels(), b = cpu.SummaryLevels();
        Check(a.size() == b.size(), "Streaming level count differs");
        for (std::size_t l = 0; l < a.size(); ++l) Same(a[l].Entries(), b[l].Entries());
        Check(gpu.NumElements() == cpu.NumElements(), "Streaming source count differs");
      }
      auto limit = SketchSummaryBudget(17, gpu.NumElements());
      metal::SketchFinalization input{gpu.SummaryLevels(), std::max(limit, gpu.SummaryLimit()),
                                      limit};
      auto actual = metal::FinalizeSketches({&input, 1});
      auto expected = cpu.GetSummary(limit);
      Same(actual.front().Entries(), expected.Entries());
      Check(metal::GetStatistics().sketch_update_dispatches > before.sketch_update_dispatches,
            "Stream never dispatched");
      source_entries += n;
      ++stream_cases;
    }
    std::printf("Checked %zu streamed sketches / %zu source entries / %zu fields\n", stream_cases,
                source_entries, fields);
    std::fflush(stdout);
  }
}
void BatchedUpdates() {
  constexpr std::size_t n = 32;
  std::vector<std::vector<WQSummaryContainer>> storage(n);
  std::vector<std::vector<WQSummary<>>> views(n);
  std::vector<WQSummaryContainer> incoming(n);
  std::vector<metal::SketchUpdate> inputs;
  for (std::size_t c = 0; c < n; ++c) {
    auto limit = std::size_t{1} << (c % 6);
    storage[c].resize(c % 4);
    for (std::size_t l = 0; l < storage[c].size(); ++l) {
      auto count = (c + l) % 3 == 0 ? 0 : limit;
      auto& s = storage[c][l];
      s.Reserve(count);
      for (std::size_t i = 0; i < count; ++i)
        s.space[i] = SummaryEntry{float(i), float(i + 1), 1.f, float(i * 2 + l)};
      s.SetSize(count);
      views[c].emplace_back(Span<SummaryEntry>{s.space}, count);
    }
    auto count = (c % 3) * limit;
    incoming[c].Reserve(count);
    for (std::size_t i = 0; i < count; ++i)
      incoming[c].space[i] = SummaryEntry{float(i) * .5f, float(i + 1) * .5f, .5f, float(i)};
    incoming[c].SetSize(count);
    inputs.push_back({views[c], incoming[c].Entries(), limit});
  }
  auto actual = metal::UpdateSketches(inputs);
  for (std::size_t c = 0; c < n; ++c) {
    current = "batched update " + std::to_string(c);
    auto const& input = inputs[c];
    WQSummaryContainer cpu;
    cpu.Reserve(std::max(incoming[c].Size(), input.limit * 2));
    cpu.CopyFrom(incoming[c]);
    std::size_t l = 0;
    while (true) {
      if (l == storage[c].size()) storage[c].emplace_back();
      cpu.SetPrune(input.limit);
      cpu.SetCombine(storage[c][l]);
      storage[c][l].Clear();
      if (cpu.Size() <= input.limit) break;
      ++l;
    }
    storage[c][l].Reserve(cpu.Size());
    storage[c][l].CopyFrom(cpu);
    Check(actual[c].carry_level == l, "Batch carry level differs");
    Check(actual[c].levels.size() == storage[c].size(), "Batch level count differs");
    for (std::size_t j = 0; j < storage[c].size(); ++j)
      Same(actual[c].levels[j].Entries(), storage[c][j].Entries());
  }
  std::puts(
      "PASS 32 batched update cases: independent limits, levels, empty inputs and carry "
      "propagation");
}
void Containers() {
  Context cpu, gpu;
  cpu.UpdateAllowUnknown(Args{{"device", "cpu"}, {"nthread", "2"}});
  gpu.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
  for (bool grouped : {false, true}) {
    for (int mode = 0; mode < 4; ++mode) {
      current = "container mode=" + std::to_string(mode);
      std::size_t rows = 4097;
      MetaInfo info;
      info.num_row_ = rows;
      info.num_col_ = 4;
      info.feature_types.HostVector() = {FeatureType::kNumerical, FeatureType::kNumerical,
                                         FeatureType::kNumerical, FeatureType::kCategorical};
      if (grouped) info.group_ptr_ = {0, 0, 1300, 2700, unsigned(rows)};
      auto& weights = info.weights_.HostVector();
      weights.resize(grouped ? 4 : rows);
      for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] =
            mode == 1 ? 0.f : (mode == 2 ? std::ldexp(float(i % 17), -140) : float(i % 17) / 13);
      std::vector<float> hessian(rows);
      for (std::size_t i = 0; i < rows; ++i) hessian[i] = mode == 3 ? 1.f : float(i % 37) / 23;
      std::vector<SparsePage> pages(3);
      std::vector<bst_idx_t> counts(4);
      for (std::size_t p = 0; p < pages.size(); ++p) {
        auto& page = pages[p];
        auto& offsets = page.offset.HostVector();
        auto& entries = page.data.HostVector();
        offsets = {0};
        for (unsigned f = 0; f < 4; ++f) {
          std::vector<::xgboost::Entry> column;
          for (std::size_t r = p * rows / 3; r < (p + 1) * rows / 3; ++r) {
            if (f == 2 || (f == 1 && r % 3)) continue;
            float v = f == 3 ? float(r % 7) : float(int((r * 29 + f * 11) % 139) - 70) / 17;
            column.emplace_back(r, v);
          }
          std::stable_sort(column.begin(), column.end(),
                           [](auto a, auto b) { return a.fvalue < b.fvalue; });
          entries.insert(entries.end(), column.begin(), column.end());
          offsets.push_back(entries.size());
          counts[f] += column.size();
        }
      }
      HostSketchContainer a(&cpu, 17, info.feature_types.ConstHostSpan(), counts, grouped);
      HostSketchContainer b(&gpu, 17, info.feature_types.ConstHostSpan(), counts, grouped);
      auto before = metal::GetStatistics();
      for (auto const& page : pages) {
        a.PushColPage(page, info, hessian);
        b.PushColPage(page, info, hessian);
      }
      auto ac = a.MakeCuts(&cpu, info), bc = b.MakeCuts(&gpu, info);
      Check(ac.Ptrs() == bc.Ptrs(), "Container cut offsets differ");
      Check(ac.Values().size() == bc.Values().size(), "Container cut sizes differ");
      for (std::size_t i = 0; i < ac.Values().size(); ++i) Same(bc.Values()[i], ac.Values()[i]);
      auto after = metal::GetStatistics();
      Check(after.sketch_sorted_dispatches - before.sketch_sorted_dispatches == 3,
            "Container sorted dispatch");
      Check(after.sketch_weight_dispatches - before.sketch_weight_dispatches == 3,
            "Container weight dispatch");
      Check(after.sketch_update_dispatches > before.sketch_update_dispatches,
            "Container updates missing");
    }
  }
  std::puts(
      "PASS 8 production sorted HostSketchContainer cases with 3 pages and categorical/empty "
      "columns");
}
void Rejections() {
  auto throws = [](auto fn) {
    bool failed = false;
    try {
      fn();
    } catch (std::exception const&) {
      failed = true;
    }
    Check(failed, "Malformed input accepted");
  };
  throws([] { metal::MakeSketchWeights(3, {}, std::vector<float>{1}, {}); });
  throws([] { metal::MakeSketchWeights(3, {}, {}, std::vector<bst_group_t>{0, 4, 3}); });
  std::vector<::xgboost::Entry> entries{{0, 2}, {1, 1}};
  std::vector<float> weights{1, 1};
  metal::SortedSketchColumn column{entries, 1};
  throws([&] { metal::MakeSortedSketches({&column, 1}, weights); });
  column = {{}, 0};
  throws([&] { metal::MakeSortedSketches({&column, 1}, weights); });
  metal::SketchUpdate update{{}, {}, 0};
  throws([&] { metal::UpdateSketches({&update, 1}); });
  Check(metal::MakeSortedSketches({}, {}).empty(), "Empty sorted batch");
  Check(metal::UpdateSketches({}).empty(), "Empty update batch");
  for (auto device : {"cpu", "mps"}) {
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", device}, {"nthread", "1"}});
    MetaInfo info;
    info.num_row_ = 3;
    info.num_col_ = 1;
    info.weights_.HostVector() = {1};
    SparsePage page;
    HostSketchContainer container(&ctx, 2, {}, {0}, false);
    throws([&] { container.PushRowPage(page, info); });
  }
}
int main() {
  try {
    Weights();
    for (std::size_t n : {0u, 1u, 31u, 257u, 4097u, 65539u}) {
      for (unsigned mode = 0; mode < 7; ++mode) Sorted(n, mode);
      std::printf("Checked %zu sorted columns / %zu fields\n", sorted_cases, fields);
      std::fflush(stdout);
    }
    Sorted(1000003, 4);
    Streams();
    BatchedUpdates();
    Containers();
    Rejections();
    std::printf(
        "PASS %zu weight cases / %zu sorted columns / %zu streamed sketches / "
        "%zu source entries / %zu checked fields; NaNs compared by classification\n",
        weight_cases, sorted_cases, stream_cases, source_entries, fields);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
