// Exercise the production cache, including growth, reuse and empty pages.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_reduction.h"
#include "../xgboost/src/common/metal_storage.h"
#include "../xgboost/src/data/gradient_index.h"
#include "xgboost/context.h"

using namespace xgboost;
namespace metal = common::metal;
void Check(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
void Same(GradientPairPrecise actual, double g, double h) {
  Check(actual.GetGrad() == g && actual.GetHess() == h, "Cache value differs from integer oracle");
}
void Run(bool dense, int max_bin, int exponent, unsigned bins) {
  Context ctx;
  ctx.Init(Args{{"device", "mps"}, {"nthread", "1"}});
  SparsePage page;
  page.offset.HostVector() = {0, 1, 2, 3, 4};
  page.data.HostVector() = {{0, 0.f}, {0, 0.f}, {0, 0.f}, {0, bins == 2 ? 1.f : 0.f}};
  common::HistogramCuts cuts{1};
  cuts.cut_ptrs_.HostVector() = {0, bins};
  cuts.cut_values_.HostVector() =
      bins == 2 ? std::vector<float>{.5f, 2.f} : std::vector<float>{2.f};
  GHistIndexMatrix gidx{&ctx, page, {}, std::move(cuts), max_bin, dense, .2};
  float big = std::ldexp(1.f, exponent), small = std::ldexp(1.f, exponent - 54);
  std::vector<GradientPair> gradients{{big, 0}, {small, 1}, {small, 1}, {-big, 0}};
  auto saved = gradients;
  auto cache = metal::MakeHistogramBuffer();
  cache->Resize(16);
  for (auto& v : cache->Data()) v = GradientPairPrecise{17.5, -23.25};
  auto hist = metal::MakeHistogram();
  auto before = metal::GetStatistics();
  auto build = [&](std::size_t page_index, std::vector<bst_idx_t> selected, std::size_t offset) {
    common::RowSetCollection rows;
    *rows.Data() = std::move(selected);
    rows.Init();
    hist->Build(page_index, gidx, rows, {0}, gradients, {cache->Data().subspan(offset, bins)},
                cache.get());
  };
  std::size_t parent = 3, sibling = 8, child = 13;
  build(0, {0, 1, 2}, parent);
  // Rounded view cannot represent the two small terms. The integer cache must
  // retain them across growth and recover them in subtraction.
  Same(cache->Data()[parent], double(big), 2.);
  build(0, {0}, sibling);
  auto view = cache->Data();
  std::vector<GradientPairPrecise> prefix(view.begin(), view.end());
  cache->Resize(257);
  Check(!std::memcmp(cache->Data().data(), prefix.data(),
                     prefix.size() * sizeof(GradientPairPrecise)),
        "Growth changed the materialized prefix");
  metal::HistogramSubtraction task{parent, sibling, child, bins};
  metal::SubtractHistograms(cache.get(), {&task, 1});
  Same(cache->Data()[child], 2. * double(small), 2.);
  if (bins == 2) Same(cache->Data()[child + 1], 0., 0.);
  Same(metal::SumHistogram(cache.get(), child, bins), 2. * double(small), 2.);

  // Three logical pages, one containing no rows. For two bins, integer root
  // reduction must recover cancellation lost in a sum of the rounded view.
  build(1, {3}, parent);
  build(2, {}, parent);
  if (bins == 1) Same(cache->Data()[parent], 2. * double(small), 2.);
  Same(metal::SumHistogram(cache.get(), parent, bins), 2. * double(small), 2.);

  // Reuse a cache slot with an empty FIRST page, then add a nonempty page.
  build(0, {}, parent);
  for (unsigned b = 0; b < bins; ++b) Same(cache->Data()[parent + b], 0., 0.);
  build(1, {1}, parent);
  Same(cache->Data()[parent], double(small), 1.);
  for (auto i : {0u, 1u, 2u, 5u, 6u, 7u, 10u, 11u, 12u, 15u}) Same(cache->Data()[i], 17.5, -23.25);
  Check(!std::memcmp(saved.data(), gradients.data(), saved.size() * sizeof(GradientPair)),
        "Histogram mutated raw gradients");

  // Reuse buffers for a different tree and a different quantizer scale.
  hist->Reset();
  for (auto& g : gradients) g = GradientPair{g.GetGrad() * 2.f, g.GetHess() * 2.f};
  build(0, {0, 1, 2}, parent);
  build(1, {3}, parent);
  Same(metal::SumHistogram(cache.get(), parent, bins), 4. * double(small), 4.);
  auto after = metal::GetStatistics();
  Check(after.histogram_integer_store_dispatches - before.histogram_integer_store_dispatches == 8,
        "Integer stores not dispatched");
  Check(after.histogram_integer_accumulation_dispatches -
                before.histogram_integer_accumulation_dispatches ==
            4,
        "Integer page additions not dispatched");
  Check(after.histogram_integer_subtraction_dispatches -
                before.histogram_integer_subtraction_dispatches ==
            1,
        "Integer subtraction not dispatched");
  Check(after.histogram_integer_sum_dispatches - before.histogram_integer_sum_dispatches == 3,
        "Integer root reduction not dispatched");
}
int main() {
  try {
    std::size_t cases = 0;
    for (bool dense : {false, true})
      for (int max_bin : {16, 512, 65537})
        for (int exponent : {-80, 0, 54, 100})
          for (unsigned bins : {1u, 2u}) {
            Run(dense, max_bin, exponent, bins);
            ++cases;
          }
    std::printf(
        "PASS %zu production integer-cache cases: widths/layouts, growth, cancellation, "
        "subtraction, empty pages, guards, tree reuse and integer root sums\n",
        cases);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
