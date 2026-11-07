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
#include "integer_cache_contract_io.h"

using namespace xgboost;
namespace metal = common::metal;
namespace hc = integer_cache_contracts;
std::unique_ptr<hc::Exporter> report;
unsigned record_index=0;
std::vector<float> Flatten(std::vector<GradientPair> const& input) {
  std::vector<float> out;for(auto v:input){out.push_back(v.GetGrad());out.push_back(v.GetHess());}return out;
}
void Append(std::vector<std::uint64_t>* out,GradientPairPrecise v) {
  out->push_back(hc::Bits<std::uint64_t>(v.GetGrad()));out->push_back(hc::Bits<std::uint64_t>(v.GetHess()));
}
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
  hc::Input input{{record_index++,unsigned(dense),unsigned(max_bin),std::uint32_t(exponent),bins},{},{},{},{}};
  hc::Observation observed;
  if(report) {
    for(auto e:page.data.ConstHostVector())input.data.push_back(e.fvalue);
    input.cuts=gidx.cut.Values();input.first=Flatten(gradients);
    for(unsigned r=0;r<4;++r)observed.bins.push_back(gidx.index[r]);
  }
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
    if(report)for(auto v:cache->Data().subspan(offset,bins))Append(&observed.values,v);
  };
  std::size_t parent = 3, sibling = 8, child = 13;
  build(0, {0, 1, 2}, parent);
  // Rounded view cannot represent the two small terms. The integer cache must
  // retain them across growth and recover them in subtraction.
  Same(cache->Data()[parent], double(big), 2.);
  build(0, {0}, sibling);
  auto view = cache->Data();
  std::vector<GradientPairPrecise> prefix(view.begin(), view.end());
  if(report)for(auto v:prefix)Append(&observed.prefix_before,v);
  cache->Resize(257);
  Check(!std::memcmp(cache->Data().data(), prefix.data(),
                     prefix.size() * sizeof(GradientPairPrecise)),
        "Growth changed the materialized prefix");
  if(report)for(auto v:cache->Data().first(prefix.size()))Append(&observed.prefix_after,v);
  metal::HistogramSubtraction task{parent, sibling, child, bins};
  metal::SubtractHistograms(cache.get(), {&task, 1});
  if(report)for(auto v:cache->Data().subspan(child,bins))Append(&observed.values,v);
  Same(cache->Data()[child], 2. * double(small), 2.);
  if (bins == 2) Same(cache->Data()[child + 1], 0., 0.);
  auto child_sum=metal::SumHistogram(cache.get(), child, bins);
  Same(child_sum, 2. * double(small), 2.);
  if(report)Append(&observed.values,child_sum);

  // Three logical pages, one containing no rows. For two bins, integer root
  // reduction must recover cancellation lost in a sum of the rounded view.
  build(1, {3}, parent);
  build(2, {}, parent);
  if (bins == 1) Same(cache->Data()[parent], 2. * double(small), 2.);
  auto parent_sum=metal::SumHistogram(cache.get(), parent, bins);
  Same(parent_sum, 2. * double(small), 2.);
  if(report)Append(&observed.values,parent_sum);

  // Reuse a cache slot with an empty FIRST page, then add a nonempty page.
  build(0, {}, parent);
  for (unsigned b = 0; b < bins; ++b) Same(cache->Data()[parent + b], 0., 0.);
  build(1, {1}, parent);
  Same(cache->Data()[parent], double(small), 1.);
  for (auto i : {0u, 1u, 2u, 5u, 6u, 7u, 10u, 11u, 12u, 15u}) {
    Same(cache->Data()[i], 17.5, -23.25);
    if(report)Append(&observed.guards,cache->Data()[i]);
  }
  Check(!std::memcmp(saved.data(), gradients.data(), saved.size() * sizeof(GradientPair)),
        "Histogram mutated raw gradients");

  // Reuse buffers for a different tree and a different quantizer scale.
  hist->Reset();
  for (auto& g : gradients) g = GradientPair{g.GetGrad() * 2.f, g.GetHess() * 2.f};
  if(report)input.second=Flatten(gradients);
  build(0, {0, 1, 2}, parent);
  build(1, {3}, parent);
  auto next_sum=metal::SumHistogram(cache.get(), parent, bins);
  Same(next_sum, 4. * double(small), 4.);
  if(report)Append(&observed.values,next_sum);
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
  if(report)report->Add(input,observed);
}
void CheckOutputLayouts(bool dense, int max_bin, bool empty) {
  Context ctx;
  ctx.Init(Args{{"device", "mps"}, {"nthread", "1"}});
  SparsePage page;
  page.offset.HostVector() = {0, 1, 2, 3, 4};
  page.data.HostVector() = {{0, 0.f}, {0, 1.f}, {0, 0.f}, {0, 1.f}};
  common::HistogramCuts cuts{1};
  cuts.cut_ptrs_.HostVector() = {0, 2};
  cuts.cut_values_.HostVector() = {.5f, 2.f};
  GHistIndexMatrix gidx{&ctx, page, {}, std::move(cuts), max_bin, dense, .2};
  std::vector<GradientPair> gradients{{1.5f, 1}, {-2.25f, 2}, {3.f, 3}, {-4.5f, 4}};
  common::RowSetCollection rows;
  *rows.Data() = {0, 1, 2, 3};
  rows.Init();
  rows.AddSplit(0, 1, 2, 3, 1);
  rows.AddSplit(1, 3, 4, empty ? 0 : 1, empty ? 3 : 2);
  std::vector<bst_node_t> nodes{3, 4, 2};
  for (std::vector<std::size_t> offsets :
       {std::vector<std::size_t>{3, 5, 7}, {3, 8, 13}, {13, 8, 3}}) {
    auto cache = metal::MakeHistogramBuffer();
    cache->Resize(20);
    for (auto& value : cache->Data()) value = GradientPairPrecise{17.5, -23.25};
    auto hist = metal::MakeHistogram();
    std::vector<common::GHistRow> outputs;
    for (auto offset : offsets) outputs.push_back(cache->Data().subspan(offset, 2));
    // First-page replacement, later-page accumulation, then replacement again.
    for (unsigned pass = 0; pass < 3; ++pass) {
      hist->Build(pass == 1 ? 1 : 0, gidx, rows, nodes, gradients, outputs, cache.get());
      std::vector<GradientPairPrecise> expected(20, GradientPairPrecise{17.5, -23.25});
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        expected[offsets[i]] = expected[offsets[i] + 1] = GradientPairPrecise{};
        for (auto row = rows[nodes[i]].begin(); row != rows[nodes[i]].end(); ++row) {
          auto bin = *row % 2;
          auto const& value = gradients[*row];
          double factor = pass == 1 ? 2. : 1.;
          expected[offsets[i] + bin] += GradientPairPrecise{
              factor * value.GetGrad(), factor * value.GetHess()};
        }
        auto sum = metal::SumHistogram(cache.get(), offsets[i], 2);
        auto total = expected[offsets[i]] + expected[offsets[i] + 1];
        Same(sum, total.GetGrad(), total.GetHess());
      }
      Check(!std::memcmp(cache->Data().data(), expected.data(), expected.size() * sizeof(expected[0])),
            "Histogram output layout changed a value or guard");
    }
  }
}
int main(int argc,char** argv) {
  try {
    report=hc::ReportOption(argc,argv);
    std::size_t cases = 0;
    for (bool dense : {false, true})
      for (int max_bin : {16, 512, 65537})
        for (int exponent : {-80, 0, 54, 100})
          for (unsigned bins : {1u, 2u}) {
            Run(dense, max_bin, exponent, bins);
            ++cases;
          }
    for (bool dense : {false, true})
      for (int max_bin : {16, 512, 65537})
        for (bool empty : {false, true}) CheckOutputLayouts(dense, max_bin, empty);
    std::puts("PASS 108 histogram layout/page cases: contiguous, gapped, reversed, "
              "empty nodes, integer sums and untouched guards");
    if(report)report->Finish();
    std::printf(
        "PASS %zu production integer-cache cases: widths/layouts, growth, cancellation, "
        "subtraction, empty pages, guards, tree reuse and integer root sums\n",
        cases);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
