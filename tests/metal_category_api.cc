// Typed category dictionaries versus source CPU algorithms and known name maps.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_category.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/data/cat_container.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, errors = 0;
void Check(bool ok, char const* text) {
  if (!ok) throw std::runtime_error(text);
}
struct Dictionary {
  cpu_impl::CatContainerImpl storage;
  std::vector<std::int32_t> segments;
  enc::HostColumnsView View() {
    storage.Finalize();
    segments = {0};
    for (auto const& col : storage.columns_v)
      segments.push_back(segments.back() + std::visit([](auto const& x) { return x.size(); }, col));
    return {storage.columns_v, segments, segments.back()};
  }
};
cpu_impl::CatStrArray Strings(std::vector<std::string> const& names) {
  cpu_impl::CatStrArray result;
  result.offsets.push_back(0);
  for (auto const& name : names) {
    result.values.insert(result.values.end(), name.begin(), name.end());
    result.offsets.push_back(result.values.size());
  }
  return result;
}
template <class T>
void Numeric(Dictionary* dict, unsigned n, std::mt19937_64* rng) {
  std::vector<T> values;
  for (unsigned i = 0; i < n; ++i) {
    std::uint64_t bits = std::uint64_t(i) * 0x9e3779b97f4a7c15ull;
    T value;
    std::memcpy(&value, &bits, sizeof(T));
    values.push_back(value);
  }
  if (n) {
    values.push_back(std::numeric_limits<T>::lowest());
    values.push_back(std::numeric_limits<T>::max());
    values.push_back(T(1));
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  std::shuffle(values.begin(), values.end(), *rng);
  dict->storage.columns.emplace_back(std::move(values));
  dict->storage.columns.emplace_back(std::vector<std::int32_t>{});
}
Dictionary Data(unsigned n, std::mt19937_64* rng) {
  Dictionary result;
  result.storage.columns.emplace_back(std::vector<std::int32_t>{});
  Numeric<std::uint8_t>(&result, n, rng);
  Numeric<std::int8_t>(&result, n, rng);
  Numeric<std::uint16_t>(&result, n, rng);
  Numeric<std::int16_t>(&result, n, rng);
  Numeric<std::uint32_t>(&result, n, rng);
  Numeric<std::int32_t>(&result, n, rng);
  Numeric<std::uint64_t>(&result, n, rng);
  Numeric<std::int64_t>(&result, n, rng);
  std::vector<std::string> names;
  for (unsigned i = 0; i < n; ++i) names.push_back(std::string(i % 19, 'a') + std::to_string(i));
  if (n) {
    for (auto name : {std::string{}, std::string("a"), std::string("a\0b", 3),
                      std::string("a\0", 2), std::string("\x80"), std::string("\xff"),
                      std::string("caf\xc3\xa9"), std::string(4097, 'z')})
      names.push_back(name);
  }
  std::shuffle(names.begin(), names.end(), *rng);
  result.storage.columns.emplace_back(Strings(names));
  result.storage.columns.emplace_back(cpu_impl::CatStrArray{});
  return result;
}
void Run(unsigned n, unsigned mode, std::mt19937_64* rng) {
  auto original = Data(n, rng);
  Dictionary incoming;
  std::vector<std::int32_t> known;
  for (auto const& col : original.storage.columns) {
    std::visit(
        [&](auto const& values) {
          std::vector<std::int32_t> order(values.size());
          std::iota(order.begin(), order.end(), 0);
          std::shuffle(order.begin(), order.end(), *rng);
          if (mode == 1 && !order.empty()) order.resize((order.size() + 1) / 2);
          if (mode == 2 && !order.empty()) order = {order.back()};
          using T = std::decay_t<decltype(values)>;
          if constexpr (std::is_same_v<T, cpu_impl::CatStrArray>) {
            if (values.offsets.empty())
              incoming.storage.columns.emplace_back(T{});
            else {
              std::vector<std::string> names;
              for (auto i : order)
                names.emplace_back(
                    reinterpret_cast<char const*>(values.values.data() + values.offsets[i]),
                    values.offsets[i + 1] - values.offsets[i]);
              incoming.storage.columns.emplace_back(Strings(names));
            }
          } else {
            T subset;
            for (auto i : order) subset.push_back(values[i]);
            incoming.storage.columns.emplace_back(std::move(subset));
          }
          known.insert(known.end(), order.begin(), order.end());
        },
        col);
  }
  auto orig = original.View(), next = incoming.View();
  std::vector<std::int32_t> expected(orig.n_total_cats), sorted(expected.size() + 2, -919);
  enc::SortNames(enc::DftHostPolicy{}, orig, expected);
  auto before = metal::GetStatistics();
  metal::SortCategories(orig, {sorted.data() + 1, expected.size()});
  Check(metal::GetStatistics().category_sort_dispatches - before.category_sort_dispatches ==
            bool(orig.n_total_cats),
        "Sort counter");
  Check(sorted.front() == -919 && sorted.back() == -919, "Sort guard");
  Check(std::equal(expected.begin(), expected.end(), sorted.begin() + 1),
        "Typed name order differs from source");
  fields += expected.size();
  std::vector<std::int32_t> mapping(known.size() + 2, -919), cpu(known.size());
  enc::Recode(enc::DftHostPolicy{}, orig, expected, next, cpu);
  Check(cpu == known, "Known name map disagrees with source oracle");
  before = metal::GetStatistics();
  metal::RecodeCategories(orig, {sorted.data() + 1, expected.size()}, next,
                          {mapping.data() + 1, known.size()});
  Check(metal::GetStatistics().category_recode_dispatches - before.category_recode_dispatches == 1,
        "Recode counter");
  Check(mapping.front() == -919 && mapping.back() == -919, "Mapping guard");
  Check(std::equal(known.begin(), known.end(), mapping.begin() + 1), "Name map differs");
  fields += known.size();
  if (n && mode == 0) {
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", "mps"}});
    CatContainer categories(orig, false), copied;
    copied.Copy(&ctx, categories);
    before = metal::GetStatistics();
    copied.Sort(&ctx);
    auto [accessor, actual] = cpu_impl::MakeCatAccessor(&ctx, next, &copied);
    Check(actual == known, "Container recoding differs");
    Check(metal::GetStatistics().category_sort_dispatches == before.category_sort_dispatches + 1 &&
              metal::GetStatistics().category_recode_dispatches ==
                  before.category_recode_dispatches + 1,
          "Container bypassed GPU");
    fields += actual.size();
    for (std::size_t f = 0; f < next.Size(); ++f) {
      auto segment = enc::MappingView{next.feature_segments, actual}[f];
      for (std::size_t i = 0; i < segment.size(); ++i)
        Check(segment[i] == known[next.feature_segments[f] + i], "Accessor mapping");
    }
  }
  ++cases;
  std::printf("PASS n%u mode%u: %zu cases / %zu indices\n", n, mode, cases, fields);
  std::fflush(stdout);
}
template <class Fn>
void Reject(Fn fn, std::string const& message, bool dispatched) {
  auto before = metal::GetStatistics();
  bool rejected = false;
  try {
    fn();
  } catch (std::exception const& e) {
    rejected = std::string(e.what()).find(message) != std::string::npos;
    if (!rejected) std::fprintf(stderr, "Wrong error: %s\n", e.what());
  }
  Check(rejected, "Missing expected error");
  Check(metal::GetStatistics().category_recode_dispatches - before.category_recode_dispatches ==
            dispatched,
        "Error dispatch count");
  ++errors;
}
void ErrorCases() {
  Dictionary original, incoming;
  original.storage.columns = {std::vector<std::uint64_t>{9007199254740993ull, 9007199254740992ull,
                                                         std::numeric_limits<std::uint64_t>::max()},
                              std::vector<std::int32_t>{2}};
  incoming.storage.columns = {
      std::vector<std::uint64_t>{9007199254740994ull, 9007199254740993ull, 4},
      std::vector<std::int32_t>{7}};
  auto o = original.View(), n = incoming.View();
  std::vector<std::int32_t> sorted(o.n_total_cats), mapping(n.n_total_cats, -991);
  enc::SortNames(enc::DftHostPolicy{}, o, sorted);
  Reject([&] { metal::RecodeCategories(o, sorted, n, mapping); },
         "0th (0-based) column: `9007199254740994`", true);
  Check(std::all_of(mapping.begin(), mapping.end(), [](int x) { return x == -991; }),
        "Failed map copied to host");
  // CUDA checks all feature types before searching any names. CPU's per-feature
  // loop would instead report the earlier missing category here.
  incoming.storage.columns[1] = std::vector<std::uint32_t>{2};
  n = incoming.View();
  Reject([&] { metal::RecodeCategories(o, sorted, n, mapping); }, "data type doesn't match", true);
  Reject([&] { metal::RecodeCategories(o, sorted, n, {}); }, "`mapping` should have", false);
  Reject([&] { metal::RecodeCategories(o, {}, n, mapping); }, "`sorted_idx` should have", false);
  auto bad = n;
  bad.columns = bad.columns.first(1);
  Reject([&] { metal::RecodeCategories(o, sorted, bad, mapping); }, "same number of columns",
         false);
  bad = n;
  bad.feature_segments = bad.feature_segments.first(1);
  Reject([&] { metal::RecodeCategories(o, sorted, bad, mapping); }, "Invalid new encoding", false);

  std::mt19937_64 rng(99);
  auto all = Data(3, &rng);
  std::vector<cpu_impl::ColumnType> types;
  for (auto const& col : all.storage.columns)
    if (std::visit([](auto const& x) { return x.size() != 0; }, col)) types.push_back(col);
  Check(types.size() == 9, "All source category types");
  for (std::size_t a = 0; a < types.size(); ++a)
    for (std::size_t b = 0; b < types.size(); ++b) {
      if (a == b) continue;
      Dictionary l, r;
      l.storage.columns = {types[a]};
      r.storage.columns = {types[b]};
      auto lv = l.View(), rv = r.View();
      std::vector<std::int32_t> order(lv.n_total_cats), result(rv.n_total_cats);
      enc::SortNames(enc::DftHostPolicy{}, lv, order);
      Reject([&] { metal::RecodeCategories(lv, order, rv, result); }, "data type doesn't match",
             true);
    }
  Dictionary l, r;
  l.storage.columns = {Strings({})};
  r.storage.columns = {cpu_impl::CatStrArray{}};
  auto lv = l.View(), rv = r.View();
  Reject([&] { metal::RecodeCategories(lv, {}, rv, {}); }, "data type doesn't match", true);
  l.storage.columns = {Strings({"", "prefix", "prefixed"})};
  r.storage.columns = {Strings({"prefix", "unknown", "another"})};
  lv = l.View();
  rv = r.View();
  std::vector<std::int32_t> order{0, 1, 2}, result(3);
  Reject([&] { metal::RecodeCategories(lv, order, rv, result); }, "column: `unknown`", true);
  Dictionary empty;
  auto v = empty.View();
  auto before = metal::GetStatistics();
  metal::SortCategories(v, {});
  metal::RecodeCategories(v, {}, v, {});
  Check(metal::GetStatistics().category_sort_dispatches == before.category_sort_dispatches &&
            metal::GetStatistics().category_recode_dispatches == before.category_recode_dispatches,
        "Empty dictionary dispatched");
}
int main() {
  try {
    std::mt19937_64 rng(4973);
    for (unsigned n : {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 4097u, 65537u})
      for (unsigned mode : {0u, 1u, 2u}) Run(n, mode, &rng);
    ErrorCases();
    std::printf("PASS %zu dictionary cases / %zu exact indices / %zu expected errors\n", cases,
                fields, errors);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
