// Production category containers: device ownership, snapshots and CPU handoffs.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/data/cat_container.h"
#include "xgboost/json.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, checks = 0;
std::string current;
void Check(bool ok, char const* message) {
  ++checks;
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void NoTransfers(metal::Statistics const& before) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes == before.vector_readback_bytes, "Implicit host mirror");
  Check(after.vector_upload_bytes == before.vector_upload_bytes, "Repeated dictionary import");
  Check(after.category_state_staging_bytes == before.category_state_staging_bytes,
        "Category payload staging");
}
void Owned(CatContainer const& cats) {
  Check(cats.DeviceCanRead() && !cats.HostCanRead(), "Dictionary lost device authority");
}
template <typename T>
void Same(common::Span<T const> actual, common::Span<T const> expected) {
  Check(actual.size() == expected.size(), "Size differs");
  for (std::size_t i = 0; i < actual.size(); ++i)
    Check(!std::memcmp(&actual[i], &expected[i], sizeof(T)), "Payload bits differ");
}
void Same(enc::HostColumnsView actual, enc::HostColumnsView expected) {
  Check(actual.n_total_cats == expected.n_total_cats && actual.Size() == expected.Size(),
        "Dictionary shape differs");
  Same<std::int32_t>(actual.feature_segments, expected.feature_segments);
  for (std::size_t f = 0; f < actual.Size(); ++f) {
    Check(actual.columns[f].index() == expected.columns[f].index(), "Dictionary type differs");
    std::visit(
        [&](auto const& values) {
          using V = std::decay_t<decltype(values)>;
          auto const& reference = std::get<V>(expected.columns[f]);
          if constexpr (std::is_same_v<V, enc::CatStrArrayView>) {
            Same<std::int32_t>(values.offsets, reference.offsets);
            Same<enc::CatCharT>(values.values, reference.values);
          } else {
            Same<typename V::value_type>(values, reference);
          }
        },
        actual.columns[f]);
  }
}
struct Dictionary {
  cpu_impl::CatContainerImpl storage;
  std::vector<std::int32_t> segments{0};
  enc::HostColumnsView View() {
    storage.Finalize();
    segments = {0};
    for (auto const& col : storage.columns_v)
      segments.push_back(segments.back() + std::visit([](auto const& x) { return x.size(); }, col));
    return {storage.columns_v, segments, segments.back()};
  }
};
template <typename T>
void Numeric(Dictionary* out, unsigned size, bool reversed) {
  std::vector<T> values;
  auto n = sizeof(T) == 1 ? std::min(size, 113u) : size;
  for (unsigned i = 0; i < n; ++i) values.push_back(T(i * 7 + 3));
  if (reversed) std::reverse(values.begin(), values.end());
  out->storage.columns.emplace_back(std::move(values));
}
Dictionary Data(unsigned size, bool reversed) {
  Dictionary out;
  out.storage.columns.emplace_back(std::vector<std::int32_t>{});
  Numeric<std::int8_t>(&out, size, reversed);
  Numeric<std::uint8_t>(&out, size, reversed);
  Numeric<std::int16_t>(&out, size, reversed);
  Numeric<std::uint16_t>(&out, size, reversed);
  Numeric<std::int32_t>(&out, size, reversed);
  Numeric<std::uint32_t>(&out, size, reversed);
  Numeric<std::int64_t>(&out, size, reversed);
  Numeric<std::uint64_t>(&out, size, reversed);
  cpu_impl::CatStrArray strings;
  strings.offsets.push_back(0);
  std::vector<std::string> names{"", "a", std::string("a\0b", 3), "caf\xc3\xa9",
                                 std::string(4097, 'z')};
  for (unsigned i = 0; i < size; ++i) names.push_back("name" + std::to_string(i));
  if (reversed) std::reverse(names.begin(), names.end());
  for (auto const& name : names) {
    strings.values.insert(strings.values.end(), name.begin(), name.end());
    strings.offsets.push_back(strings.values.size());
  }
  // Preserve unused trailing bytes too; the dictionary snapshot is exact.
  strings.values.push_back(91);
  out.storage.columns.emplace_back(std::move(strings));
  out.storage.columns.emplace_back(cpu_impl::CatStrArray{{0}, {}});
  out.storage.columns.emplace_back(cpu_impl::CatStrArray{});
  return out;
}
Json Save(CatContainer const& cats) {
  Json json{Object{}};
  cats.Save(&json);
  return json;
}
void Run(unsigned rows) {
  current = "category state/" + std::to_string(rows);
  Context gpu, cpu;
  gpu.Init({{"device", "mps"}, {"nthread", "2"}});
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  auto source = Data(rows, false), incoming = Data(rows, true);
  auto original = source.View(), next = incoming.View();
  CatContainer reference{original, false};
  reference.Sort(&cpu);
  auto serialized = Save(reference);
  std::vector<std::int32_t> expected(next.n_total_cats);
  enc::Recode(cpu_impl::EncPolicy, original, reference.RefSortedIndex(&cpu), next,
              common::Span{expected});
  CatContainer model{&gpu, original, false}, input{&gpu, next, false};
  Owned(model);
  Owned(input);
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    auto before = metal::GetStatistics();
    model.Sort(&gpu);
    auto [acc, mapping] = cpu_impl::MakeCatAccessor(&gpu, &input, &model);
    NoTransfers(before);
    Owned(model);
    Owned(input);
    Check(mapping.Device().IsMPS() && mapping.DeviceCanRead() && !mapping.HostCanRead(),
          "Mapping acquired a host mirror");
    Same<std::int32_t>(model.RefSortedIndex(&gpu), reference.RefSortedIndex(&cpu));
    Same<std::int32_t>(acc.enc.mapping, expected);
    Same(model.MetalView().View(), original);
    ++cases;
  }
  auto before = metal::GetStatistics();
  CatContainer copied;
  copied.Copy(&gpu, model);
  NoTransfers(before);
  Owned(copied);
  Owned(model);
  // Replacing the original must not alter the copied payload or its views.
  auto other = Data(rows + 2, true);
  CatContainer replacement{other.View(), false};
  replacement.Sort(&cpu);
  auto replacement_json = Save(replacement);
  model.Load(replacement_json);
  model.Sort(&gpu);
  Same(copied.MetalView().View(), original);
  Check(Json::Dump(Save(copied)) == Json::Dump(serialized), "GPU snapshot serialization differs");
  Check(copied.HostCanRead(), "Explicit serialization failed to publish CPU mirror");
  before = metal::GetStatistics();
  (void)copied.MetalView();
  copied.Sort(&gpu);
  NoTransfers(before);  // CPU publication must not invalidate immutable GPU names.
  CatContainer host;
  host.Copy(&cpu, copied);
  Check(host.HostCanRead() && !host.DeviceCanRead(), "CPU copy retains device authority");
  Check(Json::Dump(Save(host)) == Json::Dump(serialized), "CPU copy differs");
  host.Sort(&gpu);
  before = metal::GetStatistics();
  host.Sort(&gpu);
  NoTransfers(before);
  // Loading into an existing GPU-backed object replaces both representations.
  for (auto const& json : {serialized, replacement_json, serialized}) {
    model.Load(json);
    model.Sort(&gpu);
    Check(Json::Dump(Save(model)) == Json::Dump(json), "Reload appended stale dictionary columns");
    ++cases;
  }
  // Concurrent immutable views/caches are shared; each recoding owns its output.
  model.Load(serialized);
  model.Sort(&gpu);
  (void)input.MetalView();
  before = metal::GetStatistics();
  std::atomic<bool> okay{true};
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < 4; ++t) {
    threads.emplace_back([&] {
      try {
        for (unsigned r = 0; r < 4; ++r) {
          auto [acc, mapping] = cpu_impl::MakeCatAccessor(&gpu, &input, &model);
          if (mapping.HostCanRead() ||
              !std::equal(expected.begin(), expected.end(), acc.enc.mapping.begin()))
            okay = false;
        }
      } catch (...) {
        okay = false;
      }
    });
  }
  for (auto& thread : threads) thread.join();
  Check(okay, "Concurrent category recoding differs");
  NoTransfers(before);
  ++cases;
  std::printf("Checked %s: %zu cases / %zu checks\n", current.c_str(), cases, checks);
  std::fflush(stdout);
}
void Rejection() {
  current = "failed recoding must preserve output";
  Context gpu;
  gpu.Init({{"device", "mps"}});
  auto data = Data(31, false), missing = Data(32, false);
  CatContainer original{&gpu, data.View(), false}, incoming{&gpu, missing.View(), false};
  original.Sort(&gpu);
  HostDeviceVector<std::int32_t> mapping(incoming.NumCatsTotal(), 73, DeviceOrd::MPS());
  auto output = mapping.DeviceSpan();
  bool rejected = false;
  try {
    original.MetalView().Recode(original.RefSortedIndex(&gpu), incoming.MetalView(), output);
  } catch (dmlc::Error const&) {
    rejected = true;
  }
  Check(rejected, "Unknown category accepted");
  for (auto value : mapping.ConstDeviceSpan())
    Check(value == 73, "Failed recoding published partial output");
  ++cases;
}
void Bindings() {
  current = "guarded mapping, aliases and fallback";
  auto source = Data(31, false), incoming = Data(31, true);
  auto original = source.View(), next = incoming.View();
  std::vector<std::int32_t> order(original.n_total_cats), expected(next.n_total_cats);
  enc::SortNames(cpu_impl::EncPolicy, original, order);
  enc::Recode(cpu_impl::EncPolicy, original, order, next, common::Span{expected});
  auto a = metal::MakeCategoryDictionary(original), b = metal::MakeCategoryDictionary(next);
  HostDeviceVector<std::int32_t> sorted{order, DeviceOrd::MPS()};
  HostDeviceVector<std::int32_t> output(expected.size() + 8, -71, DeviceOrd::MPS());
  auto destination = output.DeviceSpan().subspan(4, expected.size());
  auto sorted_view = sorted.DeviceSpan();
  auto before = metal::GetStatistics();
  a->Recode(sorted_view, *b, destination);
  NoTransfers(before);
  Same<std::int32_t>(destination, expected);
  for (std::size_t i = 0; i < output.Size(); ++i)
    if (i < 4 || i >= 4 + expected.size())
      Check(output.ConstDeviceSpan()[i] == -71, "Mapping guard");
  before = metal::GetStatistics();
  a->Recode(sorted_view, *b, sorted_view);
  NoTransfers(before);
  Same<std::int32_t>(sorted.ConstDeviceSpan(), expected);
  auto page = std::size_t(getpagesize());
  auto* memory = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  Check(memory != MAP_FAILED && order.size() * sizeof(std::int32_t) <= page, "mmap failed");
  std::memcpy(memory, order.data(), order.size() * sizeof(std::int32_t));
  Check(mprotect(memory, page, PROT_READ) == 0, "mprotect failed");
  std::vector<unsigned char> raw(expected.size() * sizeof(std::int32_t) + 8, 0xa5);
  before = metal::GetStatistics();
  a->Recode({static_cast<std::int32_t const*>(memory), order.size()}, *b,
            {reinterpret_cast<std::int32_t*>(raw.data() + 2), expected.size()});
  Check(metal::GetStatistics().category_state_staging_bytes - before.category_state_staging_bytes ==
            (order.size() + expected.size()) * sizeof(std::int32_t),
        "Fallback byte accounting");
  Check(!std::memcmp(raw.data() + 2, expected.data(), expected.size() * sizeof(std::int32_t)),
        "Fallback mapping differs");
  for (std::size_t i = 0; i < raw.size(); ++i)
    if (i < 2 || i >= 2 + expected.size() * sizeof(std::int32_t))
      Check(raw[i] == 0xa5, "Fallback guard");
  Check(munmap(memory, page) == 0, "munmap failed");
  ++cases;
}
int main() {
  try {
    for (unsigned rows : {0u, 1u, 31u, 257u, 4097u}) Run(rows);
    Rejection();
    Bindings();
    auto stats = metal::GetStatistics();
    Check(stats.vector_live_buffers == 0 && stats.vector_capacity_bytes == 0,
          "Dictionary allocations leaked after owners died");
    std::printf("PASS: %zu category state cases / %zu checks\n", cases, checks);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
