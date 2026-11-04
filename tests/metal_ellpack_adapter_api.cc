// Direct resident adapter -> CUDA ELLPACK bytes, with independent scalar bin search.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/compressed_iterator.h"
#include "../xgboost/src/common/metal_adapter.h"
#include "../xgboost/src/common/metal_ellpack.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/data/gradient_index.h"
using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, windows = 0, fields = 0, bytes_checked = 0, errors = 0;
void Check(bool ok, char const* reason) {
  ++fields;
  if (!ok) throw std::runtime_error(reason);
}
template <class T>
data::ArrayAdapter MakeArrayAdapter(std::vector<T> const& values, std::size_t rows, unsigned cols) {
  std::ostringstream json;
  json << "{\"version\":3,\"typestr\":\"<f" << sizeof(T) << "\",\"data\":["
       << reinterpret_cast<std::uintptr_t>(values.data()) << ",false],\"shape\":[" << rows << ","
       << cols << "],\"strides\":null}";
  auto s = json.str();
  return data::ArrayAdapter{StringView{s}};
}
template <class T>
void Case(unsigned maximum, unsigned mode, std::size_t window, std::size_t rows = 19,
          bool empty_cuts = false) {
  constexpr unsigned cols = 7;
  common::HistogramCuts cuts{0};
  auto& ptr = cuts.cut_ptrs_.HostVector();
  auto& values = cuts.cut_values_.HostVector();
  ptr.assign(1, 0);
  std::vector<FeatureType> types(cols, FeatureType::kNumerical);
  types[2] = FeatureType::kCategorical;
  for (unsigned f = 0; f < cols; ++f) {
    auto width = empty_cuts ? 0u : std::max(1u, maximum - std::min(maximum - 1, f));
    for (unsigned j = 0; j < width; ++j) values.push_back(float(j) * (f == 2 ? 1 : 2));
    ptr.push_back(values.size());
  }
  std::vector<T> raw(rows * cols, -99);
  std::vector<std::vector<std::uint32_t>> bins(rows);
  std::vector<std::size_t> row_ptr{0}, hits(values.size(), 0);
  std::vector<std::uint32_t> global_bins;
  unsigned stride = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    for (unsigned f = 0; f < cols; ++f) {
      bool valid = mode == 0 || (mode == 1 && (r == 0 || (r + f) % 3)) ||
                   (mode == 2 && f < cols - 1 && (r + f) % 3 == 0);
      if (!valid) {
        bins[r].push_back(~0u);
        continue;
      }
      T v = T((r * 13 + f * 5) % (2 * maximum + 3)) - T(1);
      if (r + 1 == rows) v = T(maximum * 4);
      if (f == 2) v = std::max(T(0), v) + T(.75);
      if (r == 1 && f == 0) v = std::numeric_limits<float>::denorm_min();
      if (r == 2 && f == 0) v = -T(0);
      raw[r * cols + f] = v;
      float query = float(v);
      if (f == 2) query = float(static_cast<int>(query));
      auto first = values.begin() + ptr[f], last = values.begin() + ptr[f + 1];
      auto it =
          f == 2 ? std::lower_bound(first, last, query) : std::upper_bound(first, last, query);
      auto bin =
          static_cast<std::uint32_t>(std::min(it - values.begin(), last - values.begin() - 1));
      bins[r].push_back(bin);
      global_bins.push_back(bin);
      ++hits[bin];
    }
    stride = std::max(stride, unsigned(global_bins.size() - row_ptr.back()));
    row_ptr.push_back(global_bins.size());
  }
  auto adapter = MakeArrayAdapter(raw, rows, cols);
  auto layout = metal::MakeEllpackInfo(cuts, stride, mode == 0);
  auto expected_stride = mode == 0 ? cols : stride;
  auto symbols = empty_cuts       ? 1u
                 : mode == 0      ? maximum
                 : stride == cols ? maximum + 1
                                  : ptr.back() + 1;
  Check(layout.row_stride == expected_stride && layout.symbols == symbols, "Independent geometry");
  auto page = metal::MakeCompressedPage(layout, rows, 73);
  auto before = metal::GetStatistics();
  page->BeginAdapter(cuts, types);
  for (std::size_t begin = 0; begin < raw.size(); begin += window) {
    auto input = metal::PrepareAdapter(adapter, -99, begin, std::min(window, raw.size() - begin));
    page->PushAdapter(*input);
    ++windows;
  }
  page->EndAdapter();
  auto after = metal::GetStatistics();
  Check(after.ellpack_adapter_values - before.ellpack_adapter_values == global_bins.size(),
        "GPU binning value count");
  Check(after.adapter_debug_readback_bytes == before.adapter_debug_readback_bytes &&
            after.ellpack_debug_readback_bytes == before.ellpack_debug_readback_bytes &&
            after.ellpack_pack_input_bytes == before.ellpack_pack_input_bytes &&
            after.adapter_index_dispatches == before.adapter_index_dispatches &&
            after.binning_dispatches == before.binning_dispatches,
        "No raw readback, host bins or host-bin re-upload");
  if (rows) Check(after.ellpack_adapter_dispatches > before.ellpack_adapter_dispatches, "GPU work");
  std::vector<std::uint8_t> expected(
      common::CompressedBufferWriter::CalculateBufferSize(rows * expected_stride, symbols), 0);
  common::CompressedBufferWriter writer{symbols};
  for (std::size_t r = 0; r < rows; ++r) {
    unsigned slot = 0;
    for (unsigned f = 0; f < cols; ++f) {
      auto bin = bins[r][f];
      if (expected_stride == cols) {
        writer.WriteSymbol(expected.data(), bin == ~0u ? symbols - 1 : bin - ptr[f], r * cols + f);
      } else if (bin != ~0u) {
        writer.WriteSymbol(expected.data(), bin, r * expected_stride + slot++);
      }
    }
    if (expected_stride != cols)
      while (slot < expected_stride)
        writer.WriteSymbol(expected.data(), symbols - 1, r * expected_stride + slot++);
  }
  page->Commit(.23);
  Check(page->Readback() == expected, "Exact direct CUDA bitstream");
  bytes_checked += expected.size();
  Context ctx;
  ctx.Init({{"nthread", "1"}});
  GHistIndexMatrix restored;
  page->Restore(&ctx, cuts, maximum, .2, cases % 2, &restored);
  Check(restored.base_rowid == 73 && restored.Size() == rows, "Restored shape");
  Check(restored.index.Size() == global_bins.size() && bool(restored.metal_page),
        "Resident GPU bins");
  for (std::size_t j = 0; j < row_ptr.size(); ++j)
    Check(restored.row_ptr[j] == row_ptr[j], "Row pointer");
  for (std::size_t j = 0; j < global_bins.size(); ++j)
    Check(restored.index[j] == global_bins[j], "Independent scalar bin");
  for (std::size_t j = 0; j < hits.size(); ++j)
    Check(restored.hit_count[j] == hits[j], "Hit count");
  ++cases;
}
void Reject(std::function<void()> fn) {
  bool rejected = false;
  try {
    fn();
  } catch (std::exception const&) {
    rejected = true;
  }
  Check(rejected, "Invalid construction must fail");
  ++errors;
}
void Errors() {
  common::HistogramCuts cuts{0};
  cuts.cut_ptrs_.HostVector() = {0, 2, 4};
  cuts.cut_values_.HostVector() = {0, 2, 0, 2};
  std::vector<float> raw{1, 1, 1, 1};
  auto adapter = MakeArrayAdapter(raw, 2, 2);
  auto layout = metal::MakeEllpackInfo(cuts, 2, true);
  auto fresh = [&] {
    auto p = metal::MakeCompressedPage(layout, 2, 0);
    p->BeginAdapter(cuts, {});
    return p;
  };
  auto p = fresh();
  Reject([&] { p->EndAdapter(); });
  Reject([&] { p->Commit(0); });
  Reject([&] { p->BeginAdapter(cuts, {}); });
  auto late = metal::PrepareAdapter(adapter, -99, 1, 1);
  Reject([&] { p->PushAdapter(*late); });
  auto first = metal::PrepareAdapter(adapter, -99, 0, 1);
  p->PushAdapter(*first);
  Reject([&] { p->PushAdapter(*first); });
  auto target = metal::MakeCompressedPage(layout, 2, 0);
  Reject([&] { target->Copy(*p, 0); });
  Reject([&] { target->EndAdapter(); });
  auto other = MakeArrayAdapter(raw, 1, 4);
  auto wrong = metal::PrepareAdapter(other, -99, 0, 4);
  Reject([&] { fresh()->PushAdapter(*wrong); });
  raw[0] = std::numeric_limits<float>::infinity();
  auto inf = metal::PrepareAdapter(adapter, -99, 0, 4);
  Reject([&] { fresh()->PushAdapter(*inf); });
  raw[0] = -99;
  auto missing = metal::PrepareAdapter(adapter, -99, 0, 4);
  Reject([&] { fresh()->PushAdapter(*missing); });
  raw[0] = 1;
  auto full = metal::PrepareAdapter(adapter, -99, 0, 4);
  auto empty = cuts;
  empty.cut_ptrs_.HostVector() = {0, 0, 4};
  auto no_cut = metal::MakeCompressedPage(layout, 2, 0);
  no_cut->BeginAdapter(empty, {});
  Reject([&] { no_cut->PushAdapter(*full); });
  auto narrow_layout = metal::MakeEllpackInfo(cuts, 1, false);
  auto narrow = metal::MakeCompressedPage(narrow_layout, 2, 0);
  narrow->BeginAdapter(cuts, {});
  Reject([&] { narrow->PushAdapter(*full); });
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    for (unsigned maximum : {3u, 17u, 257u, 65537u}) {
      for (unsigned mode = 0; mode < 4; ++mode) {
        Case<float>(maximum, mode, 1);
        Case<double>(maximum, mode, 5);
        Case<float>(maximum, mode, 19);
        Case<double>(maximum, mode, 4096);
      }
      std::printf("Checked %zu cases / %zu windows / %zu exact bytes / %zu fields\n", cases,
                  windows, bytes_checked, fields);
    }
    for (unsigned mode = 0; mode < 4; ++mode) {
      Case<float>(257, mode, 1025, 4097);
      Case<float>(17, mode, 1, 0);
    }
    Case<float>(17, 3, 1, 19, true);
    Errors();
    std::printf(
        "PASS %zu direct adapter cases / %zu windows / %zu exact bytes / %zu fields / %zu expected errors\n",
        cases, windows, bytes_checked, fields, errors);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL case %zu: %s\n", cases, e.what());
    return 1;
  }
}
