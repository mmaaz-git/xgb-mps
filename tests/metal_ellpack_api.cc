// CUDA's CPU bit writer is the byte oracle; bin generation is independent of Metal.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/column_matrix.h"
#include "../xgboost/src/common/compressed_iterator.h"
#include "../xgboost/src/common/metal_ellpack.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/data/gradient_index.h"
using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, bytes_checked = 0;
void Check(bool ok, char const* what) {
  ++fields;
  if (!ok) throw std::runtime_error(what);
}
struct Fixture {
  Context ctx;
  common::HistogramCuts cuts{0};
  std::vector<std::uint32_t> widths;
  unsigned mode, maximum;
  bool Dense() const { return mode == 0; }
  bool Valid(std::size_t row, std::size_t feature) const {
    return mode == 0 || (mode == 1 && (row == 0 || (row + feature) % 5 != 0)) ||
           (mode == 2 && (row + feature) % 5 == 0);
  }
  std::uint32_t Bin(std::size_t row, std::size_t feature) const {
    return (row * 17 + feature * 31) % widths[feature];
  }
  Fixture(unsigned bins, unsigned layout) : mode{layout}, maximum{bins} {
    ctx.Init({{"nthread", "1"}});
    auto& ptr = cuts.cut_ptrs_.HostVector();
    auto& values = cuts.cut_values_.HostVector();
    ptr.assign(1, 0);
    for (unsigned f = 0; f < 7; ++f) {
      widths.push_back(std::max(1u, bins - std::min(bins - 1, f)));
      for (unsigned j = 0; j < widths.back(); ++j) values.push_back(float(j));
      ptr.push_back(values.size());
    }
  }
  std::shared_ptr<GHistIndexMatrix> Index(std::size_t start, std::size_t rows) {
    auto result = std::make_shared<GHistIndexMatrix>(rows, start + 73, common::HistogramCuts{cuts},
                                                     maximum, Dense());
    std::vector<std::uint32_t> bins;
    for (std::size_t r = 0; r < rows; ++r) {
      for (unsigned f = 0; f < 7; ++f)
        if (Valid(start + r, f)) {
          bins.push_back(cuts.Ptrs()[f] + Bin(start + r, f));
        }
      result->row_ptr[r + 1] = bins.size();
    }
    result->ResizeIndex(&ctx, bins.size(), Dense());
    if (Dense()) result->index.SetBinOffset(cuts.Ptrs());
    common::DispatchBinType(result->index.GetBinTypeSize(), [&](auto type) {
      auto data = result->index.data<decltype(type)>();
      for (std::size_t i = 0; i < bins.size(); ++i) {
        data[i] = Dense() ? bins[i] - cuts.Ptrs()[i % 7] : bins[i];
        ++result->hit_count[bins[i]];
      }
    });
    result->InitColumns(&ctx, .2);
    return result;
  }
  std::vector<std::uint32_t> Symbols(std::size_t start, std::size_t rows, unsigned stride,
                                     unsigned n_symbols) const {
    std::vector<std::uint32_t> result;
    for (std::size_t r = 0; r < rows; ++r) {
      auto begin = result.size();
      for (unsigned f = 0; f < 7; ++f) {
        if (stride == 7)
          result.push_back(Valid(start + r, f) ? Bin(start + r, f) : n_symbols - 1);
        else if (Valid(start + r, f))
          result.push_back(cuts.Ptrs()[f] + Bin(start + r, f));
      }
      result.resize(begin + stride, n_symbols - 1);
    }
    return result;
  }
  void Verify(GHistIndexMatrix const& actual, GHistIndexMatrix const& reference) {
    Check(actual.Size() == reference.Size() && actual.base_rowid == reference.base_rowid,
          "Restore geometry");
    Check(actual.index.Size() == reference.index.Size(), "Restored nnz");
    Check(bool(actual.metal_page), "Restored GPU buffers retained");
    for (std::size_t i = 0; i < actual.row_ptr.size(); ++i)
      Check(actual.row_ptr[i] == reference.row_ptr[i], "Row pointers");
    for (std::size_t i = 0; i < actual.index.Size(); ++i)
      Check(actual.index[i] == reference.index[i], "Restored global bin");
    for (std::size_t i = 0; i < actual.hit_count.size(); ++i)
      Check(actual.hit_count[i] == reference.hit_count[i], "GPU hit counts");
    auto& columns = actual.Transpose();
    common::DispatchBinType(columns.GetTypeSize(), [&](auto t) {
      for (unsigned f = 0; f < 7; ++f) {
        auto verify = [&](auto column) {
          for (std::size_t r = 0; r < actual.Size(); ++r)
            Check(column[r] == reference.GetGindex(reference.base_rowid + r, f),
                  "Column/missing restore");
        };
        if (columns.GetColumnType(f) == common::kSparseColumn)
          verify(columns.SparseColumn<decltype(t)>(f, 0));
        else if (columns.AnyMissing())
          verify(columns.DenseColumn<decltype(t), true>(f));
        else
          verify(columns.DenseColumn<decltype(t), false>(f));
      }
    });
  }
};
void Case(unsigned bins, unsigned mode, double ratio, bool empty,
          float snappy = -1.0f, std::size_t extra_bytes = 0) {
  Fixture fixture{bins, mode};
  std::vector<bst_idx_t> chunks =
      empty ? std::vector<bst_idx_t>{0, 0} : std::vector<bst_idx_t>{3, 0, 17, 13};
  auto rows = std::accumulate(chunks.begin(), chunks.end(), bst_idx_t{0});
  auto reference = fixture.Index(0, rows);
  unsigned stride = 0;
  for (std::size_t r = 0; r < rows; ++r)
    stride = std::max(stride, unsigned(reference->row_ptr[r + 1] - reference->row_ptr[r]));
  auto info = metal::MakeEllpackInfo(fixture.cuts, stride, fixture.Dense());
  auto expected_stride = fixture.Dense() ? 7u : stride;
  auto symbols = fixture.Dense() ? bins : stride == 7 ? bins + 1 : fixture.cuts.TotalBins() + 1;
  Check(info.row_stride == expected_stride && info.symbols == symbols, "CUDA symbol geometry");
  std::size_t allocation = 0;
  for (auto n : chunks)
    allocation += common::CompressedBufferWriter::CalculateBufferSize(n * expected_stride, symbols);
  allocation += extra_bytes;
  auto target = metal::MakeCompressedPage(info, rows, 73, allocation);
  bst_idx_t start = 0;
  for (auto n : chunks) {
    auto raw = fixture.Index(start, n);
    auto source = metal::MakeCompressedPage(info, n, 73 + start);
    source->Pack(*raw);
    auto values = fixture.Symbols(start, n, expected_stride, symbols);
    std::vector<std::uint8_t> expected(source->SizeBytes(), 0);
    common::CompressedBufferWriter writer{symbols};
    for (std::size_t i = 0; i < values.size(); ++i)
      writer.WriteSymbol(expected.data(), values[i], i);
    auto actual = source->Readback();
    if (actual != expected) {
      std::fprintf(stderr,
                   "pack bins=%u mode=%u rows=%llu start=%llu stride=%u symbols=%u bits=%u\n", bins,
                   mode, n, start, expected_stride, symbols, info.bits);
      for (std::size_t j = 0; j < expected.size(); ++j)
        std::fprintf(stderr, "%zu:%02x/%02x ", j, actual[j], expected[j]);
      std::fprintf(stderr, "\n");
    }
    Check(actual == expected, "Packed CUDA bytes");
    bytes_checked += expected.size();
    source->Commit(ratio, snappy, snappy >= 0);
    Check(source->Readback() == expected, "Committed source bytes");
    target->Copy(*source, start);
    start += n;
  }
  auto values = fixture.Symbols(0, rows, expected_stride, symbols);
  std::vector<std::uint8_t> expected(allocation, 0);
  common::CompressedBufferWriter writer{symbols};
  for (std::size_t i = 0; i < values.size(); ++i) writer.WriteSymbol(expected.data(), values[i], i);
  auto start_commit = metal::GetStatistics();
  target->Commit(ratio, snappy, snappy >= 0);
  auto end_commit = metal::GetStatistics();
  Check(target->NonMissing() == reference->index.Size(),
        "Committed GPU non-missing count");
  Check(target->Readback() == expected, "Concatenated CUDA bytes and zero padding");
  bytes_checked += expected.size();
  auto host_bytes = ratio == 0   ? 0
                    : ratio == 1 ? allocation
                                 : std::max(std::size_t(allocation * ratio), std::size_t{1});
  Check(target->HostBytes() == host_bytes && target->DeviceBytes() == allocation - host_bytes,
        "Exact host/device split");
  auto compressed = snappy >= 0 && info.row_stride != info.features
                        ? std::min(host_bytes, std::size_t(host_bytes * snappy))
                        : 0;
  Check(end_commit.snappy_input_bytes - start_commit.snappy_input_bytes ==
            compressed,
        "Only selected sparse host bytes use Snappy");
  Check(target->HostStorageBytes() == host_bytes - compressed +
                                          end_commit.snappy_compressed_bytes -
                                          start_commit.snappy_compressed_bytes,
        "Logical and physical cache bytes are distinct");
  auto before = metal::GetStatistics();
  for (bool copy : {false, true}) {
    GHistIndexMatrix restored;
    target->Restore(&fixture.ctx, fixture.cuts, bins, .2, copy, &restored);
    fixture.Verify(restored, *reference);
  }
  auto after = metal::GetStatistics();
  Check(after.ellpack_debug_readback_bytes == before.ellpack_debug_readback_bytes,
        "Restore never reads compressed bytes on host");
  Check(after.index_uploads == before.index_uploads, "Restore retains GPU index buffers");
  ++cases;
}
// Exercise every bit width without allocating billions of histogram cuts. This
// synthetic index is used only as a symbol source for Pack/Copy, never for bin
// search, histogram accumulation or restoration into a real matrix.
void SymbolWidths() {
  Context ctx;
  for (unsigned bits = 1; bits <= 32; ++bits) {
    std::uint32_t symbols = std::uint32_t((std::uint64_t{1} << (bits - 1)) + 1);
    common::HistogramCuts cuts{0};
    cuts.cut_ptrs_.HostVector() = {0, 1};
    cuts.cut_values_.HostVector() = {0};
    GHistIndexMatrix input{37, 0, std::move(cuts), 1 << 30, true};
    input.ResizeIndex(&ctx, 37, true);
    input.index.SetBinOffset(input.cut.Ptrs());
    std::vector<std::uint32_t> values(37);
    for (unsigned i = 0; i < 37; ++i) {
      values[i] =
          i == 0 ? symbols - 1 : std::uint32_t((std::uint64_t(i) * 2654435761ULL) % symbols);
      input.index.data<std::uint32_t>()[i] = values[i];
      input.row_ptr[i + 1] = i + 1;
    }
    metal::EllpackInfo info{1, 1, symbols, bits, true};
    auto source = metal::MakeCompressedPage(info, 37, 0);
    source->Pack(input);
    source->Commit(.23);
    auto target = metal::MakeCompressedPage(info, 74, 0);
    target->Copy(*source, 0);
    target->Copy(*source, 37);
    target->Commit(.0001);  // Split inside the header; all symbols reside in the tail.
    std::vector<std::uint8_t> expected(target->SizeBytes(), 0);
    common::CompressedBufferWriter writer{symbols};
    for (unsigned i = 0; i < 74; ++i) writer.WriteSymbol(expected.data(), values[i % 37], i);
    Check(target->Readback() == expected,
          "Every symbol width, cross-byte copy and split addressing");
    bytes_checked += expected.size();
    ++cases;
  }
}
void EmptySymbols() {
  Context ctx;
  ctx.Init({{"nthread", "1"}});
  common::HistogramCuts cuts{0};
  cuts.cut_ptrs_.HostVector().assign(4, 0);
  auto info = metal::MakeEllpackInfo(cuts, 0, false);
  Check(info.features == 3 && info.row_stride == 0 && info.symbols == 1 && info.bits == 1,
        "CUDA empty symbol alphabet");
  GHistIndexMatrix raw{17, 73, common::HistogramCuts{cuts}, 8, false};
  raw.ResizeIndex(&ctx, 0, false);
  auto packed = metal::MakeCompressedPage(info, 17, 73);
  packed->Pack(raw);
  packed->Commit(.23);
  GHistIndexMatrix output;
  auto before = metal::GetStatistics();
  packed->Restore(&ctx, cuts, 8, .2, true, &output);
  Check(output.Size() == 17 && output.Features() == 3 && output.index.Size() == 0,
        "Restore empty alphabet rows");
  for (auto offset : output.row_ptr) Check(offset == 0, "Empty alphabet row pointers");
  Check(metal::GetStatistics().ellpack_debug_readback_bytes == before.ellpack_debug_readback_bytes,
        "No empty alphabet host readback");
  ++cases;
}
void Plans() {
  metal::EllpackInfo info{7, 7, 17, 5, true};
  std::vector<bst_idx_t> rows{0, 3, 17, 13, 0};
  for (auto memory : {1024ULL, 1048576ULL})
    for (auto minimum : {-1LL, 0LL, 1LL, 33LL, 4096LL})
      for (double ratio : {std::nan(""), 0., .25, 1.})
        for (bool validation : {false, true}) {
          auto actual = metal::PlanEllpackCache(info, rows, validation, ratio, minimum, memory);
          std::vector<std::size_t> bytes, mapping;
          std::vector<bst_idx_t> sizes;
          std::size_t total = 0;
          auto limit = minimum == -1 ? memory / 16 : minimum;
          for (auto n : rows) {
            auto b = common::CompressedBufferWriter::CalculateBufferSize(n * 7, 17);
            total += b;
            if (bytes.empty() || bytes.back() >= limit) {
              bytes.push_back(b);
              sizes.push_back(n);
            } else {
              bytes.back() += b;
              sizes.back() += n;
            }
            mapping.push_back(bytes.size() - 1);
          }
          auto host = ratio;
          if (std::isnan(host))
            host = validation            ? 1.
                   : total <= memory / 2 ? 0.
                                         : (total - (memory / 2) * .85) / total;
          if (bytes.size() == 1) host = 0.;
          Check(actual.host_ratio == host && actual.min_page_bytes == limit, "CUDA cache defaults");
          Check(actual.bytes == bytes && actual.rows == sizes && actual.mapping == mapping,
                "CUDA page concatenation plan");
        }
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    Plans();
    SymbolWidths();
    EmptySymbols();
    for (auto bins : {1u, 2u, 3u, 8u, 17u, 255u, 256u, 257u, 65535u, 65536u, 65537u}) {
      for (unsigned mode = 0; mode < 4; ++mode)
        for (double ratio : {0., 1e-6, .23, .5, 1.}) Case(bins, mode, ratio, false);
      std::printf("checked %zu cases / %zu bytes / %zu fields\n", cases, bytes_checked, fields);
    }
    for (unsigned mode = 0; mode < 4; ++mode) Case(17, mode, .23, true);
    for (unsigned mode = 0; mode < 4; ++mode)
      for (double ratio : {0., 1e-6, .23, 1.})
        for (float compression : {0.f, .4f, 1.f})
          for (bool empty : {false, true})
            Case(257, mode, ratio, empty, compression);
    // Cache reads must reconstruct multiple 2MiB chunks at a nonzero byte
    // offset. Concatenation padding is part of the original logical allocation.
    Case(257, 2, .75, false, 1.f, 5u << 20);
    Case(257, 2, 1., false, .4f, 6u << 20);
    std::printf("PASS %zu cases / %zu exact bytes / %zu exact fields\n", cases, bytes_checked,
                fields);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL case %zu: %s\n", cases, e.what());
    return 1;
  }
}
