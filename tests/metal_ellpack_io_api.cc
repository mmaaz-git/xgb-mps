// Independent CUDA aligned-stream/CPU bit-writer oracle; file and view
// lifetimes.
#include "../xgboost/src/common/compressed_iterator.h"
#include "../xgboost/src/common/io.h"
#include "../xgboost/src/common/metal_ellpack.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/data/gradient_index.h"
#include "../xgboost/src/data/iterative_dmatrix.h"
#include "../xgboost/src/data/proxy_dmatrix.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>
using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, exact_bytes = 0, rejected = 0, saves = 0;
void Check(bool ok, char const *why) {
  ++fields;
  if (!ok)
    throw std::runtime_error(why);
}
template <class T> void Append(std::string *out, T v) {
  out->append(reinterpret_cast<char const *>(&v), sizeof(v));
  out->resize((out->size() + 7) / 8 * 8, '\0');
}
auto Reader(std::string const &bytes) {
  auto memory = std::make_shared<common::MallocResource>(bytes.size());
  if (!bytes.empty())
    std::memcpy(memory->Data(), bytes.data(), bytes.size());
  return std::make_unique<common::AlignedResourceReadStream>(memory);
}
std::string Write(GHistIndexMatrix const &page) {
  std::string out;
  {
    common::AlignedMemWriteStream stream{&out};
    auto n = metal::WriteCompressedPage(page, &stream);
    Check(n == out.size(), "Aligned write byte count");
  }
  return out;
}
void Case(std::filesystem::path const &file, unsigned bins, unsigned mode,
          double ratio, bst_idx_t rows, bool padded) {
  constexpr unsigned cols = 5;
  Context cpu;
  cpu.Init({{"nthread", "1"}});
  common::HistogramCuts cuts{0};
  auto &ptr = cuts.cut_ptrs_.HostVector();
  ptr.assign(1, 0);
  for (unsigned f = 0; f < cols; ++f) {
    auto width = mode == 3 ? 0u : std::max(1u, bins - std::min(bins - 1, f));
    for (unsigned j = 0; j < width; ++j)
      cuts.cut_values_.HostVector().push_back(float(j));
    ptr.push_back(cuts.Values().size());
  }
  auto valid = [&](bst_idx_t r, unsigned f) {
    return mode == 0 || (mode == 1 && (r == 0 || (r + f) % 3)) ||
           (mode == 2 && (r + f) % 5 == 0);
  };
  bool dense = mode == 0;
  GHistIndexMatrix oracle{rows, 73, common::HistogramCuts{cuts},
                          static_cast<bst_bin_t>(bins), dense};
  std::vector<std::uint32_t> global;
  unsigned stride = 0;
  for (bst_idx_t r = 0; r < rows; ++r) {
    for (unsigned f = 0; f < cols; ++f)
      if (valid(r, f)) {
        global.push_back(ptr[f] + (r * 17 + f * 31) % (ptr[f + 1] - ptr[f]));
        ++oracle.hit_count[global.back()];
      }
    stride = std::max(stride, unsigned(global.size() - oracle.row_ptr[r]));
    oracle.row_ptr[r + 1] = global.size();
  }
  oracle.ResizeIndex(&cpu, global.size(), dense);
  if (dense)
    oracle.index.SetBinOffset(cuts.Ptrs());
  common::DispatchBinType(oracle.index.GetBinTypeSize(), [&](auto t) {
    auto data = oracle.index.data<decltype(t)>();
    for (std::size_t k = 0; k < global.size(); ++k)
      data[k] = dense ? global[k] - ptr[k % cols] : global[k];
  });
  auto layout = metal::MakeEllpackInfo(cuts, stride, dense);
  auto expected_stride = dense ? cols : stride;
  auto symbols = mode == 3        ? 1u
                 : dense          ? bins
                 : stride == cols ? bins + 1
                                  : ptr.back() + 1;
  auto allocation = common::CompressedBufferWriter::CalculateBufferSize(
      rows * expected_stride, symbols);
  if (padded)
    allocation +=
        24; // Concatenated host caches may preserve extra per-page padding.
  std::vector<std::uint8_t> payload(allocation, 0);
  common::CompressedBufferWriter bitwriter{symbols};
  std::size_t k = 0;
  for (bst_idx_t r = 0; r < rows; ++r) {
    unsigned slot = 0;
    for (unsigned f = 0; f < cols; ++f) {
      if (expected_stride == cols) {
        auto symbol = valid(r, f) ? global[k++] - ptr[f] : symbols - 1;
        bitwriter.WriteSymbol(payload.data(), symbol,
                              r * expected_stride + slot++);
      } else if (valid(r, f))
        bitwriter.WriteSymbol(payload.data(), global[k++],
                              r * expected_stride + slot++);
    }
    while (slot < expected_stride)
      bitwriter.WriteSymbol(payload.data(), symbols - 1,
                            r * expected_stride + slot++);
  }
  std::string expected;
  Append(&expected, rows);
  Append(&expected, dense);
  Append(&expected, bst_idx_t(expected_stride));
  Append(&expected, std::uint64_t(payload.size()));
  expected.append(reinterpret_cast<char const *>(payload.data()),
                  payload.size());
  expected.resize((expected.size() + 7) / 8 * 8, '\0');
  Append(&expected, bst_idx_t(73));
  Append(&expected, bst_idx_t(symbols));
  auto source = metal::MakeCompressedPage(layout, rows, 73, allocation);
  source->Pack(oracle);
  source->Commit(ratio);
  GHistIndexMatrix view;
  source->View(cuts, bins, false, &view);
  auto before = metal::GetStatistics();
  auto actual = Write(view);
  Check(actual == expected,
        "CUDA field order, aligned padding and exact compressed payload");
  exact_bytes += actual.size();
  {
    common::AlignedFileWriteStream output{file.string(), "wb"};
    Check(metal::WriteCompressedPage(view, &output) == expected.size(),
          "Disk byte count");
  }
  Check(std::filesystem::file_size(file) == expected.size(),
        "Actual file length");
  auto written = metal::GetStatistics();
  Check(written.ellpack_io_write_pages - before.ellpack_io_write_pages == 2 &&
            written.ellpack_io_write_bytes - before.ellpack_io_write_bytes ==
                2 * allocation,
        "Completed page/payload write accounting");
  for (bool copy : {false, true}) {
    auto start = metal::GetStatistics();
    GHistIndexMatrix loaded;
    {
      common::PrivateMmapConstStream reader{file.string(), 0, expected.size()};
      Check(metal::ReadCompressedPage(&reader, cuts, bins, copy, &loaded),
            "Mapped read");
      Check(reader.Tell() == expected.size(), "Read consumes exact page");
    }
    auto read = metal::GetStatistics();
    Check(read.ellpack_io_read_pages - start.ellpack_io_read_pages == 1 &&
              read.ellpack_io_read_bytes - start.ellpack_io_read_bytes ==
                  allocation &&
              read.ellpack_io_dispatches - start.ellpack_io_dispatches ==
                  unsigned(copy),
          "Completed page/payload read and private-copy accounting");
    Check(!loaded.HostReadable() && loaded.index.Size() == 0 &&
              loaded.row_ptr.empty(),
          "Device-only IO");
    Check(Write(loaded) == expected, "Owned GPU bytes survive mapping close");
    Check(metal::CompressedNonMissing(loaded) == global.size(),
          "Loaded compressed GPU nnz");
    auto ready = metal::GetStatistics();
    Check(ready.ellpack_host_materializations ==
              before.ellpack_host_materializations,
          "No CPU bins during IO");
    Check(ready.ellpack_debug_readback_bytes ==
                  start.ellpack_debug_readback_bytes &&
              ready.index_uploads == start.index_uploads,
          "Production IO never uses diagnostic reads or typed-index uploads");
    metal::MaterializeHost(&cpu, .2, &loaded);
    for (std::size_t i = 0; i <= rows; ++i)
      Check(loaded.row_ptr[i] == oracle.row_ptr[i], "CPU row pointers");
    for (std::size_t i = 0; i < global.size(); ++i)
      Check(loaded.index[i] == global[i], "CPU exact bins");
    for (std::size_t i = 0; i < oracle.hit_count.size(); ++i)
      Check(loaded.hit_count[i] == oracle.hit_count[i], "CPU hit counts");
    // Explicit CPU access above is the only permitted materialization.
    before = metal::GetStatistics();
  }
  if (bins == 2 && mode == 1 && rows && ratio == 0 && !padded) {
    // Every missing field/payload prefix must fail before publishing a GPU
    // view.
    for (std::size_t n = 0; n < expected.size(); ++n) {
      auto reader = Reader(expected.substr(0, n));
      GHistIndexMatrix bad;
      Check(!metal::ReadCompressedPage(reader.get(), cuts, bins, false, &bad),
            "Reject truncated page");
      Check(!bad.metal_page && bad.row_ptr.empty(), "No partial publication");
      ++rejected;
    }
    for (auto mutation : std::vector<std::pair<std::size_t, std::uint64_t>>{
             {8, 2},
             {0, 1ULL << 32},
             {16, cols + 1},
             {24, ~0ULL},
             {expected.size() - 8, symbols + 1},
             {expected.size() - 16, ~0ULL},
             {0, 1ULL << 31}}) {
      auto bytes = expected;
      std::memcpy(bytes.data() + mutation.first, &mutation.second, 8);
      auto reader = Reader(bytes);
      GHistIndexMatrix bad;
      bool threw = false;
      try {
        metal::ReadCompressedPage(reader.get(), cuts, bins, true, &bad);
      } catch (dmlc::Error const &) {
        threw = true;
      }
      Check(threw && !bad.metal_page && bad.row_ptr.empty(),
            "Reject invalid geometry before publication");
      ++rejected;
    }
    // A valid owned read also survives deletion of its backing file.
    GHistIndexMatrix retained;
    {
      common::PrivateMmapConstStream reader{file.string(), 0, expected.size()};
      Check(metal::ReadCompressedPage(&reader, cuts, bins, false, &retained),
            "Read before unlink");
    }
    std::filesystem::remove(file);
    Check(Write(retained) == expected, "Read survives deletion");
  }
  ++cases;
}
struct Input {
  data::DMatrixProxy *proxy;
  std::vector<float> values;
  bool done = false;
  static void Reset(DataIterHandle p) { static_cast<Input *>(p)->done = false; }
  static int Next(DataIterHandle p) {
    auto &self = *static_cast<Input *>(p);
    if (self.done)
      return 0;
    self.done = true;
    std::ostringstream json;
    json << "{\"version\":3,\"typestr\":\"<f4\",\"data\":["
         << reinterpret_cast<std::uintptr_t>(self.values.data())
         << ",false],\"shape\":[37,5],\"strides\":null}";
    self.proxy->Info().Clear();
    self.proxy->SetArray(StringView{json.str()});
    self.proxy->Info().feature_types.HostVector() = {
        FeatureType::kNumerical, FeatureType::kCategorical,
        FeatureType::kNumerical, FeatureType::kNumerical,
        FeatureType::kNumerical};
    return 1;
  }
};
void SaveLoad() {
  Context gpu, cpu;
  gpu.Init({{"device", "mps"}, {"nthread", "1"}});
  std::shared_ptr<DMatrix> proxy = std::make_shared<data::DMatrixProxy>();
  Input input{static_cast<data::DMatrixProxy *>(proxy.get())};
  input.values.resize(37 * 5);
  for (unsigned i = 0; i < input.values.size(); ++i)
    input.values[i] = i % 5 == 1 ? float(i % 7) : float(i % 23) / 4;
  for (unsigned mode = 0; mode < 3; ++mode) {
    if (mode)
      for (unsigned i = 0; i < input.values.size(); ++i)
        if (mode == 2 || i % 11 == 0)
          input.values[i] = std::nanf("");
    std::shared_ptr<DMatrix> matrix{
        DMatrix::Create(static_cast<DataIterHandle>(&input),
                        static_cast<DMatrixHandle>(&proxy), {}, Input::Reset,
                        Input::Next, std::nanf(""), 1, 17, std::string{"mps"})};
    auto original =
        matrix->GetBatches<GHistIndexMatrix>(&gpu, {}).begin().Page();
    auto expected_page = Write(*original);
    std::string saved;
    {
      common::AlignedMemWriteStream out{&saved};
      dynamic_cast<data::IterativeDMatrix *>(matrix.get())->Save(&out);
    }
    std::unique_ptr<data::IterativeDMatrix> restored;
    {
      auto reader = Reader(saved);
      restored.reset(data::IterativeDMatrix::Load(&gpu, reader.get()));
      Check(reader->Tell() == saved.size(), "QDM consumes cuts and page");
    }
    matrix.reset();
    auto page = restored->GetBatches<GHistIndexMatrix>(&gpu, {}).begin().Page();
    Check(Write(*page) == expected_page, "Internal QDM exact page roundtrip");
    Check(page->cut.Ptrs() == original->cut.Ptrs() &&
              page->cut.Values() == original->cut.Values() &&
              page->cut.HasCategorical() == original->cut.HasCategorical() &&
              page->cut.MaxCategory() == original->cut.MaxCategory(),
          "Saved cut fields");
    Check(restored->Info().num_row_ == 0 && restored->Info().labels.Size() == 0,
          "CUDA page-only contract");
    bool threw = false;
    try {
      restored->GetBatches<GHistIndexMatrix>(&cpu, {});
    } catch (dmlc::Error const &) {
      threw = true;
    }
    Check(threw,
          "Loaded page-only QDM cannot pretend to have CPU training metadata");
    std::string again;
    {
      common::AlignedMemWriteStream out{&again};
      restored->Save(&out);
    }
    Check(again == saved, "Internal QDM exact file roundtrip");
    ++saves;
  }
}
int main() {
  auto file = std::filesystem::temp_directory_path() /
              ("xgb-ellpack-io-" + std::to_string(getpid()));
  try {
    for (unsigned bins : {1, 2, 17, 257, 65537})
      for (unsigned mode = 0; mode < 4; ++mode)
        for (double ratio : {0., .23, 1.})
          for (bool padded : {false, true})
            Case(file, bins, mode, ratio, 33, padded);
    for (unsigned mode = 0; mode < 4; ++mode)
      Case(file, 17, mode, .23, 0, true);
    SaveLoad();
    std::filesystem::remove(file);
    std::printf("PASS %zu IO cases / %zu exact bytes / %zu fields / %zu "
                "invalid streams / %zu QDM saves\n",
                cases, exact_bytes, fields, rejected, saves);
  } catch (std::exception const &e) {
    std::fprintf(stderr, "FAIL case %zu: %s\n", cases, e.what());
    std::filesystem::remove(file);
    return 1;
  }
}
