// Inspect production cached pages independently of unsupported Python get_data().
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/column_matrix.h"
#include "../xgboost/src/common/compressed_iterator.h"
#include "../xgboost/src/common/metal_ellpack.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/data/gradient_index.h"
#include "../xgboost/src/data/proxy_dmatrix.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, pages = 0, fields = 0;
void Check(bool ok, char const* message) {
  ++fields;
  if (!ok) throw std::runtime_error(message);
}
struct Input {
  data::DMatrixProxy* proxy;
  std::vector<bst_idx_t> sizes{0, 131, 0, 257, 641, 0};
  std::vector<float> values;
  std::vector<FeatureType> types;
  std::size_t page = 0, base = 0, calls = 0;
  bool weighted, closed = false;
  static void Reset(DataIterHandle handle) {
    auto& self = *static_cast<Input*>(handle);
    Check(!self.closed, "Cache reset called the released input");
    self.page = self.base = 0;
  }
  static int Next(DataIterHandle handle) {
    auto& self = *static_cast<Input*>(handle);
    Check(!self.closed, "Cache read called the released input");
    ++self.calls;
    if (self.page == self.sizes.size()) return 0;
    auto rows = self.sizes[self.page++];
    auto cols = self.types.size();
    std::ostringstream json;
    json << "{\"version\":3,\"typestr\":\"<f4\",\"data\":["
         << reinterpret_cast<std::uintptr_t>(self.values.data() + self.base * cols)
         << ",false],\"shape\":[" << rows << "," << cols << "],\"strides\":null}";
    self.proxy->Info().Clear();
    self.proxy->SetArray(StringView{json.str()});
    self.proxy->Info().feature_types.HostVector() = self.types;
    if (self.weighted) {
      auto& weights = self.proxy->Info().weights_.HostVector();
      weights.resize(rows);
      for (std::size_t r = 0; r < rows; ++r) weights[r] = float((self.base + r) % 9) / 4;
    }
    self.base += rows;
    return 1;
  }
};
void Case(std::filesystem::path const& dir, int bins, unsigned layout, bool cats, bool weighted,
          bool on_host = false, double ratio = std::nan(""), std::int64_t minimum = 0) {
  constexpr std::size_t cols = 7, rows = 1029;
  std::shared_ptr<DMatrix> proxy = std::make_shared<data::DMatrixProxy>();
  Input input{static_cast<data::DMatrixProxy*>(proxy.get())};
  input.weighted = weighted;
  input.types.assign(cols, FeatureType::kNumerical);
  if (cats) input.types[2] = FeatureType::kCategorical;
  input.values.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t f = 0; f < cols; ++f) {
      float value = cats && f == 2 ? float(r % 7) : float(int((r * 193 + f * 17) % 997) - 498) / 8;
      if (layout == 3 || (layout == 1 && (r + f) % 5 == 0) ||
          (layout == 2 && (f == 6 || (r + f) % 17 != 0)))
        value = -99;
      input.values[r * cols + f] = value;
    }
  }
  ExtMemConfig config{(dir / "pages").string(), on_host, float(ratio), minimum, -99, 2};
  config.device = "mps";
  auto handle = static_cast<DataIterHandle>(&input);
  auto proxy_handle = static_cast<DMatrixHandle>(&proxy);
  auto before = metal::GetStatistics();
  std::shared_ptr<DMatrix> matrix{DMatrix::Create(handle, proxy_handle, std::shared_ptr<DMatrix>{},
                                                  Input::Reset, Input::Next, bins, config)};
  auto after = metal::GetStatistics();
  Check(after.adapter_prepare_dispatches > before.adapter_prepare_dispatches, "GPU construction");
  Check(after.adapter_debug_readback_bytes == before.adapter_debug_readback_bytes,
        "No raw readback");
  // CPU quantization with the exact same reference cuts is an independent bin oracle.
  std::shared_ptr<DMatrix> cpu{DMatrix::Create(handle, proxy_handle, matrix, Input::Reset,
                                               Input::Next, -99.0f, 2, bins, std::string{"cpu"})};
  Context ctx;
  ctx.Init({{"device", "cpu"}, {"nthread", "2"}});
  GHistIndexMatrix const* oracle = nullptr;
  for (auto const& page : cpu->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{})) oracle = &page;
  Check(oracle && oracle->Size() == rows, "CPU reference shape");
  auto expected_sizes = input.sizes;
  if (on_host) {
    unsigned stride = 0, maximum = 0;
    for (std::size_t r = 0; r < rows; ++r)
      stride = std::max(stride, unsigned(oracle->row_ptr[r + 1] - oracle->row_ptr[r]));
    for (unsigned f = 0; f < cols; ++f)
      maximum = std::max(maximum, oracle->cut.Ptrs()[f + 1] - oracle->cut.Ptrs()[f]);
    auto symbols = oracle->IsDense() ? maximum
                   : stride == cols  ? maximum + 1
                                     : oracle->cut.TotalBins() + 1;
    if (oracle->IsDense()) stride = cols;
    auto limit = minimum == -1 ? metal::CacheWorkingSetBytes() / 16 : minimum;
    std::vector<std::size_t> sizes;
    expected_sizes.clear();
    for (auto n : input.sizes) {
      auto bytes = common::CompressedBufferWriter::CalculateBufferSize(n * stride, symbols);
      if (sizes.empty() || sizes.back() >= limit) {
        sizes.push_back(bytes);
        expected_sizes.push_back(n);
      } else {
        sizes.back() += bytes;
        expected_sizes.back() += n;
      }
    }
  }
  std::vector<bst_idx_t> expected_ptr{0};
  for (auto n : expected_sizes) expected_ptr.push_back(expected_ptr.back() + n);
  Check(matrix->NumBatches() == expected_sizes.size(), "Cached batch count");
  Check(matrix->BatchPtr() == expected_ptr, "Cached batch row boundaries");
  Check(std::filesystem::is_empty(dir) == on_host, "Storage follows on_host");
  input.closed = true;
  auto calls = input.calls;
  before = metal::GetStatistics();
  for (unsigned pass = 0; pass < 3; ++pass) {
    BatchParam parameter;
    parameter.prefetch_copy = pass % 2;
    // Interrupt a cache traversal to check that resetting discards pending reads safely.
    for (auto const& page : matrix->GetBatches<GHistIndexMatrix>(&ctx, parameter)) {
      Check(page.base_rowid == 0, "Reset begins at first page");
      break;
    }
    std::size_t base = 0, batch = 0;
    std::vector<std::size_t> counts(oracle->hit_count.size(), 0);
    for (auto const& page : matrix->GetBatches<GHistIndexMatrix>(&ctx, parameter)) {
      ++pages;
      Check(page.base_rowid == base && page.Size() == expected_sizes[batch++], "Page geometry");
      Check(page.index.GetBinTypeSize() == oracle->index.GetBinTypeSize(),
            "Compressed index width");
      auto offset = oracle->row_ptr[base];
      for (std::size_t r = 0; r <= page.Size(); ++r) {
        Check(page.row_ptr[r] == oracle->row_ptr[base + r] - offset, "Local row pointer");
      }
      for (std::size_t k = 0; k < page.index.Size(); ++k) {
        Check(page.index[k] == oracle->index[offset + k], "Exact cached bin");
      }
      for (std::size_t k = 0; k < counts.size(); ++k) counts[k] += page.hit_count[k];
      // Verify reconstructed feature IDs/row IDs, including nonzero global bases.
      AssignColumnBinIndex(page, [&](auto bin, auto k, auto r, auto f) {
        Check(page.index[k] == bin, "Recovered index position");
        Check(oracle->GetGindex(base + r, f) == bin, "Recovered feature/row");
      });
      auto& columns = page.Transpose();
      common::DispatchBinType(columns.GetTypeSize(), [&](auto type) {
        using T = decltype(type);
        for (std::size_t f = 0; f < cols; ++f) {
          auto verify = [&](auto column) {
            for (std::size_t r = 0; r < page.Size(); ++r) {
              Check(column[r] == oracle->GetGindex(base + r, f), "Cached column bin/missing flag");
            }
          };
          if (columns.GetColumnType(f) == common::kSparseColumn) {
            verify(columns.SparseColumn<T>(f, 0));
          } else if (columns.AnyMissing()) {
            verify(columns.DenseColumn<T, true>(f));
          } else {
            verify(columns.DenseColumn<T, false>(f));
          }
        }
      });
      base += page.Size();
    }
    Check(batch == expected_sizes.size() && base == rows, "All cached pages visited");
    for (std::size_t k = 0; k < counts.size(); ++k) {
      Check(counts[k] == oracle->hit_count[k], "Cached hit-count sum");
    }
  }
  after = metal::GetStatistics();
  Check(input.calls == calls, "Cache independence");
  Check(after.ellpack_debug_readback_bytes == before.ellpack_debug_readback_bytes,
        "No raw compressed readback");
  if (on_host) Check(after.index_uploads == before.index_uploads, "Restored buffers remain on GPU");
  Check(after.adapter_prepare_dispatches == before.adapter_prepare_dispatches &&
            after.adapter_index_dispatches == before.adapter_index_dispatches &&
            after.cuda_sketch_batches == before.cuda_sketch_batches,
        "No cache reingestion/resketch");
  cpu.reset();
  matrix.reset();
  Check(std::filesystem::is_empty(dir), "Cache removed at matrix destruction");
  ++cases;
  std::printf("case=%zu bins=%d layout=%u categories=%d weighted=%d PASS\n", cases, bins, layout,
              cats, weighted);
  std::fflush(stdout);
}
int main() {
  auto dir =
      std::filesystem::temp_directory_path() / ("xgb-metal-extmem-" + std::to_string(getpid()));
  try {
    Check(std::filesystem::create_directory(dir), "Create isolated cache directory");
    for (int bins : {8, 256, 257})
      for (unsigned layout = 0; layout < 4; ++layout)
        for (bool cats : {false, true})
          for (bool weighted : {false, true}) Case(dir, bins, layout, cats, weighted);
    for (int bins : {8, 257})
      for (unsigned layout = 0; layout < 4; ++layout)
        for (bool cats : {false, true})
          for (double ratio : {0., .23, 1.})
            for (std::int64_t minimum : {0, 128, -1})
              Case(dir, bins, layout, cats, true, true, ratio, minimum);
    std::filesystem::remove(dir);
    std::printf("PASS %zu cases / %zu cached page visits / %zu exact fields\n", cases, pages,
                fields);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL case %zu: %s\n", cases, e.what());
    std::filesystem::remove_all(dir);
    return 1;
  }
}
