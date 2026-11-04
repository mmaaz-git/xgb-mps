// Independent CPU bins and lifetime/publication checks for in-memory Metal
// ELLPACK.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/column_matrix.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_input.h"
#include "../xgboost/src/data/gradient_index.h"
#include "../xgboost/src/data/proxy_dmatrix.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, concurrent_cases = 0;
void Check(bool ok, char const *message) {
  ++fields;
  if (!ok)
    throw std::runtime_error(message);
}
struct Input {
  data::DMatrixProxy *proxy;
  std::vector<bst_idx_t> sizes;
  std::vector<float> values;
  std::vector<FeatureType> types;
  std::size_t page = 0, base = 0;
  bool weighted = false, closed = false;
  static void Reset(DataIterHandle handle) {
    auto &self = *static_cast<Input *>(handle);
    if (self.closed)
      throw std::runtime_error("Matrix conversion reset released input");
    self.page = self.base = 0;
  }
  static int Next(DataIterHandle handle) {
    auto &self = *static_cast<Input *>(handle);
    if (self.closed)
      throw std::runtime_error("Matrix conversion reread released input");
    if (self.page == self.sizes.size())
      return 0;
    auto rows = self.sizes[self.page++];
    auto cols = self.types.size();
    std::ostringstream json;
    json << "{\"version\":3,\"typestr\":\"<f4\",\"data\":["
         << reinterpret_cast<std::uintptr_t>(self.values.data() +
                                             self.base * cols)
         << ",false],\"shape\":[" << rows << "," << cols
         << "],\"strides\":null}";
    self.proxy->Info().Clear();
    self.proxy->SetArray(StringView{json.str()});
    self.proxy->Info().feature_types.HostVector() = self.types;
    if (self.weighted) {
      auto &weights = self.proxy->Info().weights_.HostVector();
      weights.resize(rows);
      for (std::size_t r = 0; r < rows; ++r)
        weights[r] = float((self.base + r) % 9) / 4;
    }
    self.base += rows;
    return 1;
  }
};
std::shared_ptr<GHistIndexMatrix const> Page(DMatrix *matrix,
                                             Context const *ctx) {
  auto batches = matrix->GetBatches<GHistIndexMatrix>(ctx, BatchParam{});
  auto it = batches.begin();
  auto result = it.Page();
  ++it;
  Check(!(it != batches.end()), "In-memory matrix has one page");
  return result;
}
void Case(int bins, unsigned layout, bool cats, bool weighted, bool stream,
          bool cpu_source, std::size_t rows = 1031) {
  constexpr std::size_t cols = 7;
  std::shared_ptr<DMatrix> proxy = std::make_shared<data::DMatrixProxy>();
  Input input{static_cast<data::DMatrixProxy *>(proxy.get())};
  input.sizes = stream
                    ? std::vector<bst_idx_t>{0, rows / 3, 0, rows - rows / 3, 0}
                    : std::vector<bst_idx_t>{rows};
  input.weighted = weighted;
  input.types.assign(cols, FeatureType::kNumerical);
  if (cats)
    input.types[2] = FeatureType::kCategorical;
  input.values.resize(std::max<std::size_t>(1, rows * cols));
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t f = 0; f < cols; ++f) {
      float value = cats && f == 2
                        ? float(r % 7)
                        : float(int((r * 193 + f * 17) % 997) - 498) / 8;
      if (layout == 3 || (layout == 1 && r && (r + f) % 5 == 0) ||
          (layout == 2 && (f == 6 || (r + f) % 17 != 0)))
        value = -99;
      input.values[r * cols + f] = value;
    }
  }
  auto handle = static_cast<DataIterHandle>(&input);
  auto proxy_handle = static_cast<DMatrixHandle>(&proxy);
  auto before = metal::GetStatistics();
  std::shared_ptr<DMatrix> matrix{DMatrix::Create(
      handle, proxy_handle, {}, Input::Reset, Input::Next, -99.f, 2, bins,
      cpu_source ? std::string{"cpu"} : std::string{"mps"})};
  auto constructed = metal::GetStatistics();
  Check(constructed.ellpack_host_materializations ==
                before.ellpack_host_materializations &&
            constructed.ellpack_host_readback_bytes ==
                before.ellpack_host_readback_bytes &&
            constructed.ellpack_pack_input_bytes ==
                before.ellpack_pack_input_bytes,
        "Construction stages no host bins or mirrors");
  if (!cpu_source) {
    Check(constructed.ellpack_adapter_values - before.ellpack_adapter_values ==
              matrix->Info().num_nonzero_,
          "Direct GPU compressed binning covers every input value");
    Check(constructed.adapter_index_dispatches ==
                  before.adapter_index_dispatches &&
              constructed.ellpack_view_pages == before.ellpack_view_pages + 1,
          "Direct construction retains one compressed view, no typed index "
          "writes");
  }
  // A CPU matrix built from identical input and the exact same cuts is the
  // oracle.
  std::shared_ptr<DMatrix> reference{
      DMatrix::Create(handle, proxy_handle, matrix, Input::Reset, Input::Next,
                      -99.f, 2, bins, std::string{"cpu"})};
  auto after_reference = metal::GetStatistics();
  Check(after_reference.ellpack_host_materializations ==
                constructed.ellpack_host_materializations &&
            after_reference.ellpack_pack_input_bytes ==
                constructed.ellpack_pack_input_bytes,
        "CPU reference metadata does not convert the source matrix");
  input.closed = true;
  Context cpu, gpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  gpu.Init({{"device", "mps"}, {"nthread", "2"}});
  auto oracle = Page(reference.get(), &cpu);
  auto get = [&](bool device) {
    // No fixture counters here: this function may execute concurrently.
    return matrix
        ->GetBatches<GHistIndexMatrix>(device ? &gpu : &cpu, BatchParam{})
        .begin()
        .Page();
  };
  std::vector<std::shared_ptr<GHistIndexMatrix const>> readers(6);
  if (bins == 17 && stream && !cats && !weighted) {
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<std::shared_ptr<GHistIndexMatrix const>>> futures;
    for (unsigned i = 0; i < readers.size(); ++i)
      futures.push_back(std::async(std::launch::async, [&, i] {
        gate.wait();
        return get(i % 2);
      }));
    start.set_value();
    for (std::size_t i = 0; i < readers.size(); ++i)
      readers[i] = futures[i].get();
    ++concurrent_cases;
  } else {
    // GPU first ensures CPU construction does not remain on the old typed path.
    readers[1] = get(true);
    for (unsigned i = 0; i < readers.size(); ++i)
      readers[i] = get(i % 2);
  }
  auto host = readers[0], device = readers[1];
  for (unsigned i = 0; i < readers.size(); ++i)
    Check(readers[i] == (i % 2 ? device : host),
          "One published representation per device");
  Check(host != device && host->HostReadable() && !device->HostReadable(),
        "Independent CPU/GPU views");
  Check(device->row_ptr.empty() && device->data.empty() &&
            device->hit_count.empty() && device->index.Size() == 0 &&
            device->Size() == rows && device->Features() == cols,
        "GPU view keeps no host arrays");
  Check(host->Size() == rows && host->index.Size() == oracle->index.Size() &&
            host->index.GetBinTypeSize() == oracle->index.GetBinTypeSize(),
        "CPU shape and bin width");
  for (std::size_t r = 0; r <= rows; ++r)
    Check(host->row_ptr[r] == oracle->row_ptr[r], "Exact CPU row pointer");
  for (std::size_t k = 0; k < host->index.Size(); ++k)
    Check(host->index[k] == oracle->index[k], "Exact CPU bin");
  for (std::size_t k = 0; k < host->hit_count.size(); ++k)
    Check(host->hit_count[k] == oracle->hit_count[k], "Exact CPU bin count");
  auto const &columns = host->Transpose();
  common::DispatchBinType(columns.GetTypeSize(), [&](auto type) {
    using T = decltype(type);
    for (std::size_t f = 0; f < cols; ++f) {
      auto verify = [&](auto column) {
        for (std::size_t r = 0; r < rows; ++r)
          Check(column[r] == oracle->GetGindex(r, f),
                "Exact CPU column and missing flag");
      };
      if (columns.GetColumnType(f) == common::kSparseColumn)
        verify(columns.SparseColumn<T>(f, 0));
      else if (columns.AnyMissing())
        verify(columns.DenseColumn<T, true>(f));
      else
        verify(columns.DenseColumn<T, false>(f));
    }
  });
  auto converted = metal::GetStatistics();
  Check(converted.ellpack_host_materializations -
                constructed.ellpack_host_materializations ==
            (cpu_source ? 0 : 1),
        "At most one host materialization, including concurrent access");
  auto upload = cpu_source && oracle->index.Size()
                    ? oracle->index.Size() * oracle->index.GetBinTypeSize() +
                          (rows + 1) * 8
                    : 0;
  Check(converted.ellpack_pack_input_bytes -
                constructed.ellpack_pack_input_bytes ==
            upload,
        "CPU-origin compression uploads bins once; direct MPS construction "
        "never does");
  // Explicit export with an MPS context must use CPU data and retain both
  // views.
  for (auto const &batch :
       matrix->GetBatches<ExtSparsePage>(&gpu, BatchParam{})) {
    Check(batch.page->Size() == rows, "Sparse export shape");
    Check(batch.page->offset.ConstHostVector().size() == rows + 1,
          "Sparse export row array");
  }
  Check(Page(matrix.get(), &gpu) == device && Page(matrix.get(), &cpu) == host,
        "Repeated batches and exports retain both representations");
  Check(metal::GetStatistics().ellpack_host_materializations ==
                converted.ellpack_host_materializations &&
            metal::GetStatistics().ellpack_pack_input_bytes ==
                converted.ellpack_pack_input_bytes,
        "No repeated conversion");
  matrix.reset();
  auto n = std::min<std::size_t>(rows, 23);
  auto prepared = metal::PrepareInput(*device, input.types, 0, n, cols, {});
  std::vector<float> actual(n * cols);
  prepared->Readback(actual);
  for (std::size_t r = 0; r < n; ++r)
    for (unsigned f = 0; f < cols; ++f) {
      auto bin = oracle->GetGindex(r, f);
      auto expected = bin < 0          ? std::nanf("")
                      : cats && f == 2 ? oracle->cut.Values()[bin]
                      : bin == oracle->cut.Ptrs()[f]
                          ? -std::numeric_limits<float>::infinity()
                          : oracle->cut.Values()[bin - 1];
      Check((std::isnan(expected) && std::isnan(actual[r * cols + f])) ||
                actual[r * cols + f] == expected,
            "GPU input survives DMatrix destruction");
    }
  auto final = metal::GetStatistics();
  Check(final.index_uploads == before.index_uploads &&
            final.ellpack_debug_readback_bytes ==
                before.ellpack_debug_readback_bytes &&
            final.adapter_debug_readback_bytes ==
                before.adapter_debug_readback_bytes,
        "No typed GPU upload or diagnostic raw readback");
  ++cases;
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    for (int bins : {2, 17, 257, 65537}) {
      for (unsigned layout = 0; layout < 4; ++layout)
        for (bool cats : {false, true})
          for (bool weighted : {false, true})
            for (bool stream : {false, true})
              for (bool cpu : {false, true})
                Case(bins, layout, cats, weighted, stream, cpu);
      std::printf("Checked %zu cases / %zu fields\n", cases, fields);
    }
    for (bool stream : {false, true})
      for (bool cpu : {false, true})
        Case(17, 0, false, false, stream, cpu, 0);
    std::printf(
        "PASS %zu cases / %zu exact fields / %zu concurrent conversion cases\n",
        cases, fields, concurrent_cases);
  } catch (std::exception const &e) {
    std::fprintf(stderr, "FAIL case %zu: %s\n", cases, e.what());
    return 1;
  }
}
