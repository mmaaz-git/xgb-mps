// Ordinary matrix cache generations, explicit CPU conversion and retained GPU
// views.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/column_matrix.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_input.h"
#include "../xgboost/src/data/adapter.h"
#include "../xgboost/src/data/gradient_index.h"
using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, concurrent_cases = 0, generations = 0;
void Check(bool ok, char const *message) {
  ++fields;
  if (!ok)
    throw std::runtime_error(message);
}
auto Page(DMatrix *matrix, Context const *ctx, BatchParam p = {}) {
  return matrix->GetBatches<GHistIndexMatrix>(ctx, p).begin().Page();
}
void Case(int bins, unsigned layout, bool cats, bool weighted, bool approx,
          std::size_t rows = 1031) {
  constexpr std::size_t cols = 7;
  std::vector<float> values(std::max<std::size_t>(1, rows * cols)), hess(rows);
  std::vector<FeatureType> types(cols, FeatureType::kNumerical);
  if (cats)
    types[2] = FeatureType::kCategorical;
  for (std::size_t r = 0; r < rows; ++r) {
    hess[r] = float(r % 7 + 1) / 4;
    for (std::size_t f = 0; f < cols; ++f) {
      float v = cats && f == 2 ? float(r % 7)
                               : float(int((r * 193 + f * 17) % 997) - 498) / 8;
      if (layout == 3 || (layout == 1 && r && (r + f) % 5 == 0) ||
          (layout == 2 && (f == 6 || (r + f) % 17 != 0)))
        v = -99;
      values[r * cols + f] = v;
    }
  }
  data::DenseAdapter adapter(values.data(), rows, cols);
  std::shared_ptr<DMatrix> matrix{DMatrix::Create(&adapter, -99.f, 2)};
  matrix->Info().feature_types.HostVector() = types;
  if (weighted) {
    auto &weights = matrix->Info().weights_.HostVector();
    weights.resize(rows);
    for (std::size_t r = 0; r < rows; ++r)
      weights[r] = float(r % 9) / 4;
  }
  Context cpu, gpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  gpu.Init({{"device", "mps"}, {"nthread", "2"}});
  auto p = approx ? BatchParam{bins, hess, true} : BatchParam{bins, .2};
  auto before = metal::GetStatistics();
  auto device = Page(matrix.get(), &gpu, p);
  auto ready = metal::GetStatistics();
  Check(ready.ellpack_csr_values - before.ellpack_csr_values ==
            matrix->Info().num_nonzero_,
        "Direct raw CSR compression covers every non-missing value");
  Check(ready.binning_dispatches == before.binning_dispatches &&
            ready.index_uploads == before.index_uploads &&
            ready.ellpack_pack_input_bytes == before.ellpack_pack_input_bytes &&
            ready.ellpack_host_materializations ==
                before.ellpack_host_materializations,
        "Construction has no host bins, host mirrors or typed bin uploads");
  Check(!device->HostReadable() && device->row_ptr.empty() &&
            device->index.Size() == 0 && device->hit_count.empty() &&
            device->Size() == rows,
        "GPU-only quantized view");
  BatchParam metadata;
  metadata.for_cuts = true;
  Check(Page(matrix.get(), &cpu, metadata) == device,
        "CPU metadata reuses GPU view");
  auto fixed = p.MakeCache();
  fixed.forbid_regen = fixed.for_cuts = true;
  Check(Page(matrix.get(), &cpu, fixed) == device,
        "Fixed reference keeps cuts and view");
  bool rejected = false;
  auto invalid = fixed;
  ++invalid.max_bin;
  try {
    Page(matrix.get(), &gpu, invalid);
  } catch (std::exception const &) {
    rejected = true;
  }
  Check(rejected, "Fixed reference rejects changed bins");
  Check(metal::GetStatistics().ellpack_host_materializations ==
            ready.ellpack_host_materializations,
        "Metadata never materializes CPU bins");
  auto raw = matrix->GetBatches<SparsePage>().begin().Page();
  GHistIndexMatrix oracle(&cpu, *raw, types, device->cut, bins,
                          matrix->IsDense(), p.sparse_thresh);
  std::vector<std::shared_ptr<GHistIndexMatrix const>> readers(6);
  auto get = [&](bool on_gpu) {
    return Page(matrix.get(), on_gpu ? &gpu : &cpu);
  };
  if (bins == 17 && !cats && !weighted) {
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<std::shared_ptr<GHistIndexMatrix const>>> futures;
    for (unsigned i = 0; i < readers.size(); ++i)
      futures.push_back(std::async(std::launch::async, [&, i] {
        gate.wait();
        return get(i % 2);
      }));
    start.set_value();
    for (unsigned i = 0; i < readers.size(); ++i)
      readers[i] = futures[i].get();
    ++concurrent_cases;
  } else {
    for (unsigned i = 0; i < readers.size(); ++i)
      readers[i] = get(i % 2);
  }
  auto host = readers[0];
  for (unsigned i = 0; i < readers.size(); ++i)
    Check(readers[i] == (i % 2 ? device : host),
          "One published view per representation");
  Check(host != device && host->HostReadable() && !device->HostReadable(),
        "Independent CPU mirror");
  Check(host->index.Size() == oracle.index.Size() &&
            host->hit_count.size() == oracle.hit_count.size(),
        "CPU index and hit shapes");
  for (std::size_t r = 0; r <= rows; ++r)
    Check(host->row_ptr[r] == oracle.row_ptr[r], "Exact CPU row offset");
  for (std::size_t k = 0; k < host->index.Size(); ++k)
    Check(host->index[k] == oracle.index[k],
          "Exact independently searched CPU bin");
  for (std::size_t k = 0; k < host->hit_count.size(); ++k)
    Check(host->hit_count[k] == oracle.hit_count[k], "Exact CPU bin hit count");
  if (!approx) {
    auto const &columns = host->Transpose();
    common::DispatchBinType(columns.GetTypeSize(), [&](auto type) {
      using T = decltype(type);
      for (unsigned f = 0; f < cols; ++f) {
        auto verify = [&](auto column) {
          for (std::size_t r = 0; r < rows; ++r)
            Check(column[r] == oracle.GetGindex(r, f),
                  "Exact dense/sparse CPU column");
        };
        if (columns.GetColumnType(f) == common::kSparseColumn)
          verify(columns.SparseColumn<T>(f, 0));
        else if (columns.AnyMissing())
          verify(columns.DenseColumn<T, true>(f));
        else
          verify(columns.DenseColumn<T, false>(f));
      }
    });
  }
  Check(metal::GetStatistics().ellpack_host_materializations ==
            before.ellpack_host_materializations + 1,
        "Exactly one CPU materialization, including concurrent access");
  // A new mutable generation must not mutate retained readers or reuse stale
  // cuts.
  if (bins == 17) {
    auto changed = p;
    changed.max_bin = 33;
    auto replacement = Page(matrix.get(), &gpu, changed);
    Check(replacement != device && replacement->max_numeric_bins_per_feat == 33,
          "Changed bins publish a new compressed generation");
    Check(Page(matrix.get(), &cpu, metadata) == replacement,
          "Metadata uses latest generation");
    Check(host->max_numeric_bins_per_feat == 17 &&
              device->max_numeric_bins_per_feat == 17,
          "Retained readers preserve old generation");
    ++generations;
  }
  matrix.reset();
  auto n = std::min<std::size_t>(rows, 23);
  auto prepared = metal::PrepareInput(*device, types, 0, n, cols, {});
  std::vector<float> actual(n * cols);
  prepared->Readback(actual);
  for (std::size_t r = 0; r < n; ++r)
    for (unsigned f = 0; f < cols; ++f) {
      auto bin = oracle.GetGindex(r, f);
      auto expected = bin < 0          ? std::nanf("")
                      : cats && f == 2 ? oracle.cut.Values()[bin]
                      : bin == oracle.cut.Ptrs()[f]
                          ? -std::numeric_limits<float>::infinity()
                          : oracle.cut.Values()[bin - 1];
      Check((std::isnan(expected) && std::isnan(actual[r * cols + f])) ||
                actual[r * cols + f] == expected,
            "Retained GPU data survives regeneration and matrix destruction");
    }
  ++cases;
}
void CPUOrigin() {
  std::vector<float> values(1031 * 7);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = float(i % 199) / 7;
  data::DenseAdapter adapter(values.data(), 1031, 7);
  std::shared_ptr<DMatrix> matrix{DMatrix::Create(&adapter, -99.f, 2)};
  Context cpu, gpu;
  cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
  gpu.Init({{"device", "mps"}, {"nthread", "2"}});
  auto p = BatchParam{17, .2};
  auto host = Page(matrix.get(), &cpu, p);
  auto before = metal::GetStatistics();
  auto metadata = p;
  metadata.forbid_regen = metadata.for_cuts = true;
  Check(Page(matrix.get(), &gpu, metadata) == host,
        "MPS metadata retains existing CPU view");
  Check(metal::GetStatistics().ellpack_pack_input_bytes ==
            before.ellpack_pack_input_bytes,
        "CPU metadata does not upload bins");
  auto device = Page(matrix.get(), &gpu); // Empty params preserve the CPU cuts.
  Check(device->cut.Values() == host->cut.Values(),
        "Fixed CPU cuts survive conversion");
  auto converted = metal::GetStatistics();
  Check(converted.ellpack_pack_input_bytes - before.ellpack_pack_input_bytes ==
            host->index.Size() + host->row_ptr.size() * 8,
        "CPU bins upload once");
  Check(Page(matrix.get(), &gpu) == device && Page(matrix.get(), &cpu) == host,
        "Converted views reused");
  auto regenerated = Page(matrix.get(), &gpu, p);
  Check(regenerated != device && metal::GetStatistics().ellpack_csr_values >
                                     converted.ellpack_csr_values,
        "Mutable MPS training regenerates CUDA cuts from raw CSR");
  Check(metal::GetStatistics().ellpack_pack_input_bytes ==
            converted.ellpack_pack_input_bytes,
        "Regeneration avoids stale CPU bin upload");
  auto cpu_again = Page(matrix.get(), &cpu, p);
  Check(cpu_again != host && cpu_again->cut.Values() == host->cut.Values(),
        "Mutable CPU training restores CPU sketch generation");
  ++generations;
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    for (int bins : {2, 17, 257, 65537}) {
      for (unsigned layout = 0; layout < 4; ++layout)
        for (bool cats : {false, true})
          for (bool weighted : {false, true})
            for (bool approx : {false, true})
              Case(bins, layout, cats, weighted, approx);
      std::printf("Checked %zu cases / %zu fields\n", cases, fields);
    }
    for (bool approx : {false, true})
      Case(17, 0, false, false, approx, 0);
    CPUOrigin();
    std::printf("PASS %zu cases / %zu exact fields / %zu concurrent cases / "
                "%zu regenerated views\n",
                cases, fields, concurrent_cases, generations);
  } catch (std::exception const &e) {
    std::fprintf(stderr, "FAIL case %zu: %s\n", cases, e.what());
    return 1;
  }
}
