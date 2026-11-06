// CUDA ELLPACK access must remain correct after the host bin mirror is overwritten.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_ellpack.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_input.h"
#include "../xgboost/src/common/metal_storage.h"
#include "../xgboost/src/common/metal_training.h"
#include "../xgboost/src/data/gradient_index.h"
using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, histograms = 0, partitions = 0, inputs = 0;
void Check(bool ok, char const* reason) {
  ++fields;
  if (!ok) throw std::runtime_error(reason);
}
void Same(float actual, float expected) {
  Check((std::isnan(actual) && std::isnan(expected)) || !std::memcmp(&actual, &expected, 4),
        "CUDA reconstructed input value");
}
void Case(unsigned maximum, unsigned mode, double ratio, bool copy,
          std::size_t n = 37, bool lazy = false, float snappy = -1.0f) {
  constexpr unsigned cols = 7;
  constexpr std::size_t base = 73;
  Context ctx;
  ctx.Init({{"device", "cpu"}, {"nthread", "1"}});
  common::HistogramCuts cuts{0};
  auto& ptrs = cuts.cut_ptrs_.HostVector();
  auto& cut_values = cuts.cut_values_.HostVector();
  ptrs.assign(1, 0);
  std::vector<FeatureType> types(cols, FeatureType::kNumerical);
  types[2] = FeatureType::kCategorical;
  for (unsigned f = 0; f < cols; ++f) {
    unsigned width = mode == 2 && f == 6 ? 0 : maximum;
    for (unsigned j = 0; j < width; ++j) cut_values.push_back(f == 2 ? float(j) : float(2 * j + 1));
    ptrs.push_back(cut_values.size());
  }
  SparsePage raw;
  raw.base_rowid = base;
  raw.offset.HostVector().assign(1, 0);
  for (std::size_t r = 0; r < n; ++r) {
    for (unsigned f = 0; f < cols; ++f) {
      bool valid = mode == 0 || (mode == 1 && (r == 0 || (r + f) % 3)) ||
                   (mode == 2 && f < 6 && (r + f) % 3 == 0);
      if (valid) {
        auto bin = (r * 17 + f * 31) % maximum;
        raw.data.HostVector().push_back({f, f == 2 ? float(bin) : float(2 * bin)});
      }
    }
    raw.offset.HostVector().push_back(raw.data.Size());
  }
  GHistIndexMatrix oracle{
      &ctx,      raw, types, common::HistogramCuts{cuts}, static_cast<bst_bin_t>(maximum),
      mode == 0, .2};
  std::size_t stride = 0;
  for (std::size_t r = 0; r < n; ++r)
    stride = std::max(stride, oracle.row_ptr[r + 1] - oracle.row_ptr[r]);
  auto layout = metal::MakeEllpackInfo(cuts, stride, mode == 0);
  auto packed = metal::MakeCompressedPage(layout, n, base);
  packed->Pack(oracle);
  packed->Commit(ratio, snappy, snappy >= 0);
  Check(packed->NonMissing() == oracle.index.Size(),
        "Actual compressed non-missing total");
  GHistIndexMatrix page;
  auto before = metal::GetStatistics();
  if (lazy) {
    packed->View(cuts, maximum, copy, &page);
    Check(!page.HostReadable() && page.row_ptr.empty() && page.data.empty() &&
              page.index.Size() == 0 && page.hit_count.empty(),
          "Device-only view allocates no host rows, bins or hits");
    Check(page.Size() == n && page.base_rowid == base &&
              page.Features() == cols,
          "Device-only shape is independent of host row pointers");
    auto stats = metal::GetStatistics();
    Check(stats.ellpack_restore_dispatches ==
                  before.ellpack_restore_dispatches &&
              stats.ellpack_host_readback_bytes ==
                  before.ellpack_host_readback_bytes,
          "View does not unpack or read back host mirrors");
    if (maximum == 2 && mode == 0 && ratio == 0 && !copy && n == 37) {
      auto rejected = [&](auto fn) {
        bool caught = false;
        try {
          fn();
        } catch (dmlc::Error const &) {
          caught = true;
        }
        Check(caught, "Device-only host access is rejected");
      };
      rejected([&] { (void)page.Transpose(); });
      rejected([&] { (void)page.RowIdx(base); });
      rejected([&] { page.InvalidateMetal(); });
    }
  } else {
    packed->Restore(&ctx, cuts, maximum, .2, copy, &page);
  }
  packed.reset();  // Consumers must retain both split buffers independently of the cache.
  Check(metal::GetStatistics().index_gpu_time_ns == before.index_gpu_time_ns,
        "Compressed page does not create a dense GPU transpose");
  std::fill(page.row_ptr.begin(), page.row_ptr.end(), 0);
  if (page.index.Size())
    std::memset(page.index.data<std::uint8_t>(), 0,
                page.index.Size() * page.index.GetBinTypeSize());
  std::fill(page.cut.cut_values_.HostVector().begin(), page.cut.cut_values_.HostVector().end(),
            -999.f);
  before = metal::GetStatistics();

  auto histogram = metal::MakeHistogram();
  auto cache = metal::MakeHistogramBuffer();
  cache->Resize(cuts.TotalBins());
  std::vector<GradientPair> gradients(base + n);
  for (std::size_t r = 0; r < gradients.size(); ++r)
    gradients[r] = GradientPair{float(int(r % 9) - 4) * .25f, 1.f};
  for (unsigned selection = 0; selection < 2; ++selection) {
    common::RowSetCollection rows;
    for (std::size_t r = n; r > 0; --r)
      if (!selection || r % 3) rows.Data()->push_back(base + r - 1);
    rows.Init();
    std::vector<GradientPairPrecise> expected(cuts.TotalBins());
    for (auto r : *rows.Data()) {
      auto local = r - base;
      for (auto j = oracle.row_ptr[local]; j < oracle.row_ptr[local + 1]; ++j) {
        auto bin = oracle.index[j];
        expected[bin] += GradientPairPrecise{gradients[r].GetGrad(), gradients[r].GetHess()};
      }
    }
    auto stats = metal::GetStatistics();
    histogram->Build(0, page, rows, {0}, gradients, {cache->Data()}, cache.get());
    for (std::size_t j = 0; j < expected.size(); ++j)
      Check(cache->Data()[j].GetGrad() == expected[j].GetGrad() &&
                cache->Data()[j].GetHess() == expected[j].GetHess(),
            "Exact compressed integer histogram");
    auto after = metal::GetStatistics();
    Check(after.ellpack_histogram_dispatches - stats.ellpack_histogram_dispatches ==
              (rows.Data()->empty() ? 0 : 1),
          "Compressed histogram dispatch");
    if (mode == 0 && rows.Data()->size() >= 16384 && maximum <= 257)
      Check(after.shared_histogram_dispatches > stats.shared_histogram_dispatches,
            "Shared compressed histogram path");
    ++histograms;
  }

  auto rows = metal::MakeDeviceRows();
  std::vector<std::uint32_t> categories{(1u << 31) | (1u << 28) | (1u << 24)};
  // Nonzero input offset exercises binding directly from persistent GPU storage.
  HostDeviceVector<std::uint32_t> resident_categories({0xdeadbeef, categories[0], 0xabababab},
                                                    DeviceOrd::MPS());
  for (unsigned f = 0; f < cols; ++f) {
    for (bool missing_left : {false, true}) {
      rows->Reset(n, base);
      metal::PartitionSplit split{};
      split.end = n;
      split.feature = f;
      split.split_bin = int(ptrs[f] + maximum / 2) - 1;
      split.default_left = missing_left;
      split.categorical = f == 2;
      split.cat_size = categories.size();
      std::vector<bst_idx_t> left, right;
      for (std::size_t r = 0; r < n; ++r) {
        auto bin = oracle.GetGindex(base + r, f);
        bool goes_left = missing_left;
        if (bin >= 0) {
          if (f == 2) {
            auto cat = unsigned(cut_values[bin]);
            goes_left = cat >= 32 || (categories[0] & (1u << (31 - cat))) == 0;
          } else
            goes_left = bin <= split.split_bin;
        }
        (goes_left ? left : right).push_back(base + r);
      }
      auto stats = metal::GetStatistics();
      auto input = missing_left ? resident_categories.ConstDeviceSpan().subspan(1, 1)
                                : common::Span<std::uint32_t const>{categories};
      auto counts = rows->Partition(page, {split}, input);
      if (missing_left) {
        auto after = metal::GetStatistics();
        Check(after.partition_category_shared_input_bytes -
                  stats.partition_category_shared_input_bytes == (n ? input.size_bytes() : 0),
              "Partition did not bind resident categories");
        Check(after.partition_category_host_copy_bytes == stats.partition_category_host_copy_bytes &&
                  after.vector_readback_bytes == stats.vector_readback_bytes &&
                  after.vector_upload_bytes == stats.vector_upload_bytes &&
                  !resident_categories.HostCanRead(),
              "Partition published category words through CPU memory");
      }
      Check(counts.size() == 1 && counts[0] == left.size(), "Exact compressed partition count");
      left.insert(left.end(), right.begin(), right.end());
      std::vector<bst_idx_t> actual(n);
      rows->CopyToHost(actual);
      Check(actual == left, "Stable compressed numeric/category partition");
      Check(metal::GetStatistics().ellpack_partition_dispatches -
                    stats.ellpack_partition_dispatches ==
                (n ? 1 : 0),
            "Compressed partition dispatch");
      ++partitions;
    }
  }

  if (n >= 4) {
    // Switch repeatedly between complete covers and partial/unordered ranges.
    // This detects stale buffer ownership and corruption of untouched rows.
    rows->Reset(n, base);
    std::vector<bst_idx_t> expected(n), actual(n);
    std::iota(expected.begin(), expected.end(), base);
    auto split = [&](std::size_t begin, std::size_t end, unsigned feature) {
      metal::PartitionSplit s{};
      s.begin = begin;
      s.end = end;
      s.feature = feature;
      s.split_bin = int(ptrs[feature] + maximum / 2) - 1;
      s.default_left = feature % 2;
      return s;
    };
    auto check = [&](std::vector<metal::PartitionSplit> const& splits) {
      std::vector<std::uint64_t> expected_counts;
      for (auto const& s : splits) {
        auto middle = std::stable_partition(expected.begin() + s.begin, expected.begin() + s.end,
            [&](bst_idx_t row) {
              auto bin = oracle.GetGindex(row, s.feature);
              return bin < 0 ? bool(s.default_left) : bin <= s.split_bin;
            });
        expected_counts.push_back(middle - (expected.begin() + s.begin));
      }
      Check(rows->Partition(page, splits, {}) == expected_counts, "Repeated partition counts");
      rows->CopyToHost(actual);
      Check(actual == expected, "Repeated stable partition preserves untouched rows");
      ++partitions;
    };
    check({split(0, n, 0)});
    check({split(1, n / 2, 1)});
    check({split(0, n / 2, 3), split(n / 2, n / 2, 0), split(n / 2, n, 4)});
    check({split(n / 2, n, 5), split(0, n / 2, 6)});
    check({split(0, n, 1)});
  }

  std::vector<std::int32_t> segments(cols + 2, maximum), mapping(maximum);
  segments[0] = segments[1] = segments[2] = 0;
  for (unsigned j = 0; j < maximum; ++j) mapping[j] = (j + 3) % maximum;
  for (std::size_t begin : {std::size_t{0}, n / 2, n}) {
    auto count = std::min<std::size_t>(n - begin, 23);
    auto stats = metal::GetStatistics();
    auto input = metal::PrepareInput(page, types, begin, count, cols + 1, {segments, mapping});
    std::vector<float> actual(count * (cols + 1));
    input->Readback(actual);
    for (std::size_t r = 0; r < count; ++r) {
      for (unsigned f = 0; f < cols + 1; ++f) {
        auto bin = f == cols ? -1 : oracle.GetGindex(base + begin + r, f);
        float expected = std::numeric_limits<float>::quiet_NaN();
        if (bin >= 0)
          expected = f == 2           ? float(mapping[unsigned(cut_values[bin])])
                     : bin == ptrs[f] ? -std::numeric_limits<float>::infinity()
                                      : cut_values[bin - 1];
        Same(actual[r * (cols + 1) + f], expected);
      }
    }
    Check(metal::GetStatistics().ellpack_input_dispatches - stats.ellpack_input_dispatches ==
              (count ? 1 : 0),
          "Compressed prediction input dispatch");
    ++inputs;
  }
  auto after = metal::GetStatistics();
  Check(after.index_uploads == before.index_uploads &&
            after.ellpack_restore_dispatches == before.ellpack_restore_dispatches &&
            after.ellpack_debug_readback_bytes == before.ellpack_debug_readback_bytes,
        "Consumers do not re-upload, restore or read back compressed storage");
  if (lazy) {
    Check(!page.HostReadable(), "GPU consumers leave host mirrors absent");
    page.cut =
        cuts; // Restore intentionally poisoned cut values for CPU access.
    auto view = page.metal_page;
    metal::MaterializeHost(&ctx, .2, &page);
    Check(page.HostReadable() && page.metal_page == view,
          "Host mirror retains GPU ownership");
    Check(page.Size() == n && page.index.Size() == oracle.index.Size(),
          "Host mirror shape");
    for (std::size_t r = 0; r <= n; ++r)
      Check(page.row_ptr[r] == oracle.row_ptr[r],
            "Lazily restored row pointer");
    for (std::size_t k = 0; k < page.index.Size(); ++k)
      Check(page.index[k] == oracle.index[k], "Lazily restored exact bin");
    for (std::size_t k = 0; k < page.hit_count.size(); ++k)
      Check(page.hit_count[k] == oracle.hit_count[k],
            "Lazily restored exact hit count");
    auto expected_bytes = (n + 1) * 8 +
                          page.index.Size() * page.index.GetBinTypeSize() +
                          cuts.TotalBins() * 4;
    auto host = metal::GetStatistics();
    Check(host.ellpack_host_materializations ==
                  after.ellpack_host_materializations + 1 &&
              host.ellpack_host_readback_bytes ==
                  after.ellpack_host_readback_bytes + expected_bytes,
          "One explicit host materialization with exact byte accounting");
    metal::MaterializeHost(&ctx, .2, &page);
    Check(metal::GetStatistics().ellpack_restore_dispatches ==
                  host.ellpack_restore_dispatches &&
              metal::GetStatistics().ellpack_host_materializations ==
                  host.ellpack_host_materializations,
          "Host materialization is idempotent");
  }
  ++cases;
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    for (bool lazy : {false, true}) {
      for (unsigned maximum : {2u, 17u, 257u, 65537u}) {
        for (unsigned mode = 0; mode < 4; ++mode)
          for (double ratio : {0., .000001, .5, 1.})
            for (bool copy : {false, true})
              Case(maximum, mode, ratio, copy, 37, lazy);
        std::printf("Checked %zu cases / %zu fields\n", cases, fields);
      }
      for (unsigned mode = 0; mode < 4; ++mode)
        Case(17, mode, .23, false, 0, lazy);
      // Seven columns exercise a partial four-feature tile. Include 9-bit
      // symbols and split head/tail storage on the large shared-histogram path.
      for (unsigned maximum : {2u, 17u, 257u})
        for (double ratio : {0., .5, 1.})
          Case(maximum, 0, ratio, false, 16513, lazy);
      Case(17, 2, .5, true, 5000, lazy); // Non-serial partition prefix.
    }
    for (unsigned mode = 0; mode < 4; ++mode)
      for (double ratio : {.23, 1.})
        for (float compression : {.4f, 1.f})
          for (bool copy : {false, true})
            Case(257, mode, ratio, copy, 513, true, compression);
    std::printf(
        "PASS %zu cases / %zu exact fields / %zu histograms / %zu partitions / %zu input chunks\n",
        cases, fields, histograms, partitions, inputs);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL case %zu: %s\n", cases, e.what());
    return 1;
  }
}
