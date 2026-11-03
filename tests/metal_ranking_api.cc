// Production grouped Metal sorting against independent stable CPU sorting.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ranking.h"

void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  try {
    using namespace xgboost;
    std::vector<std::vector<bst_group_t>> cases{{},
                                                {0, 0, 0},
                                                {1},
                                                {31, 0, 32, 33},
                                                {255, 256, 0, 1},
                                                {257, 0, 1, 256},
                                                {0, 65537, 0},
                                                {65537, 1, 4097, 0, 257}};
    std::vector<bst_group_t> many(4097), skew(65537, 1);
    for (size_t g = 0; g < many.size(); ++g) many[g] = g % 257;
    skew[32768] = 257;
    cases.push_back(many);
    cases.push_back(skew);
    std::mt19937 rng(761);
    size_t checked = 0, count = 0, small = 0, radix = 0;
    for (auto const& sizes : cases) {
      std::vector<bst_group_t> groups{0};
      for (auto size : sizes) groups.push_back(groups.back() + size);
      size_t n = groups.back();
      bool large = std::any_of(sizes.begin(), sizes.end(), [](auto v) { return v > 256; });
      for (size_t stride : {1, 3}) {
        for (int mode = 0; mode < 5; ++mode) {
          std::vector<float> data(n * stride, std::numeric_limits<float>::quiet_NaN());
          for (size_t i = 0; i < n; ++i) {
            float v = float(int(rng() % 101) - 50) / 7;
            if (mode == 1) v = i % 3 ? -0.0f : 0.0f;
            if (mode == 2) v = float(i);
            if (mode == 3) v = -float(i);
            if (mode == 4) {
              constexpr uint32_t bits[]{0,          0x80000000, 1,          0x80000001,
                                        0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
                                        0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff};
              auto b = bits[rng() % (sizeof(bits) / sizeof(bits[0]))];
              std::memcpy(&v, &b, sizeof(v));
            }
            data[i * stride] = v;
          }
          auto values =
              linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{data}, n, stride)
                  .Slice(linalg::All(), 0);
          std::vector<size_t> actual(n), expected(n);
          for (size_t g = 0; g < sizes.size(); ++g) {
            auto begin = expected.begin() + groups[g], end = expected.begin() + groups[g + 1];
            std::iota(begin, end, 0);
            std::stable_sort(begin, end, [&](size_t a, size_t b) {
              return values(groups[g] + a) > values(groups[g] + b);
            });
          }
          auto before = common::metal::GetStatistics();
          common::metal::SortGroups(values, groups, actual);
          auto after = common::metal::GetStatistics();
          Check(after.group_sort_dispatches - before.group_sort_dispatches == uint64_t(n != 0),
                "Grouped sort did not complete on GPU");
          Check(after.radix_sort_dispatches - before.radix_sort_dispatches == uint64_t(n && large),
                "Grouped sort selected an unexpected path");
          if (actual != expected) {
            for (size_t i = 0; i < n; ++i)
              if (actual[i] != expected[i]) {
                std::fprintf(stderr,
                             "rows=%zu groups=%zu stride=%zu mode=%d index=%zu: %zu != %zu\n", n,
                             sizes.size(), stride, mode, i, actual[i], expected[i]);
                break;
              }
            throw std::runtime_error("Grouped sort differs from stable CPU order");
          }
          checked += n;
          ++count;
          if (n) (large ? radix : small)++;
        }
      }
      std::printf("Checked %zu cases / %zu sorted rows\n", count, checked);
      std::fflush(stdout);
    }
    std::printf("PASS %zu grouped sort cases / %zu rows; %zu small-query and %zu radix launches\n",
                count, checked, small, radix);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
