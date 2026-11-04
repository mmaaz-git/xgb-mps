// Exact CUDA-source query metadata and immutable production cache ownership.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_ranking.h"
#include "../xgboost/src/common/ranking_utils.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, inexact_double = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
template <typename Fn>
void Reject(Fn fn) {
  bool rejected = false;
  try {
    fn();
  } catch (dmlc::Error const&) {
    rejected = true;
  }
  Check(rejected, "Invalid metadata was accepted");
}
auto View(std::vector<float> const& values) {
  return linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{values}, values.size());
}
void Case(std::vector<bst_group_t> const& sizes, std::uint32_t k, bool mean) {
  std::vector<bst_group_t> groups{0};
  for (auto size : sizes) {
    Check(std::uint64_t{groups.back()} + size <= std::numeric_limits<bst_group_t>::max(),
          "Invalid test fixture");
    groups.push_back(groups.back() + size);
  }
  auto before = metal::GetStatistics();
  auto plan = metal::MakeRankingPlan(groups, k, mean);
  auto after = metal::GetStatistics();
  Check(after.ranking_plan_dispatches == before.ranking_plan_dispatches + 1,
        "Metadata did not execute on GPU");
  Check(plan->NumPair() == k && plan->IsMean() == mean, "Pair parameters differ");
  Check(plan->Groups().size() == groups.size() &&
            std::equal(groups.begin(), groups.end(), plan->Groups().begin()),
        "Owned query offsets differ");
  Check(plan->MaxGroupSize() == (sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end())),
        "Maximum query size differs");
  auto tasks = plan->Tasks();
  Check(tasks.size() == groups.size(), "Missing prefix sentinel");
  unsigned __int128 pairs = 0, tiles = 0;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    auto const& task = tasks[g];
    Check(task.pair_begin == pairs && task.tile_begin == tiles, "Exact integer prefix differs");
    if (static_cast<unsigned __int128>(double(task.pair_begin)) != pairs) ++inexact_double;
    auto count = g < sizes.size() ? sizes[g] : 0;
    auto topk = g == sizes.size() ? 0 : mean ? k : std::min(k, count);
    Check(task.begin == groups[g] && task.count == count && task.topk == topk && task.padding == 0,
          "Query task shape differs");
    fields += 6;
    if (g == sizes.size()) break;
    // CUDA DiscreteTrapezoidArea uses the arithmetic-series area; use a wider
    // host type here to independently detect accidental intermediate overflow.
    unsigned __int128 n = count, h = k, total = n * h;
    if (!mean) {
      total = 0;
      if (n && h) {
        --n;
        h = std::min(n, h);
        if (h) total = ((n - (h - 1)) + n) * h / 2;
      }
      if (count < 100) {
        std::size_t enumerated = 0;
        for (std::size_t i = 0; i < std::min(k, count); ++i)
          for (std::size_t j = i + 1; j < count; ++j) ++enumerated;
        Check(total == enumerated, "Source formula and pair enumeration disagree");
      }
    }
    pairs += total;
    tiles += (total + 255) / 256;
  }
  Check(pairs <= std::numeric_limits<std::uint64_t>::max(), "Oracle total overflow");
  ++cases;
}

void Cache() {
  Context ctx;
  ctx.Init({{"device", "mps"}, {"nthread", "2"}});
  MetaInfo info;
  info.num_row_ = 531;
  info.group_ptr_ = {0, 0, 7, 264, 264, 531, 531};
  info.labels.Reshape(531, 1);
  for (std::size_t i = 0; i < 531; ++i) info.labels.HostView()(i, 0) = i % 3;
  std::vector<float> scores(531);
  for (std::size_t i = 0; i < scores.size(); ++i) scores[i] = int(i % 23) - 11;
  std::vector<std::size_t> expected(scores.size());
  for (std::size_t g = 1; g < info.group_ptr_.size(); ++g) {
    auto begin = info.group_ptr_[g - 1], end = info.group_ptr_[g];
    std::iota(expected.begin() + begin, expected.begin() + end, 0);
    std::stable_sort(expected.begin() + begin, expected.begin() + end,
                     [&](auto a, auto b) { return scores[begin + a] > scores[begin + b]; });
  }
  for (bool mean : {false, true}) {
    ltr::LambdaRankParam param;
    param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", mean ? "mean" : "topk"},
                                  {"lambdarank_num_pair_per_sample", "7"}});
    auto before = metal::GetStatistics();
    auto cache = std::make_unique<ltr::RankingCache>(&ctx, info, param);
    auto plan = cache->MetalPlan();
    auto identity = plan->Tasks().data();
    Check(plan->MaxGroupSize() == 267 && cache->MaxPositionSize() == (mean ? 32 : 7),
          "RankingCache did not use GPU maximum");
    for (int round = 0; round < 3; ++round) {
      auto rank = cache->SortedIdx(&ctx, scores);
      Check(std::equal(rank.begin(), rank.end(), expected.begin()), "Cached sort differs");
      Check(plan->Tasks().data() == identity, "Cached task storage was replaced");
    }
    auto after = metal::GetStatistics();
    Check(after.ranking_plan_dispatches == before.ranking_plan_dispatches + 1 &&
              after.ranking_plan_reuses == before.ranking_plan_reuses + 3,
          "Query metadata was recomputed between cache uses");
    cache.reset();  // A consumer can retain the immutable GPU buffers independently.
    std::vector<GradientPair> saved(scores.size()), fresh(scores.size());
    auto labels = info.labels.HostView().Slice(linalg::All(), 0);
    for (std::uint32_t seed : {0u, 919u, 0xffffffffu}) {
      metal::RankingGradient(metal::RankingLossKind::kPairwise, scores, labels, info.group_ptr_, {},
                             7, true, true, true, 1., {}, {}, saved, mean, seed, {}, {}, {}, plan);
      metal::RankingGradient(metal::RankingLossKind::kPairwise, scores, labels, info.group_ptr_, {},
                             7, true, true, true, 1., {}, {}, fresh, mean, seed);
      Check(std::memcmp(saved.data(), fresh.data(), saved.size() * sizeof(GradientPair)) == 0,
            "Cached and fresh objectives differ");
    }
    std::vector<std::size_t> rank(scores.size());
    // Mutation at the original host address must not silently reuse old groups.
    info.group_ptr_[2] = 8;
    Check(plan->Groups()[2] == 7, "Plan aliases caller-owned offsets");
    Reject([&] { metal::SortGroups(View(scores), info.group_ptr_, rank, plan); });
    info.group_ptr_[2] = 7;
    Reject([&] {
      metal::RankingGradient(metal::RankingLossKind::kPairwise, scores, labels, info.group_ptr_, {},
                             8, true, true, true, 1., {}, {}, saved, mean, 0, {}, {}, {}, plan);
    });
    Reject([&] {
      metal::RankingGradient(metal::RankingLossKind::kPairwise, scores, labels, info.group_ptr_, {},
                             7, true, true, true, 1., {}, {}, saved, !mean, 0, {}, {}, {}, plan);
    });
  }
  std::puts(
      "Cached GPU metadata: both methods, retained ownership, exact sorts/gradient replay, "
      "seed changes and stale-group/parameter guards PASS");
}

int main() {
  try {
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::vector<bst_group_t>> inputs{
        {},
        {0},
        {0, 0, 0},
        {1},
        {1, 0, 2, 3, 7, 32, 255, 256, 257, 4096, 4097, 65537, 0},
        {max},
        {0, max, 0},
        {max - 2, 1, 0, 1},
        {max / 2, 0, max - max / 2}};
    for (auto const& input : inputs)
      for (auto k : {0u, 1u, 2u, 7u, 32u, 256u, 65537u, max})
        for (bool mean : {false, true}) Case(input, k, mean);
    std::mt19937 rng(672311);
    for (auto count : {254, 255, 256, 257, 65534, 65535, 65536, 65537, 262145}) {
      std::vector<bst_group_t> sizes(count);
      for (auto& size : sizes) size = rng() % 65;
      for (auto k : {3u, max})
        for (bool mean : {false, true}) Case(sizes, k, mean);
    }
    Check(inexact_double > 0, "No precision-sensitive integer prefixes checked");
    for (std::vector<bst_group_t> bad : {std::vector<bst_group_t>{}, {1}, {0, 7, 3}, {0, 3, 2, 7}})
      Reject([&] { metal::MakeRankingPlan(bad, max, true); });
    Cache();
    std::printf(
        "CUDA ranking metadata PASS: %zu cases / %zu exact task fields / "
        "%zu prefixes not representable in binary64\n",
        cases, fields, inexact_double);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "Ranking metadata FAIL: %s\n", e.what());
    return 1;
  }
}
