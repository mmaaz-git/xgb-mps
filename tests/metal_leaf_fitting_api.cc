// Independent leaf sums and the upstream regularized weight formula.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_leaf.h"
#include "../xgboost/src/common/metal_training.h"
#include "../xgboost/src/tree/param.h"

using namespace xgboost;
namespace metal = common::metal;
using common::Span;
std::size_t cases = 0, pairs = 0, sums_checked = 0, weights_checked = 0;
std::string current;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Same(double actual, double expected) {
  if (std::isnan(actual) && std::isnan(expected)) return;
  if (actual != expected) {
    std::printf("GPU %.17g CPU %.17g\n", actual, expected);
    Check(false, "Gradient sum differs");
  }
  ++sums_checked;
}
void WeightSame(float actual, float expected) {
  ++weights_checked;
  if (std::isnan(actual) && std::isnan(expected)) return;
  if (std::memcmp(&actual, &expected, 4)) {
    std::printf("GPU weight %.9g CPU %.9g\n", actual, expected);
    Check(false, "Weight bits differ");
  }
}
void Weights(metal::LeafSums const& sums) {
  auto reference = sums.Data();
  for (unsigned kind = 0; kind < 5; ++kind) {
    tree::TrainParam p;
    p.reg_lambda = kind == 0 ? 0 : kind == 4 ? 1e-30f : 1.25f;
    p.reg_alpha = kind == 2 ? .75f : kind == 4 ? 1e-40f : 0;
    p.max_delta_step = kind == 3 ? .125f : 0;
    // min_child_weight governs split validity, not the final CalcWeight formula.
    p.min_child_weight = 1e20f;
    float eta = kind == 4 ? 1e-30f : .3f;
    for (unsigned bounded = 0; bounded < 4; ++bounded) {
      std::vector<metal::SplitBounds> bounds;
      if (bounded) {
        auto lower = bounded == 1 ? -.1f : bounded == 2 ? .125f : -1e-40f;
        auto upper = bounded == 1 ? .2f : bounded == 2 ? .25f : 1e-40f;
        bounds.resize(reference.size(), {lower, upper});
      }
      std::vector<float> output(reference.size() + 2, -913.f);
      auto before = metal::GetStatistics();
      sums.Weights({p.reg_lambda, p.reg_alpha, p.max_delta_step, p.min_child_weight}, bounds, eta,
                   {output.data() + 1, reference.size()});
      auto after = metal::GetStatistics();
      Check(after.leaf_weight_dispatches - before.leaf_weight_dispatches == !reference.empty(),
            "Weight counter mismatch");
      for (std::size_t i = 0; i < reference.size(); ++i) {
        float expected = tree::CalcWeight(p, reference[i]);
        if (bounded) {
          if (expected < bounds[i].lower)
            expected = bounds[i].lower;
          else if (expected > bounds[i].upper)
            expected = bounds[i].upper;
        }
        expected *= eta;
        WeightSame(output[i + 1], expected);
      }
      Check(output.front() == -913.f && output.back() == -913.f, "Weight guard overwritten");
    }
  }
}
void Run(std::size_t rows, std::size_t targets, unsigned layout, unsigned mode) {
  current = "rows=" + std::to_string(rows) + " targets=" + std::to_string(targets) +
            " layout=" + std::to_string(layout) + " mode=" + std::to_string(mode);
  std::size_t total_rows = rows + 11;
  std::size_t stride0 = layout == 1 ? 1 : layout == 2 ? targets * 2 + 3 : targets;
  std::size_t stride1 = layout == 1 ? total_rows + 3 : layout == 2 ? 2 : 1;
  if (layout == 3) stride0 = 0;
  if (layout == 4) {
    stride0 = 1;
    stride1 = 0;
  }
  auto extent = (total_rows - 1) * stride0 + (targets - 1) * stride1 + 1;
  std::vector<GradientPair> storage(extent + 8, GradientPair{981, -237});
  auto view = linalg::MatrixView<GradientPair>{
      {storage.data() + 4, extent}, {total_rows, targets}, {stride0, stride1}, DeviceOrd::CPU()};
  std::mt19937 rng(317);
  for (std::size_t r = 5; r < rows + 5; ++r) {
    for (std::size_t t = 0; t < targets; ++t) {
      float g = float(int(rng() % 2047) - 1023) / 16;
      float h = float(rng() % 511) / 8;
      if (mode == 1) {
        g = std::ldexp(g, -140);
        h = std::ldexp(h, -140);
      }
      if (mode == 2) {
        g = std::ldexp(g, 110);
        h = std::ldexp(h, 110);
      }
      if (mode == 3) {
        g = 1;
        h = t % 2 ? -1 : 0;
      }
      if (mode == 4) {
        float v[] = {0, -0.f, INFINITY, -INFINITY, NAN, 1e-40f, -1e-40f};
        g = v[t % 7];
        h = v[(t + 2) % 7];
      }
      view(r, t) = GradientPair{g, h};
    }
  }
  auto original = storage;
  auto before = metal::GetStatistics();
  auto output = metal::MakeLeafSums(7, targets);
  Check(metal::GetStatistics().leaf_sum_init_dispatches == before.leaf_sum_init_dispatches + 1,
        "Missing GPU initialization");
  std::vector<GradientPairPrecise> reference(7 * targets);
  std::size_t boundaries[]{0, rows / 3, rows * 2 / 3, rows};
  for (unsigned page = 0; page < 3; ++page) {
    auto begin = boundaries[page], end = boundaries[page + 1];
    auto indices = metal::MakeDeviceRows();
    indices->Reset(end - begin, begin + 5);
    std::vector<metal::LeafSumSegment> segments;
    for (unsigned leaf = 0; leaf < 7; ++leaf) {
      auto slot = (leaf * 3 + 2) % 7;
      auto lo = std::max(begin, rows * std::min(leaf, 5u) / 5);
      auto hi = std::min(end, rows * std::min(leaf + 1, 5u) / 5);
      if (hi <= lo) {
        segments.push_back({0, 0, slot, 0});
        continue;
      }
      segments.push_back({lo - begin, hi - begin, slot, 0});
      for (auto row = lo; row < hi; ++row) {
        for (std::size_t t = 0; t < targets; ++t) {
          reference[slot * targets + t] += GradientPairPrecise{view(row + 5, t)};
          ++pairs;
        }
      }
    }
    // Descriptor order must not matter, nor must output slots match row order.
    std::reverse(segments.begin(), segments.end());
    auto previous = metal::GetStatistics();
    indices->AddLeafGradients(segments, view, output.get());
    auto after = metal::GetStatistics();
    Check(after.leaf_sum_dispatches - previous.leaf_sum_dispatches == bool(end - begin),
          "Wrong per-page sum dispatch count");
    for (std::size_t i = 0; i < reference.size(); ++i) {
      Same(output->Data()[i].GetGrad(), reference[i].GetGrad());
      Same(output->Data()[i].GetHess(), reference[i].GetHess());
    }
  }
  Check(!std::memcmp(storage.data(), original.data(), storage.size() * sizeof(GradientPair)),
        "Gradient input/padding changed");
  Weights(*output);
  ++cases;
  if (cases % 30 == 0 || rows > 1000000) {
    std::printf("Checked %zu leaf-fitting cases / %zu source pairs / %zu weights\n", cases, pairs,
                weights_checked);
    std::fflush(stdout);
  }
}
void Rejections() {
  auto empty = metal::MakeLeafSums(0, 1);
  Weights(*empty);
  auto output = metal::MakeLeafSums(2, 1);
  auto indices = metal::MakeDeviceRows();
  indices->Reset(2, 0);
  std::vector<GradientPair> input(2, {1, 1});
  auto view = linalg::MakeTensorView(DeviceOrd::CPU(), Span{input}, 2, 1);
  for (auto segments : {std::vector<metal::LeafSumSegment>{{0, 3, 0, 0}},
                        std::vector<metal::LeafSumSegment>{{0, 2, 2, 0}},
                        std::vector<metal::LeafSumSegment>{{0, 1, 0, 0}, {1, 2, 0, 0}}}) {
    bool failed = false;
    auto before = metal::GetStatistics().leaf_sum_dispatches;
    try {
      indices->AddLeafGradients(segments, view, output.get());
    } catch (std::exception const&) {
      failed = true;
    }
    Check(failed, "Invalid leaf descriptors accepted");
    Check(metal::GetStatistics().leaf_sum_dispatches == before, "Rejected input dispatched");
  }
}
int main() {
  try {
    for (std::size_t rows : {0ul, 1ul, 31ul, 4097ul, 65539ul})
      for (std::size_t targets : {1ul, 3ul, 7ul})
        for (unsigned layout = 0; layout < 5; ++layout)
          for (unsigned mode = 0; mode < 4; ++mode) Run(rows, targets, layout, mode);
    for (unsigned layout = 0; layout < 3; ++layout) Run(1000003, 3, layout, 0);
    Run(7, 7, 0, 4);
    Rejections();
    std::printf(
        "PASS %zu leaf-fitting cases / %zu source pairs / %zu exact sum fields / %zu exact CPU "
        "weights; layouts, pages, bounds, guards and errors\n",
        cases, pairs, sums_checked, weights_checked);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
