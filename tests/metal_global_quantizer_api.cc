// CUDA global quantizer/stump control flow, independent scalar arithmetic replay.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_collective.h"
#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_leaf.h"
#include "../xgboost/src/common/metal_reduction.h"
#include "../xgboost/src/common/metal_training.h"
#include "../xgboost/src/common/stats.h"
#include "../xgboost/src/tree/fit_stump.h"
#include "../xgboost/src/tree/hist/sampler.h"
#include "cuda_gradient_reference.h"
#include "xgboost/c_api.h"

using namespace xgboost;
namespace metal = common::metal;
namespace ref = cuda_reference;
std::string current;
std::size_t cases = 0, fields = 0, rejected = 0;
void Check(bool ok, char const* reason) {
  if (!ok) throw std::runtime_error(current + ": " + reason);
}
void Api(int status) {
  if (status) throw std::runtime_error(XGBGetLastError());
}
template <typename T>
void Same(T a, T b) {
  ++fields;
  if (std::memcmp(&a, &b, sizeof(T))) {
    std::fprintf(stderr, "actual %a expected %a\n", double(a), double(b));
    Check(false, "Exact field mismatch");
  }
}
std::size_t Rows(int rank, unsigned layout) {
  if (layout == 3 || (layout == 1 && rank == 0) || (layout == 2 && rank != 0)) return 0;
  return layout == 4 ? 1 : 257 + rank * 31;
}
ref::Pair FixtureValue(int rank, std::size_t row, std::size_t target, unsigned mode) {
  float g = std::ldexp(float(int(row % 13) - 6), int(target % 3) + rank * 3 - 8);
  float h = std::ldexp(float(1 + row % 5), int(target % 4) + rank - 6);
  if (mode == 1) g = h = 0;
  if (mode == 2) {
    g = std::ldexp(float(int(row % 7) - 3), -149 + int(target % 3));
    h = std::ldexp(float(1 + row % 3), -149 + rank);
  }
  if (mode == 3) {
    // Common scales must reflect another rank's extreme values. Tiny positive
    // Hessians must still quantize to one even when global scales truncate them.
    g = std::ldexp(float(int(row % 7) - 3), rank == 0 ? 100 : -80);
    h = row == 0 ? std::ldexp(1.f, rank == 0 ? 110 : 50) : 0x1p-140f;
  }
  if (mode == 4) h = -h;
  if (mode == 5) {
    g = row == 0 ? std::ldexp(1.f, int(target % 3) * 30 - 30) : 0.f;
    h = row == 0 ? 1.f : 0.f;
  }
  return {g, h};
}
std::vector<ref::Scale> Scales(int world, unsigned layout, unsigned mode, std::size_t targets) {
  std::vector<ref::Scale> scales(targets);
  std::size_t total_rows = 0;
  for (int rank = 0; rank < world; ++rank) total_rows += Rows(rank, layout);
  for (std::size_t t = 0; t < targets; ++t) {
    double pos[2]{}, neg[2]{};
    for (int rank = 0; rank < world; ++rank) {
      double local_pos[2]{}, local_neg[2]{};
      for (std::size_t row = 0; row < Rows(rank, layout); ++row) {
        auto pair = FixtureValue(rank, row, t, mode);
        float values[]{pair.g, pair.h};
        for (unsigned k = 0; k < 2; ++k) {
          if (values[k] >= 0)
            local_pos[k] += values[k];
          else
            local_neg[k] -= values[k];
        }
      }
      for (unsigned k = 0; k < 2; ++k) {
        pos[k] += local_pos[k];
        neg[k] += local_neg[k];
      }
    }
    scales[t] = {ref::Exponent(pos[0], neg[0], total_rows),
                 ref::Exponent(pos[1], neg[1], total_rows)};
  }
  return scales;
}
ref::IntegerPair Fixed(ref::Pair v, ref::Scale scale) {
  ref::IntegerPair result{static_cast<std::int64_t>(std::ldexp(double(v.g), scale.g)),
                          static_cast<std::int64_t>(std::ldexp(double(v.h), scale.h))};
  if (v.h > 0 && !result.h) result.h = 1;
  return result;
}
void Verify(metal::QuantizedGradients const& state, std::vector<ref::IntegerPair> const& values,
            std::vector<ref::Scale> const& scales) {
  auto targets = scales.size();
  std::vector<GradientPairPrecise> result(targets);
  std::vector<GradientPairInt64> integers(targets);
  state.Sum(linalg::MakeVec(result.data(), targets), integers);
  for (std::size_t t = 0; t < targets; ++t) {
    std::uint64_t g = 0, h = 0;
    for (std::size_t r = 0; r < state.Rows(); ++r) {
      g += values[r * targets + t].g;
      h += values[r * targets + t].h;
    }
    std::int64_t sg, sh;
    std::memcpy(&sg, &g, 8);
    std::memcpy(&sh, &h, 8);
    Same(integers[t].GetQuantisedGrad(), sg);
    Same(integers[t].GetQuantisedHess(), sh);
    Same(result[t].GetGrad(), std::ldexp(double(sg), -scales[t].g));
    Same(result[t].GetHess(), std::ldexp(double(sh), -scales[t].h));
  }
}
void Run(Context const& ctx, int rank, int world, unsigned layout, unsigned mode,
         std::size_t targets) {
  current = "rank" + std::to_string(rank) + "/world" + std::to_string(world) + "/layout" +
            std::to_string(layout) + "/mode" + std::to_string(mode) + "/targets" +
            std::to_string(targets);
  auto rows = Rows(rank, layout);
  auto scales = Scales(world, layout, mode, targets);
  std::vector<ref::Pair> source(rows * targets);
  std::vector<ref::IntegerPair> fixed(rows * targets);
  linalg::Matrix<GradientPair> input({rows, targets}, DeviceOrd::MPS(),
                                     mode % 2 ? linalg::Order::kF : linalg::Order::kC);
  auto host = input.HostView();
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t t = 0; t < targets; ++t) {
      auto v = source[r * targets + t] = FixtureValue(rank, r, t, mode);
      host(r, t) = {v.g, v.h};
      fixed[r * targets + t] = Fixed(v, scales[t]);
    }
  auto device = std::as_const(input).View(DeviceOrd::MPS());
  auto before = metal::GetStatistics();
  auto collective_before = metal::GetCollectiveStatistics();
  auto state = metal::MakeQuantizedGradients(device, &ctx);
  Check(state->Rows() == rows && state->Targets() == targets, "Quantized shape changed");
  Verify(*state, fixed, scales);
  Check(metal::GetCollectiveStatistics().reduce_dispatches == collective_before.reduce_dispatches,
        "Quantizer control bounds unexpectedly use GPU collectives");
  Check(metal::GetStatistics().vector_readback_bytes == before.vector_readback_bytes,
        "Resident original gradients were mirrored to host");

  // Retain global original scales through separately sampled FLOAT value gradients.
  auto quantizer = metal::MakeGradientQuantizer(device, &ctx);
  auto leaves = metal::MakeLeafSums(rows + 1, targets, quantizer);
  quantizer.reset();
  linalg::Matrix<GradientPair> values({rows, targets}, DeviceOrd::MPS());
  auto vh = values.HostView();
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t t = 0; t < targets; ++t) {
      auto v = source[r * targets + t];
      float factor = r % 3 == 0 ? 0.f : r % 3 == 1 ? 1.25f : 1.5f;
      vh(r, t) = {ref::Multiply(v.g, factor), ref::Multiply(v.h, factor)};
    }
  auto vv = std::as_const(values).View(DeviceOrd::MPS());
  for (unsigned page = 0; page < 2; ++page) {
    auto begin = rows * page / 2, end = rows * (page + 1) / 2;
    auto indices = metal::MakeDeviceRows();
    indices->Reset(end - begin, begin);
    std::vector<metal::LeafSumSegment> segments;
    for (auto r = begin; r < end; ++r)
      segments.push_back({r - begin, r - begin + 1, static_cast<std::uint32_t>(r), 0});
    indices->AddLeafGradients(segments, vv, leaves.get());
  }
  auto sums = leaves->Data();
  for (std::size_t r = 0; r <= rows; ++r)
    for (std::size_t t = 0; t < targets; ++t) {
      ref::IntegerPair q{};
      if (r < rows) q = Fixed({vh(r, t).GetGrad(), vh(r, t).GetHess()}, scales[t]);
      Same(sums[r * targets + t].GetGrad(), std::ldexp(double(q.g), -scales[t].g));
      Same(sums[r * targets + t].GetHess(), std::ldexp(double(q.h), -scales[t].h));
    }

  // Exercise the production Sampler caller, including its RNG and GPU ownership.
  for (unsigned sampling = 0; sampling < 2; ++sampling) {
    if (sampling && mode != 0 && mode != 4) continue;  // Finite FP32 norm fixtures.
    linalg::Matrix<GradientPair> sampled({rows, targets}, DeviceOrd::MPS());
    metal::CopyGradients(device, sampled.View(DeviceOrd::MPS()));
    tree::TrainParam param;
    param.subsample = .63f;
    param.sampling_method =
        sampling ? tree::TrainParam::kGradientBased : tree::TrainParam::kUniform;
    ctx.Rng().seed(79);
    auto rng = ctx.Rng();
    auto seed = static_cast<std::uint32_t>(rng());
    tree::cpu_impl::Sampler sampler(param);
    sampler.Sample(&ctx, sampled.View(DeviceOrd::MPS()));
    Check(ctx.Rng() == rng, "Sampler seed consumption differs");
    auto expected = fixed;
    if (sampling) {
      ref::GradientSampling oracle(source, rows, targets, param.subsample, seed, scales);
      Check(!oracle.invalid_rescale, "Unexpected invalid fixture rescale");
      expected = oracle.sampled;
    } else {
      for (std::size_t r = 0; r < rows; ++r)
        if (!ref::Selected(seed, r, param.subsample))
          for (std::size_t t = 0; t < targets; ++t) expected[r * targets + t] = {};
    }
    Verify(*sampler.Quantized(), expected, scales);
  }

  // CUDA stump sums are GPU binary64 collectives, before division. Exact dyadic
  // fixtures avoid pretending a serial sum defines CUDA's parallel association.
  if (mode != 3) {
    linalg::Vector<float> output;
    collective_before = metal::GetCollectiveStatistics();
    tree::FitStump(&ctx, input, targets, &output);
    Check(output.Device().IsMPS(), "Stump output lost Metal ownership");
    auto actual = output.HostView();
    for (std::size_t t = 0; t < targets; ++t) {
      double g = 0, h = 0;
      for (int peer = 0; peer < world; ++peer)
        for (std::size_t r = 0; r < Rows(peer, layout); ++r) {
          auto v = FixtureValue(peer, r, t, mode);
          g += v.g;
          h += v.h;
        }
      auto expected = static_cast<float>(-g / std::max(h, double(1e-6f)));
      if (world == 1 && !rows) expected = 0.f;  // Existing local empty shortcut.
      Same(actual(t), expected);
    }
    if (world > 1) {
      auto after = metal::GetCollectiveStatistics();
      Check(after.reduce_dispatches > collective_before.reduce_dispatches,
            "Missing GPU stump reduction");
      Check(after.staging_bytes == collective_before.staging_bytes, "Stump sums were staged");
    }
  }
  ++cases;
}
template <typename Fn>
void Reject(Fn&& fn, char const* message) {
  try {
    fn();
  } catch (std::exception const& e) {
    Check(std::string(e.what()).find(message) != std::string::npos, "Wrong rejection reason");
    ++rejected;
    return;
  }
  Check(false, "Invalid call succeeded");
}
void Invalid(Context const& ctx, int rank, int world) {
  current = "invalid quantizers";
  linalg::Matrix<GradientPair> input({std::size_t{3}, std::size_t{2}}, DeviceOrd::MPS());
  input.Data()->Fill({2, 1});
  auto view = std::as_const(input).View(DeviceOrd::MPS());
  if (world > 1) {
    Reject([&] { metal::MakeQuantizedGradients(view); }, "requires a Context");
    Reject([&] { metal::MakeGradientQuantizer(view); }, "requires a Context");
    float out[2]{};
    Reject([&] { metal::FitStump(view, linalg::MakeVec(out, 2)); }, "requires a Context");
    float labels[6]{};
    auto label_view =
        linalg::MakeTensorView(DeviceOrd::CPU(), common::Span<float const>{labels, 6}, 3, 2);
    Reject([&] { metal::Mean(metal::MeanKind::kSample, label_view, {}, linalg::MakeVec(out, 2)); },
           "require a Context");
  }
  for (auto bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                   -std::numeric_limits<float>::infinity()}) {
    input.HostView()(1, 1) = rank == world - 1 ? GradientPair{bad, 1} : GradientPair{2, 1};
    view = std::as_const(input).View(DeviceOrd::MPS());
    Reject([&] { metal::MakeQuantizedGradients(view, &ctx); }, "finite gradients");
    Reject([&] { metal::MakeGradientQuantizer(view, &ctx); }, "finite gradients");
  }
}
std::size_t MeanRows(int rank, int world, unsigned layout) {
  std::vector<int> active;
  for (int peer = 0; peer < world; ++peer) {
    if (layout == 3 || (layout == 1 && peer == 0) || ((layout == 2 || layout == 4) && peer != 0))
      continue;
    active.push_back(peer);
  }
  auto it = std::find(active.begin(), active.end(), rank);
  if (it == active.end()) return 0;
  std::size_t total = layout == 4 ? 1 : 1024;
  return total / active.size() + (std::size_t(it - active.begin()) < total % active.size());
}
void Means(Context const& ctx, int rank, int world, unsigned layout, std::size_t targets) {
  current = "means/world" + std::to_string(world) + "/rank" + std::to_string(rank) + "/layout" +
            std::to_string(layout) + "/targets" + std::to_string(targets);
  auto rows = MeanRows(rank, world, layout);
  auto value = [](int peer, std::size_t row, std::size_t target) {
    return std::ldexp(float(int((row + peer * 11) % 17) - 8), int(target % 3) - 2);
  };
  linalg::Matrix<float> input({rows, targets}, DeviceOrd::MPS(),
                              layout % 2 ? linalg::Order::kF : linalg::Order::kC);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t target = 0; target < targets; ++target)
      input.HostView()(row, target) = value(rank, row, target);
  input.View(DeviceOrd::MPS());
  for (unsigned weighted = 0; weighted < 3; ++weighted) {
    std::vector<std::vector<float>> weights(world);
    std::size_t total_rows = 0;
    double sum_weight = 0;
    int last = -1;
    for (int peer = 0; peer < world; ++peer) {
      auto n = MeanRows(peer, world, layout);
      total_rows += n;
      weights[peer].resize(n);
      for (std::size_t row = 0; row < n; ++row) {
        auto w = weighted == 2 && peer == 0 && world > 1 ? 0.f : float(1 + (row % 2) * 2);
        weights[peer][row] = w;
        sum_weight += w;
        if (w) last = peer;
      }
    }
    if (sum_weight) {
      double next_power = 1;
      while (next_power < sum_weight) next_power *= 2;
      weights[last][0] += float(next_power - sum_weight);
      sum_weight = next_power;
    }
    HostDeviceVector<float> local_weights(weights[rank], DeviceOrd::MPS());
    linalg::Vector<float> output;
    auto before = metal::GetStatistics();
    auto coll_before = metal::GetCollectiveStatistics();
    auto invoke = [&] {
      if (weighted)
        common::WeightedSampleMean(&ctx, input, local_weights, &output);
      else
        common::SampleMean(&ctx, input, &output);
    };
    if (weighted && !sum_weight) {
      Reject(invoke, "weights must contain at least one non-zero value");
      continue;
    }
    invoke();
    Check(output.Device().IsMPS(), "Mean output lost Metal ownership");
    auto after = metal::GetStatistics();
    Check(after.vector_readback_bytes == before.vector_readback_bytes,
          "Mean inputs/sums were mirrored");
    Check(after.initialization_host_input_bytes == before.initialization_host_input_bytes,
          "Resident mean input was staged");
    if (world > 1) {
      auto coll_after = metal::GetCollectiveStatistics();
      Check(coll_after.reduce_dispatches > coll_before.reduce_dispatches,
            "Missing FP32 GPU mean reduction");
      Check(coll_after.staging_bytes == coll_before.staging_bytes, "Mean outputs were staged");
    }
    auto out = output.HostView();
    Check(out.Size() == std::max<std::size_t>(targets, 1), "Mean output shape changed");
    for (std::size_t target = 0; target < out.Size(); ++target) {
      float expected = 0;
      if (targets && total_rows)
        for (int peer = 0; peer < world; ++peer) {
          double local = 0;
          for (std::size_t row = 0; row < MeanRows(peer, world, layout); ++row) {
            double term = value(peer, row, target) / (weighted ? sum_weight : double(total_rows));
            if (weighted) term *= weights[peer][row];
            local += term;
          }
          expected += static_cast<float>(local);
        }
      Same(out(target), expected);
    }
    ++cases;
  }
}
int main(int argc, char** argv) {
  try {
    Check(argc == 2, "Communicator JSON required");
    Api(XGCommunicatorInit(argv[1]));
    int rank = XGCommunicatorGetRank(), world = XGCommunicatorGetWorldSize();
    Context ctx;
    ctx.Init(Args{{"device", "mps"}, {"nthread", "2"}});
    for (unsigned mode = 0; mode < 6; ++mode) {
      for (unsigned layout = 0; layout < 5; ++layout)
        for (std::size_t targets : {1, 3, 17}) Run(ctx, rank, world, layout, mode, targets);
      std::printf("PASS world%d rank%d mode%d: %zu cases / %zu fields\n", world, rank, mode, cases,
                  fields);
      std::fflush(stdout);
    }
    for (unsigned layout = 0; layout < 5; ++layout)
      for (std::size_t targets : {0, 1, 3, 17}) Means(ctx, rank, world, layout, targets);
    Invalid(ctx, rank, world);
    Check(metal::GetStatistics().vector_live_buffers == 0, "Leaked owned vectors");
    Api(XGCommunicatorFinalize());
    std::printf(
        "PASS world%d rank%d: %zu global quantizer/initialization cases / %zu fields / %zu "
        "rejected\n",
        world, rank, cases, fields, rejected);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s: %s\n", current.c_str(), e.what());
    return 1;
  }
}
