// Output widths around the former 16-bit limit. Small, exactly representable
// inputs isolate indexing/capacity from unrelated reduction-order differences.
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_gradient.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_leaf.h"
#include "../xgboost/src/common/metal_quantile.h"
#include "../xgboost/src/common/metal_reduction.h"
#include "../xgboost/src/common/metal_training.h"
#include "../xgboost/src/common/stats.h"
#include "../xgboost/src/tree/fit_stump.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t fields = 0;
void Check(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
void Same(double actual, double expected) {
  ++fields;
  if (actual != expected) {
    std::fprintf(stderr, "actual=%a expected=%a field=%zu\n", actual, expected, fields);
    throw std::runtime_error("Capacity result differs from exact reference");
  }
}
float Base(std::size_t target) { return static_cast<float>(int(target % 17) - 8); }
constexpr std::array<float, 4> kOffsets{6, 0, 4, 2};

void NoInitializationTransfers(metal::Statistics const& before) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes == before.vector_readback_bytes, "Unexpected host mirror");
  Check(after.initialization_host_input_bytes == before.initialization_host_input_bytes,
        "Resident input was staged");
  Check(after.initialization_host_output_bytes == before.initialization_host_output_bytes,
        "Resident output was staged");
}

void Statistics(Context const& gpu, std::size_t targets, bool quantiles) {
  auto cpu = gpu.MakeCPU();
  linalg::Matrix<float> values({std::size_t{4}, targets}, DeviceOrd::CPU());
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t t = 0; t < targets; ++t) values.HostView()(r, t) = Base(t) + kOffsets[r];
  HostDeviceVector<float> weights{4, 1, 2, 1}, empty_weights;
  for (bool weighted : {false, true}) {
    linalg::Vector<float> expected, actual;
    auto run = [&](Context const& ctx, linalg::Vector<float>* out) {
      if (quantiles) {
        common::Median(&ctx, values, weighted ? weights : empty_weights, out);
      } else if (weighted) {
        common::WeightedSampleMean(&ctx, values, weights, out);
      } else {
        common::SampleMean(&ctx, values, out);
      }
    };
    run(cpu, &expected);
    values.View(DeviceOrd::MPS());
    weights.SetDevice(DeviceOrd::MPS());
    weights.ConstDevicePointer();
    auto before = metal::GetStatistics();
    run(gpu, &actual);
    NoInitializationTransfers(before);
    auto after = metal::GetStatistics();
    Check(quantiles ? after.quantile_dispatches == before.quantile_dispatches + 1
                    : after.mean_dispatches == before.mean_dispatches + 1,
          "Missing completed GPU operation");
    Check(actual.Device().IsMPS() && actual.Size() == targets, "Output ownership/shape changed");
    for (std::size_t t = 0; t < targets; ++t) {
      Same(actual.HostView()(t), expected.HostView()(t));
      Same(actual.HostView()(t), Base(t) + (weighted ? (quantiles ? 4 : 4.25f) : 3));
    }
  }
}

void Strided(std::size_t targets) {
  auto stride = targets * 2 + 3;
  HostDeviceVector<float> storage(4 * stride, -999), weights{4, 1, 2, 1};
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t t = 0; t < targets; ++t)
      storage.HostVector()[r * stride + 2 * t] = Base(t) + kOffsets[r];
  auto original = storage.HostVector();
  storage.SetDevice(DeviceOrd::MPS());
  weights.SetDevice(DeviceOrd::MPS());
  linalg::MatrixView<float const> view{storage.ConstDeviceSpan(),
                                     {std::size_t{4}, targets},
                                     {stride, std::size_t{2}}, DeviceOrd::MPS()};
  auto weight_view = weights.ConstDeviceSpan();
  for (auto kind : {metal::MeanKind::kFloat32, metal::MeanKind::kSample,
                    metal::MeanKind::kWeightedSample}) {
    HostDeviceVector<float> output(3 * targets, -777, DeviceOrd::MPS());
    linalg::VectorView<float> out{output.DeviceSpan(), {targets}, {std::size_t{3}},
                                  DeviceOrd::MPS()};
    auto before = metal::GetStatistics();
    metal::Mean(kind, view, kind == metal::MeanKind::kWeightedSample ? weight_view
                                                                  : common::Span<float const>{}, out);
    NoInitializationTransfers(before);
    Check(metal::GetStatistics().mean_dispatches == before.mean_dispatches + 1,
          "Strided mean did not execute on GPU");
    for (std::size_t t = 0; t < targets; ++t) {
      Same(output.ConstHostVector()[3 * t],
           Base(t) + (kind == metal::MeanKind::kWeightedSample ? 4.25f : 3));
      Same(output.ConstHostVector()[3 * t + 1], -777);
      Same(output.ConstHostVector()[3 * t + 2], -777);
    }
  }
  std::vector<metal::QuantileRequest> requests;
  std::array<double, 5> alpha{0, .25, .5, .75, 1};
  for (std::size_t t = 0; t < targets; ++t)
    for (auto a : alpha) requests.push_back({static_cast<std::uint32_t>(t), 0, a});
  for (bool weighted : {false, true}) {
    HostDeviceVector<float> output(requests.size(), -777, DeviceOrd::MPS());
    auto out = output.DeviceSpan();
    auto before = metal::GetStatistics();
    metal::Quantiles(view, weighted ? weight_view : common::Span<float const>{}, requests, out);
    NoInitializationTransfers(before);
    Check(metal::GetStatistics().quantile_dispatches == before.quantile_dispatches + 1,
          "Strided quantiles did not execute on GPU");
    std::array<float, 5> offsets = weighted ? std::array<float, 5>{0, 2, 4, 6, 6}
                                           : std::array<float, 5>{0, .5f, 3, 5.5f, 6};
    for (std::size_t t = 0; t < targets; ++t)
      for (std::size_t a = 0; a < alpha.size(); ++a)
        Same(output.ConstHostVector()[5 * t + a], Base(t) + offsets[a]);
  }
  Check(storage.ConstHostVector() == original, "Strided input or its guards were modified");
}

linalg::Matrix<GradientPair> Gradients(std::size_t targets) {
  linalg::Matrix<GradientPair> values({std::size_t{4}, targets}, DeviceOrd::CPU());
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t t = 0; t < targets; ++t)
      values.HostView()(r, t) = {Base(t) + kOffsets[r], 2};
  return values;
}

void Stump(Context const& gpu, std::size_t targets) {
  auto cpu = gpu.MakeCPU();
  auto values = Gradients(targets);
  linalg::Vector<float> expected, actual;
  tree::FitStump(&cpu, values, targets, &expected);
  values.View(DeviceOrd::MPS());
  auto before = metal::GetStatistics();
  tree::FitStump(&gpu, values, targets, &actual);
  NoInitializationTransfers(before);
  Check(metal::GetStatistics().stump_dispatches == before.stump_dispatches + 1,
        "Stump did not execute on GPU");
  Check(actual.Device().IsMPS(), "Stump output lost GPU ownership");
  for (std::size_t t = 0; t < targets; ++t) {
    Same(actual.HostView()(t), expected.HostView()(t));
    Same(actual.HostView()(t), -(Base(t) / 2 + 1.5f));
  }
  linalg::Vector<GradientPairPrecise> sum({targets}, DeviceOrd::MPS());
  before = metal::GetStatistics();
  metal::SumGradients(std::as_const(values).View(DeviceOrd::MPS()), sum.View(DeviceOrd::MPS()));
  NoInitializationTransfers(before);
  Check(metal::GetStatistics().gradient_sum_dispatches == before.gradient_sum_dispatches + 1,
        "Gradient sum did not execute on GPU");
  for (std::size_t t = 0; t < targets; ++t) {
    Same(sum.HostView()(t).GetGrad(), 4 * Base(t) + 12);
    Same(sum.HostView()(t).GetHess(), 8);
  }
}

void Leaves(std::size_t targets, bool quantized) {
  auto values = Gradients(targets);
  auto view = std::as_const(values).View(DeviceOrd::MPS());
  auto before = metal::GetStatistics();
  auto quantizer = quantized ? metal::MakeGradientQuantizer(view) : nullptr;
  auto sums = metal::MakeLeafSums(3, targets, quantizer);
  auto rows = metal::MakeDeviceRows();
  rows->Reset(4, 0);
  std::vector<metal::LeafSumSegment> segments{{0, 1, 2, 0}, {1, 4, 0, 0}, {4, 4, 1, 0}};
  // Two accumulations exercise the persistent output, including an empty leaf.
  rows->AddLeafGradients(segments, view, sums.get());
  rows->AddLeafGradients(segments, view, sums.get());
  auto after = metal::GetStatistics();
  Check(after.leaf_sum_init_dispatches == before.leaf_sum_init_dispatches + 1,
        "Leaf storage was not initialized on GPU");
  Check(quantized ? after.leaf_integer_sum_dispatches == before.leaf_integer_sum_dispatches + 2
                  : after.leaf_sum_dispatches == before.leaf_sum_dispatches + 2,
        "Leaf sums did not execute on GPU");
  auto data = sums->Data();
  Check(data.size() == 3 * targets, "Leaf sum shape changed");
  for (std::size_t t = 0; t < targets; ++t) {
    Same(data[t].GetGrad(), 6 * Base(t) + 12);
    Same(data[t].GetHess(), 12);
    Same(data[targets + t].GetGrad(), 0);
    Same(data[targets + t].GetHess(), 0);
    Same(data[2 * targets + t].GetGrad(), 2 * Base(t) + 12);
    Same(data[2 * targets + t].GetHess(), 4);
  }
}

void QuantizedRoots(std::size_t targets) {
  auto values = Gradients(targets);
  auto view = std::as_const(values).View(DeviceOrd::MPS());
  auto before = metal::GetStatistics();
  auto state = metal::MakeQuantizedGradients(view);
  Check(state->Targets() == targets && state->Rows() == 4, "Integer gradient shape changed");
  auto after = metal::GetStatistics();
  Check(after.gradient_bound_dispatches == before.gradient_bound_dispatches + targets &&
            after.quantization_dispatches == before.quantization_dispatches + targets,
        "Missing GPU gradient quantization");
  linalg::Vector<GradientPairPrecise> sums({targets}, DeviceOrd::MPS());
  state->Sum(sums.View(DeviceOrd::MPS()));
  for (std::size_t t = 0; t < targets; ++t) {
    Same(sums.HostView()(t).GetGrad(), 4 * Base(t) + 12);
    Same(sums.HostView()(t).GetHess(), 8);
  }
}

int main(int argc, char** argv) {
  try {
    Context gpu;
    gpu.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    // Optional operation/width selectors also let the pre-fix library reproduce
    // each guard independently, without rebuilding it or stopping at the first.
    std::string operation = argc > 1 ? argv[1] : "initialization";
    std::vector<std::size_t> widths{65535, 65536, 65537};
    if (argc > 2) widths = {std::stoul(argv[2])};
    for (auto targets : widths) {
      std::printf("START %s targets=%zu\n", operation.c_str(), targets);
      std::fflush(stdout);
      if (operation == "initialization" || operation == "mean") Statistics(gpu, targets, false);
      if (operation == "initialization" || operation == "quantile") Statistics(gpu, targets, true);
      if (operation == "initialization" || operation == "stump") Stump(gpu, targets);
      if (operation == "initialization" || operation == "leaf") Leaves(targets, false);
      if (operation == "initialization" || operation == "strided") Strided(targets);
      if (operation == "quantizer") {
        Leaves(targets, true);
        QuantizedRoots(targets);
      }
      std::printf("PASS %s targets=%zu fields=%zu\n", operation.c_str(), targets, fields);
      std::fflush(stdout);
    }
    Check(fields > 0, "No capacity operation selected");
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
