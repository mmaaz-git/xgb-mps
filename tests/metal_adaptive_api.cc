// Actual objective calls, exact-integer global statistics, and CPU comparisons.
// Communicator tests exercise objective aggregation, not distributed learners.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../results/cuda_objective_reference_generated.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "xgboost/c_api.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"

using namespace xgboost;
namespace metal = common::metal;
struct Math {
  static float Hypot(float a, float b) { return float(std::hypot(double(a), double(b))); }
  static float Tanh(float x) { return float(std::tanh(double(x))); }
  static float FMax(float a, float b) { return std::fmax(a, b); }
};
using Ref = objective_reference::Ref<Math>;
std::size_t cases = 0, fields = 0, global_witnesses = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Api(int result) {
  if (result != 0) throw std::runtime_error(XGBGetLastError());
}
void Near(float actual, float expected) {
  ++fields;
  if (actual == expected || (std::isnan(actual) && std::isnan(expected))) return;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > 3e-6f * std::max(1.f, std::abs(expected))) {
    std::fprintf(stderr, "case %zu actual %.9g expected %.9g\n", cases, actual, expected);
    throw std::runtime_error("Objective differs from independent global/CPU reference");
  }
}
std::size_t Rows(int rank, unsigned layout) {
  if (layout == 2 || (layout == 1 && rank == 0)) return 0;
  return layout == 3 ? 65537 : 257 + rank * 128;
}
float Root(int rank, std::size_t row, unsigned target) {
  return float(1 + rank * 5 + row % 4 + target * 2);
}
float Weight(std::size_t row, unsigned mode) {
  return mode == 0 ? 1.f : mode == 1 ? float(row % 5) : mode == 2 ? 0.f : 0x1p-40f;
}
void Run(int rank, int world, unsigned kind, unsigned targets, unsigned mode, unsigned layout) {
  auto rows = Rows(rank, layout);
  auto label_targets = kind ? 1 : targets;
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "1"}});
  auto cpu_ctx = ctx.MakeCPU();
  MetaInfo info;
  info.num_row_ = rows;
  info.labels.Reshape(rows, label_targets);
  HostDeviceVector<float> predictions(rows * targets);
  if (mode) info.weights_.Resize(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    if (mode) info.weights_.HostVector()[i] = Weight(i, mode);
    for (unsigned j = 0; j < label_targets; ++j) info.labels.HostView()(i, j) = float(i % 7);
    for (unsigned j = 0; j < targets; ++j) {
      auto root = Root(rank, i, j);
      predictions.HostVector()[i * targets + j] = float(i % 7) + (i % 2 ? -1 : 1) * root * root;
    }
  }
  // Integer roots/weights and power-of-two tiny weights make every local sum
  // exact in FP32. The reference does not reproduce the GPU reduction tree.
  std::vector<double> sums(targets, 0.), local(targets, 0.);
  double weight = 0., local_weight = 0.;
  for (int worker = 0; worker < world; ++worker)
    for (std::size_t i = 0; i < Rows(worker, layout); ++i) {
      auto w = Weight(i, mode);
      weight += w;
      if (worker == rank) local_weight += w;
      for (unsigned j = 0; j < targets; ++j) {
        sums[j] += w * Root(worker, i, j);
        if (worker == rank) local[j] += w * Root(worker, i, j);
      }
    }
  std::vector<float> scales(targets), local_scales(targets);
  for (unsigned j = 0; j < targets; ++j) {
    scales[j] = weight < 1e-6 ? 0.f : float((sums[j] / weight) * (sums[j] / weight));
    local_scales[j] =
        local_weight < 1e-6 ? 0.f : float((local[j] / local_weight) * (local[j] / local_weight));
  }
  Args args{{"quantile_alpha", targets == 1 ? "[0.5]" : "[0.1, 0.5, 0.9]"}};
  char const* name = kind ? "reg:quantileerror" : "reg:absoluteerror";
  std::unique_ptr<ObjFunction> objective{ObjFunction::Create(name, &ctx)};
  std::unique_ptr<ObjFunction> cpu_objective{ObjFunction::Create(name, &cpu_ctx)};
  objective->Configure(args);
  cpu_objective->Configure(args);
  linalg::Matrix<GradientPair> output, cpu_output;
  auto before = metal::GetStatistics();
  objective->GetGradient(predictions, info, 0, &output);
  auto after = metal::GetStatistics();
  Check(after.residual_scale_dispatches - before.residual_scale_dispatches == bool(rows) &&
            after.objective_dispatches - before.objective_dispatches == bool(rows),
        "Missing GPU work or empty worker dispatched");
  Check(output.Shape(0) == rows && output.Shape(1) == targets, "Output shape");
  cpu_objective->GetGradient(predictions, info, 0, &cpu_output);
  for (std::size_t i = 0; i < rows; ++i)
    for (unsigned j = 0; j < targets; ++j) {
      float alpha = targets == 1 ? .5f : j == 0 ? .1f : j == 1 ? .5f : .9f;
      float p = predictions.HostVector()[i * targets + j], y = float(i % 7), w = Weight(i, mode);
      auto expected = Ref::Adaptive(kind, p, y, scales[j], w, alpha);
      auto local_expected = Ref::Adaptive(kind, p, y, local_scales[j], w, alpha);
      auto actual = output.HostView()(i, j), cpu = cpu_output.HostView()(i, j);
      Near(actual.GetGrad(), expected.GetGrad());
      Near(actual.GetHess(), expected.GetHess());
      Near(actual.GetGrad(), cpu.GetGrad());
      Near(actual.GetHess(), cpu.GetHess());
      global_witnesses += std::abs(expected.GetGrad() - local_expected.GetGrad()) > 1e-3f ||
                          std::abs(expected.GetHess() - local_expected.GetHess()) > 1e-3f;
    }
  ++cases;
}
// Two terms make the FP32-vs-binary64 distinction independent of association.
void Precision() {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "1"}});
  for (unsigned kind : {0u, 1u})
    for (unsigned mode : {0u, 1u}) {
      MetaInfo info;
      info.num_row_ = 2;
      info.labels.Reshape(2, 1);
      HostDeviceVector<float> predictions(std::vector<float>{1.f, 1.f});
      float a = mode ? std::numeric_limits<float>::max() : 1e8f, b = mode ? a : 1.f;
      info.weights_.HostVector() = {a, b};
      std::unique_ptr<ObjFunction> objective{
          ObjFunction::Create(kind ? "reg:quantileerror" : "reg:absoluteerror", &ctx)};
      objective->Configure({{"quantile_alpha", "[0.1]"}});
      linalg::Matrix<GradientPair> out;
      objective->GetGradient(predictions, info, 0, &out);
      double root_sum = float(a + b), weight_sum = double(a) + double(b);
      float scale = float((root_sum / weight_sum) * (root_sum / weight_sum));
      for (unsigned i = 0; i < 2; ++i) {
        auto expected = Ref::Adaptive(kind, 1.f, 0.f, scale, i ? b : a, .1f);
        Near(out.HostView()(i, 0).GetGrad(), expected.GetGrad());
        Near(out.HostView()(i, 0).GetHess(), expected.GetHess());
      }
      ++cases;
    }
}
int main(int argc, char** argv) {
  try {
    Check(argc == 2, "Pass communicator JSON");
    Api(XGCommunicatorInit(argv[1]));
    int rank = XGCommunicatorGetRank(), world = XGCommunicatorGetWorldSize();
    for (unsigned kind : {0u, 1u})
      for (unsigned targets : {1u, 3u})
        for (unsigned mode = 0; mode < 4; ++mode)
          for (unsigned layout = 0; layout < (world == 1 ? 4u : 3u); ++layout)
            Run(rank, world, kind, targets, mode, layout);
    if (world == 1) Precision();
    Check(world == 1 || global_witnesses > 0, "Missing global/local aggregation witnesses");
    Api(XGCommunicatorFinalize());
    std::printf("PASS world%d rank%d: %zu objective cases / %zu fields / %zu global witnesses\n",
                world, rank, cases, fields, global_witnesses);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
