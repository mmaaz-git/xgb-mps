// Initialization integration: device-authoritative payloads, CPU references,
// CUDA-defined host finalizers, strided copies and transactional validation.
#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/collective/broadcast.h"
#include "../xgboost/src/common/linalg_op.h"
#include "../xgboost/src/common/metal_collective.h"
#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_linalg.h"
#include "../xgboost/src/common/metal_ops.h"
#include "../xgboost/src/common/metal_quantile.h"
#include "../xgboost/src/common/metal_reduction.h"
#include "../xgboost/src/common/metal_vector.h"
#include "../xgboost/src/common/numeric.h"
#include "../xgboost/src/common/stats.h"
#include "../xgboost/src/tree/fit_stump.h"
#include "xgboost/c_api.h"
#include "xgboost/objective.h"

using namespace xgboost;
namespace metal = common::metal;
std::size_t cases = 0, fields = 0, rejected = 0;
std::string current;
std::size_t distributed_cases = 0, mean_fields = 0, expected_errors = 0;
void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(current + ": " + message);
}
void Near(double actual, double expected, double tolerance = 3e-6) {
  ++fields;
  if (actual == expected || (std::isnan(actual) && std::isnan(expected))) return;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance * std::max(1., std::abs(expected))) {
    std::fprintf(stderr, "%s actual %.17g expected %.17g\n", current.c_str(), actual, expected);
    Check(false, "Numerical reference differs");
  }
}
template <typename T>
void Resident(HostDeviceVector<T>* values) {
  values->SetDevice(DeviceOrd::MPS());
  values->DevicePointer();
}
void NoTransfers(metal::Statistics const& before, std::uint64_t host_finalizer_bytes = 0) {
  auto after = metal::GetStatistics();
  Check(after.vector_readback_bytes - before.vector_readback_bytes == host_finalizer_bytes,
        "Unexpected host mirror");
  Check(after.initialization_host_input_bytes == before.initialization_host_input_bytes,
        "Initialization input staged through host");
  Check(after.initialization_host_output_bytes == before.initialization_host_output_bytes,
        "Initialization output staged through host");
  Check(after.objective_staging_bytes == before.objective_staging_bytes,
        "Objective payload staging");
}
template <typename T>
auto Vector(common::Span<T> values, std::size_t count, std::size_t stride = 1) {
  std::size_t shape[]{count}, strides[]{stride};
  return linalg::VectorView<T>{values, shape, strides, DeviceOrd::MPS()};
}
template <typename T>
auto Matrix(common::Span<T> values, std::size_t rows, std::size_t targets, std::size_t stride0,
            std::size_t stride1 = 1) {
  std::size_t shape[]{rows, targets}, strides[]{stride0, stride1};
  return linalg::MatrixView<T>{values, shape, strides, DeviceOrd::MPS()};
}
template <typename Fn>
void Reject(Fn fn) {
  bool threw = false;
  try {
    fn();
  } catch (dmlc::Error const&) {
    threw = true;
  }
  Check(threw, "Invalid input accepted");
  ++rejected;
}
void Statistics(Context const& gpu, std::size_t rows, std::size_t targets) {
  auto cpu = gpu.MakeCPU();
  linalg::Matrix<float> labels({rows, targets}, DeviceOrd::CPU());
  linalg::Matrix<GradientPair> gradients({rows, targets}, DeviceOrd::CPU());
  HostDeviceVector<float> weights(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    weights.HostVector()[i] = .25f + float(i % 5) / 4;
    for (std::size_t j = 0; j < targets; ++j) {
      float y = .5f + float((i + j) % 11) / 4;
      labels.HostView()(i, j) = y;
      gradients.HostView()(i, j) = GradientPair{y, .25f + float(i % 7) / 4};
    }
  }
  for (unsigned op = 0; op < 6; ++op) {
    if (op == 2 && !rows) continue;  // WeightedSampleMean rejects empty weights.
    current = "statistics/" + std::to_string(op) + "/" + std::to_string(rows) + "/" +
              std::to_string(targets);
    auto run = [&](Context const& ctx, linalg::Vector<float>* out) {
      if (op == 0)
        common::Mean(
            &ctx,
            labels.View(ctx.IsMPS() ? DeviceOrd::MPS() : DeviceOrd::CPU()).Slice(linalg::All(), 0),
            out);
      if (op == 1) common::SampleMean(&ctx, labels, out);
      if (op == 2) common::WeightedSampleMean(&ctx, labels, weights, out);
      if (op == 3) common::Median(&ctx, labels, HostDeviceVector<float>{}, out);
      if (op == 4) common::Median(&ctx, labels, weights, out);
      if (op == 5) tree::FitStump(&ctx, gradients, targets, out);
    };
    linalg::Vector<float> expected, actual;
    run(cpu, &expected);
    double mean_reference = 0, magnitude = 0;
    if (op == 0 && rows) {
      for (std::size_t i = 0; i < rows; ++i) {
        float term = labels.HostView()(i, 0) / float(rows);
        mean_reference += term;
        magnitude += std::abs(double(term));
      }
    }
    auto compare = [&] {
      Check(actual.Size() == expected.Size(), "Output shape");
      if (op == 0) {
        // Both backends use FP32 terms. Bound each summation independently;
        // CPU's per-thread serial accumulation has a longer rounding path.
        auto bound = [&](std::size_t steps) {
          double product = steps * std::numeric_limits<float>::epsilon();
          return product / (1 - product) * magnitude;
        };
        Check(std::abs(double(expected.HostView()(0)) - mean_reference) <= bound(rows + 2),
              "CPU mean exceeds accumulation bound");
        Check(std::abs(double(actual.HostView()(0)) - mean_reference) <=
                  bound((rows + 255) / 256 + 8),
              "Metal mean exceeds accumulation bound");
        fields += 2;
      } else {
        for (std::size_t i = 0; i < actual.Size(); ++i)
          Near(actual.HostView()(i), expected.HostView()(i));
      }
    };
    Resident(labels.Data());
    Resident(gradients.Data());
    Resident(&weights);
    auto before = metal::GetStatistics();
    run(gpu, &actual);
    NoTransfers(before);
    Check(actual.Device().IsMPS() && !actual.Data()->HostCanRead(), "Output host state");
    auto pointer = actual.Data()->ConstDevicePointer();
    before = metal::GetStatistics();
    run(gpu, &actual);
    NoTransfers(before);
    // SampleMean and WeightedSampleMean replace the output, just as upstream.
    if (op != 1 && op != 2)
      Check(pointer == actual.Data()->ConstDevicePointer(), "Output allocation replaced");
    compare();
    // A CPU caller can consume and rewrite the result, then return it to Metal.
    run(cpu, &actual);
    Resident(labels.Data());
    Resident(gradients.Data());
    Resident(&weights);
    before = metal::GetStatistics();
    run(gpu, &actual);
    NoTransfers(before);
    compare();
    ++cases;
  }
  Resident(&weights);
  auto before = metal::GetStatistics();
  double sum = common::Reduce(&gpu, weights);
  double optional = common::SumOptionalWeights(
      &gpu, common::MakeOptionalWeights(DeviceOrd::MPS(), weights), rows);
  NoTransfers(before);
  Near(sum, common::Reduce(&cpu, weights), 0);
  Near(optional, sum, 0);  // Exact dyadic test inputs fit FP32.
  ++cases;
}
void Objectives(Context const& gpu) {
  auto cpu = gpu.MakeCPU();
  for (auto name : {"reg:squarederror", "reg:logistic", "binary:logistic", "binary:logitraw",
                    "reg:squaredlogerror", "reg:gamma", "reg:pseudohubererror", "count:poisson",
                    "reg:tweedie", "reg:expectileerror", "reg:quantileerror", "reg:absoluteerror",
                    "multi:softprob", "multi:softmax"}) {
    std::string kind{name};
    bool multi = kind.find("multi:") == 0;
    bool binary = kind.find("logistic") != std::string::npos || kind == "binary:logitraw";
    bool expectile = kind == "reg:expectileerror", absolute = kind == "reg:absoluteerror";
    bool ordered = expectile || kind == "reg:quantileerror";
    for (std::size_t rows : {33ul, 258ul, 1026ul})
      for (unsigned targets : {1u, 3u})
        for (bool weighted : {false, true}) {
          if (multi && targets == 1) continue;
          current = kind + "/" + std::to_string(rows) + "/" + std::to_string(targets) +
                    (weighted ? "/weighted" : "/unweighted");
          Args args{{"num_class", std::to_string(targets)},
                    {"expectile_alpha", targets == 1 ? "[0.5]" : "[0.1,0.5,0.9]"},
                    {"quantile_alpha", targets == 1 ? "[0.5]" : "[0.1,0.5,0.9]"},
                    {"scale_pos_weight", "1.7"}};
          std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, &gpu)},
              ref{ObjFunction::Create(name, &cpu)};
          obj->Configure(args);
          ref->Configure(args);
          MetaInfo info;
          info.num_row_ = rows;
          auto label_targets = multi || ordered ? 1u : targets;
          info.labels.Reshape(rows, label_targets);
          for (std::size_t i = 0; i < info.labels.Size(); ++i)
            info.labels.Data()->HostVector()[i] = multi    ? float(i % targets)
                                                  : binary ? float(i % 2)
                                                           : .5f + float(i % 11) / 4;
          if (weighted) {
            info.weights_.Resize(rows);
            for (std::size_t i = 0; i < rows; ++i)
              info.weights_.HostVector()[i] = .25f + float(i % 5) / 4;
          }
          linalg::Vector<float> expected, actual;
          ref->InitEstimation(info, &expected);
          Resident(info.labels.Data());
          Resident(&info.weights_);
          auto before = metal::GetStatistics();
          obj->InitEstimation(info, &actual);
          // Upstream expectile reads its scalar mean and final target vector;
          // absolute error reads its target means and fitted weights.
          NoTransfers(before, expectile ? (targets + 1) * 4 : absolute ? targets * 8 : 0);
          Check(!info.labels.Data()->HostCanRead() && !info.weights_.HostCanRead(),
                "Initialization materialized metadata");
          if (!expectile && !absolute)
            Check(!actual.Data()->HostCanRead() && actual.Device().IsMPS(),
                  "Base score host state");
          Check(actual.Size() == expected.Size(), "Base score shape");
          for (std::size_t i = 0; i < actual.Size(); ++i)
            Near(actual.HostView()(i), expected.HostView()(i));
          ++cases;
        }
    std::printf("Checked initialization storage %s\n", name);
    std::fflush(stdout);
  }
}

std::size_t WorkerRows(int rank, unsigned layout) {
  return layout == 3 || (layout == 1 && rank == 0) || (layout == 2 && rank != 0) ? 0
                                                                                 : 33 + rank * 19;
}
float WorkerWeight(int rank, std::size_t row, unsigned mode) {
  if (!mode || (mode == 2 && rank == 0)) return 1.f;
  return mode == 3 && rank == 0 ? 0.f : .25f + float(row % 5) / 4;
}
void SameWorkers(Context const& ctx, linalg::Vector<float> const& values) {
  auto local = values.HostView().Values();
  std::vector<float> root(local.data(), local.data() + local.size());
  std::uint64_t size = root.size();
  collective::SafeColl(collective::Broadcast(&ctx, linalg::MakeVec(&size, 1), 0));
  root.resize(size);
  collective::SafeColl(collective::Broadcast(&ctx, linalg::MakeVec(root.data(), size), 0));
  Check(local.size() == root.size(), "Workers produced different base-score shapes");
  Check(!std::memcmp(local.data(), root.data(), local.size_bytes()),
        "Workers produced different base-score bits");
}
void DistributedObjectives(Context const& gpu, int rank, int world) {
  auto cpu = gpu.MakeCPU();
  for (auto name : {"reg:squarederror", "reg:logistic", "binary:logistic", "binary:logitraw",
                    "reg:squaredlogerror", "reg:gamma", "reg:pseudohubererror", "count:poisson",
                    "reg:tweedie", "reg:expectileerror", "reg:quantileerror", "reg:absoluteerror",
                    "multi:softprob", "multi:softmax"}) {
    std::string kind{name};
    bool multi = kind.find("multi:") == 0;
    bool binary = kind.find("logistic") != std::string::npos || kind == "binary:logitraw";
    bool expectile = kind == "reg:expectileerror", absolute = kind == "reg:absoluteerror";
    bool ordered = expectile || kind == "reg:quantileerror";
    for (std::size_t targets : {1ul, 3ul}) {
      if (multi && targets == 1) continue;
      for (unsigned layout = 0; layout < 4; ++layout)
        for (unsigned mode = 0; mode < 4; ++mode)
          for (unsigned newton = 0; newton < (binary ? 2u : 1u); ++newton) {
            current = kind + "/world" + std::to_string(world) + "/rank" + std::to_string(rank) +
                      "/targets" + std::to_string(targets) + "/layout" + std::to_string(layout) +
                      "/weights" + std::to_string(mode) + "/newton" + std::to_string(newton);
            auto rows = WorkerRows(rank, layout);
            auto columns = multi || ordered ? 1ul : targets;
            auto label = [&](int peer, std::size_t row, std::size_t column) {
              return multi    ? float(row % targets)
                     : binary ? float((row + column) % 2)
                              : .5f + float((row + column) % 11) / 4 + peer * 2.f;
            };
            MetaInfo info;
            info.num_row_ = rows;
            info.labels.Reshape(rows, columns);
            for (std::size_t r = 0; r < rows; ++r)
              for (std::size_t t = 0; t < columns; ++t)
                info.labels.HostView()(r, t) = label(rank, r, t);
            if (mode && !(mode == 2 && rank == 0)) {
              info.weights_.Resize(rows);
              for (std::size_t r = 0; r < rows; ++r)
                info.weights_.HostVector()[r] = WorkerWeight(rank, r, mode);
            }
            double sum_weight = 0;
            std::vector<double> numerator(targets, 0);
            for (int peer = 0; peer < world; ++peer)
              for (std::size_t r = 0; r < WorkerRows(peer, layout); ++r) {
                double w = WorkerWeight(peer, r, mode);
                sum_weight += w;
                for (std::size_t t = 0; t < targets; ++t)
                  numerator[t] += double(label(peer, r, t)) * w;
              }
            Args args{{"num_class", std::to_string(targets)},
                      {"expectile_alpha", targets == 1 ? "[0.5]" : "[0.1,0.5,0.9]"},
                      {"quantile_alpha", targets == 1 ? "[0.5]" : "[0.1,0.5,0.9]"},
                      {"scale_pos_weight", newton ? "1.7" : "1"}};
            std::unique_ptr<ObjFunction> reference{ObjFunction::Create(name, &cpu)},
                objective{ObjFunction::Create(name, &gpu)};
            reference->Configure(args);
            objective->Configure(args);
            linalg::Vector<float> expected, actual;
            auto initialize = [&](ObjFunction const& obj, linalg::Vector<float>* out,
                                  std::string* error) {
              try {
                obj.InitEstimation(info, out);
                return true;
              } catch (dmlc::Error const& e) {
                *error = e.what();
                return false;
              }
            };
            std::string cpu_error, gpu_error;
            auto cpu_ok = initialize(*reference, &expected, &cpu_error);
            Resident(info.labels.Data());
            Resident(&info.weights_);
            auto before = metal::GetStatistics();
            auto collective_before = metal::GetCollectiveStatistics();
            auto gpu_ok = initialize(*objective, &actual, &gpu_error);
            if (cpu_ok != gpu_ok)
              std::fprintf(stderr, "CPU: %s\nMetal: %s\n", cpu_error.c_str(), gpu_error.c_str());
            Check(cpu_ok == gpu_ok, "CPU/Metal initialization rejection differs");
            if (!gpu_ok) {
              Check(sum_weight == 0, "Unexpected rejection of positive-weight data");
              ++expected_errors;
              ++distributed_cases;
              continue;
            }
            std::uint64_t finalizer = expectile                    ? (targets + 1) * 4
                                      : absolute && sum_weight > 0 ? targets * 8
                                                                   : 0;
            NoTransfers(before, finalizer);
            Check(!info.labels.Data()->HostCanRead() && !info.weights_.HostCanRead(),
                  "Distributed initialization mirrored metadata");
            auto collective_after = metal::GetCollectiveStatistics();
            Check((collective_after.reduce_dispatches > collective_before.reduce_dispatches) ==
                      (world > 1 && !(absolute && sum_weight == 0)),
                  "Missing GPU initialization collective");
            Check(collective_after.staging_bytes == collective_before.staging_bytes,
                  "Initialization collective staged payloads");
            Check(actual.Size() == targets && expected.Size() == targets, "Base-score shape");
            for (std::size_t t = 0; t < targets; ++t) {
              Near(actual.HostView()(t), expected.HostView()(t));
              if (kind == "reg:squarederror") {
                double mean = sum_weight > 0 ? numerator[t] / sum_weight : 0;
                Near(actual.HostView()(t), mean);
                ++mean_fields;
              }
            }
            SameWorkers(gpu, actual);
            ++distributed_cases;
          }
    }
    std::printf(
        "PASS world%d rank%d %s: %zu initialization cases / %zu "
        "expected errors\n",
        world, rank, name, distributed_cases, expected_errors);
    std::fflush(stdout);
  }
}
void Bindings() {
  for (std::size_t rows : {0ul, 1ul, 33ul, 257ul, 4097ul})
    for (std::size_t stride : {1ul, 3ul}) {
      current = "strided/" + std::to_string(rows) + "/" + std::to_string(stride);
      constexpr std::size_t targets = 3;
      std::vector<float> source(rows * targets * stride + 5, -71);
      std::vector<GradientPair> pairs(source.size(), GradientPair{-71, -91});
      std::vector<double> sums(targets, 0), hess(targets, 0);
      for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < targets; ++j) {
          auto k = 1 + (i * targets + j) * stride;
          source[k] = .5f + float((i + j) % 13) / 4;
          pairs[k] = GradientPair{source[k], .25f + float(i % 7) / 4};
          sums[j] += source[k];
          hess[j] += pairs[k].GetHess();
        }
      HostDeviceVector<float> input(source, DeviceOrd::MPS()),
          out(targets * stride + 5, -91, DeviceOrd::MPS());
      HostDeviceVector<GradientPair> gpairs(pairs, DeviceOrd::MPS());
      HostDeviceVector<GradientPairPrecise> sum_out(
          targets * stride + 5, GradientPairPrecise{-71, -91}, DeviceOrd::MPS());
      auto values =
          Matrix(input.ConstDeviceSpan().subspan(1), rows, targets, targets * stride, stride);
      auto gradients =
          Matrix(gpairs.ConstDeviceSpan().subspan(1), rows, targets, targets * stride, stride);
      auto output = Vector(out.DeviceSpan().subspan(1), targets, stride);
      auto precise = Vector(sum_out.DeviceSpan().subspan(1), targets, stride);
      auto before = metal::GetStatistics();
      metal::Mean(metal::MeanKind::kSample, values, {}, output);
      metal::SumGradients(gradients, precise);
      NoTransfers(before);
      for (std::size_t i = 0; i < out.Size(); ++i) {
        bool active = i && (i - 1) % stride == 0 && (i - 1) / stride < targets;
        auto j = active ? (i - 1) / stride : 0;
        Near(out.ConstHostVector()[i], active ? (rows ? float(sums[j] / rows) : 0) : -91);
        Near(sum_out.ConstHostVector()[i].GetGrad(), active ? sums[j] : -71, 0);
        Near(sum_out.ConstHostVector()[i].GetHess(), active ? hess[j] : -91, 0);
      }
      // Source-defined stump weight, plus untouched padding after the scatter.
      output = Vector(out.DeviceSpan().subspan(1), targets, stride);
      before = metal::GetStatistics();
      metal::FitStump(gradients, output);
      NoTransfers(before);
      for (std::size_t j = 0; j < targets; ++j)
        Near(out.ConstHostVector()[1 + j * stride],
             tree::CalcUnregularizedWeight(sums[j], hess[j]));
      ++cases;
    }

  // Center must snapshot an aliased mean before any output thread can overwrite it.
  for (std::size_t stride : {1ul, 3ul})
    for (bool alias : {false, true}) {
      current = "center/" + std::to_string(stride) + (alias ? "/alias" : "");
      std::vector<float> source(257 * stride + 5, -91);
      for (std::size_t i = 0; i < 257; ++i) source[1 + i * stride] = float(i) / 4 - 3;
      HostDeviceVector<float> values(source, DeviceOrd::MPS()), scalar(1, .25f, DeviceOrd::MPS());
      auto output = Vector(values.DeviceSpan().subspan(1), 257, stride);
      auto mean = alias ? Vector(values.ConstDeviceSpan().subspan(1), 1)
                        : Vector(scalar.ConstDeviceSpan(), 1);
      auto before = metal::GetStatistics();
      metal::Center(output, mean);
      NoTransfers(before);
      for (std::size_t i = 0; i < source.size(); ++i) {
        bool active = i && (i - 1) % stride == 0 && (i - 1) / stride < 257;
        Near(values.ConstHostVector()[i], source[i] - (active ? (alias ? -3.f : .25f) : 0), 0);
      }
      ++cases;
    }
  // Invalid reductions must leave the original device output intact.
  current = "validation";
  HostDeviceVector<float> labels(std::vector<float>{0, 1, 2, 3}, DeviceOrd::MPS()),
      weights(4, 0.f, DeviceOrd::MPS()), output(11, -91.f, DeviceOrd::MPS());
  auto matrix = Matrix(labels.ConstDeviceSpan(), 4, 1, 1);
  auto before = metal::GetStatistics();
  Reject([&] {
    metal::Mean(metal::MeanKind::kWeightedSample, matrix, weights.ConstDeviceSpan(),
                Vector(output.DeviceSpan(), 1));
  });
  Reject([&] {
    metal::SmallHistogram(matrix, common::OptionalWeights{1.f}, Vector(output.DeviceSpan(), 2, 3));
  });
  labels.Fill(NAN);
  Reject([&] {
    metal::SmallHistogram(matrix, common::OptionalWeights{1.f}, Vector(output.DeviceSpan(), 4, 2));
  });
  NoTransfers(before);
  for (float value : output.ConstHostVector()) Near(value, -91, 0);
  ++cases;
}
void Fallbacks() {
  current = "readonly and offset inputs";
  std::size_t rows = 257, page = getpagesize();
  auto mapping = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  Check(mapping != MAP_FAILED, "mmap");
  auto input = static_cast<float*>(mapping) + 1;
  for (std::size_t i = 0; i < rows; ++i) input[i] = float(i);
  Check(mprotect(mapping, page, PROT_READ) == 0, "mprotect");
  HostDeviceVector<float> output(3, -91.f, DeviceOrd::MPS());
  std::vector<metal::QuantileRequest> requests{{0, 0, .5}};
  auto before = metal::GetStatistics();
  metal::Quantiles(Matrix(common::Span<float const>{input, rows}, rows, 1, 1), {}, requests,
                   output.DeviceSpan().subspan(1, 1));
  auto after = metal::GetStatistics();
  Check(after.initialization_host_input_bytes - before.initialization_host_input_bytes == rows * 4,
        "Read-only input fallback not counted");
  Check(after.initialization_host_output_bytes == before.initialization_host_output_bytes,
        "Resident output staged");
  Near(output.ConstHostVector()[1], 128, 0);
  Near(output.ConstHostVector()[0], -91, 0);
  Near(output.ConstHostVector()[2], -91, 0);
  Check(munmap(mapping, page) == 0, "munmap");
  ++cases;

  // C++ GradientPair alignment is four, while the reduction's Metal float2
  // input needs eight. The input fallback must preserve the exact dyadic sum.
  current = "four-byte gradient input";
  std::vector<float> raw(2 * rows + 2, -91);
  auto pairs = reinterpret_cast<GradientPair*>(raw.data() + 1);
  Check(reinterpret_cast<std::uintptr_t>(pairs) % 8 == 4, "Fixture alignment");
  for (std::size_t i = 0; i < rows; ++i) pairs[i] = GradientPair{float(i), .5f};
  HostDeviceVector<GradientPairPrecise> sum(1, GradientPairPrecise{}, DeviceOrd::MPS());
  before = metal::GetStatistics();
  metal::SumGradients(Matrix(common::Span<GradientPair const>{pairs, rows}, rows, 1, 1),
                      Vector(sum.DeviceSpan(), 1));
  after = metal::GetStatistics();
  Check(after.initialization_host_input_bytes - before.initialization_host_input_bytes == rows * 8,
        "Four-byte input fallback not counted");
  Check(after.initialization_host_output_bytes == before.initialization_host_output_bytes,
        "Sum output staged");
  Near(sum.ConstHostVector()[0].GetGrad(), rows * (rows - 1) / 2, 0);
  Near(sum.ConstHostVector()[0].GetHess(), rows * .5, 0);
  Near(raw.front(), -91, 0);
  Near(raw.back(), -91, 0);
  ++cases;

  // Mean and quantile outputs can alias input storage. Publication occurs only
  // after the source reduction/sort has completed; the other elements survive.
  for (bool quantile : {false, true}) {
    current = quantile ? "aliased quantile" : "aliased mean";
    std::vector<float> source(rows + 2, -91);
    for (std::size_t i = 0; i < rows; ++i) source[i + 1] = float(i);
    HostDeviceVector<float> values(source, DeviceOrd::MPS());
    auto view = Matrix(values.ConstDeviceSpan().subspan(1, rows), rows, 1, 1);
    auto out = values.DeviceSpan().subspan(1, 1);
    before = metal::GetStatistics();
    if (quantile)
      metal::Quantiles(view, {}, requests, out);
    else
      metal::Mean(metal::MeanKind::kSample, view, {}, Vector(out, 1));
    NoTransfers(before);
    source[1] = 128;
    for (std::size_t i = 0; i < source.size(); ++i) Near(values.ConstHostVector()[i], source[i], 0);
    ++cases;
  }

  current = "center exceptional FP32";
  std::vector<float> source{0.f,      -0.f,      0x1p-149f, -0x1p-149f, 0x1p-126f, -0x1p-126f,
                            INFINITY, -INFINITY, NAN,       1.f,        -1.f};
  for (float shift : {0.f, -0.f, 0x1p-149f, -0x1p-149f, INFINITY, -INFINITY, NAN}) {
    HostDeviceVector<float> values(source, DeviceOrd::MPS()), scalar(1, shift, DeviceOrd::MPS());
    before = metal::GetStatistics();
    metal::Center(Vector(values.DeviceSpan(), source.size()), Vector(scalar.ConstDeviceSpan(), 1));
    NoTransfers(before);
    for (std::size_t i = 0; i < source.size(); ++i) {
      float expected = source[i] - shift, actual = values.ConstHostVector()[i];
      Check((std::isnan(actual) && std::isnan(expected)) || !std::memcmp(&actual, &expected, 4),
            "Center changed FP32 subtraction bits");
      ++fields;
    }
    ++cases;
  }
}
int main(int argc, char** argv) {
  try {
    Check(argc == 1 || argc == 2, "Expected optional communicator JSON");
    Check(!XGCommunicatorInit(argc == 2 ? argv[1] : "{}"), "Communicator initialization failed");
    int rank = XGCommunicatorGetRank(), world = XGCommunicatorGetWorldSize();
    Context gpu;
    gpu.UpdateAllowUnknown(Args{{"device", "mps"}, {"nthread", "2"}});
    if (world == 1) {
      for (std::size_t rows : {0ul, 1ul, 33ul, 257ul, 4097ul})
        for (std::size_t targets : {1ul, 3ul}) Statistics(gpu, rows, targets);
      Objectives(gpu);
      Bindings();
      Fallbacks();
    }
    DistributedObjectives(gpu, rank, world);
    auto stats = metal::GetStatistics();
    Check(!stats.vector_live_buffers && !stats.vector_capacity_bytes, "Live vector leak");
    Check(!XGCommunicatorFinalize(), "Communicator finalization failed");
    std::printf("PASS %zu initialization storage cases / %zu field checks / %zu rejected calls\n",
                cases, fields, rejected);
    std::printf(
        "PASS world%d rank%d: %zu initialization cases / %zu direct "
        "mean checks / "
        "%zu expected errors\n",
        world, rank, distributed_cases, mean_fields, expected_errors);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
