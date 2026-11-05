// Consume exact exported AFT inputs; evaluate real CUDA source and objective.
#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>

#include "../xgboost/src/common/cuda_context.cuh"
#include "../xgboost/src/common/device_helpers.cuh"
#include "../xgboost/src/common/survival_util.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/objective.h"
#include "aft_contract_io.h"

namespace common = xgboost::common;
namespace aft = aft_contracts;
using aft::Require;

void Cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
bool Same(float a, float b) {
  return aft::Bits<std::uint32_t>(a) == aft::Bits<std::uint32_t>(b) ||
         (std::isnan(a) && std::isnan(b));
}

template <typename Distribution>
std::vector<aft::Observation> Observe(xgboost::Context const* ctx, aft::Input const& input) {
  using namespace xgboost;
  MetaInfo info;
  info.num_row_ = input.lower.size();
  info.labels_lower_bound_.HostVector() = input.lower;
  info.labels_upper_bound_.HostVector() = input.upper;
  info.weights_.HostVector() = input.weights;
  info.labels_lower_bound_.SetDevice(ctx->Device());
  info.labels_upper_bound_.SetDevice(ctx->Device());
  info.weights_.SetDevice(ctx->Device());
  HostDeviceVector<float> predictions(input.prediction, ctx->Device());
  auto lower = info.labels_lower_bound_.ConstDeviceSpan();
  auto upper = info.labels_upper_bound_.ConstDeviceSpan();
  auto weights = info.weights_.ConstDeviceSpan();
  auto pred = predictions.ConstDeviceSpan();
  dh::device_vector<aft::Observation> storage(input.lower.size());
  auto out = dh::ToSpan(storage);
  float scale = input.scale;
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(std::size_t{0}),
                    out.size(), [=] __device__(std::size_t i) {
    using Loss = common::AFTLoss<Distribution>;
    double l = lower[i], u = upper[i], p = pred[i], sigma = scale;
    double g = Loss::Gradient(l, u, p, sigma), h = Loss::Hessian(l, u, p, sigma);
    // Source likelihood before regularization, for sensitivity classification.
    // This is a separate trace expression, not a capture of an API temporary.
    double likelihood = l == u ? Distribution::PDF((log(l)-p)/sigma)/(sigma*l) :
        (isinf(u) ? 1. : Distribution::CDF((log(u)-p)/sigma)) -
        (l <= 0. ? 0. : Distribution::CDF((log(l)-p)/sigma));
    float weight = weights.empty() ? 1.f : weights[i];
    out[i] = {g, h, likelihood, static_cast<float>(g)*weight, static_cast<float>(h)*weight};
  });
  Cuda(cudaDeviceSynchronize());
  std::vector<aft::Observation> observations(out.size());
  Cuda(cudaMemcpy(observations.data(), out.data(), observations.size()*sizeof(aft::Observation),
                  cudaMemcpyDeviceToHost));

  // The 75 ordinary cases exercise the actual objective registry/consumer too.
  // The 21 invalid-scale/domain guard cases remain direct source-function tests.
  if (input.mode < 5) {
    char const* distributions[]{"normal", "logistic", "extreme"};
    std::ostringstream sigma;
    sigma << std::setprecision(std::numeric_limits<float>::max_digits10) << scale;
    std::unique_ptr<ObjFunction> objective{ObjFunction::Create("survival:aft", ctx)};
    objective->Configure({{"aft_loss_distribution", distributions[input.distribution]},
                          {"aft_loss_distribution_scale", sigma.str()}});
    linalg::Matrix<GradientPair> actual;
    objective->GetGradient(predictions, info, 0, &actual);
    Require(actual.Data()->Device().IsCUDA(), "AFT objective output is not CUDA-owned");
    auto host = actual.HostView();
    for (std::size_t i = 0; i < observations.size(); ++i) {
      auto g = host(i, 0).GetGrad(), h = host(i, 0).GetHess();
      Require(Same(g, observations[i].weighted_gradient) && Same(h, observations[i].weighted_hessian),
              "CUDA production objective differs from separately executed CUDA source functions");
      observations[i].weighted_gradient = g;
      observations[i].weighted_hessian = h;
    }
  }
  return observations;
}

int main(int argc, char** argv) {
  try {
    Require(argc == 3, "Usage: cuda_aft_contracts input-corpus.bin new-cuda-observations.bin");
    int devices = 0;
    Cuda(cudaGetDeviceCount(&devices));
    Require(devices > 0, "NVIDIA execution is required; no skipped pass or CPU fallback");
    Cuda(cudaSetDevice(0));
    auto ctx = xgboost::Context{}.MakeCUDA(0);
    Require(ctx.Device().IsCUDA(), "Context is not CUDA");
    cudaDeviceProp gpu{};
    Cuda(cudaGetDeviceProperties(&gpu, 0));
    int driver = 0, runtime = 0;
    Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
    std::string metadata_path = std::string(argv[2])+".json";
    Require(!std::ifstream{metadata_path}.good(), "AFT metadata output already exists");
    aft::InputReader inputs(argv[1]);
    aft::Writer report(argv[2], aft::kCuda);
    std::size_t rows = 0;
    for (std::uint32_t i = 0; i < aft::kCases; ++i) {
      auto input = inputs.Next();
      auto observed = input.distribution == 0 ? Observe<common::NormalDistribution>(&ctx, input) :
                      input.distribution == 1 ? Observe<common::LogisticDistribution>(&ctx, input) :
                                                Observe<common::ExtremeDistribution>(&ctx, input);
      report.Add(input, observed);
      rows += input.lower.size();
      std::printf("AFT CUDA %u/96: distribution=%u mode=%u rows=%zu, cumulative=%zu\n",
                  i+1, input.distribution, input.mode, input.lower.size(), rows);
      std::fflush(stdout);
    }
    inputs.Finish();
    report.Finish();
    using namespace xgboost;
    Json metadata{Object{}};
    metadata["status"] = String("completed");
    metadata["backend"] = String("cuda");
    metadata["device"] = String(gpu.name);
    metadata["compute_capability"] = String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
    metadata["driver"] = Integer(driver); metadata["runtime"] = Integer(runtime);
    metadata["nvcc"] = String(std::to_string(__CUDACC_VER_MAJOR__)+"."+
        std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
    metadata["cases"] = Integer(aft::kCases); metadata["rows"] = Integer(rows);
    metadata["objective_cases"] = Integer(75); metadata["source_guard_cases"] = Integer(21);
    metadata["input_file"] = String(argv[1]); metadata["observation_file"] = String(argv[2]);
    std::string text;
    Json::Dump(metadata, &text);
    std::ofstream meta{metadata_path};
    meta.exceptions(std::ios::failbit | std::ios::badbit);
    meta << text << '\n';
    std::printf("PASS AFT CUDA execution/source checks: 96 cases / %zu rows; cross-backend comparison remains\n", rows);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL AFT CUDA: %s\n", error.what());
    return 1;
  }
}
