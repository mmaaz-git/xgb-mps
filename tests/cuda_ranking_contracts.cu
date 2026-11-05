// Observe unchanged production ranking consumers on all 64 Metal rounding inputs.
// The shared cost-factor race is not repaired here. Each repetition starts fresh.
#include <cuda_runtime.h>
#include <cuda/std/tuple>
#include <cub/version.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/version.h>
#include <cmath>
#include <cstdio>
#include <memory>

#include "../xgboost/src/common/cuda_context.cuh"
#include "../xgboost/src/common/deterministic.cuh"
#include "../xgboost/src/common/device_helpers.cuh"
#include "../xgboost/src/objective/lambdarank_obj.h"
#include "xgboost/json.h"
#include "ranking_rounding_io.h"

using namespace xgboost;
namespace rr = ranking_rounding;
using GradCostNorm = cuda::std::tuple<GradientPair, double, double>;
void Cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T> std::vector<T> ReadDevice(T const* data, std::size_t n) {
  std::vector<T> host(n);
  if (n) Cuda(cudaMemcpy(host.data(), data, n*sizeof(T), cudaMemcpyDeviceToHost));
  return host;
}

rr::Observation Observe(Context const* ctx, rr::Input const& in) {
  auto s = in.shape;
  MetaInfo info;
  info.num_row_ = s.rows;
  info.group_ptr_ = {0, 0, 128, 128, 128+s.largest, 129+s.largest};
  info.labels.Data()->HostVector() = in.labels; info.labels.Reshape(s.rows, 1);
  info.labels.SetDevice(ctx->Device()); info.weights_.HostVector() = in.weights;
  HostDeviceVector<float> predictions(in.predictions, ctx->Device());
  ltr::LambdaRankParam param;
  param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", s.mean ? "mean" : "topk"},
      {"lambdarank_num_pair_per_sample", "8"}, {"lambdarank_unbiased", "true"},
      {"lambdarank_normalization", s.normalized ? "true" : "false"},
      {"lambdarank_score_normalization", s.mode == 3 ? "true" : "false"},
      {"lambdarank_bias_norm", "1.5"}});
  std::shared_ptr<ltr::RankingCache> cache;
  if (s.objective) cache = std::make_shared<ltr::MAPCache>(ctx, info, param);
  else cache = std::make_shared<ltr::RankingCache>(ctx, info, param);
  rr::Require(cache->MaxPositionSize() == s.positions, "CUDA bias shape differs");
  linalg::Vector<double> positive(in.positive.begin(), in.positive.end(), {s.positions}, ctx->Device());
  linalg::Vector<double> negative(in.negative.begin(), in.negative.end(), {s.positions}, ctx->Device());
  auto li_full = linalg::Zeros<double>(ctx, s.rows), lj_full = linalg::Zeros<double>(ctx, s.rows);
  auto li = linalg::Zeros<double>(ctx, s.positions), lj = linalg::Zeros<double>(ctx, s.positions);
  linalg::Matrix<GradientPair> gradients;
  if (s.objective) {
    obj::cuda_impl::LambdaRankGetGradientMAP(ctx, 0xffffffffu, predictions, info,
        std::static_pointer_cast<ltr::MAPCache>(cache), positive.View(ctx->Device()),
        negative.View(ctx->Device()), li_full.View(ctx->Device()), lj_full.View(ctx->Device()), &gradients);
  } else {
    obj::cuda_impl::LambdaRankGetGradientPairwise(ctx, 0xffffffffu, predictions, info,
        cache, positive.View(ctx->Device()), negative.View(ctx->Device()),
        li_full.View(ctx->Device()), lj_full.View(ctx->Device()), &gradients);
  }
  rr::Require(gradients.Data()->Device().IsCUDA(), "Ranking output is not CUDA-owned");

  // Capture actual cached rounding factors and reconstruct per-query diagnostics
  // from the production reduction's retained maxima/sums. Per-query cost factors
  // are a separate source-expression evaluation; the shared winner is read below.
  auto maxima = cache->MaxLambdas<GradCostNorm>(ctx, 5);
  auto rounding = cache->CUDARounding(ctx);
  auto groups = cache->DataGroupPtr(ctx);
  auto weights = info.weights_.ConstDeviceSpan();
  auto pos = positive.View(ctx->Device()), neg = negative.View(ctx->Device());
  double weight_norm = cache->WeightNorm();
  dh::device_vector<rr::Group> trace_storage(5);
  auto trace = dh::ToSpan(trace_storage);
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(std::size_t{0}),
      std::size_t{5}, [=] __device__(std::size_t g) {
        auto n = groups[g+1]-groups[g];
        double minimum = INFINITY;
        for (std::size_t i = 0; i < pos.Size(); ++i) {
          minimum = std::min(minimum, std::abs(pos(i)));
          minimum = std::min(minimum, std::abs(neg(i)));
        }
        auto cost = cuda::std::get<1>(maxima[g]);
        auto sum = cuda::std::get<2>(maxima[g]);
        double norm = !s.normalized ? 1. : s.mean ? 1./8. :
                      sum > 0. ? std::log2(1.+sum)/sum : 1.;
        trace[g] = {static_cast<float>(norm), rounding(g).GetGrad(), rounding(g).GetHess(),
            static_cast<float>(weights[g]*weight_norm),
            common::CreateRoundingFactor(static_cast<float>(n)*(cost/minimum), n)};
      });
  obj::cuda_impl::LambdaRankUpdatePositionBias(ctx, li_full.View(ctx->Device()),
      lj_full.View(ctx->Device()), &positive, &negative, &li, &lj, cache);
  Cuda(cudaDeviceSynchronize());
  rr::Observation out;
  auto global = cache->CUDACostRounding(ctx);
  out.global_cost_rounding = ReadDevice(global.data(), 1)[0]; out.weight_norm = weight_norm;
  out.groups = ReadDevice(trace.data(), trace.size());
  for (auto g : gradients.HostView().Values()) {
    out.gradients.push_back(g.GetGrad()); out.gradients.push_back(g.GetHess());
  }
  if (s.objective) {
    auto map = std::static_pointer_cast<ltr::MAPCache>(cache);
    auto nrel = map->NumRelevant(ctx), acc = map->Acc(ctx);
    auto a = ReadDevice(nrel.data(), nrel.size()), b = ReadDevice(acc.data(), acc.size());
    for (std::size_t i = 0; i < a.size(); ++i) { out.prefix.push_back(a[i]); out.prefix.push_back(b[i]); }
  }
  out.positive = positive.Data()->ConstHostVector(); out.negative = negative.Data()->ConstHostVector();
  return out;
}

int main(int argc, char** argv) {
  try {
    rr::Require(argc == 3, "Usage: cuda_ranking_contracts input-only.bin new-cuda-observations.bin");
    std::string metadata_path = std::string(argv[2])+".json";
    rr::Require(!std::filesystem::exists(metadata_path), "Ranking metadata already exists");
    int devices = 0; Cuda(cudaGetDeviceCount(&devices));
    rr::Require(devices > 0, "Actual NVIDIA GPU required; no skipped pass or CPU fallback");
    Cuda(cudaSetDevice(0)); auto ctx = Context{}.MakeCUDA(0);
    cudaDeviceProp gpu{}; Cuda(cudaGetDeviceProperties(&gpu, 0));
    int driver = 0, runtime = 0;
    Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
    rr::InputReader input(argv[1]); rr::Writer report(argv[2], rr::kCuda);
    std::size_t executions = 0;
    for (std::uint32_t i = 0; i < rr::kCases; ++i) {
      auto in = input.Next(); report.Case(in);
      auto repeats = rr::Repetitions(rr::kCuda, in.shape);
      for (std::uint32_t r = 0; r < repeats; ++r) {
        report.Observe(Observe(&ctx, in)); ++executions;
      }
      std::printf("CUDA ranking %u/64: rows=%u repeats=%u cumulative=%zu\n",
                  i+1, in.shape.rows, repeats, executions); std::fflush(stdout);
    }
    input.Finish(); report.Finish();
    Json meta{Object{}};
    meta["status"] = String("completed-observations; comparison remains");
    meta["suite"] = String("ranking-rounding-v1"); meta["backend"] = String("cuda");
    meta["device"] = String(gpu.name);
    meta["compute_capability"] = String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
    meta["nvcc"] = String(std::to_string(__CUDACC_VER_MAJOR__)+"."+
        std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
    meta["driver"] = Integer(driver); meta["runtime"] = Integer(runtime);
    meta["thrust_version"] = Integer(THRUST_VERSION); meta["cub_version"] = Integer(CUB_VERSION);
    meta["cases"] = Integer(64); meta["executions"] = Integer(executions);
    meta["input_file"] = String(argv[1]); meta["observation_file"] = String(argv[2]);
    std::string text; Json::Dump(meta, &text);
    std::ofstream stream{metadata_path}; stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream << text << '\n';
    std::printf("PASS %zu CUDA ranking executions; race/numerical comparison remains\n", executions);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL CUDA ranking: %s\n", error.what()); return 1;
  }
}
