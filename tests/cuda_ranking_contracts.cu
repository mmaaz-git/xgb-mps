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
#include "ranking_objective_report.h"
#include "ranking_primitive_io.h"

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

void RunObjectives(char const* input_path, char const* output_path) {
  namespace ro = ranking_objective;
  std::string metadata_path = std::string(output_path)+".json";
  rr::Require(!std::filesystem::exists(metadata_path), "Objective metadata already exists");
  int devices = 0; Cuda(cudaGetDeviceCount(&devices));
  rr::Require(devices > 0, "Actual NVIDIA GPU required; no skipped pass or CPU fallback");
  Cuda(cudaSetDevice(0)); auto ctx = Context{}.MakeCUDA(0);
  ro::InputReader input(input_path); ro::Writer report(output_path, 2, input.Suite());
  std::unique_ptr<MetaInfo> info;
  std::unique_ptr<ObjFunction> objective;
  std::size_t rows = 0, sequences = 0;
  for (std::size_t index = 0; index < input.Size(); ++index) {
    auto in = input.Next(); auto const& s = in.spec;
    if (s.iteration == 0) {
      // Preserve the original lifecycle: one objective/cache/MetaInfo per
      // sequence, reused for every later iteration without importing bias.
      objective.reset(); info = std::make_unique<MetaInfo>();
      ctx.Init({{"device", "cuda:0"}, {"nthread", s.batched ? "8" : "2"}});
      info->num_row_ = s.rows; info->group_ptr_ = s.group_ptr;
      info->labels.Data()->HostVector() = in.labels; info->labels.Reshape(s.rows, 1);
      info->labels.SetDevice(ctx.Device()); info->weights_.HostVector() = in.weights;
      objective.reset(ObjFunction::Create(ro::Objective(s.objective), &ctx));
      objective->Configure(ro::Arguments(s)); ++sequences;
    }
    rr::Require(info && objective, "Objective state missing for continued sequence");
    HostDeviceVector<float> predictions(in.predictions, ctx.Device());
    ctx.Rng().seed(s.seed);
    linalg::Matrix<GradientPair> gradients;
    objective->GetGradient(predictions, *info, s.iteration, &gradients);
    rr::Require(gradients.Data()->Device().IsCUDA() && gradients.Data()->DeviceCanWrite() &&
                !gradients.Data()->HostCanRead(), "Objective gradients are not resident CUDA outputs");
    Cuda(cudaDeviceSynchronize());
    auto observed = ro::Observe(&ctx, objective.get(), &gradients, s);
    RandomEngine rng(s.seed);
    if (s.mean) static_cast<void>(rng());
    rr::Require(observed.next_rng == static_cast<std::uint32_t>(rng()),
                "Public objective consumed an unexpected number of parent RNG draws");
    report.Add(in, observed); rows += s.rows;
    if ((index+1)%60 == 0 || s.batched || index+1 == input.Size()) {
      std::printf("CUDA objective %zu/%zu: sequence=%u iteration=%u rows=%u cumulative=%zu\n",
                  index+1, input.Size(), s.sequence, s.iteration, s.rows, rows); std::fflush(stdout);
    }
  }
  input.Finish(); report.Finish();
  cudaDeviceProp gpu{}; Cuda(cudaGetDeviceProperties(&gpu, 0));
  int driver = 0, runtime = 0;
  Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};
  meta["status"] = String("completed-observations; comparison remains");
  meta["suite"] = String("ranking-objective-v1"); meta["corpus"] = String(input.Suite() ? "unbiased" : "gradient");
  meta["backend"] = String("cuda"); meta["device"] = String(gpu.name);
  meta["compute_capability"] = String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"] = String(std::to_string(__CUDACC_VER_MAJOR__)+"."+
      std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"] = Integer(driver); meta["runtime"] = Integer(runtime);
  meta["thrust_version"] = Integer(THRUST_VERSION); meta["cub_version"] = Integer(CUB_VERSION);
  meta["calls"] = Integer(input.Size()); meta["sequences"] = Integer(sequences); meta["rows"] = Integer(rows);
  meta["input_file"] = String(input_path); meta["observation_file"] = String(output_path);
  meta["bias_initialization"] = String("Production iter=0 initialization; subsequent state stays in the objective; no Metal bias inputs");
  std::string text; Json::Dump(meta, &text);
  std::ofstream stream{metadata_path}; stream.exceptions(std::ios::badbit | std::ios::failbit);
  stream << text << '\n';
  std::printf("PASS %zu public CUDA objective calls / %zu sequences / %zu rows; comparison remains\n",
              input.Size(), sequences, rows);
}

void RunPrimitives(char const* input_path, char const* output_path) {
  namespace rp = ranking_primitive;
  std::string metadata_path = std::string(output_path)+".json";
  rr::Require(!std::filesystem::exists(metadata_path) && !std::filesystem::exists(output_path),
              "Primitive output/metadata already exists");
  auto queries = rp::LoadInputs(input_path);
  int devices = 0; Cuda(cudaGetDeviceCount(&devices));
  rr::Require(devices > 0, "Actual NVIDIA GPU required; no skipped pass or CPU fallback");
  Cuda(cudaSetDevice(0)); auto ctx = Context{}.MakeCUDA(0);
  dh::device_vector<rp::Query> source(queries);
  dh::device_vector<rp::Result> storage(queries.size());
  auto in = dh::ToSpan(source);
  auto out = dh::ToSpan(storage);
  thrust::for_each_n(ctx.CUDACtx()->CTP(), thrust::make_counting_iterator(std::size_t{0}),
      queries.size(), [=] __device__(std::size_t i) {
        auto q = in[i];
        float maximum = __uint_as_float(q.maximum), value = __uint_as_float(q.value);
        auto fgs = static_cast<float>(q.rows);
        float bound = fgs * maximum;
        double cost_bound = static_cast<double>(fgs) * (q.cost / q.minimum);
        // Invoke the actual source, including its unspecified nonfinite frexp
        // exponent. Do not substitute Metal's explicit exponent-zero policy.
        float m = common::CreateRoundingFactor(bound, q.rows);
        double cm = common::CreateRoundingFactor(cost_bound, q.rows);
        // Separate evaluations of the source delta expressions classify the
        // finite/nonfinite domain; they are not private-temporary captures.
        float delta = bound / (1.f - (2.f * static_cast<float>(q.rows)) *
                                      std::numeric_limits<float>::epsilon());
        double cost_delta = cost_bound / (1. - (2. * static_cast<double>(q.rows)) *
                                              std::numeric_limits<double>::epsilon());
        out[i] = {__float_as_uint(m),
                  __float_as_uint(common::TruncateWithRounding(m, value)),
                  __float_as_uint(common::TruncateWithRounding(m, -value)), __float_as_uint(delta),
                  cm, common::TruncateWithRounding(cm, q.term),
                  common::TruncateWithRounding(cm, -q.term), cost_delta};
      });
  Cuda(cudaDeviceSynchronize());
  auto observations = ReadDevice(out.data(), out.size());
  rp::Save(output_path, rp::kCuda, queries, observations.data());
  cudaDeviceProp gpu{}; Cuda(cudaGetDeviceProperties(&gpu, 0));
  int driver = 0, runtime = 0;
  Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};
  meta["status"] = String("completed-observations; comparison remains");
  meta["suite"] = String("ranking-primitive-v1"); meta["backend"] = String("cuda");
  meta["device"] = String(gpu.name);
  meta["compute_capability"] = String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"] = String(std::to_string(__CUDACC_VER_MAJOR__)+"."+
      std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"] = Integer(driver); meta["runtime"] = Integer(runtime);
  meta["thrust_version"] = Integer(THRUST_VERSION); meta["cub_version"] = Integer(CUB_VERSION);
  meta["cases"] = Integer(queries.size()); meta["result_fields"] = Integer(queries.size()*6);
  meta["diagnostic_fields"] = Integer(queries.size()*2);
  meta["input_file"] = String(input_path); meta["observation_file"] = String(output_path);
  meta["diagnostics"] = String("Separate source delta evaluations; not private-temporary captures");
  std::string text; Json::Dump(meta, &text);
  std::ofstream stream{metadata_path}; stream.exceptions(std::ios::badbit | std::ios::failbit);
  stream << text << '\n';
  std::printf("PASS CUDA primitive execution: %zu cases / %zu result and diagnostic fields; comparison remains\n",
              queries.size(), queries.size()*8);
}

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--objectives") {
      RunObjectives(argv[2], argv[3]); return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--primitives") {
      RunPrimitives(argv[2], argv[3]); return 0;
    }
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
