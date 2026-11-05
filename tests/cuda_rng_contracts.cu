// Actual NVIDIA execution of the unchanged shared RNG and ranking-pair corpus.
// Every observed value is checked exactly; a summary is emitted only after all
// checks complete. The independent reference is not used inside CUDA kernels.
#include <cuda_runtime.h>
#include <cub/version.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/version.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../xgboost/src/common/cuda_context.cuh"
#include "../xgboost/src/common/device_helpers.cuh"
#include "../xgboost/src/common/random.cuh"
#include "../xgboost/src/objective/lambdarank_obj.cuh"
#include "../xgboost/src/tree/gpu_hist/sampler.cuh"
#include "xgboost/json.h"
#include "cuda_ranking_pairs.h"
#include "rng_contract_fixtures.h"

#if !__has_include(<cuda/std/random>)
#error "This contract targets modern CCCL; legacy Thrust streams require separate expectations."
#endif
static_assert(std::is_same_v<xgboost::common::cuda_impl::DefaultRng, cuda::std::philox4x64>);
static_assert(sizeof(std::size_t) == 8);

namespace common = xgboost::common;
namespace fixture = rng_contracts;
using namespace xgboost;

void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T> std::vector<T> Read(T const* data, std::size_t size) {
  Cuda(cudaDeviceSynchronize());
  std::vector<T> result(size);
  if (size) Cuda(cudaMemcpy(result.data(), data, size*sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}
void Equal(std::uint64_t actual, std::uint64_t expected, std::size_t row, char const* field) {
  if (actual != expected) {
    throw std::runtime_error(std::string(field)+" row="+std::to_string(row)+
        " actual="+std::to_string(actual)+" expected="+std::to_string(expected));
  }
}
std::uint32_t Bits(float value) {
  std::uint32_t word;
  std::memcpy(&word, &value, sizeof(word));
  return word;
}

// Count actual URBG calls by the production distribution, including rejection.
// This delegates every random word to the vendored DefaultRng alias.
struct CountedEngine {
  using Engine = common::cuda_impl::DefaultRng;
  using result_type = Engine::result_type;
  Engine engine;
  std::uint64_t position;
  __device__ CountedEngine(std::uint32_t seed, std::uint64_t start) : engine{seed}, position{start} {
    engine.discard(start);
  }
  __host__ __device__ static constexpr result_type min() { return Engine::min(); }
  __host__ __device__ static constexpr result_type max() { return Engine::max(); }
  __device__ result_type operator()() { ++position; return engine(); }
};

void Philox(Context const* ctx, Json* report) {
  auto queries = fixture::PhiloxQueries();
  Check(queries.size() == 1060276, "Incomplete Philox corpus");
  Equal(cuda_reference::Philox(20111115, 9999), 3409172418970261260ULL, 9999,
        "Published Philox check");
  dh::device_vector<fixture::PhiloxQuery> inputs(queries);
  struct Output { std::uint64_t word; std::uint32_t weight; };
  dh::device_vector<Output> storage(queries.size());
  auto in = dh::ToSpan(inputs); auto out = dh::ToSpan(storage);
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(std::size_t{0}),
      out.size(), [=] __device__(std::size_t i) {
        common::cuda_impl::DefaultRng rng{in[i].seed};
        rng.discard(in[i].row);
        out[i] = {rng(), __float_as_uint(tree::cuda_impl::RandomWeight{in[i].seed}(in[i].row))};
      });
  auto observed = Read(out.data(), out.size());
  for (std::size_t i = 0; i < queries.size(); ++i) {
    Equal(observed[i].word, cuda_reference::Philox(queries[i].seed, queries[i].row), i, "Philox");
    Equal(observed[i].weight, Bits(cuda_reference::Weight(queries[i].seed, queries[i].row)),
          i, "RandomWeight bits");
  }
  (*report)["philox_queries"] = Integer(queries.size());
  std::printf("PASS %zu exact CUDA Philox words and production RandomWeight outputs\n", queries.size());
  std::fflush(stdout);
}

void Distribution(Context const* ctx, Json* report) {
  auto queries = fixture::RankingQueries();
  Check(queries.size()/4 == 41200, "Incomplete ranking distribution corpus");
  dh::device_vector<std::uint64_t> inputs(queries);
  struct Output { std::uint64_t value, position, next_word; };
  dh::device_vector<Output> storage(queries.size()/4*16);
  auto in = dh::ToSpan(inputs); auto out = dh::ToSpan(storage);
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(std::size_t{0}),
      out.size(), [=] __device__(std::size_t i) {
        auto c = i/16*4;
        CountedEngine rng{static_cast<std::uint32_t>(in[c]), in[c+2]+i%16};
        common::cuda_impl::UniformIntDistribution<std::size_t> dist(0, in[c+1]-1);
        auto value = dist(rng);
        auto position = rng.position;
        out[i] = {value, position, rng()};
      });
  auto observed = Read(out.data(), out.size());
  for (std::size_t i = 0; i < observed.size(); ++i) {
    auto c = i/16*4;
    auto position = queries[c+2]+i%16;
    auto expected = cuda_reference::RankingUniform(queries[c], &position, queries[c+1]);
    Equal(observed[i].value, expected, i, "Ranking draw");
    Equal(observed[i].position, position, i, "Ranking consumed position");
    Equal(observed[i].next_word, cuda_reference::Philox(queries[c], position), i, "Next engine word");
  }
  (*report)["distribution_cases"] = Integer(queries.size()/4);
  (*report)["distribution_draws"] = Integer(observed.size());
  std::printf("PASS %zu exact CUDA ranking draws, consumed positions and next engine words\n", observed.size());
  std::fflush(stdout);
}

void Pairs(Context const* ctx, Json* report) {
  std::size_t cases = 0, checked = 0, reversed = 0, homogeneous = 0;
  (*report)["pair_cases"] = Array{};
  fixture::PairCases([&](fixture::PairCase const& c) {
    MetaInfo info;
    info.num_row_ = c.predictions.size();
    info.group_ptr_ = {0};
    for (auto size : c.sizes) info.group_ptr_.push_back(info.group_ptr_.back()+size);
    // MetaInfo stores one contiguous target; the Metal test additionally checks
    // the same target through a stride-three view. Padding is not label data.
    std::vector<float> labels(info.num_row_);
    for (std::size_t i = 0; i < labels.size(); ++i) labels[i] = c.labels[3*i];
    info.labels.Data()->HostVector() = labels;
    info.labels.Reshape(labels.size(), 1);
    info.labels.SetDevice(ctx->Device());
    HostDeviceVector<float> predictions(c.predictions, ctx->Device());
    ltr::LambdaRankParam param;
    param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", "mean"},
                                 {"lambdarank_num_pair_per_sample", std::to_string(c.count)}});
    auto cache = std::make_shared<ltr::RankingCache>(ctx, info, param);
    auto threads = cache->CUDAThreadsGroupPtr();
    Check(cache->CUDAThreads() == labels.size()*c.count, "Wrong production pair count");
    obj::cuda_impl::KernelInputs args{};
    args.d_group_ptr = cache->DataGroupPtr(ctx);
    args.d_threads_group_ptr = threads;
    args.predts = predictions.ConstDeviceSpan();
    args.labels = info.labels.View(ctx->Device());
    args.seed = c.seed;
    // The empty corpus still constructs and checks the real cache's zero-work
    // metadata. No pair kernel is launched when it has no rows.
    if (!labels.empty()) {
      args.d_sorted_idx = cache->SortedIdx(ctx, args.predts);
      args.d_y_sorted_idx = obj::cuda_impl::SortY(ctx, info, args.d_sorted_idx, cache);
    }
    struct Pair { std::uint64_t first, second; };
    dh::device_vector<Pair> storage(cache->CUDAThreads());
    auto out = dh::ToSpan(storage);
    if (!out.empty()) {
      thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(std::size_t{0}),
          out.size(), [=] __device__(std::size_t idx) {
            auto group = dh::SegmentId(args.d_threads_group_ptr, idx);
            auto [a, b] = obj::cuda_impl::MakePairsOp<false>{args}(idx, group);
            out[idx] = {a, b};
          });
    }
    auto observed = Read(out.data(), out.size());
    auto actual_rank = Read(args.d_sorted_idx.data(), args.d_sorted_idx.size());
    for (std::size_t g = 0; g < c.sizes.size(); ++g) {
      std::vector<std::size_t> rank(c.sizes[g]);
      std::iota(rank.begin(), rank.end(), 0);
      auto start = info.group_ptr_[g];
      std::stable_sort(rank.begin(), rank.end(), [&](auto a, auto b) {
        return c.predictions[start+a] > c.predictions[start+b];
      });
      for (std::size_t i = 0; i < rank.size(); ++i)
        Equal(actual_rank[start+i], rank[i], i, "Production prediction rank");
      std::vector<float> y(labels.begin()+start, labels.begin()+info.group_ptr_[g+1]);
      auto expected = cuda_reference::MeanPairs(y, rank, c.seed+static_cast<std::uint32_t>(g), c.count);
      for (std::size_t i = 0; i < expected.size(); ++i) {
        auto actual = observed[std::size_t{start}*c.count+i];
        Equal(actual.first, expected[i][0], i, "Production pair first");
        Equal(actual.second, expected[i][1], i, "Production pair second");
        reversed += actual.first > actual.second;
        homogeneous += actual.first == actual.second;
        ++checked;
      }
    }
    Json result{Object{}};
    result["seed"] = Integer(c.seed); result["count"] = Integer(c.count);
    result["mode"] = Integer(c.mode); result["rows"] = Integer(labels.size());
    result["pairs"] = Integer(observed.size()); result["status"] = String("exact");
    get<Array>((*report)["pair_cases"]).push_back(result);
    ++cases;
    std::printf("CUDA ranking case %zu/74: seed=%u k=%u mode=%d rows=%zu; cumulative pairs=%zu\n",
                cases, c.seed, c.count, c.mode, labels.size(), checked);
    std::fflush(stdout);
  });
  Check(cases == 74 && checked == 9068696 && reversed && homogeneous,
        "Incomplete production ranking pair corpus");
  (*report)["pair_count"] = Integer(checked); (*report)["reversed_pairs"] = Integer(reversed);
  (*report)["homogeneous_pairs"] = Integer(homogeneous);
}

int main(int argc, char** argv) {
  Json report{Object{}};
  std::ofstream output;
  int result = 0;
  try {
    Check(argc == 2, "Usage: cuda_rng_contracts new-report.json");
    Check(!std::ifstream{argv[1]}.good(), "Report already exists");
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(argv[1]);
    report["suite"] = String("rng-v1"); report["backend"] = String("cuda");
    report["rng_branch"] = String("cuda::std::philox4x64");
    report["cccl_cub_version"] = Integer(CUB_VERSION);
    report["thrust_version"] = Integer(THRUST_VERSION);
    report["nvcc"] = String(std::to_string(__CUDACC_VER_MAJOR__)+"."+
        std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
    int devices = 0;
    Cuda(cudaGetDeviceCount(&devices)); Check(devices > 0, "NVIDIA GPU required; no skipped pass");
    Cuda(cudaSetDevice(0));
    auto ctx = Context{}.MakeCUDA(0);
    cudaDeviceProp device{}; Cuda(cudaGetDeviceProperties(&device, 0));
    report["device"] = String(device.name);
    report["compute_capability"] = String(std::to_string(device.major)+"."+std::to_string(device.minor));
    int driver = 0, runtime = 0;
    Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
    report["driver"] = Integer(driver); report["runtime"] = Integer(runtime);
    report["comparison"] = String("Every actual GPU result checked exactly against the shared independent host oracle; no tolerance");
    report["scope"] = String("P5-N06 ranking RNG/pairs and training Philox/RandomWeight primitives; no unweighted shuffle or full gradient comparison");
    Philox(&ctx, &report);
    Distribution(&ctx, &report);
    Pairs(&ctx, &report);
    report["status"] = String("passed-exact-corpus");
    std::printf("PASS actual NVIDIA RNG and ranking pair corpus; broader CUDA parity remains\n");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL CUDA RNG: %s\n", error.what());
    report["status"] = String("failed"); report["error"] = String(error.what()); result = 1;
  }
  if (output.is_open()) {
    std::string text; Json::Dump(report, &text); output << text << '\n';
  }
  return result;
}
