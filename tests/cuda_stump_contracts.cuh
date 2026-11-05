// Actual CUDA stump consumer plus a separately executed source reduction trace.
// The trace observes its own sums; it does not capture the API's private buffer.
#pragma once
#include <thrust/reduce.h>
#include "../xgboost/src/collective/aggregator.h"
#include "../xgboost/src/tree/fit_stump.h"
#include "stump_contract_io.h"

namespace xgboost::tree::cuda_impl {
void FitStump(Context const*,linalg::MatrixView<GradientPair const>,linalg::VectorView<float>);
}
void Stumps(char const* input_path,char const* output_path) {
  namespace sc=stump_contracts;
  using namespace xgboost;
  std::string metadata_path=std::string(output_path)+".json";
  Check(!std::filesystem::exists(metadata_path),"Stump metadata already exists");
  int devices=0; Cuda(cudaGetDeviceCount(&devices));
  Check(devices>0,"Actual NVIDIA GPU required; no skipped pass or CPU fallback");
  Cuda(cudaSetDevice(0)); auto ctx=Context{}.MakeCUDA(0); ctx.Init({{"device","cuda:0"},{"nthread","2"}});
  sc::InputReader inputs(input_path); sc::Writer report(output_path,sc::kCuda);
  std::size_t pairs=0,fields=0;
  for (unsigned index=0; index<sc::kCases; ++index) {
    auto in=inputs.Next(); auto s=in.spec;
    std::vector<GradientPair> host;
    for (unsigned i=0; i<s.pairs; ++i) host.emplace_back(in.storage[2*i],in.storage[2*i+1]);
    HostDeviceVector<GradientPair> storage{host,ctx.Device()};
    linalg::MatrixView<GradientPair const> gpair{storage.ConstDeviceSpan(),{s.rows,s.targets},
                                               {s.stride0,s.stride1},ctx.Device()};
    auto api=linalg::Constant(&ctx,0.f,s.targets);
    tree::cuda_impl::FitStump(&ctx,gpair,api.View(ctx.Device()));

    // Replay the unchanged iterator/reduce_by_key/GlobalSum/weight sequence in
    // tree/fit_stump.cu. This diagnostic is separate from the actual call above.
    auto sums=linalg::Constant(&ctx,GradientPairPrecise{},s.targets);
    auto key_it=dh::MakeTransformIterator<bst_target_t>(thrust::make_counting_iterator(0ul),
        [=] XGBOOST_DEVICE(std::size_t i)->bst_target_t { return i/gpair.Shape(0); });
    auto grad_it=dh::MakeTransformIterator<GradientPairPrecise>(thrust::make_counting_iterator(0ul),
        [=] XGBOOST_DEVICE(std::size_t i)->GradientPairPrecise {
          auto target=i/gpair.Shape(0); auto sample=i%gpair.Shape(0);
          return GradientPairPrecise{gpair(sample,target)};
        });
    auto d_sum=sums.View(ctx.Device());
    thrust::reduce_by_key(ctx.CUDACtx()->CTP(),key_it,key_it+gpair.Size(),grad_it,
                         thrust::make_discard_iterator(),dh::tbegin(d_sum.Values()));
    auto rc=collective::GlobalSum(&ctx,linalg::MakeVec(reinterpret_cast<double*>(d_sum.Values().data()),
                                                     d_sum.Size()*2,ctx.Device()));
    collective::SafeColl(rc);
    auto weights=linalg::Constant(&ctx,0.f,s.targets); auto out=weights.View(ctx.Device());
    thrust::for_each_n(ctx.CUDACtx()->CTP(),thrust::make_counting_iterator(0ul),s.targets,
        [=] XGBOOST_DEVICE(std::size_t i) mutable {
          out(i)=static_cast<float>(tree::CalcUnregularizedWeight(d_sum(i).GetGrad(),d_sum(i).GetHess()));
        });
    Cuda(cudaDeviceSynchronize());
    auto unchanged=Read(storage.ConstDeviceSpan().data(),s.pairs);
    Check(!std::memcmp(unchanged.data(),host.data(),host.size()*sizeof(GradientPair)),
          "CUDA stump modified input or padding");
    sc::Observation observed;
    for (auto sum:Read(d_sum.Values().data(),s.targets)) {
      observed.sums.push_back(sum.GetGrad()); observed.sums.push_back(sum.GetHess());
    }
    observed.weights=Read(out.Values().data(),s.targets);
    observed.api_weights=Read(api.View(ctx.Device()).Values().data(),s.targets);
    report.Add(in,observed); pairs+=s.rows*s.targets; fields+=4*s.targets;
    if ((index+1)%45==0) {
      std::printf("CUDA root/stump %u/%u: %zu gradient pairs / %zu fields\n",index+1,sc::kCases,pairs,fields);
      std::fflush(stdout);
    }
  }
  inputs.Finish(); report.Finish();
  cudaDeviceProp gpu{}; Cuda(cudaGetDeviceProperties(&gpu,0));
  int driver=0,runtime=0; Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};
  meta["status"]=String("completed observations; comparison remains"); meta["suite"]=String("stump-v1");
  meta["backend"]=String("cuda"); meta["device"]=String(gpu.name);
  meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"]=Integer(driver); meta["runtime"]=Integer(runtime);
  meta["cub_version"]=Integer(CUB_VERSION); meta["thrust_version"]=Integer(THRUST_VERSION);
  meta["cases"]=Integer(sc::kCases); meta["gradient_pairs"]=Integer(pairs); meta["output_fields"]=Integer(fields);
  meta["input_file"]=String(input_path); meta["observation_file"]=String(output_path);
  meta["scope"]=String("Actual tree::cuda_impl::FitStump on original strided inputs plus separate source reduction/weight trace; no private API sum capture; single worker only");
  std::string text; Json::Dump(meta,&text); std::ofstream stream{metadata_path};
  stream.exceptions(std::ios::badbit|std::ios::failbit); stream<<text<<'\n';
}
