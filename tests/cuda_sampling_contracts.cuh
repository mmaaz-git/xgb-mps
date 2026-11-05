// Actual production CUDA quantizer/sampler consumers for the full native corpus.
#pragma once
#include "../xgboost/src/tree/updater_gpu_hist.cuh"
#include "sampling_contract_io.h"

inline void SamplingRngCheck() {
  std::mt19937 inputs{1937};
  unsigned count=0;
  auto check=[&](std::uint32_t seed) {
    auto engine=sampling_contracts::RowSeedEngine(seed);auto before=engine;
    Check(engine()==seed && engine!=before,"Sampling RNG alignment/consumption failed");++count;
  };
  for(std::uint32_t seed:{0u,1u,7919u,0x80000000u,0xffffffffu})check(seed);
  for(unsigned i=0;i<10000;++i)check(inputs());
  std::printf("PASS sampling MT19937 row-seed alignment: %u cases; no GPU execution\n",count);
}

namespace cuda_sampling_trace {
namespace sc=sampling_contracts;
using namespace xgboost;
namespace impl=tree::cuda_impl;
inline std::vector<GradientPair> Pairs(std::vector<float> const& data) {
  std::vector<GradientPair> result;result.reserve(data.size()/2);
  for(std::size_t i=0;i<data.size();i+=2)result.emplace_back(data[i],data[i+1]);
  return result;
}
inline void Append(sc::Words* out,std::vector<GradientPair> const& data) {
  for(auto p:data){out->push_back(sc::Bits<std::uint32_t>(p.GetGrad()));out->push_back(sc::Bits<std::uint32_t>(p.GetHess()));}
}
inline void Roots(Context const* ctx,linalg::Matrix<GradientPairInt64>* integers,
                  tree::GradientQuantiserGroup const& quantizer,sc::Words* out) {
  auto view=integers->View(ctx->Device());auto targets=view.Shape(1);
  std::vector<GradientPairInt64> exact;
  std::vector<GradientPairPrecise> floating;
  if(targets==1) {
    // The scalar updater's actual reduction primitive and host finalizer.
    auto begin=linalg::tcbegin(view);
    auto sum=dh::Reduce(ctx->CUDACtx()->CTP(),begin,begin+view.Size(),GradientPairInt64{},cuda::std::plus<GradientPairInt64>{});
    collective::SafeColl(collective::GlobalSum(ctx,linalg::MakeVec(reinterpret_cast<std::int64_t*>(&sum),2)));
    exact.push_back(sum);floating.push_back(quantizer[0].ToFloatingPoint(sum));
  } else {
    // Actual vector updater root reducer. This observes the sampled integer
    // matrix; it does not instantiate a complete tree/histogram updater.
    dh::device_vector<GradientPairInt64> sums(targets,GradientPairInt64{});
    impl::CalcRootSum(ctx,view,dh::ToSpan(sums));
    auto d=dh::ToSpan(sums);
    collective::SafeColl(collective::GlobalSum(ctx,linalg::MakeVec(reinterpret_cast<std::int64_t*>(d.data()),2*targets,ctx->Device())));
    dh::device_vector<GradientPairPrecise> converted(targets);auto result=dh::ToSpan(converted);
    auto q=quantizer.DeviceSpan();
    dh::LaunchN(targets,ctx->CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t t){result[t]=q[t].ToFloatingPoint(d[t]);});
    exact=Read(d.data(),targets);floating=Read(result.data(),targets);
  }
  for(auto p:floating){out->push_back(sc::Bits<std::uint64_t>(p.GetGrad()));out->push_back(sc::Bits<std::uint64_t>(p.GetHess()));}
  for(auto p:exact){out->push_back(sc::Bits<std::uint64_t>(p.GetQuantisedGrad()));out->push_back(sc::Bits<std::uint64_t>(p.GetQuantisedHess()));}
}
inline std::vector<GradientPair> Replay(Context const* ctx,impl::Sampler* sampler,sc::Spec s,
                                       std::vector<float> const& initial) {
  auto shape=sc::Layout(s);auto host=Pairs(initial);
  HostDeviceVector<GradientPair> backing{host,ctx->Device()};
  linalg::Matrix<GradientPair> dense{{s.rows,shape.replay_targets},ctx->Device()};
  auto input=backing.DeviceSpan();auto values=dense.View(ctx->Device());
  unsigned offset=s.family==0?4:0;
  // The CUDA public replay takes an owning contiguous matrix. Project the
  // original padded view on GPU, then scatter only logical elements back.
  dh::LaunchN(s.rows*shape.replay_targets,ctx->CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i) mutable {
    auto r=i/shape.replay_targets,t=i%shape.replay_targets;values(r,t)=input[offset+r*shape.replay_stride+t];
  });
  sampler->ApplySampling(ctx,&dense);
  dh::LaunchN(s.rows*shape.replay_targets,ctx->CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i){
    auto r=i/shape.replay_targets,t=i%shape.replay_targets;input[offset+r*shape.replay_stride+t]=values(r,t);
  });
  return Read(input.data(),input.size());
}
}  // namespace cuda_sampling_trace

inline void Sampling(char const* input_path,char const* output_path) {
  namespace trace=cuda_sampling_trace;namespace sc=sampling_contracts;
  using namespace xgboost;
  std::string metadata_path=std::string(output_path)+".json";
  Check(!std::filesystem::exists(metadata_path),"Sampling metadata already exists");
  int devices=0;Cuda(cudaGetDeviceCount(&devices));Check(devices>0,"Actual NVIDIA GPU required; no CPU fallback or skipped pass");
  Cuda(cudaSetDevice(0));auto ctx=Context{}.MakeCUDA(0);ctx.Init({{"device","cuda:0"},{"nthread","2"}});
  sc::InputReader inputs(input_path);sc::Writer report(output_path,sc::kCuda);
  std::unique_ptr<tree::cuda_impl::Sampler> lifecycle;
  std::size_t fields=0,executed=0,excluded=0,projections=0;
  for(unsigned index=0;index<sc::kCases;++index) {
    auto in=inputs.Next();auto s=in.spec;auto shape=sc::Layout(s);sc::Observation observed;
    if(s.domain) {
      // Frozen int64/nonfinite conversion domain exclusions. Do not execute
      // undefined casts or manufacture roots/replay values for these records.
      observed.status=2;++excluded;
    } else {
      auto fraction=sc::Bits<float>(s.fraction);
      std::unique_ptr<tree::cuda_impl::Sampler> single;
      tree::cuda_impl::Sampler* sampler=nullptr;
      if(s.family==1) {
        if(s.step==0) {
          ctx.Init({{"nthread",std::to_string(s.threads)}});ctx.Rng().seed(701);
          lifecycle=std::make_unique<tree::cuda_impl::Sampler>(s.rows,fraction,tree::TrainParam::kGradientBased);
        }
        Check(bool(lifecycle),"Missing live sampling lifecycle");sampler=lifecycle.get();
      } else {
        ctx.Rng()=sc::RowSeedEngine(s.seed);
        single=std::make_unique<tree::cuda_impl::Sampler>(s.rows,fraction,tree::TrainParam::kGradientBased);
        sampler=single.get();
      }
      auto expected_engine=ctx.Rng();if(fraction<1)expected_engine();
      auto host=trace::Pairs(in.storage);
      HostDeviceVector<GradientPair> storage{host,ctx.Device()};
      auto backing=storage.DeviceSpan();
      linalg::MatrixView<GradientPair const> source{backing.subspan(shape.offset,shape.extent),
          {s.rows,s.targets},{shape.stride0,shape.stride1},ctx.Device()};
      tree::GradientQuantiserGroup quantizer{&ctx,source};
      linalg::Matrix<GradientPairInt64> integers;
      tree::CalcQuantizedGpairs(&ctx,source,quantizer.DeviceSpan(),&integers);
      sampler->Sample(&ctx,integers.View(ctx.Device()),quantizer.DeviceSpan());
      Check(ctx.Rng()==expected_engine,"CUDA sampler parent RNG consumption differs");
      auto info=sampler->GetSamplingInfo();
      if(s.family!=1 && fraction<1)Check(info.seed==s.seed,"CUDA sampler did not consume the requested row seed");
      auto unchanged=Read(backing.data(),backing.size());
      Check(host.empty() || !std::memcmp(host.data(),unchanged.data(),host.size()*sizeof(GradientPair)),"CUDA quantizer/sampler modified source or padding");
      if(s.family==0 && fraction<1) {
        Check(info.reg_abs_grad.size()==s.rows && info.threshold_index<info.thresholds.size(),"Missing actual CUDA sampling diagnostics");
        for(float value:Read(info.reg_abs_grad.data(),s.rows))observed.actual.push_back(sc::Bits<std::uint32_t>(value));
        auto threshold=Read(info.thresholds.data()+info.threshold_index,1);
        observed.actual.push_back(sc::Bits<std::uint32_t>(threshold[0]));
      }
      if(s.family!=2) {
        trace::Roots(&ctx,&integers,quantizer,&observed.actual);
        if(s.family==1) {
          linalg::Matrix<GradientPair> values{{s.rows,s.targets},ctx.Device()};
          values.Data()->HostVector()=host;values.Data()->SetDevice(ctx.Device());
          sampler->ApplySampling(&ctx,&values);
          auto view=values.View(ctx.Device());trace::Append(&observed.actual,Read(view.Values().data(),view.Size()));
        }
        trace::Append(&observed.actual,trace::Replay(&ctx,sampler,s,in.replay));
        if(s.family==0) {
          ++projections;
          dh::LaunchN(backing.size(),ctx.CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i){backing[i]=GradientPair{919,313};});
          trace::Roots(&ctx,&integers,quantizer,&observed.actual);
        } else {
          auto next=ctx.Rng();observed.actual.push_back(next());
          if(s.step==2){ctx.Rng()();lifecycle.reset();}
        }
      }
      Cuda(cudaDeviceSynchronize());++executed;
    }
    report.Add(in,observed);fields+=observed.actual.size();
    if((index+1)%20==0 || index+1==sc::kCases) {
      std::printf("CUDA sampling %u/%u: executed=%zu excluded=%zu fields=%zu\n",index+1,sc::kCases,executed,excluded,fields);std::fflush(stdout);
    }
  }
  inputs.Finish();report.Finish();
  cudaDeviceProp gpu{};Cuda(cudaGetDeviceProperties(&gpu,0));int driver=0,runtime=0;
  Cuda(cudaDriverGetVersion(&driver));Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};meta["status"]=String("completed observations with domain exclusions; comparison remains");
  meta["suite"]=String("sampling-v1");meta["backend"]=String("cuda");meta["device"]=String(gpu.name);
  meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"]=Integer(driver);meta["runtime"]=Integer(runtime);meta["cub_version"]=Integer(CUB_VERSION);meta["thrust_version"]=Integer(THRUST_VERSION);
  meta["cases"]=Integer(sc::kCases);meta["executed_cases"]=Integer(executed);meta["domain_exclusions"]=Integer(excluded);
  meta["output_fields"]=Integer(fields);meta["padded_replay_projections"]=Integer(projections);
  meta["input_file"]=String(input_path);meta["observation_file"]=String(output_path);
  meta["scope"]=String("Actual quantizer, integer sampling, SamplingInfo and floating replay APIs; scalar/vector source root reducers; 18 persistent lifecycle sequences; original padded replay projected via GPU gather/scatter; no private CDF capture");
  meta["seed_alignment"]=String("Low-level explicit row seeds use verified test-supplied MT19937 states, not ordinary seed(row_seed); lifecycle uses original parent seed701");
  std::string text;Json::Dump(meta,&text);std::ofstream stream{metadata_path};stream.exceptions(std::ios::badbit|std::ios::failbit);stream<<text<<'\n';
}
