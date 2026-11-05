// Actual production CUDA leaf accumulation and weight consumers.
#pragma once
#include "../xgboost/src/tree/gpu_hist/leaf_sum.cuh"
#include "integer_leaf_contract_io.h"

namespace cuda_integer_leaf_trace {
namespace lc=integer_leaf_contracts;
using namespace xgboost;
inline std::vector<GradientPair> Pairs(std::vector<float> const& data) {
  std::vector<GradientPair> result;result.reserve(data.size()/2);
  for(std::size_t i=0;i<data.size();i+=2)result.emplace_back(data[i],data[i+1]);
  return result;
}
inline void Snapshot(Context const* ctx,tree::GradientQuantiserGroup const& quantizer,
                     linalg::Matrix<GradientPairInt64>* sums,std::vector<unsigned> const& order,
                     lc::Observation* observed) {
  auto values=sums->View(ctx->Device());auto q=quantizer.DeviceSpan();auto targets=values.Shape(1);
  dh::device_vector<GradientPairPrecise> converted(values.Size());auto out=dh::ToSpan(converted);
  dh::LaunchN(values.Size(),ctx->CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i){
    out[i]=q[i%targets].ToFloatingPoint(values(i/targets,i%targets));
  });
  auto actual=Read(out.data(),out.size());auto exact=Read(values.Values().data(),values.Size());
  std::vector<std::uint64_t> canonical(2*actual.size()),integers(canonical.size());
  for(std::size_t row=0;row<order.size();++row)for(std::size_t t=0;t<targets;++t) {
    auto i=row*targets+t,at=2*(order[row]*targets+t);
    canonical[at]=lc::Bits<std::uint64_t>(actual[i].GetGrad());
    canonical[at+1]=lc::Bits<std::uint64_t>(actual[i].GetHess());
    integers[at]=lc::Bits<std::uint64_t>(exact[i].GetQuantisedGrad());
    integers[at+1]=lc::Bits<std::uint64_t>(exact[i].GetQuantisedHess());
  }
  observed->sums.insert(observed->sums.end(),canonical.begin(),canonical.end());
  observed->integers.insert(observed->integers.end(),integers.begin(),integers.end());
}
inline void Weights(Context const* ctx,tree::GradientQuantiserGroup const& quantizer,
                    linalg::Matrix<GradientPairInt64>* sums,std::vector<unsigned> const& order,
                    lc::Observation* observed) {
  auto targets=sums->Shape(1),n=sums->Size();
  std::vector<bst_node_t> host_nodes(order.begin(),order.end());
  dh::device_vector<bst_node_t> nodes(host_nodes);
  for(unsigned kind=0;kind<5;++kind) {
    tree::TrainParam param;
    param.reg_lambda=kind==0?0.f:1.25f;param.reg_alpha=kind==2?.75f:0.f;
    param.max_delta_step=kind==3?.125f:0.f;param.min_child_weight=1e20f;param.learning_rate=.3f;
    dh::device_vector<float> lower(n,.125f),upper(n,.25f),weights(n+2,-913.f);
    tree::TreeEvaluator::SplitEvaluator<tree::EvalParam> evaluator{
        nullptr,lower.data().get(),upper.data().get(),kind==4,static_cast<bst_target_t>(targets)};
    auto out=linalg::MakeTensorView(ctx,dh::ToSpan(weights).subspan(1,n),order.size(),targets);
    tree::cuda_impl::LeafWeight(ctx,tree::EvalParam{param},evaluator,dh::ToSpan(nodes),
                               quantizer.DeviceSpan(),sums->View(ctx->Device()),out);
    auto actual=Read(weights);std::vector<std::uint32_t> canonical(n+2);
    canonical.front()=lc::Bits<std::uint32_t>(actual.front());canonical.back()=lc::Bits<std::uint32_t>(actual.back());
    for(std::size_t r=0;r<order.size();++r)for(std::size_t t=0;t<targets;++t)
      canonical[1+order[r]*targets+t]=lc::Bits<std::uint32_t>(actual[1+r*targets+t]);
    observed->weights.insert(observed->weights.end(),canonical.begin(),canonical.end());
  }
}
}  // namespace cuda_integer_leaf_trace

inline void IntegerLeaves(char const* input_path,char const* output_path) {
  namespace lc=integer_leaf_contracts;namespace trace=cuda_integer_leaf_trace;
  using namespace xgboost;using tree::LeafInfo;using tree::cuda_impl::RowIndexT;
  std::string metadata_path=std::string(output_path)+".json";
  Check(!std::filesystem::exists(metadata_path),"Integer leaf metadata already exists");
#if THRUST_MAJOR_VERSION < 3
  throw std::runtime_error("Full paged leaf replay requires Thrust3 accumulation; older source branch overwrites pages");
#endif
  int devices=0;Cuda(cudaGetDeviceCount(&devices));Check(devices>0,"Actual NVIDIA GPU required; no skipped pass or CPU fallback");
  Cuda(cudaSetDevice(0));auto ctx=Context{}.MakeCUDA(0);ctx.Init({{"device","cuda:0"},{"nthread","2"}});
  lc::InputReader inputs(input_path);lc::Writer report(output_path,lc::kCuda);
  std::size_t executed=0,excluded=0,sum_fields=0,weight_fields=0;
  for(unsigned index=0;index<lc::kCases;++index) {
    auto input=inputs.Next();auto s=input.spec;auto shape=lc::Layout(s);lc::Observation observed;
    if(s.domain) {observed.status=2;++excluded;} else {
      auto original=trace::Pairs(input.original),sampled=trace::Pairs(input.sampled);
      HostDeviceVector<GradientPair> original_buffer{original,ctx.Device()},sampled_buffer{sampled,ctx.Device()};
      auto backing=original_buffer.DeviceSpan();
      linalg::MatrixView<GradientPair const> source{backing.subspan(shape.offset,shape.extent),
          {shape.rows,s.targets},{shape.stride0,shape.stride1},ctx.Device()};
      tree::GradientQuantiserGroup quantizer{&ctx,source};
      auto original_after=Read(backing.data(),backing.size());
      Check(!std::memcmp(original.data(),original_after.data(),original.size()*sizeof(GradientPair)),"CUDA leaf quantizer changed original backing");
      if(s.family==0) {
        dh::LaunchN(backing.size(),ctx.CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i){backing[i]=GradientPair{919,313};});
      } else if(s.family>=2) {
        // Retain the original quantizer while mutating the original allocation,
        // as the native curvature/range cases do after rejecting/rescaling rows.
        Cuda(cudaMemcpyAsync(backing.data(),sampled.data(),sampled.size()*sizeof(GradientPair),
                             cudaMemcpyHostToDevice,ctx.CUDACtx()->Stream()));
      }
      auto values_backing=s.family==0?sampled_buffer.DeviceSpan():backing;
      linalg::MatrixView<GradientPair const> values{values_backing.subspan(shape.offset,shape.extent),
          {shape.rows,s.targets},{shape.stride0,shape.stride1},ctx.Device()};
      linalg::Matrix<GradientPairInt64> sums{{shape.leaves,s.targets},ctx.Device()};
      std::vector<unsigned> order;
      if(s.family==0)for(int leaf=6;leaf>=0;--leaf)order.push_back((leaf*3+2)%7);
      else for(unsigned leaf=0;leaf<shape.leaves;++leaf)order.push_back(leaf);
      auto add=[&](std::vector<LeafInfo> const& leaves,std::vector<RowIndexT> const& indices) {
        dh::device_vector<RowIndexT> rows(indices);
        tree::cuda_impl::LeafGradSum(&ctx,leaves,quantizer.DeviceSpan(),dh::ToSpan(rows),values,sums.View(ctx.Device()));
        trace::Snapshot(&ctx,quantizer,&sums,order,&observed);
      };
      if(s.family==0) {
        unsigned boundaries[]{0,s.rows/3,s.rows*2/3,s.rows};
        for(unsigned page=0;page<3;++page) {
          auto begin=boundaries[page],end=boundaries[page+1];
          std::vector<RowIndexT> rows;for(auto r=begin;r<end;++r)rows.push_back(r+5);
          std::vector<LeafInfo> leaves;
          for(int leaf=6;leaf>=0;--leaf) {
            unsigned lo=std::max(begin,s.rows*std::min(unsigned(leaf),5u)/5);
            unsigned hi=std::min(end,s.rows*std::min(unsigned(leaf+1),5u)/5);
            LeafInfo info{};info.nidx=(leaf*3+2)%7;
            if(hi>lo)info.node.segment={lo-begin,hi-begin};
            leaves.push_back(info);
          }
          add(leaves,rows);
        }
        trace::Weights(&ctx,quantizer,&sums,order,&observed);
      } else if(s.family==1) {
        LeafInfo empty{},nonempty{};empty.nidx=0;nonempty.nidx=1;
        if(s.mode) {
          nonempty.node.segment={0,3};add({empty,nonempty},{0,1,2});
          nonempty.node.segment={0,1};add({empty,nonempty},{3});
        } else {nonempty.node.segment={0,4};add({empty,nonempty},{0,1,2,3});}
      } else {
        LeafInfo leaf{};leaf.nidx=0;leaf.node.segment={0,s.rows};
        std::vector<RowIndexT> rows;for(unsigned r=0;r<s.rows;++r)rows.push_back(r);add({leaf},rows);
      }
      auto final=Read(values_backing.data(),values_backing.size());
      Check(!std::memcmp(sampled.data(),final.data(),sampled.size()*sizeof(GradientPair)),"CUDA leaf accumulation changed sampled backing/guards");
      ++executed;
    }
    report.Add(input,observed);sum_fields+=observed.sums.size();weight_fields+=observed.weights.size();
    if((index+1)%25==0 || index+1==lc::kCases) {
      std::printf("CUDA integer leaf %u/%u: executed=%zu excluded=%zu sums=%zu weights=%zu\n",index+1,lc::kCases,executed,excluded,sum_fields,weight_fields);std::fflush(stdout);
    }
  }
  inputs.Finish();report.Finish();
  cudaDeviceProp gpu{};Cuda(cudaGetDeviceProperties(&gpu,0));int driver=0,runtime=0;
  Cuda(cudaDriverGetVersion(&driver));Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};meta["suite"]=String("integer-leaf-v1");meta["backend"]=String("cuda");meta["device"]=String(gpu.name);
  meta["status"]=String("completed observations with domain exclusions; comparison remains");
  meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"]=Integer(driver);meta["runtime"]=Integer(runtime);meta["cub_version"]=Integer(CUB_VERSION);meta["thrust_version"]=Integer(THRUST_VERSION);
  meta["cases"]=Integer(lc::kCases);meta["executed_cases"]=Integer(executed);meta["domain_exclusions"]=Integer(excluded);
  meta["converted_sum_fields"]=Integer(sum_fields);meta["actual_integer_fields"]=Integer(sum_fields);meta["weight_fields_including_guards"]=Integer(weight_fields);
  meta["input_file"]=String(input_path);meta["observation_file"]=String(output_path);
  meta["scope"]=String("Actual quantizer, persistent LeafGradSum pages and LeafWeight APIs; actual integer diagnostics separate from converted sums; slots reordered for reporting only; no Metal private integer capture; single worker");
  std::string text;Json::Dump(meta,&text);std::ofstream stream{metadata_path};stream.exceptions(std::ios::badbit|std::ios::failbit);stream<<text<<'\n';
}
