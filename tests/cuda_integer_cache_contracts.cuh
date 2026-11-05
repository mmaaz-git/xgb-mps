// Production ELLPACK histogram builder/cache replay; no replacement histogram kernel.
#pragma once
#include "../xgboost/src/data/gradient_index.h"
#include "../xgboost/src/data/ellpack_page.cuh"
#include "../xgboost/src/tree/gpu_hist/histogram.cuh"
#include "integer_cache_contract_io.h"

namespace cuda_integer_cache_trace {
namespace hc=integer_cache_contracts;
using namespace xgboost;
template <typename Accessor> struct DecodeBins {
  Accessor accessor;
  common::Span<std::uint32_t> output;
  __device__ void operator()(std::size_t row) const {
    output[row]=accessor.GetBinIndex(row,0);
  }
};
inline std::vector<GradientPair> Pairs(std::vector<float> const& data) {
  std::vector<GradientPair> out;for(std::size_t i=0;i<data.size();i+=2)out.emplace_back(data[i],data[i+1]);return out;
}
inline void Integer(std::vector<std::uint64_t>* out,GradientPairInt64 v) {
  out->push_back(hc::Bits<std::uint64_t>(v.GetQuantisedGrad()));out->push_back(hc::Bits<std::uint64_t>(v.GetQuantisedHess()));
}
inline void Floating(std::vector<std::uint64_t>* out,GradientPairPrecise v) {
  out->push_back(hc::Bits<std::uint64_t>(v.GetGrad()));out->push_back(hc::Bits<std::uint64_t>(v.GetHess()));
}
inline void Snapshot(Context const* ctx,tree::GradientQuantiserGroup const& quantizer,
                     common::Span<GradientPairInt64> values,hc::Observation* observed) {
  dh::device_vector<GradientPairPrecise> converted(values.size());auto output=dh::ToSpan(converted);auto q=quantizer.DeviceSpan();
  dh::LaunchN(values.size(),ctx->CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i){output[i]=q[0].ToFloatingPoint(values[i]);});
  for(auto value:Read(output.data(),output.size()))Floating(&observed->values,value);
  for(auto value:Read(values.data(),values.size()))Integer(&observed->integers,value);
}
inline void Root(Context const* ctx,tree::GradientQuantiserGroup const& quantizer,
                 common::Span<GradientPairInt64> values,hc::Observation* observed) {
  // Same reduction/finalizer primitives as the scalar updater, here applied to
  // the actual histogram bins. This is a diagnostic, not its private root call.
  auto sum=dh::Reduce(ctx->CUDACtx()->CTP(),dh::tbegin(values),dh::tend(values),GradientPairInt64{},cuda::std::plus<GradientPairInt64>{});
  Integer(&observed->integers,sum);Floating(&observed->values,quantizer[0].ToFloatingPoint(sum));
}
inline std::vector<std::uint64_t> Prefix(tree::DeviceHistogramBuilder* builder) {
  std::vector<std::uint64_t> out;
  for(unsigned node=0;node<16;++node){auto values=builder->GetNodeHistogram(node);for(auto v:Read(values.data(),values.size()))Integer(&out,v);}
  return out;
}
}  // namespace cuda_integer_cache_trace

inline void IntegerCache(char const* input_path,char const* output_path) {
  namespace hc=integer_cache_contracts;namespace trace=cuda_integer_cache_trace;using namespace xgboost;
  std::string metadata_path=std::string(output_path)+".json";Check(!std::filesystem::exists(metadata_path),"Cache metadata already exists");
  int devices=0;Cuda(cudaGetDeviceCount(&devices));Check(devices>0,"Actual NVIDIA GPU required; no CPU fallback or skipped pass");
  Cuda(cudaSetDevice(0));auto ctx=Context{}.MakeCUDA(0);ctx.Init({{"device","cuda:0"},{"nthread","1"}});
  Context host;host.Init({{"device","cpu"},{"nthread","1"}});
  hc::InputReader inputs(input_path);hc::Writer report(output_path,hc::kCuda);
  std::size_t fields=0;
  for(unsigned index=0;index<hc::kCases;++index) {
    auto input=inputs.Next();auto s=input.spec;hc::Observation observed;
    SparsePage page;page.offset.HostVector()={0,1,2,3,4};
    for(float value:input.data)page.data.HostVector().emplace_back(0,value);
    common::HistogramCuts cuts{1};cuts.cut_ptrs_.HostVector()={0,s.bins};cuts.cut_values_.HostVector()=input.cuts;
    // Preserve each original GHistIndex width/layout, then use the actual
    // CUDA conversion constructor instead of bypassing packing with fake bins.
    GHistIndexMatrix gidx{&host,page,{},std::move(cuts),int(s.max_bin),bool(s.dense),.2};
    EllpackPageImpl ellpack{&ctx,gidx,{}};
    dh::device_vector<std::uint32_t> decoded(4);auto bin_output=dh::ToSpan(decoded);
    ellpack.Visit(&ctx,{},[&](auto accessor){
      dh::LaunchN(4,ctx.CUDACtx()->Stream(),trace::DecodeBins<decltype(accessor)>{accessor,bin_output});
    });
    observed.bins=Read(decoded);
    tree::FeatureGroups features{ellpack.Cuts(),ellpack.IsDenseCompressed(),tree::DftStHistShmemBytes(ctx.Ordinal())};
    tree::DeviceHistogramBuilder builder;
    auto initialize=[&](bool guards){
      builder.Reset(&ctx,1024,s.bins,false);std::vector<bst_node_t> nodes;for(int i=0;i<16;++i)nodes.push_back(i);
      builder.AllocateHistograms(&ctx,nodes);
      if(guards)for(int node:nodes){auto values=builder.GetNodeHistogram(node);
        dh::LaunchN(values.size(),ctx.CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i){values[i]=GradientPairInt64{17,-23};});}
    };
    initialize(true);
    linalg::Matrix<GradientPair> gradients{{4u,1u},ctx.Device()};linalg::Matrix<GradientPairInt64> integers;
    std::unique_ptr<tree::GradientQuantiserGroup> quantizer;
    auto set_gradients=[&](std::vector<float> const& values){
      gradients.Data()->HostVector()=trace::Pairs(values);gradients.Data()->SetDevice(ctx.Device());
      quantizer=std::make_unique<tree::GradientQuantiserGroup>(&ctx,gradients.View(ctx.Device()));
      tree::CalcQuantizedGpairs(&ctx,gradients.View(ctx.Device()),quantizer->DeviceSpan(),&integers);
    };
    set_gradients(input.first);
    auto build=[&](unsigned page_index,std::vector<std::uint32_t> const& rows,bst_node_t node){
      auto values=builder.GetNodeHistogram(node);
      // CUDA accumulation adds to caller-owned storage. Clear only this logical
      // slot before its first page, including an empty first page, matching the
      // reset performed by Metal's higher-level Build(page_index=0) wrapper.
      if(page_index==0)dh::LaunchN(values.size(),ctx.CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i){values[i]=GradientPairInt64{};});
      dh::device_vector<std::uint32_t> selected(rows);
      builder.BuildHistogram(&ctx,ellpack.GetDeviceEllpack(&ctx),features.DeviceAccessor(ctx.Device()),
                             integers.View(ctx.Device()).Values(),dh::ToSpan(selected),values);
      trace::Snapshot(&ctx,*quantizer,values,&observed);
    };
    constexpr bst_node_t parent=3,sibling=8,child=13;
    build(0,{0,1,2},parent);build(0,{0},sibling);
    observed.prefix_before=trace::Prefix(&builder);
    std::vector<bst_node_t> more;for(int node=16;node<257;++node)more.push_back(node);
    builder.AllocateHistograms(&ctx,more);observed.prefix_after=trace::Prefix(&builder);
    Check(observed.prefix_before==observed.prefix_after,"CUDA cache growth changed existing integer slots");
    Check(builder.SubtractionTrick(&ctx,parent,sibling,child),"CUDA subtraction unexpectedly unavailable");
    trace::Snapshot(&ctx,*quantizer,builder.GetNodeHistogram(child),&observed);
    trace::Root(&ctx,*quantizer,builder.GetNodeHistogram(child),&observed);
    build(1,{3},parent);build(2,{},parent);
    trace::Root(&ctx,*quantizer,builder.GetNodeHistogram(parent),&observed);
    build(0,{},parent);build(1,{1},parent);
    for(int node:{0,1,2,5,6,7,10,11,12,15}){
      auto values=builder.GetNodeHistogram(node);for(auto v:Read(values.data(),values.size())){
        Check(v.GetQuantisedGrad()==17 && v.GetQuantisedHess()==-23,"CUDA untouched integer cache guard changed");trace::Integer(&observed.guards,v);
      }
    }
    auto untouched=gradients.Data()->ConstHostVector(),first=trace::Pairs(input.first);
    Check(!std::memcmp(untouched.data(),first.data(),first.size()*sizeof(GradientPair)),"CUDA histogram changed raw gradients");
    initialize(false);set_gradients(input.second);build(0,{0,1,2},parent);build(1,{3},parent);
    trace::Root(&ctx,*quantizer,builder.GetNodeHistogram(parent),&observed);
    report.Add(input,observed);fields+=observed.values.size()+observed.bins.size();
    if((index+1)%8==0){std::printf("CUDA cache %u/%u: actual common fields=%zu\n",index+1,hc::kCases,fields);std::fflush(stdout);}
  }
  inputs.Finish();report.Finish();
  cudaDeviceProp gpu{};Cuda(cudaGetDeviceProperties(&gpu,0));int driver=0,runtime=0;
  Cuda(cudaDriverGetVersion(&driver));Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};meta["suite"]=String("integer-cache-v1");meta["backend"]=String("cuda");meta["device"]=String(gpu.name);
  meta["status"]=String("completed actual observations; comparison remains");meta["cases"]=Integer(hc::kCases);meta["common_actual_fields"]=Integer(fields);
  meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"]=Integer(driver);meta["runtime"]=Integer(runtime);meta["cub_version"]=Integer(CUB_VERSION);meta["thrust_version"]=Integer(THRUST_VERSION);
  meta["input_file"]=String(input_path);meta["observation_file"]=String(output_path);
  meta["scope"]=String("Actual GHistIndex-to-ELLPACK constructor, production histogram build/cache allocation/subtraction/reset; caller clears first-page slot; integer histogram diagnostics use production reduction/finalizer primitives; single worker");
  meta["storage_adaptation"]=String("Metal16->257 materialized pairs versus CUDA16->257 histogram slots; backend-specific prefixes/guards; no claim that reserved CUDA capacity reallocates, no private Metal integer-buffer capture");
  std::string text;Json::Dump(meta,&text);std::ofstream stream{metadata_path};stream.exceptions(std::ios::badbit|std::ios::failbit);stream<<text<<'\n';
}
