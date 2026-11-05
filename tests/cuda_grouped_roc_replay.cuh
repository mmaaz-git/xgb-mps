// Actual linked RankingAUC calls. Never execute mixed groups with zero pairs:
// their result slots are unwritten by the source's transform-output iterator.
#pragma once
#include "grouped_roc_contract_io.h"
namespace grouped_roc_replay {
namespace gr=grouped_roc;
using namespace xgboost;
inline std::vector<double> Observe(Context const& ctx,std::vector<float> const& p,
    std::vector<float> const& y,std::vector<std::uint32_t> const& ptr,
    std::vector<float> const& weights,bool expect_limit=false) {
  MetaInfo info;
  info.num_row_=p.size();info.group_ptr_=ptr;info.labels.Reshape(y.size(),1);
  info.labels.Data()->HostVector()=y;info.labels.SetDevice(ctx.Device());
  info.weights_.HostVector()=weights;info.weights_.SetDevice(ctx.Device());
  HostDeviceVector<float> predictions{p,ctx.Device()};
  std::shared_ptr<metric::cuda_impl::DeviceAUCCache> cache;
  std::vector<double> result;
  bool guarded=false;
  try {
    auto value=metric::cuda_impl::RankingAUC(&ctx,predictions.ConstDeviceSpan(),info,&cache);
    result={value.first,double(value.second)};
  } catch (dmlc::Error const& error) {
    auto message=std::string(error.what());
    gr::Require(expect_limit && message.find("n_threads")!=std::string::npos &&
                message.find("2147483647")!=std::string::npos,
                "Unexpected CUDA grouped ROC error; not the pair-count guard");
    guarded=true;
  }
  gr::Require(guarded==expect_limit,"CUDA grouped ROC pair-count guard did not execute");
  ::Cuda(cudaDeviceSynchronize());
  auto same=[](auto const& a,auto const& b) {
    return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)));
  };
  gr::Require(same(predictions.ConstHostVector(),p) && same(info.labels.Data()->ConstHostVector(),y) &&
              same(info.weights_.ConstHostVector(),weights) && info.group_ptr_==ptr,
              "CUDA grouped ROC modified source inputs");
  return result;
}
inline int Run(char const* input,char const* output) {
  auto metadata=std::string(output)+".json";
  gr::Require(!std::filesystem::exists(metadata),"Grouped ROC metadata already exists");
  int count=0;::Cuda(cudaGetDeviceCount(&count));gr::Require(count>0,"Actual NVIDIA GPU required");
  ::Cuda(cudaSetDevice(0));auto ctx=Context{}.MakeCUDA(0);ctx.Init({{"device","cuda:0"},{"nthread","1"}});
  gr::InputReader reader(input);gr::Writer writer(output,gr::kCuda);
  unsigned isolated=0,adapted=0,removed=0,guards=0,short_calls=0;
  for (unsigned i=0;i<gr::kCases;++i) {
    auto in=reader.Next();auto compact=gr::Compact(in);gr::Observation out;
    auto retained=compact.ptr.size()-1;
    if (retained!=in.spec.groups) { ++adapted;removed+=in.spec.groups-retained; }
    if (in.spec.family==gr::kLarge) {
      Observe(ctx,compact.predictions,compact.labels,compact.ptr,compact.weights,true);
      out.guarded=1;++guards;
    } else {
      out.common=Observe(ctx,compact.predictions,compact.labels,compact.ptr,compact.weights);
      for (unsigned g=0;g<in.spec.groups;++g) {
        std::vector<float> p,y,w;
        for (unsigned row=in.ptr[g];row<in.ptr[g+1];++row) {
          p.push_back(in.predictions[gr::Offset(in.spec)+row*gr::Stride(in.spec)]);
          y.push_back(in.labels[gr::Offset(in.spec)+row*gr::Stride(in.spec)]);
        }
        if (!in.weights.empty()) w.push_back(in.weights[g]);
        auto pair=Observe(ctx,p,y,{0,unsigned(p.size())},w);
        out.isolated.insert(out.isolated.end(),pair.begin(),pair.end());
        ++isolated;short_calls+=p.size()<3;
      }
    }
    writer.Add(in,out);
    if ((i+1)%24==0 || i+1==gr::kCases) {
      std::printf("CUDA grouped ROC %u/%u; %u isolated queries, %u pair-count guards\n",
                  i+1,gr::kCases,isolated,guards);std::fflush(stdout);
    }
  }
  reader.Finish();writer.Finish();
  cudaDeviceProp gpu{};::Cuda(cudaGetDeviceProperties(&gpu,0));int driver=0,runtime=0;
  ::Cuda(cudaDriverGetVersion(&driver));::Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};meta["suite"]=String("grouped-roc-v1");meta["backend"]=String("cuda");
  meta["device"]=String(gpu.name);meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"]=Integer(driver);meta["runtime"]=Integer(runtime);meta["cases"]=Integer(gr::kCases);
  meta["cub_version"]=Integer(CUB_VERSION);meta["thrust_version"]=Integer(THRUST_VERSION);
  meta["isolated_query_calls"]=Integer(isolated);meta["isolated_short_queries"]=Integer(short_calls);
  meta["adapted_records"]=Integer(adapted);meta["removed_zero_pair_groups"]=Integer(removed);
  meta["observed_pair_count_guards"]=Integer(guards);
  meta["scope"]=String("Actual linked RankingAUC. Mixed calls remove only zero-pair groups; all original queries also run in isolation, including length0/1/2. Million-row inputs must hit the actual int32 pair-count guard. No uninitialized aggregate slots are read or synthesized.");
  std::string text;Json::Dump(meta,&text);std::ofstream stream{metadata};
  stream.exceptions(std::ios::badbit|std::ios::failbit);stream<<text<<'\n';return 0;
}
}  // namespace grouped_roc_replay
