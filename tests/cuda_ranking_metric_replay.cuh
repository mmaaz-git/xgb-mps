// Original linked rank_metric.cu consumers. Explicit state inputs only bridge
// raw Metal API parameters; intermediate outputs come from the actual consumers.
#pragma once
#include "../xgboost/src/common/algorithm.cuh"
#include "../xgboost/src/metric/rank_metric.h"
#include "ranking_metric_io.h"
namespace ranking_metric_replay {
namespace rm=ranking_metric;
using namespace xgboost;
template <typename Tag,typename Tag::Type Member> struct StateAccess {
  friend typename Tag::Type StateMember(Tag) { return Member; }
};
struct CacheParam { using Type=ltr::LambdaRankParam ltr::RankingCache::*;friend Type StateMember(CacheParam); };
struct NumPairs { using Type=std::size_t ltr::LambdaRankParam::*;friend Type StateMember(NumPairs); };
struct Sorted { using Type=HostDeviceVector<std::size_t> ltr::RankingCache::*;friend Type StateMember(Sorted); };
struct Inverse { using Type=linalg::Vector<double> ltr::NDCGCache::*;friend Type StateMember(Inverse); };
template struct StateAccess<CacheParam,&ltr::RankingCache::param_>;
template struct StateAccess<NumPairs,&ltr::LambdaRankParam::lambdarank_num_pair_per_sample>;
template struct StateAccess<Sorted,&ltr::RankingCache::sorted_idx_cache_>;
template struct StateAccess<Inverse,&ltr::NDCGCache::inv_idcg_>;
inline ltr::LambdaRankParam Param(rm::Spec s) {
  ltr::LambdaRankParam p;
  // UINT32_MAX is NumPair's default sentinel, not an explicit full list. Mean
  // mode gives the exact untruncated TopK supplied by this raw-score corpus;
  // the public metric default remains a separate lifecycle comparison.
  p.UpdateAllowUnknown(Args{{"lambdarank_pair_method",s.topk==~0u ? "mean" : "topk"},
      {"lambdarank_num_pair_per_sample",std::to_string(s.topk==~0u ? 1u : std::max(1u,s.topk))},
      {"ndcg_exp_gain",s.kind==1 ? "true" : "false"}});
  return p;
}
inline void RawTopK(ltr::RankingCache* cache,rm::Spec s) {
  if (!s.topk) (cache->*StateMember(CacheParam{})).*StateMember(NumPairs{})=0;
  rm::Require(cache->Param().TopK()==s.topk,"CUDA metric TopK differs from supplied raw input");
}
inline rm::Observation Observe(Context const& ctx,rm::Input const& in) {
  auto s=in.spec;rm::Observation out;
  std::vector<float> p(s.rows),y(s.family==rm::kSort ? 0u : s.rows);
  for (unsigned i=0;i<s.rows;++i) { p[i]=in.predictions[i*s.stride];if (!y.empty()) y[i]=in.labels[i*s.stride]; }
  HostDeviceVector<float> predictions{p,ctx.Device()};
  if (s.family==rm::kSort) {
    HostDeviceVector<unsigned> ptr{in.ptr,ctx.Device()};HostDeviceVector<std::size_t> order(s.rows,0,ctx.Device());
    common::SegmentedArgSort<false,true>(&ctx,predictions.ConstDeviceSpan(),ptr.ConstDeviceSpan(),order.DeviceSpan());
    ::Cuda(cudaDeviceSynchronize());
    auto const& h=order.ConstHostVector();out.order.assign(h.begin(),h.end());
    rm::Require(ptr.ConstHostVector()==in.ptr,"CUDA sort modified query boundaries");
  } else {
    MetaInfo info;info.num_row_=s.rows;info.group_ptr_=in.ptr;
    info.labels.Reshape(s.rows,1);info.labels.Data()->HostVector()=y;info.labels.SetDevice(ctx.Device());
    info.weights_.HostVector()=in.weights;info.weights_.SetDevice(ctx.Device());
    auto param=Param(s);
    if (!s.topk) {
      bool rejected=false;
      try { auto invalid=param;invalid.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample","0"}}); }
      catch (dmlc::Error const& error) {
        rm::Require(std::string(error.what()).find("lambdarank_num_pair_per_sample")!=std::string::npos,
                    "Unexpected CUDA top-k parameter rejection");rejected=true;
      }
      rm::Require(rejected,"CUDA zero-topk parameter guard did not execute");out.guards|=1;
    }
    std::shared_ptr<ltr::RankingCache> base;
    metric::PackedReduceResult result;
    if (rm::NDCG(s)) {
      std::shared_ptr<ltr::NDCGCache> cache;
      try { cache=std::make_shared<ltr::NDCGCache>(&ctx,info,param); }
      catch (dmlc::Error const& error) {
        rm::Require(rm::NoGroups(s) && std::string(error.what()).find("d_group_ptr.size()")!=std::string::npos,
                    "Unexpected CUDA NDCG cache guard");out.guards|=2;
      }
      rm::Require(bool(cache)==!rm::NoGroups(s),"CUDA zero-group NDCG guard did not execute");
      if (cache) {
        RawTopK(cache.get(),s);
        // The raw Metal score API accepts supplied inverse-IDCG, including NaN
        // and nonpositive values. Inject these *inputs*, not expected results.
        (cache.get()->*StateMember(Inverse{})).Data()->HostVector()=in.inverse;
        result=metric::cuda_impl::NDCGScore(&ctx,info,predictions,s.minus,cache);
        auto reducer=cache->Dcg(&ctx);out.reducer=::ReadDevice(reducer.Values().data(),reducer.Size());base=cache;
      }
    } else if (s.kind==3) {
      auto cache=std::make_shared<ltr::MAPCache>(&ctx,info,param);RawTopK(cache.get(),s);
      result=metric::cuda_impl::MAPScore(&ctx,info,predictions,s.minus,cache);
      auto reducer=cache->Map(&ctx),prefix=cache->NumRelevant(&ctx);
      out.reducer=::ReadDevice(reducer.data(),reducer.size());out.prefix=::ReadDevice(prefix.data(),prefix.size());base=cache;
    } else {
      auto cache=std::make_shared<ltr::PreCache>(&ctx,info,param);RawTopK(cache.get(),s);
      result=metric::cuda_impl::PreScore(&ctx,info,predictions,cache);
      auto reducer=cache->Pre(&ctx);out.reducer=::ReadDevice(reducer.data(),reducer.size());base=cache;
    }
    if (base) {
      out.common={result.Residue(),result.Weights()};
      auto const& sorted=(base.get()->*StateMember(Sorted{})).ConstHostVector();
      out.order.assign(sorted.begin(),sorted.end());
    }
    ::Cuda(cudaDeviceSynchronize());
    auto equal=[](auto const& a,auto const& b) {
      return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)));
    };
    rm::Require(equal(info.labels.Data()->ConstHostVector(),y) && equal(info.weights_.ConstHostVector(),in.weights) &&
                info.group_ptr_==in.ptr,"CUDA ranking metric modified labels/weights/groups");
  }
  auto const& h=predictions.ConstHostVector();
  rm::Require(h.size()==p.size() && (p.empty() || !std::memcmp(h.data(),p.data(),p.size()*sizeof(float))),
              "CUDA ranking metric modified predictions");
  return out;
}
inline int Run(char const* input,char const* output) {
  auto metadata=std::string(output)+".json";rm::Require(!std::filesystem::exists(metadata),"Ranking metric metadata already exists");
  int count=0;::Cuda(cudaGetDeviceCount(&count));rm::Require(count>0,"Actual NVIDIA GPU required");
  ::Cuda(cudaSetDevice(0));auto ctx=Context{}.MakeCUDA(0);ctx.Init({{"device","cuda:0"},{"nthread","1"}});
  rm::InputReader reader(input);rm::Writer writer(output,rm::kCuda);
  unsigned topk_guards=0,group_guards=0;std::size_t rows=0;
  for (unsigned i=0;i<rm::kCases;++i) {
    auto in=reader.Next();auto out=Observe(ctx,in);writer.Add(in,out);rows+=in.spec.rows;
    topk_guards+=bool(out.guards&1);group_guards+=bool(out.guards&2);
    if ((i+1)%64==0 || i+1==rm::kCases) {
      std::printf("CUDA ranking metrics %u/%u; %zu rows, %u topk and %u group guards\n",i+1,rm::kCases,rows,topk_guards,group_guards);
      std::fflush(stdout);
    }
  }
  reader.Finish();writer.Finish();cudaDeviceProp gpu{};::Cuda(cudaGetDeviceProperties(&gpu,0));
  int driver=0,runtime=0;::Cuda(cudaDriverGetVersion(&driver));::Cuda(cudaRuntimeGetVersion(&runtime));
  Json meta{Object{}};meta["suite"]=String("ranking-metric-v1");meta["backend"]=String("cuda");
  meta["status"]=String("completed-observations; comparison remains");
  meta["input_file"]=String(input);meta["observation_file"]=String(output);
  meta["device"]=String(gpu.name);meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"]=Integer(driver);meta["runtime"]=Integer(runtime);meta["cases"]=Integer(rm::kCases);meta["rows"]=Integer(rows);
  meta["cub_version"]=Integer(CUB_VERSION);meta["thrust_version"]=Integer(THRUST_VERSION);
  meta["topk_guards"]=Integer(topk_guards);meta["ndcg_group_guards"]=Integer(group_guards);
  meta["scope"]=String("Original linked score consumers and segmented sort. Raw topk0 uses explicit cache-state injection after observing the public parameter guard. Supplied NDCG inverse-IDCG is an input. Edge11 compares a separate actual Metal call with standard discounts; original custom-discount call retained only on Metal. Zero-group NDCG records actual constructor rejection, no fabricated score. Reducer/prefix/order are observed producer buffers. Full cache and public lifecycle corpora remain separate work.");
  std::string text;Json::Dump(meta,&text);std::ofstream stream{metadata};stream.exceptions(std::ios::badbit|std::ios::failbit);
  stream<<text<<'\n';return 0;
}
}  // namespace ranking_metric_replay
