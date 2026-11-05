// Grouped PR calls the linked CUDA library directly. A separate namespace copy
// prepares observation boundaries for the binary/multi-output comparison; its
// generated source and arithmetic-preservation record are retained by CI.
#include <cuda_runtime.h>
#include <cub/version.cuh>
#include <thrust/version.h>
#include <cstdio>
#include <fstream>
#include "../xgboost/src/metric/auc.h"
#include "grouped_pr_contract_io.h"
#include "xgboost/json.h"
#include "metric_registry_probe.h"
#include "auc_capture.cuh"
#include "auc_probe.inc"

namespace gp = grouped_pr;
using namespace xgboost;
void Cuda(cudaError_t code) {
  if (code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
#include "cuda_auc_replay.cuh"
std::vector<double> Observe(Context const& ctx,std::vector<float> const& p,
                            std::vector<float> const& y,std::vector<std::uint32_t> const& ptr,
                            std::vector<float> const& weights) {
  MetaInfo info;
  info.num_row_=p.size();info.group_ptr_=ptr;info.labels.Reshape(y.size(),1);
  info.labels.Data()->HostVector()=y;info.labels.SetDevice(ctx.Device());
  info.weights_.HostVector()=weights;info.weights_.SetDevice(ctx.Device());
  HostDeviceVector<float> predictions{p,ctx.Device()};
  std::shared_ptr<metric::cuda_impl::DeviceAUCCache> cache;
  auto result=metric::cuda_impl::RankingPRAUC(&ctx,predictions.ConstDeviceSpan(),info,&cache);
  Cuda(cudaDeviceSynchronize());
  auto same=[](auto const& a,auto const& b) {
    return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)));
  };
  gp::Require(same(predictions.ConstHostVector(),p) && same(info.labels.Data()->ConstHostVector(),y) &&
              same(info.weights_.ConstHostVector(),weights) && info.group_ptr_==ptr,
              "CUDA grouped PR modified source inputs");
  return {result.first,double(result.second)};
}
int main(int argc,char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--metric-registry") {
      MetricRegistrySignature();
      return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--auc") return auc_replay::Run(argv[2], argv[3]);
    gp::Require(argc==4 && std::string(argv[1])=="--grouped-pr",
                "Usage: cuda_auc_contracts (--grouped-pr|--auc) input-only.bin new-cuda.bin");
    std::string metadata=std::string(argv[3])+".json";
    gp::Require(!std::filesystem::exists(metadata),"AUC metadata already exists");
    int count=0;Cuda(cudaGetDeviceCount(&count));gp::Require(count>0,"Actual NVIDIA GPU required");
    Cuda(cudaSetDevice(0));auto ctx=Context{}.MakeCUDA(0);ctx.Init({{"device","cuda:0"},{"nthread","1"}});
    gp::InputReader reader(argv[2]);gp::Writer writer(argv[3],gp::kCuda);
    unsigned isolated=0,adapted=0,rejections=0;
    for (unsigned index=0;index<gp::kCases;++index) {
      auto in=reader.Next();auto compact=gp::Compact(in);gp::Observation out;
      if (in.spec.rows && gp::Nonempty(in).size()!=in.spec.groups) ++adapted;
      try {
        out.common=Observe(ctx,compact.predictions,compact.labels,compact.ptr,compact.weights);
      } catch (dmlc::Error const& error) {
        gp::Require(in.spec.family==2 && std::string(error.what()).find(
            "PR-AUC supports only binary relevance for learning to rank.")!=std::string::npos,
            "Unexpected CUDA PR error; not a successful label rejection");
        out.rejected=1;++rejections;
      }
      gp::Require(out.rejected==unsigned(in.spec.family==2),"CUDA failed to reject invalid PR label");
      if (!out.rejected) {
        for (std::size_t g=0;g+1<compact.ptr.size();++g) {
          auto begin=compact.ptr[g],end=compact.ptr[g+1];
          std::vector<float> p(compact.predictions.begin()+begin,compact.predictions.begin()+end);
          std::vector<float> y(compact.labels.begin()+begin,compact.labels.begin()+end),w;
          if (!compact.weights.empty()) w.push_back(compact.weights[g]);
          auto pair=Observe(ctx,p,y,{0,end-begin},w);
          out.isolated.insert(out.isolated.end(),pair.begin(),pair.end());++isolated;
        }
      }
      writer.Add(in,out);
      if ((index+1)%24==0 || index+1==gp::kCases) {
        std::printf("CUDA grouped PR %u/%u; isolated queries=%u, empty-group adaptations=%u\n",
                     index+1,gp::kCases,isolated,adapted);std::fflush(stdout);
      }
    }
    reader.Finish();writer.Finish();
    cudaDeviceProp gpu{};Cuda(cudaGetDeviceProperties(&gpu,0));int driver=0,runtime=0;
    Cuda(cudaDriverGetVersion(&driver));Cuda(cudaRuntimeGetVersion(&runtime));
    Json meta{Object{}};meta["suite"]=String("grouped-pr-v1");meta["backend"]=String("cuda");
    meta["device"]=String(gpu.name);meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
    meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
    meta["driver"]=Integer(driver);meta["runtime"]=Integer(runtime);meta["cases"]=Integer(gp::kCases);
    meta["cub_version"]=Integer(CUB_VERSION);meta["thrust_version"]=Integer(THRUST_VERSION);
    meta["isolated_query_calls"]=Integer(isolated);meta["empty_group_adaptations"]=Integer(adapted);
    meta["observed_label_rejections"]=Integer(rejections);
    meta["scope"]=String("Actual linked cuda_impl::RankingPRAUC; contiguous packing and explicit empty-query removal. Aggregate compacted and isolated-query calls are separate actual observations. Original nonempty input layouts with empty queries are not executed on CUDA.");
    std::string text;Json::Dump(meta,&text);std::ofstream stream{metadata};
    stream.exceptions(std::ios::badbit|std::ios::failbit);stream<<text<<'\n';
  } catch (std::exception const& error) {std::fprintf(stderr,"FAIL CUDA AUC: %s\n",error.what());return 1;}
}
