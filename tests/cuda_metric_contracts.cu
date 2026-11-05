// Complete metric corpus: actual CUDA source policies/reducers and separately
// identified source-expression probes. No reference outputs enter GPU inputs.
#include <cuda_runtime.h>
#include <cub/version.cuh>
#include <thrust/version.h>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include "cuda_metric_source.h"
#include "metric_contract_io.h"
#include "metric_registry_probe.h"
#include "../xgboost/src/common/device_helpers.cuh"
#include "xgboost/json.h"

namespace mc=metric_contracts;
namespace metric=xgboost::metric;
namespace common=xgboost::common;
using xgboost::Context;
using xgboost::HostDeviceVector;
void Cuda(cudaError_t s) { if (s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
std::array<double,2> Pair(metric::PackedReduceResult result) { return {result.Residue(),result.Weights()}; }
std::string Parameter(mc::Spec s) {
  std::ostringstream out; out<<std::setprecision(std::numeric_limits<float>::max_digits10)<<mc::Bits<float>(s.parameter);
  return out.str();
}
template <typename Fn>
void Terms(Context const* ctx,mc::Spec s,Fn loss,mc::Observation* observed) {
  dh::device_vector<float> storage(s.fields); auto out=dh::ToSpan(storage);
  dh::LaunchN(s.fields/2,ctx->CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i) {
    auto term=loss(i,i/s.targets,i%s.targets);
    out[2*i]=std::get<0>(term); out[2*i+1]=std::get<1>(term);
  });
  Cuda(cudaDeviceSynchronize()); observed->actual.resize(s.fields);
  if (s.fields) Cuda(cudaMemcpy(observed->actual.data(),out.data(),out.size_bytes(),cudaMemcpyDeviceToHost));
  observed->reductions=mc::ReductionReferences(observed->actual);
}
struct Data {
  xgboost::MetaInfo info;
  HostDeviceVector<float> predictions,upper,alphas;
  Data(Context const* ctx,mc::Input const& in)
      : predictions{in.predictions,ctx->Device()},upper{in.upper,ctx->Device()},alphas{in.alphas,ctx->Device()} {
    info.num_row_=in.spec.rows;
    unsigned targets=in.spec.kind>=11 && in.spec.family==1 ? 1 : in.spec.targets;
    if (in.spec.family==5) targets=1;
    info.labels.Reshape(in.spec.rows,targets); info.labels.Data()->HostVector()=in.labels;
    info.labels.SetDevice(ctx->Device()); info.weights_.HostVector()=in.weights;
    info.weights_.SetDevice(ctx->Device());
  }
};
template <typename Fn>
void Reduce(Context const* ctx,mc::Spec s,Data const& data,Fn loss,mc::Observation* out) {
  Terms(ctx,s,loss,out);
  // The source's actual FP32-term -> PackedReduceResult -> Thrust reduction.
  out->sums=Pair(metric::Reduce(ctx,data.info,loss,s.levels));
}
template <typename Policy>
void Ordinary(Context const* ctx,mc::Spec s,Data const& d,Policy policy,mc::Observation* out) {
  auto y=d.info.labels.View(ctx->Device()); auto p=d.predictions.ConstDeviceSpan();
  auto w=common::MakeOptionalWeights(ctx->Device(),d.info.weights_);
  auto loss=[=] XGBOOST_DEVICE(std::size_t i,std::size_t row,std::size_t target) {
    float wt=w[row]; float residue=policy.EvalRow(y(row,target),p[i]);
    return std::make_tuple(residue*wt,wt);
  };
  Reduce(ctx,s,d,loss,out);
}
template <typename Policy>
void Multi(Context const* ctx,mc::Spec s,Data const& d,mc::Observation* out) {
  metric::MultiClassMetricsReduction<Policy> reducer;
  if (s.family!=5) {
    auto p=d.predictions.ConstDeviceSpan(), y=d.info.labels.Data()->ConstDeviceSpan();
    auto w=common::MakeOptionalWeights(ctx->Device(),d.info.weights_);
    auto loss=[=] XGBOOST_DEVICE(std::size_t i,std::size_t,std::size_t) {
      float wt=w[i]; auto label=static_cast<int>(y[i]);
      return std::make_tuple(Policy::EvalRow(label,p.data()+i*s.targets,s.targets)*wt,wt);
    };
    Terms(ctx,s,loss,out);
  }
  try { out->sums=Pair(reducer.Reduce(ctx,s.targets,d.info.weights_,*d.info.labels.Data(),d.predictions)); }
  catch (dmlc::Error const& error) {
    mc::Require(s.family==5 && std::string(error.what()).find("MultiClassEvaluation: label must be in")!=std::string::npos,
                "Unexpected CUDA multiclass exception");
    out->status=1;
  }
}
template <typename Policy>
void Survival(Context const* ctx,Data const& d,Policy policy,mc::Observation* out) {
  metric::ElementWiseSurvivalMetricsReduction<Policy> reducer; reducer.Configure(policy);
  out->sums=Pair(reducer.Reduce(ctx,d.info.weights_,*d.info.labels.Data(),d.upper,d.predictions));
}
template <typename Distribution>
void AFT(Context const* ctx,Data const& d,mc::Spec s,mc::Observation* out) {
  metric::EvalAFTNLogLik<Distribution> policy;
  policy.Configure({{"aft_loss_distribution_scale",Parameter(s)}});
  Survival(ctx,d,policy,out);
}
mc::Observation Observe(Context const* ctx,mc::Input const& in) {
  auto s=in.spec; mc::Observation out;
  if (s.family==5 && s.mode>=2) {
    // Frozen NUM-OP-36: nonfinite/out-of-int32 casts are Metal guard extensions
    // outside the CUDA contract. Retain the case without executing the cast.
    out.status=2; return out;
  }
  if (s.family==0) {
    dh::device_vector<float> values(in.predictions), storage(s.fields);
    auto v=dh::ToSpan(values), result=dh::ToSpan(storage);
    dh::LaunchN(s.rows,ctx->CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i) {
      float x=v[i]; result[5*i]=std::log(x); result[5*i+1]=std::log1p(x);
      result[5*i+2]=std::exp(x); result[5*i+3]=std::sqrt(x); result[5*i+4]=common::LogGamma(x);
    });
    Cuda(cudaDeviceSynchronize()); out.actual.resize(s.fields);
    Cuda(cudaMemcpy(out.actual.data(),result.data(),result.size_bytes(),cudaMemcpyDeviceToHost));
    return out;
  }
  Data d{ctx,in};
  if (s.family==6) {
    // Empty MetricSums returns before inspecting its parameters. Observe the
    // actual source empty reducer; no loss/parameter expression is evaluated.
    if (s.kind==11) Multi<metric::EvalMultiLogLoss>(ctx,s,d,&out);
    else if (s.kind==12) Multi<metric::EvalMatchError>(ctx,s,d,&out);
    else out.sums=Pair(metric::Reduce(ctx,d.info,
        [] XGBOOST_DEVICE(std::size_t,std::size_t,std::size_t) { return std::make_tuple(0.f,0.f); }));
  } else if (s.family==3 || s.family==4) {
    if (s.accuracy) Survival(ctx,d,metric::EvalIntervalRegressionAccuracy{},&out);
    else if (s.kind==0) AFT<common::NormalDistribution>(ctx,d,s,&out);
    else if (s.kind==1) AFT<common::LogisticDistribution>(ctx,d,s,&out);
    else AFT<common::ExtremeDistribution>(ctx,d,s,&out);
  } else if (s.family==2) {
    auto y=d.info.labels.View(ctx->Device()); auto p=d.predictions.ConstDeviceSpan();
    auto a=d.alphas.ConstDeviceSpan(); auto w=common::MakeOptionalWeights(ctx->Device(),d.info.weights_);
    // These lambdas are the source expressions in QuantileError/ExpectileError;
    // they are separately evaluated probes, not captures of private temporaries.
    if (s.kind==0) {
      auto loss=[=] XGBOOST_DEVICE(std::size_t i,std::size_t,std::size_t) {
        auto row=i/(s.targets*s.levels),level=i/s.targets%s.levels,target=i%s.targets;
        auto diff=y(row,target)-p[i]; float sign=diff>=0.f;
        auto residue=(a[level]*sign*diff)-(1.f-a[level])*(1.f-sign)*diff;
        auto wt=w[row]; return std::make_tuple(residue*wt,wt);
      };
      Reduce(ctx,s,d,loss,&out);
    } else {
      auto loss=[=] XGBOOST_DEVICE(std::size_t i,std::size_t,std::size_t) {
        auto row=i/(s.targets*s.levels),level=i/s.targets%s.levels,target=i%s.targets;
        auto diff=p[i]-y(row,target); auto scale=diff>=0.f ? (1.f-a[level]) : a[level];
        auto wt=w[row]; return std::make_tuple(scale*diff*diff*wt,wt);
      };
      Reduce(ctx,s,d,loss,&out);
    }
  } else if (s.kind==10) {
    auto y=d.info.labels.View(ctx->Device()); auto p=d.predictions.ConstDeviceSpan();
    auto w=common::MakeOptionalWeights(ctx->Device(),d.info.weights_); float slope=mc::Bits<float>(s.parameter);
    // Source expression in PseudoErrorLoss::Eval, with the original cast/order.
    auto loss=[=] XGBOOST_DEVICE(std::size_t i,std::size_t row,std::size_t target) {
      float wt=w[row]; auto a=y(row,target)-p[i];
      auto v=common::Sqr(slope)*(std::sqrt((1+common::Sqr(a/slope)))-1)*wt;
      return std::make_tuple(v,wt);
    };
    Reduce(ctx,s,d,loss,&out);
  } else {
    switch (s.kind) {
      case 0: Ordinary(ctx,s,d,metric::EvalRowRMSE{},&out); break;
      case 1: Ordinary(ctx,s,d,metric::EvalRowRMSLE{},&out); break;
      case 2: Ordinary(ctx,s,d,metric::EvalRowMAE{},&out); break;
      case 3: Ordinary(ctx,s,d,metric::EvalRowMAPE{},&out); break;
      case 4: Ordinary(ctx,s,d,metric::EvalRowLogLoss{},&out); break;
      case 5: { metric::EvalError p{Parameter(s).c_str()};
        mc::Require(mc::Bits<std::uint32_t>(p.MetalParameter())==s.parameter,"Error parameter lost bits");
        Ordinary(ctx,s,d,p,&out); break; }
      case 6: Ordinary(ctx,s,d,metric::EvalPoissonNegLogLik{},&out); break;
      case 7: Ordinary(ctx,s,d,metric::EvalGammaDeviance{},&out); break;
      case 8: Ordinary(ctx,s,d,metric::EvalGammaNLogLik{},&out); break;
      case 9: { metric::EvalTweedieNLogLik p{Parameter(s).c_str()};
        mc::Require(mc::Bits<std::uint32_t>(p.MetalParameter())==s.parameter,"Tweedie parameter lost bits");
        Ordinary(ctx,s,d,p,&out); break; }
      case 11: Multi<metric::EvalMultiLogLoss>(ctx,s,d,&out); break;
      case 12: Multi<metric::EvalMatchError>(ctx,s,d,&out); break;
      default: throw std::runtime_error("Unknown metric policy");
    }
  }
  return out;
}
int main(int argc,char** argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--metric-registry") { MetricRegistrySignature(); return 0; }
    mc::Require(argc==3,"Usage: cuda_metric_contracts input-only.bin new-cuda-observations.bin");
    std::string metadata_path=std::string(argv[2])+".json";
    mc::Require(!std::filesystem::exists(metadata_path),"Metric metadata already exists");
    int devices=0; Cuda(cudaGetDeviceCount(&devices)); mc::Require(devices>0,"Actual NVIDIA GPU required; no skipped pass or CPU fallback");
    Cuda(cudaSetDevice(0)); auto ctx=Context{}.MakeCUDA(0); ctx.Init({{"device","cuda:0"},{"nthread","2"}});
    mc::InputReader input(argv[1]); mc::Writer report(argv[2],mc::kCuda);
    std::size_t fields=0,rejections=0,excluded=0;
    for (unsigned i=0; i<mc::kCases; ++i) {
      auto in=input.Next(); auto out=Observe(&ctx,in); report.Add(in,out);
      fields+=out.actual.size()+(in.spec.family && !out.status ? 2 : 0);
      rejections+=out.status==1; excluded+=out.status==2;
      if ((i+1)%20==0 || i+1==mc::kCases) {
        std::printf("CUDA metrics %u/%u: family=%u fields=%zu rejections=%zu outside_domain=%zu\n",
                    i+1,mc::kCases,in.spec.family,fields,rejections,excluded);
        std::fflush(stdout);
      }
    }
    input.Finish(); report.Finish();
    cudaDeviceProp gpu{}; Cuda(cudaGetDeviceProperties(&gpu,0));
    int driver=0,runtime=0; Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
    using namespace xgboost;
    Json meta{Object{}};
    meta["status"]=String("completed observations; comparison remains"); meta["suite"]=String("metric-v1");
    meta["backend"]=String("cuda"); meta["device"]=String(gpu.name);
    meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
    meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
    meta["driver"]=Integer(driver); meta["runtime"]=Integer(runtime); meta["cub_version"]=Integer(CUB_VERSION);
    meta["thrust_version"]=Integer(THRUST_VERSION); meta["cases"]=Integer(mc::kCases);
    meta["observed_fields"]=Integer(fields); meta["caught_label_rejections"]=Integer(rejections);
    meta["executed_cases"]=Integer(mc::kCases-excluded); meta["outside_domain_cases"]=Integer(excluded);
    meta["input_file"]=String(argv[1]); meta["observation_file"]=String(argv[2]);
    meta["scope"]=String("Actual source row policies and elementwise/multiclass/survival reducers; pseudo-Huber/quantile/expectile source-expression probes; separate primitive math observations; empty source reducer boundaries; no production metric registry changes or GPU input oracles");
    std::string text; Json::Dump(meta,&text); std::ofstream stream{metadata_path};
    stream.exceptions(std::ios::badbit|std::ios::failbit); stream<<text<<'\n';
    std::printf("CUDA metric observations completed: %u cases / %zu fields; comparison remains\n",mc::kCases,fields);
  } catch (std::exception const& e) { std::fprintf(stderr,"FAIL CUDA metric: %s\n",e.what()); return 1; }
}
