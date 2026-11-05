// Original CUDA coordinate-reduction/residual methods on every defined action.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include "cuda_linear_source.h"
#include "linear_contract_io.h"
#include "linear_registry_probe.h"
#include "xgboost/json.h"

namespace lc = linear_contracts;
using namespace xgboost;
void Cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::vector<GradientPair> Pairs(std::vector<float> const& words) {
  std::vector<GradientPair> pairs; pairs.reserve(words.size()/2);
  for (std::size_t i=0;i<words.size();i+=2) pairs.emplace_back(words[i],words[i+1]);
  return pairs;
}
std::vector<float> Flatten(std::vector<GradientPair> const& pairs) {
  std::vector<float> out;out.reserve(pairs.size()*2);
  for (auto pair:pairs) {out.push_back(pair.GetGrad());out.push_back(pair.GetHess());}
  return out;
}
std::vector<GradientPair> Snapshot(cuda_linear_trace::Probe* probe,bool guards) {
  auto& device=probe->Residuals();
  std::vector<GradientPair> out(device.size()+(guards?2:0),GradientPair{-999,998});
  if (!device.empty()) Cuda(cudaMemcpy(out.data()+(guards?1:0),device.data().get(),
                                     device.size()*sizeof(GradientPair),cudaMemcpyDeviceToHost));
  return out;
}

int main(int argc,char** argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--linear-registry") {LinearRegistrySignature();return 0;}
    lc::Require(argc==3,"Usage: cuda_linear_contracts input-only.bin new-cuda.bin");
    std::string metadata_path=std::string(argv[2])+".json";
    lc::Require(!std::filesystem::exists(metadata_path),"Linear metadata already exists");
    int devices=0;Cuda(cudaGetDeviceCount(&devices));
    lc::Require(devices>0,"Actual NVIDIA GPU required; no CPU fallback or skipped pass");
    Cuda(cudaSetDevice(0));auto ctx=Context{}.MakeCUDA(0);ctx.Init({{"device","cuda:0"},{"nthread","1"}});
    lc::InputReader input(argv[1]);lc::Writer output(argv[2],lc::kCuda);
    std::size_t sum_fields=0,residual_fields=0,excluded_residuals=0,rejections=0;
    for (unsigned index=0;index<lc::kCases;++index) {
      auto in=input.NextCase();auto s=in.spec;output.Start(in);
      std::unique_ptr<cuda_linear_trace::Probe> probe;
      std::vector<Entry> entries;
      if (s.family<lc::kInvalidFeatureLow) {
        probe=std::make_unique<cuda_linear_trace::Probe>(&ctx);
        probe->SetRows(s.rows);probe->Offsets().assign(in.offsets.begin(),in.offsets.end());
        for (std::size_t i=0;i<in.rows.size();++i) entries.emplace_back(in.rows[i],in.entries[i]);
        probe->Data().resize(entries.size());probe->Residuals().resize(std::size_t{s.rows}*s.groups);
        if (!entries.empty()) Cuda(cudaMemcpy(probe->Data().data().get(),entries.data(),
                                             entries.size()*sizeof(Entry),cudaMemcpyHostToDevice));
      }
      for (auto action:lc::Actions(s)) {
        auto state=input.NextStep();output.StartStep(state);
        lc::Observation observed;observed.status=lc::Status(lc::kCuda,action);
        if (action.domain==2) {
          ++rejections;output.EndStep(observed);continue;
        }
        // Operation replay: reload the exact portable input before each action.
        // In particular this does not pretend that a duplicate-column write ran.
        auto source=Pairs(state);probe->SetGradients(source);
        for (unsigned group=0;group<s.groups;++group) {
          if (action.gradient) {
            auto pair=action.feature<0 ? probe->GetBiasGradient(group,s.groups)
                                      : probe->GetGradient(group,s.groups,action.feature);
            observed.gradients.push_back(pair.GetGrad());observed.gradients.push_back(pair.GetHess());
          }
          if (action.update && action.domain==0) {
            if (action.feature<0) probe->UpdateBiasResidual(action.delta,group,s.groups);
            else probe->UpdateResidual(action.delta,group,s.groups,action.feature);
          }
        }
        if (action.copy && action.domain==0) observed.residuals=Flatten(Snapshot(probe.get(),s.family==lc::kMain));
        if (action.domain==1) ++excluded_residuals;
        if (!action.update || action.domain==1) {
          auto unchanged=Snapshot(probe.get(),false);
          lc::Require(unchanged.size()==source.size() &&
                      (source.empty() || !std::memcmp(unchanged.data(),source.data(),source.size()*sizeof(GradientPair))),
                      "CUDA reduction modified its input state");
        }
        sum_fields+=observed.gradients.size();residual_fields+=observed.residuals.size();
        output.EndStep(observed);
      }
      if (probe) {
        if (!entries.empty()) {
          std::vector<Entry> actual(entries.size());
          Cuda(cudaMemcpy(actual.data(),probe->Data().data().get(),actual.size()*sizeof(Entry),cudaMemcpyDeviceToHost));
          lc::Require(!std::memcmp(actual.data(),entries.data(),entries.size()*sizeof(Entry)),"CUDA changed CSC inputs");
        }
        lc::Require(probe->Offsets().size()==in.offsets.size() &&
                    std::equal(probe->Offsets().begin(),probe->Offsets().end(),in.offsets.begin()),"CUDA changed CSC offsets");
      }
      if ((index+1)%12==0 || index+1==lc::kCases) {
        std::printf("CUDA linear %u/%u: sums=%zu residual fields=%zu\n",index+1,lc::kCases,sum_fields,residual_fields);std::fflush(stdout);
      }
    }
    input.Finish();output.Finish();
    cudaDeviceProp gpu{};Cuda(cudaGetDeviceProperties(&gpu,0));int driver=0,runtime=0;
    Cuda(cudaDriverGetVersion(&driver));Cuda(cudaRuntimeGetVersion(&runtime));
    Json meta{Object{}};meta["suite"]=String("linear-v1");meta["backend"]=String("cuda");
    meta["status"]=String("completed actual operation observations with explicit exclusions; comparison remains");
    meta["device"]=String(gpu.name);meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
    meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
    meta["driver"]=Integer(driver);meta["runtime"]=Integer(runtime);meta["cub_version"]=Integer(CUB_VERSION);meta["thrust_version"]=Integer(THRUST_VERSION);
    meta["cases"]=Integer(lc::kCases);meta["steps"]=Integer(lc::kSteps);
    meta["actual_sum_fields"]=Integer(sum_fields);meta["actual_residual_fields_including_copy_guards"]=Integer(residual_fields);
    meta["duplicate_residual_exclusions"]=Integer(excluded_residuals);meta["unsafe_rejection_exclusions"]=Integer(rejections);
    meta["input_file"]=String(argv[1]);meta["observation_file"]=String(argv[2]);
    meta["scope"]=String("Unchanged GPUCoordinateUpdater method bodies compiled with hidden visibility and isolated registration; private-state adapter loads explicit CSC/input states; original UpdateGpair uploads; no LazyInitDevice/full updater/selector or persistent sequence claim");
    meta["domains"]=String("Duplicate-column gradient reductions executed; concurrent duplicate residual writes excluded. Unsafe feature/row rejection calls not executed. Supplied empty states support zero-length reduction methods, not the LazyInitDevice empty-input early return.");
    std::string text;Json::Dump(meta,&text);std::ofstream stream{metadata_path};stream.exceptions(std::ios::badbit|std::ios::failbit);stream<<text<<'\n';
  } catch (std::exception const& error) {std::fprintf(stderr,"FAIL CUDA linear: %s\n",error.what());return 1;}
}
