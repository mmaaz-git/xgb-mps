// NVIDIA execution of the shared P5-N01/08/13/14 witnesses. No Metal oracle is
// substituted for CUDA. Order-dependent differences are observations for review.
#include <cuda_runtime.h>
#include <cub/version.cuh>
#include <thrust/version.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/hist_util.cuh"
#include "../xgboost/src/common/quantile.cuh"
#include "../xgboost/src/common/stats.cuh"
#include "../xgboost/src/common/numeric.h"
#include "../xgboost/src/common/optional_weight.h"
#include "../xgboost/src/common/stats.h"
#include "stats_contract_io.h"
#include "../xgboost/src/common/linalg_op.h"
#include "linalg_contract_io.h"
#include "metric_registry_probe.h"
#include "xgboost/json.h"
#include "numerical_contract_fixtures.h"
#include "numerical_contract_report.h"

namespace fixture = numerical_contracts;
namespace common = xgboost::common;
using xgboost::Context;
using xgboost::HostDeviceVector;
using common::SketchContainer;
using common::SketchEntry;

void Check(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void Cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <typename T> std::vector<T> Read(T const* data, std::size_t size) {
  Cuda(cudaDeviceSynchronize());
  std::vector<T> result(size);
  if (size) Cuda(cudaMemcpy(result.data(), data, size*sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}
template <typename T> std::vector<T> Read(dh::device_vector<T> const& data) {
  return Read(data.data().get(), data.size());
}
std::vector<SketchEntry> Entries(std::vector<fixture::Entry> const& values) {
  std::vector<SketchEntry> result;
  for (auto e : values) result.emplace_back(e.rmin, e.rmax, e.wmin, e.value);
  return result;
}

void Threshold(Context const* ctx, fixture::Report& report, int exponent, unsigned variant) {
  auto q = fixture::MakeThreshold(exponent, variant);
  dh::device_vector<float> values(q.values), weights(q.weights);
  dh::device_vector<std::size_t> ptr(std::vector<std::size_t>{0, 2});
  auto alpha = thrust::make_constant_iterator(q.alpha);
  HostDeviceVector<float> result;
  common::SegmentedWeightedQuantile(ctx, alpha, ptr.cbegin(), ptr.cend(), values.cbegin(),
                                    values.cend(), weights.cbegin(), weights.cend(), &result);
  auto const& selected = result.ConstHostVector();
  Check(selected.size() == 1 && selected[0] == q.expected,
        "CUDA binary64 threshold selected the wrong observation");
  report.Case("weighted_threshold", 2, exponent, variant,
              {{"values", fixture::FloatWords(q.values)}, {"weights", fixture::FloatWords(q.weights)},
               {"alpha", fixture::FloatWords(&q.alpha, 1)}},
              {{"api_selected", fixture::FloatWords(selected)}});
}
void Quantile(Context const* ctx, fixture::Report& report, unsigned rows, int exponent) {
  auto q = fixture::MakeQuantile(rows, exponent);
  dh::device_vector<float> values(q.values), weights(q.weights);
  dh::device_vector<std::size_t> ptr(std::vector<std::size_t>{0, rows});
  auto alpha = thrust::make_constant_iterator(1.0);
  HostDeviceVector<float> result;
  common::SegmentedWeightedQuantile(ctx, alpha, ptr.cbegin(), ptr.cend(), values.cbegin(),
                                    values.cend(), weights.cbegin(), weights.cend(), &result);
  auto api_selected = result.ConstHostVector();

  // The production API does not expose its temporary CDF. Repeat its actual
  // source primitives/functors to observe a CDF and GPU selection. This is a
  // separate execution, not a capture of the API's private temporary buffer.
  dh::device_vector<std::size_t> sorted;
  common::SegmentedArgMergeSort(ctx, ptr.cbegin(), ptr.cend(), values.cbegin(), values.cend(), &sorted);
  auto key = dh::MakeIndexTransformIter(
      common::detail::SegOp<decltype(ptr.cbegin())>{ptr.cbegin(), ptr.cend()});
  auto weight = dh::MakeIndexTransformIter(
      common::detail::WeightOp<decltype(weights.cbegin())>{weights.cbegin(), dh::ToSpan(sorted)});
  dh::device_vector<float> cdf(rows);
  thrust::inclusive_scan_by_key(ctx->CUDACtx()->CTP(), key, key+rows, weight, cdf.begin());
  HostDeviceVector<float> selected(1, 0.f, ctx->Device());
  auto view = xgboost::linalg::MakeVec(ctx->Device(), selected.DeviceSpan());
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), 1,
                    common::detail::MakeWQSegOp(ptr.cbegin(), values.cbegin(), alpha,
                                                dh::ToSpan(cdf), dh::ToSpan(sorted), view));
  auto observed = Read(cdf);
  auto picked = selected.ConstHostVector();
  for (unsigned i = 0; i < rows; ++i) {
    Check(observed[i] == q.serial[i] || observed[i] == q.regrouped[i],
          "CDF is outside this fixture's possible positive-sum results");
  }
  auto index = std::lower_bound(observed.begin(), observed.end(), double(observed.back())) - observed.begin();
  Check(picked[0] == q.values[index], "GPU source selection differs from its observed CDF");
  Check(api_selected.size() == 1 && (api_selected[0] == 0 || api_selected[0] == float(rows-1)),
        "Production weighted quantile selected an impossible witness value");
  report.Case("weighted_cdf", rows, exponent, 0,
              {{"values", fixture::FloatWords(q.values)}, {"weights", fixture::FloatWords(q.weights)}},
              {{"cdf", fixture::FloatWords(observed)}, {"selected", fixture::FloatWords(picked)},
               {"api_selected", fixture::FloatWords(api_selected)}});
}

void Prefix(Context const* ctx, fixture::Report& report, unsigned rows, int exponent) {
  auto p = fixture::MakePrefix(rows, exponent);
  std::vector<xgboost::Entry> input;
  for (unsigned i = 0; i < rows; ++i) input.emplace_back(p.ids[i], float(i % 3));
  dh::device_vector<xgboost::Entry> entries(input);
  dh::device_vector<float> weights(p.values);
  // Calls the actual sort + in-place segmented scan used by CUDA sketch input.
  common::detail::SortByWeight(ctx, &weights, &entries);
  auto observed = Read(weights);
  auto sorted = Read(entries);
  float large = std::ldexp(0x1p24f, exponent), small = std::ldexp(1.f, exponent);
  for (unsigned i = 0; i < rows; ++i) {
    Check(sorted[i].index == input[i].index && sorted[i].fvalue == input[i].fvalue,
          "Already sorted fixture changed its ordering");
    Check(observed[i] == large || (i % 3 == 2 && observed[i] == fixture::Add(large, 2*small)),
          "Prefix is outside this fixture's possible positive-sum results");
  }
  report.Case("raw_prefix", rows, exponent, 0,
              {{"weights", fixture::FloatWords(p.values)}, {"ids", {p.ids.begin(), p.ids.end()}}},
              {{"cdf", fixture::FloatWords(observed)}});
}

void Summary(Context const* ctx, fixture::Report& report, int exponent, bool repair,
             unsigned iteration) {
  auto s = repair ? fixture::MakeRepair(exponent) : fixture::MakeReverse(exponent);
  auto input = Entries(s.input);
  dh::device_vector<SketchEntry> entries(input);
  using Offset = SketchContainer::OffsetT;
  dh::device_vector<Offset> ptr(std::vector<Offset>{s.ptr.begin(), s.ptr.end()});
  HostDeviceVector<xgboost::FeatureType> types;
  SketchContainer sketch(types, 16, s.ptr.size()-1, ctx->Device());
  std::vector<SketchEntry> observed;
  std::vector<Offset> observed_ptr;
  if (repair) {
    // All values within each column are unique. The empty-container Merge
    // copies them, performs identity unique compaction, then invokes the actual
    // in-place FixError once. A fresh container is used for each repetition.
    sketch.Merge(ctx, dh::ToSpan(ptr), dh::ToSpan(entries));
    observed = Read(sketch.Data().data(), sketch.Data().size());
    observed_ptr = Read(sketch.ColumnsPtr().data(), sketch.ColumnsPtr().size());
    Check(observed.size() == input.size(), "Repair changed summary size");
    Check(observed_ptr.size() == s.ptr.size() &&
              std::equal(observed_ptr.begin(), observed_ptr.end(), s.ptr.begin()),
          "Repair changed column boundaries");
    for (std::size_t i = 0; i < input.size(); ++i) {
      Check(observed[i].value == input[i].value && observed[i].wmin == input[i].wmin,
            "Repair changed a value or weight");
      Check(observed[i].rmin >= input[i].rmin && observed[i].rmax >= input[i].rmax &&
                observed[i].rmax >= fixture::Add(observed[i].rmin, observed[i].wmin),
            "Repair violated its per-entry equations");
    }
  } else {
    auto n = sketch.ScanInput(ctx, dh::ToSpan(entries), dh::ToSpan(ptr));
    observed = Read(entries.data().get(), n);
    observed_ptr = Read(ptr);
    Check(n == 4, "Reverse scan/unique changed the number of distinct values");
    std::vector<Offset> expected_ptr{0, 0, 1, 3, 4};
    Check(observed_ptr == expected_ptr, "Reverse scan/unique changed column boundaries");
    std::size_t retained[]{0, 3, 4, 6};
    for (std::size_t i = 0; i < n; ++i) {
      Check(observed[i].value == input[retained[i]].value &&
                observed[i].rmin == input[retained[i]].rmin,
            "Reverse scan/unique did not retain the first value/rank");
    }
  }
  report.Case(repair ? "repair" : "reverse_unique", s.input.size(), exponent, iteration,
              {{"entries", fixture::EntryWords(s.input.data(), s.input.size())}, {"ptr", s.ptr}},
              {{"entries", fixture::EntryWords(observed.data(), observed.size())},
               {"ptr", {observed_ptr.begin(), observed_ptr.end()}}});
}

void Linalg(char const* input_path, char const* output_path) {
  namespace lc = linalg_contracts;
  namespace la = xgboost::linalg;
  std::string metadata_path=std::string(output_path)+".json";
  Check(!std::filesystem::exists(metadata_path),"Linalg metadata already exists");
  int devices=0; Cuda(cudaGetDeviceCount(&devices));
  Check(devices>0,"Actual NVIDIA GPU required; no skipped pass or CPU fallback");
  Cuda(cudaSetDevice(0));
  auto ctx=Context{}.MakeCUDA(0); ctx.Init({{"device","cuda:0"},{"nthread","2"}});
  lc::InputReader input(input_path); lc::Writer report(output_path,lc::kCuda);
  std::size_t fields=0, executed=0, projections=0;
  auto same_bits=[](std::vector<float> const& a,std::vector<float> const& b) {
    return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)));
  };
  for (unsigned index=0; index<lc::kCases; ++index) {
    auto in=input.Next(); auto s=in.spec; lc::Observation observed;
    if (s.kind==3) {
      // NUM-OP-11: nonfinite/too-many-run rejection is a Metal extension of
      // undefined CUDA buffer behavior. No CUDA call and no fabricated results.
      observed.status=2;
    } else {
      dh::device_vector<float> values(in.values), weights(in.weights);
      dh::device_vector<float> output(in.initial);
      auto vector=[&](dh::device_vector<float>& v,std::size_t n,std::size_t stride) {
        std::size_t shape[1]{n}, strides[1]{stride};
        return la::VectorView<float>{dh::ToSpan(v),shape,strides,ctx.Device()};
      };
      if (s.kind==1 || s.kind==5) {
        auto view=vector(values,s.rows,s.stride0);
        if (s.op==0) la::VecScaMul(&ctx,view,s.parameter);
        if (s.op==1) la::VecScaDiv(&ctx,view,s.parameter);
        if (s.op==2) la::LogE(&ctx,view,float(s.parameter));
        if (s.op==3) {
          // Exact source expression from SoftmaxMultiClassObj::InitEstimation.
          // This tests the transform, not the entire objective call chain.
          dh::device_vector<float> mean(1,float(s.parameter));
          auto d_mean=la::MakeVec(mean.data().get(),1,ctx.Device());
          la::TransformKernel(&ctx,view,[=] XGBOOST_DEVICE(float v) { return v-d_mean(0); });
          Cuda(cudaDeviceSynchronize());
        }
        observed.actual=Read(values);
      } else if (s.kind==2) {
        std::size_t shape[2]{s.rows,s.columns}, strides[2]{s.stride0,s.stride1};
        la::MatrixView<float const> matrix{dh::ToSpan(values),shape,strides,ctx.Device()};
        if (s.columns==1) {
          la::Vector<float> mean;
          common::Mean(&ctx,matrix.Slice(la::All(),0),&mean);
          Check(mean.Data()->Device().IsCUDA() && !mean.Data()->HostCanRead(),
                "Linalg mean output not resident on CUDA");
          observed.actual=mean.Data()->ConstHostVector();
        } else {
          // The Metal batched primitive corresponds to one source FP32 Mean
          // per column. Preserve all input/output strides and guard storage.
          for (unsigned t=0; t<s.columns; ++t) {
            auto out=la::MakeVec(output.data().get()+t*s.out_stride,1,ctx.Device());
            common::cuda_impl::Mean(&ctx,matrix.Slice(la::All(),t),out);
          }
          observed.actual=Read(output);
        }
      } else {
        // CUDA ArgSort consumes contiguous keys. Gather only logical labels,
        // leaving the original strided backing storage unchanged. This does
        // not assert that the CUDA histogram supports strided input labels.
        dh::device_vector<float> packed(s.rows);
        auto src=dh::ToSpan(values), dst=dh::ToSpan(packed);
        dh::LaunchN(s.rows,ctx.CUDACtx()->Stream(),[=] XGBOOST_DEVICE(std::size_t i) {
          dst[i]=src[i*s.stride0];
        });
        auto gathered=Read(packed);
        for (std::size_t i=0; i<s.rows; ++i)
          Check(lc::Bits<std::uint32_t>(gathered[i])==lc::Bits<std::uint32_t>(in.values[i*s.stride0]),
                "CUDA histogram gather changed logical input bits");
        ++projections;
        std::size_t shape[2]{s.rows,1};
        la::MatrixView<float const> labels{dh::ToSpan(packed),shape,ctx.Device()};
        auto optional=s.kind==0 && s.mode ? common::OptionalWeights{dh::ToSpan(weights)}
                                         : common::OptionalWeights{float(s.parameter)};
        la::SmallHistogram(&ctx,labels,optional,vector(output,s.columns,s.out_stride));
        observed.actual=Read(output);
      }
      if (s.kind!=1 && s.kind!=5) Check(same_bits(Read(values),in.values),"Linalg modified input bits");
      Check(same_bits(Read(weights),in.weights),"Linalg modified weight bits");
      ++executed;
    }
    fields+=observed.actual.size(); report.Add(in,observed);
    if ((index+1)%20==0 || index+1==lc::kCases) {
      std::printf("CUDA linalg %u/%u: executed=%zu outside_domain=%u observed_fields=%zu\n",
                  index+1,lc::kCases,executed,unsigned(index+1-executed),fields);
      std::fflush(stdout);
    }
  }
  input.Finish(); report.Finish();
  cudaDeviceProp gpu{}; Cuda(cudaGetDeviceProperties(&gpu,0));
  int driver=0,runtime=0; Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
  using namespace xgboost;
  Json meta{Object{}};
  meta["status"]=String("completed common-domain observations; three Metal-only guards not executed; comparison remains");
  meta["suite"]=String("linalg-v1"); meta["backend"]=String("cuda"); meta["device"]=String(gpu.name);
  meta["compute_capability"]=String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"]=String(std::to_string(__CUDACC_VER_MAJOR__)+"."+
      std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"]=Integer(driver); meta["runtime"]=Integer(runtime);
  meta["thrust_version"]=Integer(THRUST_VERSION); meta["cub_version"]=Integer(CUB_VERSION);
  meta["cases"]=Integer(lc::kCases); meta["executed_cases"]=Integer(executed);
  meta["outside_domain_cases"]=Integer(3); meta["output_fields"]=Integer(fields);
  meta["histogram_contiguous_projections"]=Integer(projections);
  meta["scope"]=String("CUDA histogram uses bit-checked GPU gather of logical labels; multi-column means invoke scalar CUDA Mean per column; subtraction invokes multiclass source transform expression; no CPU/reference outputs supplied");
  meta["input_file"]=String(input_path); meta["observation_file"]=String(output_path);
  std::string text; Json::Dump(meta,&text);
  std::ofstream stream{metadata_path}; stream.exceptions(std::ios::badbit|std::ios::failbit); stream<<text<<'\n';
  std::printf("CUDA linalg observations completed: %zu executed cases, 3 outside domain; comparison remains\n",executed);
}

void Statistics(char const* input_path, char const* output_path) {
  namespace sc = stats_contracts;
  std::string metadata_path = std::string(output_path)+".json";
  Check(!std::filesystem::exists(metadata_path), "Statistics metadata already exists");
  int devices = 0; Cuda(cudaGetDeviceCount(&devices));
  Check(devices > 0, "Actual NVIDIA GPU required; no skipped pass or CPU fallback");
  Cuda(cudaSetDevice(0));
  auto ctx = Context{}.MakeCUDA(0); ctx.Init({{"device","cuda:0"},{"nthread","2"}});
  sc::InputReader input(input_path); sc::Writer report(output_path, sc::kCuda);
  std::size_t fields = 0;
  for (unsigned index = 0; index < sc::kCases; ++index) {
    auto in = input.Next(); auto s = in.spec; sc::Observation observed;
    if (s.kind == 0) {
      xgboost::linalg::Matrix<float> values({s.rows, s.columns}, ctx.Device());
      values.Data()->HostVector() = in.values;
      HostDeviceVector<float> weights(in.weights, ctx.Device());
      xgboost::linalg::Vector<float> actual;
      auto capture = [&] {
        if (s.rows) Check(actual.Data()->Device().IsCUDA() && !actual.Data()->HostCanRead(),
                          "Statistics output is not resident on CUDA before observation");
        for (auto value : actual.HostView().Values())
          observed.actual.push_back(sc::Bits<std::uint32_t>(value));
      };
      common::Median(&ctx, values, weights, &actual); capture();
      common::SampleMean(&ctx, values, &actual); capture();
      common::Mean(&ctx, values.View(ctx.Device()).Slice(xgboost::linalg::All(), 0), &actual); capture();
      unsigned status = 2;  // Weighted mean not called in mode zero.
      if (s.mode) {
        status = 0;
        if (!s.rows || s.mode == 2) {
          try { common::WeightedSampleMean(&ctx, values, weights, &actual); }
          catch (std::exception const& error) {
            Check(std::string(error.what()).find("weights must contain at least one non-zero value") != std::string::npos,
                  "Unexpected CUDA weighted-mean exception");
            status = 1;
          }
        } else {
          common::WeightedSampleMean(&ctx, values, weights, &actual); capture();
        }
      }
      observed.actual.push_back(status);
    } else if (s.kind == 3) {
      double value = common::SumOptionalWeights(&ctx, common::OptionalWeights{sc::Bits<float>(s.scalar)}, s.rows);
      observed.actual.push_back(sc::Bits<std::uint64_t>(value));
    } else {
      HostDeviceVector<float> values(in.values, ctx.Device());
      double generic = common::Reduce(&ctx, values);
      auto weights = common::MakeOptionalWeights(ctx.Device(), values);
      double optional = common::SumOptionalWeights(&ctx, weights, s.rows);
      if (s.rows) Check(values.Device().IsCUDA() && values.DeviceCanRead(), "Reduction input is not CUDA-resident");
      observed.actual = {sc::Bits<std::uint64_t>(generic), sc::Bits<std::uint64_t>(optional)};
    }
    Cuda(cudaDeviceSynchronize()); report.Add(in, observed); fields += observed.actual.size();
    if ((index+1)%14 == 0 || index+1 == sc::kCases) {
      std::printf("CUDA statistics %u/%u: family=%u rows=%u observed_fields=%zu\n", index+1, sc::kCases, s.kind, s.rows, fields);
      std::fflush(stdout);
    }
  }
  input.Finish(); report.Finish();
  cudaDeviceProp gpu{}; Cuda(cudaGetDeviceProperties(&gpu, 0));
  int driver = 0, runtime = 0; Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
  using namespace xgboost;
  Json meta{Object{}};
  meta["status"] = String("completed-observations; comparison remains");
  meta["suite"] = String("stats-v1"); meta["backend"] = String("cuda"); meta["device"] = String(gpu.name);
  meta["compute_capability"] = String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
  meta["nvcc"] = String(std::to_string(__CUDACC_VER_MAJOR__)+"."+
      std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
  meta["driver"] = Integer(driver); meta["runtime"] = Integer(runtime);
  meta["thrust_version"] = Integer(THRUST_VERSION); meta["cub_version"] = Integer(CUB_VERSION);
  meta["cases"] = Integer(sc::kCases); meta["output_fields"] = Integer(fields);
  meta["input_file"] = String(input_path); meta["observation_file"] = String(output_path);
  meta["scope"] = String("Actual public statistics APIs; scalar optional weights and empty generic sums retain source host boundaries; no reference values supplied");
  std::string text; Json::Dump(meta, &text);
  std::ofstream stream{metadata_path}; stream.exceptions(std::ios::badbit | std::ios::failbit); stream << text << '\n';
  std::printf("PASS CUDA statistics execution: %u cases / %zu fields; comparison remains\n", sc::kCases, fields);
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--metric-registry") { MetricRegistrySignature(); return 0; }
    if (argc == 4 && std::string(argv[1]) == "--linalg") { Linalg(argv[2], argv[3]); return 0; }
    if (argc == 4 && std::string(argv[1]) == "--statistics") { Statistics(argv[2], argv[3]); return 0; }
    Check(argc == 2, "Usage: cuda_numerical_contracts_test new-report.jsonl");
    int devices = 0;
    Cuda(cudaGetDeviceCount(&devices));
    Check(devices > 0, "No NVIDIA GPU: execution is required; this is not a skipped pass");
    Cuda(cudaSetDevice(0));
    cudaDeviceProp device{};
    Cuda(cudaGetDeviceProperties(&device, 0));
    int driver = 0, runtime = 0;
    Cuda(cudaDriverGetVersion(&driver));
    Cuda(cudaRuntimeGetVersion(&runtime));
    auto ctx = Context{}.MakeCUDA(0);
    Check(ctx.Device().IsCUDA(), "XGBoost context is not CUDA");
    fixture::Report report(argv[1]);
    report.Metadata({{"backend", "cuda"}, {"fixture_suite", "numerical-v2"}, {"device", device.name},
                     {"compute_capability", std::to_string(device.major)+"."+std::to_string(device.minor)},
                     {"driver", std::to_string(driver)}, {"runtime", std::to_string(runtime)},
                     {"cuda_headers", std::to_string(CUDART_VERSION)},
                     {"nvcc", std::to_string(__CUDACC_VER_MAJOR__)+"."+
                                  std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__)},
                     {"thrust", std::to_string(THRUST_VERSION)}, {"cub", std::to_string(CUB_VERSION)},
                     {"weighted_cdf", "production API plus separately executed source scan/select trace"},
                     {"repair", "fresh Merge invokes FixError once; 32 repetitions per input"}});
    unsigned completed = 0;
    for (int exponent : {0, -149, 100}) {
      for (unsigned variant = 0; variant < 6; ++variant) { Threshold(&ctx, report, exponent, variant); ++completed; }
      for (unsigned rows : {3u, 257u}) { Quantile(&ctx, report, rows, exponent); ++completed; }
      for (unsigned rows : {3u, 261u, 513u}) { Prefix(&ctx, report, rows, exponent); ++completed; }
      Summary(&ctx, report, exponent, false, 0); ++completed;
      for (unsigned repeat = 0; repeat < 32; ++repeat) {
        Summary(&ctx, report, exponent, true, repeat); ++completed;
      }
      std::printf("Recorded exponent %d; %u CUDA executions completed\n", exponent, completed);
      std::fflush(stdout);
    }
    report.Finish();
    std::printf("PASS execution/invariants: %u cases. Backend differences require comparison/review.\n", completed);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
