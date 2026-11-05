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

int main(int argc, char** argv) {
  try {
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
