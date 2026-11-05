// Observe actual NVIDIA arithmetic/libdevice on the exact exported Metal inputs.
// CPU expectations are not sent to kernels and differences are retained for review.
#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/version.h>
#include <cub/version.cuh>
#include <cmath>
#include <cstdio>

#include "../xgboost/src/common/cuda_context.cuh"
#include "../xgboost/src/common/device_helpers.cuh"
#include "xgboost/context.h"
#include "xgboost/json.h"
#include "math_contract_io.h"

namespace math = math_contracts;
void Cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
  try {
    math::Require(argc == 3, "Usage: cuda_math_contracts input-only.bin new-cuda-observations.bin");
    std::string metadata_path = std::string(argv[2])+".json";
    math::Require(!std::filesystem::exists(metadata_path), "Math metadata already exists");
    int devices = 0;
    Cuda(cudaGetDeviceCount(&devices));
    math::Require(devices > 0, "Actual NVIDIA GPU required; no skipped pass or CPU fallback");
    Cuda(cudaSetDevice(0));
    auto ctx = xgboost::Context{}.MakeCUDA(0);
    cudaDeviceProp gpu{}; Cuda(cudaGetDeviceProperties(&gpu, 0));
    int driver = 0, runtime = 0;
    Cuda(cudaDriverGetVersion(&driver)); Cuda(cudaRuntimeGetVersion(&runtime));
    math::InputReader input(argv[1]);
    math::Writer report(argv[2], math::kCuda, math::kAll);
    std::size_t fields = 0;
    for (unsigned id = 0; id < 4; ++id) {
      auto words = input.Next(id);
      auto shape = math::kShapes[id];
      dh::device_vector<std::uint64_t> source(words), storage(std::size_t{shape.rows}*shape.outputs);
      auto in = dh::ToSpan(source), out = dh::ToSpan(storage);
      thrust::for_each_n(ctx.CUDACtx()->CTP(), thrust::make_counting_iterator(std::size_t{0}),
          shape.rows, [=] __device__(std::size_t i) {
            if (id == 0) {
              double a = __longlong_as_double(in[2*i]), b = __longlong_as_double(in[2*i+1]);
              out[6*i] = __double_as_longlong(a+b); out[6*i+1] = __double_as_longlong(a-b);
              out[6*i+2] = __double_as_longlong(a*b); out[6*i+3] = __double_as_longlong(a/b);
              out[6*i+4] = __float_as_uint(static_cast<float>(a));
              out[6*i+5] = __float_as_uint(static_cast<float>(b));
            } else if (id == 1) {
              out[i] = __double_as_longlong(static_cast<double>(__uint_as_float(in[i])));
            } else if (id == 2) {
              double x = __longlong_as_double(in[i]);
              out[3*i] = __double_as_longlong(exp(x));
              out[3*i+1] = __double_as_longlong(log(x));
              out[3*i+2] = __double_as_longlong(erf(x));
            } else {
              float a = __uint_as_float(in[2*i]), b = __uint_as_float(in[2*i+1]);
              out[2*i] = __float_as_uint(hypotf(a, b));
              out[2*i+1] = __float_as_uint(tanhf(a));
            }
          });
      Cuda(cudaDeviceSynchronize());
      math::Words observed(out.size());
      Cuda(cudaMemcpy(observed.data(), out.data(), out.size_bytes(), cudaMemcpyDeviceToHost));
      report.Add(id, words, observed);
      fields += observed.size();
      std::printf("CUDA math group %u/4: %u rows / %zu output fields; cumulative=%zu\n",
                  id+1, shape.rows, observed.size(), fields);
      std::fflush(stdout);
    }
    input.Finish(); report.Finish();
    using namespace xgboost;
    Json metadata{Object{}};
    metadata["status"] = String("completed-observations; comparison remains");
    metadata["suite"] = String("math-v1"); metadata["backend"] = String("cuda");
    metadata["device"] = String(gpu.name);
    metadata["compute_capability"] = String(std::to_string(gpu.major)+"."+std::to_string(gpu.minor));
    metadata["nvcc"] = String(std::to_string(__CUDACC_VER_MAJOR__)+"."+
        std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__));
    metadata["driver"] = Integer(driver); metadata["runtime"] = Integer(runtime);
    metadata["thrust_version"] = Integer(THRUST_VERSION); metadata["cub_version"] = Integer(CUB_VERSION);
    metadata["groups"] = Integer(4); metadata["output_fields"] = Integer(fields);
    metadata["input_file"] = String(argv[1]); metadata["observation_file"] = String(argv[2]);
    std::string text; Json::Dump(metadata, &text);
    std::ofstream stream{metadata_path}; stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream << text << '\n';
    std::printf("PASS CUDA math execution: %zu observed fields; no equivalence claim\n", fields);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL CUDA math: %s\n", error.what());
    return 1;
  }
}
