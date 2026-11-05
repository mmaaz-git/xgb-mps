#!/bin/bash
# Compile/link validation only. This job has no NVIDIA GPU and runs no GPU tests.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-cuda-ci
date -u
nvcc --version
cmake --version
dpkg-query -W libnccl-dev libnvcomp6-dev-cuda-13 libgrpc++-dev protobuf-compiler

echo 'Checking that CUDA uses the modern Philox4x64 branch targeted by Metal'
cat > build-cuda-ci/rng-branch.cu <<'CPP'
#include <type_traits>
#include "src/common/random.cuh"
#if !__has_include(<cuda/std/random>)
#error "The Metal random-stream target requires the modern CUDA random header."
#endif
static_assert(std::is_same_v<xgboost::common::cuda_impl::DefaultRng, cuda::std::philox4x64>);
CPP
nvcc -std=c++17 -I xgboost -c build-cuda-ci/rng-branch.cu -o build-cuda-ci/rng-branch.o

echo 'Configuring CUDA with NCCL, optional nvCOMP and federated consumers'
cmake -S xgboost -B build-cuda-ci -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DUSE_CUDA=ON -DUSE_METAL=OFF -DUSE_NCCL=ON -DBUILD_WITH_SHARED_NCCL=ON \
  -DUSE_NVCOMP=ON -DPLUGIN_FEDERATED=ON -DUSE_OPENMP=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
echo 'Compiling and linking libxgboost (no GPU execution)'
cmake --build build-cuda-ci --parallel 2
echo 'PASS CUDA compilation/linking; GPU runtime behavior remains untested'
date -u
