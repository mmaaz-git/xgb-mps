#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
libomp_root="${LIBOMP_ROOT:-$(brew --prefix libomp)}"
clang++ -O2 -std=c++17 -ffp-contract=off -dynamiclib \
  -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
  -I xgboost/include -I xgboost/dmlc-core/include -I build-metal/dmlc-core/include \
  -Xpreprocessor -fopenmp -I "$libomp_root/include" tests/cuda_linear_reference.cc \
  -L xgboost/lib -lxgboost -L "$libomp_root/lib" -lomp \
  -Wl,-rpath,"$PWD/xgboost/lib" -o results/libcuda_linear_reference.dylib
