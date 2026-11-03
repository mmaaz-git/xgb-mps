#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
echo "Configuring XGBoost with Metal and OpenMP"
cmake -S xgboost -B build-metal -G Ninja -DUSE_METAL=ON -DUSE_OPENMP=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
echo "Building XGBoost (4 parallel compiler jobs)"
cmake --build build-metal -j 4
echo "Build complete: xgboost/lib/libxgboost.dylib"
