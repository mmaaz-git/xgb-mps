#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
echo "Compiling standalone tests against the production Metal shader"
clang++ -std=c++17 -fobjc-arc tests/metal_kernel.mm \
  -framework Metal -framework Foundation -o results/metal_kernel_test
echo "Checking GPU integer histograms against the independent CPU reference"
results/metal_kernel_test xgboost/src/common/metal_histogram.metal
echo "Checking objective kernels against independent CPU mathematics"
clang++ -std=c++17 -fobjc-arc tests/metal_objective.mm \
  -framework Metal -framework Foundation -o results/metal_objective_test
results/metal_objective_test xgboost/src/common/metal_histogram.metal
echo "Running training equivalence tests; a real GPU is required"
.venv/bin/python -m pytest -q tests --require-metal
