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
echo "Checking Metal partitioning and index transposition against CPU references"
clang++ -std=c++17 -fobjc-arc tests/metal_partition.mm \
  -framework Metal -framework Foundation -o results/metal_partition_test
results/metal_partition_test xgboost/src/common/metal_histogram.metal
echo "Checking software binary64 against CPU arithmetic"
clang++ -std=c++17 -fobjc-arc tests/metal_fp64.mm \
  -framework Metal -framework Foundation -o results/metal_fp64_test
results/metal_fp64_test xgboost/src/common/metal_histogram.metal
echo "Checking scalar and vector split kernels against independent CPU references"
clang++ -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_split.mm \
  -framework Metal -framework Foundation -o results/metal_split_test
results/metal_split_test xgboost/src/common/metal_histogram.metal
echo "Checking GPU histogram conversion, accumulation, subtraction and gathers"
clang++ -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_storage.mm \
  -framework Metal -framework Foundation -o results/metal_storage_test
results/metal_storage_test xgboost/src/common/metal_histogram.metal
echo "Checking adaptive objective scales and gradients against CPU references"
clang++ -std=c++17 -ffp-contract=off -fobjc-arc tests/metal_adaptive.mm \
  -framework Metal -framework Foundation -o results/metal_adaptive_test
results/metal_adaptive_test xgboost/src/common/metal_histogram.metal
echo "Checking GPU sampling against the CPU random engine and Bernoulli distribution"
clang++ -std=c++17 -fobjc-arc tests/metal_sampling.mm \
  -framework Metal -framework Foundation -o results/metal_sampling_test
results/metal_sampling_test xgboost/src/common/metal_histogram.metal
echo "Running training equivalence tests; a real GPU is required"
.venv/bin/python -m pytest -q tests --require-metal
