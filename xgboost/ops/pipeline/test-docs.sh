#!/bin/bash
## Run doctests embedded in the Sphinx documentation.

set -euo pipefail

echo "--- Build libxgboost for documentation snippets"
cmake_args=(
  -S .
  -B build/doc-test
  -DHIDE_CXX_SYMBOLS=ON
  -DUSE_OPENMP=ON
  -DCMAKE_BUILD_TYPE=Release
)
if command -v sccache >/dev/null 2>&1; then
  cmake_args+=(
    -DCMAKE_C_COMPILER_LAUNCHER=sccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
  )
fi
cmake "${cmake_args[@]}"
cmake --build build/doc-test --target xgboost --parallel "$(nproc)"

echo "--- Run Sphinx doctest builder"
python3 -m sphinx -b doctest -a -E doc doc/_build/doctest
