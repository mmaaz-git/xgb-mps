#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
libomp_root="${LIBOMP_ROOT:-$(brew --prefix libomp)}"
export DYLD_LIBRARY_PATH="$libomp_root/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
echo "Configuring isolated Metal/federated build (requires gRPC and Protobuf)"
cmake -S xgboost -B build-metal-federated -G Ninja -DUSE_METAL=ON -DUSE_OPENMP=ON \
  -DPLUGIN_FEDERATED=ON -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-metal-federated -j4
echo "Preparing an isolated Python package with the federated library"
.venv/bin/python - <<'PY'
from pathlib import Path
root = Path.cwd()
source = root / "xgboost/python-package/xgboost"
target = root / "results/federated-python/xgboost"
target.mkdir(parents=True, exist_ok=True)
for entry in source.iterdir():
    if entry.name in {"lib", "__pycache__"}:
        continue
    link = target / entry.name
    if link.is_symlink():
        assert link.resolve() == entry.resolve(), link
    else:
        assert not link.exists(), link
        link.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
link = target / "lib"
library = root / "build-metal-federated/lib"
if link.is_symlink():
    assert link.resolve() == library.resolve(), link
else:
    assert not link.exists(), link
    link.symlink_to(library, target_is_directory=True)
PY
export PYTHONPATH="$PWD/results/federated-python${PYTHONPATH:+:$PYTHONPATH}"
echo "Building federated transport and sketch reference checks"
for pair in "metal_federated_api metal_federated_api_test" \
            "metal_sketch_collective_api metal_federated_sketch_api_test"; do
  read -r source output <<< "$pair"
  clang++ -O2 -std=c++17 -ffp-contract=off \
    -DDMLC_CORE_USE_CMAKE -DDMLC_LOG_CUSTOMIZE=1 -DXGBOOST_USE_METAL=1 \
    -I xgboost/include -I xgboost/dmlc-core/include -I build-metal-federated/dmlc-core/include \
    -Xpreprocessor -fopenmp -I "$libomp_root/include" "tests/$source.cc" \
    -L build-metal-federated/lib -lxgboost -L "$libomp_root/lib" -lomp \
    -Wl,-rpath,"$PWD/build-metal-federated/lib" -o "results/$output"
done
MTL_DEBUG_LAYER=1 .venv/bin/python tests/run_federated_metal.py "$@"
if [[ $# -eq 0 ]]; then
  MTL_DEBUG_LAYER=1 .venv/bin/python tests/run_federated_metal.py --secure --worlds 2
fi
