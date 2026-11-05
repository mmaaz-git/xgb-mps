# Running the shared numerical witnesses on NVIDIA

VERIFY-2 requires actual GPU execution. The first harness covers **P5-N08,
P5-N13 and P5-N14**, using the same 21 inputs as the Metal numerical-contract
test. The other fourteen numerical contracts and broader caller/integration
comparisons remain work; this harness is not a full parity certificate.

No NVIDIA execution result is published yet. The CUDA CI job compiles/links the
harness on a CPU runner and retains its exact compiler command and input/library
hashes. A runtime invocation with no accessible NVIDIA GPU fails; it cannot
silently pass by skipping or falling back to CPU.

## What executes

| Witness | CUDA path | Metal observation and boundary |
| --- | --- | --- |
| Weighted CDF, 3/257 rows | Production `SegmentedWeightedQuantile`, plus a separately executed trace using its source sort/scan/selection functors. | Production quantile API and direct production CDF/select kernels. The CUDA trace is a separate execution, not a capture of the API's private temporary CDF. Both selections are retained. |
| Raw segmented prefixes, 3/261/513 rows | Production `detail::SortByWeight`, including its in-place segmented scan. | Production hierarchical prefix kernels; cases cross 256-entry tile boundaries. |
| Reverse duplicate accumulation | Production `SketchContainer::ScanInput`, including unique compaction. | Observed production reverse-scan output is projected onto retained first entries on the host for comparison. This report projection does not test Metal GPU compaction; existing sketch integration fixtures cover that separately. |
| Neighboring rank repair | A fresh empty `SketchContainer::Merge` copies the unique-valued input and calls production `FixError` once. | Production deterministic left-to-right repair kernel. Synthetic rank violations isolate the operation, without claiming exact training-data reachability. |

All inputs run at exponents `0`, `-149` and `100`. CUDA repeats each repair input
32 times, always with fresh storage, producing **114 execution records for 21
input cases**. Repetition characterizes observed scheduling on that GPU; it does
not enumerate every possible outcome of the source's neighboring read/write race.
Neither runner changes production formulas or widens ordinary tolerances.

Reports use JSON Lines and unsigned integer words for floating-point bits, so
subnormals, signed zeros and nonfinite payloads are not changed by decimal JSON
serialization. Every case carries its full input and output arrays. Metadata
records the GPU and compiler/runtime versions; the CUDA build record also keeps
architecture, flags, CCCL configuration and the linked library hash.

## Produce a Metal observation file

From the configured macOS checkout:

```sh
bash scripts/test.sh --native metal_numerical_contracts
MTL_DEBUG_LAYER=1 results/metal_numerical_contracts_test \
  xgboost/src/common/metal_histogram.metal --report results/metal-observations.jsonl
python3 scripts/compare_numerical_contracts.py inspect results/metal-observations.jsonl
```

Use a new output name; existing reports are never overwritten. The native test
continues to enforce its exact Metal expectations before emitting observations.
The existing [local acceptance](LOCAL_ACCEPTANCE.md) records the fixed production
revision and limits; source changes require a new matching observation file.
The [published M3 observations](../validation/cuda-comparison-metal-m3.jsonl) use
the accepted `68c800e` production tree and the `0074861` harness. Their
[provenance](../validation/cuda-comparison-harness-m3.json) retains build/test
commands, hashes and the corrected reporting-code compile failure. These are
Metal observations; there is no corresponding NVIDIA execution file yet.

## Build and execute on NVIDIA

Use a Linux NVIDIA machine with a compatible driver, CUDA toolkit, its CCCL CMake
package, CMake, Ninja, a supported host compiler, OpenMP and Python 3. The CI
configuration uses NVCC 13.4.92. This standalone consumer currently supports a
shared XGBoost library with visible C++ symbols and without RMM.

For a fresh single-device comparison build, optional distributed/codec plugins
can be disabled. Set the architecture for the test GPU; `80` matches compiler CI.

```sh
cmake -S xgboost -B build-cuda-run -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DUSE_CUDA=ON -DUSE_METAL=OFF -DUSE_NCCL=OFF -DUSE_NVCOMP=OFF \
  -DPLUGIN_FEDERATED=OFF -DPLUGIN_RMM=OFF \
  -DHIDE_CXX_SYMBOLS=OFF -DBUILD_STATIC_LIB=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
python3 scripts/build_cuda_comparison.py --build-dir build-cuda-run \
  --output-dir results/cuda-comparison-build
results/cuda-comparison-build/cmake/cuda_numerical_contracts \
  results/cuda-observations.jsonl
```

The build driver first compiles the harness object using the configured compiler,
CCCL, architectures and arithmetic flags. It then checks/rebuilds production
dependencies and links the executable. This exposes harness compiler errors
before the expensive library build. It writes `build.json`, including terminal
status; CI separates the object and link phases and rejects changed inputs
between them. Git metadata reads trust only the selected checkout for that
invocation, accommodating container checkout ownership. The CI's
`scripts/ci_cuda_build.sh` does this against its CUDA/NCCL/nvCOMP/federated build,
but **does not run the final GPU command**.

Keep the runtime log, observation file and build record together. Run the same
source/fixture revision on both platforms. The comparison command rejects
missing inputs, missing repetitions, truncated files and changed fixture arrays:

```sh
python3 scripts/compare_numerical_contracts.py compare \
  results/metal-observations.jsonl results/cuda-observations.jsonl \
  --output results/numerical-comparison.json
```

Exit `0` means outputs match on these inputs. Exit `1` means invalid evidence or
a structural/source-preserved-field mismatch. Exit `2` preserves observed
numerical/read-order differences for contract review; it does not automatically
classify them as either defects or acceptable drift. The report retains changed
fields and sample bit patterns, points to the full source reports by hash, and
counts distinct outputs across repetitions. Review against the
[per-operation contracts](LOCAL_ACCEPTANCE.md#seventeen-numerical-contracts),
including selected-observation changes and source-race policies.

The comparison tool's Python tests use explicitly synthetic data. Passing those
integrity tests is not a substitute for the NVIDIA command above.
