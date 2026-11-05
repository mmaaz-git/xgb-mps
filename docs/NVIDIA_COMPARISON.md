# Running the shared numerical witnesses on NVIDIA

VERIFY-2 requires actual GPU execution. The current native harness,
`numerical-v2`, covers **P5-N01, P5-N08, P5-N13 and P5-N14**, using the same 39
inputs as the Metal numerical-contract test. A second native harness covers
**P5-N02**, using the complete AFT corpus described below. A third covers
**P5-N06** ranking RNG and raw pairs. The other eleven numerical contracts and
broader caller/integration comparisons remain work;
these harnesses are not a full parity certificate.

No NVIDIA execution result is published yet. The CUDA CI job compiles/links the
harness on a CPU runner and retains its exact compiler command and input/library
hashes. A runtime invocation with no accessible NVIDIA GPU fails; it cannot
silently pass by skipping or falling back to CPU.

## What executes

| Witness | CUDA path | Metal observation and boundary |
| --- | --- | --- |
| Weighted threshold precision | Production `SegmentedWeightedQuantile` with FP32 alpha promoted by its binary64 threshold expression. | Production quantile API with native GPU inputs/output. Values `[0,1]`, weights `[1,9]`, three adjacent FP32 alphas around `.1`, both input orders and three weight scales isolate threshold narrowing from CDF association. |
| Weighted CDF, 3/257 rows | Production `SegmentedWeightedQuantile`, plus a separately executed trace using its source sort/scan/selection functors. | Production quantile API and direct production CDF/select kernels. The CUDA trace is a separate execution, not a capture of the API's private temporary CDF. Both selections are retained. |
| Raw segmented prefixes, 3/261/513 rows | Production `detail::SortByWeight`, including its in-place segmented scan. | Production hierarchical prefix kernels; cases cross 256-entry tile boundaries. |
| Reverse duplicate accumulation | Production `SketchContainer::ScanInput`, including unique compaction. | Observed production reverse-scan output is projected onto retained first entries on the host for comparison. This report projection does not test Metal GPU compaction; existing sketch integration fixtures cover that separately. |
| Neighboring rank repair | A fresh empty `SketchContainer::Merge` copies the unique-valued input and calls production `FixError` once. | Production deterministic left-to-right repair kernel. Synthetic rank violations isolate the operation, without claiming exact training-data reachability. |

All inputs run at exponents `0`, `-149` and `100`. CUDA repeats each repair input
32 times, always with fresh storage, producing **132 execution records for 39
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
The [current M3 observations](../validation/cuda-comparison-metal-v2-m3.jsonl)
use the accepted `68c800e` production tree. Their
[provenance](../validation/cuda-comparison-threshold-m3.json) retains commands,
hashes, 39 witnesses / 3,491 exact native checks and twelve report-integrity
tests. All 21 older payloads are unchanged; the 18 new cases enforce exact
threshold selection. Historical [v1 observations](../validation/cuda-comparison-metal-m3.jsonl)
and [provenance](../validation/cuda-comparison-harness-m3.json) remain readable.
Comparison requires matching suites and rejects a v1/v2 pair. These are Metal
observations; no corresponding NVIDIA execution file exists yet.

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

The build driver first compiles all three harness objects using the configured compiler,
CCCL, architectures and arithmetic flags. It then checks/rebuilds production
dependencies and links the executables. This exposes harness compiler errors
before the expensive library build. It writes `build.json`, including terminal
status; CI separates the object and link phases and rejects changed inputs
between them. Git metadata reads trust only the selected checkout for that
invocation, accommodating container checkout ownership. The CI's
`scripts/ci_cuda_build.sh` does this against its CUDA/NCCL/nvCOMP/federated build,
but **does not run the final GPU command**.
When production enables nvCOMP, the consumer imports that same CMake package,
retaining its package-specific library path. A small dependency executable is
compiled and linked before the harness object; it is never run on CPU-only CI.

Keep the runtime log, observation file and build record together. Run the same
source/fixture revision on both platforms. The comparison command rejects
missing inputs, missing repetitions, truncated files and changed fixture arrays:

```sh
python3 scripts/compare_numerical_contracts.py compare \
  results/metal-observations.jsonl results/cuda-observations.jsonl \
  --output results/numerical-comparison.json
```

Exit `0` means outputs match on these inputs. Exit `1` means invalid evidence,
an exact weighted-threshold mismatch, or a structural/source-preserved-field
mismatch. Exit `2` preserves observed
numerical/read-order differences for contract review; it does not automatically
classify them as either defects or acceptable drift. The report retains changed
fields and sample bit patterns, points to the full source reports by hash, and
counts distinct outputs across repetitions. Review against the
[per-operation contracts](LOCAL_ACCEPTANCE.md#seventeen-numerical-contracts),
including selected-observation changes and source-race policies.

The comparison tool's Python tests use explicitly synthetic data. Passing those
integrity tests is not a substitute for the NVIDIA command above.

## AFT survival gradients and Hessians

The P5-N02 harness exports the existing Metal test's complete corpus rather than
regenerating random inputs or host-library logarithms on Linux. The
[input file](../validation/aft-contract-inputs-m3.bin.gz) contains **96 cases /
316,683 rows**: normal, logistic and extreme distributions; five ordinary
scales; unweighted, weighted, zero-weighted, subnormal-weighted and uncensored
cases; then seven domain-guard cases per distribution. The
[Metal observations](../validation/aft-contract-metal-m3.bin.gz) retain every
row. The [run record](../validation/aft-contract-m3-run.json) includes source,
library and artifact hashes and the unchanged local results: 99,831 sensitive
rows, 322 CPU-tolerance field differences, and zero production/source field
differences. These are local results, not NVIDIA agreement.

`cuda_aft_contracts` reads only the input file, evaluates vendored
`AFTLoss<Distribution>::Gradient/Hessian` on GPU, and runs the actual
`survival:aft` objective for all 75 ordinary cases. Its weighted outputs must
agree with the separately executed CUDA source functions. The 21 domain-guard
cases call source functions directly; they do not claim that invalid parameters
are accepted by public training. A separately evaluated likelihood expression
classifies sensitive rows; it does not capture an objective's private temporary.
This runner covers gradients and Hessians, not survival metrics or prediction
transformation.

After building the CUDA executables as above:

```sh
python3 scripts/compare_aft_contracts.py unpack \
  validation/aft-contract-inputs-m3.bin.gz --output results/aft-inputs.bin
results/cuda-comparison-build/cmake/cuda_aft_contracts \
  results/aft-inputs.bin results/aft-cuda.bin
python3 scripts/compare_aft_contracts.py compare \
  validation/aft-contract-metal-m3.bin.gz results/aft-cuda.bin \
  --output results/aft-comparison.json
```

Keep `aft-cuda.bin.json`, the runtime log and `build.json` with the observations.
They record the actual GPU, CUDA/compiler versions, build flags, source inputs
and linked library. Use matching production and harness sources. The comparator
enforces identical input bits and complete framing; it does not independently
authenticate the runtime metadata or build provenance.

The ordinary weighted-gradient/Hessian tolerance remains `atol=1e-6, rtol=2e-5`.
Sensitive tails are rows with unequal bounds and likelihood not greater than
the FP32 value `1e-5` on either backend; domain guards are reported separately.
Exit `1` denotes invalid evidence or an ordinary tolerance failure. Exit `2`
retains any other bit differences for review, including sensitive results,
nonfinite classifications, NaN payloads and within-tolerance differences.
Exit `0` means exact agreement only on the recorded inputs. No tail difference
is automatically accepted and no global mathematical error bound is claimed.

To regenerate Metal observations after relevant source changes:

```sh
bash scripts/test.sh --native metal_aft_source
MTL_DEBUG_LAYER=1 results/metal_aft_source_test \
  xgboost/src/common/metal_histogram.metal --report-prefix results/new-aft
python3 scripts/compare_aft_contracts.py inspect results/new-aft.metal.bin
```

The portable binary format uses explicit little-endian words. The header is
`XGBAFT1\0`, followed by uint32 backend kind (`0` input, `1` Metal, `2` CUDA) and
case count. Each case has five uint32 words: distribution, mode, scale FP32 bits,
row count and weight count. FP32 bit arrays follow for lower bounds, upper
bounds, predictions and weights. Observation files then store each row's three
FP64 fields (gradient, Hessian, likelihood) and two FP32 weighted outputs. The
footer is `AFTE` plus uint32 case count; trailing bytes are rejected. Signed
zeros, subnormals and nonfinite payloads survive interchange. Gzip is packaging
only; the CUDA reader consumes the uncompressed input-only file.

## Exact RNG streams and ranking pairs

`cuda_rng_contracts` uses shared generators for the entire existing Metal corpus:

| Corpus | NVIDIA execution | Exact checks |
| --- | --- | --- |
| 1,060,276 Philox queries | Vendored `DefaultRng` and production `RandomWeight` | Every engine word and FP32 uniform-draw bit pattern, including 64-bit discard boundaries |
| 41,200 seed/range/discard cases; 659,200 draws | Vendored `UniformIntDistribution<size_t>` using an instrumented production engine | Draw, consumed engine position and next engine word; singleton ranges and rejection sampling retained |
| 74 cases; 9,068,696 raw ranking pairs | Production `RankingCache`, prediction/label sorts and `MakePairsOp<false>` | Every prediction rank and both indices of every pair; seed/group wrap, homogeneous labels, empty queries and the large reused-batch Metal case retained |

The shared fixture header preserves the old inputs bit for bit, including signed
zeros and subnormal labels. It uses standard integer engines, without
implementation-defined standard random distributions. The Metal pair test still
passes labels through its stride-three view. CUDA's `MetaInfo` stores the same
target as one contiguous column; padding is not treated as label data. The
empty case checks the production CUDA cache's zero-work metadata and launches
no pair kernel. This does not compare the complete ranking gradient computation.

```sh
results/cuda-comparison-build/cmake/cuda_rng_contracts results/cuda-rng-report.json
```

Success requires **every actual GPU result** to match the shared independent
host oracle exactly. This is the same oracle against which the Metal results
were checked; expected values never feed the CUDA kernels. The JSON summary
records counts, each completed pair case, hardware, compiler/CCCL versions and
the selected RNG branch. It does not contain full output arrays. A mismatch
returns failure with its field, row, actual and expected values; retain the log
and report, including any partial failure record, alongside `build.json`.
That build record identifies the fixture, oracle, harness and production source
hashes. Use matching source inputs on both machines; there is no separate
tolerance-based report comparator for this exact contract.

Compilation requires the modern `cuda::std::philox4x64` branch and rejects an
older Thrust fallback. The Philox/`RandomWeight` portion also contributes evidence
for training RNG, but the separate unweighted column shuffle and broader
sampling caller comparisons remain open. It therefore adds one complete
contract runner, P5-N06. No NVIDIA result exists yet. The
[M3 record](../validation/rng-contract-m3-run.json) contains the three passing
Metal groups and the input-preservation check. To rerun those groups:

```sh
bash scripts/test.sh --native metal_cuda_rng metal_ranking_rng metal_ranking_pairs_api
```

## Production sketch, training and prediction comparison

`scripts/compare_gpu_pipeline.py` complements the numerical witnesses with fresh
ordinary DMatrix inputs through production training, exported cuts and prediction.
It requires a matching production-tree/library build record and rejects CPU
fallback. It records all cut bits, predictive tree fields by path, sampled margin
bits, input hashes, hardware, build information and the exact harness-file hash.
Each GPU model is also evaluated on CPU at the original `2e-5` tolerances.
This checks prediction of the same model; it does not assert that separately
trained CPU and GPU models must be identical.

The `fixtures-v1` suite has eight cases: dense, missing/empty-column, categorical,
row-weighted, subnormal-weighted, zero-weighted, group-weighted ranking and
Hessian-weighted `approx`. Integer-generated inputs and binary fractions avoid
RNG/library-math variation; cross-platform input hashes must still agree.
Each case trains three trees. `higgs-v1` uses all 10.5M training rows for the
ordinary sketch path, trains one tree and records 1,024 held-out predictions.
It is a correctness experiment, not a performance benchmark or a replay of all
1,000 training rounds. Prepare HIGGS using `scripts/prepare_published_data.py`.

After the CUDA build above, use an environment with NumPy and SciPy:

```sh
python3 scripts/compare_gpu_pipeline.py run --device cuda \
  --build-record results/cuda-comparison-build/build.json \
  --output results/gpu-pipeline-fixtures-cuda.json
python3 scripts/compare_gpu_pipeline.py run --device cuda --suite higgs-v1 \
  --build-record results/cuda-comparison-build/build.json \
  --output results/gpu-pipeline-higgs-cuda.json
python3 scripts/compare_gpu_pipeline.py compare \
  validation/gpu-pipeline-fixtures-m3.json results/gpu-pipeline-fixtures-cuda.json \
  --output results/gpu-pipeline-fixtures-comparison.json
python3 scripts/compare_gpu_pipeline.py compare \
  validation/gpu-pipeline-higgs-m3.json results/gpu-pipeline-higgs-cuda.json \
  --output results/gpu-pipeline-higgs-comparison.json
```

Run the same harness source on both machines. Exit codes are `0` for identical
recorded outputs, `2` for differences requiring review, and `1` for invalid or
mismatched evidence. Changed cuts, leaf weights and margins remain visible;
there is no automatic waiver for floating-point differences. Negative-infinity
bin lower bounds and split thresholds are retained as bits: these are valid
source-defined boundaries, not nonfinite prediction failures.

The [M3 fixture observations](../validation/gpu-pipeline-fixtures-m3.json) and
[HIGGS observations](../validation/gpu-pipeline-higgs-m3.json) are complete on
production `68c800e`. All sampled GPU predictions match CPU evaluation of the
same model bit for bit. HIGGS dispatched five sketch batches, and its cuts and
first predictive tree exactly reproduce the earlier full benchmark. The
[run record](../validation/gpu-pipeline-m3-run.json) includes hashes, commands,
the initial validator error and 18 passing synthetic integrity tests. NVIDIA
results are still absent. These integration cases do not close the eleven
remaining numerical contracts or cover external memory, native device inputs,
distributed execution and all other caller combinations.
