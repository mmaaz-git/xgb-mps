# Running the shared numerical witnesses on NVIDIA

VERIFY-2 requires actual GPU execution. The current native harness,
`numerical-v2`, covers **P5-N01, P5-N08, P5-N13 and P5-N14**, using the same 39
inputs as the Metal numerical-contract test. A second native harness covers
**P5-N02**, using the complete AFT corpus described below. A third covers
**P5-N06** ranking RNG and raw pairs plus **training-rng-modern** streams and
unweighted column sampling. A fourth exports the existing **P5-N03** primitive
math corpus. A fifth adds **P5-N04/05** observations from the full existing
ranking-rounding, ranking-gradient, repeated unbiased-objective and primitive
rounding boundary corpora. The numerical executable also has a **P5-N07** mode
for the complete statistics API corpus, a separate linear-algebra mode, and a
root/stump mode preserving all 405 existing reduction cases, and a sampling
mode preserving the complete current CUDA-derived sampling corpus.
The integer leaf and cache modes retain every original page, weight and lifecycle check.
A sixth executable covers the complete metric-term/reduction corpus, including
its previously uncounted math, label and empty-input checks.
A seventh executable adds the full linear operation corpus, extending
**training-reduction-association** coverage. The other five numerical contracts and
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

The build driver first compiles all six harness objects using the configured compiler,
CCCL, architectures and arithmetic flags. It then checks/rebuilds production
dependencies and links the executables. This exposes harness compiler errors
before the expensive library build. It writes `build.json`, including terminal
status; CI separates the object and link phases and rejects changed inputs
between them. Git metadata reads trust only the selected checkout for that
invocation, accommodating container checkout ownership. The CI's
`scripts/ci_cuda_build.sh` does this against its CUDA/NCCL/nvCOMP/federated build,
but **does not run the final GPU command**.
When production enables nvCOMP, the consumer imports that same CMake package,
resolves the actual shared object behind its development symlink, and adds that
runtime directory to the harness search path. Each executable retains a direct
nvCOMP dependency because ELF RUNPATH is not transitive. A small address-only
host executable checks library loading before the production build; it calls
neither nvCOMP nor CUDA functions. The resolved paths are preserved in CI evidence.

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

`cuda_rng_contracts` (`rng-v2`) uses shared generators for the existing Metal RNG
and ranking corpus and the unweighted column-sampling cases:

| Corpus | NVIDIA execution | Exact checks |
| --- | --- | --- |
| 1,060,276 Philox queries | Vendored `DefaultRng` and production `RandomWeight` | Every engine word and FP32 uniform-draw bit pattern, including 64-bit discard boundaries |
| 41,200 seed/range/discard cases; 659,200 draws | Vendored `UniformIntDistribution<size_t>` using an instrumented production engine | Draw, consumed engine position and next engine word; singleton ranges and rejection sampling retained |
| 74 cases; 9,068,696 raw ranking pairs | Production `RankingCache`, prediction/label sorts and `MakePairsOp<false>` | Every prediction rank and both indices of every pair; seed/group wrap, homogeneous labels, empty queries and the large reused-batch Metal case retained |
| 128 unweighted column cases; 1,320,913 permutation entries | Thrust shuffle with its separate `default_random_engine`, followed by prefix retention and GPU sorting, as in `random.cu` | Full permutations and 646,087 selected IDs; includes the million-column case, unsigned feature-ID boundaries and empty input |
| 15 unweighted tree/level/node hierarchies; 210 outputs | Actual production `ColumnSampler`, including its CUDA initialization and `SampleFeature` consumer | Selected IDs, cached-level behavior across two trees, and exact parent RNG consumption |

The shared fixture headers preserve the old inputs bit for bit, including signed
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
for training RNG. The unweighted shuffle primitive is a separate execution of
the source's Thrust calls, not a capture of a private `ColumnSampler` temporary;
the hierarchy cases invoke the actual production consumer. Weighted selection's
log-key arithmetic and broader sampling caller combinations still need comparison.
Together these provide runners for P5-N06 and training-rng-modern; no NVIDIA
result exists yet. The
[M3 record](../validation/rng-contract-m3-run.json) contains the three passing
Metal RNG groups and their input-preservation check. The
[column extension record](../validation/rng-column-contract-m3-run.json) adds the
unchanged column matrix and full passing Metal column group (including its
additional weighted/local storage checks). Those additional checks are not
claimed as new CUDA weighted-selection evidence. To rerun the four groups:

```sh
bash scripts/test.sh --native metal_cuda_rng metal_ranking_rng metal_ranking_pairs_api metal_column_sampling_api
```

## Primitive arithmetic, casts and math functions

`cuda_math_contracts` consumes the exact input bits exported by three existing
Metal tests. Random inputs generated by macOS standard-library distributions
are exported, not regenerated on Linux. The [input file](../validation/math-contract-inputs-m3.bin.gz)
contains no expected outputs; the separate
[Metal observations](../validation/math-contract-metal-m3.bin.gz) retain every result.

| Group | Inputs | Observed output fields |
| --- | --- | ---: |
| Binary64 arithmetic and narrowing | 100,196 pairs | 400,784 add/subtract/multiply/divide results and 200,392 FP32 casts |
| Widening | 100,007 FP32 values | 100,007 binary64 casts |
| Binary64 math | 80,060 values | 240,180 exp/log/erf results |
| FP32 math | 500,400 pairs | 1,000,800 hypotf/tanhf results |

The NVIDIA kernel evaluates native arithmetic, casts and libdevice functions on
these inputs. It records **1,942,163 output fields** without substituting CPU
expectations. This is primitive coverage for P5-N03, not a full objective/metric
comparison or proof of P5-N12's compiler/contraction contract.

```sh
python3 scripts/compare_math_contracts.py unpack \
  validation/math-contract-inputs-m3.bin.gz --output results/math-inputs.bin
results/cuda-comparison-build/cmake/cuda_math_contracts \
  results/math-inputs.bin results/math-cuda.bin
python3 scripts/compare_math_contracts.py compare \
  validation/math-contract-metal-m3.bin.gz results/math-cuda.bin \
  --output results/math-comparison.json
```

Keep `math-cuda.bin.json`, the runtime log and matching `build.json` with both
observation files. The records identify the actual GPU, driver/runtime,
NVCC/CCCL versions, compiler flags and source/library hashes. The comparator
requires identical full input bits and complete framing; it does not authenticate
the runtime or build provenance independently. NumPy is its only nonstandard
Python dependency. Use new output paths.

Arithmetic and casts require exact bits, except that both-NaN payload differences
are reported and allowed under the existing primitive contract. For math
functions, NaN/Inf classification changes fail; all other non-NaN bit differences
require review, including signed zero and one-ULP differences. Exit `1` means
invalid evidence or a contract failure, `2` means math differences require
review, and `0` means agreement under this contract on these recorded inputs.
Counts, finite ULP maxima and sample bit patterns accompany the complete source
files' hashes. Passing CPU bounds are not added together into a new GPU allowance.

The [M3 run record](../validation/math-contract-m3-run.json) retains the unchanged
CPU checks: exact arithmetic/casts with the stated NaN policy; exp/log/erf maxima
of 1/3/3 binary64 ULP against the original four-ULP budget; hypot/tanh maxima of
0/0 against the original one-FP32-ULP budget. The adaptive test's 12 source cases
also pass, but their 2,359,332 fields are not exported by this primitive runner.
Sixteen synthetic integrity tests and a byte-identical C++ input round trip pass.
These sampled results establish neither a universal error bound nor NVIDIA agreement.

To regenerate the Metal observations after relevant changes:

```sh
bash scripts/test.sh --native metal_fp64 metal_transcendental metal_adaptive_source
MTL_DEBUG_LAYER=1 results/metal_fp64_test \
  xgboost/src/common/metal_histogram.metal --report results/new-math-fp64.bin
MTL_DEBUG_LAYER=1 results/metal_transcendental_test \
  xgboost/src/common/metal_histogram.metal --report results/new-math-transcendental.bin
MTL_DEBUG_LAYER=1 results/metal_adaptive_source_test \
  xgboost/src/common/metal_histogram.metal --report results/new-math-adaptive.bin
python3 scripts/compare_math_contracts.py pack \
  results/new-math-fp64.bin results/new-math-transcendental.bin results/new-math-adaptive.bin \
  --inputs results/new-math-inputs.bin.gz --metal results/new-math-metal.bin.gz
```

The little-endian format starts with `XGBMAT1\0`, uint32 backend kind (`0` inputs,
`1` Metal, `2` CUDA) and a four-bit group mask. Included groups appear in ascending
order, each with four uint32 words (ID, fixed row count, input width, output width),
then row-major uint64 input cells and, for observations, output cells. FP32 cells
use only the low 32 bits. `MEND` and the uint32 mask finish the file; trailing
bytes are rejected. Partial Metal exports use masks 3, 4 and 8; packing and CUDA
execution require all groups. Gzip is packaging only.

## Ranking rounding, gradients and bias updates

`cuda_ranking_contracts` consumes the exact inputs from all 64 cases in
`metal_ranking_rounding_api`: pairwise/MAP, top-k/mean sampling, normalization
on/off, largest queries of 257/65,537 rows and four existing bias/label modes.
The empty queries, soft binary labels, extreme weights, zero bias and small-bias
cases remain. The [inputs](../validation/ranking-rounding-inputs-m3.bin.gz)
contain predictions, labels, five group weights and the initial binary64 bias
vectors. They contain no expected outputs.

The [Metal export](../validation/ranking-rounding-metal-m3.bin.gz) retains
**6,345,280 result/diagnostic fields**: 4,227,328 gradient/Hessian fields,
2,113,664 MAP prefix fields, 2,560 updated bias values, 1,600 query-statistic
fields and 128 scalar fields. The [run record](../validation/ranking-rounding-m3-run.json)
includes unchanged checks for 121 untracked-cost maxima, 16 combined-weight
witnesses and 48 global factors different from the last query's factor.
This existing low-level Metal test supplies the CPU cache's `WeightNorm`;
the CUDA consumer computes its own cache normalization from the same weights.

CUDA invokes the vendored `LambdaRankGetGradientPairwise/MAP` and
`LambdaRankUpdatePositionBias`, with production caches, sorts, pair generation,
CUB reductions and atomic accumulation. It reads the actual cached gradient
rounding factors, shared cost factor and MAP prefixes. Query normalization,
combined weights and individual cost factors are **separate source-expression
evaluations using retained production reduction results**; they are not captures
of otherwise private temporary expressions or overwritten query writes.

Each smaller case executes 32 times with freshly initialized cache, gradients,
cost buffers and bias vectors; each larger case executes once: **1,056 planned
CUDA executions**. Every repetition's full outputs are retained. The CUDA source
allows multiple queries to write its shared cost factor. Metal's declared
policy is the deterministic maximum, so the comparator records observed CUDA
winners and distinct complete outputs without demanding a unique race winner.
These repetitions cannot enumerate all possible race outcomes.

```sh
python3 scripts/compare_ranking_contracts.py unpack \
  validation/ranking-rounding-inputs-m3.bin.gz --output results/ranking-inputs.bin
results/cuda-comparison-build/cmake/cuda_ranking_contracts \
  results/ranking-inputs.bin results/ranking-cuda.bin
python3 scripts/compare_ranking_contracts.py compare \
  validation/ranking-rounding-metal-m3.bin.gz results/ranking-cuda.bin \
  --output results/ranking-comparison.json
```

Retain `ranking-cuda.bin.json`, the runtime log and matching `build.json` with
both reports. The comparison enforces identical input bits, case order, shapes,
all repetitions and exact EOF. Hardware provenance still requires reviewing
the build/runtime records; valid binary framing does not authenticate GPU execution.

The existing rounding-test limits remain: gradients use `2e-5 * max(1, abs(ref))`,
MAP prefixes and comparable bias updates `2e-12`, normalization `2e-7`, and
gradient/Hessian rounding plus combined FP32 weights require exact bits.
All bit differences, including those
within these limits, remain review items. Shared-factor differences and their
dependent bias outputs require review, as do the zero-bias cases whose cost
rounding uses nonfinite `frexp` inputs with unspecified NVIDIA exponents.
These cases do not waive gradient or prefix checks. Exit `1` means invalid
evidence or a declared comparison failure, `2` means differences require review,
and `0` means bit-identical outputs on these recorded executions.

This mode covers the complete existing rounding corpus. The public-objective
mode below covers the gradient and unbiased suites, including NDCG and repeated
state. A separate primitive mode below retains the rounding boundary test.
No NVIDIA execution result is available yet. Fourteen synthetic integrity tests,
a C++ input round trip and the unchanged Metal rounding checks pass locally.

To regenerate the Metal exports:

```sh
bash scripts/test.sh --native metal_ranking_rounding_api
MTL_DEBUG_LAYER=1 results/metal_ranking_rounding_api_test --report-prefix results/new-ranking
```

The little-endian format starts with `XGBRND1\0`, uint32 backend kind (`0` inputs,
`1` Metal, `2` CUDA) and 64 cases. Each case stores eight uint32 words: objective,
mean flag, normalization flag, largest query, mode, rows, bias positions and
repetition count. Input arrays follow as FP32 prediction/label/weight bits and
FP64 initial bias bits. Each observation starts with a uint32 repetition index,
two FP64 scalars, five query records (four FP32 fields and one FP64 factor),
row-major FP32 gradients, row-major FP64 MAP prefixes if applicable, and both
FP64 updated bias vectors. `RNDE` plus the uint32 case count finishes the file.
The fixed corpus uses eight pairs, seed `0xffffffff`, bias norm `1.5` and group
pointers `[0, 0, 128, 128, 128+largest, 129+largest]`.

## Public ranking objective sequences

`cuda_ranking_contracts --objectives` replays every call from
`metal_ranking_gradient_api` and `metal_ranking_unbiased_api` through the actual
CUDA `ObjFunction` registry, configuration, gradient and serialization APIs.
The [run record](../validation/ranking-objective-m3-run.json) retains exact input
bits, source/library hashes and the unchanged native checks.

| Corpus | Calls / sequences | Rows | Observations |
| --- | --- | --- | --- |
| Gradient | 1,442 / 1,442 | 2,444,200 | 4,888,400 gradient/Hessian fields; 1,442 parent RNG words |
| Unbiased | 2,596 / 866 | 2,949,920 | 5,899,840 gradient/Hessian fields; 118,528 serialized bias fields; 2,596 parent RNG words |

Both corpora retain pairwise/NDCG/MAP, top-k/mean, normalization settings, query
sizes, weights and label modes. The totals include two gradient calls with
327,680 rows each and four unbiased calls with 524,288 rows each. The ordinary
unbiased sequences each retain three iterations with their original changing
predictions; the two batched sequences retain two iterations each.

CUDA creates a fresh objective and `MetaInfo` only at the start of a sequence.
Subsequent calls reuse both, preserving the objective's actual binary64 bias
state. Input files contain predictions, labels, query boundaries, weights,
parameters and seeds; **no Metal bias state or expected outputs are supplied**.
Iteration zero initializes bias through production code. Results must be CUDA
resident before host observation. Each call checks parent RNG consumption:
one draw for mean sampling, none for top-k. Reading the next word from an engine
copy leaves the real engine unchanged.

Run both corpora after compiling the extended ranking executable:

```sh
python3 scripts/compare_ranking_objectives.py unpack \
  validation/ranking-gradient-inputs-m3.bin.gz --output results/ranking-gradient-inputs.bin
results/cuda-comparison-build/cmake/cuda_ranking_contracts --objectives \
  results/ranking-gradient-inputs.bin results/ranking-gradient-cuda.bin
python3 scripts/compare_ranking_objectives.py compare \
  validation/ranking-gradient-metal-m3.bin.gz results/ranking-gradient-cuda.bin \
  --output results/ranking-gradient-comparison.json

python3 scripts/compare_ranking_objectives.py unpack \
  validation/ranking-unbiased-inputs-m3.bin.gz --output results/ranking-unbiased-inputs.bin
results/cuda-comparison-build/cmake/cuda_ranking_contracts --objectives \
  results/ranking-unbiased-inputs.bin results/ranking-unbiased-cuda.bin
python3 scripts/compare_ranking_objectives.py compare \
  validation/ranking-unbiased-metal-m3.bin.gz results/ranking-unbiased-cuda.bin \
  --output results/ranking-unbiased-comparison.json
```

Use new output paths. Retain both `.bin.json` runtime metadata files, logs and
the matching `build.json`. The reader rejects observation files as CUDA inputs,
changed label/weight bits within a sequence, missing/reordered calls, malformed
shapes and incomplete/trailing data. The comparator requires matching input
bits and exact parent RNG words. Ordinary gradient and serialized bias budgets
remain `2e-6 + 2e-5 * abs(Metal)`. The gradient suite additionally retains its
strict `2 * 2^-149 + 2e-5 * abs(Metal)` budget whenever either output is a
nonzero subnormal. Classification changes fail; every other bit difference
remains a review item, including signed zeros and differences within tolerance.
Exit codes are `1` for invalid evidence or failed checks, `2` for other
differences requiring review, and `0` for identical recorded outputs.

`SaveConfig` exposes FP32 bias arrays, not the full internal binary64 state.
The report identifies the first differing iteration and whether the preceding
serialized bias differs, without treating matching serialization as proof of
identical internal state. CUDA's shared-factor race can affect later unbiased
iterations; it does not automatically waive a failed comparison. This mode
executes each call once; the separate rounding mode provides repeated fresh
executions for race characterization.

Both full Metal groups and 17 synthetic report-integrity tests pass. Complete
C++ input round trips preserve every byte; an independent host engine matches
all 4,038 recorded parent RNG words. The original ordinary unbiased checks
assert 118,368 serialized bias fields; the additional 160 batched bias fields
are observations without a separate native bias oracle. CUDA compilation/linking
passed in [run 42073244457](https://github.com/mmaaz-git/xgb-mps/actions/runs/42073244457);
actual NVIDIA execution remains pending. Reloads, changed
metadata shapes and other input/state combinations remain broader caller work.

To regenerate the Metal observations with the original checks enabled:

```sh
bash scripts/test.sh --native metal_ranking_gradient_api metal_ranking_unbiased_api
MTL_DEBUG_LAYER=1 results/metal_ranking_gradient_api_test --report-prefix results/new-gradient
MTL_DEBUG_LAYER=1 results/metal_ranking_unbiased_api_test --report-prefix results/new-unbiased
```

The versioned little-endian format is defined in `tests/ranking_objective_io.h`
and independently checked by `scripts/compare_ranking_objectives.py`. Its
`XGBROB1\0` header identifies backend kind, corpus and complete call count;
each call carries a fixed parameter identity and all input words. Observations
append FP32 gradient/Hessian and serialized bias words plus the next parent RNG
word. `ROBE`, corpus ID and call count terminate the file at exact EOF. Gzip is
packaging only. Valid framing alone does not authenticate GPU execution.

## Ranking primitive rounding boundaries

`cuda_ranking_contracts --primitives` consumes all **162,851 existing cases**
from `metal_ranking_rounding`: powers and neighboring values, subnormals,
overflow, zero bias, large row counts and random bit patterns. It invokes the
vendored `CreateRoundingFactor<float/double>` and `TruncateWithRounding` functions
with the production caller's bound expressions. The
[run record](../validation/ranking-primitive-m3-run.json) retains all **977,106
original result fields**, exact native checks and the same 7,284 FP32 / 2,400
binary64 asymmetric truncation witnesses.

The export adds two delta diagnostics per case: separate evaluations of the
source expression passed to `frexp`, not captures of private temporaries.
Metal explicitly uses exponent zero for nonfinite inputs; NVIDIA leaves that
exponent unspecified. CUDA executes its source without adding Metal's guard.
Both diagnostics must be nonfinite before factor differences enter policy
review. A truncation difference inherits that review only when actual factors
also differ; equal-factor truncations still require exact results. This does
not prove the cause of every dependent difference.

```sh
python3 scripts/compare_ranking_primitives.py unpack \
  validation/ranking-primitive-inputs-m3.bin.gz --output results/ranking-primitive-inputs.bin
results/cuda-comparison-build/cmake/cuda_ranking_contracts --primitives \
  results/ranking-primitive-inputs.bin results/ranking-primitive-cuda.bin
python3 scripts/compare_ranking_primitives.py compare \
  validation/ranking-primitive-metal-m3.bin.gz results/ranking-primitive-cuda.bin \
  --output results/ranking-primitive-comparison.json
```

Use fresh output paths and preserve `.bin.json`, runtime logs and the matching
`build.json`. The comparator requires the complete corpus and exact input bits.
Finite-domain results and diagnostics require exact bits, except that pairs of
NaNs retain their payload/sign differences for review. Signed-zero mismatches
fail exact checks. Diagnostic classification or finite/nonfinite disagreements
are not waived. Exit `1` means invalid evidence or an exact-check failure,
`2` means other differences require review, and `0` means identical recorded
outputs. None of these outcomes establishes behavior on untested inputs.

The full Metal test and 17 synthetic report tests pass. The C++ input round trip
is byte-identical. New Metal diagnostics match independent host expressions
apart from 405 FP32 and 96 binary64 NaN payload/sign differences. Their nonfinite
counts are 19,602 and 25,959 respectively. These host checks are not NVIDIA
execution. CUDA compilation/linking passed in
[run 42079045498](https://github.com/mmaaz-git/xgb-mps/actions/runs/42079045498);
runtime comparisons remain pending.

```sh
bash scripts/test.sh --native metal_ranking_rounding
MTL_DEBUG_LAYER=1 results/metal_ranking_rounding_test \
  xgboost/src/common/metal_histogram.metal --report-prefix results/new-ranking-primitive
```

`tests/ranking_primitive_io.h` specifies the little-endian `XGBRPR1\0` format:
backend kind and fixed case count, all input fields without native struct
padding, then eight observed fields per case for Metal/CUDA reports. `RPRE`
and the case count finish the file at exact EOF. CUDA rejects observation files
as input. Gzip is packaging only; framing does not authenticate GPU provenance.

## Statistics reductions, means and medians

`cuda_numerical_contracts --statistics` replays the full `metal_stats_api` corpus
through public `Median`, `SampleMean`, strided `Mean`, `WeightedSampleMean`,
`Reduce` and `SumOptionalWeights` APIs. The
[record](../validation/stats-contract-m3-run.json) contains **122 cases / 515
observed fields**: 56 matrix cases, 63 generic/optional-weight reduction cases,
two precision witnesses and one scalar-weight case. Every input bit is retained,
including empty, nonfinite, subnormal, zero-weight and wide-exponent cases.

CUDA generic sums use `plus<float>` despite returning double. Optional-weight
sums use binary64 accumulation; scalar optional weights use the source's host
multiply. Means also have distinct term and accumulator precision. The replay
uses those APIs directly, transfers array weights to CUDA, and retains host
publication where the source returns a scalar. Nonempty tensor results must be
CUDA resident before observation. Expected weighted-mean rejections must carry
the source's zero-total-weight error; unrelated exceptions fail the run.

```sh
python3 scripts/compare_stats_contracts.py unpack \
  validation/stats-contract-inputs-m3.bin.gz --output results/stats-inputs.bin
results/cuda-comparison-build/cmake/cuda_numerical_contracts --statistics \
  results/stats-inputs.bin results/stats-cuda.bin
python3 scripts/compare_stats_contracts.py compare \
  validation/stats-contract-metal-m3.bin.gz results/stats-cuda.bin \
  --output results/stats-comparison.json
```

The Metal observations include the original CPU reference values and compensated
sum/magnitude diagnostics. **CUDA input files contain no references or expected
outputs.** The comparator checks both GPU observations independently against
the references, even when the GPUs agree with each other. It preserves the
original FP32 mean tolerance, exact median checks, reduction classification
checks and operation-specific forward-error bounds. Generic sums must also be
widened FP32 values. All GPU-to-GPU bit differences remain review items,
including differences within bounds and permitted NaN payload/signed-zero
differences. The two-`FLT_MAX` optional-weight witness additionally checks the
known exact binary64 sum; the original native test checked that it was finite.

For wide-exponent generic sums, the original bound is
`steps * eps / (1 - steps * eps) * sum_abs`, where `eps = 2^-23` and each repeated
`ceil(n / 256)` level contributes eight steps. Optional-weight sums retain
`32 * 2^-52 * sum_abs`. Other reduction modes retain exact numerical equality
or matching NaN classification. The generic sum also checks the compensated
reference in the original modes. These are the corpus's accepted bounds, not
a universal error guarantee for every CUB architecture or reduction tree.

Both original Metal execution and export mode pass, as do 17 synthetic integrity
tests. All 515 exported Metal fields satisfy the comparator. C++ input round
trip and host API type checks pass. CUDA compile/link passed in
[run 42079045498](https://github.com/mmaaz-git/xgb-mps/actions/runs/42079045498);
runtime validation remains pending. This statistics replay does not cover the separate linalg, metric or
training-reduction corpora.

```sh
bash scripts/test.sh --native metal_stats_api
MTL_DEBUG_LAYER=1 results/metal_stats_api_test --report-prefix results/new-stats
```

Use fresh output paths and retain `.bin.json`, runtime logs and `build.json`.
The `XGBSTA1\0` little-endian format in `tests/stats_contract_io.h` requires all
122 parameter identities in order, complete input/observation arrays and exact
EOF after `STAE` plus the case count. FP32 output/status words have zero upper
32 bits; reductions retain binary64 words. Only Metal reports carry reference
words and diagnostics. Exit `1` means invalid evidence or failed reference
checks, `2` means other bit differences require review, and `0` means identical
recorded GPU outputs that also pass those checks. Framing does not authenticate
hardware execution.

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
results are still absent. These integration cases do not close the numerical
contracts or cover external memory, native device inputs,
distributed execution and all other caller combinations.

## Linear algebra and packed histograms

`cuda_numerical_contracts --linalg` replays the complete `metal_linalg_api` corpus:
141 histograms, 80 vector operations, 33 mean cases, three invalid-input guards,
one packed-label/default-weight case and two previously uncounted empty
operations. The [record](../validation/linalg-contract-m3-run.json) contains
**260 records / 10,508,701 Metal output fields**. The original native checks still
report 258 cases / 10,499,275 checked fields; the export also records guard storage
and the additional observations. The single original RNG stream, parameter order,
edge values and binary64 scalar parameters are retained.

```sh
python3 scripts/compare_linalg_contracts.py unpack \
  validation/linalg-contract-inputs-m3.bin.gz --output results/linalg-inputs.bin
results/cuda-comparison-build/cmake/cuda_numerical_contracts --linalg \
  results/linalg-inputs.bin results/linalg-cuda.bin
python3 scripts/compare_linalg_contracts.py compare \
  validation/linalg-contract-metal-m3.bin.gz results/linalg-cuda.bin \
  --output results/linalg-comparison.json
```

CUDA calls `SmallHistogram`, `VecScaMul`, `VecScaDiv`, `LogE` and `Mean` directly.
For the two batched mean cases it invokes the source FP32 `cuda_impl::Mean` once
per column, retaining strided input/output storage. Subtraction invokes the
multiclass initialization source's `TransformKernel` expression, `v - d_mean(0)`;
it is not a replay of the complete objective.

There are two explicit domain/layout adaptations:

- CUDA histogram sorting reads contiguous keys, whereas the Metal tests include
  padded, strided labels. A GPU gather projects the logical labels to contiguous
  CUDA storage, checks every gathered bit, then calls the unchanged CUDA helper.
  Original backing inputs and weights are checked unchanged. This establishes a
  common logical input; it does not claim CUDA strided-label support. Output
  strides and guards are retained.
- The frozen audit's NUM-OP-11 defines nonfinite-label and excessive-run rejection
  as Metal extensions outside the CUDA contract. All three cases and their Metal
  rejection observations remain in the corpus. CUDA marks them **not executed**
  and writes no output values for them. They are never counted as CUDA passes.

Both GPUs are independently checked against the Metal report's original vector
references and reduction sum/magnitude/count diagnostics. None of these references
is present in CUDA input files. Scale, division and subtraction retain zero-ULP
checks; logarithm retains its one-ULP budget and original nonfinite semantics.
Reductions retain the original exact modes or
`gamma(ceil(n / 256) + 8, 2^-23) * sum_abs` bound. The cancellation witness retains
its Metal-specific zero assertion; CUDA receives the association bound, not a
requirement to reproduce Metal's reduction tree. Output guard storage is
additionally required bit-exact, including the vector logarithm case where the
original general one-ULP check also covered guards.

Every GPU bit difference requires review, even if both results pass the reference
check. Comparison exits 1 for malformed reports/reference failures and 2 for
review, including the three domain exclusions even if all common-domain fields
are identical. Counts include every differing field; diagnostic JSON shows up to
32 examples per case/policy, with full bits retained in the binary reports.
No overall full-execution or full-parity success is inferred from exclusions.

Both Metal modes, all exported Metal reference checks and 16 synthetic integrity
tests pass. C++ input round-trip and host view/API type checks pass. CUDA compile/link
passed in [run 42083981584](https://github.com/mmaaz-git/xgb-mps/actions/runs/42083981584);
actual NVIDIA execution remains pending. No production code or
original test tolerances changed.

```sh
bash scripts/test.sh --native metal_linalg_api
MTL_DEBUG_LAYER=1 results/metal_linalg_api_test --report-prefix results/new-linalg
python3 scripts/compare_linalg_contracts.py validate-metal results/new-linalg.metal.bin
```

Use fresh output paths and retain `.bin.json`, runtime logs and `build.json`.
The little-endian `XGBLIN1\0` format requires all 260 fixed identities in order,
all backing inputs and actual outputs, backend-specific execution statuses,
the completion marker and exact EOF. Metal-only reference diagnostics serialize
binary64 sums/magnitudes and the original FP32 cast separately; the published
observations were produced on the M3, where `long double` is binary64.

## Metric terms, reductions and guards

`cuda_metric_contracts` consumes the complete existing `metal_metric_api` corpus.
The [record](../validation/metric-contract-m3-run.json) retains **375 records**:
287 ordinary metric cases, 32 quantile/expectile cases, 30 survival/ordered-NaN
cases, one 100,057-input math corpus, twelve invalid-label checks and thirteen
empty calls. Original generators, ordering and tolerances are unchanged. The
native test still reports 349 counted cases and 15,628,618 row checks; the export
also retains 500,285 math results, 724 sum fields and every status.

```sh
python3 scripts/compare_metric_contracts.py unpack \
  validation/metric-contract-inputs-m3.bin.xz --output results/metric-inputs.bin
results/cuda-comparison-build/cmake/cuda_metric_contracts \
  results/metric-inputs.bin results/metric-cuda.bin
python3 scripts/compare_metric_contracts.py compare \
  validation/metric-contract-metal-m3.bin.xz results/metric-cuda.bin \
  --output results/metric-comparison.json
```

The CUDA executable compiles the unchanged elementwise, multiclass and survival
source policies/reducers through `tests/cuda_metric_source.h`. Registration
expressions populate private, unregistered entries; hidden symbol visibility
prevents these definitions from replacing library symbols. The build driver
compares registered names/factory types with the ordinary numerical executable
and checks the metric executable's defined dynamic symbols. These are host-only
checks, separate from GPU execution. A host prototype passes on the M3; the
six CUDA objects, all 197 production steps and six links completed in
[run 42092645863](https://github.com/mmaaz-git/xgb-mps/actions/runs/42092645863).
The run then failed before the registry check entered main: Linux could not load
`libnvcomp.so.5`. Its failed log and exact source hashes are retained. The runtime
search/dependency fix and early loader probe passed in
[run 42100004279](https://github.com/mmaaz-git/xgb-mps/actions/runs/42100004279),
along with all six consumers and the 24-entry host registry/factory and symbol
isolation checks. NVIDIA execution remains unvalidated.

Ordinary row policies and multiclass/survival reducers use the actual source
functions. Pseudo-Huber, quantile and expectile terms separately evaluate the
source expressions from their metric methods, then use the actual source
reducer. They are not captures of those methods' private temporaries. Math
observations invoke the CUDA float functions and `common::LogGamma`. Empty
cases exercise source reducer boundaries without evaluating a loss or its
constructor parameters; this does not claim constructor-parameter equivalence.

Four finite invalid-label cases exercise CUDA's actual rejection path. Under
the frozen audit's **NUM-OP-36**, the other eight use nonfinite or out-of-int32
casts outside CUDA's defined contract. Their Metal checks and exact inputs are
retained; CUDA marks them **not executed**, without fabricated sums. The planned
CUDA execution therefore covers 367 cases, with eight explicit exclusions.

Both GPUs are independently checked against the original CPU row/math references.
Ordinary loss budgets remain zero or four ULP as specified by the native test;
weights and quantile/expectile terms retain zero ULP. Math retains its
`1/1/1/0/6` log/log1p/exp/sqrt/lgamma budgets and explicit signed-zero checks.
Zero ULP uses the original numerical equality/NaN semantics, not universal bit
identity. All GPU output bit differences remain review items.

For each reduction, the comparator independently recomputes the original
Neumaier sum and magnitude from **that backend's observed FP32 terms**. It also
checks the recorded diagnostics against this recomputation. CUDA's reduction is
not judged against Metal's terms when their primitive math differs. The original
`32 * epsilon64 * sum_abs` reduction bound remains unchanged. Survival retains
its analytic `2e-14` checks and exact weight sums; empty outputs remain numerically
zero. References and analytic expectations occur only in Metal observations,
never in the CUDA inputs.

Both native modes and all exported Metal checks pass, alongside 18 report tests
and six mocked build-driver tests. The updated CUDA-domain status policy
reproduces both earlier Metal/input exports byte for byte. C++ input round trips
are exact, and observation files are rejected as CUDA inputs. These checks are
not NVIDIA execution evidence. The CUDA compiler/link and host isolation gates
passed in [run 42100004279](https://github.com/mmaaz-git/xgb-mps/actions/runs/42100004279);
actual NVIDIA runtime comparison remains pending.

```sh
bash scripts/test.sh --native metal_metric_api
MTL_DEBUG_LAYER=1 results/metal_metric_api_test \
  xgboost/src/common/metal_histogram.metal --report-prefix results/new-metric
python3 scripts/compare_metric_contracts.py validate-metal results/new-metric.metal.bin
```

Use fresh output paths and retain `.bin.json`, runtime logs and `build.json`.
The little-endian `XGBMET1\0` format requires all fixed identities, complete
arrays, backend/domain statuses, a completion marker and exact EOF. Published
XZ archives are lossless; raw-byte hashes are retained. The reader also accepts
raw and gzip files. Comparison exits 1 for malformed reports or failed checks,
and 2 for numerical/domain review, including exclusions when common-domain
results agree. Every differing output field is counted; JSON gives up to 32
examples per case/field while the binary reports retain every value.

## Root sums and stump weights

The `--stumps` mode retains the complete `metal_reduction.mm` corpus:
**405 cases / 11,964,810 gradient pairs**, with 1–65,537 rows, 1/3/7 targets,
contiguous/transposed/padded layouts, integers, subnormals, fractional values,
wide exponents, and zero/negative Hessians. All 31,910,480 backing input words,
including padding, are serialized exactly. The original generator, case order,
sum bounds, final-weight checks and unchanged-input checks remain intact.

CUDA invokes the actual `tree::cuda_impl::FitStump` on each original strided
view. The production API exposes only weights. A **separate source trace**
repeats its `GradientPairPrecise` transform, Thrust `reduce_by_key`, `GlobalSum`
and `CalcUnregularizedWeight` sequence to record sums and diagnostic weights.
These are not captures of the production call's private sums. Both executions
use CUDA; neither receives reference outputs. This suite runs on one worker,
so it does not establish distributed collective equivalence.

The comparator independently reconstructs the original Neumaier references
and sequential magnitudes from input terms, verifies the saved Metal references,
and checks each GPU's sums. Modes 0/1/4 retain exact numerical sum equality;
modes 2/3 retain `8e-14 * sum_abs`. Each diagnostic weight must equal the exact
FP32 bits of binary64 `-gradient / max(hessian, double(1e-6f))` using that
execution's own sums. For actual CUDA API weights, whose sums are private,
the comparator propagates those same original sum intervals through the weight
formula and FP32 rounding, with no extra tolerance. Exact modes retain bitwise
weight checks. All API/trace and GPU-to-GPU bit differences remain review items,
even when both observations satisfy their independent reference checks.

Normal and Metal-validation export runs pass. All **4,455 Metal output fields**
validate independently; nine synthetic integrity tests and six mocked build-driver
tests pass. The C++ reader reproduces the full input file byte for byte and
rejects observation files as input. CUDA compilation and host checks passed in
run 42100004279; NVIDIA execution is pending. See the
[run record](../validation/stump-contract-m3-run.json). This
extends P5-N07 coverage without completing another numerical contract.

```sh
python3 scripts/compare_stump_contracts.py unpack \
  validation/stump-contract-inputs-m3.bin.xz results/stump-inputs.bin
results/cuda-comparison-build/cmake/cuda_numerical_contracts --stumps \
  results/stump-inputs.bin results/stump-cuda.bin
python3 scripts/compare_stump_contracts.py compare \
  validation/stump-contract-metal-m3.bin.xz results/stump-cuda.bin \
  --output results/stump-comparison.json
```

To export a fresh Metal report:

```sh
bash scripts/test.sh --native metal_reduction
MTL_DEBUG_LAYER=1 results/metal_reduction_test \
  xgboost/src/common/metal_histogram.metal --report-prefix results/new-stump
python3 scripts/compare_stump_contracts.py validate-metal results/new-stump.metal.bin
```

Use new paths and preserve runtime metadata/logs plus `build.json`. The
little-endian `XGBSTP1\0` format requires every fixed identity, full backing
storage, backend-specific observations, `STPE` plus the count, and exact EOF.
Only Metal reports contain references. Raw/gzip/XZ inputs are accepted.
Comparison exits 1 for malformed evidence or reference failures, 2 for bit
differences requiring review, and 0 for observed identity in this corpus.
It counts all 5,940 cross-backend fields, including CUDA API weights, plus
1,485 API/trace weight comparisons; binary reports retain every value.

## Current CUDA-derived gradient sampling

The `--sampling` mode preserves the complete `metal_cuda_gradient_api.cc`
corpus: **143 low-level attempts, 54 lifecycle calls and two norm guards**.
The older 498-case `metal_gradient_sampling_api` tests a legacy CPU-derived
algorithm that production no longer uses; it is not this replay's CUDA target.
The [run record](../validation/sampling-contract-m3-run.json) contains exact
source, input, observation and log hashes. The original Metal checks pass in
ordinary and export modes. All **19,522,746 observation words** pass their
reference checks, and 19 synthetic report-integrity tests pass. CUDA compilation
and host checks passed in run42103889374; NVIDIA execution remains pending.

CUDA constructs the actual `GradientQuantiserGroup`, quantizes with
`CalcQuantizedGpairs`, and calls `cuda_impl::Sampler::Sample` and `ApplySampling`.
It records public `SamplingInfo` norms/thresholds, exact integer roots, converted
roots, floating replay and parent RNG state. Scalar roots use the scalar
updater's reduction/finalizer primitives; vector roots call its actual inline
`CalcRootSum`. This does not execute an entire tree updater or capture the
sampler's private CDF. The 18 lifecycle sequences keep each sampler and parent
engine alive across all three original calls.

Three details matter when interpreting these observations:

- Low-level Metal cases supply a row seed directly, while CUDA draws it from
  the parent MT19937 engine. The harness supplies a valid engine state whose
  first draw is that exact row seed. It does **not** call `seed(row_seed)` and
  assume the first draw matches. The host check covers 10,005 seeds; ordinary
  lifecycle cases retain the original `seed(701)` sequence and consumption.
- CUDA's floating replay API takes an owning matrix. Original padded replay
  views are gathered into that matrix on GPU, sampled by the actual API, then
  scattered back on GPU. All padding is checked unchanged. This adaptation
  does not claim a strided CUDA replay API exists.
- Seven cases exceed CUDA's int64 rescaling domain; a further positive-budget
  case has nonfinite norms before conversion. Metal's rejection is recorded;
  CUDA marks these eight cases **not executed**, without fabricated outputs.
  The nonfinite-norm case with zero budget follows CUDA's defined early-return
  path and remains among the **191 planned CUDA executions**.

Each backend is independently checked against the original source-oracle
references saved only in the Metal report. FP32 checks retain their original
exact-bit or both-NaN rule; root doubles retain numerical equality, and integer
and RNG words remain exact. Replay padding, unsampled identities and zero-budget
results also have checks derived from the inputs. These references and hashes
document the fixed native run; the file format alone cannot authenticate GPU
execution. Every GPU-to-GPU bit difference remains visible, including signed
zeros allowed by a numerical check. No tolerances were widened.

After compiling on a CUDA host, use fresh paths:

```sh
python3 scripts/compare_sampling_contracts.py unpack \
  validation/sampling-contract-inputs-m3.bin.xz results/sampling-inputs.bin
results/cuda-comparison-build/cmake/cuda_numerical_contracts --sampling \
  results/sampling-inputs.bin results/sampling-cuda.bin
python3 scripts/compare_sampling_contracts.py compare \
  validation/sampling-contract-metal-m3.bin.xz results/sampling-cuda.bin \
  --output results/sampling-comparison.json
```

Preserve the CUDA output's `.json` metadata, runtime log and compiler `build.json`.
The separate `--sampling-rng` command runs only the host seed-alignment check and
does not initialize a GPU. Compilation, linking and all 10,005 host seed checks
passed in [run 42103889374](https://github.com/mmaaz-git/xgb-mps/actions/runs/42103889374);
NVIDIA execution remains pending. To export fresh Metal observations:

```sh
bash scripts/test.sh --native metal_cuda_gradient_api
MTL_DEBUG_LAYER=1 results/metal_cuda_gradient_api_test \
  --report-prefix results/new-sampling
python3 scripts/compare_sampling_contracts.py validate-metal results/new-sampling.metal.bin
```

The `XGBSMP1\0` interchange retains every source/replay backing word, all fixed
case identities, backend-specific statuses and observations, a completion marker
and exact EOF. Input-only files contain no expected outputs. Comparison exits 1
for malformed records or reference failures and 2 for required review, including
the eight exclusions even when all common-domain outputs are identical.

## Integer leaf accumulation and weights

The `--integer-leaves` mode retains all of `metal_integer_leaf_api.cc`: 242
ordinary strided/broadcast cases, eight cancellation cases, one original-scale
curvature case and five conversion-range cases. The last six are checked by
the original test but absent from its printed 250-case counter. There are
**256 records**, with three undefined CUDA conversions explicitly excluded.
The [record](../validation/integer-leaf-contract-m3-run.json) links the inputs,
Metal outputs and exact provenance. Both native modes, 15 synthetic report
tests and the C++ input round trip pass. CUDA compilation/linking passed in
[run 42108648407](https://github.com/mmaaz-git/xgb-mps/actions/runs/42108648407);
NVIDIA execution remains pending.

CUDA builds the actual original `GradientQuantiserGroup`, calls `LeafGradSum`
for every page into one persistent integer matrix, and obtains all five weight
settings through `LeafWeight`. Reversed leaf segments retain their original
slot mapping; output reordering only maps API rows back to canonical slots.
Bounds and output guards remain present. Curvature/range cases mutate the
original allocation after creating the quantizer; ordinary cases overwrite
the original source while retaining its quantizer and separate sampled inputs.
The harness requires the Thrust 3 accumulation branch: the older branch explicitly
overwrites pages and cannot represent this full paged corpus.

Metal's `LeafSums` API exposes converted binary64 sums and weights, not its
private integer buffer. The comparison therefore keeps three distinct things:

- **20,466 converted sum fields** and **19,430 weight fields**, including guards,
  are actual observations on both backends. Every intermediate page is retained.
- CUDA's 20,466 integer words are actual `LeafGradSum` buffer observations.
- Metal's saved integer words are independent CPU-oracle references. They are
  never counted as Metal GPU observations.

The comparator reconstructs original scales from row-ordered positive/negative
sums, converts sampled terms with the positive-Hessian floor, accumulates exact
integer page sums, then derives converted sums and weights directly from input
bits. Saved references and both backends are checked independently. The
original numerical-double equality and exact weight bits remain unchanged;
GPU-to-GPU signed-zero differences still require review. Three out-of-range or
nonfinite conversion guards remain Metal rejections/CUDA exclusions, never
fabricated successful CUDA tests. This is a single-worker leaf API replay, not
a full updater or histogram-cache comparison.

After building on a CUDA host, use fresh paths:

```sh
python3 scripts/compare_integer_leaf_contracts.py unpack \
  validation/integer-leaf-contract-inputs-m3.bin.xz results/integer-leaf-inputs.bin
results/cuda-comparison-build/cmake/cuda_numerical_contracts --integer-leaves \
  results/integer-leaf-inputs.bin results/integer-leaf-cuda.bin
python3 scripts/compare_integer_leaf_contracts.py compare \
  validation/integer-leaf-contract-metal-m3.bin.xz results/integer-leaf-cuda.bin \
  --output results/integer-leaf-comparison.json
```

To export fresh Metal observations:

```sh
bash scripts/test.sh --native metal_integer_leaf_api
MTL_DEBUG_LAYER=1 results/metal_integer_leaf_api_test --report-prefix results/new-integer-leaf
python3 scripts/compare_integer_leaf_contracts.py validate-metal results/new-integer-leaf.metal.bin
```

Preserve runtime `.json` metadata, logs and compiler `build.json`. The `XGBILF1\0`
format checks fixed case identities, full source/sample backing storage,
backend-specific statuses, observations/references, `ILFE` plus the count and
exact EOF. CUDA accepts input-only records. Comparison exits 1 for malformed
evidence or reference failures and 2 for required review, including the three
exclusions even when all common-domain outputs agree.

## Integer histogram-cache lifecycle

The `--integer-cache` mode preserves all 48 cases in
`metal_integer_cache_api.cc`: both gradient-index layouts, all three index widths,
four gradient exponents and one/two bins. It retains all original dispatch
assertions and checks growth, integer subtraction, cancellation, empty subsequent
pages, an empty first page when reusing a slot, untouched guards/source gradients,
and a new tree with a new quantizer. The
[record](../validation/integer-cache-contract-m3-run.json) contains source hashes,
Metal evidence and full portable inputs/observations.

CUDA constructs the actual ELLPACK page from the original gradient index, then
calls `DeviceHistogramBuilder`, its cache allocator and `SubtractionTrick`.
The caller clears a slot for its first page because the CUDA builder adds to
caller-owned storage; Metal's wrapper performs that reset internally. CUDA's
root diagnostic applies its production integer reduction and quantizer to the
actual histogram bins. It does not expose the updater's private root call,
which normally reduces gradient rows.

Across backends, compare 1,584 converted binary64 values and 192 decoded bin
IDs. CUDA also records 1,584 actual integer diagnostics. Metal does not expose
its private integer storage. Growth prefixes and guards are checked independently:
Metal uses materialized double pairs, while CUDA uses integer node histograms.
The original Metal 16-to-257-pair growth maps to 16-to-257 CUDA node slots; this
does not claim physical CUDA reallocation, since its cache reserves capacity.
There are 4,032 Metal and 6,048 CUDA storage fields to check.

The comparator reconstructs every original input bit and quantizer, accumulates
integer bins before converting, and independently validates all snapshots, roots,
prefixes and guards. Identical wrong outputs cannot pass. Original numerical
double equality remains; cross-backend bit differences still require review.
Exit codes are 0 for identical outputs with all reference checks passing, 2 for
differences requiring review, and 1 for malformed inputs or failed checks.

```sh
python3 scripts/compare_integer_cache_contracts.py unpack \
  validation/integer-cache-contract-inputs-m3.bin.xz results/integer-cache-inputs.bin
results/cuda-comparison-build/cmake/cuda_numerical_contracts --integer-cache \
  results/integer-cache-inputs.bin results/integer-cache-cuda.bin
python3 scripts/compare_integer_cache_contracts.py compare \
  validation/integer-cache-contract-metal-m3.bin.xz results/integer-cache-cuda.bin \
  --output results/integer-cache-comparison.json
```

To produce new Metal observations, use a new output prefix:

```sh
bash scripts/test.sh --native metal_integer_cache_api
MTL_DEBUG_LAYER=1 results/metal_integer_cache_api_test --report-prefix results/new-integer-cache
python3 scripts/compare_integer_cache_contracts.py validate-metal results/new-integer-cache.metal.bin
```

Both Metal modes, the independent checker, 16 synthetic report tests and the C++
input round trip pass. CUDA compilation/linking passed in run42108648407;
NVIDIA execution remains pending. This is a
single-worker API replay; it does not validate the full external-memory scheduler
or establish complete training-reduction parity.

## Linear coordinate reductions and residuals

The `cuda_linear_contracts` executable retains the complete existing
`metal_linear_api.cc` corpus: 49 main cases, four exceptional cases and three
rejection guards, for **56 records / 893 actions**. This includes the million-row
case, zero rows, multiple groups, negative Hessians, duplicate CSC row IDs,
subnormals, large and nonfinite values, and the original source/entry/copy guards.
The [record](../validation/linear-contract-m3-run.json) links the full artifacts
and their source hashes.

The normal and export Metal runs retain the original counts: 4,118,809 source
pairs, 3,564 checked sum fields (2,991 exact/nonfinite) and 148,277,125 residual
checks. The reports additionally serialize the previously uncounted edge cases,
copy guards, the order fixture's Hessian and the overflow fixture's Hessian:
3,570 sum fields and 148,280,662 residual/copy fields. The independent checker
reconstructs every original input, including the seeded MT19937 sequence and
duplicate CSC entries. It validates each supplied action state, source terms,
FP32 reduction bound and residual expression. Original finite residual bits and
NaN classification remain unchanged; no tolerance is widened.

CUDA invokes `GPUCoordinateUpdater::GetBiasGradient`, `GetGradient`,
`UpdateBiasResidual` and `UpdateResidual` from the actual vendored source.
Its class and method bodies are unchanged. A test-only member-pointer adapter
initializes its private CSC/gradient storage and calls the original
`UpdateGpair` upload. Registry expressions populate unregistered test entries;
hidden visibility prevents production symbol replacement. Compiler CI checks
the actual registered names/factory types against a baseline executable and
rejects exported production linear definitions. These are host checks, not GPU
execution evidence.

This is an **operation replay**, not an execution of `LazyInitDevice`, the full
updater/selector loop or a continuous CUDA update sequence. Each action loads
its supplied input state. Those states preserve the original test's CPU
reference sequence and are independently reconstructed by the comparator.
This adaptation is necessary to compare defined operations following the
original corpus's duplicate-column updates: CUDA's concurrent duplicate writes
race, while Metal supports serial duplicate runs. The 123 affected residual
updates are explicit CUDA exclusions; their preceding gradient reductions still
execute. The three unsafe row/feature guards are Metal rejection observations
and CUDA exclusions, never fabricated successful CUDA rejections. Empty states
exercise well-formed zero-length primitive calls, not the updater constructor's
empty-input early-return path.

There are 123,570,886 planned common actual fields: 3,570 sums and 123,567,316
residual/copy fields. Comparison returns 1 for malformed data or failed checks
and 2 for required domain/difference review, even if every common output matches.
NaN payload and zero-sign bit differences remain visible despite the native
classification/numeric rules.

```sh
python3 scripts/compare_linear_contracts.py unpack \
  validation/linear-contract-inputs-m3.bin.xz results/linear-inputs.bin
results/cuda-comparison-build/cmake/cuda_linear_contracts \
  results/linear-inputs.bin results/linear-cuda.bin
python3 scripts/compare_linear_contracts.py compare \
  validation/linear-contract-metal-m3.bin.xz results/linear-cuda.bin \
  --output results/linear-comparison.json
```

To export again, use new paths. The full input and Metal files require about
1.9 GB before compression; the checker memory-maps uncompressed files.

```sh
bash scripts/test.sh --native metal_linear_api
MTL_DEBUG_LAYER=1 results/metal_linear_api_test --report-prefix results/new-linear
python3 scripts/compare_linear_contracts.py validate-metal results/new-linear.metal.bin
```

Metal execution, the independent checker, 20 report-integrity tests and the full
C++ input round trip pass. CUDA compilation/linking and host registry/symbol
isolation passed in [run 42113854537](https://github.com/mmaaz-git/xgb-mps/actions/runs/42113854537).
NVIDIA execution and full updater/selector comparisons remain pending.
Primitive association bounds are not universal CUB
architecture guarantees; preserve concrete failures for review alongside the
separate compiler/contraction contract.

## Grouped precision–recall AUC

`cuda_auc_contracts --grouped-pr` retains the entire existing
`metal_ranking_pr_api.cc` corpus: **319 records**, comprising 297 main cases,
sixteen edge cases and six invalid-label rejections. The main cases cover
14,198,354 rows and 4,660 original query outputs. Zero/singleton queries, three
storage strides, tied signed zeros, subnormal scores/weights, soft and NaN labels,
negative/nonfinite weights, overflowing products, and million-row queries remain.
Original dispatch, input/guard, CPU/source/integral and tiny-value relative checks
are unchanged. The [record](../validation/grouped-pr-contract-m3-run.json) contains
source hashes and local evidence.

The original Metal calls are saved exactly. A separate common replay packs the
same logical values and removes empty queries, then observes the combined result
and individual one-query calls. CUDA invokes the **linked production
`metric::cuda_impl::RankingPRAUC`**, without copying its formulas, including its
source into a test translation unit, or replacing registrations. Its six rejection
records require the actual invalid-label error, not an arbitrary exception.

Compaction is an explicit domain adaptation: in `auc.cu`, grouped PR can read
`fptp[LastOf(empty_group)]`, and `reduce_by_key` stores totals without slots for
absent groups. A leading empty query can underflow the index; internal empty
queries can misalign the total assigned to later queries. This is source analysis,
not an observed NVIDIA result. There are 301 adapted records, including six
rejection cases. CUDA does not execute those original empty-query layouts, so
their source behavior is not established by this replay. Zero-row inputs reach
the source's early return. Negative/nonfinite weights remain direct implementation
tests; normal public metadata validation rejects them.

Isolated-query results are actual separate calls on each backend, **not captured
per-query internals of the combined CUDA call**. Metal additionally retains all
original per-query outputs and all four padded output guards. The reports expose
20,336 Metal fields; 8,014 planned common fields come from 313 combined calls and
3,694 isolated-query calls. The independent checker reconstructs all 153,869,127
original FP32 input words, including every strided gap, using the original seeded
MT19937 algorithm. It checks FP32 source term casts, binary64 serial reference
integration, exact counts/rejections and the original `2e-12 + 2e-12*abs(expected)`
tolerance. The tied cast fixtures retain exact comparisons; tiny tied curves
retain the separate relative check. This reference is not a CUB association model
or a universal NVCC/libdevice error bound.

One complete compressed report is stored, avoiding a duplicate copy of its large
input payload. `unpack` extracts input-only records with a separately recorded
hash; the CUDA reader rejects observation files. Use new output paths:

```sh
python3 scripts/compare_grouped_pr_contracts.py unpack \
  validation/grouped-pr-contract-metal-m3.bin.xz results/grouped-pr-inputs.bin
results/cuda-comparison-build/cmake/cuda_auc_contracts --grouped-pr \
  results/grouped-pr-inputs.bin results/grouped-pr-cuda.bin
python3 scripts/compare_grouped_pr_contracts.py compare \
  validation/grouped-pr-contract-metal-m3.bin.xz results/grouped-pr-cuda.bin \
  --output results/grouped-pr-comparison.json
```

Original Metal execution and the full export pass. Independent references,
23 report-integrity tests, eleven mocked build-driver tests and the complete C++
input round trip pass. CUDA compilation and actual NVIDIA execution remain
pending. Comparison exits 1 for malformed input or reference failure and 2 for
review, even if all common outputs match, because the domain adaptation remains.
All output bit differences, including zero signs and NaN payloads, are reported.

To export again, allow approximately 1.2 GB for the two raw files:

```sh
bash scripts/test.sh --native metal_ranking_pr_api
MTL_DEBUG_LAYER=1 results/metal_ranking_pr_api_test --report-prefix results/new-grouped-pr
python3 scripts/compare_grouped_pr_contracts.py validate-metal results/new-grouped-pr.metal.bin
```

This covers grouped PR only. Binary/multi-output AUC, their pathological prefix
diagnostics, grouped ROC, public metric caller combinations and the remaining
ranking metrics are separate work; P5-N09/10/11 are not closed by this batch.

## Binary and multi-output AUC observation preparation

The CUDA comparison build now prepares a separate `auc_probe` namespace copy of
the pinned `src/metric/auc.cu`. `scripts/prepare_auc_probe.py` preserves the source
arithmetic and inserts eight readback hooks for terms, prefixes, original PR
totals, curve results and normalization inputs. It checks the original source
hash and reconstructs its exact bytes from the edit record. Production source
and the grouped-PR library calls are unchanged.

`tests/auc_capture.cuh` records actual producer buffers, with explicit presence
flags for stages the source skips. Enabled capture adds device synchronization
and host copies; it is a test instrument, not a performance measurement. CI
retains the generated source and edit record, checks metric registry identity,
and rejects exported metric definitions from the copy.

Eight source/capture-helper checks and fourteen mocked build-driver checks pass
locally. CUDA compilation of this preparation is pending; these host checks do
not prove CUDA execution or numerical agreement. The full 528-record Metal AUC
corpus still needs its exporter, CUDA replay and comparison checks. Replay must
also compare the instrumented results with calls to the original linked library.
This preparation does not increase the count of completed comparison contracts.
