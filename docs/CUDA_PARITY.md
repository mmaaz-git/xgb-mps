# CUDA / Metal status

This is the current summary. Detailed implementation history is preserved in the
[archived ledger](archive/CUDA_PARITY-progress-2026-11-04.md). Historical sections
describe different revisions and must not be read as current status.

The target is CUDA-accelerated behavior in vendored XGBoost 3.4.2. `device="mps"`
selects custom **Metal compute kernels**; Apple's MPS library does not replace
XGBoost's CUDA kernels. **Full coverage remains unproven and known GPU-state gaps
remain.** See the [ordered plan](PLAN.md).

## What exists

| Area | Current implementation | Boundary |
| --- | --- | --- |
| Tree training | Hist/approx histograms, quantization, sampling, scalar/vector split arithmetic, partitioning, constraints and leaf fitting have Metal paths | Bound gathers and numeric thresholds still use host payloads |
| Objectives/metrics | Regression, multiclass, ranking, AFT, adaptive objectives and metrics have Metal paths | Numerical comparisons have operation-specific limits |
| Prediction | Tree/leaf prediction, native outputs, contributions/interactions have Metal paths | Source-defined host model setup/serialization remains |
| Data | Native tensors/columns, CSR packing, sketches, quantile matrices, dictionaries, metadata, external caches and optional Snappy | Native ordinary entries stay resident through sketch/compression; producer readiness is synchronous |
| Other consumers | GPU coordinate descent, collectives, distributed tree learners, optional federated transport | Host linear selectors/model updates follow CUDA; distributed evidence uses workers on one M3 |

These are implemented paths, not blanket completeness claims. The
[operation map](CUDA_OPERATION_MAP.md) links 42 families / 109 source roots to
implementation/test entry points. Its checker verifies inventory/references,
not every branch's behavior.

## Confirmed open work

| Plan ID | Remaining CUDA behavior or evidence |
| --- | --- |
| PORT-7 | Native quantile row/group weights pass through host expansion/readback. |
| PORT-8 | Active scalar/vector split evaluators gather GPU monotonic bounds on CPU. |
| PORT-10 | Selected numeric cut/−∞ threshold materialization runs on CPU; categorical bit construction already runs on GPU. |
| PORT-11 | Sketch categorical detection and pointer initialization still perform CUDA GPU work on CPU. |

PORT-4/5 classification is complete for the frozen [audit](CUDA_AUDIT.md):
174 logical operations, 1,078 assigned lexical sites, 17 numerical contracts.
Sixteen missing/defective rows mapped to ten tasks at the audit revision. NUM-1
is now locally fixed at `2404827`. Binary64
threshold comparisons and independent boundary/alias regressions pass in three
native groups and 34 Python tests; [evidence](../validation/cuda-num1-m3.txt).
PORT-9 is locally fixed at `1034657`: vector leaf finalization replays retained
sampling state; four native groups, 52 Python tests and CPU build/API/updater
checks pass ([evidence](../validation/cuda-port9-m3.txt)). PORT-1 is locally fixed
at `c78cb43`: persistent GPU weights, root/model scaling, alias snapshots, bound
inputs and compact prediction-cache gathers are validated
([evidence](../validation/cuda-port1-m3.txt)). PORT-2 is locally fixed at `6a606e1`:
selected vector categories persist through queues, model growth and partitioning
([evidence](../validation/cuda-port2-m3.txt)). PORT-3 is locally fixed at `ab87872`:
exact vector roots, queued sums, parent gathers and child propagation stay on Metal
([evidence](../validation/cuda-port3-m3.txt)). PORT-6 is locally fixed at `2bfb914`:
feature sampling, interaction filtering and feature-indexed split construction
consume resident IDs through actual scalar/vector evaluators
([evidence](../validation/cuda-port6-m3.txt)). Four implementation tasks remain.
The frozen audit retains its original findings. This is a finite
implementation queue, not proof that no unlisted defects remain. Numerical
acceptance fixtures and NVIDIA comparisons remain separate verification work.

Scalar control fields, topology, serialization and finalizers that CUDA itself
runs on the host are not missing kernels merely because they access host memory.
MPS execution currently shares CPU storage defaults, making caller-level review
necessary. Blanket replacement of host views is unsafe.

## Numerical and platform contracts

The vendored CUDA implementation defines formulas, precision, RNG and update
order. CPU is an oracle where those agree. Changing CUDA formulas solely to
match CPU is not an acceptable hidden fix. Local source-derived oracles also do
not establish NVIDIA runtime equivalence.

- **AFT:** sensitive CDF cancellation amplifies small transcendental differences.
  Well-conditioned CPU checks remain; sensitive rows have source-expression and
  intermediate checks. No CPU recomputation or widened tolerance erases the
  documented differences. [Details](archive/CUDA_PARITY-progress-2026-11-04.md#aft-device-execution-and-numerical-behavior).
- **Ranking:** CUDA truncation, RNG and pair order differ from CPU. Degenerate/racy
  source cases have explicit deterministic policies that must stay visible.
  [Details](archive/CUDA_PARITY-progress-2026-11-04.md#ranking-gradient-numerical-behavior).
- **AUC/scans/reductions:** association, ties, compiler contraction and NVIDIA math
  libraries can differ. Exact local fixtures and sampled tolerances do not imply
  universal bitwise equality.
- **Interchange/scheduling:** Metal DLPack/native adapters replace CUDA pointers.
  Completion is synchronous; a portable CUDA-style stream handshake is absent.
  Optimizations must preserve ownership/readiness.

These are explicit limits or tracked work. The [review](REVIEW.md) separates
reproduced defects, confirmed gaps and architecture risks.

## Evidence boundary

Latest full local regression covers `2bfb914`: **102 native groups and 3,603
Python passes, one expected skip**, plus CPU build/API/schema and nine updater
tests. Both PyTorch and MLX GPU interop tests ran.
[Configuration, source hashes and results](../validation/cuda-port6-m3.txt).
Optional federated learners were last tested at `dd44829`
([record](../validation/cuda-metal-native-csr-storage-m3.txt)). Earlier validation
records retain their original scope and failures.

CUDA compile CI passed at `a531694`, including PORT-3
([run 42010137944](https://github.com/mmaaz-git/xgb-mps/actions/runs/42010137944)).
The PORT-6 source needs its own CI result. CI compiles/links on
a CPU runner. **No NVIDIA runtime comparison has been performed.** Other Apple
hardware is also untested. [Historical benchmarks](PUBLISHED_BENCHMARKS.md)
predate subsequent correctness changes and are not current-HEAD measurements.
