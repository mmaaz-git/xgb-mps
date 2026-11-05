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
| Tree training | Hist/approx histograms, quantization, sampling, scalar/vector split arithmetic, partitioning, constraints and leaf fitting have Metal paths | Vector payload ownership is incomplete |
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
| PORT-1 | Vector weight producers, root learning-rate multiplication, model growth and alias snapshots still use host views/copies in baseline main. Quarantined draft is unvalidated and contains an unsafe non-MPS branch. |
| PORT-2 | Selected vector categorical bitsets return to host candidates before model copying; CUDA retains device category storage. |
| PORT-3 | Exact vector node/selected-child sums move through host containers; CUDA retains device node/split sums. |
| NUM-1 | Weighted-quantile threshold narrows CUDA's binary64 comparison to FP32; a two-row production diagnostic selects the wrong value. |
| PORT-9 | Vector leaf validity incorrectly uses zero Hessians instead of the actual sampler mask. |
| PORT-6 | Sampled/allowed feature-ID payloads and consumers still use CPU arrays. |
| PORT-7 | Native quantile row/group weights pass through host expansion/readback. |
| PORT-8 | Active scalar/vector split evaluators gather GPU monotonic bounds on CPU. |
| PORT-10 | Selected numeric cut/−∞ threshold materialization runs on CPU; categorical bit construction already runs on GPU. |
| PORT-11 | Sketch categorical detection and pointer initialization still perform CUDA GPU work on CPU. |

PORT-4/5 classification is complete for the frozen [audit](CUDA_AUDIT.md):
174 logical operations, 1,078 assigned lexical sites, 17 numerical contracts.
Sixteen missing/defective rows map to the ten tasks above. This is a finite
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

Last full regression before review: implementation `dd44829`, recorded at
`2eaad4d`; **100 native groups, 3,554 Python passes, one expected skip**, CPU-only
build/API checks and optional federated learners.
[Configuration and results](../validation/cuda-metal-native-csr-storage-m3.txt).
Cleanup has separate [validation evidence](../validation/repo-review-2026-11-05.txt).

CUDA compile CI also passed at audit baseline `382f78c`
([run 41987882440](https://github.com/mmaaz-git/xgb-mps/actions/runs/41987882440)). It compiles/links on
a CPU runner. **No NVIDIA runtime comparison has been performed.** Other Apple
hardware is also untested. [Historical benchmarks](PUBLISHED_BENCHMARKS.md)
predate subsequent correctness changes and are not current-HEAD measurements.
