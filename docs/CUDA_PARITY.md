# CUDA / Metal status

This is the implementation and support summary. Current tested revisions,
regression results and CI evidence are maintained in [VALIDATION.md](VALIDATION.md).
Detailed implementation history is preserved in the
[archived ledger](archive/CUDA_PARITY-progress-2026-11-04.md). Historical sections
describe different revisions and must not be read as current status.

The target is CUDA-accelerated behavior in vendored XGBoost 3.4.2. `device="mps"`
selects custom **Metal compute kernels**; Apple's MPS library does not replace
XGBoost's CUDA kernels. **Full coverage remains unproven; verification work
remains after the audit's ten implementation fixes.** See the [ordered plan](PLAN.md).

## What exists

| Area | Current implementation | Boundary |
| --- | --- | --- |
| Tree training | Hist/approx histograms, quantization, sampling, scalar/vector split arithmetic, partitioning, constraints and leaf fitting have Metal paths | Source-defined host orchestration remains; NVIDIA runtime comparisons are pending |
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
| VERIFY-2 | Shared fixtures executed on actual NVIDIA hardware; local source oracles are insufficient. |
| PERF-1 | Current-revision large-workload timings and optimization after local correctness acceptance. |

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
([evidence](../validation/cuda-port6-m3.txt)). PORT-7 is locally fixed at `796e9fa`:
native quantile row/group weights and expansion stay on Metal through quantile
and host/disk external-memory construction; constructor authority/copy checks,
independent cut oracles and Torch/MLX cases pass
([evidence](../validation/cuda-port7-m3.txt)). PORT-8 is locally fixed at `ed5afcf`:
GPU bound updates feed resident snapshots, scalar/vector feature evaluation and
root finalization; growth, aliases, lifetimes and complete-tree checks pass
([evidence](../validation/cuda-port8-m3.txt)). PORT-10 is locally fixed at `30639c6`:
selected thresholds resolve on GPU, with resident cuts through winner/category
consumers; exact metadata and complete-tree checks pass
([evidence](../validation/cuda-port10-m3.txt)). PORT-11 is locally fixed at `68c800e`:
GPU pointer initialization, categorical reduction and owned type snapshots pass
constructor/lifetime, subsequent-batch and federated checks
([evidence](../validation/cuda-port11-m3.txt)). All ten audited implementation tasks
and VERIFY-1 local acceptance are complete; NVIDIA execution and performance
remain open.
The frozen audit retains its original findings. This is a finite
implementation queue, not proof that no unlisted defects remain. Numerical
acceptance is recorded in the [local acceptance report](LOCAL_ACCEPTANCE.md); NVIDIA comparisons remain open.

Subsequent AUC comparison preparation found a capacity gap outside those ten
tasks. PORT-12 (`4a05813`) removes the artificial 65,535-output AUC limit and makes
empty normalization return NaN, as CUDA's reduction does. Native and Python
checks exercise 65,536 outputs through the public metric/learner paths. The
[focused record](../validation/auc-capacity-m3-run.json) preserves the failing
baseline and exact tested source. PORT-13 (`41d4dc2`) corrects the five remaining
artificial limits in gradient columns, leaf sums, gradient sums, means and
quantiles. Its [boundary and caller tests](../validation/output-capacity-m3-run.json)
pass at 65,535/65,536/65,537 outputs, including a complete 65,536-target vector tree
with exact CPU/analytic predictions. Kernel index and device buffer limits remain;
extreme-width per-column quantization is memory-intensive. The
[full regression record](../validation/output-capacity-full-m3-run.json) records
validation at `41d4dc2`; see [VALIDATION.md](VALIDATION.md) for the evidence summary.
No NVIDIA execution or arbitrary-width training claim is implied.

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

All 17 contracts now have named local fixtures and external requirements in the
[acceptance report](LOCAL_ACCEPTANCE.md). New CDF/scan/repair witnesses explicitly
show arithmetic/read-order differences without claiming measured NVIDIA output.

These are explicit limits or tracked work. The [review](REVIEW.md) separates
reproduced defects, confirmed gaps and architecture risks.

## Evidence boundary

Use [VALIDATION.md](VALIDATION.md) for the current evidence summary, including
which revisions received full regression, focused checks, compiler CI and timing.
Every record applies to its recorded source and configuration.

Historical acceptance evidence remains available: the
[`68c800e` record](../validation/cuda-port11-m3.txt) includes CPU build/API/schema,
updater, PyTorch/MLX interop and optional federated checks. Federated transport,
sketches and learners were tested with all workers on the same M3, including TLS.
The [VERIFY-1 record](../validation/cuda-verify1-m3-run.json) contains the subsequent
local numerical witnesses. The historical CUDA compile milestone at `bf1ebea`
([run 42027936238](https://github.com/mmaaz-git/xgb-mps/actions/runs/42027936238))
is retained as evidence for that revision, not as the latest CI result.

Compiler CI runs on a CPU runner. **No NVIDIA runtime comparison has been
performed.** Other Apple hardware is also untested.
[Published benchmarks](PUBLISHED_BENCHMARKS.md) retain their measured revisions
and are not evidence for an unmeasured checkout.
