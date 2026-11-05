# Completion plan

Updated 2026-11-05 after the [caller/numerical audit](CUDA_AUDIT.md).
The bounded audit is complete; the user resumed feature development. This is the work queue;
chronological progress logs are not a second plan.

## Target and completion claims

Port CUDA-accelerated operations in the **vendored XGBoost 3.4.2 snapshot**
(`22a84e0`) to Apple GPU execution. Preserve CUDA algorithms and payload placement
where applicable. Host orchestration, serialization and other work CUDA itself
performs on CPU remain host work. Metal replaces NVIDIA-specific memory/runtime
mechanisms; it cannot accept CUDA device pointers.

Track three separate claims:

1. **Coverage:** every required CUDA operation/caller has a named Metal path and
   integration evidence, or an explicit platform adaptation. No unexplained gaps.
2. **Correctness:** source-derived checks, relevant CPU comparisons, storage and
   lifetime checks, CPU builds and CUDA compilation pass. Actual NVIDIA runtime
   comparisons are a separate requirement and have not been performed.
3. **Performance:** repeated end-to-end measurements on fixed workloads. No
   promised speedup ratio; slower-than-CPU results remain visible.

The audit classified 174 logical operations and all 1,078 listed lexical sites
across 42 families / 109 CUDA roots at `382f78c`. Sixteen missing/defective rows
map to **10 implementation tasks** below; NUM-1 is now fixed and nine remain.
These unequal-sized rows are not a
completion percentage, and this bounded audit cannot prove the absence of
unlisted defects. **Current: PORT-9, then vector state and the remaining
GPU-placement tasks.** No new kernels were added during
classification.

## Ordered work queue

| ID | Deliverable | Status / dependency | Completion criterion |
| --- | --- | --- | --- |
| CLEAN-1 | Review, concise docs, selected-test runner, reproduced API fixes | Complete | Findings published; history archived; all 100 native groups preserved; current library built first; diagnostic schemas agree; metadata alias regression passes |
| PORT-4 | Classify frozen CUDA caller inventory | Classification complete | All 1,078 listed sites assigned; 174 logical rows have source/Metal/test references and dispositions; findings are separate tasks, not a parity certificate |
| PORT-5 | Classify numerical-contract ledger | Classification complete; acceptance work remains | Seventeen contracts record authority, comparisons, limits and actions; NUM-1 is locally fixed; required divergence fixtures and external comparisons remain VERIFY-1/2 work |
| NUM-1 | Correct weighted-quantile threshold precision | Locally complete at `2404827` | Binary64 comparison and independent boundary/alias regressions pass; three native groups and 34 Python tests; [evidence](../validation/cuda-num1-m3.txt). NVIDIA comparison remains VERIFY-2 |
| PORT-9 | Correct vector row validity after sampling | Current | Preserve actual sampler mask for selected zero-Hessian rows; cover no/uniform/MVS sampling, reduced gradients and empty pages; preserve separate scalar semantics |
| PORT-1 | GPU vector root/child/model weights | Open; after correctness fixes | Finalize → persistent weights → root scale / model expansion / constraints / prediction-cache leaf gathers stays on Metal; growth preserves queued candidates; non-MPS CPU updater still uses host views |
| PORT-2 | GPU selected categorical bitsets | Open; after PORT-1 | Vector category payload stays resident through workspace reuse, candidate queuing and model growth; numeric/one-hot/partition and alias cases pass |
| PORT-3 | GPU exact vector node/child sums | Open; after PORT-1 | Integer selected-child and node sums stay resident through evaluation/propagation; no floating reconstruction; scalar source-defined publication unchanged |
| PORT-6 | Resident sampled/allowed feature IDs | Open; after vector state | GPU weighted gather/shuffle/sort, constraint filtering and split-consumer feature IDs; ordinary host launch descriptors remain permitted |
| PORT-7 | Resident native quantile weights | Open; after PORT-6 | Keep row/group weights and GPU expansion resident through native adapter sketch construction; weighted/grouped/repeated-batch fixtures |
| PORT-8 | GPU split-bound gathers | Open; after PORT-7 | Scalar/vector active evaluators consume GPU monotonic bounds; exclude unsupported reduced-gradient/monotone refit combination |
| PORT-10 | GPU numeric split thresholds | Open; after PORT-8 | Selected cut and −∞ materialization executes on GPU; category-bit construction already exists, with categorical cases retained as regression guards |
| PORT-11 | GPU sketch initialization | Open; after PORT-10 | GPU categorical-type reduction and pointer-payload clear; numeric/categorical/mixed/empty and subsequent-batch fixtures |
| VERIFY-1 | Integration at a fixed revision | After required implementation tasks | Full native/Python run, CPU build/API + updater checks, optional federated checks, CUDA compile; complete required numerical divergence fixtures; record dependencies/skips and no unexplained required-operation gaps |
| VERIFY-2 | NVIDIA execution comparison | External hardware required | Shared fixtures run against vendored CUDA on NVIDIA; publish numerical contracts and failures, not just compiler success |
| PERF-1 | Profile/optimize a frozen benchmark matrix | After local coverage/correctness milestone | Repeated CPU/MPS timings and correctness for large HIGGS, wide synthetic, sparse and vector workloads; improvements and regressions published |

VERIFY-2 is an evidence gap, not a reason to keep editing unrelated Metal code
while hardware is unavailable. Cross-Apple-hardware claims also require actual runs.

## PORT-1 is a bounded change

The interrupted implementation is preserved **locally**, unmerged, on
`wip/vector-weight-storage-review` at `ca00777`. Review it as a draft; do not
cherry-pick it wholesale.

- Fix the reviewed unsafe CUDA-context branch first: only MPS requests a Metal
  weight view; the common CPU updater's other branch uses a host view. CUDA
  compilation cannot detect a CPU dereference of device memory.
- Cover root-only trees, depthwise/lossguide growth, monotonic constraints,
  categorical inputs, forest/reduced-target callers, saved-model continuation,
  workspace reuse, aliases and delayed candidates after growth.
- Run GPU consumers **before** CPU comparisons. Assert storage authority and
  payload-copy/readback counters; dispatch counts alone are insufficient.
- Compare fixed CUDA-source weight fixtures and CPU cases where algorithms agree.
  Do not widen tolerances without changing and documenting the numerical contract.
- Finish with focused native/Python checks and CPU-build evidence, then stop.
  PORT-2 and PORT-3 remain open after this task lands.

## PORT-4 audit boundary

The [published audit](CUDA_AUDIT.md) freezes the source/caller table at `382f78c`.
The [ledger](../validation/cuda-caller-audit.json) and reproducible lexical
inventory retain source locations, actual MPS execution/storage paths and one
disposition per operation:

- implemented, with a specific test;
- host work also performed by CUDA, with source evidence;
- Metal platform adaptation with an explicit contract;
- missing operation, with a new task ID and acceptance case.

Classification is complete for this frozen ledger; stop the broad audit and
implement ticket by ticket. New findings update this queue
explicitly; they do not silently expand PORT-1 or reset completed work.
“Mapped” is not a closure status.
GPU-required work left on CPU cannot be closed as a platform adaptation, and
documenting a numerical defect does not fix or close it.

## Testing and work discipline

```sh
# Inspect without starting builds/tests.
bash scripts/test.sh --list
bash scripts/test.sh --native metal_model_state_api --dry-run

# Example focused integration; the current library is rebuilt first.
bash scripts/test.sh --native metal_model_state_api \
  --python tests/test_metal_model_state.py

# Integration milestone, after relevant focused checks pass.
bash scripts/test.sh --full
```

CPU-only and federated configurations are separate gates; see
[validation instructions](VALIDATION.md). Interop acceptance requires PyTorch
**and** MLX: `--require-metal` alone does not prevent optional-dependency skips.

Before a task, state its concrete gap, touched subsystem and exit condition;
inspect existing outputs. Keep one implementation task active. Long work runs
in a named `tmux` session with logs; inspect that job instead of restarting it.
After focused checks pass, broaden/repeat testing only for a new failure or risk.
Run full regression at integration milestones and shared-runtime changes, not
after every documentation edit. Report terminal failures accurately.

## Explicit follow-ups, not an immediate rewrite

- **MAINT-1:** extract runtime/memory plumbing, then one family at a time from
  the large `.mm`/`.metal` files. Preserve arithmetic/APIs/lifetimes; independently
  review each diff and run focused plus integration checks.
- **PERF-2:** use checked amortized vector growth or reserve. Prove O(log n)
  allocations/O(n) copied bytes while preserving size/tail semantics, then measure.
- **PERF-3:** profile eager pipeline creation and per-operation waits. Lazy creation
  or batched submission needs explicit lifetime/readiness tests.
- **CI-1:** add CPU runtime smoke on existing hosted Linux runners; this has no
  GPU-hardware dependency.
- **CI-2:** add an Apple-GPU job when such a runner is available. Keep CUDA
  compilation, CUDA execution and Metal execution distinct.

No wholesale rewrite, mass reformat, upstream-version migration or removal of
numerical evidence is part of this cleanup. Structural changes get their own
bounded diff and acceptance criteria.
