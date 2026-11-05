# Completion plan

Updated 2026-11-04 local / 2026-11-05 UTC after the [repository review](REVIEW.md).
Feature development is paused during review cleanup. This is the work queue;
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

The 42-family/109-file inventory is not a completion percentage. A credible finish
estimate requires closing PORT-4's caller inventory. **Next: PORT-4 and PORT-5
classification, without new kernels; then PORT-1 as the first implementation.**

## Ordered work queue

| ID | Deliverable | Status / dependency | Completion criterion |
| --- | --- | --- | --- |
| CLEAN-1 | Review, concise docs, selected-test runner, reproduced API fixes | Complete | Findings published; history archived; all 100 native groups preserved; current library built first; diagnostic schemas agree; metadata alias regression passes |
| PORT-4 | Close remaining CUDA caller inventory | Next; read-only audit after CLEAN-1 | Frozen source/caller table has no unknown rows; discovered gaps become separately scoped tasks with source location and acceptance test |
| PORT-5 | Resolve numerical-contract ledger | Alongside PORT-4 | AFT tails, ranking edge policies, AUC/reduction association, RNG/compiler variants have explicit authority, comparisons, divergence fixtures and dispositions; actual defects remain open tasks |
| PORT-1 | GPU vector root/child/model weights | Paused; after scope classification | Finalize → persistent weights → root scale / model expansion / constraints stays on Metal; growth preserves queued candidates; non-MPS CPU updater still uses host views |
| PORT-2 | GPU selected categorical bitsets | Open; after PORT-1 | Vector category payload stays resident through workspace reuse, candidate queuing and model growth; numeric/one-hot/partition and alias cases pass |
| PORT-3 | GPU exact vector node/child sums | Open; after PORT-1 | Integer selected-child and node sums stay resident through evaluation/propagation; no floating reconstruction; scalar source-defined publication unchanged |
| VERIFY-1 | Integration at a fixed revision | After required PORT tasks | Full native/Python run, CPU build/API + updater checks, required tests of the optional federated build, CUDA compile; record dependencies/skips and no unexplained required-operation gaps |
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

## PORT-4 must produce a table

Freeze the table to the vendored source and one implementation commit. Enumerate
the remaining caller rows once and publish total, unclassified and missing counts.
Start from the [operation map](CUDA_OPERATION_MAP.md) and existing
[transform/linalg audit](../validation/cuda-metal-transform-callers.txt).
Preserve completed evidence instead of repeating it. Each remaining caller gets
its CUDA source location, actual MPS execution/storage path and one disposition:

- implemented, with a specific test;
- host work also performed by CUDA, with source evidence;
- Metal platform adaptation with an explicit contract;
- missing operation, with a new task ID and acceptance case.

Publish the completed table and outstanding count **before implementing additional
audit findings**. New findings update this queue explicitly; they do not silently
expand PORT-1 or reset completed work. “Mapped” is not a closure status.
Once all rows are classified, stop auditing and implement ticket by ticket.
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
