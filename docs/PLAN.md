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
map to **10 implementation tasks** below; NUM-1, PORT-9 and PORT-1/2/3/6/7/8 are locally fixed,
and two remain.
These unequal-sized rows are not a
completion percentage, and this bounded audit cannot prove the absence of
unlisted defects. **Next: PORT-10 numeric thresholds, then PORT-11 sketch initialization.** No new kernels were added during
classification.

## Ordered work queue

| ID | Deliverable | Status / dependency | Completion criterion |
| --- | --- | --- | --- |
| CLEAN-1 | Review, concise docs, selected-test runner, reproduced API fixes | Complete | Findings published; history archived; all 100 native groups preserved; current library built first; diagnostic schemas agree; metadata alias regression passes |
| PORT-4 | Classify frozen CUDA caller inventory | Classification complete | All 1,078 listed sites assigned; 174 logical rows have source/Metal/test references and dispositions; findings are separate tasks, not a parity certificate |
| PORT-5 | Classify numerical-contract ledger | Classification complete; acceptance work remains | Seventeen contracts record authority, comparisons, limits and actions; NUM-1 is locally fixed; required divergence fixtures and external comparisons remain VERIFY-1/2 work |
| NUM-1 | Correct weighted-quantile threshold precision | Locally complete at `2404827` | Binary64 comparison and independent boundary/alias regressions pass; three native groups and 34 Python tests; [evidence](../validation/cuda-num1-m3.txt). NVIDIA comparison remains VERIFY-2 |
| PORT-9 | Correct vector row validity after sampling | Locally complete at `1034657` | Retained sampler state controls positions; 42 new exact none/uniform/MVS cases plus zero-Hessian production updater coverage; four native groups, 52 Python tests and CPU build/API/updater checks pass; [evidence](../validation/cuda-port9-m3.txt) |
| PORT-1 | GPU vector root/child/model weights | Locally complete at `c78cb43`; CUDA compile passed at `383b448` | Resident producer/storage/consumer chain, growth/aliases and non-MPS host views checked; 100 native groups plus final predictor check, 3,595 Python passes and CPU checks; [evidence](../validation/cuda-port1-m3.txt) |
| PORT-2 | GPU selected categorical bitsets | Locally complete at `6a606e1`; CUDA compile passed at `99cbeb3` | Resident selection, queued state, model/view/partition consumers; 100 native groups, 3,597 Python passes and CPU checks; [evidence](../validation/cuda-port2-m3.txt) |
| PORT-3 | GPU exact vector node/child sums | Locally complete at `ab87872`; CUDA compile passed at `a531694` | Resident exact roots, queued child storage, parent gathers and propagation; 101 native groups, 3,603 Python passes and CPU checks; [evidence](../validation/cuda-port3-m3.txt) |
| PORT-6 | Resident sampled/allowed feature IDs | Locally complete at `2bfb914`; CUDA CI pending | GPU sampler/filter/task/gather/consumer chain, exact assembly and lifetime checks; 102 native groups, 3,603 Python passes and CPU checks; [evidence](../validation/cuda-port6-m3.txt) |
| PORT-7 | Resident native quantile weights | Locally complete at `796e9fa`; CUDA CI pending | Native row/group weights and expansion stay GPU-owned through quantile and host/disk constructors; six native groups, 820 Python passes and CPU checks; [evidence](../validation/cuda-port7-m3.txt) |
| PORT-8 | GPU split-bound gathers | Locally complete at `ed5afcf`; CUDA CI pending | GPU gather/snapshot/feature/finalizer chain and root consumers; 103 native groups, 3,647 Python passes and CPU checks; [evidence](../validation/cuda-port8-m3.txt). Unsupported reduced-gradient/monotone refit restriction unchanged |
| PORT-10 | GPU numeric split thresholds | Next | Selected cut and −∞ materialization executes on GPU; category-bit construction already exists, with categorical cases retained as regression guards |
| PORT-11 | GPU sketch initialization | Open; after PORT-10 | GPU categorical-type reduction and pointer-payload clear; numeric/categorical/mixed/empty and subsequent-batch fixtures |
| VERIFY-1 | Integration at a fixed revision | After required implementation tasks | Full native/Python run, CPU build/API + updater checks, optional federated checks, CUDA compile; complete required numerical divergence fixtures; record dependencies/skips and no unexplained required-operation gaps |
| VERIFY-2 | NVIDIA execution comparison | External hardware required | Shared fixtures run against vendored CUDA on NVIDIA; publish numerical contracts and failures, not just compiler success |
| PERF-1 | Profile/optimize a frozen benchmark matrix | After local coverage/correctness milestone | Repeated CPU/MPS timings and correctness for large HIGGS, wide synthetic, sparse and vector workloads; improvements and regressions published |

VERIFY-2 is an evidence gap, not a reason to keep editing unrelated Metal code
while hardware is unavailable. Cross-Apple-hardware claims also require actual runs.

## Next bounded change: PORT-10

Materialize selected numeric split thresholds on GPU. Scalar and vector
evaluators currently use the selected bin to read a cut value or construct
negative infinity on CPU; the CUDA split kernels perform these operations.

- Feed cut values into GPU winner/finalization code and return the selected
  threshold with the existing source-defined selected-node metadata.
- Preserve scalar exclusive-bin and vector forward/backward formulas, including
  first-bin negative infinity. Invalid/root-only candidates must avoid cut loads.
- Check exact threshold bits, missing directions and ties with CUDA-source
  fixtures; retain OHE/partition category regressions and full-tree checks.
- Categorical bit construction already runs on GPU. Sketch initialization remains
  PORT-11; this task does not expand into unrelated selection or arithmetic work.

Resident split-bound gathers are recorded under
[PORT-8](../validation/cuda-port8-m3.txt): GPU updates feed owned snapshots,
feature evaluation and finalization before CPU comparisons; aliases, growth,
source lifetime and scalar/vector complete-tree checks pass. The unsupported
reduced-gradient/monotonic combination remains rejected, as in CUDA.

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
