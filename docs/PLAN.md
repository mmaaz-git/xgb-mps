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
map to **10 implementation tasks** below; all ten are locally fixed and validated.
These unequal-sized rows are not a
completion percentage, and this bounded audit cannot prove the absence of
unlisted defects. **VERIFY-1 local acceptance is complete. Next: VERIFY-2 and PERF-1.** No new kernels were added during
classification.

## Ordered work queue

| ID | Deliverable | Status / dependency | Completion criterion |
| --- | --- | --- | --- |
| CLEAN-1 | Review, concise docs, selected-test runner, reproduced API fixes | Complete | Findings published; history archived; all 100 native groups preserved; current library built first; diagnostic schemas agree; metadata alias regression passes |
| PORT-4 | Classify frozen CUDA caller inventory | Classification complete | All 1,078 listed sites assigned; 174 logical rows have source/Metal/test references and dispositions; findings are separate tasks, not a parity certificate |
| PORT-5 | Classify numerical-contract ledger | Local contract reconciliation complete; NVIDIA comparisons remain | All seventeen contracts have named passing fixtures and explicit limits in the [acceptance report](LOCAL_ACCEPTANCE.md); external execution remains VERIFY-2 |
| NUM-1 | Correct weighted-quantile threshold precision | Locally complete at `2404827` | Binary64 comparison and independent boundary/alias regressions pass; three native groups and 34 Python tests; [evidence](../validation/cuda-num1-m3.txt). NVIDIA comparison remains VERIFY-2 |
| PORT-9 | Correct vector row validity after sampling | Locally complete at `1034657` | Retained sampler state controls positions; 42 new exact none/uniform/MVS cases plus zero-Hessian production updater coverage; four native groups, 52 Python tests and CPU build/API/updater checks pass; [evidence](../validation/cuda-port9-m3.txt) |
| PORT-1 | GPU vector root/child/model weights | Locally complete at `c78cb43`; CUDA compile passed at `383b448` | Resident producer/storage/consumer chain, growth/aliases and non-MPS host views checked; 100 native groups plus final predictor check, 3,595 Python passes and CPU checks; [evidence](../validation/cuda-port1-m3.txt) |
| PORT-2 | GPU selected categorical bitsets | Locally complete at `6a606e1`; CUDA compile passed at `99cbeb3` | Resident selection, queued state, model/view/partition consumers; 100 native groups, 3,597 Python passes and CPU checks; [evidence](../validation/cuda-port2-m3.txt) |
| PORT-3 | GPU exact vector node/child sums | Locally complete at `ab87872`; CUDA compile passed at `a531694` | Resident exact roots, queued child storage, parent gathers and propagation; 101 native groups, 3,603 Python passes and CPU checks; [evidence](../validation/cuda-port3-m3.txt) |
| PORT-6 | Resident sampled/allowed feature IDs | Locally complete at `2bfb914`; CUDA compile passed at `0409c7f` | GPU sampler/filter/task/gather/consumer chain, exact assembly and lifetime checks; 102 native groups, 3,603 Python passes and CPU checks; [evidence](../validation/cuda-port6-m3.txt) |
| PORT-7 | Resident native quantile weights | Locally complete at `796e9fa`; CUDA compile passed at `0409c7f` | Native row/group weights and expansion stay GPU-owned through quantile and host/disk constructors; six native groups, 820 Python passes and CPU checks; [evidence](../validation/cuda-port7-m3.txt) |
| PORT-8 | GPU split-bound gathers | Locally complete at `ed5afcf`; CUDA compile passed at `0409c7f` | GPU gather/snapshot/feature/finalizer chain and root consumers; 103 native groups, 3,647 Python passes and CPU checks; [evidence](../validation/cuda-port8-m3.txt). Unsupported reduced-gradient/monotone refit restriction unchanged |
| PORT-10 | GPU numeric split thresholds | Locally complete at `30639c6`; CUDA compile passed at `bf1ebea` | Resident cuts feed GPU threshold selection and category handling; 1,624 exact threshold cases, 103 native groups, 3,647 Python passes and CPU checks; [evidence](../validation/cuda-port10-m3.txt) |
| PORT-11 | GPU sketch initialization | Locally complete at `68c800e`; CUDA compile passed at `bf1ebea` | GPU type reduction, owned snapshot and pointer clear; 72 new constructor/lifetime cases, 103 native groups, 3,647 Python passes, CPU checks and 14 federated groups; [evidence](../validation/cuda-port11-m3.txt) |
| VERIFY-1 | Integration at a fixed revision | Locally complete; NVIDIA execution remains separate | [Acceptance report](LOCAL_ACCEPTANCE.md) maps all 17 contracts and 16 original gap rows; full Metal/CPU/federated evidence at `68c800e`, focused numerical fixtures at `0c66454`; CUDA compile passed at `bf1ebea` |
| VERIFY-2 | NVIDIA execution comparison | Runners addressing ten contracts and production sketch/training/prediction added; Metal evidence published, external NVIDIA hardware required | [Harness instructions](NVIDIA_COMPARISON.md); shared fixtures must run against vendored CUDA on NVIDIA; seven contracts lack dedicated runners and broader caller coverage remains, including ranking gradients/unbiased objectives; publish differences, not just compiler success |
| PERF-1 | Profile/optimize a frozen benchmark matrix | Full HIGGS repeated at `68c800e`; ordinary strict equivalence failed, common-cut 100-round diagnostic passes | Repeated CPU/MPS timings and correctness for large HIGGS, wide synthetic, sparse and vector workloads; improvements and regressions published |

VERIFY-2 is an evidence gap, not a reason to keep editing unrelated Metal code
while hardware is unavailable. Cross-Apple-hardware claims also require actual runs.

## Next bounded work: NVIDIA comparison and performance

[VERIFY-1 local acceptance](LOCAL_ACCEPTANCE.md) is complete: all seventeen
numerical contracts and sixteen original gap rows have named evidence. Full
Metal/CPU/federated gates cover `68c800e`; the new 104th native group passed
separately without production changes. [CUDA run
42027936238](https://github.com/mmaaz-git/xgb-mps/actions/runs/42027936238) at
`bf1ebea` passed all 197 compile/link steps on the same production tree.

VERIFY-2 needs an actual NVIDIA runner. The [native harness](NVIDIA_COMPARISON.md)
connects P5-N01/08/13/14 inputs to vendored quantile/sketch consumers and records
full output bits; its 39 Metal witnesses and twelve report-integrity tests pass.
The eighteen added P5-N01 cases isolate binary64 threshold precision using
adjacent alphas, reversed input order and three weight scales. A production
pipeline runner also records eight ordinary-matrix cases and full HIGGS sketch
construction, with source/input hashes, GPU-execution checks and CPU evaluation
of each GPU model. Its Metal reports and 18 synthetic integrity tests pass;
HIGGS cuts and its first tree reproduce the original benchmark exactly. Actual
CUDA execution and the remaining contract/caller harnesses are still needed.
Integrate the remaining source fixtures with CUDA consumers, recording GPU model,
NVCC/CCCL/libdevice versions and flags. Use the per-contract exact, bounded,
race-policy and domain-extension rules in the acceptance report. Portable inputs
alone are not a completed NVIDIA harness; compiler success is not execution.
Resolve concrete discrepancies instead of widening ordinary tolerances.

The P5-N02 AFT runner now consumes the complete existing Metal corpus as exact
input bits: 96 cases / 316,683 rows, with all three distributions, weights,
sensitive tails and domain guards retained. Its Metal export reproduces the
previous 322 CPU-tolerance field differences and zero production/source field
differences. The CUDA runner uses the actual objective for 75 ordinary cases;
21 domain guards call the source functions directly. Eleven synthetic report
checks pass. See the
[AFT run record](../validation/aft-contract-m3-run.json).
The numerical-v2 CUDA consumer and optional nvCOMP link probe passed in
[run 42045096489](https://github.com/mmaaz-git/xgb-mps/actions/runs/42045096489).
The AFT and numerical consumers also compiled and linked successfully in
[run 42051624772](https://github.com/mmaaz-git/xgb-mps/actions/runs/42051624772);
that build predates the RNG consumer.

P5-N06 now has a runner for the unchanged ranking corpus: 659,200 distribution
draws and 9,068,696 raw pairs across 74 production cases. Shared fixture inputs
were checked bit for bit against the old generators, and all three affected
Metal groups pass. The CUDA runner invokes the production ranking cache, sorts
and `MakePairsOp`, comparing each observed result exactly with the same
independent host oracle used by Metal. It also covers 1,060,276 Philox/uniform
queries through production `RandomWeight`. The
[run record](../validation/rng-contract-m3-run.json) retains the initial Metal
checks; the extension record below tracks the current CUDA compiler result.

The `rng-v2` extension covers the training RNG contract's distinct Thrust shuffle
stream: all 128 existing unweighted primitive cases / 1,320,913 permutation
entries, plus 15 actual `ColumnSampler` hierarchies / 210 outputs and parent
engine consumption. The shared matrix preserves the old column test's complete
input/case ordering; its full Metal group passes with the original 550 cases
and 33 resident chains. Weighted log-key arithmetic and broader caller
combinations remain separate comparison work. The
[extension record](../validation/rng-column-contract-m3-run.json) links the base
RNG evidence without relabeling it as NVIDIA execution. The corrected RNG,
numerical and AFT consumers compiled and linked in
[run 42058206746](https://github.com/mmaaz-git/xgb-mps/actions/runs/42058206746).

The P5-N03 primitive-math runner exports the unchanged binary64 arithmetic/cast,
exp/log/erf and FP32 hypot/tanh corpora: **1,942,163 observed fields**. Three Metal
groups and 16 synthetic report-integrity tests pass. Existing CPU budgets remain
unchanged; GPU-to-GPU math differences require review, even at one ULP. See the
[run record](../validation/math-contract-m3-run.json). The math consumer compiled
and linked in [run 42062682076](https://github.com/mmaaz-git/xgb-mps/actions/runs/42062682076);
NVIDIA execution remains pending. This does not close P5-N12's
compiler/contraction requirement or establish universal error bounds.
The ranking-rounding runner adds P5-N04/05 observations from all 64 existing
cases: 6,345,280 result/diagnostic fields. The Metal export and 14 synthetic
integrity tests pass. CUDA calls the production gradient and bias-update
consumers, with 32 fresh repetitions of each smaller case to characterize the
shared-factor race. Its compile/runtime validation remains pending. The
[record](../validation/ranking-rounding-m3-run.json) distinguishes cached fields
from separately evaluated source diagnostics. The broader gradient and repeated
unbiased-objective suites still need matching exports and consumers.
Runners now address ten contracts; seven still lack dedicated runners, broader
caller coverage remains, and none has actual NVIDIA execution evidence.

Continue VERIFY-2 in the following bounded batches. Reuse the accepted Metal
corpora and invoke the corresponding vendored CUDA consumers; publish complete
inputs and observations with each batch. A runner's completion does not close
its contract until actual NVIDIA results have been reviewed.

| Order | Remaining contract work | Required comparison |
| --- | --- | --- |
| 1 | P5-N04/05: ranking rounding, gradients and unbiased updates | Complete compiler/runtime validation of the 64-case rounding runner; export the full gradient and repeated unbiased-objective corpora, including NDCG and batched cases. Retain the separate primitive-rounding boundaries. |
| 2 | P5-N07 and training-reduction-association | Stats, linalg, metric and training reductions under their declared term precision and association bounds |
| 3 | P5-N09/10/11: PR, ROC and ranking metrics | Retain pathological inputs, fallback branches, top-k/soft labels and prefix/IDCG diagnostics; distinguish common valid CUDA inputs from Metal extensions |
| 4 | training-shap-order | Same models/inputs through actual prediction consumers; retain cross-tree accumulation differences |
| Across all batches | P5-N12: compiler and contraction behavior | Record and inspect emitted CUDA behavior and matching runtime results; primitive tests or compiler flags alone do not close this contract |

After these batches, reconcile broader caller combinations against the frozen
operation ledger, including weighted sampling and the objective/metric paths
outside each primitive corpus. If NVIDIA hardware becomes available, run the
ready quantile/training comparison first and process concrete discrepancies.

PERF-1 can now proceed on the M3 while external NVIDIA evidence is unavailable.
Freeze the accepted source and workload parameters; inspect prior outputs before
running repeated CPU/MPS HIGGS, wide, sparse and vector benchmarks. Separate
construction, training and prediction timing, retain correctness results and CPU
wins, then profile a measured bottleneck before changing production code.

The [full HIGGS replay](PUBLISHED_BENCHMARKS.md#accepted-production-tree-68c800e)
finished all three CPU/MPS pairs (10.5M training rows, 1,000 rounds): median
288 s CPU / 469 s MPS, with the unchanged strict margin comparison failing in
every pair. CPU prediction of the saved MPS model matches all 500,000 margins
bit for bit. Fresh one-round replays reproduce each backend's first tree exactly;
24 of 28 features have different quantile cuts, and the first root threshold
already differs. A separate 100-round common-cut diagnostic now passes all
500,000 test margins at the original tolerance; all splits agree and evaluation
histories match. Small leaf-weight differences remain. The CPU control reproduces
the original CPU cuts and first 100 predictive trees exactly. This implicates
quantile construction but does not prove CUDA sketch equivalence or explain
every difference after 1,000 rounds. Retain the original failure; do not change
formulas solely to obtain CPU identity or widen tolerances. Prioritize actual
CUDA sketch outputs in VERIFY-2, then expand its remaining contract/caller
coverage. Wide, sparse and vector timings remain PERF-1 work.
Histogram, objective and partition timings are the first profiling candidates;
the observed system-wide swap pressure must accompany published timings.

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
