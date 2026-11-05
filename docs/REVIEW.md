# Repository review and cleanup

2026-11-04 local / 2026-11-05 UTC. Reviewed baseline `2eaad4d` and the interrupted
vector-weight draft, now isolated at local branch
`wip/vector-weight-storage-review`, commit `ca00777`.

## Assessment

The project contains substantial real Metal implementations and useful tests.
It also accumulated serious maintainability and planning debt: a transcript-like
README, overlapping status documents, a large shared runtime, and tests that could
link a stale library. Passing aggregate test counts did not replace code review.
Discarding the port or rewriting it wholesale is not supported by these findings.

Three independent reviewers covered native runtime/storage, tests/CI/evidence,
and documentation/public API. The root integrated and checked their fixes.
The review inspected the entire unfinished diff and sampled critical baseline
paths; **it is not a proof of correctness of all 212 kernels**. No NVIDIA execution
or performance benchmark was performed during this review.

## Findings and disposition

P1 denotes unsafe behavior or a blocker for the stated full-port goal. P2 denotes
a reproduced API defect or material maintenance/validation risk; these labels
are review priorities, not a security severity classification.

| Finding | Evidence | Disposition |
| --- | --- | --- |
| **P1: unsafe CUDA view in unfinished root-weight change** | At `ca00777`, `updater_quantile_hist.cc:280` selects `ctx_->Device()` then runs CPU `std::transform` for non-MPS contexts. Manual updater selection can retain a CUDA context. | Draft quarantined, absent from main. Fix before PORT-1; source-confirmed unsafe path, not reproduced on NVIDIA. |
| **P2: CPU diagnostic schema missing 17 fields** | Existing CPU library returned 261 keys vs 278 in Metal; AUC/ranking lookups could raise `KeyError`. | Fixed with one shared stats-to-JSON mapping and an unchanged plain Statistics header. Added actual CPU/Metal schema comparison; both builds pass. |
| **P2: aliased metadata capsule consumed twice** | `set_info` and quantile setup independently transformed one raw DLPack capsule used as label and weight/base margin. | Deduplicate exports within the call, retain across quantile passes; focused ownership/single-use regressions added. Runtime verification recorded separately. |
| **P1 coverage gap: three vector payload paths incomplete** | CPU weight/root work, selected category bits and exact node/child sums differ from CUDA device storage in `multi_evaluate_splits.cuh`. | Separate PORT-1/2/3, each with producer/consumer and growth/lifetime acceptance. Weight work alone cannot close vector coverage. |
| **P2: test script could use a stale library** | Old `scripts/test.sh` compiled native tests but never rebuilt the library, including after Statistics ABI edits. | New runner builds first, selects tests, records commands/status/source and library fingerprints. |
| **P2: repetitive, all-or-nothing validation** | 727-line shell script repeated compile/link commands; focused runs required ad hoc shell scripts. | Six-line wrapper plus one runner and native manifest. All 100 original compiler/run/environment combinations preserved and mechanically compared. |
| **P2: optional dependency skips can weaken a “full” result** | `--require-metal` checks GPU availability, but PyTorch/MLX `importorskip` remains allowed. CPU/federated gates are separate too. | Document exact gate/dependency boundaries; mandatory interop acceptance must inspect skips. No claim these gaps invalidate the recorded baseline run. |
| **P2: current status buried in historical prose** | README 3,458 lines; build instructions at line 897; test totals 3,554 and 1,278 both presented as current. Parity ledger 4,559 lines. | Concise entry point/current ledger; both originals archived with repaired links. One ordered plan with acceptance criteria. |
| **P2: exact-capacity allocation growth** | `MetalVectorBuffer::Resize` copies the old prefix on each small capacity increase, yielding quadratic traffic under incremental growth. | PERF-2: checked amortized growth/reserve with copy-count tests and measurements. Impact not benchmarked in review. |
| **P2: runtime/context coupling** | Roughly 19,000 runtime/shader lines, 212 kernels, eager pipeline creation; MPS context also reports CPU storage defaults. | MAINT-1 bounded extractions; PORT-4 caller classification. No global Device/HostSpan rewrite. |
| **P2: per-operation waits and weak historical benchmark provenance** | Synchronous completion after many small operations; some older reports lack a top-level revision. | PERF-1/3 fixed-revision measurements after coverage. Old speedups are historical, not current claims. |

## What changed in this cleanup

- Preserved unfinished feature work on a local WIP branch instead of mixing it
  into cleanup or claiming it passed. Main remains based on the tested baseline.
- Fixed the two independently reproduced API defects above. The 276 statistics
  member types/order and Metal JSON expressions were independently verified
  unchanged; the duplicate CPU counter mapping was removed.
- Replaced the test harness without deleting native checks. Build-first execution,
  focused selection, per-step logs and failure status are now standard.
- Added same-branch/PR concurrency cancellation for superseded CUDA compile jobs.
  The three initial dependency failures were followed by three passing runs.
- Separated README, [plan](PLAN.md), [current status](CUDA_PARITY.md),
  [implementation guide](ARCHITECTURE.md) and [validation instructions](VALIDATION.md)
  from archived development history.

No training arithmetic, Metal kernel, allocation strategy, numerical tolerance,
or asynchronous execution contract was changed by this cleanup. No broad test
deletion or benchmark rerun was used to make the repository look healthier.

## Validation and review limits

[Cleanup validation](../validation/repo-review-2026-11-05.txt) records exact checks,
failures/fixes and final results. The prior full regression was at `dd44829`:
100 native groups / 3,554 Python passes / one expected skip. It is not silently
relabelled as a full regression of these changes.

CUDA CI checks compilation/linking only. The last pre-cleanup run
[41982974034](https://github.com/mmaaz-git/xgb-mps/actions/runs/41982974034) passed.
Compilation cannot detect the quarantined host/device pointer defect and is not
NVIDIA or Metal execution evidence.

Independent notes retain source locations, reasoning and limitations:
[runtime](reviews/2026-11-05-runtime.md),
[validation/CI](reviews/2026-11-05-validation.md),
[documentation](reviews/2026-11-05-docs.md),
[public API](reviews/2026-11-05-public-api.md).

## Follow-up audit

The subsequent [caller/numerical audit](CUDA_AUDIT.md) is now published at frozen
revision `382f78c`: 174 logical operations and 1,078 classified lexical sites,
with ten implementation tasks. It found two correctness defects and additional
GPU-placement gaps. The [updated plan](PLAN.md) puts those correctness fixes
before vector state work. The user subsequently resumed implementation;
NUM-1 is locally fixed at `2404827` and PORT-9 at `1034657`. PORT-1 is next;
see the plan and [latest validation record](../validation/cuda-port9-m3.txt).
