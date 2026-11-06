# Historical CUDA caller and numerical audit

Completed 2026-11-05 against implementation **`382f78c`** and vendored upstream
import **`22a84e0`**. Three independent agents reviewed training, data/runtime,
and objectives/metrics; the root reviewed shared sort/scan helpers and integrated
the results. Reviewers cross-checked the two correctness findings and narrowed
the numeric-threshold task to exclude already implemented categorical work.

**This is a frozen snapshot of `382f78c`, not the current implementation backlog.**
All ten implementation tasks identified below were subsequently completed and
accepted locally; see the [local acceptance report](LOCAL_ACCEPTANCE.md) and
[fix records](CUDA_PARITY.md). Current tested revisions and CI evidence are in
[VALIDATION.md](VALIDATION.md); the [plan](PLAN.md) defines the remaining scope.
The original findings and counts below are retained for traceability.

The first two fixes were **NUM-1 at `2404827`**
([record](../validation/cuda-num1-m3.txt)) and **PORT-9 at `1034657`**
([record](../validation/cuda-port9-m3.txt)), with focused local validation.

At the audited revision, the port was substantial but incomplete: this audit
identified ten implementation tasks, including two correctness defects. The
inventory does not support a completion percentage or a reliable calendar
estimate. Local completion of those tasks does not establish NVIDIA runtime
equivalence or a performance result.

## Classified scope at the audited revision

| Measure | Result |
| --- | ---: |
| Original CUDA source/header roots | 109 |
| Operation families | 42 |
| Lexical candidate line-sites, all assigned | 1,078 |
| Manually classified logical operations | 174 |
| Implemented paths | 113 |
| Host work also present in CUDA | 26 |
| Explicit Metal/platform adaptations | 19 |
| Missing or defective operation rows | 16 |
| Implementation tasks covering those rows | 10 |
| Numerical contracts recorded | 17 |
| Unclassified rows in this frozen ledger | 0 |

Logical rows differ in size and can share helper implementations. These counts
must not be divided to estimate completion. The lexical inventory includes
dispatches, templates, runtime support and unused definitions; it is not a C++
semantic parser. Manual caller tracing adds the actual operation and storage
meaning. Zero unclassified listed rows does not prove there are no unlisted
branches or defects.

## Original implementation findings

These are the findings as recorded at `382f78c`. All ten tasks below are now
locally complete; this table is not a work queue.

| ID | Finding | Acceptance focus |
| --- | --- | --- |
| **NUM-1** | Weighted-quantile threshold is narrowed to FP32; an existing test repeats the error | Preserve CUDA's binary64 threshold comparison; independent native and public-objective boundary regressions |
| **PORT-9** | Vector leaf validity is inferred from zero Hessians instead of sampler state | Keep selected zero-Hessian rows valid; no-sampling, uniform/MVS, reduced-gradient and empty-page fixtures; preserve scalar CUDA semantics |
| PORT-1 | Vector weights, root learning-rate scaling, model copies and prediction-cache leaf gathers use CPU payloads | GPU producer → persistent model → consumer, including growth, aliasing and delayed candidates |
| PORT-2 | Selected vector categorical bitsets return through host candidates | Resident category payload through candidate queuing, model growth and partition consumers |
| PORT-3 | Exact vector root/node/selected-child sums use host containers | Preserve exact integer sums on GPU through propagation; no reconstruction from floats |
| PORT-6 | Sampled/allowed feature IDs return to CPU consumers | Resident sampling, constraint filtering and split-consumer IDs; host launch descriptors remain allowed |
| PORT-7 | Native quantile row/group weights are read/expanded on host | Retain GPU weights and group expansion through sketch construction |
| PORT-8 | Active split evaluators gather monotonic bounds on CPU | GPU scalar/vector bound gathers; exclude the publicly rejected reduced-gradient/monotone refit combination |
| PORT-10 | Selected numeric split thresholds are materialized on CPU | GPU cut/−∞ lookup; retain categorical regression guards, without inventing a missing OHE kernel |
| PORT-11 | Sketch initialization detects categorical types and initializes pointer payload on CPU | GPU type reduction and pointer clear; host dimensions/control stay host |

Full source locations and acceptance cases are in the
[machine-readable ledger](../validation/cuda-caller-audit.json). The fixes and
their acceptance evidence are reconciled in [CUDA_PARITY.md](CUDA_PARITY.md)
and [LOCAL_ACCEPTANCE.md](LOCAL_ACCEPTANCE.md).

### The two correctness findings at `382f78c`

For labels `[0, 1]`, weights `[1, 9]`, and `alpha=float32(0.1)`, CUDA's source
expression produces a threshold of `1.0000000149011612`, selecting **1**. The
audited Metal implementation narrowed that threshold to **1**, selecting **0**.
A tiny run through the then-existing local MPS library reproduced base
score/predictions of zero and GPU dispatches.
This is an exact precision mismatch, independent of scan association. The CUDA
result was derived from source, not executed on NVIDIA.
[Diagnostic record](../validation/cuda-audit-num-1-repro.json),
[independent review](../validation/cuda-audit-num-1-crosscheck.json).

For vector trees with no subsampling, CUDA's sampler marks a selected row valid
even when every Hessian is zero. At the audited revision, the MPS leaf-position
kernel instead invalidated it. Both reviewers traced this through the active
vector updater. This finding is source-confirmed; no new production reproducer
was run. Scalar CUDA uses a
different, intentional quantized-Hessian rule and must remain separate.
[Independent review](../validation/cuda-audit-training-crosscheck.json).

## Numerical and platform boundaries

The ledger records source authority, behavior at the audited revision, existing
comparisons and required action for AFT tails/math domains, ranking race/nonfinite policies,
modern versus legacy RNG branches, weighted-CDF association, AUC conditioning,
sketch repair order, reductions, compiler contraction and SHAP accumulation.
Documented variance is not permission to hide a precision defect. NUM-1 and the
required local divergence fixtures were subsequently completed; the
[acceptance report](LOCAL_ACCEPTANCE.md) records their evidence and the remaining
NVIDIA verification requirements.

Metal buffer/runtime mechanisms replace CUDA-specific APIs. Source-defined host
orchestration, serialization and scalar publication do not require new kernels.
Specialized Metal consumers can replace CUB/Thrust templates without a generic
C++ callable launcher. Pair-index enumeration uses an explicit integer mapping
adaptation; extreme-index CUDA floating-rounding equivalence is unverified.

## Evidence and reproducibility

Independent reports: [training](audits/2026-11-05-training.md),
[data/runtime](audits/2026-11-05-data.md),
[objectives/metrics](audits/2026-11-05-numeric.md). Shared helper rows are `ROOT-*`
in the consolidated ledger. Existing transform/linalg evidence is retained.

```sh
python3 scripts/check_cuda_audit.py --check-inventory
python3 scripts/check_cuda_coverage.py
```

The [audit validation record](../validation/cuda-caller-audit-2026-11-05.txt)
includes check results and the original diagnostic command.

These checks verify frozen-source inventory, assignments, identifiers and
references. They do not verify semantic equivalence or GPU execution. Existing
tests were inspected, not rerun; the quantile diagnostic used the existing
library without a new build. No production code, tests or tolerances were
changed in this audit. No new benchmark was run.

CUDA compilation/linking at the audited revision passed in
[run 41987882440](https://github.com/mmaaz-git/xgb-mps/actions/runs/41987882440).
It ran on a CPU runner and is not NVIDIA runtime evidence. At audit publication,
the prior full local regression applied to `dd44829`, not automatically to the
audit revision. See [VALIDATION.md](VALIDATION.md) for later validation evidence.
