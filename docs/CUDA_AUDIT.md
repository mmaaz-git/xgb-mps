# CUDA caller and numerical audit

Completed 2026-11-05 against implementation **`382f78c`** and vendored upstream
import **`22a84e0`**. Three independent agents reviewed training, data/runtime,
and objectives/metrics; the root reviewed shared sort/scan helpers and integrated
the results. Reviewers cross-checked the two correctness findings and narrowed
the numeric-threshold task to exclude already implemented categorical work.

This is a frozen audit. Since publication, **NUM-1 was fixed at `2404827`** with
focused local validation ([record](../validation/cuda-num1-m3.txt)). Nine audited
implementation tasks remain; the [plan](PLAN.md) tracks current status.

**The port is substantial but incomplete. This audit identifies 10 implementation
tasks, including two correctness defects.** It does not support a completion
percentage or a reliable calendar estimate. Full integration, NVIDIA execution
comparisons and performance work remain separate milestones.

## Classified scope

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

## Required implementation work

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
[machine-readable ledger](../validation/cuda-caller-audit.json). The
[ordered plan](PLAN.md) puts correctness fixes first, then dependent vector state,
then the remaining placement tasks. Each implementation is a bounded change.

### The two correctness findings

For labels `[0, 1]`, weights `[1, 9]`, and `alpha=float32(0.1)`, CUDA's source
expression produces a threshold of `1.0000000149011612`, selecting **1**. Metal
narrows that threshold to **1**, selecting **0**. A tiny run through the existing
local MPS library reproduced base score/predictions of zero and GPU dispatches.
This is an exact precision mismatch, independent of scan association. The CUDA
result was derived from source, not executed on NVIDIA.
[Diagnostic record](../validation/cuda-audit-num-1-repro.json),
[independent review](../validation/cuda-audit-num-1-crosscheck.json).

For vector trees with no subsampling, CUDA's sampler marks a selected row valid
even when every Hessian is zero. The MPS leaf-position kernel instead invalidates
it. Both reviewers traced this through the active vector updater. This finding
is source-confirmed; no new production reproducer was run. Scalar CUDA uses a
different, intentional quantized-Hessian rule and must remain separate.
[Independent review](../validation/cuda-audit-training-crosscheck.json).

## Numerical and platform boundaries

The ledger records source authority, current behavior, existing comparisons and
required action for AFT tails/math domains, ranking race/nonfinite policies,
modern versus legacy RNG branches, weighted-CDF association, AUC conditioning,
sketch repair order, reductions, compiler contraction and SHAP accumulation.
Documented variance is not permission to hide a precision defect. NUM-1 remains
open; required divergence fixtures remain acceptance work under the plan.

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
It ran on a CPU runner and is not NVIDIA runtime evidence. The prior full local
regression still applies to `dd44829`, not automatically to this audit revision.
