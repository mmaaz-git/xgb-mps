# Scope and remaining work

Updated 2026-11-05 after a three-agent review of `6f3dcbd` and cleanup `383f19d`.
The user replaced the open-ended full-parity goal with the Python usability,
HIGGS diagnosis, correctness and measured-performance phase below, and confirmed
the Python phase as the priority. The Python phase is complete. The user then
chose to remove R, Java/Scala and Spark, including the Python Spark adapter,
from this fork. Those integrations are excluded, not unfinished MPS tasks.
Python, the shared C++ engine/C API and existing Dask support remain.
This file is the current plan; the [previous plan](archive/PLAN-before-scope-reset-2026-11-05.md)
retains completed tasks and comparison history.

## What “full MPS implementation” means

Within the Python/shared-engine scope, support the user-facing CUDA-accelerated
operations and their required GPU work
and data placement in the vendored XGBoost 3.4.2 snapshot on the documented Apple
GPU platform. Preserve the algorithms, required
precision, data ownership and GPU execution. Use CPU comparisons where the
algorithms agree, and operation-appropriate checks where they differ.

Feature coverage, numerical validation and performance are separate claims.
Universal bitwise reproduction of every NVIDIA math-library result, compiler
choice or racing update is not a useful definition of feature coverage.
Existing correctness tests and tolerances remain intact in this cleanup.

The original audit's ten implementation tasks and two later capacity fixes have
local acceptance. The new bounded review found **no confirmed missing major
core CUDA-supported training/prediction feature family**. This is not proof that every branch is correct.
See the [review](REVIEW.md), [support boundaries](CUDA_PARITY.md) and
[local acceptance](LOCAL_ACCEPTANCE.md).

## Remaining work

| Area | Actual gap | What would close it |
| --- | --- | --- |
| Quantile/training agreement | All 6,184 saved HIGGS MPS cuts match CUDA-source equations; CPU uses a different sketch. Fixed-cut training agrees in the recorded 100-round check. | NVIDIA execution remains a separate validation question; the CPU cut difference alone is not a demonstrated MPS defect. |
| Cross-backend validation | Ranking, SHAP and linear training exist, but NVIDIA numerical and public-workflow comparisons are incomplete. | Representative end-to-end comparisons of the implemented features; add focused coverage for actual discrepancies. An unfinished replay program is not an unimplemented feature. |
| Deployment coverage | Testing is concentrated on one M3. Distributed/federated tests used multiple workers on that machine. | Test additional Apple hardware and actual multi-host use before claiming those configurations validated. |
| Platform boundaries | Device ordinal 0, synchronous native-input readiness and explicit allocation/index limits are documented. | State the supported surface clearly; extend a boundary only when a required, supported CUDA use case is identified. Do not infer a CUDA gap from a Metal guard alone. |
| Maintainability | Runtime/shader monoliths, legacy test-only paths and numerical-policy complexity remain. | Small extractions or proven dead-code removals, each retaining relevant regression checks. |
| Performance | Full HIGGS remains slower on this M3 than CPU in the latest 1,000-round measurement. | Optimize measured bottlenecks after correctness; report complete before/after results, including CPU wins. |

## Completed Python acceptance checklist

The cleanup is complete; findings are in [REVIEW.md](REVIEW.md). The Python phase
used this acceptance checklist:

1. **Python installation and use:** build a wheel, install it into a fresh
   environment, verify the imported package/library come from that environment,
   and run existing workflow tests outside the checkout without source-path
   injection. Exercise training, prediction, save/load and CPU model interchange.
2. **HIGGS diagnosis:** inspect the saved benchmark, cuts and common-cut evidence;
   identify the source of the cut difference as far as the evidence permits.
   Use an existing NVIDIA comparison if hardware is available. Clearly separate
   an established defect, expected algorithm differences and unresolved behavior.
3. **Correctness:** fix any demonstrated defect from these checks and retain a
   focused regression. Preserve required CUDA algorithms/precision and existing
   tolerances. Do not add comparison infrastructure without a concrete failure.
4. **Performance:** after correctness checks, optimize a measured bottleneck and
   evaluate it with the existing benchmark tools, paired timings and correctness
   checks. Retain only supported improvements; record unsuccessful experiments.

Results: item 1 passed again on the optimized build, which is also installed in
the project's `.venv` ([fresh installation](../validation/python-install-m3.json));
item 2 has a source-level explanation ([all HIGGS cuts](../validation/higgs-cut-source-m3.json)).
Neither check demonstrated a production defect requiring item 3 changes.
Item 4 reduced the measured logistic-gradient GPU cost from 7.78 to 1.12 seconds
median by reusing the existing checked sigmoid implementation. All six HIGGS
models/prediction arrays match exactly. Total training timing was variable;
the 1.081x median ratio is not a general speedup claim.
[Performance and focused regression evidence](../validation/sigmoid-reuse-m3.json).
NVIDIA runtime equivalence is not claimed by these results.

Installation instructions, revision-specific evidence, the HIGGS conclusion and
the performance result are published. This completes the agreed Python phase;
the remaining validation/deployment work in the table is a backlog, not an
automatic extension into exhaustive private-function replay or bitwise identity.

NVIDIA access is required to execute CUDA comparisons; none is configured.
That limits a CUDA-equivalence claim, not the ability to use or maintain this
experimental MPS backend. The [advanced comparison guide](NVIDIA_COMPARISON.md)
is a reference, not an automatic task list.

## Cleanup priorities

- **Done in this pass:** remove two unused host split bridges and their eager
  pipelines; let linear tests build their own reference only when needed;
  exclude the two Metal-only implementation files from CUDA triggers; reconcile
  status documents and archive completed planning history.
- **Next maintenance candidate:** isolate or retire the legacy CPU-derived
  sampling implementation after preserving its useful reference role. Production
  already uses the CUDA-derived sampling path.
- **Structural follow-up:** extract runtime/storage and one kernel family at a
  time. Keep file movement separate from changes to arithmetic or scheduling.
- **Numerical-policy follow-up:** distinguish required binary64 operations from
  the extra policy of emulating FP32 transcendental/division behavior. Neither
  deleting all emulation nor silently weakening tests is justified by this review.

Core/Python regression tests and recorded evidence are retained. Dedicated
R/JVM/Spark tests were removed with their bindings; earlier audit records still
describe the revisions they examined. The approximately
680 MB validation corpus should eventually have an artifact-storage policy;
this cleanup does not delete it or rewrite Git history. The unfinished
`wip/ranking-cache-comparison` branch remains outside main and is not accepted
coverage. The rejected logistic-specialization experiment stays rejected.
