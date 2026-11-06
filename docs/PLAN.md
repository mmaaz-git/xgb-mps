# Scope and remaining work

Updated 2026-11-05 after a three-agent review of `6f3dcbd`.
The automatic full-parity goal is **paused**. This cleanup does not resume it.
This file is the current plan; the [previous plan](archive/PLAN-before-scope-reset-2026-11-05.md)
retains completed tasks and comparison history.

## What “full MPS implementation” means

Support the user-facing CUDA-accelerated operations and their required GPU work
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
| Spark integration | The device setter rejects MPS; GPU resource/ordinal and interchange paths recognize CUDA. Generic host-data execution may work but is unvalidated. | Add first-class MPS selection/resource behavior and test Spark train/transform. |
| R/JVM distribution | Normal package build tools lack first-class Metal configuration; explicit CMake combinations exist. | Provide and validate standard Metal-enabled build/install paths. Core API support alone is not a packaging result. |
| Quantile/training agreement | HIGGS CPU and MPS cuts differ; fixed-cut training agrees. This is not yet diagnosed against NVIDIA. | Run the existing quantile/training comparison and resolve any demonstrated MPS defect. |
| Cross-backend validation | Ranking, SHAP and linear training exist, but NVIDIA numerical and public-workflow comparisons are incomplete. | Representative end-to-end comparisons of the implemented features; add focused coverage for actual discrepancies. An unfinished replay program is not an unimplemented feature. |
| Deployment coverage | Testing is concentrated on one M3. Distributed/federated tests used multiple workers on that machine. | Test additional Apple hardware and actual multi-host use before claiming those configurations validated. |
| Platform boundaries | Device ordinal 0, synchronous native-input readiness and explicit allocation/index limits are documented. | State the supported surface clearly; extend a boundary only when a required, supported CUDA use case is identified. Do not infer a CUDA gap from a Metal guard alone. |
| Maintainability | Runtime/shader monoliths, legacy test-only paths and numerical-policy complexity remain. | Small extractions or proven dead-code removals, each retaining relevant regression checks. |
| Performance | Full HIGGS remains slower on this M3 than CPU in the latest 1,000-round measurement. | Optimize measured bottlenecks after correctness; report complete before/after results, including CPU wins. |

## Immediate execution order

1. This bounded cleanup and its focused checks are complete. Keep the
   implementation goal paused; the findings are in [REVIEW.md](REVIEW.md).
2. Before resuming development, choose one concrete deliverable from the table.
   The highest-value correctness check is the existing quantile/training comparison.
3. Use demonstrated failures or required unsupported use cases to select code
   changes. Do not automatically add more private-function replay programs,
   synthetic edge-case corpora or benchmark batches.

NVIDIA access is required to execute CUDA comparisons; none is configured.
Spark and packaging integration are real remaining ecosystem work, independent
of that access; this cleanup identifies them without starting another porting batch.
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

All regression tests and recorded evidence are retained. The approximately
680 MB validation corpus should eventually have an artifact-storage policy;
this cleanup does not delete it or rewrite Git history. The unfinished
`wip/ranking-cache-comparison` branch remains outside main and is not accepted
coverage. The rejected logistic-specialization experiment stays rejected.
