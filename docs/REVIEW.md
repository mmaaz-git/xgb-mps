# Review of the 6f3dcbd baseline

2026-11-05; reviewed baseline `6f3dcbd`. Three subagents independently examined
feature coverage, production complexity, and repository/tests/CI hygiene. This
was a bounded source review using the existing audits and tests, not a new
exhaustive inventory or a proof of full CUDA parity.

**No confirmed missing major core CUDA training/prediction family was found.** The
remaining ranking, SHAP and linear items in older plans mostly describe missing
comparison coverage. Their Metal implementations and integration tests already
exist. The [current plan](PLAN.md) separates implementation, validation,
maintenance and performance work.

> Subsequent scope decision: R, Java/Scala and Spark bindings were removed at
> the user's request. The ecosystem finding below describes the reviewed revision,
> not an outstanding task. See [PLAN.md](PLAN.md) for the current Python scope.

## What exists, and what remains uncertain

| Area | Source and existing tests | Remaining question |
| --- | --- | --- |
| Hist/approx and quantiles | `hist_util.cc` dispatches Metal sketch construction; vector root/weight paths exist in `updater_quantile_hist.cc`; sketch and split tests cover dense/sparse, categories, weights and external pages. | HIGGS CPU/MPS cuts differ. Common-cut training agrees; actual NVIDIA comparison remains. |
| Ranking | `rank_metric.cc` has Metal metric paths; `test_metal_ranking.py` checks cache reuse and metadata lifetime. | NVIDIA numerical and caller/lifecycle comparison. |
| Exact SHAP | Metal entry points in `predictor/interpretability/shap.h` and `shap.cc`; `test_metal_shap.py` covers contributions/interactions and several model/data forms. | NVIDIA accumulation-order and numerical comparison. |
| Linear training | `updater_coordinate.cc` uses Metal reductions/residual updates; `test_metal_linear.py` covers all five selectors and CUDA-source snapshot behavior. | Actual CUDA updater comparison. |
| Distributed/federated trees | Metal collective and learner implementations, with local multi-worker tests. | Other Apple GPUs and real multi-host deployment. |

**Ecosystem gaps at the reviewed revision (bindings since removed).** In `python-package/xgboost/spark/core.py`,
`set_device` accepts only CPU/CUDA/GPU, GPU detection uses `use_cuda`, and worker
GPU assignment selects CUDA ordinals. Generic `setParams(device="mps")` can
bypass the setter, so this is an incomplete first-class integration, not proof
that every host-data Spark call fails. JNI's ordinary build script exposes CUDA
but no Metal flag; R's regular Makevars omit Metal sources/link settings. Top-level
CMake supports the binding options alongside Metal, so standard packaging and
runtime acceptance are unfinished rather than proven absent core APIs.

Python wheel machinery can package a prebuilt Metal library or build with
explicit CMake flags; Metal defaults off. Shared Dask parameter validation accepts
`mps`; no missing Dask core dispatch was established by this review.

These restrictions are shared with upstream CUDA, not missing MPS features:
`tree_method="exact"`, conditional SHAP, distributed `gblinear`, and coordinate
descent over multiple CSC blocks. Approximate SHAP has a deliberate CPU path;
the CUDA GPU predictor does not implement it either.

Metal device ordinal 0, synchronous producer readiness and explicit size limits
remain support boundaries. A UINT32 guard alone does not establish a narrower
capability than CUDA: upstream grid-stride arithmetic also needs to be examined
before claiming an extreme-size CUDA operation works.

## Cleanup performed

- Removed the unused non-quantized `metal::EvaluateSplits` and
  `metal::EvaluateMultiSplits` host bridges, declarations and eager pipeline
  creation: **145 net lines removed**. Tracked-code searches found no callers.
  Production still uses `EvaluateFeatureSplits` and the quantized implementations.
  Raw shader kernels/tests, shared helpers and statistics layout are retained.
- Removed the unconditional linear-reference build from the test runner.
  `test_metal_linear.py` already rebuilds its reference when missing or stale.
- Limited CUDA trigger exclusions to the two implementation files included only
  under `USE_METAL`. Shared headers, stubs, CMake and CUDA files still trigger CI. Source-only sparse
  checkout avoids fetching recorded GPU corpora that this compiler job does not use.
- Replaced the chronological plan with a short current queue; corrected stale
  test counts, CI status/policy and claims that resolved audit tasks remain open.

No training arithmetic, tolerance, regression test or evidence corpus was removed.
No new kernels, comparison framework or performance experiment was added.

## Necessary complexity versus optional policy

Some binary64 support is necessary: CUDA weighted quantiles use a double
threshold (`common/stats.cuh`), and multiclass gradients accumulate a double
sum (`objective/multiclass_obj.cu`). Replacing these with float can change the
algorithm or selected observation.

The FP32 arithmetic policy is a separate question. CUDA sigmoid uses `expf` and
FP32 division (`common/math.h`); Metal's `xgb_metric_math::F` routes division and
transcendentals through software binary64, and ranking helpers target CPU
rounding. That particular implementation is not dictated by the CUDA source.
It is a plausible simplification target, but this review does not establish that
native replacements would meet the required accuracy. Existing checks must remain
meaningful; numerical changes need an explicit accuracy decision and evidence.

Legacy `metal::GradientSample` is used only by its native test, and
`metal::UniformSample` has no callers. Production uses `MakeQuantizedGradients`
and `SampleUniform`/`SampleGradient`. Their legacy kernels/pipelines were left
untouched here; isolation or retirement is a separate small cleanup.

The baseline had roughly 11,500 host-runtime lines, 8,500 shader lines and 218
kernels in two main files. Structural extraction is warranted. Eager pipeline
creation, synchronous waits and exact-capacity buffer growth are performance
concerns, not missing feature families.

## Evidence and history

The test runner remains useful: it builds first, selects focused tests and records
source/library provenance. No tracked build products or Python caches were found.
Most repository bulk is intentional evidence: 680 MB in `validation/`, including
671 MB of compressed corpora. Relocating that data needs a deliberate link/manifest
migration; neither blind deletion nor history rewriting is part of this cleanup.

Current validation is recorded in [VALIDATION.md](VALIDATION.md). The
[previous review](archive/REVIEW-before-scope-reset-2026-11-05.md) and
[frozen caller audit](CUDA_AUDIT.md) preserve earlier findings. The audit's ten
implementation tasks have subsequent [local acceptance](LOCAL_ACCEPTANCE.md).
