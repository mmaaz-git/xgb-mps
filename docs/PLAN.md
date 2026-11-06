# Scope and next steps

Updated 2026-11-05. This fork provides **Python bindings only**, backed by the
shared XGBoost C++ engine and its CPU, CUDA and Metal implementations. Existing
Dask support remains. R, Java/Scala and Spark, including `xgboost.spark`, have
been removed and are outside the project scope.

## Current status

The Python MPS backend is usable and remains experimental. The bounded
[review](REVIEW.md) found no confirmed missing major core CUDA-supported
training/prediction family. That does not establish correctness for every branch,
NVIDIA numerical equivalence, or a general performance advantage.

Completed checks and results:

- CPU/Metal builds and Python-only wheel/source packaging passed. A fresh wheel
  installation passed 77 workflow/model-state tests with one expected skip, plus
  regression/classification train/predict/save/load and CPU model interchange.
  [Packaging evidence](../validation/python-only-m3.json).
- All 6,184 saved HIGGS MPS cuts match the CUDA-source sketch equations. CPU uses
  a different sketch; the recorded fixed-cut training comparison agrees.
  [Diagnosis](../validation/higgs-cut-source-m3.json).
- Reusing the checked sigmoid reduced logistic-gradient GPU time from 7.78 to
  1.12 seconds median. Total training timings varied substantially, so this is
  not a general speedup claim. All six models and prediction arrays matched.
  [Performance evidence](../validation/sigmoid-reuse-m3.json).

## Next steps

1. **Test real Python workloads.** Compare predictive quality and model
   interchange with CPU; use operation-appropriate checks where CUDA and CPU
   algorithms differ. Fix demonstrated bugs or gaps with focused regressions.
2. **Profile representative workloads.** Measure end-to-end time, GPU work,
   transfers and memory pressure. Include large datasets and cases where CPU
   wins. Reuse the existing benchmark tools and recorded baselines.
3. **Optimize measured bottlenecks.** Preserve algorithms, required precision,
   data ownership and test tolerances. Report paired before/after timings and
   correctness results; retain only supported improvements.

Broader validation needs actual NVIDIA execution, additional Apple GPUs and
multi-host testing. Current hardware evidence is concentrated on one M3;
CUDA CI compiles the backend but does not execute GPU kernels.
[Validation boundaries](VALIDATION.md) and [platform limits](CUDA_PARITY.md).

Maintenance should support a concrete bug, bottleneck or reviewability problem.
There is no new broad rewrite or exhaustive private-function comparison project
in this plan. Existing audits, regression tests and validation corpora remain;
[earlier plans](archive/PLAN-before-scope-reset-2026-11-05.md) preserve historical
work. The unmerged `wip/ranking-cache-comparison` branch is not accepted coverage.
