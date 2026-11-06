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

- The full baseline passed 106 native groups and 4,001 Python tests, with two
  expected skips. New workflows cover preprocessing and cross-validation,
  weighted categorical classification, early stopping, grouped ranking and
  persistence. [Workflow evidence](../validation/python-workflows-full-m3.json).
- Large-workload profiling now lives in `scripts/profile_training.py`, separate
  from tests. A four-feature histogram tile reuses row IDs and gradients; median
  histogram GPU time improved 2.04× on wide data and 1.62× on HIGGS. Focused
  correctness and installed-package checks passed; before/after models and
  predictions are bit-identical. CPU remains faster on both measured workloads.
  [Optimization evidence](../validation/histogram-tiling-m3.json).
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

1. **Investigate row partitioning and first-round setup.** In the latest HIGGS
   measurements, partition GPU time is about 7.7 seconds and sketch time about
   5 seconds. Inspect memory traffic and synchronization before changing code;
   GPU counters can overlap and cannot be summed into total training time.
2. **Optimize one measured cost at a time.** Preserve algorithms, precision,
   ownership and test tolerances. Use focused correctness checks and paired
   before/after workloads with CPU controls; keep all timings, including losses.
3. **Recheck end-to-end performance at a milestone.** The current experiments
   use full dataset sizes but 100 rounds. A later 1,000-round HIGGS replay and
   broader dataset checks should follow meaningful cumulative improvements.
   Existing real-workflow tests remain regression gates.

Broader validation needs actual NVIDIA execution, additional Apple GPUs and
multi-host testing. Current hardware evidence is concentrated on one M3;
CUDA CI compiles the backend but does not execute GPU kernels.
[Validation boundaries](VALIDATION.md) and [platform limits](CUDA_PARITY.md).

Maintenance should support a concrete bug, bottleneck or reviewability problem.
There is no new broad rewrite or exhaustive private-function comparison project
in this plan. Existing audits, regression tests and validation corpora remain;
[earlier plans](archive/PLAN-before-scope-reset-2026-11-05.md) preserve historical
work. The unmerged `wip/ranking-cache-comparison` branch is not accepted coverage.
