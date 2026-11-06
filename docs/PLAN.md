# Scope and next steps

Updated 2026-11-06. This fork provides **Python bindings only**, backed by the
shared XGBoost C++ engine and its CPU, CUDA and Metal implementations. Existing
Dask support remains. R, Java/Scala and Spark, including `xgboost.spark`, have
been removed and are outside the project scope.

## Current status

The Python MPS backend is usable and remains experimental. The bounded
[review](REVIEW.md) found no confirmed missing major core CUDA-supported
training/prediction family. That does not establish correctness for every branch,
NVIDIA numerical equivalence, or a general performance advantage.

Completed checks and results:

- High-dimensional split evaluation now reuses identical missing-direction gains
  and conservatively screens dense numeric candidates. On 1,024 / 4,096 columns,
  paired median MPS training improves 1.84× / 2.06× and split GPU time improves
  5.17× / 5.29×. All 12 MPS models, margins and metric histories match before/after;
  five native groups, 178 Python tests and three installed workflows passed.
  MPS is close to CPU at 1,024 columns; CPU remains faster at 4,096 columns.
  Independently trained CPU/MPS models still differ.
  [Optimization evidence](../validation/quantized-split-tuning-m3.json),
  [original datasets and comparisons](../validation/high-dimensional-m3.json).
- Partitioning now avoids a redundant row-ID copy for complete ordered ranges;
  sketch feature sorting omits all-zero high-key passes. Final checks passed
  four native groups, 171 Python tests and three installed-package workflows.
  Before/after models and margin arrays match exactly within each dataset across
  24 MPS runs. Targeted GPU costs improved; total training results remain mixed.
  [Latest profiling and correctness evidence](../validation/partition-sketch-tuning-m3.json).
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

1. **Profile the remaining split-stage overhead and histogram costs.** At 4,096
   columns, split GPU time is now 12.82 seconds and histogram GPU time 10.92 seconds
   within 55.97 seconds of median training. Instrument CPU preparation, submission
   and waits before choosing another change; GPU family timings can overlap and
   their sum cannot be subtracted to infer CPU time. Preserve precision and split
   selection. Retain held-out quality and cross-model differences; these datasets
   still have no common-cut diagnosis.
2. **Inspect sketch scratch-memory lifetimes.** The sampled wide worker reached
   a 6.9 GB physical-footprint peak. Check whether sorted records and temporary
   buffers can be reused without changing stable order or GPU ownership.
3. **Optimize one measured cost at a time.** Preserve algorithms, precision,
   ownership and test tolerances. Use focused correctness checks and paired
   before/after workloads with CPU controls; keep all timings, including losses.
4. **Recheck end-to-end performance at a milestone.** The current experiments
   use full dataset sizes but 100 rounds. A later 1,000-round HIGGS replay and
   broader dataset checks need more stable timing conditions.
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
