# Scope and next steps

Updated 2026-11-06. This fork provides **Python bindings only**, backed by the
shared XGBoost C++ engine and its CPU, CUDA and Metal implementations. Python's
Dask support remains. R, Java/Scala and Spark, including `xgboost.spark`, have
been removed and are outside the project scope.

## Current status

The Python MPS backend is usable and remains experimental. The bounded
[review](REVIEW.md) found no confirmed missing major core CUDA-supported
training/prediction family. This is not proof of correctness for every branch,
NVIDIA numerical equivalence, or a general performance advantage.

The latest full regression on `28908e1` passed all 106 native groups and 4,001
Python tests, with two expected platform skips. The subsequent optimization
uses exact bit ordering for FP32 comparisons and removes redundant input-staging
writes. It passed seven native groups, 427 Python tests and three installed
workflows. Gain arithmetic, tie rules and test tolerances are unchanged.

The benchmark matrix now includes dense wide, missing-value, categorical,
seven-class Covertype (500 rounds), and full-row HIGGS (1,000 rounds), with
three CPU/candidate repeats each. All 26 MPS models, margin arrays and metric
histories match their before/after references exactly. Categorical training
improves 1.80×; other results are mixed. CPU wins all three synthetic cases,
and wide timings coincide with substantial paging.
[Current measurements](PUBLISHED_BENCHMARKS.md#current-m3-results) and
[acceptance evidence](../validation/regression-and-benchmarks-m3.json) define the scope.

## Next steps

1. **Address wide-workload memory pressure with comparable measurements.**
   Inspect histogram and split scratch footprint and repeated copies before
   changing allocation policy. Current RSS does not capture complete Metal
   physical footprint. Cross-tree scratch retention was deferred because its
   longer high-water lifetime needs stronger memory evidence; the slower
   direct-store histogram experiment was rejected.
2. **Profile the next measured costs.** HIGGS counters point to row partitioning
   and histograms; Covertype points to objectives and histograms. Counters can
   overlap, so use host timings and a focused trace to select one change.
   Preserve command timing boundaries and compare whole-training time as well
   as the targeted kernel.
   Keep CPU controls and publish regressions alongside gains. A quieter machine
   is needed before generalizing the unstable wide-workload measurements.
3. **Broaden hardware and numerical validation when available.** Actual NVIDIA
   execution, additional Apple GPUs and multi-host operation remain unvalidated.
   CUDA CI compiles but does not execute GPU kernels. The independently trained
   wide CPU/MPS models still lack a common-cut diagnosis of their differences.

Preserve algorithms, precision, ownership and test tolerances. Existing Python
workflows remain regression checks. Maintenance must address a concrete bug,
bottleneck or reviewability problem; no broad rewrite or exhaustive private-
function comparison project is planned.

[Validation history](VALIDATION.md), [platform limits](CUDA_PARITY.md) and
[archived plans](archive/PLAN-before-scope-reset-2026-11-05.md) retain earlier
milestones. The unmerged `wip/ranking-cache-comparison` branch is not accepted
coverage.
