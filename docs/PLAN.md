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

The requested regression, missing-data optimization and wide-memory work is
complete for `61b2042`. The final backend passed **106 native groups and 4,001
Python tests**, with two expected platform skips, plus three installed-wheel
workflows. All before/after benchmark models and predictions are exact.

In three-repeat comparisons, 20%-missing training falls from 18.62 to 12.15
seconds; CPU takes 10.13 seconds in the same comparison. Wide-data median physical
peak falls from 6.43 to 6.04 GiB (6.2%) in separate diagnostics. Its 100-round
training median is 6.1% slower in this comparison; the memory saving does not
establish a speed improvement.
The [latest comparisons](PUBLISHED_BENCHMARKS.md#missing-value-and-memory-optimization-m3)
and [acceptance evidence](../validation/missing-and-memory-m3.json) include all
controls, exact commands, quality differences and physical-memory limits.

## Remaining work

1. **Continue only from measured costs.** Missing-data histogram/split work
   remains a cost; the measured CPU/MPS results depend on the workload. The wide
   physical peak has moved from sketch construction to tree construction;
   retained histogram/split storage remains substantial. Further changes need
   paired whole-training controls, exact outputs and physical-memory measurements.
2. **Broaden validation.** Investigate the pre-existing wide CPU/MPS model
   difference with common cuts. Actual NVIDIA execution, additional Apple GPUs
   and multi-host operation remain unvalidated. CUDA CI only compiles.

Preserve algorithms, precision, ownership and test tolerances. Large performance
workloads remain in scripts, separate from correctness tests. No broad rewrite
or exhaustive private-function comparison project is planned.

[Validation history](VALIDATION.md), [platform limits](CUDA_PARITY.md) and
[archived plans](archive/PLAN-before-scope-reset-2026-11-05.md) retain earlier
milestones. The unmerged `wip/ranking-cache-comparison` branch is not accepted
coverage.
