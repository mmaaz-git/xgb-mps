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

The latest full regression on `28908e1` passed 106 native groups and 4,001 Python
tests, with two expected platform skips. The current memory/partition change
passed five native groups, 148 Python tests and three installed workflows.
It removes a redundant histogram allocation and caches columns for repeated
partition gathers on eligible large, narrow pages. Existing kernels, arithmetic,
tie rules and test tolerances are unchanged.

All 20 MPS workers in the current 30-run comparison reproduce baseline models,
margins and every-round metrics exactly. HIGGS at 100 rounds improves 1.22× by
three-repeat median. Full 1,000-round HIGGS improves from 271.79 to 250.14 seconds
in a single run per variant; it needs repeats before generalizing that ratio.
The wide case saves 256 MiB of Metal allocation but has essentially unchanged
training time and no demonstrated physical-peak reduction. The HIGGS cache adds
about 279 MiB net Metal allocation. CPU wins the synthetic controls.
[Measurements](PUBLISHED_BENCHMARKS.md#current-m3-results) and
[acceptance evidence](../validation/memory-and-partition-m3.json) define the scope.

## Next steps

1. **Locate the remaining wide-data physical-memory peak.** The redundant
   histogram buffer is removed on the contiguous first-page path. Separate
   data/matrix/quantization/training phases before changing more allocation
   policy; the process peak did not fall with the 256 MiB Metal saving.
   Inspect measured histogram/split scratch and copies, preserve bounded
   lifetimes, and use a quieter machine before generalizing timing differences.
2. **Continue from measured HIGGS costs.** The column cache improves partition
   locality using existing kernels. Histograms remain a substantial cost;
   two/eight-feature tiles were slower than the existing four-feature tile
   and were rejected. Dense-byte/SIMD-scan variants were inconclusive and
   rejected too. Any next kernel change needs stable-row microbenchmarks,
   whole-training CPU/before/after controls, memory accounting and exact outputs.
   Keep command timing boundaries; counters overlap and must not be summed.
3. **Broaden numerical and hardware validation.** Diagnose the pre-existing
   wide CPU/MPS model difference with common cuts. Actual NVIDIA execution,
   additional Apple GPUs and multi-host operation remain unvalidated. CUDA CI
   only compiles. The current changes have focused acceptance; the earlier
   full suite does not become a full run of the current revision.

Preserve algorithms, precision, ownership and test tolerances. Existing Python
workflows remain regression checks. Maintenance must address a concrete bug,
bottleneck or reviewability problem; no broad rewrite or exhaustive private-
function comparison project is planned.

[Validation history](VALIDATION.md), [platform limits](CUDA_PARITY.md) and
[archived plans](archive/PLAN-before-scope-reset-2026-11-05.md) retain earlier
milestones. The unmerged `wip/ranking-cache-comparison` branch is not accepted
coverage.
