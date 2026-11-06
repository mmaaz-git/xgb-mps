# Scope and next steps

Updated 2026-11-06. This fork provides **Python bindings only**, backed by the
shared XGBoost C++ engine and its CPU, CUDA and Metal implementations. Python's
Dask support remains. R, Java/Scala and Spark, including `xgboost.spark`, have
been removed and are outside the project scope.

## Current status

The Python MPS backend is usable and remains experimental. The bounded
[review](REVIEW.md) found no confirmed missing major core CUDA-supported
training/prediction family. That does not establish correctness for every branch,
NVIDIA numerical equivalence, or a general performance advantage.

The latest milestone batches split-stage commands, widens eligibility for
existing shared histogram kernels on dense wide data, and reduces conservative
bound construction cost. Exact split scoring, integer statistics and tie rules
are preserved. Eight native groups, 300 Python tests and three installed-package
workflows passed. All 18 MPS models, margin arrays and per-round metric histories
match before/after across the two synthetic workloads and full-row HIGGS.

Median training improves 1.06× / 1.45× / 1.06× on 1,024 columns, 4,096 columns and
HIGGS, respectively. Individual pairs and CPU comparisons vary; one HIGGS pair
regresses slightly. Independent CPU/MPS training still produces different models.
[Current measurements](PUBLISHED_BENCHMARKS.md#current-m3-results) and
[acceptance evidence](../validation/parallel-tuning-m3.json) define the scope.

## Next steps

1. **Choose the next bottleneck from comparable measurements.** Split commands
   and histogram work remain substantial on wide data; HIGGS spends about seven
   GPU seconds each on partitioning and histograms. Measure stages using stable
   boundaries before changing them. Batching changed the split counter from
   separate intervals to a contiguous span, so its ratio is not a scoring-kernel
   speedup. CPU wait samples also include necessary GPU execution.
2. **Check a longer real workload at a milestone.** The latest HIGGS runs use all
   10.5M training rows but 100 rounds. A later 1,000-round replay and broader
   dataset checks should retain CPU controls, all timings and memory/power
   observations. Improve timing stability before making general speedup claims.
3. **Broaden validation when hardware is available.** Actual NVIDIA execution,
   additional Apple GPUs and multi-host operation remain unvalidated. CUDA CI
   compiles the backend but does not execute GPU kernels. The wide datasets have
   no common-cut diagnosis of their CPU/MPS training differences.

Preserve algorithms, precision, ownership and test tolerances. Optimize one
measured cost at a time, with focused correctness checks and paired workloads.
Existing real Python workflows remain regression gates. Maintenance must address
a concrete bug, bottleneck or reviewability problem; no broad rewrite or
exhaustive private-function comparison project is planned.

[Validation history](VALIDATION.md), [platform limits](CUDA_PARITY.md) and
[archived plans](archive/PLAN-before-scope-reset-2026-11-05.md) retain earlier
milestones. The unmerged `wip/ranking-cache-comparison` branch is not accepted
coverage.
