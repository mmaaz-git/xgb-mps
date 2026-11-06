# Replaying published CUDA workloads on Apple Metal

The purpose is to test workloads on which CUDA demonstrated substantial gains,
and identify where this backend falls short. A historical GPU/CPU ratio is not a
prediction for an M3. These runs compare the current CPU `hist` implementation
with this repository's Metal implementation on the same machine. CUDA is not
available on this Mac.

## Primary sources

| Publication | Workload | Reported comparison |
| --- | --- | --- |
| [NVIDIA, September 2017](https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/) | HIGGS: 10.5M training rows, 28 features, 1,000 rounds | 4.15×, one Tesla P100 versus two Xeon E5-2698 v3 CPUs (32 cores) |
| [Mitchell et al., June 2018, Table 2](https://arxiv.org/html/1806.11248v1) | HIGGS, 500 rounds | 509.29 / 38.41 s = 13.26× |
| Same paper | Covertype, 500 rounds | 3532.26 / 107.70 s = 32.80× |

The 2018 machine has **eight V100s and 64 CPU cores**, and its
[benchmark script](https://github.com/RAMitchell/GBM-Benchmarks/blob/a0bbed08c918b0a82e9a5e2207d1f43134b445e0/benchmark.py)
defaults to `n_gpus=-1` (all GPUs). Those figures are not single-GPU speedups.
The repository README's example results differ from the paper: notably Covertype
is 464.05 / 103.39 s = **4.49×**, not 32.80×. We retain that discrepancy rather
than imply the paper's exact run is fully specified by the current repository.

Both sources compare histogram algorithms. This avoids the misleading comparison
of GPU `hist` with the much slower CPU `exact` algorithm. The 2017 script is the
main full-size single-GPU workload; Covertype adds multiclass trees and a case
where the present backend may be slower.

## Presets and measurement boundaries

`nvidia2017-higgs` uses the original first 10,500,000 training rows and last
500,000 test rows, `binary:logitraw`, `error`, depth 6, learning rate 0.3, 256 bins,
and 1,000 rounds. The article leaves most parameters at their defaults; depth,
learning rate and bins are verified against
[XGBoost v0.7 tree parameters](https://github.com/dmlc/xgboost/blob/v0.7/src/tree/param.h).
The old fixed base score of 0.5 is set explicitly, avoiding modern automatic
intercept estimation. Test evaluation runs every round. The original `error`
metric on raw margins is retained; conventional accuracy at margin zero and
binary log loss are reported separately after timing.

`gbm2018-covtype` uses two `train_test_split` calls with `random_state=0`, first
reserving 20% test, then 25% of the remainder for validation. It trains depth 6,
learning rate 0.1, 500 rounds, evaluating train and validation every round.
The reference loader leaves class labels at 1 through 7; the benchmark computes
`num_class=max(label)+1`. Our preset therefore preserves **eight output classes,
including unused class zero**, instead of silently changing its workload.
Inputs are converted to FP32, which also matches XGBoost's stored feature values.

For both presets:

- The primary timing is `xgb.train`, including first feature quantization and
  per-round evaluation. Download, CSV conversion, DMatrix construction and final
  test prediction are excluded. DMatrix and DMatrix-plus-training times are also
  recorded, so the setup cost is visible.
- Each run uses a fresh worker process and fresh matrices. A small untimed model
  warms its runtime. CPU and MPS order alternates between repeats; raw runs and
  medians are retained. Progress logging is periodic rather than every round.
- CPU uses all eight M3 CPU cores. No concurrent tests, builds or other benchmark
  jobs run during timing. Unrelated user applications remain open. Peak process
  RSS and system-wide swap are recorded; swap is not attributed to this process.
- Wall and monotonic clocks must agree within one second. Sleep or a clock jump
  invalidates the worker result. Completed GPU histogram counters are required
  for MPS and must be zero for the CPU training interval.
- Every repeat compares all test margins with `atol=rtol=2e-5`. Failures remain
  explicitly marked `prediction_equivalence_failed`. Accuracy, log loss and
  class disagreement are additional information, not substitutes for that check.
  Worker models and margin arrays remain locally available for diagnosis.
- The current library and historical XGBoost have different implementations.
  These are workload replays, not exact replications of old hardware/software.
  Any row-count or round-count override is recorded and is a scaling experiment.

## Reproduce

```sh
.venv/bin/python scripts/prepare_published_data.py higgs covtype
.venv/bin/python scripts/benchmark_published.py --preset nvidia2017-higgs \
  --threads 8 --repeats 3 --output results/published-higgs.json
.venv/bin/python scripts/benchmark_published.py --preset gbm2018-covtype \
  --threads 8 --repeats 3 --output results/published-covtype.json
```

The preparation script streams compressed CSVs into memory-mapped arrays in
100,000-row chunks, verifies dimensions, finite values and label domains, and
stores source URLs and SHA-256 hashes. Completed caches/results are reused or
protected against accidental overwrite. Run long jobs in `tmux` and inspect
existing output first. Download/conversion is a separate job from timing.

Datasets: [HIGGS, D. Whiteson (2014)](https://doi.org/10.24432/C5V312) and
[Covertype, J. Blackard (1998)](https://doi.org/10.24432/C50K5N), distributed by UCI
under CC BY 4.0. Dataset files stay in the ignored local cache; this repository
contains provenance, commands and results.

## Accepted production tree `68c800e`

The 2026-11-05 replay uses the full 10.5M/500K HIGGS split and 1,000 rounds,
with three alternating CPU/MPS pairs. The production tree and library hashes
were unchanged throughout; the benchmark harness was at `0074861`. This is the
accepted production code at that milestone, including the CUDA-style sketch and
subsequent coverage fixes. Later reader/cleanup changes have separate evidence.

| Measurement | CPU | MPS |
| --- | ---: | ---: |
| Training runs, seconds | 195.167, 287.918, 289.030 | 458.656, 469.100, 486.718 |
| Median training, seconds | 287.918 | 469.100 |
| Median DMatrix + training, seconds | 289.307 | 470.713 |
| Test accuracy, all repeats | 0.761210 | 0.761654 |

CPU/MPS median training ratio is **0.614×**: MPS takes about 1.63 times as long.
Each pair fails strict margin equivalence on **499,937 of 500,000** test values:
maximum absolute difference 5.15643, RMS 0.360498, and 30,266 class disagreements.
Accuracy differs by 0.000444 and log loss by 0.000106572; similar quality does not
turn the failed equivalence check into a pass.

These measurements were on AC power with no concurrent local build/test jobs.
User applications remained open. Recorded system-wide swap use ranged from
19,608 to 22,778 MB; this is not the benchmark's own swap use. Peak process RSS
was approximately 3.78–4.06 GB. CPU timing varied substantially, so the median
and all individual runs are retained. Wall/monotonic differences were below
0.017 seconds. These results do not isolate hardware limits from implementation
costs or system memory pressure.

The first MPS run records 176.56 s of histogram GPU time, 76.50 s objective time,
75.02 s partition time and 10.48 s split time. Wait time overlaps GPU execution
and must not be added as independent work. Diagnose correctness before tuning
these measured costs.

The saved MPS model produces **bit-identical margins for all 500,000 test rows**
when evaluated with the CPU predictor, with no Metal prediction dispatches.
The separately trained models already differ at tree zero's root: both choose
feature 25, but CPU uses threshold 1.0758163 and MPS 1.0759506. Fresh one-round
replays reproduce both first trees exactly. Their exported quantile cuts differ
for 24 of 28 features. This locates a difference before prediction, but does not
prove that quantile construction explains every subsequent divergence.

A separate **100-round common-cut diagnostic** uses a CPU reference cut set,
with CPU and Metal quantile matrices independently constructed in fresh workers.
All 500,000 margins pass the original tolerance: maximum difference `1.43e-6`,
RMS `3.29e-8`; 452,930 are bit-identical. Every split agrees and test-error
histories match exactly. Ninety-seven trees have at least one differing leaf
weight; the first difference is one FP32 ULP in tree 3. The CPU control's cuts
and first 100 predictive trees exactly reproduce the original CPU baseline.
These observations implicate quantile construction, while retaining the
1,000-round failure and the limit of this 100-round diagnostic. They do not
establish Metal/CUDA sketch agreement; that needs NVIDIA execution.

A subsequent source-reference check resolves the cut-construction question at
the source-equation level: **all 6,184 saved MPS cuts across all 28 features match
the existing CUDA-source scalar reference exactly**. It preserves the five
original 64M-entry windows and their partial-row counts. CPU cuts differ on 24
features; CPU's streaming queue/level sketch and CUDA's sorted-window sketch are
different algorithms. The MPS root threshold `1.0759506` is in the reference cuts;
the CPU threshold `1.0758163` is not. This difference does not demonstrate a Metal
sketch bug. The initial unweighted lookup was vectorized for this full-size
diagnostic and checked against the existing scalar lookup on 15 cases; later
pruning, merging, duplicate repair and queries reuse the existing reference.

This is CPU evaluation of CUDA source equations, **not NVIDIA execution**. It
does not certify compiler rounding or independently replay all 1,000 training
rounds. The recorded cross-model margin comparison remains failed, and the
100-round common-cut check retains its original scope.
[Source-reference evidence](../validation/higgs-cut-source-m3.json).

[Raw runs](../validation/perf1-higgs-1000-m3.json),
[source/dataset provenance](../validation/perf1-higgs-1000-provenance-m3.json),
the [first-divergence diagnosis](../validation/perf1-higgs-diagnosis-m3.json)
(diagnostic source at `52069b2`), and
[common-cut diagnosis](../validation/perf1-higgs-common-cuts-m3.json).
Models and prediction arrays remain in the ignored local results directory.
Reproduce the diagnostic after a completed benchmark and provenance check:

```sh
.venv/bin/python scripts/diagnose_higgs.py \
  --benchmark results/perf1-higgs-1000.json \
  --provenance results/perf1-higgs-1000-provenance.json \
  --output-dir results/perf1-higgs-diagnosis

# Separate correctness experiment; not a replacement performance measurement.
.venv/bin/python scripts/diagnose_higgs.py \
  --benchmark results/perf1-higgs-1000.json \
  --provenance results/perf1-higgs-1000-provenance.json \
  --output-dir results/perf1-higgs-common-cuts --common-cuts --rounds 100

# Inspect saved cuts using source equations; no library rebuild or training.
.venv/bin/python scripts/diagnose_higgs.py \
  --benchmark results/perf1-higgs-1000.json \
  --provenance results/perf1-higgs-1000-provenance.json \
  --source-cuts-from results/perf1-higgs-diagnosis \
  --output-dir results/higgs-cut-source-20261105
```

## Initial M3 measurements

These use the backend at commit `989ee7a`, before the low-bin histogram change.

| Workload | CPU training | MPS training | CPU / MPS | Validation |
| --- | ---: | ---: | ---: | --- |
| Full HIGGS, **100 rounds**, one paired pilot | 23.168 s | 24.579 s | 0.943× | All 500,000 test margins pass; maximum difference `9.54e-7` |
| Full Covertype, **500 rounds**, median of three pairs | 30.194 s | 92.504 s | 0.326× | Seven margins fail strict comparison; all predicted classes agree |

The HIGGS pilot has the full 10.5M/500K split, but fewer rounds than the published
benchmark. Its CPU/MPS accuracy is 0.741512 and every reported test-error value
agrees. The initial three Covertype pairs give the same numerical comparison:
seven of 929,624 margins differ beyond tolerance, maximum difference 0.126695 and
RMS 0.000238. All differences are in output class 6. All 116,203 predicted classes
agree, accuracy is 0.894564, and test log loss differs by `1.12e-8`. This is a
strict-equivalence failure even though classification quality is unchanged.

Raw [HIGGS pilot](../validation/published-higgs-100-baseline-m3.json),
[Covertype runs](../validation/published-covtype-500-baseline-m3.json), and
[two-round integration check](../validation/published-covtype-smoke-m3.json)
include all timings, counters, evaluation histories, memory snapshots and clocks.
No CUDA-scale training gain has been reproduced in these initial runs.

The HIGGS pilot spends 9.22 s in histogram GPU commands and 4.69 s in partition
commands, versus 0.29 s in split evaluation. The first full Covertype run spends
35.20 s in histogram commands and 8.80 s in split evaluation. GPU command times
are diagnostic components, not separate training timings; wait time overlaps
GPU execution and must not be added to it.

## Full 1,000-round HIGGS replay

With the low-bin change at `6e87d03`, the full published HIGGS workload completes
in **190.852 s CPU / 193.740 s MPS (0.985×)**. This is **one paired run**, not a
three-repeat estimate. DMatrix-plus-training is 192.970 / 195.336 s. Both models
classify every test row identically, with **76.121% accuracy**. All 500,000 test
margins pass the strict check; maximum difference is `3.34e-6`, RMS `3.04e-7`.
Per-round error histories differ by at most `2e-6` (one test row), and final test
log loss differs by `8.8e-12`.

Peak process RSS is 3.77 GB CPU / 4.01 GB MPS. Wall/monotonic clock differences
are below 0.002 s, so no sleep interval invalidated this run. Histograms account
for 82.65 s of MPS GPU command time and partition operations 45.35 s; split
evaluation accounts for 5.78 s. These are the next substantial GPU costs to
investigate. The measured training ratio is effectively a tie, substantially
below the historical P100 ratio. Hardware and software differences prevent
attributing the whole gap to either the chip or the implementation.

[Full raw report](../validation/published-higgs-1000-m3.json). Reproduce with:

```sh
.venv/bin/python scripts/benchmark_published.py --preset nvidia2017-higgs \
  --threads 8 --repeats 1 --output results/published-higgs-1000-v1.json
```

## Exact binary-column histogram accumulation

The new path sums one- and two-bin features in lane registers, reduces within
SIMD groups, and uses global integer atomic additions. This avoids one set of
contended threadgroup atomic updates per row. The sums are exact modulo 64 bits,
as with the existing histogram arithmetic. Tests now cover 576 histogram cases,
including partial SIMD groups, empty nodes, negative values, low-word carries,
zero gradients, page offsets and all index widths. The complete native suite
and 583 Python tests pass under Metal API Validation (one Python test skipped).

A randomized, same-process comparison of the baseline and current shader uses
one million rows, 32 features, five repeats and an independent integer reference
for every result. At the root with 32 rows per lane, the measured kernel gains
are **5.21× for one bin**, **3.59× for two bins**, and **0.99× for 256 bins**.
On 32 balanced, fragmented nodes, the two-bin case at the production eight rows
per lane is **0.96×**. These are histogram kernel measurements, not training
speedups; slower layouts are retained in the
[one-bin](../validation/histogram-lowbin-1-m3.json),
[two-bin](../validation/histogram-lowbin-2-m3.json) and
[256-bin](../validation/histogram-lowbin-256-m3.json) reports.

```sh
git show 989ee7a:xgboost/src/common/metal_histogram.metal > results/baseline.metal
clang++ -O2 -std=c++17 -fobjc-arc scripts/metal_histogram_benchmark.mm \
  -framework Metal -framework Foundation -o results/metal_histogram_compare
results/metal_histogram_compare xgboost/src/common/metal_histogram.metal \
  results/baseline.metal 2 > results/histogram-comparison.json
```

## Covertype after the binary-column change

Three new alternating CPU/MPS pairs at `6e87d03` give medians of **33.320 s CPU /
70.289 s MPS (0.474×)**. Compared with the initial MPS median of 92.504 s, observed
MPS training time is **24.0% lower**. Median histogram GPU time drops from
**35.172 to 15.616 s**, with exactly **21,500 histogram dispatches** in every run.
In the first pair, split evaluation stays at 8.80 / 8.77 s and partition work at
3.22 / 3.19 s before/after, consistent with the targeted histogram improvement.

These before/after training runs are separate sessions. The CPU median moved
from 30.194 to 33.320 s, so the wall-time reduction includes environmental
variation. The randomized kernel comparison and GPU component timings provide
additional evidence for the optimization; they do not establish a guaranteed
24% gain on other runs or machines. Within these measurements, CPU remains
about twice as fast as MPS on Covertype.

All six before/after margin-array comparisons (three CPU and three MPS) are
**bit-for-bit identical**, verified with both array comparisons and SHA-256
hashes. The optimization changes performance without changing this workload's
predictions. Each new CPU/MPS pair still has the same seven strict margin
mismatches and zero class disagreements. Accuracy remains 89.4564%. The report
is explicitly `prediction_equivalence_failed`, not a claim of complete numerical
equivalence. Maximum wall/monotonic disagreement across all twelve initial and
new Covertype workers is 0.00111 s.

[Raw repeated runs](../validation/published-covtype-500-lowbin-m3.json) and
[before/after verification](../validation/published-covtype-lowbin-verification-m3.json).
Reproduce the updated workload with:

```sh
.venv/bin/python scripts/benchmark_published.py --preset gbm2018-covtype \
  --threads 8 --repeats 3 --output results/published-covtype-lowbin-v1.json
```

The next performance investigations are gradient/index reuse in wide histograms,
row-partition memory traffic, and synchronization overhead from many small GPU
commands. The current results do not establish a hardware-only limit. The target
is useful, measured MPS gains over this machine's CPU; historical CUDA ratios help
select workloads rather than define an identical speedup requirement.


## Compressed ELLPACK reader

On 2026-11-05, profiling the existing HIGGS run led to a small change in
`xgb_ellpack_read`: read packed symbols by byte rather than fetching each bit
separately, with direct paths for one- and eight-bit symbols. The bitstream,
head/tail split handling, histogram arithmetic and training algorithm are unchanged.
The baseline is the production tree at `dc568a5`; the candidate shader and
library hashes are retained in the [run record](../validation/ellpack-read-m3-run.json).

Three fresh-process pairs use all 10.5 million training rows and 500,000 test
rows, **100 rounds**, depth six, 256 bins and eight CPU threads. The existing
HIGGS worker is unchanged, with separate frozen Python/library copies; loaded
library paths were independently checked. Within-pair order was randomized.

| Median time | Before | After |
| --- | ---: | ---: |
| Training | 64.754 s | 55.194 s |
| Histogram GPU time | 25.214 s | 12.916 s |
| Partition GPU time | 8.106 s | 7.852 s |

Training takes **14.8% less time (1.17× speedup)** in this comparison. Baseline
trials were 49.389, 83.526 and 64.754 seconds; candidate trials were 38.400,
58.064 and 55.194 seconds. All six saved models and all 500,000-margin arrays
are bit-identical. All workers used AC power; system-wide swap was high and
wall times varied considerably. These are before/after MPS measurements, not
CPU/NVIDIA speedups or a repeat of the published 1,000-round workload.
The earlier full HIGGS CPU/MPS result and its strict-equivalence failure remain.

The extended existing histogram microbenchmark compares compressed pages using
one million rows, 32 features, five repeats, randomized baseline/candidate order,
and an independent exact integer reference for every result. Across root,
balanced and skewed node layouts and both row batch sizes, median kernel ratios
are **1.25–2.56× for eight-bit symbols**, **1.08–1.48× for four-bit symbols**, and
**0.97–1.08× for one-bit symbols**. Every timing is retained. An initial general
byte decoder regressed one-bit kernels; the direct one-bit path corrected that
regression before the final measurements.

The existing six focused native groups and 160 Python tests pass. Native cases
cover all symbol widths 1–32, cross-byte copies and head/tail boundaries; Python
checks exercise ordinary, native and external-memory matrices. No tolerance was
changed. This is focused acceptance of the reader change; the last full regression
remains at `41d4dc2`. [Complete acceptance log](../validation/ellpack-read-m3.txt).

To reproduce the compressed kernel comparison, use fresh output files:

```sh
git show dc568a5:xgboost/src/common/metal_histogram.metal > results/reader-baseline.metal
clang++ -O2 -std=c++17 -fobjc-arc scripts/metal_histogram_benchmark.mm \
  -framework Metal -framework Foundation -o results/reader-compare
for bins in 2 16 256; do
  results/reader-compare xgboost/src/common/metal_histogram.metal \
    results/reader-baseline.metal "$bins" --ellpack > "results/reader-kernel-$bins.json"
done
```

For training, build the baseline and candidate in separate checkouts with
`bash scripts/build.sh`, then alternate three fresh workers per build against
the same prepared HIGGS cache. Use a different output path for each worker:

```sh
.venv/bin/python scripts/benchmark_published.py --preset nvidia2017-higgs \
  --worker mps --cache /absolute/path/to/.cache/datasets --threads 8 \
  --rounds 100 --progress-every 25 --output results/reader-trial.json
```

The run record includes all six worker reports, power and memory observations,
source/library hashes, dataset provenance, kernel timings and focused-test status.


## Rejected logistic specialization

A separate experiment compiled the logistic branch of the objective kernel as
its own pipeline, keeping the shared arithmetic unchanged. Three native groups
and 59 Python tests passed; all six 100-round HIGGS models and prediction arrays
were bit-identical. Three before/after pairs gave median training times of
**51.509 / 50.406 seconds (1.02×)** and objective GPU times of
**8.953 / 8.619 seconds (1.04×)**. Objective GPU time was slower in the first pair;
training times ranged from 39.9 to 55.0 seconds across the experiment.

This did not establish a convincing benefit for the additional pipeline, so the
candidate was removed and the accepted reader source and library restored.
The [decision record](../validation/objective-specialization-m3-run.json) links
compressed raw reports, the discarded patch and focused-test evidence. No
production arithmetic or tolerance was changed, and this experiment adds no
NVIDIA runtime evidence. Do not repeat it unchanged.
