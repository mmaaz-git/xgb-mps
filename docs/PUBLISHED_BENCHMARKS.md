# Performance benchmarks

Current results are first; published CUDA references and earlier local milestones
remain below as historical evidence. Do not combine ratios from different sessions
or revisions. CUDA is not available on this Mac.

## Current M3 results

The 2026-11-06 memory/partition milestone starts from `048dc86`. It removes a
redundant histogram allocation and adds a bounded column cache for partitioning
large, narrow dense pages. Existing shaders, integer arithmetic, split tie rules
and test tolerances are unchanged. [Acceptance evidence](../validation/memory-and-partition-m3.json)
records exact source/library hashes, all 30 training workers and commands.

The final build passed five native groups and 148 Python tests, followed by
three installed-package workflows outside the checkout. These are focused
checks; the full 106-native/4,001-Python run on `28908e1` predates both this and
the previous optimization. There was no NVIDIA execution.

### Repeated whole-training measurements

Each cell is a median of three fresh-process runs on the M3 (10-core GPU,
eight CPU cores, 16 GB RAM), rotating CPU/previous-MPS/new-MPS order. Runtime
initialization is warmed. All use 100 rounds, depth 6 and 256 bins. Training
includes quantization, column-cache construction when eligible, and held-out
evaluation each round. Loading, matrix creation and final prediction are separate.

| Training data | CPU | Previous MPS | New MPS | Previous / new MPS |
| --- | ---: | ---: | ---: | ---: |
| 25K × 4,096 dense, synthetic | 39.00 s | 44.29 s | 44.48 s | 1.00× |
| HIGGS: 10.5M × 28 | 37.78 s | 37.13 s | 30.50 s | 1.22× |
| 100K × 32 categorical, synthetic | 2.09 s | 5.24 s | 5.09 s | 1.03× |

HIGGS improves **1.22×** by median. The wide case is essentially flat
(44.29 → 44.48 s); its allocation saving is not a demonstrated training-speed
gain. CPU remains faster on both synthetic controls. Other applications and system paging were
not controlled. No builds, tests or other benchmarks from this task overlapped
these timings. All repeats, including regressions, are retained below.

| Workload / variant | Training seconds, in repeat order |
| --- | --- |
| 25K × 4,096 dense, synthetic / CPU | 28.939, 39.004, 39.085 |
| 25K × 4,096 dense, synthetic / previous MPS | 40.356, 44.289, 45.458 |
| 25K × 4,096 dense, synthetic / new MPS | 46.615, 44.478, 43.822 |
| HIGGS: 10.5M × 28 / CPU | 37.784, 39.896, 36.317 |
| HIGGS: 10.5M × 28 / previous MPS | 38.128, 35.443, 37.126 |
| HIGGS: 10.5M × 28 / new MPS | 31.089, 30.496, 30.469 |
| 100K × 32 categorical, synthetic / CPU | 2.090, 1.860, 2.141 |
| 100K × 32 categorical, synthetic / previous MPS | 5.241, 5.251, 5.096 |
| 100K × 32 categorical, synthetic / new MPS | 5.094, 5.093, 5.214 |

Including matrix construction, CPU/new-MPS median seconds are:

- 25K × 4,096 dense, synthetic: 39.206 / 44.693.
- HIGGS: 10.5M × 28: 39.098 / 31.834.
- 100K × 32 categorical, synthetic: 2.097 / 5.099.

### Full-length HIGGS confirmation

A separate **single run per variant**, in previous-MPS/CPU/new-MPS order,
uses all 10.5M training rows, 500K held-out rows and 1,000 rounds:

| CPU | Previous MPS | New MPS | Previous / new MPS | CPU / new MPS |
| ---: | ---: | ---: | ---: | ---: |
| 296.40 s | 271.79 s | 250.14 s | 1.09× | 1.18× |

This confirms the full-length workload and exact before/after output. One run
per variant does not establish the repeatability of the long-run speedup.
Do not substitute these values for the previous milestone's three-run medians.

### Memory and partition costs

Separate 30-round diagnostic runs sample Metal `currentAllocatedSize` and
`task_vm_info` after each iteration. Physical peak is process-lifetime peak,
which can precede training. Values below are MiB; these instrumented single-run
timings are not used in the training medians above.

| Workload | Previous / new maximum sampled Metal allocation | Previous / new process peak physical footprint |
| --- | ---: | ---: |
| 25K × 4,096 dense | 4468.8 / 4212.8 | 6543.1 / 6545.2 |
| 10.5M × 28 HIGGS | 3078.5 / 3357.4 | 9733.6 / 9760.4 |

The wide case saves **256 MiB of Metal buffer allocation** by accumulating first-page
histograms directly in contiguous integer-cache destinations. Later pages and
noncontiguous output retain the original staging/addition path. This does not
establish a lower physical-memory peak or eliminate system paging.

HIGGS gains a 294 MB (280.4 MiB) column cache; net measured Metal allocation rises
about **279 MiB** after the histogram saving. It is limited to dense eight-bit
compressed pages with 256 symbols, 4–64 features, at least 2^20 rows and contiguous
full-head/full-tail storage. Each page's cache is capped at the smaller of
512 MiB and one thirty-second of Metal's recommended working-set size. This is
not a global budget. Wide, small and split-storage pages use the original path.
The page owns the cache and initializes it once, including concurrent first use.

Seven paired isolated repeats on five stable within-node row layouts gave
**1.6–2.25× partition speedups** with exact CPU stable-partition output. The
294 MB transpose took 17.228 ms in that diagnostic. Whole-training measurements
include cache creation; initial transpose time is in `index_gpu_time_ns`, separate
from partition commands. Neither isolated timings nor counter sums are training
time. GPU counter families may overlap.

Median HIGGS GPU-family durations in the repeated 100-round runs:

| Variant | Partition | Histogram |
| --- | ---: | ---: |
| Previous MPS | 7.052 s | 6.853 s |
| New MPS | 4.319 s | 6.773 s |

The remaining histogram kernel was also profiled: two- and eight-feature tiles
were slower than the existing four-feature tile and were rejected. Dense-byte
classifier specialization and a SIMD rank scan gave small inconsistent gains
and were also rejected. An allocation-lifetime probe disproved the suspected
autorelease retention; no pool change or persistent split-scratch policy was added.

### Correctness and held-out quality

All 20 MPS training workers reproduce their baseline models, margin arrays and
per-round metric histories exactly. Every worker also checks same-model CPU
prediction exactly. New native checks cover output offsets, guards, empty nodes,
page replacement/addition, cache lifetime, concurrent first use, reuse and fallback
layouts. Independently trained CPU/MPS models can still differ:

| Workload | CPU accuracy | MPS accuracy | CPU log loss | MPS log loss |
| --- | ---: | ---: | ---: | ---: |
| 25K × 4,096 dense, synthetic | 0.7036 | 0.6848 | 0.5963 | 0.5873 |
| HIGGS: 10.5M × 28 | 0.7415 | 0.7422 | 0.5113 | 0.5110 |
| 100K × 32 categorical, synthetic | 0.7178 | 0.7178 | 0.6190 | 0.6190 |
| HIGGS: 10.5M × 28, 1,000 rounds | 0.7612 | 0.7617 | 0.4828 | 0.4827 |

The wide CPU/MPS accuracy difference already existed in the baseline. Exact
before/after MPS agreement does not resolve its cause. Power-source and
wall/monotonic-clock checks pass for every accepted worker. System swap is
system-wide, and neither RSS nor Metal allocation measures total system GPU
physical memory.

### Reproduce the workloads

Use a fresh output directory for each run. Prepare HIGGS once if it is not cached:

```sh
.venv/bin/python scripts/prepare_published_data.py higgs
.venv/bin/python scripts/profile_training.py --case wide --rows 25000 \
  --features 4096 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide-4096
.venv/bin/python scripts/profile_training.py --case higgs --rounds 100 \
  --threads 8 --repeats 3 --output results/profile-higgs-100
.venv/bin/python scripts/profile_training.py --case categorical --rows 100000 \
  --features 32 --categories 256 --informative-features 16 --rounds 100 \
  --threads 8 --repeats 3 --output results/profile-categorical
```

Use `--rounds 1000` for full-length HIGGS. The public record includes every
before/after worker command and frozen `--package-dir` library hash; the commands
above measure the built package in the current checkout. Historical milestones follow.

## Regression and broader benchmark milestone (048dc86)

The 2026-11-06 milestone starts from `28908e1`. Its full standard regression
passed 106 native groups and 4,001 Python tests, with two expected platform
skips. The final optimization then passed seven native groups, 427 Python tests
and three installed-package workflows. Exact sources, commands, all timings and
output hashes are in the [acceptance record](../validation/regression-and-benchmarks-m3.json).

Each CPU/candidate cell below is the median of three fresh-process runs on this
M3 (10-core GPU, eight CPU cores, 16 GB RAM). Runtime initialization is warmed.
Training includes quantization and held-out evaluation every round; loading,
matrix creation and final prediction are separate. All cases use depth 6 and
256 bins. GPU timing counters may overlap and must not be summed.

| Training data | Rounds | CPU | MPS |
| --- | ---: | ---: | ---: |
| 100K × 32 categorical, synthetic | 100 | 1.48 s | 5.12 s |
| 100K × 128, 20% missing, synthetic | 100 | 1.78 s | 5.20 s |
| 25K × 4,096 dense, synthetic | 100 | 44.82 s | 62.44 s |
| Covertype: 464,809 × 54 | 500 | 72.54 s | 142.96 s |
| HIGGS: 10.5M × 28 | 1,000 | 389.05 s | 308.96 s |

The categorical synthetic workload uses 256 values per feature and random
per-category effects on 16 informative columns. Dense and missing-value
synthetic cases use 64 informative columns; missing entries are introduced
after generating labels. Covertype maps classes 1–7 to 0–6 and uses a stratified
80/20 split (seed 1701), with 116,203 held-out rows. Its seven-class, 500-round
configuration differs from the historical eight-class/60–20–20 replay below.
HIGGS uses all 10.5M training rows, 500K held-out rows and the full 1,000 rounds.

### Repeats and before/after comparison

| Workload / variant | Training seconds, in repeat order |
| --- | --- |
| 100K × 32 categorical, synthetic / CPU | 1.427, 1.479, 1.520 |
| 100K × 32 categorical, synthetic / previous MPS | 9.194, 8.889, 9.457 |
| 100K × 32 categorical, synthetic / new MPS | 5.118, 5.098, 5.170 |
| 100K × 128, 20% missing, synthetic / CPU | 1.730, 1.779, 1.860 |
| 100K × 128, 20% missing, synthetic / previous MPS | 5.333, 5.159, 5.162 |
| 100K × 128, 20% missing, synthetic / new MPS | 5.295, 5.203, 5.126 |
| 25K × 4,096 dense, synthetic / CPU | 30.780, 44.815, 46.327 |
| 25K × 4,096 dense, synthetic / previous MPS | 81.556, 50.450, 58.732 |
| 25K × 4,096 dense, synthetic / new MPS | 79.061, 62.437, 59.031 |
| Covertype: 464,809 × 54 / CPU | 67.909, 80.437, 72.545 |
| Covertype: 464,809 × 54 / previous MPS (one correctness reference) | 141.388 |
| Covertype: 464,809 × 54 / new MPS | 137.886, 142.957, 144.295 |
| HIGGS: 10.5M × 28 / CPU | 384.968, 395.181, 389.052 |
| HIGGS: 10.5M × 28 / previous MPS (one correctness reference) | 305.300 |
| HIGGS: 10.5M × 28 / new MPS | 308.965, 308.270, 321.067 |

The three short workloads rotate CPU/baseline/candidate order across repeats.
Covertype and HIGGS alternate three CPU/candidate pairs, with one additional
baseline MPS reference each. Those single long baseline timings do not support
a robust before/after speedup claim.

Categorical training improves 1.80× by median (9.194 → 5.118 seconds), with all
three pairs improving. Missing-value training is effectively flat
(5.162 → 5.203 seconds). The wide median is slower (58.732 → 62.437 seconds);
its runs vary substantially and system swap grows during the experiment. CPU
is faster than new MPS in every pair of these three synthetic cases. Preserve
all repeats rather than selecting favorable values or combining them with
earlier sessions. Other applications were not controlled, and no local builds,
correctness tests or other benchmarks from this task overlapped these timings.

Including matrix construction, CPU/new-MPS median seconds are:

- 100K × 32 categorical, synthetic: 1.485 / 5.124.
- 100K × 128, 20% missing, synthetic: 1.807 / 5.234.
- 25K × 4,096 dense, synthetic: 45.083 / 62.783.
- Covertype: 464,809 × 54: 72.645 / 143.031.
- HIGGS: 10.5M × 28: 391.032 / 310.989.

### Held-out quality

These compare independently trained CPU and MPS models. Before/after MPS
outputs are exact, but independent CPU/MPS training can produce different trees.
The wide accuracy difference below is also present in the baseline MPS model;
its numerical cause has not been isolated. Test tolerances were not relaxed.

| Workload | CPU accuracy | MPS accuracy | CPU log loss | MPS log loss |
| --- | ---: | ---: | ---: | ---: |
| 100K × 32 categorical, synthetic | 0.7178 | 0.7178 | 0.6190 | 0.6190 |
| 100K × 128, 20% missing, synthetic | 0.6989 | 0.6989 | 0.5729 | 0.5748 |
| 25K × 4,096 dense, synthetic | 0.7036 | 0.6848 | 0.5963 | 0.5873 |
| Covertype: 464,809 × 54 | 0.9463 | 0.9461 | 0.1510 | 0.1512 |
| HIGGS: 10.5M × 28 | 0.7612 | 0.7617 | 0.4828 | 0.4827 |

### Where to profile next

Median recorded GPU-family durations across the three candidate runs:

| Workload | Row partition | Histogram | Objective | Split |
| --- | ---: | ---: | ---: | ---: |
| Covertype: 464,809 × 54 | 11.72 s | 24.01 s | 24.73 s | 3.86 s |
| HIGGS: 10.5M × 28 | 69.51 s | 66.91 s | 12.80 s | 2.29 s |

These diagnostic counters can overlap; they are not additive and do not isolate
CPU overhead. They point to row partitioning/histograms on HIGGS and
objective/histogram work on Covertype as candidates for the next focused profile.
Wide-workload memory pressure needs separate investigation before attributing
its timing variation to any kernel.

### What changed and what was rejected

- FP32 ordering now compares bit patterns directly, preserving NaNs, signed
  zeros, subnormals, infinities and strict ties. Gain calculations still use
  their original precision. Isolated partitioned-category scoring improves
  1.75–2.30×; numeric/one-hot microbenchmarks are mixed, including a 9.1%
  regression for one numeric-with-missing shape. The whole-training missing
  control above is effectively flat.
- Packed input initializes only alignment padding before copying its payload.
  A separate CPU sample of the accepted baseline observed 3.1% of main-thread
  samples in redundant staging zero fill. Removing it preserves ownership,
  offsets, counters and command boundaries. No isolated whole-training gain
  is attributed to this change.
- A direct-store histogram prototype was rejected: all 1,024 outputs matched
  CPU exactly, but 56 of 64 timings regressed.
- Cross-tree scratch retention was deferred after the current profile showed
  only 1.1% of main-thread samples in split allocations, with a potentially
  longer high-water memory lifetime. It is absent from the accepted code.

All 26 MPS benchmark runs reproduce the corresponding baseline models, margins
and metric histories exactly, including 1,000-round HIGGS and 500-round
Covertype. Every MPS worker also checks same-model CPU prediction exactly.
Independent CPU/MPS training differences and held-out quality are recorded;
no NVIDIA kernels were executed. New direct comparisons cover 1,400,784 FP32
pairs; scalar/vector split, input layout, categories and Python workflow checks
also pass. The full suite precedes this optimization; its subsequent checks
are focused acceptance.

Peak RSS is process-wide. Current end-of-worker RSS is sampled after prediction,
model saving and CPU validation, not immediately after training, and does not
measure the complete Metal physical footprint. Swap is system-wide. All timings
pass power-source and wall/monotonic-clock checks; detailed memory and power
observations are retained in the record.

### Reproduce the workloads

Prepare the two UCI caches once. Use new output directories for each run.

```sh
.venv/bin/python scripts/prepare_published_data.py higgs covtype
.venv/bin/python scripts/profile_training.py --case categorical --rows 100000 \
  --features 32 --categories 256 --informative-features 16 --rounds 100 \
  --threads 8 --repeats 3 --output results/profile-categorical
.venv/bin/python scripts/profile_training.py --case wide --rows 100000 \
  --features 128 --informative-features 64 --missing-fraction 0.2 --rounds 100 \
  --threads 8 --repeats 3 --output results/profile-missing
.venv/bin/python scripts/profile_training.py --case wide --rows 25000 \
  --features 4096 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide-4096
.venv/bin/python scripts/profile_training.py --case covtype --rounds 500 \
  --threads 8 --repeats 3 --output results/profile-covtype
.venv/bin/python scripts/profile_training.py --case higgs --rounds 1000 \
  --threads 8 --repeats 3 --output results/profile-higgs-1000
```

Before/after worker commands use isolated `--package-dir` paths and are listed
in the acceptance record. The historical measurements below retain their own
revisions and workload definitions.

## Command batching and histogram dispatch milestone (28908e1)

The 2026-11-06 comparison starts from `9919a31`, the previous accepted MPS build.
Three subagents developed and cross-reviewed command batching, wider eligibility
for existing shared histogram kernels, and cheaper conservative split bounds.
The candidate's exact source/library hashes and all commands are in the
[acceptance record](../validation/parallel-tuning-m3.json).

All runs use this M3's eight CPU cores and 10-core GPU, 16 GB RAM, 100 rounds,
depth 6 and 256 bins. Each variant runs in a fresh process with a warmed runtime;
CPU/original-MPS/candidate-MPS order rotates across three repeats. Training
includes quantization and held-out evaluation each round; generation/loading,
matrix construction and final prediction are separate. Synthetic data has 64
informative columns and no column sampling. HIGGS uses all 10.5M training and
500K test rows, but 100 rather than the historical publication's 1,000 rounds.

| Workload | CPU median | Previous MPS median | New MPS median | Previous / new MPS |
| --- | ---: | ---: | ---: | ---: |
| 100K × 1,024 synthetic | 18.385 s | 16.155 s | 15.173 s | 1.06× |
| 25K × 4,096 synthetic | 35.605 s | 45.637 s | 31.405 s | 1.45× |
| 10.5M × 28 HIGGS | 33.690 s | 33.131 s | 31.244 s | 1.06× |

All six synthetic before/after pairs improve. At 4,096 columns the median of
the **paired** improvement ratios is 1.25×, below the 1.45× ratio of medians
because timings vary. One HIGGS pair regresses by 0.9%; its median improvement is
modest. CPU wins two of three 4,096-column pairs despite having a higher median.
New MPS is faster than CPU in all three pairs for each other workload. These
observations do not establish a general advantage over CPU or CUDA.

| Workload / variant | Training runs in repeat order, seconds |
| --- | --- |
| 1,024 columns CPU | 17.724, 18.385, 18.398 |
| 1,024 columns previous MPS | 17.515, 15.943, 16.155 |
| 1,024 columns new MPS | 15.522, 15.173, 15.021 |
| 4,096 columns CPU | 27.045, 35.605, 36.500 |
| 4,096 columns previous MPS | 38.307, 47.251, 45.637 |
| 4,096 columns new MPS | 30.737, 31.405, 40.664 |
| HIGGS CPU | 33.690, 32.938, 33.929 |
| HIGGS previous MPS | 33.173, 33.131, 30.999 |
| HIGGS new MPS | 31.222, 31.244, 31.277 |

Including matrix setup, median CPU/previous-MPS/new-MPS times are
18.581 / 16.356 / 15.392 seconds, 35.801 / 45.856 / 31.612 seconds, and
35.041 / 33.882 / 32.495 seconds, respectively.

The changes address three specific costs:

- Resident split evaluation reduces command completions from five to two and
  reuses encoders across nodes. Separate integer/precise parent staging protects
  queued inputs; public calls remain synchronous. A separate baseline CPU sample
  found 3,329 of 3,469 main-thread samples waiting for Metal completion. This
  includes necessary GPU work and is not an estimate of removable overhead.
- Dense compressed pages with at least 1,024 features can use existing shared
  histogram kernels at 4,096 selected rows, retaining the prior 16,384-row rule
  and all capacity guards. No histogram arithmetic changed. Median histogram
  GPU time falls from 3.796 to 2.786 seconds at 1,024 columns and 10.531 to 6.094
  seconds at 4,096 columns. HIGGS's histogram time changes from 6.557 to 6.816
  seconds; it does not use the new wide-page condition.
- Split screening constructs widened FP32 intervals directly from integer sums,
  retaining binary64 scoring for survivors and unbounded fallback when validity
  is uncertain. The standalone dense kernel improves 1.11–1.21×. Missing-data
  microbenchmarks are mostly unchanged, with a 7% regression in one shape.

**Split-counter boundaries changed.** After allocation readback, the baseline
sums four command intervals; the candidate measures one contiguous span through
task/gather, scoring, finalization and winner extraction. This includes gaps
between encoders and scheduling effects. The first readback and standalone
operations remain separately timed. At 4,096 columns the counter rises from
12.296 to 13.881 seconds; this does not establish a scoring-kernel regression,
and the entire rise cannot be attributed to instrumentation alone. Use whole
training for the combined comparison and unchanged-boundary standalone kernel
timings for the bound optimization. GPU families can overlap.

All 18 MPS models, prediction arrays and per-round metric histories match
before/after and reproduce the prior saved outputs exactly. All nine CPU controls
also reproduce their saved outputs. Each MPS model's CPU prediction matches
exactly; independently trained CPU/MPS models remain different. Eight native
groups, 300 Python tests and three installed-package workflows passed. New
checks include 1,044,483 bound cases, queued staging ownership and wide compressed
histograms. No precision tolerance changed and no NVIDIA execution was performed.

This task ran no concurrent builds/tests/benchmarks during timing. Unrelated
sessions and applications were not controlled. All runs used AC power and passed
clock checks; maximum wall/monotonic discrepancy was 0.0091 seconds. System-wide
swap was substantial, and peak process RSS ranged from 2.35 to 6.58 GB. All raw
timings, quality differences, memory/power observations, source hashes and
microbenchmark commands are retained in the acceptance record.

Reproduce current-build profiles in new output directories. For before/after
comparisons, use `--package-dir` with isolated packages; the record lists the
exact rotated worker commands used above.

```sh
.venv/bin/python scripts/profile_training.py --case wide --rows 100000 \
  --features 1024 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide-1024
.venv/bin/python scripts/profile_training.py --case wide --rows 25000 \
  --features 4096 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide-4096
.venv/bin/python scripts/profile_training.py --case higgs --rounds 100 \
  --threads 8 --repeats 3 --output results/profile-higgs
```

## Published CUDA workloads and earlier milestones

The replays below test workloads on which CUDA demonstrated substantial gains,
and identify where this backend fell short at each recorded revision. A historical
GPU/CPU ratio is not a prediction for an M3. Each local comparison uses CPU `hist`
and the corresponding Metal implementation on the same machine.

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

## Reusing the checked sigmoid implementation

The Python usability phase identified an existing implementation that could
reduce the measured logistic-objective cost. Ranking already uses a two-float
exponential expansion with explicit FP32 boundaries and software division.
The objective sigmoid now reuses that helper for non-NaN inputs; NaNs retain
the general path. This adds no new pipeline and changes no test tolerances.

Five native groups and 112 Python tests passed, with one expected skip. The
existing 2,996,040-input sigmoid corpus now also checks the objective entry point;
every result matches the independent binary64 reference. The objective API
checks passed 14,305,782 exact fields, including subnormal/nonfinite cases.

Three pairs used the full 10.5M/500K HIGGS split and **100 rounds**, separate
baseline/candidate package and library snapshots, fresh worker processes and
randomized pair order. Every saved model and prediction array is bit-identical
across all six runs.

| Pair | Baseline training (s) | Candidate training (s) |
| --- | ---: | ---: |
| 1 | 46.438 | 33.698 |
| 2 | 52.546 | 48.602 |
| 3 | 54.580 | 60.761 |
| Median | 52.546 | 48.602 |

Median objective GPU time fell from **7.781 to 1.124 seconds (6.92x)** and improved
in every pair. This component improvement supports retaining the small change.
The total-training median ratio is **1.081x**, but one candidate run was slower
and timing varied substantially. All six runs used battery power; system-wide
swap was approximately 22–24 GB and user applications remained open. These
conditions do not establish a reliable general end-to-end speedup. No build or
test job ran concurrently with the timed workers.

This is an MPS before/after experiment, not a CPU/NVIDIA speedup measurement or
a rerun of the published 1,000-round workload. The existing benchmark worker
command above applies to the two source/library snapshots. The
[result record](../validation/sigmoid-reuse-m3.json) retains exact source/library
hashes, every timing, model hashes, power, memory and clock observations.
The final optimized wheel also passed a [fresh installation check](../validation/python-install-m3.json).

## Reusing gradients across dense histogram columns

On 2026-11-05, the new `scripts/profile_training.py` measured HIGGS and a wide
synthetic classification dataset after the full correctness baseline passed.
The resulting optimization processes four adjacent dense ELLPACK features per
workgroup, sharing row IDs and quantized gradients. Integer sums, quantization,
tree algorithms and test tolerances are unchanged. Binary columns and layouts
that cannot fit four histograms in threadgroup memory retain the existing path.

Three repeats per CPU/baseline-MPS/optimized-MPS variant used fresh processes,
isolated package/library copies, eight CPU threads and **100 rounds**. Variant
order rotated between repeats; all workers ran sequentially on AC power.
Training includes quantization and per-round held-out evaluation, after warming
the runtime. Data loading, DMatrix construction and final prediction are separate.

| Workload | CPU median | MPS before | MPS after | Histogram GPU before / after |
| --- | ---: | ---: | ---: | ---: |
| Wide: 1M training rows × 128 features | 9.919 s | 16.691 s | 11.481 s | 3.992 / 1.954 s |
| HIGGS: 10.5M training rows × 28 features | 36.531 s | 40.792 s | 37.894 s | 10.950 / 6.740 s |

Histogram GPU time improved in every pair, with ratios of median times of
**2.04× wide / 1.62× HIGGS**. Ratios of total-training medians are **1.45× / 1.08×**; medians of
the paired training ratios are **1.21× / 1.06×**. These different summaries reflect
substantial timing variability. CPU remains faster on both workloads.

| Workload / variant | Training runs in repeat order, seconds |
| --- | --- |
| Wide CPU | 8.584, 9.919, 14.988 |
| Wide MPS before | 13.880, 16.691, 19.731 |
| Wide MPS after | 11.481, 11.432, 18.193 |
| HIGGS CPU | 38.576, 35.601, 36.531 |
| HIGGS MPS before | 41.592, 39.570, 40.792 |
| HIGGS MPS after | 37.894, 37.248, 38.643 |

All six MPS models and prediction arrays per dataset are bit-identical. Each
MPS model also produces identical margins with the CPU predictor. Independently
trained CPU/MPS models differ; the report retains their held-out quality and
margin differences. Four native groups and 89 Python tests passed, with one
expected skip, followed by the final compressed-consumer assertions and three
installed-package workflows. The full 106-group/4,001-test baseline predates
this kernel change.

User applications remained open, with roughly 21–24 GB of system-wide swap;
this is not the benchmark's own swap allocation. No assistant build/test job
overlapped timing. The result supports this optimization, not a general CPU
speedup, a cold-start improvement, or a repeat of the published 1,000-round run.
HIGGS partition GPU time remains about 7.7 seconds and sketch time about 5 seconds;
these are the next costs to investigate. Counter families can overlap.

The [acceptance record](../validation/histogram-tiling-m3.json) retains all 18
timings, source/library hashes, memory/power observations, output comparisons,
and the earlier exploratory timings. To profile the current build, use a new
output directory for each invocation:

```sh
.venv/bin/python scripts/profile_training.py --case wide --rows 1000000 \
  --features 128 --rounds 100 --threads 8 --repeats 3 --output results/profile-wide
.venv/bin/python scripts/profile_training.py --case higgs --rounds 100 \
  --threads 8 --repeats 3 --output results/profile-higgs
```

For before/after comparisons, build separate revisions and point `--package-dir`
at their respective Python packages. The record specifies the exact worker order
and command used here; retain fresh workers and the CPU controls.

## Removing partition copies and unnecessary feature-sort passes

The next profiling pass starts at `581f4c1`. Dense partition diagnostics showed
classification dominated prefix scanning. Complete ordered partitions now scatter
into the existing spare row buffer and swap ownership, avoiding an intermediate
row-ID copy. Partial or unordered ranges retain the original path. A separate
setup change limits stable feature-key radix sorting to the bits needed by the
feature count. For 28 or 128 features, this removes six of eight radix passes;
the value sort, weighted tie order and sketch arithmetic are unchanged.

Five randomized alternating kernel measurements check every result against CPU
references. The median sum of partition-stage GPU times improves by **1.07–1.22×** across
1M/10.5M rows and 1/8/32 nodes. For 4,194,304 feature-key records, full-width
versus bounded sorting gives:

| Features | Full 32-bit median | Bounded median | Stage speedup |
| --- | ---: | ---: | ---: |
| 28 | 19.745 ms | 4.827 ms | 4.09× |
| 128 | 18.382 ms | 4.496 ms | 4.09× |
| 257 | 18.391 ms | 6.873 ms | 2.68× |
| 4097 | 19.021 ms | 10.131 ms | 1.88× |

These measure selected stages, not whole training. In the combined 100-round
training experiment, median complete-sketch GPU time changes from **2.730 to
2.254 seconds** on wide data and **6.187 to 4.456 seconds** on HIGGS. Partition
GPU time changes from **0.972 to 0.888 seconds** and **8.724 to 7.578 seconds**.
Both targeted components improve in every pair.

Total training results are mixed. Each experiment uses three CPU/baseline-MPS/
candidate-MPS repeats per dataset, with rotated order and fresh isolated packages.
The partition-only experiment has median MPS times of **11.580 / 11.647 seconds**
on wide data and **47.211 / 42.185 seconds** on HIGGS. The combined experiment is:

| Workload / variant | Training runs in repeat order, seconds |
| --- | --- |
| Wide CPU | 19.320, 18.622, 18.498 |
| Wide MPS baseline | 22.088, 17.779, 17.421 |
| Wide MPS combined | 23.625, 20.753, 17.942 |
| HIGGS CPU | 58.595, 49.723, 126.096 |
| HIGGS MPS baseline | 47.543, 90.534, 103.801 |
| HIGGS MPS combined | 45.165, 48.282, 71.456 |

The combined wide candidate is slower overall in all three pairs. HIGGS baseline
timing varies by more than twofold, with large setup variation. The apparent
HIGGS median speedup is **inconclusive as a general end-to-end improvement**.
All runs used AC power and passed clock checks; user applications remained open
and system-wide swap was substantial. This task ran no concurrent builds, tests
or other benchmarks; unrelated sessions and applications were not controlled.
The small changes are retained for their consistent targeted
GPU improvements and exact outputs, without claiming reliable whole-training gains.

Four native groups and 171 Python tests passed on the combined build, followed
by three installed-package workflows. Across both experiments, all 12 MPS models
and margin arrays **per dataset** match exactly. CPU prediction of each MPS model
also agrees exactly; independent CPU training still uses a different sketch.
No tolerance or precision was changed.

A separate five-second CPU sampling diagnostic, excluded from timing comparisons,
finds **2,671 of 3,594 main-thread samples (74%)** waiting for Metal command
completion, particularly in split evaluation and prediction. This includes actual
GPU execution and scheduling, not just avoidable overhead. The sampled wide worker
reports a **6.9 GB peak physical footprint**, which differs from process RSS.
Command dependencies and temporary-buffer lifetimes are the next investigations.

The [complete record](../validation/partition-sketch-tuning-m3.json) retains all
36 training timings, all source/library snapshots, memory/power observations,
kernel timings, correctness checks and the sampling evidence. Large-workload
commands in the previous section apply. Reproduce the stage diagnostics with:

```sh
mkdir -p results
git show 581f4c1:xgboost/src/common/metal_histogram.metal > results/partition-baseline.metal
clang++ -O2 -std=c++17 -fobjc-arc scripts/metal_partition_benchmark.mm \
  -framework Metal -framework Foundation -o results/partition-benchmark
results/partition-benchmark xgboost/src/common/metal_histogram.metal \
  results/partition-baseline.metal --swap > results/partition-stages.json
clang++ -O2 -std=c++17 -fobjc-arc scripts/metal_feature_sort_benchmark.mm \
  -framework Metal -framework Foundation -o results/feature-sort-benchmark
results/feature-sort-benchmark xgboost/src/common/metal_histogram.metal \
  > results/feature-sort-stages.json
```

## High-dimensional synthetic workloads

The 2026-11-06 experiment uses the unchanged engine at `59f0451` and adds an
optional `--informative-features` flag to the profiler. These are new synthetic
workloads, not published CUDA replays. Dense FP32 normal features have 64 signal
columns spread across the full width, an additive linear/quadratic target and
stochastic binary labels. No column sampling is used. Both cases have 102.4M
training feature values (409.6 MB before XGBoost allocations); the wider case
has fewer rows to bound memory. This is not a constant-row scaling study.

Each case uses 100 rounds, depth 6, 256 bins, eight CPU threads and three
alternating CPU/MPS pairs in fresh processes. The runtime is warmed; training
includes quantization and held-out evaluation every round. Data generation,
matrix construction and final prediction are excluded. Held-out sets contain
10,000 and 2,500 rows, respectively. Dataset hashes and shapes agree across
CPU/MPS workers. The profiler's default three-feature target remains unchanged.

| Training shape | CPU median | MPS median | MPS / CPU time | Split GPU median | Histogram GPU median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100,000 × 1,024 | 18.762 s | 27.353 s | 1.46× | 14.439 s | 3.427 s |
| 25,000 × 4,096 | 45.624 s | 110.075 s | 2.41× | 69.203 s | 12.063 s |

CPU is faster in all six matched pairs. Including matrix setup gives median
CPU/MPS times of 19.002 / 27.566 seconds and 45.924 / 110.335 seconds.
Individual training times in repeat order are:

| Workload | CPU runs, seconds | MPS runs, seconds |
| --- | --- | --- |
| 1,024 columns | 14.097, 18.762, 28.318 | 27.353, 25.727, 34.491 |
| 4,096 columns | 44.268, 45.624, 51.388 | 107.050, 110.075, 116.575 |

Split evaluation is the largest measured GPU cost. The quantized split gain
path uses software binary64 arithmetic; its contribution, scan/reduction costs
and dispatch geometry need to be isolated before an optimization is selected.
The measured GPU time means launch overhead alone does not explain the slowdown.
Family counters can overlap and must not be summed as independent work.

Every MPS model produces bit-identical margins with the CPU predictor, and all
three saved models/margin arrays match within each device and dataset. All
predictions are finite. Independently trained CPU/MPS models are **not numerically
equivalent**: 9,999 of 10,000 and all 2,500 margins fail `atol=rtol=2e-5`.

| Columns | CPU / MPS accuracy | CPU / MPS log loss | Class disagreement |
| --- | --- | --- | ---: |
| 1,024 | 0.7311 / 0.7304 | 0.530385 / 0.531930 | 19.57% |
| 4,096 | 0.7036 / 0.6848 | 0.596307 / 0.587342 | 27.24% |

Different CPU/CUDA-style sketches can yield different trained models, but these
datasets have no common-cut diagnosis. Similar aggregate quality does not prove
training equivalence. No engine arithmetic, precision or tolerances changed.

All runs used AC power and passed clock checks. This task ran no concurrent
builds, tests or other benchmarks; unrelated sessions and applications were
not controlled. Recorded system-wide swap use was about 13,548–17,834 MB and
peak process RSS ranged from 2.26 to 4.59 GB. Timing drift limits the exact ratios;
these observations do not isolate hardware limits from implementation costs.

The [recorded evidence](../validation/high-dimensional-m3.json) includes all
12 workers, source/library hashes, dataset descriptions, output checks, quality
differences and hashes of local logs/models. A small CPU/MPS smoke run also passed.
Reproduce with new output directories:

```sh
.venv/bin/python scripts/profile_training.py --case wide --rows 100000 \
  --features 1024 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide-1024
.venv/bin/python scripts/profile_training.py --case wide --rows 25000 \
  --features 4096 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide-4096
```

## Quantized split screening and identical-gain reuse

The next change starts from `fd4059a` and addresses the high-dimensional split
bottleneck above. With zero integer missing sums, both missing directions have
identical child sums, so their full-precision gain is computed once. Dense
numeric features with at least 32 bins can also reuse the existing conservative
gain intervals when L1 regularization, delta-step limits, monotone constraints
and leaf-weight bounds are absent. A candidate is skipped only when its upper
bound is strictly below another candidate's lower bound. Ties and potentially
winning candidates retain the original binary64 evaluation; unsupported ranges
receive unbounded intervals. Numeric upper bounds reuse category-weight scratch.

The kernel benchmark separates identical-gain reuse from the combined change,
with five randomized paired measurements, an untimed warmup and exact checks
against the independent CPU implementation of CUDA source formulas:

| Dense feature tasks, 256 bins each | Original | Reuse only | Combined | Combined speedup |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.437 ms | 0.232 ms | 0.227 ms | 1.92× |
| 1,024 | 2.556 ms | 1.336 ms | 0.624 ms | 4.09× |
| 8,192 | 18.153 ms | 9.493 ms | 3.539 ms | 5.13× |

The same benchmark with nonzero missing sums retains the original calculation
and measures about 1–3% slower for the combined kernel in this sample. Its raw
timings and the earlier uncached-bound experiment are retained in the evidence.
These are selected kernel costs, not whole-training speedups.

For full training, the two datasets above are unchanged. Three repeats rotate
CPU/original-MPS/optimized-MPS order, each in a fresh process and using isolated
original/candidate packages. All use 100 rounds, eight CPU threads, 64 informative
columns and no column sampling. Median training times are:

| Training shape | CPU | MPS original | MPS optimized | MPS improvement |
| --- | ---: | ---: | ---: | ---: |
| 100,000 × 1,024 | 19.989 s | 34.371 s | 18.711 s | 1.84× |
| 25,000 × 4,096 | 39.006 s | 115.309 s | 55.973 s | 2.06× |

Split GPU medians fall from **17.631 to 3.413 seconds** and **67.894 to 12.823
seconds**: improvements of **5.17× / 5.29×**. All six before/after training pairs
improve, but CPU remains faster on the 4,096-column workload. At 1,024 columns,
the optimized MPS median is only 1.07× faster than CPU, insufficient to claim a
general CPU advantage. Individual training times in repeat order are:

| Workload / variant | Training runs, seconds |
| --- | --- |
| 1,024 columns CPU | 18.829, 19.989, 22.261 |
| 1,024 columns MPS original | 30.496, 34.930, 34.371 |
| 1,024 columns MPS optimized | 18.388, 18.711, 19.370 |
| 4,096 columns CPU | 30.465, 44.839, 39.006 |
| 4,096 columns MPS original | 122.996, 115.309, 107.097 |
| 4,096 columns MPS optimized | 61.756, 55.973, 51.586 |

All 12 MPS saved models, prediction arrays and per-round metric histories match
before/after and reproduce the original high-dimensional experiment exactly.
Every MPS worker also checks CPU prediction of its model exactly. The six CPU
controls reproduce their original outputs too. Independently trained CPU/MPS
models still differ by the amounts recorded above; no tolerance or precision
was changed and no NVIDIA execution was performed.

Five native groups and 178 Python tests passed, including the expanded 1,992-task
scalar split oracle and 1,000,300 gain-interval checks. Three workflows passed
against the newly installed wheel outside the checkout. These are focused checks;
the historical full-suite run predates this optimization.

This task ran no concurrent builds, tests or other benchmarks during timing.
Unrelated sessions and applications were not controlled. All runs used AC power
and passed clock checks; substantial system-wide swap and timing variation remain
limitations. GPU counter families can overlap. The next profiling target is the
remaining split-stage preparation/submission/waits and histogram work, rather
than treating the GPU-time reduction as an equivalent whole-training reduction.

The [complete evidence](../validation/quantized-split-tuning-m3.json) includes
source/library hashes, all 18 training runs and their commands, kernel timings,
memory/power observations, output comparisons and installation checks. The
profiling commands above accept `--package-dir` for isolated revisions. Reproduce
the direct kernel comparison with:

```sh
git show fd4059a:xgboost/src/common/metal_histogram.metal > results/split-baseline.metal
clang++ -O2 -std=c++17 -ffp-contract=off -fobjc-arc \
  scripts/metal_quantized_split_benchmark.mm -framework Metal -framework Foundation \
  -o results/split-benchmark
results/split-benchmark xgboost/src/common/metal_histogram.metal \
  results/split-baseline.metal > results/split-kernel-timings.json
```
