# Performance benchmarks

Measurements below use an Apple M3 with a 10-core GPU, eight CPU cores and
16 GB unified memory. Ratios above 1 favor MPS. They describe these workloads,
not a general GPU advantage; unrelated applications, temperature and system-wide
paging were not controlled. Do not combine ratios from different sessions.

## Wide-data optimization (M3)

Backend `614f15a`: **45 fresh-process runs**, three each of CPU, previous MPS
(`61b2042`, also the backend at `f12489d`) and current MPS per workload, with
rotated execution order. All use 100 rounds, depth 6, 256 bins and eight threads.
Training includes quantization and held-out evaluation after runtime warmup;
loading, matrix construction and final prediction are separate. No project builds
or tests overlapped. The machine was busy with other applications and substantial
paging, so these are local observations, not controlled hardware comparisons.
[Every timing, command, source/library hash and check](../validation/wide-quality-and-performance-m3.json).

| Training data | CPU median | MPS median | CPU / MPS |
| --- | ---: | ---: | ---: |
| 25K × 4,096, dense | 37.92 s | 27.01 s | 1.40× |
| 125K × 1,024, dense | 23.57 s | 14.17 s | 1.66× |
| 25K × 1,024, 20% missing | 14.73 s | 24.60 s | 0.60× |
| 1M × 128, dense | 14.87 s | 12.55 s | 1.19× |
| HIGGS, 10.5M × 28 | 39.66 s | 32.33 s | 1.23× |

MPS uses conservative bounds to skip features that cannot win, and reads scalar
histograms directly from their GPU cache. In this paired comparison, the
4,096-column MPS median fell from **42.68 to 27.01 s (36.7%)**; the
1,024-column dense case fell from **17.85 to 14.17 s (20.6%)**.

The narrow controls did not improve: 128-column dense MPS medians were
**12.48 → 12.55 s**; HIGGS was **29.96 → 32.33 s (7.9% slower by median)**.
Two HIGGS pairs improved and one regressed; all repeats are retained, and no
HIGGS speed improvement is claimed from this busy-machine session.

All 30 MPS runs reproduce the previous model bytes, margins and every-round
metrics exactly. CPU inference of each MPS model is also exact.

Synthetic cases use a fixed seed, noisy logistic targets and 16 informative
columns spread across the matrix, or 64 for the 4,096-column case. Missing values
are added after generating labels. HIGGS uses the first 10.5M rows for training
and the last 500K for testing, `binary:logitraw`, `eval_metric="error"`, learning
rate 0.3 and base score 0.5. Independently trained CPU/MPS models differ:

| Workload | CPU accuracy / log loss | MPS accuracy / log loss |
| --- | ---: | ---: |
| 25K × 4,096, dense | 0.7036 / 0.5963 | 0.6848 / 0.5873 |
| 125K × 1,024, dense | 0.7494 / 0.4995 | 0.7554 / 0.4928 |
| 25K × 1,024, 20% missing | 0.6956 / 0.6023 | 0.6972 / 0.6024 |
| 1M × 128, dense | 0.7710 / 0.4735 | 0.7706 / 0.4738 |
| HIGGS, 10.5M × 28 | 0.7415 / 0.5113 | 0.7422 / 0.5110 |

Earlier three-repeat measurements on `61b2042` remain relevant for narrower
missing-data cases: **1M × 128, 20% missing: CPU 10.13 s / MPS 12.15 s**;
**80% missing: CPU 5.07 s / MPS 7.58 s**. Those shapes were not retimed here.
[Earlier missing-data evidence](../validation/missing-and-memory-m3.json).

### Wide-data quality diagnosis

For the 25K × 4,096 workload above, the first CPU/MPS difference is already at
tree 0's root: feature 585, threshold −0.3399919 versus −0.33969516. Replaying
100 rounds with identical cuts in both directions isolates the cause:

| Cuts shared by CPU and MPS | CPU accuracy / log loss | MPS accuracy / log loss | Maximum margin difference |
| --- | ---: | ---: | ---: |
| CPU cuts | 0.7036 / 0.5963 | 0.7036 / 0.5963 | 4.77e-7 |
| MPS cuts | 0.6848 / 0.5873 | 0.6848 / 0.5873 | 4.77e-7 |

All split features, thresholds and missing directions match across all 100 trees
with either shared cut set; only small leaf-weight rounding differences remain.
Every one of the **1,052,672 MPS cuts** matches the independent CUDA-source sketch
reference evaluated on CPU. This explains this workload's difference through
quantization, without substituting CPU cuts in production. It is not an NVIDIA
execution comparison. Accuracy and log loss favor different independently
trained models here. [Diagnosis, cut-reference hashes and comparisons](../validation/wide-quality-and-performance-m3.json).

### Physical memory

Separate diagnostics use 25K × 4,096 data and **30 rounds**, with three fresh
processes per variant in alternating order. Physical peak is the Mach task
lifetime high-water mark. Metal allocation is sampled every 20 ms and may miss
short allocations; these are different measurements.

| Median peak | Previous MPS | Current MPS |
| --- | ---: | ---: |
| Physical memory | 6,318.5 MiB | 5,893.3 MiB |
| Sampled Metal allocation | 4,553.5 MiB | 4,298.0 MiB |

Physical peak fell **6.7%** and all six diagnostic outputs match exactly.
Direct scalar histogram reads remove a temporary gathered copy; they do not
modify the cached histogram. Individual repeats and system-memory observations
are retained in the evidence record.

## Synthetic workload survey (M3)

This earlier survey used backend `c33c56c`: 48 fresh processes, three repeats
per CPU/MPS cell, 100 rounds, depth 6, 256 bins and eight threads. It covers shapes
not repeated in the latest comparison. [All runs and quality measurements](../validation/synthetic-workloads-m3.json).

| Training data | CPU median | MPS median | CPU / MPS |
| --- | ---: | ---: | ---: |
| 100K × 32, dense | 0.485 s | 1.757 s | 0.28× |
| 1M × 32, dense | 2.610 s | 4.236 s | 0.62× |
| 4M × 32, dense | 11.438 s | 10.371 s | 1.10× |
| 1M × 128, dense | 10.258 s | 9.235 s | 1.11× |
| 250K × 512, dense | 19.125 s | 13.611 s | 1.41× |
| 125K × 1,024, dense | 21.423 s | 15.693 s | 1.37× |
| 1M × 128, 20% missing | 15.334 s | 22.760 s | 0.67× |
| 1M × 32, categorical (64 values) | 7.000 s | 5.821 s | 1.20× |

All cases have 16 informative features. Dense features are FP32 normal values;
categorical data uses random per-category effects. Test size is one tenth of
training size, capped at 100K. Same-device repeats and CPU inference of each MPS
model are exact; independently trained CPU/MPS models have different quality.

## Reproduce and profile

Run from the repository root with a built package. Use new output directories,
inspect existing results first, and serialize large jobs. The profiler records
all worker timings, data/library hashes, quality, memory and power observations.
GPU timing counters may overlap and must not be added as wall time.

```sh
.venv/bin/python scripts/prepare_published_data.py higgs covtype
.venv/bin/python scripts/profile_training.py --case higgs --rounds 100 \
  --threads 8 --repeats 3 --output results/profile-higgs-100
.venv/bin/python scripts/profile_training.py --case wide --rows 25000 \
  --features 1024 --informative-features 16 --missing-fraction 0.2 \
  --rounds 100 --threads 8 --repeats 3 --output results/profile-missing
.venv/bin/python scripts/profile_training.py --case wide --rows 25000 \
  --features 4096 --informative-features 64 --rounds 100 --threads 8 \
  --repeats 3 --output results/profile-wide
```

Change dimensions for the survey cases; use `--case categorical --categories 64`
for its categorical case. Commands use the current checkout, not a historical
binary. Large workloads belong in scripts, separate from correctness tests.
The isolated [sparse histogram profiler](../scripts/metal_sparse_histogram_benchmark.mm)
includes build/run instructions and checks every timed output against exact
CPU integer sums. Other `scripts/metal_*_benchmark.mm` tools isolate individual
operations; their explicit kernel choices do not represent production dispatch.

To isolate cut construction from numeric tree training, add `--cut-device cpu`
or `--cut-device mps` to a profiler command. This exports the shared cuts and
compares saved trees as well as predictions. It is a correctness diagnostic:
quantization moves into setup, so its training times are not comparable with the
ordinary benchmarks above.

## Published workloads and earlier evidence

The [2017 NVIDIA HIGGS benchmark](https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/)
used a Tesla P100. The [2018 GPU XGBoost paper](https://arxiv.org/html/1806.11248v1)
used eight V100s and 64 CPU cores; its ratios are not single-GPU speedups or
predictions for an M3. Local replays compare CPU `hist` with MPS `hist`.

```sh
.venv/bin/python scripts/benchmark_published.py --preset nvidia2017-higgs \
  --threads 8 --repeats 3 --output results/published-higgs.json
.venv/bin/python scripts/benchmark_published.py --preset gbm2018-covtype \
  --threads 8 --repeats 3 --output results/published-covtype.json
```

These presets preserve historical parameters: HIGGS runs 1,000 rounds;
Covertype runs 500 rounds with the original 60/20/20 split and eight output
classes, including unused class zero. Their strict independently trained
CPU/MPS margin comparison can report `prediction_equivalence_failed`; it is
separate from the profiler's same-model CPU inference check.

Datasets [HIGGS](https://doi.org/10.24432/C5V312) and
[Covertype](https://doi.org/10.24432/C50K5N) are distributed by UCI under CC BY 4.0.
Preparation saves source URLs and hashes in the ignored dataset cache.
Earlier [full-length HIGGS and memory/partition results](../validation/memory-and-partition-m3.json),
[broader regression/Covertype comparisons](../validation/regression-and-benchmarks-m3.json)
and other records in [validation/](../validation/) retain their measured revisions.
Superseded progress narratives remain in Git history.
