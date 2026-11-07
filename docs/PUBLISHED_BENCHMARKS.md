# Performance benchmarks

These local measurements use an Apple M3 with a 10-core GPU, eight CPU cores and
16 GB unified memory. Ratios above 1 favor MPS. They describe these workloads,
not a general GPU advantage.

## November 7 rerun

All **14 training workloads** were rerun on backend `614f15a`: **84 accepted
fresh-process runs**, three CPU and three MPS runs per case, with alternating
device order. The same current library serves both devices. No project builds or
tests overlapped. Power checks before and after accepted runs report AC power.
One HIGGS run crossed a power change and was excluded, then repeated after AC
power returned; its record is retained. Unrelated applications, temperature and
system paging were not controlled. Driver utilization snapshots are retained;
they are not a calibrated measure of GPU occupancy.
[Every timing, command, data/library hash, quality result and system observation](../validation/benchmark-refresh-m3.json).

All cases use `hist`, depth 6, 256 bins and eight threads. The table below uses
**100 rounds**. Training includes quantization and per-round held-out evaluation
after runtime warmup; loading, matrix construction and final prediction are
separate. Matrix-plus-training medians are also in the evidence record.

| Training data | CPU median | MPS median | CPU / MPS |
| --- | ---: | ---: | ---: |
| 100K × 32, dense | 0.39 s | 1.58 s | 0.24× |
| 1M × 32, dense | 2.51 s | 4.20 s | 0.60× |
| 4M × 32, dense | 10.81 s | 9.11 s | 1.19× |
| 1M × 128, dense | 9.61 s | 9.03 s | 1.06× |
| 250K × 512, dense | 19.42 s | 13.63 s | 1.42× |
| 125K × 1,024, dense | 23.52 s | 15.23 s | 1.54× |
| 25K × 4,096, dense | 39.65 s | 29.45 s | 1.35× |
| 1M × 128, 20% missing | 15.64 s | 17.56 s | 0.89× |
| 1M × 128, 80% missing | 6.67 s | 10.08 s | 0.66× |
| 25K × 1,024, 20% missing | 15.88 s | 26.40 s | 0.60× |
| 1M × 32, categorical (64 values) | 9.11 s | 7.20 s | 1.26× |
| HIGGS, 10.5M × 28 | 38.33 s | 34.25 s | 1.12× |

Synthetic cases use seed 1701, noisy logistic targets and 16 informative columns
spread across the matrix, or 64 for the 4,096-column case. Dense features are FP32
normal values; categorical features use random per-category effects. Missing
values are added after generating labels. Test size is one tenth of training
size, capped at 100K. HIGGS uses the first 10.5M rows for training and the last
500K for testing, `binary:logitraw`, learning rate 0.3 and base score 0.5.

All same-device repeated model bytes, margins and every-round metrics are exact.
Fresh CPU inference of all **42 saved MPS models** is exact. The five workloads
from the prior wide-data comparison also reproduce that session's outputs
exactly. Independently trained CPU/MPS models differ; held-out accuracy, log loss, margin differences
and class disagreements are recorded for every case. These are compatibility
and repeatability checks, not evidence of CUDA execution.

## Full-length published workloads

These are fresh three-repeat medians from the same rerun, using the published
presets below. Their round counts and evaluation schedules differ from the
100-round table.

| Training data | CPU median | MPS median | CPU / MPS |
| --- | ---: | ---: | ---: |
| HIGGS, 10.5M × 28, 1,000 rounds | 331.19 s | 270.16 s | 1.23× |
| Covertype, 348,606 × 54, 500 rounds | 47.70 s | 123.90 s | 0.38× |

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
separate from fresh CPU inference of a saved MPS model. Both benchmark drivers
reload models for this check outside timed training.

Datasets [HIGGS](https://doi.org/10.24432/C5V312) and
[Covertype](https://doi.org/10.24432/C50K5N) are distributed by UCI under CC BY 4.0.
Preparation saves source URLs and hashes in the ignored dataset cache.

## Wide-data quality diagnosis

For the 25K × 4,096 workload, the first CPU/MPS difference is already at
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

## Physical memory

Six separate diagnostics use 25K × 4,096 data and **30 rounds**, with three fresh
processes per variant in alternating order. They compare previous MPS `61b2042`
with current MPS `614f15a`. Physical peak is the Mach task lifetime high-water
mark. Metal allocation is sampled every 20 ms and may miss short allocations;
these are different measurements.

| Median peak | Previous MPS | Current MPS |
| --- | ---: | ---: |
| Physical memory | 6,468.6 MiB | 6,007.8 MiB |
| Sampled Metal allocation | 4,553.5 MiB | 4,298.0 MiB |

Current physical peak is **7.1% lower** in this comparison. All six
diagnostic models, margins and metrics match exactly. Direct scalar histogram
reads remove a temporary gathered copy without modifying cached histograms.

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

## Earlier evidence

Earlier measurements retain their own revisions and conditions; do not combine
their timings with this rerun. The [wide-data optimization and quality diagnosis](../validation/wide-quality-and-performance-m3.json)
records paired before/after performance and shared-cut checks. The original
[synthetic survey](../validation/synthetic-workloads-m3.json),
[missing-value and memory comparison](../validation/missing-and-memory-m3.json),
[HIGGS memory/partition results](../validation/memory-and-partition-m3.json) and
[regression/Covertype comparisons](../validation/regression-and-benchmarks-m3.json)
remain available. Superseded progress narratives are in Git history.
