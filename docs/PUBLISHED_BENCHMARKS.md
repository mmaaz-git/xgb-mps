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
