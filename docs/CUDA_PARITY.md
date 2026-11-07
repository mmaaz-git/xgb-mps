# Coverage and limits

This fork targets CUDA-accelerated XGBoost 3.4.2 behavior exposed through **Python**
and the shared engine. `device="mps"` selects custom Metal kernels. Python's Dask
support remains; R, Java/Scala and Spark bindings were removed.

The backend is usable and experimental. A bounded source review found no confirmed
missing major core CUDA training/prediction family, but full coverage and NVIDIA
numerical equivalence remain unproven. [Validation](VALIDATION.md) records exactly
which revisions and configurations were tested.

## Implemented paths

| Area | Metal implementation |
| --- | --- |
| Trees | Hist/approx, quantization, sampling, scalar/vector splits, partitioning, constraints and leaf fitting |
| Objectives and metrics | Regression, multiclass, ranking, AFT, adaptive objectives and metrics |
| Prediction | Batch/inplace prediction, leaf outputs, native outputs, exact SHAP contributions/interactions |
| Data | Ordinary/quantile matrices, dense/sparse/missing/categorical inputs, native tensors/columns, CSR packing, sketches, metadata and external-memory caches |
| Other | Coordinate-descent linear training, collectives, distributed trees and optional federated gRPC transport |

Host topology, serialization, scalar publication and finalizers that CUDA itself
runs on CPU remain host work. A kernel's existence alone does not establish that
every caller selects it or retains GPU storage; ownership boundaries are in the
[architecture guide](ARCHITECTURE.md).

## Platform and API boundaries

- Only Metal device ordinal **0** is supported. Producer completion is synchronous;
  there is no portable CUDA-style stream handshake for Metal DLPack.
- `tree_method="exact"`, conditional SHAP and distributed `gblinear` are unsupported,
  as in upstream CUDA. Approximate SHAP follows a CPU path. Linear GPU training
  requires `updater="coord_descent"`; multiple CSC blocks are unsupported.
- Device buffer lengths and kernel index widths impose limits. Boundary checks
  include 65,535/65,536/65,537 outputs and a 65,536-target vector tree; they do not
  establish arbitrary-width or arbitrarily large training support.
- Local execution evidence is from one Apple M3. Distributed/federated tests use
  workers on that machine, including TLS. Other Apple GPUs and multi-host operation
  remain unvalidated. CUDA CI is a compile/link check on a CPU runner.

## Numerical contracts

Vendored CUDA defines formulas, precision, RNG and update order. CPU is an oracle
where those agree. Source-derived oracles and same-model CPU predictions do not
establish NVIDIA execution equivalence, and independently trained CPU/MPS models
can differ.

| Area | Preserved behavior and limits |
| --- | --- |
| Histograms and splits | Exact integer sums remain authoritative. Split selection preserves precision and tie rules; software binary64 supplies operations absent from Metal hardware. |
| Weighted quantiles | Threshold comparison remains binary64. Different legal weighted-CDF associations can select different observations; casting the threshold to FP32 is not an acceptable workaround. |
| AFT | Ordinary CPU comparisons pass within their declared tolerance. Sensitive-tail cancellation amplifies math differences; the historical sensitive corpus retains 322 CPU-tolerance exceedances, with separate source/intermediate checks. No CPU repair or widened ordinary tolerance hides them. |
| Ranking | CUDA truncation, RNG and pair order differ from CPU. Fixtures cover the modern Philox/CCCL branch, not legacy streams. Metal replaces a CUDA shared-factor race with a deterministic GPU maximum; nonfinite exponent inputs use exponent 0/factor 1. These policies cannot predict a racing NVIDIA write. |
| AUC and ranking metrics | Prefix/reduction association, contraction and transcendental implementations can differ. Source PR cancellation pathologies are retained. Grouped ROC uses exact count accumulation in place of a pair scan; safe empty/singleton behavior and larger pair domains are explicit extensions. Near-zero IDCG denominators amplify error. |
| Sketches and SHAP | Prefix association, neighboring-rank repair order and cross-tree accumulation can differ across schedules. Alternative-order fixtures demonstrate possibilities, not observed NVIDIA output. |
| Arithmetic | Metal fast math remains disabled. Sampled math budgets, finite-domain tests and exact local fixtures are not universal accuracy guarantees. CUDA compiler, dependency and contraction settings must accompany future NVIDIA comparisons. |

The [numerical acceptance record](../validation/cuda-verify1-m3-run.json) maps all
17 recorded contracts to fixtures, declared limits and remaining comparisons.
[NVIDIA comparison instructions](VALIDATION.md#nvidia-comparison) describe the
separate hardware evidence requirement.

## Remaining work

1. **Optimize measured costs.** Missing-data histogram/split work and wide-data
   tree-construction storage remain substantial. Require paired whole-training
   controls, unchanged benchmark outputs and physical-memory measurements;
   [current benchmarks](PUBLISHED_BENCHMARKS.md) include CPU wins and regressions.
2. **Broaden correctness evidence.** Investigate the existing wide CPU/MPS training
   difference using common cuts; execute shared fixtures on NVIDIA and validate
   other Apple hardware and real multi-host operation.

Preserve algorithms, precision, ownership and test tolerances. Large profiling
workloads belong in scripts, separate from correctness tests. No broad rewrite
or exhaustive private-function comparison project is planned.

## Audit evidence

The frozen [caller audit](../validation/cuda-caller-audit.json) at `382f78c`
classified 174 logical operations and 1,078 lexical sites. Its ten implementation
tasks were subsequently fixed and locally accepted; the original open rows are
historical findings, not the current backlog. The
[operation map](../validation/cuda-operation-map.json) covers 42 families and 109
CUDA source/header roots. These inventories are source maps, not completion
percentages or correctness proofs.

```sh
python3 scripts/check_cuda_audit.py --check-inventory
python3 scripts/check_cuda_coverage.py
```

Detailed progress narratives and independent reviews remain in Git history.
