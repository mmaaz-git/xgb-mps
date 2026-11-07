# Implementation guide

The port extends vendored XGBoost's learner/model APIs. The public device name is
`mps`; computation uses custom Metal shaders. See [coverage](CUDA_PARITY.md) for
supported operations and numerical limits.

## Entry points

| Layer | Locations relative to `xgboost/` | Responsibility |
| --- | --- | --- |
| Python/native API | `python-package/xgboost/{core.py,_metal.py}`, `src/c_api/c_api.cc` | Descriptors, producers, models, outputs and diagnostics |
| Learners | `src/{tree,objective,metric,predictor}` | Metal dispatch and source-defined CPU orchestration |
| Family interfaces | `src/common/metal_*.h` | Operation contracts, ownership and views |
| Host runtime | `src/common/metal_histogram.mm` | Pipelines, buffers, dispatch and completion |
| GPU arithmetic | `src/common/metal_histogram.metal` | Kernels, scans, sorts, trees and software binary64 |
| Owned arrays | `src/common/{host_device_vector_metal.cc,metal_vector.h}` | GPU authority and lazy host views |
| Diagnostics | `src/common/metal_statistics.h`, runtime and C API | Counters and shared CPU/Metal JSON schema |

## Ownership and execution

`Context("mps")` retains a CPU default `DeviceOrd` for generic storage, so
`IsMPS()` and `IsCPU()` can both be true. Trace producers, consumers and explicit
host-access boundaries when changing dispatch. CPU-addressable shared Metal
buffers do not prove absence of CPU copies or arithmetic: `HostDeviceVector`
tracks authority, and requesting a host view can create a mirror. Residency tests
must invoke the next GPU consumer before inspecting host values.

Operations generally wait before returning. Borrowed inputs must survive to
completion; asynchronous work requires a separate readiness/lifetime contract.
Packed host input owns its copied payload and zero-fills only alignment padding.
Resident feature splits complete once for allocation/error readback, then again
after task construction, scoring and winner extraction. Scalar splits read
immutable histogram-cache segments directly; vector splits gather/interleave
targets. Nodes share encoders; integer and precise parent staging remain separate.

## Python interoperability

PyTorch MPS and MLX tensors can enter ordinary/quantile matrices and native
prediction through DLPack. `inplace_predict` with native Metal input returns an
owned `xgb.MetalOutput`; `output_device="mps"` explicitly requests Metal output.
Use `torch.from_dlpack(result)` or `mlx.core.from_dlpack(result)` to consume it on
GPU, or `result.numpy()` for host readback. The sklearn interface returns NumPy.

An imported capsule is consumed once. Aliased metadata shares one captured export
within an API call, including iterator passes; independent calls need fresh
exports. Output exports retain their allocation; `__dlpack__(copy=True)` makes
an independent GPU copy. Only `stream=None` is supported. Loaded PyTorch/MLX
runtimes are synchronized; arbitrary producers must finish writes before export.

## Histogram and sketch storage

First-page histograms write directly into contiguous integer-cache output slots,
clear only that range, then materialize binary64 views. Later pages and
discontiguous destinations retain staging/store-add paths so existing cache
entries survive. Integer sums and quantization scales remain authoritative.

Wide scalar split batches (at least 1,024 columns and 4,096 feature tasks) first
compute conservative raw-gain intervals. A feature is skipped only if its upper
bound is strictly below another feature's lower bound for that node. Missing,
constrained and uncertain cases retain exhaustive evaluation; ties survive.
This removes losing software-binary64 evaluations without changing cut values,
gain arithmetic or adding a host readback.

Moderately missing compressed pages use shared histograms over **2,048 global
bins**, with exact two-word integer accumulation. Eligibility requires 128–256
features, 16,384–65,536 bins, at least 65,536 selected rows, 32 KiB threadgroup
memory, and an ELLPACK stride in `[ceil(3*features/4), features)`. Maximum stride
is a performance heuristic, not average density. Other geometries use existing
kernels. Null padding lies beyond the final valid bin range.

Two optional per-page caches improve repeated reads:

| Cache | Eligibility and bound |
| --- | --- |
| Sparse range starts | One byte per row/range; 16,000,000 bytes for 1M × 128 with 256 bins. Cap: minimum of 64 MiB, recommended working set / 128 and maximum buffer length. Over-budget pages use uncached binary search. |
| Partition columns | Dense eight-bit compressed pages with 256 symbols, 4–64 features, at least 2²⁰ rows, and contiguous full-head or full-tail storage. Cap: minimum of 512 MiB and recommended working set / 32. Other pages keep compressed classification. |

Both preserve canonical compressed bins and publish with `std::call_once` after
GPU completion. The page owns the cache; these are per-page caps, not a global
budget. Preparation counts toward training time and `index_gpu_time_ns`.

Sketch value/feature sorts reuse record, rank and count scratch within one input
window. Feature keys replace their own records in place. Unweighted sketches
omit unused scan inputs and rank their own entries in place; weighted ranks need
a separate destination because they read the previous prefix. Sort stability,
window boundaries, FP32 weighted scan association, pruning and cuts are unchanged.

## Split screening

Conservative intervals skip expensive binary64 gain evaluation only when a
candidate cannot win. Survivors retain original float weights/squares, binary64
intermediates, final float gain and tie order. Screening requires numerical
features, zero L1 penalty and maximum leaf step, no monotonic constraint, and
infinite weight bounds. IEEE-bit zero checks preserve subnormal restrictions.

For gradient `g`, Hessian `h` and L2 penalty `λ`, let `Q = g²/(h + λ)`. A computed
weight `w = -(g/(h + λ))(1 + δ)` gives unrounded quadratic gain `Q(1 - δ²)`.
Float-square error contributes at most `(1 + δ)² ε`, with `ε = 2^-23`; including
binary64 arithmetic and final rounding stays below `4ε` in the supported normal,
finite range. The implementation widens by **32ε**.

Adjacent floats bracket converted binary64 inputs. Outward square, denominator
addition and opposite-endpoint division bound `Q`; outward child-gain addition
and parent subtraction bound the precise evaluator's result. Nonzero gradient
magnitudes and positive Hessians must lie in `[1e-8, 1e8]`, with `λ` in `[0, 1e8]`.
Exact-zero gradients have zero gain. Other inputs get unbounded intervals and
full evaluation; values near zero widen to normal endpoints to tolerate Metal's
subnormal flushing. These bounds use basic-operation rounding with fast math
disabled, as specified in Apple's
[Metal language specification, §§8.1, 8.2, 8.4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

Quantized screening additionally requires at least 32 bins and zero integer
missing sums. It widens integer-to-float conversions by two ULPs, limits scale
exponents to `[-96, 96]`, and requires Hessian lower bounds to prove child
eligibility before raising the cutoff. It compares raw child gains; its cutoff
starts at `-MAXFLOAT`, preserving negative/zero candidates. The precise evaluator
subtracts parent gain and starts at zero. Both reject only **upper < cutoff**,
retaining all possible winners and ties. Quantized numeric tasks reuse
category-weight scratch; the precise path uses spare suffix storage or recomputes
missing-direction bounds. No precision or selection tolerance changes.

FP32 split comparisons use IEEE bits: NaNs are unordered, signed zeros compare
equal, and negative magnitudes reverse order. This preserves subnormals without
software binary64 conversion. Tests cover independent interval containment,
exact split choices, near ties and complete training comparisons.

## Maintenance

Family headers provide extraction boundaries if a concrete maintenance need
justifies splitting the runtime/shader. Keep structural changes separate from
arithmetic changes and preserve APIs, lifetimes and shader dependencies.
`tests/` is executable validation, `validation/` is revision-qualified evidence,
and ignored `results/` contains local logs and caches.
