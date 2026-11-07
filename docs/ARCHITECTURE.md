# Implementation guide

The port extends vendored XGBoost rather than replacing its learner/model APIs.
The public device name is `mps`; computation uses custom Metal shaders. See the
[review](REVIEW.md) and [plan](PLAN.md) for engineering limits and next work.

## Entry points and ownership

| Layer | Main locations, relative to `xgboost/` | Responsibility |
| --- | --- | --- |
| Python/native API | `python-package/xgboost/{core.py,_metal.py}`, `src/c_api/c_api.cc` | Validate descriptors, retain producers, expose models/outputs/diagnostics |
| Learners | `src/tree`, `src/objective`, `src/metric`, `src/predictor` | Metal dispatch with source-defined CPU orchestration |
| Family interfaces | `src/common/metal_*.h` | Operation contracts, buffer ownership and views |
| Host runtime | `src/common/metal_histogram.mm` | Pipelines, buffer binding, dispatch/completion; many families currently share this file |
| GPU arithmetic | `src/common/metal_histogram.metal` | Kernels, scans, sorts, tree operations and software binary64 |
| Owned arrays | `src/common/host_device_vector_metal.cc`, `metal_vector.h` | GPU authority, lazy host views, fills/copies/growth |
| Diagnostics | `metal_statistics.h`, `metal_histogram.mm`, `c_api.cc` | Plain counter schema, snapshots and shared Metal/CPU JSON mapping |

`Context("mps")` still has a CPU default `DeviceOrd` for generic storage, so
`IsMPS()` and `IsCPU()` can both be true. A kernel's existence does not guarantee
every caller selects it or preserves GPU storage. Trace producer, consumer,
lifetime and explicit host-access boundary; changing the context globally is a
separate migration.

Owned shared Metal buffers are CPU-addressable, but direct CPU access is still
a placement decision. `HostDeviceVector` tracks authority; requesting a host view
can create a mirror. Residency tests must call the next GPU consumer before host
comparison. Shared memory does not prove absence of CPU copies or arithmetic.

Native DLPack capsules are consumed once. Capture a shared capsule once within an
API call and retain its export across iterator passes; separate calls need fresh
exports. Current operations generally wait before returning. Asynchronous work
requires an explicit ownership/readiness contract.

Resident feature split evaluation has two command completions: one for the
allocation length/error readback, then one after task construction, histogram
gathering, scoring, finalization and winner extraction. Nodes share encoders
within each stage. Integer and precise parent inputs have separate staging
buffers; resident vector parents bind their owned buffer directly. Borrowed
inputs remain alive until the synchronous call returns, and outputs are read
only after completion.

Packed host input owns its copied payload. Only alignment padding is zero-filled;
the payload is appended directly, preserving offsets and completion lifetime.

First-page histogram builds write directly into contiguous integer-cache output
slots, then materialize their binary64 view with the existing conversion kernel.
Only that output range is cleared. Later pages and discontiguous destinations
retain the temporary histogram and store/add path; they must preserve existing
cache entries. The quantization scale and integer sums remain authoritative.

Large, narrow compressed pages can own a second, column-major byte index for row
partitioning. This improves repeated feature gathers while histograms, prediction
and exports retain the canonical compressed representation. Eligibility requires
a dense eight-bit page with 256 symbols, 4–64 features, at least 2^20 rows, and
contiguous full-head or full-tail storage. Each cache is bounded by the smaller
of 512 MiB and one thirty-second of Metal's recommended working-set size; this
is a per-page limit, not a global memory budget. Wide, small and split-storage
pages keep the compressed classifier. `std::call_once` publishes the cache after
the existing transpose kernel completes. The page owns its lifetime, and repeated
partitions reuse it. Transposition time is recorded in `index_gpu_time_ns` and
included in whole-training benchmarks.

## Numerical and refactoring boundaries

Quantized histogram and split paths preserve integer state. Reconstructing exact
integers from floating views loses that contract. CUDA-source checks, CPU
comparisons and NVIDIA execution are distinct evidence; see [the ledger](CUDA_PARITY.md).

Scalar quantized splits reuse identical missing-direction gains when both integer
missing sums are zero. Numeric features with at least 32 bins and zero missing
sums can use conservative gain intervals when L1 regularization, delta-step limits,
monotone constraints and leaf-weight bounds are absent. These reject candidates whose
upper bound is strictly below another candidate's lower bound. Surviving
candidates use the original binary64 calculation; ties are retained. Values
outside the interval helper's supported range receive unbounded intervals and
fall through. Screening forms widened FP32 intervals directly from integer sums;
the Hessian lower bound must prove child eligibility before a gain lower bound
can raise the cutoff. Category-weight scratch holds numeric upper bounds, with each
lane reading only its own entries. No precision or selection tolerance changes.

FP32 ordering in quantized splits uses exact integer bit comparisons. NaNs are
unordered, signed zeros compare equal, and negative magnitudes sort in reverse.
This preserves subnormals without converting each operand to software binary64;
it changes no gain arithmetic or tie rule.

At the `6f3dcbd` review baseline the runtime/shader contained roughly 20,000 lines
and 218 kernel entry points. If a concrete maintenance need justifies future
extraction, existing family headers supply boundaries. Extract one family at a
time, preserving arithmetic, APIs, lifetimes and shader dependencies; keep
behavioral changes separate from extraction.

`tests/` is executable validation, `validation/` is revision-qualified evidence,
and ignored `results/` holds local logs/caches. `docs/archive/` preserves history;
README, the current parity ledger and the plan describe current status.
