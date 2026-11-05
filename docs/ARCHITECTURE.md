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

## Numerical and refactoring boundaries

Quantized histogram and split paths preserve integer state. Reconstructing exact
integers from floating views loses that contract. CUDA-source checks, CPU
comparisons and NVIDIA execution are distinct evidence; see [the ledger](CUDA_PARITY.md).

The runtime/shader contain roughly 19,000 lines and 212 kernel entry points at
the reviewed baseline. Existing family headers supply extraction boundaries.
Extract infrastructure, then one family per change; preserve arithmetic, APIs,
lifetimes and shader dependencies. Keep algorithm changes, lazy pipeline creation
and asynchronous submission separate from extraction.

`tests/` is executable validation, `validation/` is revision-qualified evidence,
and ignored `results/` holds local logs/caches. `docs/archive/` preserves history;
README, the current parity ledger and the plan describe current status.
