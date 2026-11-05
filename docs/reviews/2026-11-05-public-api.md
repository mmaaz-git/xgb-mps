> Independent review notes from the baseline/draft inspection. See [current dispositions](../REVIEW.md) and [final validation](../../validation/repo-review-2026-11-05.txt). Paths/line numbers refer to the reviewed revision.

# Public API and test-contract review — 2026-11-04

Scope: Python Metal descriptors/capsules, metadata import, prediction output
selection, sklearn wrapping, representative CPU comparisons, reload tests, and
optional-dependency skip behavior. Native storage and the broader test runner are
covered by the other reviewers. Baseline source: `2eaad4d`.

## Confirmed defect, fixed pending runtime validation

**P2 — Aliased raw Metal metadata capsules are consumed more than once within one API call.**

- Baseline `xgboost/python-package/xgboost/core.py:1653–1658` independently
  transforms each QuantileDMatrix metadata value. Passing one raw export as both
  `label` and `weight` (or `base_margin`) renames it on the first transform;
  `is_metal()` calls `_capsule()` for the second and raises
  `ValueError: DLPack capsule has already been consumed.`
- Baseline `core.py:957–975` similarly passes each metadata field through a
  separate setter. This affects ordinary DMatrix/set_info and DataIter callbacks
  used by external-memory matrices.
- `_metal.py:364–378` already provides `capture_tensors` to deduplicate aliased
  exports for custom gradients. The metadata paths were not using it.
- Pure-Python reproduction is retained as
  `results/review-metadata-capsule-repro.py`. It loads `_metal.py` with a stub C
  bridge, constructs a synthetic managed capsule, and evaluates the actual
  QuantileDMatrix intake assignment extracted from baseline/current source ASTs.
  No native library or GPU is loaded. Baseline raises; current intake retains
  the same wrapper for both fields and still rejects independent second use.

Changes authorized by root and made by this reviewer:

- `core.py::DMatrix.set_info` captures the eight numeric metadata fields together
  before dispatching the existing field-specific setters/validators.
- `core.py::QuantileDMatrix._init` retains deduplicated exports before the
  repeated iterator passes. Shape validation and native import remain in their
  existing paths. Metadata exports live through all setters/passes.
- `tests/test_metal_metadata.py::test_aliased_metadata_capsule`: 32 cases covering
  ordinary construction, set_info, quantile construction and external-memory
  callbacks; label+weight and label+base_margin; legacy/versioned capsules; both
  PyTorch/MLX. Tests also check native imports, metadata-specific shared output/no host copies, owned metadata
  after producer mutation/deletion, and rejection of independent capsule reuse.
- An external-memory callback makes one fresh export per invocation; this change
  deliberately does not allow reusing a consumed raw capsule across independent
  callbacks or calls. Cross feature/metadata aliases were not changed or tested.

Acceptance: new regressions and existing metadata/tensor/custom-gradient capsule
and lifetime tests pass against the clean rebuilt library; ordinary CPU metadata
continues to work. Root owns that runtime run. At report time only py_compile,
Black, diff-check and the no-GPU reproduction had run here.

## Confirmed validation gap

**P2 — `--require-metal` does not require the producer interoperability dependencies.**

`tests/conftest.py:10–16` fails on unavailable Metal, but
`tests/test_metal_tensor.py:16–18` and
`tests/test_metal_output.py:55,310` still use `pytest.importorskip` for torch/MLX.
A machine with Metal but without those packages can therefore produce a green
run while skipping large tensor/metadata/output matrices. This does not invalidate
the recorded 3,554-pass/one-skip M3 run, where those dependencies were present.

Recommendation: strict full/interop tier preflight for required producer packages,
or make the narrower tier explicit in the result summary. Keep optional imports
for ordinary users. Report selected tests and skips, not only exit status. This
finding was sent to the validation reviewer for runner integration.

## Evidence distinctions (not defects in these tests)

- `tests/test_metal_updater.py:43–48` loads the same trained model into a CPU
  Booster and compares prediction. That is valuable CPU reader/serialization
  compatibility, not independent CPU training equivalence. The test naming is
  accurate; aggregate documentation must not broaden its conclusion.
- `tests/test_metal_output.py:89–112` compares owned output with a warmed GPU
  prediction and then a CPU reader. This exercises publication/output ownership;
  its no-readback interval intentionally begins after warmup. It does not prove
  that the first call has zero upload/readback. Separate initial-state tests or
  explicitly bounded claims are needed when discussing end-to-end residency.
- `tests/test_metal.py:97–108` does train CPU and MPS independently and compares
  margins. Its positive-weight fixture avoids known valid CPU/CUDA ties, with
  source-oracle tests covering those boundaries. Do not replace that distinction
  with a universal bitwise CPU-equality requirement or widen tolerances merely
  to increase a pass count.

## Contracts that appear deliberately handled

- Raw capsule ownership is renamed once, producer state is retained, and failure
  before rename does not steal the capsule. The existing tests include counted
  deleters and both DLPack layouts.
- `MetalOutput` has explicit host readback, independent ownership after model
  deletion, shared writable exports, optional independent GPU copies, and rejects
  unsupported stream/device requests. sklearn converts it to NumPy deliberately.
- `_metal.py:92–101` synchronizes loaded torch/MLX runtimes; arbitrary producers
  must finish their writes. This is a documented synchronous interoperability
  limitation, not portable asynchronous stream support.
- The missing-GPU test is deliberately skipped on this GPU machine and checks for
  a hard error on a machine without Metal. That expected skip is distinct from
  missing optional producer dependencies.

## README dependency sanity check

The quickstart's numpy/scipy/pandas/pytest requirements match direct imports in
port tests; scikit-learn is needed for the README estimator example. Python >=3.11
is required by package metadata. PyTorch/MLX are optional for basic use but needed
for complete producer coverage. Snappy is a native-test dependency, not linked by
the production Metal library. Federated tests have separate gRPC/Protobuf needs.
No additional mandatory Python dependency was found in the port test import scan.

This review did not load the quarantined/inconsistent local native library, run
GPU training, benchmarks, or install dependencies. It does not certify full API
coverage or the native implementation.

## Focused runtime follow-up

Root's first rebuilt-library run produced 703 passes and 16 failures in the new
quantile/external cases. The failures were an incorrect test assertion:
`input_staging_bytes` also counts staging of this fixture's NumPy feature matrix,
so it cannot isolate native metadata publication. No production failure was
identified by those failures. The test now checks `metadata_host_copy_bytes`
unchanged and at least two metadata payloads' worth of
`metadata_shared_output_bytes`, while retaining native-import, snapshot/value,
and independent single-consumption checks. No production code changed for this
correction. Root will rerun the affected 32-case regression.
