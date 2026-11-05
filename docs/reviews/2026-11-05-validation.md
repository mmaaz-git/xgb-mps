> Independent review notes from the baseline/draft inspection. Findings may have been fixed during integration; see [current dispositions](../REVIEW.md). Paths/line numbers refer to the reviewed revision unless stated otherwise.

# Validation, build, CI and benchmark review

Reviewed 2026-11-05 UTC. Baseline: `2eaad4dd0b49dca26ba12320c01b5f052e039ed2`.
This review inspected the standard/optional build runners, native test inventory,
pytest gating, CUDA CI history, structural coverage checker, representative
training/source-reference tests, benchmark drivers and recorded validation.
It did not rerun the 3,554-case suite or inspect every test assertion. Root is
running a representative integration check of the cleanup separately.

## Findings, in priority order

### V1 — High: test runner could link stale implementation against new headers

**Confirmed, fixed in this cleanup.** Original `scripts/test.sh:9–17` immediately
compiled a native test against `xgboost/lib`, and its remaining 100 groups never
built the library or checked source/library identity. This is especially unsafe
when `metal_histogram.h::Statistics` changes layout. The prior work-in-progress
library and restored main already formed exactly this stale-build situation.

The replacement runner builds current sources first, recompiles every selected
native test, then logs each command, actual exit status, revision, tracked diff
hash, untracked file hashes, manifest hash and resulting library hash. Its
`--list` and `--dry-run` modes do not build. Rebuilding a selected test is
intentional; caching a binary without header dependency tracking would revive
the ABI problem.

Acceptance: clean or dirty current source builds before any selected native or
Python execution; a failed build stops the run; evidence identifies the library
and selected tests. Root's integration check supplies runtime verification.

### V2 — Medium: the default validation command was an all-or-nothing task

**Confirmed, fixed in this cleanup.** Original `scripts/test.sh` had 727 lines,
100 compiler commands, no selector, and a full pytest run at lines 724–727.
Most compilation flags were duplicated. The latest recorded standard run took
9m45s; repeatedly invoking it while making small related fixes adds latency
without improving coverage in proportion to the work.

`tests/native_tests.json` now contains the exact 100 groups in original execution
order, with 11 shared compiler profiles (preserving existing flags, framework
ordering and Objective-C ARC differences). `scripts/run_tests.py` selects
native IDs/globs and Python files or node IDs. `scripts/test.sh` is a six-line
compatibility entry point; no arguments still means the full standard suite.

Mechanical migration verification passed:

- All 100 old compiler argv lists equal expanded manifest compiler argv lists.
- All 100 old run argv/environment pairs equal the manifest, in execution order.
- The only native source outside the standard manifest is
  `tests/metal_federated_api.cc`, still handled by the optional federated runner.
- List, full/focused/glob dry-run, unknown-selector rejection, Python compilation
  and shell syntax checks pass. No heavy job was run by this reviewer.

### V3 — Medium: “full” and GPU-required do not mean all integration dependencies

**Confirmed, still open.** `tests/conftest.py:10–16` makes `--require-metal` fail
when Metal is unavailable. However `tests/test_metal_tensor.py:16/18` and
`tests/test_metal_output.py:55/310` use `pytest.importorskip` for PyTorch/MLX.
Missing optional runtimes can silently remove producer-interop coverage from a
future green run. The recorded 3,554-pass/one-skip run had these runtimes; this is
not evidence of a hidden omission in that recorded run.

Also, standard `scripts/test.sh` ends at pytest: CPU-only build/API checks,
optional federated builds and NVIDIA execution are separate. Existing records
mention extra manual CPU commands; the standard command alone does not reproduce
all those claims.

Acceptance: document separate standard, CPU-only, producer-interop and optional
federated gates; require installed producer runtimes in the interop gate and
report skips/reasons. Keep targeted local runs available. Add shared-code CPU
regressions from upstream where applicable instead of assuming Metal Python
tests plus unavailable-API checks fully validate CPU behavior.

### V4 — Medium: inventory closure is not a completed operation audit

**Confirmed, correctly caveated but unfinished.**
`scripts/check_cuda_coverage.py:31–64` verifies identifiers, source sets and
existing paths. It neither executes tests nor resolves every CUDA helper caller.
The checker explicitly asserts `completeness == "not_proven"`; this is honest.
It should not be elevated to a completion gate by itself.

Source-reference tests such as `tests/cuda_multi_split_reference.py:1–4`
explicitly do not execute NVIDIA code. They are useful independent-language
checks, but shared assumptions between a port and its hand-written oracle can
agree while differing from the source. Similarly, loading the same Metal-trained
model into a CPU predictor tests model/predictor compatibility, not independent
CPU training equivalence. Separate CPU-training checks do exist, for example
`tests/test_metal_training.py:50–65` and `:69–81`.

Acceptance: each remaining work item needs a concrete CUDA caller, Metal
consumer, execution/storage assertion, numerical oracle and a named test.
Classify evidence as CPU-training equivalence, CPU-reader compatibility,
CUDA-source oracle, Metal placement or real NVIDIA execution. Define completion
of implementation separately from unavailable cross-hardware validation.

### V5 — Medium: historical benchmark provenance is uneven

**Confirmed, still open.** Current training/published-data drivers record
revision, dirty status and timing boundaries (`scripts/benchmark.py:96–105`,
`scripts/benchmark_published.py:174–185`), alternate execution order and record
sleep/clock discontinuities. These are useful controls.

But `scripts/benchmark_prediction.py:35–37`,
`scripts/benchmark_auc.py:29–33`, and `scripts/benchmark_ranking.py:30–34` omit
the source revision/build fingerprint. Inspection found 79 historical
`validation/*benchmark*.json` records without a top-level `commit` field. A
historical measurement is not invalid solely for this reason, but it cannot
establish the current implementation's speed without recovering provenance.
Metric/kernel-only speedups also cannot stand for end-to-end training gains.

Acceptance: retain historical data with explicit revision/unknown labels;
standardize future reports on revision, dirty diff, library hash, hardware,
thread count, dataset shape/provenance, timing boundary, warmup, repeated
alternating runs, accuracy result and clock validation. Rebenchmark a bounded
representative matrix once implementation coverage stabilizes, not after every
correctness patch. Include regressions and CPU-faster workloads.

### V6 — Low: some tests' purposes are unclear

**Confirmed ambiguity, not a proven redundant setup.**
`tests/test_metal_training.py:26–27` trains a CPU booster named `cpu` that is
never used in any subsequent assertion in that test. It may intentionally warm
the DMatrix cache before Metal access; deleting it blindly could remove a
transition scenario. Either explain it as cache preparation and drop the unused
binding, or compare the trained CPU model if the intended claim is equivalence.

## GitHub Actions diagnosis

The only current workflow is CUDA compile/link on a Linux CPU runner:
`.github/workflows/cuda-compile.yml:21–42`. It does not run Metal or NVIDIA GPU
tests. The three initial failures were dependency configuration failures:

- `41977126433`: attempted replacement of held NCCL packages.
- `41977330547`: nvCOMP CMake package discovery.
- `41977632903`: nvCOMP 6 removed APIs used by vendored CUDA code.

All three subsequent runs passed: `41978324629`, `41980647083`, and
`41982974034` (baseline `2eaad4d`, completed 2026-11-05T07:15:05Z).
This is consistent with `validation/cuda-compile-ci-13.4.txt` and current CLI
status. No active run was cancelled by this review.

Cleanup adds concurrency cancellation for future superseded builds of the same
branch/PR. A cheap inventory/manifest check in CI would be useful later; do not
mistake it for Metal CI. Actual Metal GPU validation needs an appropriate Apple
GPU runner; it is presently local.

## Runner usage and proposed validation gates

```sh
bash scripts/test.sh --list
bash scripts/test.sh --native metal_model_state_api metal_split_categories --dry-run
bash scripts/test.sh --native metal_model_state_api metal_split_categories \
  --python tests/test_metal_model_state.py --log-dir results/my-focused-run
bash scripts/test.sh --python tests/test_metal_training.py::test_partition_block_boundaries
bash scripts/test.sh --full
```

Explicit log directories must be new; default log directories use UTC time and
PID. Each actual run has `summary.json` plus numbered command logs. Failed or
interrupted runs retain truthful exit codes and stop instead of reporting a
partial suite as passed. Child process groups are terminated on interruption.

Recommended gates:

1. Every change: static/reference checks plus focused native/Python consumers;
   always rebuild the changed library and selected native clients.
2. Integration checkpoint: full standard Metal suite and targeted CPU-only
   shared-code regressions; require interop dependencies where advertised.
3. Collective changes: add the optional federated/transport selections affected.
4. Shared CUDA compilation changes: retain successful CUDA CI at the revision.
5. Performance milestone: fixed reproducible dataset matrix, correctness and
   timing together; no speedup guarantee as a correctness requirement.

Do not rerun the full suite for documentation-only changes. Do not claim a
feature is complete from test counts or inventory counts. Existing tests provide
substantial evidence; the review found process/reproducibility gaps, not grounds
to discard the test suite or rewrite it wholesale.
