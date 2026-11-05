# Local acceptance of the CUDA port

Updated 2026-11-05. Production source: `68c800e`; numerical fixtures: `0c66454`.

**The ten implementation tasks and seventeen numerical contracts from the frozen
audit now have local acceptance evidence, including a passing CUDA compile/link
gate. VERIFY-1 is complete. Full CUDA parity is not established.** No NVIDIA execution
comparison has been performed, and current-source performance work remains open.

The [machine-readable record](../validation/cuda-verify1-m3-run.json) maps all
sixteen original missing/defective rows to their ten fixes, and every numerical
contract to named passing fixtures and outstanding comparisons. The
[original audit](CUDA_AUDIT.md) remains unchanged; its source locations and open
findings describe `382f78c`, not current implementation status.

## Gates and provenance

| Gate | Evidence and boundary |
| --- | --- |
| Full Metal regression | At `68c800e`: 103 native groups, 3,647 Python passes and one expected skip. PyTorch and MLX GPU interop ran. [Record](../validation/cuda-port11-m3.txt). |
| CPU build and API/updater checks | Same source: 305 disabled-API/schema/cache fields and nine updater tests passed against the separate CPU library. |
| Optional federated build/runtime | Same source: 14 transport, sketch and learner groups, including TLS. Every worker was on this M3; no multi-host claim. |
| Numerical witnesses | Four native groups and 49 Python tests passed. The new group has 21 witnesses and 3,431 exact checks. [Raw witness output](../validation/cuda-verify1-witnesses-m3.txt). |
| CUDA compile/link | [Run 42027936238](https://github.com/mmaaz-git/xgb-mps/actions/runs/42027936238) at `bf1ebea` passed all 197 steps, including the final shared-library link. CUDA, NCCL, nvCOMP and federated consumers compiled on a CPU runner. |
| NVIDIA execution | Not run. Portable inputs and source oracles exist; NVIDIA harness integration and actual hardware runs remain VERIFY-2. |

The production Git tree and Metal library SHA256 are unchanged from the full
run. All 103 existing native manifest entries and profiles are unchanged. The
104th group passed separately; **there was no single 104-group full invocation**.
The execution record retains the tested manifest hash and the committed hash:
only JSON formatting changed after the focused run, with parsed equality checked.
No production arithmetic or tolerance changed during this verification step.

Focused-run command (inspect retained outputs before repeating):

```sh
bash scripts/test.sh --native metal_initialization_state_api metal_quantile \
  metal_numerical_contracts metal_cuda_sketch_api \
  --python tests/test_metal_quantiles.py tests/test_metal_sketch.py
```

Hardware and dependency versions are retained in the full-run record: Apple M3,
10-core GPU, 16 GiB, macOS 26.6.2. The sole full-run skip is the no-GPU-only test,
which is inapplicable on this GPU-equipped machine. This evidence does not cover
other Apple GPUs, multi-host operation, or every representable input size.

## Seventeen numerical contracts

CUDA source defines the formulas, precision and RNG branch. CPU comparisons
apply where those agree. “Local acceptance” means that the named behavior has
passing fixtures under its declared limits; it does not mean NVIDIA agreement.
The machine-readable record links each row to exact run logs and SHA256 hashes.

| Contract | Locally checked behavior | Remaining comparison or limit |
| --- | --- | --- |
| training-rng-modern | Exact modern Philox/uniform fixtures and Thrust-style unweighted shuffle keys. | Pin NVIDIA compiler/CCCL branch; legacy streams are distinct. |
| training-reduction-association | Exact integer histograms/leaf sums; source floating terms and forward-error bounds. | Actual CUB association/contraction; legacy CUDA external-page limitations. |
| training-shap-order | Eight-point quadrature, finishing expressions, finite-field replay and exceptional classifications. | NVIDIA cross-tree atomic order need not equal Metal row/tree scheduling. |
| P5-N01 | NUM-1 binary64 threshold fix, adjacent boundaries, native ownership, striding and output aliases. | Execute the same boundaries on CUDA; CPU threshold precision differs. |
| P5-N02 | Ordinary AFT tolerance plus separate sensitive-tail/source-expression fixtures. The sensitive corpus retains 322 CPU-tolerance exceedances. | Actual CUDA sensitive-tail corpus; no CPU repair or widened ordinary tolerance. |
| P5-N03 | Software binary64, casts, extreme/subnormal/nonfinite inputs and sampled math budgets. | NVIDIA/other Apple math. Sampled AFT four-ULP bounds are not universal guarantees. |
| P5-N04 | Deterministic ranking maximum factor and explicit nonfinite exponent policy; global-versus-last-query witness. | Characterize CUDA's racing writes without inventing a unique expected race winner. |
| P5-N05 | Independent signed truncation, raw normalization, combined weights, soft-label MAP and replay. | NVCC/CUB arithmetic and contraction flags. |
| P5-N06 | Ranking seed wrap, discard/rejection, singleton/empty cases and exact modern pair order. | Same branch pinning as training RNG; no claim for older Thrust streams. |
| P5-N07 | FP32/binary64 precision witnesses, cancellation, overflow and bounded reduction association. | Huge 1,073,741,823-element batches and other reduction architectures are untested. |
| P5-N08 | New weighted-CDF witnesses isolate association: different legal grouping changes the selected observation. | Publish actual NVIDIA CDFs/selections; illustrative regrouping is not observed CUB output. |
| P5-N09 | PR source pathology retained, with exact terms and independent 320-bit prefix diagnostics. | NVIDIA association/NaN/error accounting; no hidden stabilization of the source formula. |
| P5-N10 | ROC count-sum adaptation preserves source FP32 pair-weight terms and ordinary `2e-12` comparisons. | Compare common valid CUDA domain; safe empty/singleton and million-row cases are labeled extensions. |
| P5-N11 | Ranking fallback/top-k/soft labels and Decimal80 cancellation diagnostics; ordinary `2e-12` metric tolerance. | Actual NVIDIA branches and IDCG inverses; near-zero denominators amplify relative error. |
| P5-N12 | Metal fast math disabled, explicit software operations and host contraction disabled for references. | External evidence only: emitted NVCC/libdevice behavior and GPU/compiler/CCCL/FMA settings. |
| P5-N13 | New neighboring-rank chains distinguish repaired and unrepaired predecessor reads. | Synthetic operation witness, not claimed training reachability or predicted CUDA race output. |
| P5-N14 | New segmented-prefix tile boundaries and reverse-duplicate association witnesses; existing exact predicate/partition tests. | Actual NVIDIA summaries; a Metal-association oracle is not proof of CUB identity. |

The new [portable fixtures](../tests/numerical_contract_fixtures.h) and
[Metal runner](../tests/metal_numerical_contracts.mm) cover exponents `0`, `-149`
and `100`. Weighted quantiles use `alpha=1` so the threshold itself is exact,
separating association from the fixed NUM-1 narrowing bug. Inputs with a large
weight followed by two small weights select row zero under the production serial
CDF and the last row under the explicitly regrouped CDF. Prefix fixtures cross
256-entry tiles; summary fixtures include empty columns and guard checks.
Alternative outputs describe specific arithmetic/read orders, never measurements
of NVIDIA execution.

## Resolution of the sixteen original gap rows

The detailed acceptance records retain the original tests, source commits,
failures and limits. The current full suite contains their regression groups;
the machine record names each original operation ID and passing current group.

| Task | Implemented producer/consumer path | Original rows | Acceptance record |
| --- | --- | ---: | --- |
| NUM-1 | Binary64 quantile threshold through resident initialization. | 1 | [Quantile precision](../validation/cuda-num1-m3.txt) |
| PORT-9 | Retained sampler decisions through vector leaf-position validity. | 1 | [Sampling validity](../validation/cuda-port9-m3.txt) |
| PORT-1 | Resident weights through queued candidates, root/model copies and prediction cache. | 4 | [Vector weights](../validation/cuda-port1-m3.txt) |
| PORT-2 | Selected category storage through model growth and partitioning. | 2 | [Categorical payloads](../validation/cuda-port2-m3.txt) |
| PORT-3 | Exact integer roots/children through persistent sums and evaluator gathers. | 2 | [Vector sums](../validation/cuda-port3-m3.txt) |
| PORT-6 | Sampled/allowed feature IDs through task assembly and split evaluation. | 2 | [Feature selection](../validation/cuda-port6-m3.txt) |
| PORT-7 | Native row/group weights through expansion, snapshots and matrix sketch constructors. | 1 | [Quantile weights](../validation/cuda-port7-m3.txt) |
| PORT-8 | GPU bound updates through snapshots, feature evaluation and finalization. | 1 | [Split bounds](../validation/cuda-port8-m3.txt) |
| PORT-10 | Resident cuts through GPU winner thresholds and evaluator metadata. | 1 | [Thresholds](../validation/cuda-port10-m3.txt) |
| PORT-11 | GPU pointer/type initialization through later sketch pushes and merges. | 1 | [Sketch initialization](../validation/cuda-port11-m3.txt) |

Source-defined host topology, selected control metadata, serialization and
finalizers remain host work. Residency checks combine known GPU producers and
consumers, ownership tests, and transfer counters. Counters alone cannot detect
every CPU dereference of unified memory. Closing this bounded list does not prove
that no unlisted defect exists.

## Next actions

1. For VERIFY-2, wire shared fixtures into the vendored CUDA paths and execute
   them on NVIDIA with GPU/compiler/dependency versions and flags recorded.
   Separate exact, bounded, source-race and domain-extension cases as above.
2. With local acceptance complete, freeze a source revision and run repeated large
   HIGGS, wide, sparse and vector CPU/MPS benchmarks. Publish CPU wins too.

NVIDIA availability is an external evidence requirement. It does not justify
unrelated Metal changes. The [completion plan](PLAN.md) remains the work queue.
