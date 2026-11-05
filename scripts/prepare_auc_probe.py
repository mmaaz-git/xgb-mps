"""Generate a test-only AUC source copy with explicit observation boundaries.

The original CUDA source remains untouched. Only its namespace, its registry
file tag, and the listed readback hooks differ. The source hash and a reversible
edit record prevent silently instrumenting a different implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "760e940b441b75ad9d03b52ad552a4398ce65bf6415feaca54530bdd7724ea60"
NAMESPACE = "namespace xgboost::metric::cuda_impl {"
PROBE_NAMESPACE = "namespace xgboost::metric::auc_probe {"
HOOK_PREFIX = "  ::auc_capture::"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def generate(source):
    if digest(source) != SOURCE_SHA256:
        raise ValueError("AUC source changed; review observation boundaries before updating the hash")
    original = source.decode("utf-8")
    text = original
    edits = []

    def replace(before, after, *, start=None, end=None):
        nonlocal text
        lo, hi = 0, len(text)
        if start is not None:
            if text.count(start) != 1 or text.count(end) != 1:
                raise ValueError("Ambiguous AUC function boundary")
            lo = text.index(start)
            hi = text.index(end, lo)
        section = text[lo:hi]
        if section.count(before) != 1:
            raise ValueError("Ambiguous AUC observation anchor")
        offset = lo + section.index(before)
        text = text[:offset] + after + text[offset + len(before):]
        edits.append(dict(offset=offset, before=before, after=after))

    replace(NAMESPACE, PROBE_NAMESPACE)
    replace("DMLC_REGISTRY_FILE_TAG(auc_gpu);", "// Test copy has no registry file tag.")
    binary = dict(start="std::tuple<double, double, double> BinaryAUC(",
                  end="std::tuple<double, double, double> BinaryROCAUC(")
    multi = dict(start="double MultiAUC(", end="void MultiSortedIdx(")
    binary_pr = dict(start="std::tuple<double, double, double> BinaryPRAUC(",
                     end="double MultiPRAUC(")
    multi_pr = dict(start="double MultiPRAUC(", end="std::pair<double, uint32_t> RankingPRAUCImpl(")
    normalization = dict(start="double ScaleOutputs(", end="void SegmentedFPTP(")

    anchor = "              [=] XGBOOST_DEVICE(size_t i) { d_fptp[i] = get_fp_tp(i); });"
    replace(anchor, anchor + "\n" + HOOK_PREFIX + "Terms(ctx, d_fptp, d_sorted_idx);", **binary)
    anchor = "                        d_fptp.size());"
    replace(anchor, anchor + "\n" + HOOK_PREFIX + "Prefixes(ctx, d_fptp);", **binary)
    anchor = "  dh::LaunchN(d_sorted_idx.size(), [=] XGBOOST_DEVICE(size_t i) { d_fptp[i] = get_fp_tp(i); });"
    replace(anchor, anchor + "\n" + HOOK_PREFIX + "Terms(ctx, d_fptp, d_sorted_idx);", **multi)
    anchor = "  SegmentedFPTP(ctx, d_fptp, get_output_id);"
    replace(anchor, anchor + "\n" + HOOK_PREFIX + "Prefixes(ctx, d_fptp);", **multi)
    anchor = "                     get_output_id, s_d_auc);"
    replace(anchor, anchor + "\n" + HOOK_PREFIX + "Curves(ctx, d_fptp, s_d_auc, n_samples);", **multi)
    anchor = "      ctx->CUDACtx()->CTP(), it, it + labels.Size(), Pair{0.0, 0.0}, PairPlus<double, double>{});"
    replace(anchor, anchor + "\n" + HOOK_PREFIX + "BinaryTotals(total_neg, total_pos);", **binary_pr)
    anchor = "                        PairPlus<double, double>{});"
    replace(anchor, anchor + "\n" + HOOK_PREFIX + "MultiTotals(ctx, dh::ToSpan(totals));", **multi_pr)
    anchor = "  auto reduce_in = dh::MakeTransformIterator<Pair>("
    replace(anchor, HOOK_PREFIX + "Normalization(ctx, local_area, output_weights, auc);\n" + anchor,
            **normalization)

    # Verify reversibility by recorded byte positions, not a broad regex that
    # might also erase an unintended arithmetic edit.
    restored = text
    for edit in reversed(edits):
        at, after = edit["offset"], edit["after"]
        if restored[at:at + len(after)] != after:
            raise ValueError("AUC edit record does not reconstruct the source")
        restored = restored[:at] + edit["before"] + restored[at + len(after):]
    if restored != original:
        raise ValueError("Instrumented AUC source contains unrecorded changes")
    generated = text.encode("utf-8")
    record = dict(source_sha256=digest(source), generated_sha256=digest(generated),
                  edits=edits, gpu_execution=False,
                  scope="Test-only namespace copy; eight readback hooks, no arithmetic edits. "
                        "Hooks add device synchronization and host copies when enabled.")
    return generated, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generated, record = generate(args.source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(generated)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Prepared {args.output}: eight observation hooks; source {record['source_sha256']}")


if __name__ == "__main__":
    main()
