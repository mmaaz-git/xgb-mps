"""Synthetic report integrity tests. These do not execute CUDA or establish parity."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("pipeline", Path(__file__).parents[1] / "scripts/compare_gpu_pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def reports():
    def fingerprint(shape):
        return {"shape": shape, "dtype": "<f4", "sha256": "0"*64}
    # A format-only HIGGS-shaped observation; no values were run on a GPU.
    case = {"name": "higgs", "inputs": {"x": fingerprint([10_500_000, 28]),
            "y": fingerprint([10_500_000]), "test": fingerprint([1024, 28])},
            "feature_types": ["q"]*28, "params": {"tree_method": "hist"}, "rounds": 1,
            "cuts": {"ptr": list(range(0, 85, 3)), "bits": [0xff800000, 0, 0x3f800000]*28},
            "model": {"base_score_bits": [0], "trees": [{"root": {"leaf_bits": 0}}]},
            "margins_bits": [0]*1024, "cpu_prediction_check": {"outside_tolerance": 0},
            "execution": {"device": "mps", "updater": "grow_quantile_histmaker", "metal_counters": {
                "cuda_sketch_batches": 5, "histogram_dispatches": 1,
                "prediction_dispatches": 1, "sketch_finalize_dispatches": 0}}}
    metal = {"schema_version": 1, "kind": "gpu-pipeline", "status": "completed", "device": "mps",
             "suite": "higgs-v1", "production_tree": "a"*40, "script_sha256": "b"*64, "cases": [case]}
    cuda = copy.deepcopy(metal)
    cuda["device"] = "cuda"
    cuda["cases"][0]["execution"] = {"device": "cuda:0", "updater": "grow_gpu_hist", "metal_counters": {}}
    return metal, cuda


def test_exact_and_changed_observations():
    metal, cuda = reports()
    assert pipeline.compare(metal, cuda)["status"] == "identical-on-recorded-cases"
    cuda["cases"][0]["cuts"]["bits"][11] = 1
    cuda["cases"][0]["model"]["trees"][0]["root"]["leaf_bits"] = 1
    result = pipeline.compare(metal, cuda)
    assert result["status"] == "differences-require-review"
    assert result["differences"][0]["features_with_different_cuts"] == [3]
    assert result["differences"][0]["first_tree_difference"]["path"] == "root"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(status="running"),
    lambda r: r["cases"].clear(),
    lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
    lambda r: r["cases"][0]["cuts"]["ptr"].__setitem__(-1, 999),
    lambda r: r["cases"][0]["margins_bits"].pop(),
    lambda r: r["cases"][0]["margins_bits"].__setitem__(0, 0x7fc00000),
    lambda r: r["cases"][0]["cuts"]["bits"].__setitem__(0, 0),
    lambda r: r["cases"][0]["model"]["trees"][0].clear(),
    lambda r: r["cases"][0]["execution"].update(device="cpu"),
    lambda r: r["cases"][0]["execution"]["metal_counters"].update(cuda_sketch_batches=0),
    lambda r: r["cases"][0]["cpu_prediction_check"].update(outside_tolerance=1),
    lambda r: r["cases"][0]["inputs"]["x"].update(shape=[100, 28]),
])
def test_incomplete_or_invalid_evidence_rejected(mutation):
    metal, cuda = reports()
    mutation(metal)
    with pytest.raises(ValueError):
        pipeline.compare(metal, cuda)


@pytest.mark.parametrize("field", ["production_tree", "script_sha256"])
def test_mismatched_source_rejected(field):
    metal, cuda = reports()
    cuda[field] = "c"*len(cuda[field])
    with pytest.raises(ValueError, match="Mismatched"):
        pipeline.compare(metal, cuda)


def test_mismatched_inputs_and_cuda_fallback_rejected():
    metal, cuda = reports()
    cuda["cases"][0]["inputs"]["x"]["sha256"] = "1"*64
    with pytest.raises(ValueError, match="mismatched inputs"):
        pipeline.compare(metal, cuda)
    metal, cuda = reports()
    cuda["cases"][0]["execution"]["updater"] = "grow_quantile_histmaker"
    with pytest.raises(ValueError, match="CUDA updater"):
        pipeline.compare(metal, cuda)


def test_truncated_split_and_unreachable_node_rejected():
    metal, cuda = reports()
    tree = metal["cases"][0]["model"]["trees"][0]
    tree["root"] = {"feature": 0, "default_left": 0, "split_type": 0, "threshold_bits": 0}
    with pytest.raises(ValueError, match="Missing tree path"):
        pipeline.compare(metal, cuda)
    metal, cuda = reports()
    metal["cases"][0]["model"]["trees"][0]["rootL"] = {"leaf_bits": 0}
    with pytest.raises(ValueError, match="Unreachable"):
        pipeline.compare(metal, cuda)


def test_infinite_split_boundary_preserved_but_not_nonfinite_leaf():
    metal, cuda = reports()
    for record in (metal, cuda):
        record["cases"][0]["model"]["trees"][0] = {
            "root": {"feature": 0, "default_left": 0, "split_type": 0, "threshold_bits": 0xff800000},
            "rootL": {"leaf_bits": 0}, "rootR": {"leaf_bits": 0}}
    assert pipeline.compare(metal, cuda)["status"] == "identical-on-recorded-cases"
    metal["cases"][0]["model"]["trees"][0]["rootL"]["leaf_bits"] = 0x7f800000
    with pytest.raises(ValueError, match="nonfinite"):
        pipeline.compare(metal, cuda)
