"""Check coverage-map references and inventory closure, not semantic completeness."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "validation/cuda-operation-map.json"


def cuda_roots(revision):
    paths = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "xgboost/src",
            "xgboost/plugin",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return {path for path in paths if path.endswith((".cu", ".cuh"))}


def main():
    data = json.loads(MAP.read_text())
    baseline = cuda_roots(data["upstream_import"])
    current = cuda_roots("HEAD")
    mapped = set()
    ids = set()
    for group in data["groups"]:
        assert group["id"] not in ids, group["id"]
        ids.add(group["id"])
        assert group["operations"] and group["status"] and group["tests"], group["id"]
        mapped.update(group["cuda_sources"])
        for path in (
            group["cuda_sources"]
            + group["metal_dispatch_and_interfaces"]
            + group["tests"]
        ):
            assert (ROOT / path).is_file(), (group["id"], path)
    assert mapped == baseline, (
        "Baseline inventory mismatch",
        baseline - mapped,
        mapped - baseline,
    )
    assert mapped == current, (
        "Current inventory mismatch",
        current - mapped,
        mapped - current,
    )
    assert len(mapped) == data["source_files"]
    for path in (
        data["implementation_bodies"]
        + [data["latest_local_regression"]]
        + data.get("caller_audit_records", [])
    ):
        assert (ROOT / path).is_file(), path
    assert (
        data["completeness"] == "not_proven"
    ), "This structural check cannot certify parity."
    print(
        f"PASS {len(ids)} operation families / {len(mapped)} CUDA source/header roots; references exist"
    )
    print(
        "This verifies inventory/reference closure only, not GPU execution or full CUDA parity."
    )


if __name__ == "__main__":
    main()
