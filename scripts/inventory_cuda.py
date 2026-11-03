"""Regenerate the CUDA file ledger; file extensions do not imply feature parity."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "xgboost/src"
files = sorted(str(p.relative_to(source)) for p in source.rglob("*") if p.suffix in {".cu", ".cuh"})
report = dict(upstream_version="3.4.2", scope="All .cu/.cuh files in xgboost/src; not a complete symbol audit",
              count=len(files), files=files)
output = root / "validation/cuda-inventory.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(f"Wrote {len(files)} CUDA source/header paths to {output}")
