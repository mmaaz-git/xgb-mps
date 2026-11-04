"""Cache UCI benchmark inputs as FP32 arrays without expanding the large CSVs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd


DATASETS = {
    "higgs": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
        "citation": "Whiteson, D. (2014). HIGGS. https://doi.org/10.24432/C5V312",
        "rows": 11_000_000, "features": 28, "label_column": 0,
    },
    "covtype": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
        "citation": "Blackard, J. (1998). Covertype. https://doi.org/10.24432/C50K5N",
        "rows": 581_012, "features": 54, "label_column": 54,
    },
}


def prepare(name, cache):
    spec = DATASETS[name]
    dest = cache / name
    manifest = dest / "manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        for field, shape in [("X", (spec["rows"], spec["features"])), ("y", (spec["rows"],))]:
            array = np.load(dest / f"{field}.npy", mmap_mode="r")
            assert array.shape == shape and array.dtype == np.float32
        print(f"Using completed {name} cache: {manifest}", flush=True)
        return metadata
    archive = cache / f"{name}.csv.gz"
    cache.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        part = archive.with_suffix(archive.suffix + ".part")
        print(f"Downloading {spec['url']} to {part}", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "4", "--retry-delay", "10",
                        "--connect-timeout", "60", "--continue-at", "-", "--output", str(part),
                        spec["url"]], check=True)
        part.rename(archive)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Hashing {archive}", flush=True)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    x = np.lib.format.open_memmap(dest / "X.partial.npy", mode="w+", dtype=np.float32,
                                 shape=(spec["rows"], spec["features"]))
    y = np.lib.format.open_memmap(dest / "y.partial.npy", mode="w+", dtype=np.float32,
                                 shape=(spec["rows"],))
    start, begin = time.perf_counter(), 0
    for chunk in pd.read_csv(archive, header=None, dtype=np.float32, chunksize=100_000):
        end = begin + len(chunk)
        assert end <= spec["rows"] and chunk.shape[1] == spec["features"] + 1
        values = chunk.to_numpy()
        assert np.isfinite(values).all()
        y[begin:end] = values[:, spec["label_column"]]
        x[begin:end] = values[:, 1:] if name == "higgs" else values[:, :-1]
        begin = end
        print(f"{name}: {end:,}/{spec['rows']:,} rows, {time.perf_counter()-start:.1f}s", flush=True)
    assert begin == spec["rows"]
    assert np.isin(y, [0, 1] if name == "higgs" else np.arange(1, 8)).all()
    x.flush()
    y.flush()
    del x, y
    for field in ("X", "y"):
        (dest / f"{field}.partial.npy").replace(dest / f"{field}.npy")
    metadata = spec | {"name": name, "license": "CC BY 4.0", "dtype": "float32",
                       "archive_bytes": archive.stat().st_size, "archive_sha256": digest.hexdigest(),
                       "conversion_seconds": time.perf_counter() - start,
                       "preprocessing": "Original row order; header=None; labels unchanged; no scaling"}
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {manifest}", flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", choices=DATASETS, nargs="+")
    parser.add_argument("--cache", type=Path, default=Path(".cache/datasets"))
    args = parser.parse_args()
    for name in args.datasets:
        prepare(name, args.cache)
