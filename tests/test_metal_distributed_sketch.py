"""Distributed quantile constructors against an independent CUDA sketch replay."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xgboost as xgb
from xgboost import collective

import cuda_sketch_reference as oracle


class Input:
    """Host reference metadata; constructing this invokes no XGBoost operations."""
    def __init__(self, x, weights, types):
        rows, cols = np.nonzero(~np.isnan(x))
        self.csr = sp.csr_matrix((x[rows, cols], (rows, cols)), shape=x.shape)
        self.weights, self.feature_types = weights, types

    def get_data(self):
        return self.csr

    def get_weight(self):
        return self.weights

    def get_uint_info(self, _):
        return np.array([], dtype=np.uint32)

    def num_col(self):
        return self.csr.shape[1]

    def num_row(self):
        return self.csr.shape[0]


def fixture(rank, layout, weighted, categorical, streamed):
    rows = 197 + rank * 19
    if layout == 1 and rank == 0 or layout == 2 and rank != 0 or layout == 4 or layout == 5 and rank == 0:
        rows = 0
    r, f = np.indices((rows, 5))
    x = (((r * 127 + f * 31 + rank * 97) % 1009 - 504) / 8).astype(np.float32)
    if categorical:
        x[:, 1] = (np.arange(rows) * 11 + rank * 3) % 73
    x[(r + f) % 13 == 0] = np.nan
    x[:, 3] = np.nan
    if layout == 3:
        x[:] = np.nan
    weights = ((np.arange(rows) % 7 + 1) / 8).astype(np.float32) if weighted else np.array([], dtype=np.float32)
    types = ["q", "c" if categorical else "q", "q", "q", "q"]
    bounds = [0, 0, rows // 3, rows, rows] if streamed else [0, rows]
    return x, weights, types, bounds


def reference(world, layout, weighted, categorical, streamed, bins):
    states, total = [], 0
    for rank in range(world):
        x, weights, types, bounds = fixture(rank, layout, weighted, categorical, streamed)
        columns, seen = oracle.cuts(Input(x, weights, types), bins, page_rows=np.diff(bounds).tolist(),
                                   adapter=True, return_summary=True)
        states.append(columns)
        total += seen
    if world > 1:
        budget = oracle.budget(total, bins)

        def prune(columns):
            return [oracle.unique(oracle.prune(col, budget, types[f] == "c", index_unique=True))
                    for f, col in enumerate(columns)]

        states = [prune(columns) for columns in states]

        def tree(begin, count):
            if count == 1:
                return states[begin]
            split = 1
            while split * 2 < count:
                split *= 2
            left, right = tree(begin, split), tree(begin + split, count - split)
            return prune([oracle.merge(a, b) for a, b in zip(left, right)])

        columns = tree(0, world)
    else:
        columns = states[0]
    ptr, values = [0], []
    for f, col in enumerate(columns):
        if types[f] == "c":
            values.extend(range(int(col[-1, 3] if len(col) else 0) + 1))
        else:
            values.append(np.float32(-np.inf))
            values.extend(oracle.query(col, bins))
        ptr.append(len(values))
    return np.asarray(ptr, dtype=np.uint64), np.asarray(values, dtype=np.float32)


class Pages(xgb.DataIter):
    def __init__(self, batches, weights, types, prefix, on_host):
        self.batches, self.weights, self.types, self.index = batches, weights, types, 0
        super().__init__(cache_prefix=prefix, on_host=on_host)

    def reset(self):
        self.index = 0

    def next(self, input_data):
        if self.index == len(self.batches):
            return False
        input_data(data=self.batches[self.index], weight=self.weights[self.index], feature_types=self.types)
        self.index += 1
        return True


def worker(config, directory):
    with collective.CommunicatorContext(**config):
        rank, world = collective.get_rank(), collective.get_world_size()
        checked = 0
        for storage in ("array", "iterator", "host", "disk", "columnar", "native"):
            # Native input exercises both conversion ownership and the collective;
            # the ordinary formats cover all empty/schema/cache combinations.
            layouts = (0, 1, 5) if storage == "native" else range(6)
            for layout in layouts:
                if layout == 5 and world == 1:
                    continue  # Entire world without a schema is invalid, as in CUDA.
                for weighted, categorical in ((False, False), (True, True)):
                    streamed = storage in ("iterator", "host", "disk")
                    x, weight, types, bounds = fixture(rank, layout, weighted, categorical, streamed)
                    expected = reference(world, layout, weighted, categorical, streamed, 7)
                    unknown_schema = layout == 5 and rank == 0
                    if unknown_schema:
                        x = np.empty((0, 0), dtype=np.float32)
                        types = None
                    weights = [weight[a:b] if weighted else None for a, b in zip(bounds[:-1], bounds[1:])]
                    batches = [x[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
                    if storage == "columnar" and not unknown_schema:
                        batches = [pd.DataFrame(batch) for batch in batches]
                    # Upstream PandasTransformed and CudfTransformed both index
                    # columns[0]: a schema-less dataframe is not supported there.
                    # Use its equivalent empty dense input on that worker.
                    elif storage == "native":
                        import torch
                        batches = [torch.tensor(batch, device="mps") for batch in batches]
                    prefix = str(Path(directory) / f"rank{rank}-{storage}-{layout}-{weighted}") if storage in ("host", "disk") else None
                    args = dict(max_bin=7, device="mps", nthread=2, enable_categorical=True)
                    if streamed:
                        pages = Pages(batches, weights, types, prefix, storage != "disk")
                        kind = xgb.ExtMemQuantileDMatrix if prefix else xgb.QuantileDMatrix
                        q = kind(pages, **args)
                    else:
                        q = xgb.QuantileDMatrix(batches[0], weight=weights[0], feature_types=types, **args)
                    for actual, wanted in zip(q.get_quantile_cut(), expected):
                        np.testing.assert_array_equal(actual.view(np.uint32), wanted.view(np.uint32))
                    assert q.num_row() == len(x) and q.num_col() == 5
                    assert q.num_nonmissing() == np.count_nonzero(~np.isnan(x))
                    # A reference constructor must reuse global cuts without another sketch.
                    before = xgb.metal_info()
                    child = xgb.QuantileDMatrix(batches[0], feature_types=types, ref=q, **args)
                    assert xgb.metal_info()["cuda_sketch_batches"] == before["cuda_sketch_batches"]
                    for actual, wanted in zip(child.get_quantile_cut(), expected):
                        np.testing.assert_array_equal(actual.view(np.uint32), wanted.view(np.uint32))
                    del child, q
                    if streamed:
                        del pages
                    gc.collect()
                    if prefix:
                        assert not list(Path(directory).glob(Path(prefix).name + "*"))
                    checked += 1
            print(f"PASS rank{rank} world{world}: {storage}, {checked} quantile cases", flush=True)


@pytest.mark.parametrize("world", [1, 2, 3])
def test_distributed_quantile_construction(metal, tmp_path, world):
    root = Path(__file__).resolve().parents[1]
    tracker = xgb.RabitTracker(n_workers=world, host_ip="127.0.0.1", timeout=120)
    tracker.start()
    workers, logs = [], []
    try:
        for task in range(world):
            config = dict(tracker.worker_args(), dmlc_task_id=str(task), dmlc_timeout=120)
            log = open(tmp_path / f"worker-{task}.log", "w+")
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable, __file__, json.dumps(config), str(tmp_path)],
                                             stdout=log, stderr=subprocess.STDOUT, cwd=root,
                                             env=dict(os.environ, MTL_DEBUG_LAYER="1")))
        deadline = time.monotonic() + 240
        for process in workers:
            assert process.wait(timeout=max(1, deadline - time.monotonic())) == 0
        tracker.wait_for(timeout=120)
    finally:
        for process in workers:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.seek(0)
            print(log.read(), end="")
            log.close()


if __name__ == "__main__":
    worker(json.loads(sys.argv[1]), sys.argv[2])
