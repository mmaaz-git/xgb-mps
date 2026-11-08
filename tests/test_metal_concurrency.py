"""Shared, initially host-resident metadata must survive concurrent training."""

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import xgboost as xgb


def train_shared(objective):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20000, 32)).astype(np.float32)
    y = x[:, 0] - x[:, 1]
    weight = rng.random(len(x)).astype(np.float32)
    params = dict(device="mps", seed=0, objective=objective)
    if objective == "multi:softprob":
        y = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75])).astype(np.float32)
        params["num_class"] = 4
    query = xgb.DMatrix(x[:100])
    reference = xgb.train(params, xgb.DMatrix(x, label=y, weight=weight), 5)
    expected = reference.predict(query)
    expected_model = reference.save_raw()
    before = xgb.metal_info()["histogram_dispatches"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for batch in range(20):
            # Every batch must exercise first use of fresh labels and weights.
            shared = xgb.DMatrix(x, label=y, weight=weight)
            start = threading.Barrier(8)

            def train(_):
                start.wait()
                model = xgb.train(params, shared, 5)
                return model.predict(query), model.save_raw()

            for prediction, model in pool.map(train, range(8)):
                np.testing.assert_array_equal(prediction, expected)
                assert model == expected_model, "Concurrent training changed the model"
            print(f"{objective}: batch {batch + 1}/20 passed", flush=True)
    assert xgb.metal_info()["histogram_dispatches"] > before


@pytest.mark.parametrize("objective", ["reg:squarederror", "multi:softprob"])
def test_shared_dmatrix_training(metal, objective):
    # A native regression can segfault; report it as a failed test rather than
    # losing the remainder of the Python suite. Use the package under test.
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(Path(xgb.__file__).resolve().parent.parent)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), objective],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    train_shared(sys.argv[1])
