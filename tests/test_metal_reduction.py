"""Integrated coverage of all scalar/vector root-reduction routes."""
import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb


@pytest.mark.parametrize("layout", ["dense", "missing", "sparse"])
@pytest.mark.parametrize("targets", [1, 3])
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_root_reductions_execute_on_gpu(metal, layout, targets, strategy):
    rng = np.random.default_rng(590)
    x = rng.normal(size=(4097, 9)).astype(np.float32)
    y = np.column_stack([x[:, 0] * (i + 1) + x[:, i + 1] for i in range(targets)])
    if layout == "missing":
        x[rng.random(x.shape) < 0.3] = np.nan
    elif layout == "sparse":
        x[rng.random(x.shape) < 0.7] = 0
        x = sp.csr_matrix(x)
    weights = rng.uniform(0.1, 3, x.shape[0]).astype(np.float32)
    weights[::13] = 0
    d = xgb.DMatrix(x, label=y, weight=weights)
    p = dict(tree_method="hist", nthread=2, multi_strategy=strategy, base_score=0,
             max_depth=4, max_bin=31, subsample=0.63, seed=29)
    cpu = xgb.train(p | {"device": "cpu"}, d, 5)
    before = xgb.metal_info()
    gpu = xgb.train(p | {"device": "mps"}, d, 5)
    after = xgb.metal_info()
    assert after["gradient_sum_dispatches"] - before["gradient_sum_dispatches"] == (
        5 * (targets if strategy == "one_output_per_tree" else 1)
    )
    assert after["stump_dispatches"] == before["stump_dispatches"]
    assert after["gradient_sum_gpu_time_ns"] > before["gradient_sum_gpu_time_ns"]
    np.testing.assert_allclose(gpu.predict(d), cpu.predict(d), atol=2e-5, rtol=2e-5)
