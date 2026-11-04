"""Public metadata imports keep owned GPU state until explicit host access."""

import gc

import numpy as np
import pytest
from test_metal_tensor import producer

import xgboost as xgb


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize(
    "field",
    [
        "label",
        "weight",
        "base_margin",
        "label_lower_bound",
        "label_upper_bound",
        "feature_weights",
    ],
)
def test_lazy_metadata_mirrors(metal, name, field):
    gc.collect()
    initial = xgb.metal_info()
    matrix = xgb.DMatrix(np.zeros((257, 4), dtype=np.float32), nthread=2)
    shape = (
        (4,)
        if field == "feature_weights"
        else (257, 3)
        if field == "base_margin"
        else (257,)
    )
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 16 + 1
    before = xgb.metal_info()
    matrix.set_float_info(field, producer(name, values))
    after = xgb.metal_info()
    assert after["vector_allocations"] == before["vector_allocations"] + 1
    assert after["vector_readback_bytes"] == before["vector_readback_bytes"]
    assert after["metadata_host_copy_bytes"] == before["metadata_host_copy_bytes"]
    assert (
        after["metadata_shared_output_bytes"] - before["metadata_shared_output_bytes"]
        == values.nbytes
    )
    np.testing.assert_array_equal(matrix.get_float_info(field), values)
    read = xgb.metal_info()
    assert (
        read["vector_readback_bytes"] - after["vector_readback_bytes"] == values.nbytes
    )
    np.testing.assert_array_equal(matrix.get_float_info(field), values)
    assert xgb.metal_info()["vector_readback_bytes"] == read["vector_readback_bytes"]

    # Same shape reuses the allocation and invalidates just the old host mirror.
    values += 3
    before = xgb.metal_info()
    matrix.set_float_info(field, producer(name, values))
    after = xgb.metal_info()
    for key in ("vector_allocations", "vector_readback_bytes", "vector_upload_bytes"):
        assert after[key] == before[key], key
    np.testing.assert_array_equal(matrix.get_float_info(field), values)
    assert (
        xgb.metal_info()["vector_readback_bytes"] - after["vector_readback_bytes"]
        == values.nbytes
    )

    # Mutable host setters and empty GPU replacements must invalidate stale data.
    matrix.set_float_info(field, np.full(shape, 2, dtype=np.float32))
    np.testing.assert_array_equal(
        matrix.get_float_info(field), np.full(values.shape, 2)
    )
    matrix.set_float_info(field, producer(name, np.empty(0, dtype=np.float32)))
    assert matrix.get_float_info(field).size == 0
    matrix.set_float_info(field, producer(name, values))
    np.testing.assert_array_equal(matrix.get_float_info(field), values)
    del matrix
    gc.collect()
    final = xgb.metal_info()
    assert final["vector_live_buffers"] == initial["vector_live_buffers"]
    assert final["vector_capacity_bytes"] == initial["vector_capacity_bytes"]


class MetadataBatches(xgb.DataIter):
    def __init__(self, name, x, y, margin, weight, mixed):
        super().__init__(release_data=True)
        self.name, self.x, self.y = name, x, y
        self.margin, self.weight, self.mixed = margin, weight, mixed
        self.at = 0
        self.offsets = [0, 0, 31, 31, 129, 257]

    def reset(self):
        self.at = 0

    def next(self, input_data):
        if self.at + 1 == len(self.offsets):
            return False
        begin, end = self.offsets[self.at : self.at + 2]
        native = self.name is not None and (not self.mixed or self.at % 2 == 0)

        def value(data):
            part = data[begin:end]
            return producer(self.name, part) if native else part

        input_data(
            data=self.x[begin:end],
            label=value(self.y),
            base_margin=value(self.margin),
            weight=value(self.weight),
        )
        self.at += 1
        return True


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_metadata_stack_copy_and_training(metal, name, mixed, device, tmp_path):
    rng = np.random.default_rng(651)
    x = rng.normal(size=(257, 4)).astype(np.float32)
    y = np.column_stack([x[:, 0], -x[:, 1], x[:, 2] / 2])
    margin = rng.normal(size=y.shape).astype(np.float32)
    weight = rng.uniform(0.1, 3, len(x)).astype(np.float32)
    before = xgb.metal_info()
    matrix = xgb.DMatrix(MetadataBatches(name, x, y, margin, weight, mixed), nthread=2)
    assert xgb.metal_info()["vector_copy_dispatches"] > before["vector_copy_dispatches"]
    for key, value in [("label", y), ("base_margin", margin), ("weight", weight)]:
        np.testing.assert_array_equal(matrix.get_float_info(key), value)
    # Keep feature-page boundaries identical: CUDA-derived weighted sketches can
    # produce different cuts for one matrix versus incrementally merged pages.
    host = xgb.DMatrix(MetadataBatches(None, x, y, margin, weight, False), nthread=2)
    params = dict(
        device=device,
        tree_method="hist",
        nthread=2,
        base_score=0.5,
        max_depth=3,
        max_bin=17,
        multi_strategy="multi_output_tree",
    )
    a, b = xgb.train(params, matrix, 3), xgb.train(params, host, 3)
    np.testing.assert_array_equal(a.save_raw(), b.save_raw())
    np.testing.assert_array_equal(a.predict(matrix), b.predict(host))
    # Public DMatrix slicing intentionally remains CPU upstream; storage origins
    # must not change its row ordering, repeated-index or serialization behavior.
    rows = np.array([256, 0, 17, 17, 1], dtype=np.int32)
    with pytest.raises(
        xgb.core.XGBoostError, match="not supported for external memory"
    ):
        matrix.slice(rows)
    resident = xgb.DMatrix(
        x,
        label=producer(name, y),
        base_margin=producer(name, margin),
        weight=producer(name, weight),
        nthread=2,
    )
    sliced = resident.slice(rows)
    np.testing.assert_array_equal(sliced.get_label(), y[rows])
    path = tmp_path / "matrix.buffer"
    resident.save_binary(path)
    loaded = xgb.DMatrix(path, nthread=2)
    np.testing.assert_array_equal(a.predict(loaded), a.predict(resident))
