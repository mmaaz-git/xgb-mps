"""Native metadata follows CUDA import/validation, without NumPy input conversion."""

import gc

import numpy as np
import pytest
from test_metal_extmem_adapter import DiskPages
from test_metal_adapter import Pages
from test_metal_tensor import producer
from cuda_sketch_reference import assert_cuts

import xgboost as xgb

FIELDS = [
    "label",
    "weight",
    "base_margin",
    "label_lower_bound",
    "label_upper_bound",
    "feature_weights",
]


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.int32, np.int64, np.uint8, np.bool_]
)
@pytest.mark.parametrize("layout", ["vector", "column", "slice", "broadcast"])
def test_native_metadata_values(metal, name, field, dtype, layout):
    values = (np.arange(274) % 19).astype(dtype)
    raw = producer(name, values)
    if layout == "column":
        raw, values = raw.reshape(274, 1), values.reshape(274, 1)
    elif layout == "slice":
        raw, values = raw[1::2], values[1::2]
    elif layout == "broadcast":
        if name == "torch":
            raw = raw[1:2].expand(274)
        else:
            import mlx.core as mx

            raw = mx.broadcast_to(raw[1:2], (274,))
        values = np.ones(274, dtype=dtype)
    matrix = xgb.DMatrix(np.zeros((values.size, 3)), nthread=2)
    before = xgb.metal_info()
    matrix.set_info(**{field: raw})
    after = xgb.metal_info()
    assert after["tensor_imports"] == before["tensor_imports"] + 1
    assert after["metadata_values"] == before["metadata_values"] + values.size
    assert after["metadata_copy_dispatches"] + after["metadata_convert_dispatches"] > (
        before["metadata_copy_dispatches"] + before["metadata_convert_dispatches"]
    )
    assert after["input_staging_bytes"] == before["input_staging_bytes"]
    assert after["input_debug_readback_bytes"] == before["input_debug_readback_bytes"]
    # The public getter squeezes single-column matrices, as for host metadata.
    expected = values.astype(np.float32).reshape(-1)
    np.testing.assert_array_equal(matrix.get_float_info(field).reshape(-1), expected)
    raw[:] = 0
    del raw
    gc.collect()
    np.testing.assert_array_equal(matrix.get_float_info(field).reshape(-1), expected)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("field", ["label", "base_margin"])
def test_metadata_matrix_capsules_and_shapes(metal, name, field):
    values = np.arange(63, dtype=np.float32).reshape(9, 7) / 4
    raw = producer(name, values).T
    data = producer(name, np.ones((7, 3), dtype=np.float32))
    kwargs = {field: raw.__dlpack__(max_version=(1, 0))}
    matrix = xgb.QuantileDMatrix(data, nthread=2, **kwargs)
    np.testing.assert_array_equal(matrix.get_float_info(field), values.T)
    # CUDA metadata preserves this two-dimensional shape, rather than applying
    # the CPU setter's inference from the matrix row count.
    matrix.set_info(**{field: producer(name, values)})
    assert matrix.get_float_info(field).shape == (9, 7)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("kind", ["ordinary", "set_info", "quantile", "external"])
@pytest.mark.parametrize("field", ["weight", "base_margin"])
@pytest.mark.parametrize("versioned", [False, True])
def test_aliased_metadata_capsule(metal, name, kind, field, versioned):
    values = np.arange(1, 14, dtype=np.float32)
    raw = producer(name, values)
    features = np.arange(39, dtype=np.float32).reshape(13, 3)
    exports = []

    def metadata():
        kwargs = {"max_version": (1, 0)} if versioned else {}
        capsule = raw.__dlpack__(**kwargs)
        exports.append(capsule)
        return {"label": capsule, field: capsule}

    class AliasedPages(xgb.DataIter):
        def __init__(self):
            self.used = False
            super().__init__(on_host=True, min_cache_page_bytes=0)

        def reset(self):
            self.used = False

        def next(self, input_data):
            if self.used:
                return False
            # Each callback exports once; both metadata fields share that export.
            # Independent callbacks must not reuse a consumed raw capsule.
            input_data(data=features, **metadata())
            self.used = True
            return True

    before = xgb.metal_info()
    if kind == "ordinary":
        matrix = xgb.DMatrix(features, nthread=2, **metadata())
    elif kind == "set_info":
        matrix = xgb.DMatrix(features, nthread=2)
        matrix.set_info(**metadata())
    elif kind == "quantile":
        matrix = xgb.QuantileDMatrix(
            features, device="mps", max_bin=17, nthread=2, **metadata()
        )
    else:
        iterator = AliasedPages()
        matrix = xgb.ExtMemQuantileDMatrix(
            iterator, device="mps", max_bin=17, nthread=2
        )
        del iterator
    after = xgb.metal_info()
    assert after["tensor_imports"] >= before["tensor_imports"] + 2
    # Feature input is deliberately a host array, so quantile construction can
    # stage it. Check the native metadata publication counters specifically.
    assert after["metadata_host_copy_bytes"] == before["metadata_host_copy_bytes"]
    assert (
        after["metadata_shared_output_bytes"] - before["metadata_shared_output_bytes"]
        >= 2 * values.nbytes
    )
    # Imported metadata owns its snapshot after the producer and callback die.
    raw[:] = 0
    del raw
    gc.collect()
    np.testing.assert_array_equal(matrix.get_label(), values)
    np.testing.assert_array_equal(matrix.get_float_info(field), values)
    for capsule in exports:
        with pytest.raises(ValueError, match="already been consumed"):
            matrix.set_info(label=capsule)
    np.testing.assert_array_equal(matrix.get_label(), values)


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("field", FIELDS)
def test_metadata_empty(metal, name, field):
    matrix = xgb.DMatrix(np.ones((0, 3)), nthread=2)
    matrix.set_info(**{field: producer(name, np.empty(0, dtype=np.float16))})
    assert matrix.get_float_info(field).size == 0


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_metadata_validation(metal, name):
    matrix = xgb.DMatrix(np.zeros((7, 3)), nthread=2)
    for field in ["label", "weight", "feature_weights"]:
        for bad in [np.nan, np.inf, -np.inf] + (
            [] if field == "label" else [-1.0, -1e-40]
        ):
            values = np.ones(7, dtype=np.float32)
            values[3] = bad
            with pytest.raises(
                xgb.core.XGBoostError,
                match="Label contains|Weights must|Feature weight",
            ):
                matrix.set_info(**{field: producer(name, values)})
        # Zero (including negative zero) is allowed by CUDA's actual predicate.
        matrix.set_info(
            **{field: producer(name, np.array([0.0, -0.0], dtype=np.float32))}
        )
        np.testing.assert_array_equal(
            matrix.get_float_info(field).view(np.uint32), [0, 0x80000000]
        )
    for field in ["base_margin", "label_lower_bound", "label_upper_bound"]:
        values = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
        matrix.set_info(**{field: producer(name, values)})
        np.testing.assert_array_equal(matrix.get_float_info(field), values)
    with pytest.raises(ValueError, match="Invalid shape"):
        matrix.set_weight(producer(name, np.ones((7, 2), dtype=np.float32)))


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float16, np.float32])
def test_qid_import(metal, name, dtype):
    ids = (np.arange(1031) // 11).astype(dtype)
    matrix = xgb.DMatrix(np.zeros((1031, 3)), nthread=2)
    before = xgb.metal_info()
    matrix.set_info(qid=producer(name, ids))
    after = xgb.metal_info()
    assert after["metadata_qid_dispatches"] > before["metadata_qid_dispatches"]
    np.testing.assert_array_equal(
        matrix.get_uint_info("group_ptr"), np.r_[np.arange(0, 1031, 11), 1031]
    )
    old = matrix.get_uint_info("group_ptr").copy()
    ids[5] = ids[-1]
    with pytest.raises(xgb.core.XGBoostError, match="qid.*sorted"):
        matrix.set_info(qid=producer(name, ids))
    np.testing.assert_array_equal(matrix.get_uint_info("group_ptr"), old)


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_group_sizes_and_qid_casts(metal, name):
    matrix = xgb.DMatrix(np.ones((7, 3)), nthread=2)
    for dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16]:
        matrix.set_group(producer(name, np.array([0, 3, 0, 4], dtype=dtype)))
        np.testing.assert_array_equal(
            matrix.get_uint_info("group_ptr"), [0, 0, 3, 3, 7]
        )
    with pytest.raises(xgb.core.XGBoostError, match="Expected integer"):
        matrix.set_group(producer(name, np.array([3, 4], dtype=np.float32)))
    matrix.set_info(
        qid=producer(name, np.array([2**32, 2**32, 2**32 + 1], dtype=np.int64))
    )
    np.testing.assert_array_equal(matrix.get_uint_info("group_ptr"), [0, 2, 3])
    matrix.set_info(
        qid=producer(name, np.array([-0.5, 0.1, 0.9, 1.9], dtype=np.float32))
    )
    np.testing.assert_array_equal(matrix.get_uint_info("group_ptr"), [0, 3, 4])
    for invalid in [np.nan, np.inf, -1.0, float(2**32)]:
        with pytest.raises(xgb.core.XGBoostError, match="uint32-representable"):
            matrix.set_info(qid=producer(name, np.array([invalid], dtype=np.float32)))
    for field in ["group", "qid"]:
        matrix.set_info(**{field: producer(name, np.array([], dtype=np.int32))})
        np.testing.assert_array_equal(matrix.get_uint_info("group_ptr"), [0])


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize(
    "kind", ["regression", "multiclass", "multitarget", "ranking", "survival"]
)
def test_gpu_metadata_training(metal, tmp_path, name, kind):
    rng = np.random.default_rng(713)
    values = rng.normal(size=(257, 7)).astype(np.float32)
    labels = (np.arange(257) % 3).astype(np.float32)
    params = dict(device="mps", nthread=2, max_depth=3, max_bin=17, base_score=0.5)
    meta = dict(
        label=labels,
        weight=np.ones(257, dtype=np.float32),
        base_margin=np.zeros(257, dtype=np.float32),
        feature_weights=np.arange(1, 8, dtype=np.float32),
    )
    if kind == "multiclass":
        params.update(objective="multi:softprob", num_class=3)
        meta["base_margin"] = np.zeros((257, 3), dtype=np.float32)
    elif kind == "multitarget":
        params.update(multi_strategy="multi_output_tree")
        meta["label"] = np.stack([labels, labels / 2], axis=1)
        meta["base_margin"] = np.zeros((257, 2), dtype=np.float32)
    elif kind == "ranking":
        params.update(objective="rank:pairwise")
        meta["qid"] = np.arange(257, dtype=np.int32) // 9
        meta["weight"] = np.ones(29, dtype=np.float32)
    elif kind == "survival":
        params.update(objective="survival:aft")
        meta.pop("label")
        meta["label_lower_bound"] = labels + 1
        meta["label_upper_bound"] = labels + 2
        meta["label_upper_bound"][::3] = np.inf
    raw = producer(name, values)
    native_meta = {key: producer(name, value) for key, value in meta.items()}
    q = xgb.QuantileDMatrix(raw, max_bin=17, nthread=2, **native_meta)
    host = xgb.QuantileDMatrix(values, max_bin=17, nthread=2, device="mps", **meta)
    actual = xgb.train(params, q, 4)
    expected = xgb.train(params, host, 4)
    np.testing.assert_array_equal(actual.predict(q), expected.predict(host))
    np.testing.assert_array_equal(
        actual.inplace_predict(raw, base_margin=native_meta["base_margin"]).numpy(),
        expected.inplace_predict(values, base_margin=meta["base_margin"]),
    )
    result = actual.predict(q)
    margin = actual.predict(q, output_margin=True)
    actual.set_param(dict(device="cpu"))
    np.testing.assert_array_equal(actual.predict(q, output_margin=True), margin)
    if kind == "multiclass":
        # CPU libm and the GPU softmax can differ by an FP32 ULP. Margins and
        # native/host-input Metal fits above remain exact comparisons.
        np.testing.assert_allclose(actual.predict(q), result, rtol=2e-7, atol=2e-8)
    else:
        np.testing.assert_array_equal(actual.predict(q), result)
    # The same imports also serve CPU training; compare whole fitted models
    # against host metadata to isolate import correctness from sketch differences.
    cpu_native = xgb.DMatrix(values, nthread=2, **native_meta)
    cpu_host = xgb.DMatrix(values, nthread=2, **meta)
    cpu_params = dict(params, device="cpu")
    fitted_native = xgb.train(cpu_params, cpu_native, 4)
    fitted_host = xgb.train(cpu_params, cpu_host, 4)
    assert fitted_native.save_raw() == fitted_host.save_raw()


@pytest.mark.parametrize("name", ["torch", "mlx"])
def test_metadata_scalars(metal, name):
    matrix = xgb.DMatrix(np.ones((1, 1)), nthread=2)
    for field in FIELDS:
        matrix.set_info(**{field: producer(name, np.array(3.5, dtype=np.float16))})
        np.testing.assert_array_equal(matrix.get_float_info(field), [3.5])
    matrix.set_info(qid=producer(name, np.array(7, dtype=np.int32)))
    np.testing.assert_array_equal(matrix.get_uint_info("group_ptr"), [0, 1])
    matrix.set_group(producer(name, np.array(3, dtype=np.int32)))
    np.testing.assert_array_equal(matrix.get_uint_info("group_ptr"), [0, 3])


@pytest.mark.parametrize("name", ["torch", "mlx"])
@pytest.mark.parametrize("constructor", ["memory", "host", "disk"])
@pytest.mark.parametrize("weight_kind", ["none", "row", "group", "zero"])
@pytest.mark.parametrize("input_kind", ["native", "mixed"])
def test_gpu_metadata_iterator(metal, tmp_path, name, constructor, weight_kind, input_kind):
    values = (np.arange(259 * 7).reshape(259, 7) % 31).astype(np.float32)
    labels = np.arange(259, dtype=np.float32) % 7
    split = [0, 0, 129, 129, 259, 259]
    slices = [slice(a, b) for a, b in zip(split[:-1], split[1:])]
    raw, y = (producer(name, v) for v in [values, labels])
    batches = [raw[s] for s in slices]
    if input_kind == "mixed":
        # Native weights also remain resident when the next data batch is on CPU.
        batches[3] = values[slices[3]]
    metadata = dict(label=[y[s] for s in slices])
    weights = np.zeros(259, dtype=np.float32) if weight_kind == "zero" else (
        np.arange(259, dtype=np.float32) % 9 / 4
    )
    expected_weights = np.array([], dtype=np.float32)
    if weight_kind == "group":
        groups = [np.array([n // 2, n - n // 2], dtype=np.uint32)
                  if n else np.array([], dtype=np.uint32) for n in np.diff(split)]
        group_weights = [np.array([0.5, 1.75], dtype=np.float32)
                         if len(g) else np.array([], dtype=np.float32) for g in groups]
        metadata.update(group=groups, weight=[producer(name, w) for w in group_weights])
        expected_weights = np.concatenate(group_weights)
    elif weight_kind != "none":
        w = producer(name, weights)
        metadata["weight"] = [w[s] for s in slices]
        expected_weights = weights
        del w

    def construct(prefix, ref=None):
        if constructor == "memory":
            it = Pages(batches, **metadata)
            result = xgb.QuantileDMatrix(it, ref=ref, max_bin=17, nthread=2)
        else:
            it = DiskPages(tmp_path / prefix, batches, on_host=constructor == "host", **metadata)
            result = xgb.ExtMemQuantileDMatrix(it, ref=ref, max_bin=17, nthread=2)
            it.closed = True
        return result

    before = xgb.metal_info()
    q = construct("native")
    after = xgb.metal_info()
    for key in ["vector_readback_bytes", "metadata_host_copy_bytes",
                "adapter_debug_readback_bytes"]:
        assert after[key] == before[key], key
    if input_kind == "native":
        assert after["input_staging_bytes"] == before["input_staging_bytes"]
    assert after["sketch_weight_dispatches"] - before["sketch_weight_dispatches"] == (
        2 if weight_kind == "group" else 0
    )
    assert after["gradient_state_shared_output_bytes"] - before["gradient_state_shared_output_bytes"] == (
        259 * 4 if weight_kind == "group" else 0
    )
    validation = construct("reference", ref=q)
    reused = xgb.metal_info()
    for key in ["vector_readback_bytes", "metadata_host_copy_bytes",
                "cuda_sketch_batches", "sketch_weight_dispatches", "gradient_state_shared_output_bytes",
                "gradient_state_staging_bytes"]:
        assert reused[key] == after[key], key
    if input_kind == "native":
        assert reused["input_staging_bytes"] == after["input_staging_bytes"]
    del raw, y, batches, metadata
    gc.collect()
    np.testing.assert_array_equal(q.get_label(), labels)
    np.testing.assert_array_equal(q.get_weight(), expected_weights)
    np.testing.assert_array_equal(validation.get_weight(), expected_weights)
    csr = xgb.DMatrix(values, nthread=2).get_data()
    assert_cuts(q, 17, csr=csr, page_rows=np.diff(split).tolist(), adapter=True)
    for actual, expected in zip(validation.get_quantile_cut(), q.get_quantile_cut()):
        np.testing.assert_array_equal(actual, expected)
    # A zero total weight has no automatic weighted-mean intercept.
    params = dict(device="mps", max_bin=17, nthread=2, max_depth=3, base_score=0.5,
                  objective="rank:pairwise" if weight_kind == "group" else "reg:squarederror")
    model = xgb.train(params, q, 3)
    np.testing.assert_array_equal(model.predict(q), model.inplace_predict(values))
    del model, q, validation
    gc.collect()
    assert not list(tmp_path.iterdir())
