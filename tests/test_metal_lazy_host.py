"""CPU mirrors are demanded by CPU consumers, never by compressed GPU traversal."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from test_metal_host_cache import HostPages


HOST_COUNTERS = ("ellpack_restore_dispatches", "ellpack_host_materializations",
                 "ellpack_host_readback_bytes")


def assert_no_host(before):
    after = xgb.metal_info()
    for key in HOST_COUNTERS:
        assert after[key] == before[key], key


@pytest.mark.parametrize("layout", ["dense", "relative", "sparse", "categorical", "missing"])
@pytest.mark.parametrize("ratio", [0., .25, 1.])
def test_lazy_host_consumers(metal, tmp_path, layout, ratio):
    rng = np.random.default_rng(1801)
    x = rng.normal(size=(517, 5)).astype(np.float32)
    y = (np.arange(len(x)) % 7).astype(np.float32) / 4
    if layout != "dense":
        x[rng.random(x.shape) < (.8 if layout == "sparse" else .2)] = np.nan
        if layout == "relative":
            x[0] = 1
        if layout in {"sparse", "missing"}:
            x[:, 4] = np.nan
        if layout == "missing":
            x[:] = np.nan
    if layout == "categorical":
        x = pd.DataFrame(x, columns=list("abcde"))
        x.b = pd.Categorical.from_codes(np.arange(len(x)) % 7, categories=list("abcdefg"))
        x.loc[::13, "b"] = np.nan
    split = [0, 0, 137, 137, 312, 517, 517]
    chunks = [x.iloc[b:e] if isinstance(x, pd.DataFrame) else x[b:e]
              for b, e in zip(split[:-1], split[1:])]
    labels = [y[b:e] for b, e in zip(split[:-1], split[1:])]
    iterator = HostPages(tmp_path / "cache", chunks, 0, label=labels)
    before = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(iterator, device="mps", nthread=2, max_bin=17,
                                  cache_host_ratio=ratio)
    q.get_quantile_cut()  # Metadata-only access must not force a CPU mirror.
    assert_no_host(before)
    iterator.closed = True
    params = dict(device="mps", nthread=2, max_bin=17, max_depth=2, base_score=.5)
    model = xgb.train(params, q, 3)
    gpu_outputs = []
    options = [{"output_margin": True}, {"pred_leaf": True}, {"pred_contribs": True},
               {"pred_interactions": True}]
    for option in options:
        gpu_outputs.append(model.predict(q, strict_shape=True, **option))
    assert_no_host(before)

    cpu = xgb.Booster(model_file=model.save_raw())
    cpu.set_param(dict(device="cpu", nthread=2))
    for option, expected in zip(options, gpu_outputs):
        start = xgb.metal_info()
        actual = cpu.predict(q, strict_shape=True, **option)
        assert xgb.metal_info()["ellpack_host_materializations"] > start["ellpack_host_materializations"]
        if "pred_leaf" in option or "output_margin" in option:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        # Device switches reuse the cache but must never reuse host-only work.
        start = xgb.metal_info()
        fresh_gpu = xgb.Booster(model_file=model.save_raw())
        fresh_gpu.set_param(params)
        repeated = fresh_gpu.predict(q, strict_shape=True, **option)
        np.testing.assert_array_equal(repeated, expected)
        assert_no_host(start)

    # Approximate SHAP is a CPU fallback (upstream CUDA does not implement it).
    # It must explicitly acquire CPU batches even when device="mps" is selected.
    for option in [{"pred_contribs": True}, {"pred_interactions": True}]:
        start = xgb.metal_info()
        actual = model.predict(q, strict_shape=True, approx_contribs=True, **option)
        assert xgb.metal_info()["ellpack_host_materializations"] > start["ellpack_host_materializations"]
        expected = cpu.predict(q, strict_shape=True, approx_contribs=True, **option)
        np.testing.assert_array_equal(actual, expected)

    start = xgb.metal_info()
    cpu_matrix = xgb.QuantileDMatrix(x, y, ref=q, max_bin=17, nthread=2)
    assert_no_host(start)
    cpu_model = xgb.train({**params, "device": "cpu"}, q, 3)
    np.testing.assert_array_equal(cpu_model.predict(q), cpu_model.predict(cpu_matrix))
    # Retraining on GPU after CPU training still leaves every host mirror absent.
    start = xgb.metal_info()
    repeated_model = xgb.train(params, q, 3)
    np.testing.assert_array_equal(repeated_model.predict(q, output_margin=True, strict_shape=True),
                                  gpu_outputs[0])
    assert_no_host(start)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["dart", "multiclass", "vector"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("ratio", [0., 1.])
def test_lazy_host_other_boosters(metal, tmp_path, kind, sparse, ratio):
    rng = np.random.default_rng(1802)
    x = rng.normal(size=(271, 7)).astype(np.float32)
    if sparse:
        x[rng.random(x.shape) < .8] = np.nan
        x[:, 6] = np.nan
    y = (np.arange(len(x)) % 3).astype(np.float32)
    params = dict(device="mps", nthread=2, max_bin=17, max_depth=2)
    if kind == "dart":
        params.update(booster="dart", rate_drop=.5, one_drop=True)
    elif kind == "multiclass":
        params.update(objective="multi:softprob", num_class=3)
    else:
        y = np.column_stack([y, 2 - y])
        params.update(multi_strategy="multi_output_tree")
    iterator = HostPages(tmp_path / "cache", [x[:139], x[139:]], 0,
                         label=[y[:139], y[139:]])
    start = xgb.metal_info()
    q = xgb.ExtMemQuantileDMatrix(iterator, device="mps", nthread=2, max_bin=17,
                                  cache_host_ratio=ratio)
    iterator.closed = True
    model = xgb.train(params, q, 4)
    options = [{"output_margin": True}, {"pred_leaf": True}, {"pred_contribs": True},
               {"pred_interactions": True}]
    def predict(booster, option):
        # Upstream rejects strict_shape specifically for vector-leaf IDs.
        strict = not (kind == "vector" and "pred_leaf" in option)
        return booster.predict(q, strict_shape=strict, **option)
    expected = [predict(model, option) for option in options]
    assert_no_host(start)
    cpu = xgb.Booster(model_file=model.save_raw())
    cpu.set_param(dict(device="cpu", nthread=2))
    for option, reference in zip(options, expected):
        actual = predict(cpu, option)
        if "pred_leaf" in option or "output_margin" in option:
            np.testing.assert_array_equal(actual, reference)
        else:
            np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)
    assert not list(tmp_path.iterdir())
