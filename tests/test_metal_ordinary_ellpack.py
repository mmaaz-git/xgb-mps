"""Ordinary matrices keep quantized GPU data compressed across public consumers."""
import numpy as np
import pytest
import xgboost as xgb

from test_metal_inmemory_ellpack import data, assert_export
from test_metal_lazy_host import assert_no_host


@pytest.mark.parametrize("layout", ["dense", "relative", "sparse", "categorical", "missing"])
@pytest.mark.parametrize("method,kind", [
    ("hist", "scalar"), ("approx", "scalar"), ("approx_variable", "scalar"),
    ("hist", "vector"),  # Upstream permits vector leaves only with hist.
    ("hist", "dart"), ("approx", "dart"), ("approx_variable", "dart"),
])
def test_ordinary_compressed_consumers(metal, layout, method, kind):
    x, y = data(layout)
    if kind == "vector":
        y = np.column_stack([y, y * -.7 + .1, y * 2 + .5])
    weight = (np.arange(len(x)) % 9).astype(np.float32) / 4
    matrix = xgb.DMatrix(x, y, weight=weight, nthread=2)
    original = matrix.get_data()
    params = dict(device="mps", tree_method="hist" if method == "hist" else "approx", nthread=2, max_bin=17, max_depth=3, base_score=.5)
    if method == "approx_variable":
        params["objective"] = "reg:pseudohubererror"
    if kind == "vector":
        params["multi_strategy"] = "multi_output_tree"
    if kind == "dart":
        params.update(booster="dart", rate_drop=.3)
    before = xgb.metal_info()
    model = xgb.train(params, matrix, 3)
    trained = xgb.metal_info()
    generations = 3 if method == "approx_variable" else 1
    assert trained["ellpack_view_pages"] - before["ellpack_view_pages"] == generations
    assert trained["ellpack_csr_values"] - before["ellpack_csr_values"] == matrix.num_nonmissing() * generations
    assert trained["ellpack_histogram_dispatches"] > before["ellpack_histogram_dispatches"]
    for key in ["binning_dispatches", "index_uploads", "ellpack_pack_input_bytes"]:
        assert trained[key] == before[key], key
    assert_no_host(before)
    cuts = matrix.get_quantile_cut()
    if method == "hist":
        reference = xgb.QuantileDMatrix(x, y, ref=matrix, device="cpu", max_bin=17, nthread=2)
        for actual, expected in zip(reference.get_quantile_cut(), cuts):
            np.testing.assert_array_equal(actual, expected)
    else:
        # An approx cache has a NaN sparse threshold; the existing fixed-reference
        # contract rejects a hist QDM's different threshold. Metadata access itself
        # must remain read-only even on this rejected conversion.
        with pytest.raises(xgb.core.XGBoostError, match="Inconsistent sparse threshold"):
            xgb.QuantileDMatrix(x, y, ref=matrix, device="cpu", max_bin=17, nthread=2)
    assert_no_host(before)
    # Ordinary get_data exports original CSR, never a quantized reconstruction.
    assert_export(matrix.get_data(), original)
    assert_no_host(before)
    cpu = xgb.Booster(model_file=model.save_raw())
    cpu.set_param(dict(device="cpu", nthread=2))
    for option in [dict(output_margin=True), dict(pred_leaf=True), dict(pred_contribs=True),
                   dict(pred_interactions=True), dict(pred_contribs=True, approx_contribs=True)]:
        strict = not (kind == "vector" and "pred_leaf" in option)
        if kind == "vector" and option.get("approx_contribs"):
            # Upstream CPU and CUDA do not implement approximate vector SHAP.
            for candidate in [cpu, model]:
                with pytest.raises(xgb.core.XGBoostError, match="multi-target tree"):
                    candidate.predict(matrix, strict_shape=strict, **option)
            continue
        expected = cpu.predict(matrix, strict_shape=strict, **option)
        actual = model.predict(matrix, strict_shape=strict, **option)
        if "pred_leaf" in option or "output_margin" in option or option.get("approx_contribs"):
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert_no_host(before)  # CPU ordinary prediction uses original raw CSR too.
    reloaded = xgb.Booster(model_file=model.save_raw())
    reloaded.set_param(dict(device="mps", nthread=2))
    np.testing.assert_array_equal(reloaded.predict(matrix), model.predict(matrix))
    if method == "hist":
        before_repeat = xgb.metal_info()
        repeated = xgb.train(params, matrix, 3)
        np.testing.assert_array_equal(repeated.predict(matrix), model.predict(matrix))
        for key in ["ellpack_csr_dispatches", "ellpack_view_pages", "ellpack_pack_input_bytes",
                    "ellpack_host_materializations"]:
            assert xgb.metal_info()[key] == before_repeat[key], key


@pytest.mark.parametrize("layout", ["dense", "relative", "sparse", "categorical", "missing"])
def test_ordinary_cache_generation_and_fixed_reference(metal, layout):
    x, y = data(layout)
    matrix = xgb.DMatrix(x, y, nthread=2)
    before = xgb.metal_info()
    for bins in [17, 33, 17]:
        params = dict(nthread=2, max_bin=bins, max_depth=2, base_score=.5)
        xgb.train({**params, "device": "cpu"}, matrix, 2)
        cpu_cuts = matrix.get_quantile_cut()
        checkpoint = xgb.metal_info()
        # An MPS child requesting CPU reference cuts must not compress the reference.
        child = xgb.QuantileDMatrix(x, y, ref=matrix, device="mps", nthread=2, max_bin=bins)
        for a, b in zip(child.get_quantile_cut(), cpu_cuts):
            np.testing.assert_array_equal(a, b)
        assert xgb.metal_info()["ellpack_pack_input_bytes"] == checkpoint["ellpack_pack_input_bytes"]
        checkpoint = xgb.metal_info()
        gpu = xgb.train({**params, "device": "mps"}, matrix, 2)
        assert xgb.metal_info()["ellpack_view_pages"] - checkpoint["ellpack_view_pages"] == 1
        assert xgb.metal_info()["ellpack_csr_values"] - checkpoint["ellpack_csr_values"] == matrix.num_nonmissing()
        fixed = xgb.QuantileDMatrix(x, y, ref=matrix, nthread=2, max_bin=bins)
        oracle = xgb.train({**params, "device": "cpu"}, fixed, 2)
        np.testing.assert_allclose(gpu.predict(matrix), oracle.predict(fixed), rtol=2e-5, atol=2e-5)
        cuts = matrix.get_quantile_cut()
        checkpoint = xgb.metal_info()
        for device in ["cpu", "mps"]:
            with pytest.raises(xgb.core.XGBoostError, match="max_bin"):
                xgb.QuantileDMatrix(x, y, ref=matrix, device=device, nthread=2, max_bin=bins + 1)
        for a, b in zip(matrix.get_quantile_cut(), cuts):
            np.testing.assert_array_equal(a, b)
        assert xgb.metal_info()["ellpack_csr_dispatches"] == checkpoint["ellpack_csr_dispatches"]
        assert_no_host(before)
    assert xgb.metal_info()["ellpack_pack_input_bytes"] == before["ellpack_pack_input_bytes"]
