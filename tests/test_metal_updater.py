"""Explicit updater dispatch, update sequences and portable configuration loading."""

import copy
import json
import pickle

import numpy as np
import pytest

import xgboost as xgb


NAMES = {
    "hist": ("grow_quantile_histmaker", "grow_gpu_hist"),
    "approx": ("grow_histmaker", "grow_gpu_approx"),
}


def fixture(kind="scalar"):
    rng = np.random.default_rng(851)
    x = rng.integers(0, 13, (193, 4)).astype(np.float32)
    y = 2 * x[:, 0] - x[:, 1] + x[:, 2] / 2
    x[::11, 1] = np.nan
    types = ["c", "q", "q", "q"] if kind == "category" else None
    params = dict(
        nthread=2,
        max_bin=17,
        max_depth=3,
        base_score=0.5,
        seed=91,
        max_cached_hist_node=3,
    )
    if kind == "vector":
        y = np.column_stack([y, -y, y / 2])
        params["multi_strategy"] = "multi_output_tree"
    return xgb.DMatrix(x, label=y, feature_types=types, nthread=2), params


def config(model):
    return json.loads(model.save_config())["learner"]["gradient_booster"]


def assert_cpu_prediction(model, data):
    cpu = xgb.Booster(
        params={"device": "cpu", "nthread": 2}, model_file=model.save_raw()
    )
    np.testing.assert_allclose(
        model.predict(data), cpu.predict(data), atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize(
    "method,kind",
    [
        ("hist", "scalar"),
        ("hist", "category"),
        ("hist", "vector"),
        ("approx", "scalar"),
        ("approx", "category"),
    ],
)
@pytest.mark.parametrize("spelling", [0, 1])
@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("booster", ["gbtree", "dart"])
def test_explicit_growth(metal, method, kind, spelling, sample, booster):
    data, params = fixture(kind)
    params.update(device="mps", tree_method=method, booster=booster)
    if booster == "dart":
        params.update(rate_drop=0.3, one_drop=True)
    if sample:
        params.update(sampling_method="gradient_based", subsample=0.7)
    expected = xgb.train(params, data, 3)
    # Manual selection takes precedence over tree_method. The vector-tree front
    # end still requires hist/auto upstream, independently of updater selection.
    manual = params | dict(
        updater=NAMES[method][spelling],
        tree_method="hist" if kind == "vector" else "exact",
    )
    before = xgb.metal_info()
    actual = xgb.train(manual, data, 3)
    assert xgb.metal_info()["histogram_dispatches"] > before["histogram_dispatches"]
    assert actual.save_raw() == expected.save_raw()
    np.testing.assert_array_equal(actual.predict(data), expected.predict(data))
    saved = config(actual)
    assert saved["specified_updater"]
    assert saved["gbtree_train_param"]["updater_seq"] == NAMES[method][0]
    assert [up["name"] for up in saved["updater"]] == [NAMES[method][0]]
    assert_cpu_prediction(actual, data)


@pytest.mark.parametrize("sequence", ["prune", "refresh", "refresh,prune"])
@pytest.mark.parametrize("refresh_leaf", [False, True])
@pytest.mark.parametrize("booster", ["gbtree", "dart"])
def test_existing_tree_updates(metal, sequence, refresh_leaf, booster):
    data, params = fixture()
    params.update(device="mps", tree_method="hist", booster=booster)
    base = xgb.train(params, data, 3).save_raw()
    results = []
    for device in ["cpu", "mps"]:
        options = params | dict(
            device=device,
            updater=sequence,
            process_type="update",
            refresh_leaf=refresh_leaf,
            gamma=12,
        )
        results.append(xgb.train(options, data, 3, xgb_model=base))
    cpu, gpu = results
    assert gpu.save_raw() == cpu.save_raw()
    np.testing.assert_array_equal(gpu.predict(data), cpu.predict(data))
    restored = xgb.Booster(
        params={"device": "mps", "nthread": 2}, model_file=gpu.save_raw()
    )
    np.testing.assert_array_equal(restored.predict(data), gpu.predict(data))
    assert restored.num_boosted_rounds() == 3


@pytest.mark.parametrize("method", ["hist", "approx"])
@pytest.mark.parametrize("sequence", ["prune", "prune,refresh"])
def test_growth_sequence(metal, method, sequence):
    data, params = fixture()
    models = []
    for name in NAMES[method]:
        before = xgb.metal_info()
        models.append(
            xgb.train(
                params | dict(device="mps", updater=f"{name},{sequence}"), data, 3
            )
        )
        assert xgb.metal_info()["histogram_dispatches"] > before["histogram_dispatches"]
    assert models[0].save_raw() == models[1].save_raw()
    assert [up["name"] for up in config(models[0])["updater"]] == [
        NAMES[method][0],
        *sequence.split(","),
    ]
    assert_cpu_prediction(models[0], data)


@pytest.mark.parametrize("method", ["hist", "approx"])
@pytest.mark.parametrize("booster", ["gbtree", "dart"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_cuda_named_configuration(request, method, booster, legacy, device):
    if device == "mps":
        request.getfixturevalue("metal")
    data, params = fixture()
    options = params | dict(
        device=device,
        tree_method=method,
        booster=booster,
        updater=NAMES[method][device == "mps"],
    )
    source = xgb.train(options, data, 2)
    saved = json.loads(source.save_config())
    saved["learner"]["generic_param"]["device"] = device
    native = copy.deepcopy(saved)
    gb = saved["learner"]["gradient_booster"]
    # Source-defined CUDA configuration schema, synthesized on this non-CUDA
    # machine. This tests parsing/dispatch, not execution equivalence to CUDA.
    gb["updater"][0]["name"] = NAMES[method][1]
    gb["gbtree_train_param"].update(
        updater=NAMES[method][1], updater_seq=NAMES[method][1]
    )
    if legacy:
        gb["updater"] = {up["name"]: up for up in gb["updater"]}
    resumed = []
    for settings in [native, saved]:
        model = xgb.Booster(model_file=source.save_raw())
        model.load_config(json.dumps(settings))
        assert config(model)["specified_updater"]
        assert config(model)["gbtree_train_param"]["updater_seq"] == NAMES[method][0]
        before = xgb.metal_info()
        model.update(data, 2)
        assert (
            xgb.metal_info()["histogram_dispatches"] > before["histogram_dispatches"]
        ) == (device == "mps")
        up = config(model)["updater"][0]
        assert up["name"] == NAMES[method][0]
        assert up["hist_train_param"]["max_cached_hist_node"] == "3"
        resumed.append(model)
    assert resumed[0].save_raw() == resumed[1].save_raw()
    np.testing.assert_array_equal(resumed[0].predict(data), resumed[1].predict(data))
    clone = pickle.loads(pickle.dumps(resumed[1]))
    np.testing.assert_array_equal(clone.predict(data), resumed[1].predict(data))
    assert_cpu_prediction(clone, data)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_manual_selection_survives_reset_and_slice(request, device):
    if device == "mps":
        request.getfixturevalue("metal")
    data, params = fixture()
    # A different automatic method makes silently losing the flag observable.
    model = xgb.train(
        params
        | dict(device=device, tree_method="hist", updater="grow_histmaker,prune"),
        data,
        3,
    )
    expected = ["grow_histmaker", "prune"]
    for copy_model in [
        model,
        model[1:3],
        pickle.loads(pickle.dumps(model)),
        model.copy(),
    ]:
        saved = config(copy_model)
        assert saved["specified_updater"]
        assert [up["name"] for up in saved["updater"]] == expected
        copy_model.reset()
        assert [up["name"] for up in config(copy_model)["updater"]] == expected
        assert_cpu_prediction(copy_model, data)


@pytest.mark.parametrize("method", ["hist", "approx"])
def test_common_updater_device_transitions(metal, method):
    data, params = fixture()
    model = xgb.Booster(
        params | dict(device="cpu", tree_method=method, updater=NAMES[method][0]),
        cache=[data],
    )
    # Compare a reused updater with a model/config roundtrip that discards its
    # workspaces. Use the common spelling, which is valid on both backends.
    for iteration, device in enumerate(["cpu", "mps", "cpu", "mps"]):
        model.set_param({"device": device})
        reference = xgb.Booster(model_file=model.save_raw())
        reference.load_config(model.save_config())
        reference.set_param({"device": device, "updater": NAMES[method][0]})
        before = xgb.metal_info()
        model.update(data, iteration)
        after = xgb.metal_info()
        assert (after["histogram_dispatches"] > before["histogram_dispatches"]) == (
            device == "mps"
        )
        reference.update(data, iteration)
        assert model.save_raw() == reference.save_raw()
        assert_cpu_prediction(model, data)


@pytest.mark.parametrize(
    "updater,process,message",
    [
        ("missing_updater", "default", "Unknown tree updater"),
        ("prune", "default", "can not be used to create new trees"),
        ("refresh", "default", "can not be used to create new trees"),
        ("grow_gpu_hist", "update", "can not be used to modify existing trees"),
        ("grow_gpu_approx", "update", "can not be used to modify existing trees"),
    ],
)
def test_updater_errors(metal, updater, process, message):
    data, params = fixture()
    base = xgb.train(params | dict(device="mps"), data, 1)
    with pytest.raises(xgb.core.XGBoostError, match=message):
        xgb.train(
            params | dict(device="mps", updater=updater, process_type=process),
            data,
            1,
            xgb_model=base,
        )
