"""Public distributed training with actual Metal learners and row-partitioned data."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import scipy.sparse as sp
import xgboost as xgb
from xgboost import collective
from xgboost.objective import TreeObjective


class Reduced(TreeObjective):
    def __call__(self, iteration, prediction, data):
        # Upstream empty margin predictions do not carry their output width.
        pred = prediction.reshape(-1, 3)
        label = data.get_label().reshape(-1, 3)
        weights = data.get_weight()
        weight = weights[:, None] if weights.size else 1.
        return (pred - label) * weight, np.ones_like(pred) * weight

    def split_grad(self, iteration, grad, hess):
        return np.mean(grad, axis=1), np.mean(hess, axis=1)


def features(rows):
    r = np.arange(rows)
    x = np.column_stack([r % 8, r // 8 % 8, r * 3 % 7, (r + 1) % 4]).astype(np.float32)
    x[r % 11 == 0, 3] = np.nan
    return x


class Pages(xgb.DataIter):
    def __init__(self, x, y, weights, types, prefix, on_host):
        self.x, self.y, self.weights, self.types = x, y, weights, types
        self.bounds = [0, 0, len(x) // 3, len(x), len(x)]
        self.index = 0
        super().__init__(cache_prefix=prefix, on_host=on_host)

    def reset(self):
        self.index = 0

    def next(self, input_data):
        if self.index == len(self.bounds) - 1:
            return False
        start, end = self.bounds[self.index:self.index + 2]
        input_data(data=self.x[start:end], label=self.y[start:end],
                   weight=None if self.weights is None else self.weights[start:end],
                   feature_types=self.types)
        self.index += 1
        return True


def scenarios():
    # (objective family, matrix, rank layout, weighting, sampler, booster, method)
    for family in ("square", "binary", "multiclass", "vector", "reduced"):
        for layout in ("populated", "empty_root", "only_root"):
            yield family, "dmatrix", layout, "mixed", "none", "gbtree", "hist"
    for storage in ("quantile", "host", "disk", "sparse", "categorical", "native"):
        yield "square", storage, "empty_root", "weighted", "none", "gbtree", "hist"
        yield "reduced", storage, "populated", "mixed", "gradient", "gbtree", "hist"
    for sampler in ("none", "uniform", "gradient"):
        yield "square", "dmatrix", "empty_root", "weighted", sampler, "dart", "hist"
        yield "square", "dmatrix", "only_root", "weighted", sampler, "gbtree", "approx"
    for family in ("logistic", "logitraw", "hinge", "squaredlog", "gamma", "pseudohuber",
                   "poisson", "tweedie", "expectile", "quantile", "absolute", "softmax",
                   "aft_normal", "aft_logistic", "aft_extreme", "rank_ndcg", "rank_map",
                   "rank_pairwise"):
        for layout in ("populated", "empty_root"):
            yield family, "dmatrix", layout, "mixed", "none", "gbtree", "hist"


def pooled_numeric_fixture(family, storage, layout, weighting, world):
    """Independent CPU reference inputs for finite-bin, unsampled federated cases."""
    xs, ys, ws = [], [], []
    for rank in range(world):
        rows = 97 + rank * 11
        if layout == "empty_root" and rank == 0 or layout == "only_root" and rank != 0:
            rows = 0
        x = features(rows)
        y = np.where(x[:, 0] < 4, 0., 4.).astype(np.float32) + .25 * x[:, 1]
        if family in ("vector", "reduced"):
            y = np.column_stack([y, 1. - .5 * y, .25 * y + .5 * x[:, 2]])
        w = (.25 + np.arange(rows) % 5 / 4).astype(np.float32)
        if weighting == "mixed" and rank == 0:
            w = np.ones(rows, dtype=np.float32)
        xs.append(x)
        ys.append(y)
        ws.append(w)
    x = np.concatenate(xs)
    return (sp.csr_matrix(x) if storage == "sparse" else x), np.concatenate(ys), np.concatenate(ws)


def worker(config, directory):
    federated = config.get("dmlc_communicator") == "federated"
    pending_cpu = []
    with collective.CommunicatorContext(**config):
        rank, world = collective.get_rank(), collective.get_world_size()
        probe_x = features(64)
        checked = cpu_checks = 0
        for case, (family, storage, layout, weighting, sampler, booster, method) in enumerate(scenarios()):
            if world == 1 and layout == "empty_root":
                continue
            context = f"world{world}/rank{rank}/case{case}/{family}/{storage}/{layout}/{sampler}/{booster}/{method}"
            print(f"START {context}", flush=True)
            rows = 97 + rank * 11
            if layout == "empty_root" and rank == 0 or layout == "only_root" and rank != 0:
                rows = 0
            x = features(rows)
            y = np.where(x[:, 0] < 4, 0., 4.).astype(np.float32) + .25 * x[:, 1]
            params = dict(device="mps", nthread=2, tree_method=method, booster=booster,
                          max_depth=3, max_bin=16, eta=.3, seed=79, reg_lambda=1.25,
                          objective="reg:squarederror", eval_metric="rmse")
            objective = None
            if family in ("binary", "logistic", "logitraw", "hinge"):
                y = (x[:, 0] >= 4).astype(np.float32)
                params.update(objective="binary:logistic", eval_metric=["logloss", "auc", "aucpr"])
                if family == "logistic":
                    params.update(objective="reg:logistic")
                elif family == "logitraw":
                    params.update(objective="binary:logitraw", eval_metric="auc")
                elif family == "hinge":
                    params.update(objective="binary:hinge", eval_metric="error")
            elif family in ("multiclass", "softmax"):
                y = (x[:, 0] % 3).astype(np.float32)
                params.update(objective="multi:softprob", num_class=3,
                              eval_metric=["mlogloss", "auc", "aucpr"])
                if family == "softmax":
                    params.update(objective="multi:softmax")
            elif family in ("vector", "reduced"):
                y = np.column_stack([y, 1. - .5 * y, .25 * y + .5 * x[:, 2]])
                params.update(num_target=3, multi_strategy="multi_output_tree")
                if family == "reduced":
                    objective = Reduced()
            elif family in ("squaredlog", "gamma", "pseudohuber", "poisson", "tweedie",
                            "expectile", "quantile", "absolute"):
                objectives = dict(squaredlog="reg:squaredlogerror", gamma="reg:gamma",
                    pseudohuber="reg:pseudohubererror", poisson="count:poisson",
                    tweedie="reg:tweedie", expectile="reg:expectileerror",
                    quantile="reg:quantileerror", absolute="reg:absoluteerror")
                y += 1.
                params.update(objective=objectives[family])
                params.pop("eval_metric")
                if family == "expectile":
                    params.update(expectile_alpha=[.1, .5, .9])
                elif family == "quantile":
                    # CUDA averages local quantiles for automatic initialization;
                    # an empty worker contributes NaN. Supply a valid intercept.
                    params.update(quantile_alpha=[.1, .5, .9], base_score=1.)
            elif family.startswith("aft_"):
                y = np.exp(.15 * y).astype(np.float32)
                params.update(objective="survival:aft", aft_loss_distribution=family[4:],
                              eval_metric=["aft-nloglik", "interval-regression-accuracy"])
            elif family.startswith("rank_"):
                y = (x[:, 0] % (2 if family == "rank_map" else 4)).astype(np.float32)
                params.update(objective="rank:" + family[5:])
                params.pop("eval_metric")
            if sampler != "none":
                params.update(subsample=.7,
                              sampling_method="gradient_based" if sampler == "gradient" else "uniform")
            if booster == "dart":
                params.update(rate_drop=.3, skip_drop=0., one_drop=True)
            groups = np.diff(np.minimum(np.arange(0, rows + 8, 8), rows)).astype(np.uint32)
            weight_rows = len(groups) if family.startswith("rank_") else rows
            weights = (.25 + np.arange(weight_rows) % 5 / 4).astype(np.float32)
            if weighting == "mixed" and rank == 0:
                weights = None
            types = ["q", "q", "c" if storage == "categorical" else "q", "q"]
            source = sp.csr_matrix(x) if storage == "sparse" else x
            if storage == "native":
                import torch
                source = torch.tensor(x, device="mps")
            probe_source = sp.csr_matrix(probe_x) if storage == "sparse" else probe_x
            prefix = str(Path(directory) / f"train-{case}-{rank}") if storage in ("host", "disk") else None
            pages = None
            if prefix:
                pages = Pages(source, y, weights, types, prefix, storage == "host")
                data = xgb.ExtMemQuantileDMatrix(pages, device="mps", max_bin=16, nthread=2,
                                               enable_categorical=True)
            elif storage == "quantile":
                data = xgb.QuantileDMatrix(source, label=y, weight=weights, feature_types=types,
                    device="mps", max_bin=16, nthread=2, enable_categorical=True)
            else:
                data = xgb.DMatrix(source, label=y, weight=weights, feature_types=types,
                                   nthread=2, enable_categorical=True)
            if family.startswith("rank_"):
                data.set_group(groups)
            elif family.startswith("aft_"):
                lower, upper = y.copy(), y.copy()
                lower[1::5] = 0
                upper[2::5] = np.inf
                lower[3::5] *= .7
                upper[3::5] *= 1.4
                data.set_float_info("label_lower_bound", lower)
                data.set_float_info("label_upper_bound", upper)
            history = {}
            before = xgb.metal_info()
            model = xgb.train(params, data, 3, obj=objective, evals=[(data, "train")],
                              evals_result=history, verbose_eval=False)
            after = xgb.metal_info()
            assert after["quantized_gradient_sum_dispatches"] > before["quantized_gradient_sum_dispatches"], context
            assert after["histogram_integer_store_dispatches"] > before["histogram_integer_store_dispatches"], context
            if family == "reduced":
                assert after["leaf_weight_dispatches"] - before["leaf_weight_dispatches"] == 3, context
            raw = bytes(model.save_raw(raw_format="json"))
            root_raw = collective.broadcast(raw if rank == 0 else None, 0)
            state, root_state = json.loads(raw), json.loads(root_raw)
            # Upstream Booster._assign_dmatrix_features skips empty workers.
            # Validate that metadata rule separately; compare all learned state.
            assert state["learner"]["feature_types"] == (types if rows else []), context
            state["learner"].pop("feature_types")
            root_state["learner"].pop("feature_types")
            if state != root_state:
                Path(directory, f"model-{case}-{rank}.json").write_bytes(raw)
                Path(directory, f"root-{case}-{rank}.json").write_bytes(root_raw)
            assert state == root_state, context
            assert history == collective.broadcast(history if rank == 0 else None, 0), context
            assert all(np.isfinite(values).all() for values in history["train"].values()), context
            probe = xgb.DMatrix(probe_source, feature_types=types, nthread=2, enable_categorical=True)
            pred = model.predict(probe)
            np.testing.assert_array_equal(pred, collective.broadcast(pred if rank == 0 else None, 0), err_msg=context)
            assert np.isfinite(pred).all(), context
            # Exhaustive finite numeric bins and unsampled tree fixtures provide
            # a meaningful CPU comparison; CUDA sampler streams remain separate.
            if sampler == "none" and booster == "gbtree" and storage != "categorical" and family in ("square", "vector", "reduced"):
                if federated:
                    # Upstream CPU HostSketchContainer calls peer channels that
                    # FederatedComm rejects. Compare pooled CPU training after
                    # leaving the communicator. These numeric bins are exhaustive
                    # and sampling is off, so this remains a matching algorithm.
                    pending_cpu.append((params, family, storage, layout, weighting, types,
                                        pred.copy(), objective, context))
                else:
                    cpu_source = sp.csr_matrix(x) if storage == "sparse" else x
                    cpu_data = xgb.DMatrix(cpu_source, label=y, weight=weights, feature_types=types,
                                           nthread=2, enable_categorical=True)
                    cpu = xgb.train(params | {"device": "cpu"}, cpu_data, 3, obj=objective)
                    np.testing.assert_allclose(pred, cpu.predict(probe), atol=2e-5, rtol=2e-5, err_msg=context)
                    cpu_checks += pred.size
                    del cpu, cpu_data
            reloaded = xgb.Booster(model_file=bytearray(raw))
            reloaded.set_param(dict(device="mps", nthread=2))
            np.testing.assert_array_equal(reloaded.predict(probe), pred, err_msg=context)
            del data, model, reloaded, probe, pages
            gc.collect()
            if prefix:
                assert not list(Path(directory).glob(Path(prefix).name + "*")), context
            checked += 1
            print(f"PASS {context}: {checked} cases / {cpu_checks} CPU prediction checks", flush=True)
        data = xgb.DMatrix(features(8), label=np.arange(8, dtype=np.float32), nthread=2)
        with pytest.raises(xgb.core.XGBoostError, match="Distributed Metal gblinear"):
            xgb.train(dict(device="mps", booster="gblinear", nthread=2), data, 1)
    for params, family, storage, layout, weighting, types, pred, objective, context in pending_cpu:
        x, y, weights = pooled_numeric_fixture(family, storage, layout, weighting, world)
        cpu_data = xgb.DMatrix(x, label=y, weight=weights, feature_types=types, nthread=2)
        probe_source = sp.csr_matrix(probe_x) if storage == "sparse" else probe_x
        probe = xgb.DMatrix(probe_source, feature_types=types, nthread=2)
        cpu = xgb.train(params | {"device": "cpu"}, cpu_data, 3, obj=objective)
        np.testing.assert_allclose(pred, cpu.predict(probe), atol=2e-5, rtol=2e-5, err_msg=context)
        cpu_checks += pred.size
        print(f"PASS pooled CPU reference: {context}", flush=True)
    print(f"PASS world{world} rank{rank}: {checked} learner cases / {cpu_checks} CPU prediction checks", flush=True)


@pytest.mark.parametrize("world", [1, 2, 3])
def test_distributed_metal_training(metal, tmp_path, world):
    root = Path(__file__).resolve().parents[1]
    tracker = xgb.RabitTracker(n_workers=world, host_ip="127.0.0.1", timeout=180)
    tracker.start()
    workers, logs = [], []
    try:
        for task in range(world):
            config = dict(tracker.worker_args(), dmlc_task_id=str(task), dmlc_timeout=180)
            log = open(tmp_path / f"worker-{task}.log", "w+")
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable, __file__, json.dumps(config), str(tmp_path)],
                stdout=log, stderr=subprocess.STDOUT, cwd=root,
                env=dict(os.environ, MTL_DEBUG_LAYER="1")))
        deadline = time.monotonic() + 300
        while True:
            codes = [process.poll() for process in workers]
            assert all(code is None or code == 0 for code in codes), codes
            if all(code == 0 for code in codes):
                break
            assert time.monotonic() < deadline, codes
            time.sleep(.05)
        tracker.wait_for(timeout=180)
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
