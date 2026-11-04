"""Host oracle for XGBoost's CUDA Philox row masks; no NVIDIA execution implied.

A CPU booster can check downstream tree/leaf arithmetic when supplied the CUDA
mask explicitly. This intentionally does not ask CPU subsampling to use its own
unrelated RNG. Tests using this helper disable column sampling so the documented
per-tree seed consumption is unambiguous.
"""
from functools import lru_cache

import numpy as np
from xgboost.objective import TreeObjective


@lru_cache(maxsize=128)
def weights(seed, rows):
    mask = (1 << 64) - 1
    draws = []
    for block in range((rows + 3) // 4):
        x = [block, 0, 0, 0]
        key = [int(seed) & 0xffffffff, 0]
        for _ in range(10):
            a, b = x[2] * 0xca5a826395121157, x[0] * 0xd2e7470ee14c6c93
            x = [(a >> 64) ^ x[1] ^ key[0], a & mask,
                 (b >> 64) ^ x[3] ^ key[1], b & mask]
            key = [(key[0] + 0x9e3779b97f4a7c15) & mask,
                   (key[1] + 0xbb67ae8584caa73b) & mask]
        # Round directly from integer to float32, without a double-rounding
        # intermediate at 24-bit midpoints.
        for word in x:
            shift = max(0, word.bit_length() - 24)
            value = word >> shift
            if shift:
                remainder, half = word & ((1 << shift) - 1), 1 << (shift - 1)
                value += remainder > half or (remainder == half and value & 1)
            draws.append(np.ldexp(np.float32(value), shift - 64))
    return np.array(draws[:rows], dtype=np.float32)


class UniformSquaredError(TreeObjective):
    def __init__(self, params, delegate=None, rounds_before=0):
        assert params.get("seed_per_iteration") is True
        assert all(params.get(key, 1.) == 1. for key in
                   ["colsample_bytree", "colsample_bylevel", "colsample_bynode"])
        assert params.get("num_parallel_tree", 1) == 1
        self.params = params
        self.delegate = delegate
        self.rounds_before = rounds_before

    def __call__(self, iteration, prediction, data):
        if self.delegate is not None:
            grad, hess = self.delegate(iteration, prediction, data)
        else:
            label = data.get_label().reshape(prediction.shape)
            weight = data.get_weight()
            weight = (weight.reshape((-1,) + (1,) * (prediction.ndim - 1))
                      if weight.size else 1.)
            grad, hess = (prediction - label) * weight, np.ones_like(prediction) * weight
        grad, hess = np.array(grad, copy=True), np.array(hess, copy=True)
        probability = np.float32(self.params["subsample"])
        if probability >= 1:
            return grad, hess
        seed = (self.params.get("seed", 0) * 127 + self.rounds_before + iteration) & 0xffffffff
        rng = np.random.RandomState(seed)  # mt19937 seed() matches XGBoost's host engine.
        targets = grad.shape[1] if grad.ndim > 1 else 1
        shared = self.params.get("multi_strategy") == "multi_output_tree"
        for target in range(1 if shared else targets):
            row_seed = int(rng.randint(0, 1 << 32, dtype=np.uint32))
            keep = (probability > 0) & (weights(row_seed, len(grad)) <= probability)
            if shared or grad.ndim == 1:
                grad[~keep], hess[~keep] = 0., 0.
            else:
                grad[~keep, target], hess[~keep, target] = 0., 0.
        return grad, hess

    def split_grad(self, iteration, grad, hess):
        if self.delegate is not None:
            return self.delegate.split_grad(iteration, grad, hess)
        return None
