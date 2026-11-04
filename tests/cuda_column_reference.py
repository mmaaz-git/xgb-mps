"""Scalar CUDA-source column selection, not NVIDIA execution or libdevice emulation."""
import math

import numpy as np

from cuda_uniform_reference import weights as philox_uniforms


def shuffle(size, seed):
    state = int(seed) % 2147483647 or 1
    keys = []
    for _ in range(24):
        state = state * 48271 % 2147483647
        keys.append(int(((state - 1) / 2147483646.) * 2.**32))
    bits = max(8, (max(1, size) - 1).bit_length())
    left_bits, right_bits = bits // 2, (bits + 1) // 2
    left_mask, right_mask = (1 << left_bits) - 1, (1 << right_bits) - 1
    order = []
    for i in range(1 << bits):
        left, right = i >> right_bits, i & right_mask
        for key in keys:
            product = (0xD2B74407B1CE6E93 * left) & ((1 << 64) - 1)
            left, right = ((product >> 32) ^ key ^ right) & left_mask, (
                ((product & 0xffffffff) << (right_bits - left_bits)) |
                (right >> left_bits)) & right_mask
        value = (left << right_bits) | right
        if value < size:
            order.append(value)
    return np.array(order, dtype=np.uint32)


def sample(features, feature_weights, seed, rate):
    if rate == 1:
        return features
    count = max(1, int(np.float32(np.float32(rate) * len(features))))
    if feature_weights is None:
        order = shuffle(len(features), seed)
    else:
        u = philox_uniforms(seed, len(features))
        # Host binary64 log followed by FP32 rounding is an independent math
        # reference. Public fixtures avoid ambiguous priority boundaries; native
        # tests separately check exact Apple log rounding and deliberately tied keys.
        logarithms = np.array([math.log(float(v)) if v else -np.inf for v in u], dtype=np.float32)
        keys = logarithms / np.maximum(feature_weights[features], np.float32(1e-6))
        order = np.argsort(-keys, kind="stable")
        if count < len(order):
            boundary = keys[order[count - 1]], keys[order[count]]
            gap = float(boundary[0]) - float(boundary[1])
            assert gap > 8 * max(abs(float(np.spacing(v))) for v in boundary), "Ambiguous log rounding"
    return np.sort(features[order[:count]])
