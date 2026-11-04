"""CUDA-source gradient sampling oracle, without NVIDIA execution.

Exact norms, threshold and integer rescaling are checked natively. The custom
objective below supplies these decisions to a CPU booster to check downstream
training at the existing prediction tolerances. The public objective boundary is
FP32, so that downstream comparison is not an exact integer-state oracle.

Reduction policy: CCCL 3d0d040ac0dc5907daf2a0731edb378e5562099d, SM60-SM90,
fixed segments. Scalar norms model one permitted binary64 FMA contraction; NVCC is unverified.
"""
import math
from fractions import Fraction
import numpy as np
from cuda_uniform_reference import UniformSquaredError, weights


def norm_fma(a, b, c):
    if hasattr(math, "fma"):
        return math.fma(a, b, c)
    # Python < 3.13: exact rational arithmetic, rounded once to binary64.
    # Norm operands here are finite and nonnegative.
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def exponents(values):
    values = np.asarray(values, dtype=np.float64)
    bound = np.maximum(np.maximum(values, 0).sum(axis=0),
                       np.maximum(-values, 0).sum(axis=0))
    bound /= 1. - 2. * len(values) * np.finfo(np.float64).eps
    return 62 - np.frexp(bound)[1]


def sample(grad, hess, fraction, seed, return_quantized=False):
    """Return sampled integer-derived split floats, replay factors, and exact roots."""
    grad, hess = np.asarray(grad, dtype=np.float32), np.asarray(hess, dtype=np.float32)
    shape = grad.shape
    g, h = grad.reshape(len(grad), -1).astype(np.float64), hess.reshape(len(hess), -1).astype(np.float64)
    ge, he = exponents(g), exponents(h)
    qg, qh = np.ldexp(g, ge).astype(np.int64), np.ldexp(h, he).astype(np.int64)
    qh[(h > 0) & (qh == 0)] = 1
    g, h = np.ldexp(qg.astype(np.float64), -ge), np.ldexp(qh.astype(np.float64), -he)
    factors = np.ones(len(g), dtype=np.float32)
    if fraction < 1.:
        # Separate binary64 squares then a fused multiply-add, narrowed once.
        terms = np.array([norm_fma(float(np.float32(.1)), float(b*b), float(a*a))
                          for a, b in zip(g.flat, h.flat)], dtype=np.float32).reshape(g.shape)
        targets = terms.shape[1]
        if targets <= 16:
            total = terms[:, 0].copy()
            for t in range(1, targets):
                total += terms[:, t]
        else:
            width = 32 if targets <= 512 else 256
            lanes = np.zeros((len(g), width), dtype=np.float32)
            for t in range(targets):
                lanes[:, t % width] += terms[:, t]
            for offset in [1, 2, 4, 8, 16]:
                previous = lanes.copy()
                for t in range(width):
                    if t % 32 + offset < 32:
                        lanes[:, t] += previous[:, t+offset]
            total = lanes[:, 0].copy()
            for warp in range(1, width // 32):
                total += lanes[:, warp*32]
        norms = np.sqrt(total)
        budget = int(np.float32(len(g)) * np.float32(fraction))
        threshold = np.float32(np.inf)
        if budget:
            ordered = np.sort(norms)
            assert np.isfinite(ordered).all()
            exponent = 62 - math.frexp(math.fsum(ordered) /
                                      (1. - 2. * len(g) * np.finfo(np.float64).eps))[1]
            cumulative = np.cumsum(np.ldexp(ordered.astype(np.float64), exponent).astype(np.int64))
            cumulative = np.ldexp(cumulative.astype(np.float64), -exponent).astype(np.float32)
            ordered = np.append(ordered, np.float32(np.inf))
            threshold = ordered[1]
            for i, prefix in enumerate(cumulative):
                denominator = np.float32(budget) - np.float32(len(g)-i-1)
                if denominator <= 0:
                    continue
                candidate = np.float32(prefix / denominator)
                if ordered[i] < candidate <= ordered[i+1]:
                    threshold = candidate
                    break
        probability = (np.zeros_like(norms) if np.isinf(threshold) else
                       norms / max(threshold, np.float32(1e-6)))
        keep = (probability >= 1.) | ((probability > 0.) & (weights(seed, len(g)) <= probability))
        factors[~keep] = 0.
        rescale = keep & (probability < 1.)
        factors[rescale] = np.float32(1.) / probability[rescale]
        g *= factors[:, None].astype(np.float64)
        h *= factors[:, None].astype(np.float64)
        fixed_g, fixed_h = np.ldexp(g, ge), np.ldexp(h, he)
        assert np.all((fixed_g >= -2.**63) & (fixed_g < 2.**63))
        assert np.all((fixed_h >= -2.**63) & (fixed_h < 2.**63))
        qg, qh = fixed_g.astype(np.int64), fixed_h.astype(np.int64)
        qh[(h > 0) & (qh == 0)] = 1
    roots = (np.ldexp(qg.sum(axis=0).astype(np.float64), -ge),
             np.ldexp(qh.sum(axis=0).astype(np.float64), -he))
    sampled = (np.ldexp(qg.astype(np.float64), -ge).astype(np.float32).reshape(shape),
               np.ldexp(qh.astype(np.float64), -he).astype(np.float32).reshape(shape))
    if return_quantized:
        return sampled, factors, roots, (qg, qh, ge, he)
    return sampled, factors, roots


class GradientSquaredError(UniformSquaredError):
    def __call__(self, iteration, prediction, data):
        if self.delegate is None:
            label = data.get_label().reshape(prediction.shape)
            weight = data.get_weight()
            weight = weight.reshape((-1,) + (1,) * (prediction.ndim-1)) if weight.size else 1.
            grad, hess = (prediction-label)*weight, np.ones_like(prediction)*weight
        else:
            grad, hess = self.delegate(iteration, prediction, data)
        grad, hess = np.array(grad, dtype=np.float32), np.array(hess, dtype=np.float32)
        split = self.delegate.split_grad(iteration, grad, hess) if self.delegate else None
        sg, sh = split if split is not None else (grad, hess)
        sg, sh = np.array(sg, copy=True), np.array(sh, copy=True)
        seed = (self.params.get("seed", 0)*127 + self.rounds_before + iteration) & 0xffffffff
        rng = np.random.RandomState(seed)
        targets = sg.shape[1] if sg.ndim > 1 else 1
        shared = self.params.get("multi_strategy") == "multi_output_tree"
        for target in range(1 if shared else targets):
            key = (...,) if shared or sg.ndim == 1 else (slice(None), target)
            row_seed = int(rng.randint(0, 1 << 32, dtype=np.uint32))
            sampled, factors, _ = sample(sg[key], sh[key], np.float32(self.params['subsample']), row_seed)
            sg[key], sh[key] = sampled
            if split is not None:
                factor = factors.reshape((-1,) + (1,) * (grad.ndim-1))
                grad *= factor
                hess *= factor
        self.saved_split = (sg, sh) if split is not None else None
        return (grad, hess) if split is not None else (sg, sh)

    def split_grad(self, iteration, grad, hess):
        return self.saved_split
