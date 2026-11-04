"""Scalar CUDA tree-split oracle executed on the host, without NVIDIA validation.

Integer histograms, exclusive numeric prefixes, missing-direction ties and raw
child gain ordering follow gpu_hist/evaluate_splits.cu. The oracle consumes the
actual cuts: CUDA sketch construction and column sampling are separate work.
"""
import json
import math

import numpy as np
from xgboost.objective import TreeObjective

from cuda_gradient_reference import exponents


class RecordedSquaredError(TreeObjective):
    def __init__(self):
        self.gradients = []

    def __call__(self, iteration, prediction, data):
        label = data.get_label().reshape(prediction.shape)
        weight = data.get_weight()
        weight = (weight.reshape((-1,) + (1,) * (prediction.ndim - 1)) if weight.size else 1.)
        grad = ((prediction - label) * weight).astype(np.float32)
        hess = (np.ones_like(prediction) * weight).astype(np.float32)
        self.gradients.append((grad.copy(), hess.copy()))
        return grad, hess


def quantize(grad, hess, keep=None):
    ge, he = int(exponents(grad)), int(exponents(hess))
    g = [int(math.ldexp(float(x), ge)) for x in grad]
    h = [max(1, int(math.ldexp(float(x), he))) if x > 0 else
         int(math.ldexp(float(x), he)) for x in hess]
    if keep is not None:
        g = [v if k else 0 for v, k in zip(g, keep, strict=True)]
        h = [v if k else 0 for v, k in zip(h, keep, strict=True)]
    return list(zip(g, h, strict=True)), (ge, he)


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def dequantize(pair, scale):
    return math.ldexp(float(pair[0]), -scale[0]), math.ldexp(float(pair[1]), -scale[1])


def weight(pair, p, bounds=(-np.inf, np.inf)):
    g, h = pair
    w = 0.
    if h > 0:
        alpha = float(np.float32(p.get("reg_alpha", 0)))
        threshold = g - alpha if g > alpha else g + alpha if g < -alpha else 0.
        w = -threshold / (h + float(np.float32(p.get("reg_lambda", 1))))
        delta = float(np.float32(p.get("max_delta_step", 0)))
        if delta and abs(w) > delta:
            w = math.copysign(delta, w)
    return np.float32(np.clip(np.float32(w), *bounds))


def gain(pair, w, p):
    g, h = pair
    if h <= 0:
        return np.float32(0)
    square = float(np.float32(w * w))
    return np.float32(-(2. * g * float(w) +
                        (h + float(np.float32(p.get("reg_lambda", 1)))) * square +
                        2. * float(np.float32(p.get("reg_alpha", 0))) * abs(float(w))))


def evaluate(hist, parent, scale, p, kind, monotone, bounds):
    n = len(hist)
    total = (sum(v[0] for v in hist), sum(v[1] for v in hist))
    missing = sub(parent, total)
    best = None
    best_gain = -np.finfo(np.float32).max

    def score(left, right):
        l, r = dequantize(left, scale), dequantize(right, scale)
        minimum = float(np.float32(p.get("min_child_weight", 1)))
        if not (l[1] > 0 and r[1] > 0 and l[1] >= minimum and r[1] >= minimum):
            return np.float32(-np.inf)
        lw, rw = weight(l, p, bounds), weight(r, p, bounds)
        if (monotone > 0 and not lw <= rw) or (monotone < 0 and not lw >= rw):
            return np.float32(-np.inf)
        return np.float32(gain(l, lw, p) + gain(r, rw, p))

    def update(value, left, right, i, ml, cats=None):
        nonlocal best, best_gain
        if value > best_gain:
            best_gain = value
            best = dict(gain=value, left=left, right=right, bin=i, missing=ml, cats=cats)

    if kind < 2:
        prefix = (0, 0)
        for i, value in enumerate(hist):
            scan = prefix if kind == 0 else sub(total, value)
            ml_sum = add(scan, missing)
            gl, gr = score(ml_sum, sub(parent, ml_sum)), score(scan, sub(parent, scan))
            ml = gl > gr
            left = ml_sum if ml else scan
            update(gl if ml else gr, left, sub(parent, left), i, ml, [i] if kind else None)
            prefix = add(prefix, value)
    else:
        order = sorted(range(n), key=lambda i: weight(dequantize(hist[i], scale), p))
        limit = min(n, p.get("max_cat_threshold", 64))
        right = (0, 0)
        for i in range(limit - 1):
            right = add(right, hist[order[i]])
            left = sub(parent, right)
            update(score(left, right), left, right, i, True, order[:i+1])
        left = (0, 0)
        for i in range(n - 1, n - limit, -1):
            left = add(left, hist[order[i]])
            right = sub(parent, left)
            update(score(left, right), left, right, i, False, order[:i])
    return best


def trees(model):
    return json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]


def audit_tree(tree, data, cuts, types, grad, hess, p, keep=None, selected_feature_only=False, integers=None):
    """Check every split and leaf; returned counts keep integration coverage visible.

    With column sampling enabled, callers can audit the winner within the chosen
    feature. Unsampled tests check the global winner over all allowed features.
    """
    pairs, scale = quantize(grad, hess, keep) if integers is None else integers
    ptr, values = cuts
    constraints = p.get("monotone_constraints", [0] * data.shape[1])
    if isinstance(constraints, str):
        constraints = json.loads(constraints.replace("(", "[").replace(")", "]"))
    constrained = any(constraints)
    interactions = p.get("interaction_constraints")
    if isinstance(interactions, str):
        interactions = json.loads(interactions)
    # A feature omitted from the groups can interact only with itself.
    if interactions is not None:
        included = {f for group in interactions for f in group}
        interactions = interactions + [[f] for f in range(data.shape[1]) if f not in included]
    binned = np.full(data.shape, -1, dtype=np.int32)
    for f in range(data.shape[1]):
        present = ~np.isnan(data[:, f])
        cut = values[ptr[f]:ptr[f+1]]
        if len(cut):
            side = "left" if types[f] == "c" else "right"
            binned[present, f] = np.minimum(np.searchsorted(cut, data[present, f], side=side), len(cut)-1)
    cat_map = {}
    for node, start, count in zip(tree["categories_nodes"], tree["categories_segments"],
                                   tree["categories_sizes"], strict=True):
        cat_map[node] = tree["categories"][start:start+count]
    splits, leaves = 0, 0

    total_leaves = sum(child < 0 for child in tree["left_children"])
    leaf_limit = p.get("max_leaves", 0)
    depth_limit = p.get("max_depth", 6)
    capped = leaf_limit > 0 and total_leaves >= leaf_limit

    def visit(node, rows, bounds, path, depth):
        nonlocal splits, leaves
        parent = (sum(pairs[r][0] for r in rows), sum(pairs[r][1] for r in rows))
        fp = dequantize(parent, scale)
        pw = weight(fp, p, bounds)
        np.testing.assert_array_equal(np.float32(tree["sum_hessian"][node]), np.float32(fp[1]))
        left_id, right_id = tree["left_children"][node], tree["right_children"][node]
        stored_weight = pw if left_id >= 0 or node == 0 else np.float32(pw * np.float32(p.get("learning_rate", p.get("eta", .3))))
        np.testing.assert_array_equal(np.float32(tree["base_weights"][node]), stored_weight)
        if left_id < 0:
            leaves += 1
            np.testing.assert_array_equal(np.float32(tree["split_conditions"][node]),
                                          np.float32(pw * np.float32(p.get("learning_rate", p.get("eta", .3)))))
            # Without column sampling, a leaf below the depth/leaf limits must
            # have no valid improving candidate. This catches premature stops.
            if selected_feature_only or capped or (depth_limit > 0 and depth >= depth_limit):
                return
        feature = tree["split_indices"][node]
        allowed = set(range(data.shape[1]))
        if interactions is not None and path:
            # CUDA constraints.cu accumulates unions along the path. CPU instead
            # requires a group to contain the entire history; overlapping groups differ.
            allowed = path | {f for group in interactions if path.intersection(group) for f in group}
        if left_id >= 0:
            assert feature in allowed
        best = None
        for f in ([feature] if selected_feature_only else sorted(allowed)):
            cut = values[ptr[f]:ptr[f+1]]
            if not len(cut):
                continue
            hist = [(0, 0)] * len(cut)
            for row in rows:
                bin_idx = binned[row, f]
                if bin_idx >= 0:
                    hist[bin_idx] = add(hist[bin_idx], pairs[row])
            kind = 0 if types[f] != "c" else 1 if len(cut) < p.get("max_cat_to_onehot", 4) else 2
            candidate = evaluate(hist, parent, scale, p, kind, constraints[f], bounds)
            if candidate is not None and (best is None or candidate["gain"] > best["gain"]):
                best = candidate | {"feature": f, "kind": kind}
        if left_id < 0:
            improvement = (-np.inf if best is None else
                           np.float32(best["gain"] - gain(fp, pw, p)))
            assert (improvement <= np.float32(1e-6) or
                    improvement < np.float32(p.get("gamma", p.get("min_split_loss", 0)))), (
                        "Premature leaf", node, improvement, best)
            return
        assert best is not None
        assert feature == best["feature"], (node, feature, best)
        assert bool(tree["default_left"][node]) == best["missing"], (node, best)
        cut = values[ptr[feature]:ptr[feature+1]]
        if best["kind"]:
            expected_cats = sorted(int(cut[i]) for i in best["cats"])
            assert cat_map[node] == expected_cats, (node, cat_map[node], expected_cats)
            go_left = ~np.isin(data[rows, feature], expected_cats)
        else:
            threshold = -np.inf if best["bin"] == 0 else cut[best["bin"]-1]
            np.testing.assert_array_equal(np.float32(tree["split_conditions"][node]), np.float32(threshold))
            go_left = data[rows, feature] < threshold
        go_left[np.isnan(data[rows, feature])] = best["missing"]
        lrows, rrows = rows[go_left], rows[~go_left]
        for selected, expected in [(lrows,best["left"]), (rrows,best["right"])]:
            actual = (sum(pairs[r][0] for r in selected), sum(pairs[r][1] for r in selected))
            assert actual == expected, (node, actual, expected)
        root_gain = gain(fp, pw, p)
        np.testing.assert_array_equal(np.float32(tree["loss_changes"][node]),
                                      np.float32(best["gain"] - root_gain))
        lb, rb = bounds, bounds
        if constrained:
            lw, rw = weight(dequantize(best["left"], scale), p, bounds), weight(dequantize(best["right"], scale), p, bounds)
            mid = np.float32(np.float32(lw + rw) / np.float32(2))
            if constraints[feature] > 0:
                lb, rb = (bounds[0],mid), (mid,bounds[1])
            elif constraints[feature] < 0:
                lb, rb = (mid,bounds[1]), (bounds[0],mid)
        splits += 1
        visit(left_id,lrows,lb,path | {feature},depth+1)
        visit(right_id,rrows,rb,path | {feature},depth+1)

    limit = np.finfo(np.float32).max if constrained else np.inf
    visit(0,np.arange(len(data)),(-limit,limit),set(),0)
    return splits, leaves


def dense_values(data):
    """Preserve sparse missing entries, including explicit observed zeros."""
    if not hasattr(data, "indptr"):
        return data
    output = np.full(data.shape, np.nan, dtype=np.float32)
    coo = data.tocoo()
    output[coo.row, coo.col] = coo.data
    return output


def audit_recorded(model, matrix, data, recorded, p):
    """Audit scalar trees with exact CUDA-derived uniform/gradient sampling."""
    from cuda_uniform_reference import weights
    from cuda_gradient_reference import sample

    data = dense_values(data)
    types = matrix.feature_types or ["q"] * data.shape[1]
    selected_only = any(p.get(key, 1.) < 1. for key in
                        ["colsample_bytree", "colsample_bylevel", "colsample_bynode"])
    counts = [0, 0]
    model_trees = trees(model)
    offset = 0
    for iteration, (grad, hess) in enumerate(recorded.gradients):
        grad, hess = grad.reshape(len(data), -1), hess.reshape(len(data), -1)
        fraction = np.float32(p.get("subsample", 1.))
        if fraction < 1.:
            assert not selected_only and p.get("seed_per_iteration") is True
            seed = (p.get("seed", 0)*127 + iteration) & 0xffffffff
            rng = np.random.RandomState(seed)
        for target in range(grad.shape[1]):
            g, h = grad[:, target], hess[:, target]
            keep, integers = None, None
            if fraction < 1.:
                row_seed = int(rng.randint(0, 1 << 32, dtype=np.uint32))
                if p.get("sampling_method", "uniform") == "uniform":
                    keep = (fraction > 0) & (weights(row_seed, len(data)) <= fraction)
                else:
                    _, _, _, (qg, qh, ge, he) = sample(g,h,fraction,row_seed,return_quantized=True)
                    integers = ([(int(a),int(b)) for a,b in zip(qg[:,0],qh[:,0],strict=True)],
                                (int(ge[0]),int(he[0])))
            result = audit_tree(model_trees[offset],data,matrix.get_quantile_cut(),types,g,h,p,
                                keep,selected_only,integers)
            counts = [a+b for a,b in zip(counts,result,strict=True)]
            offset += 1
    assert offset == len(model_trees)
    return counts
