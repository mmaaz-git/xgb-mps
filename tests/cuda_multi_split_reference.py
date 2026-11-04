"""CUDA vector-tree source oracle; this does not execute on NVIDIA hardware.

Uses integer inclusive scans, double argmax in each warp-sized chunk and float
updates between chunks. Cuts and sampled feature sets are separate contracts.
"""
import json
import math

import numpy as np

from cuda_split_reference import add, sub, dequantize, quantize, weight, trees, dense_values


def hessian_trace(values):
    # Python 3.12+ sum(float) uses compensation; CUDA accumulates sequentially.
    result = 0.
    for value in values:
        result += value[1]
    return result


def gain_weight(pair, w, p):
    g, h = pair
    # Preserve the type of w*w: float for constrained/parent gains, double for
    # unconstrained max_delta_step gains.
    square = float(w * w)
    return -(2. * g * float(w) + (h + float(np.float32(p.get("reg_lambda", 1)))) * square +
             2. * float(np.float32(p.get("reg_alpha", 0))) * abs(float(w)))


def unbounded_weight(pair, p):
    g, h = pair
    if h <= 0:
        return 0.
    alpha = float(np.float32(p.get("reg_alpha", 0)))
    g = g - alpha if g > alpha else g + alpha if g < -alpha else 0.
    w = -g / (h + float(np.float32(p.get("reg_lambda", 1))))
    delta = float(np.float32(p.get("max_delta_step", 0)))
    return math.copysign(delta, w) if delta and abs(w) > delta else w


def gain(pair, p):
    g, h = pair
    if h <= 0:
        return 0.
    if p.get("max_delta_step", 0):
        return gain_weight(pair, unbounded_weight(pair, p), p)
    alpha = float(np.float32(p.get("reg_alpha", 0)))
    g = g - alpha if g > alpha else g + alpha if g < -alpha else 0.
    return g * g / (h + float(np.float32(p.get("reg_lambda", 1))))


def split_weights(left, right, p, monotone, bounds):
    lw, rw = [], []
    for l, r, b in zip(left, right, bounds, strict=True):
        a, c = weight(l, p, b), weight(r, p, b)
        if (monotone > 0 and not a <= c) or (monotone < 0 and not a >= c):
            pooled = p | {key: np.float32(2 * np.float32(p.get(key, default)))
                          for key, default in [("reg_alpha", 0), ("reg_lambda", 1)]}
            a = c = weight(add(l, r), pooled, b)
        lw.append(a)
        rw.append(c)
    return lw, rw


def evaluate(hist, parent, scales, p, kind, monotone, bounds, constrained):
    n, nt = len(hist), len(parent)
    total = [(sum(v[t][0] for v in hist), sum(v[t][1] for v in hist)) for t in range(nt)]
    order = list(range(n))
    if kind == 2:
        pw = [weight(dequantize(parent[t], scales[t]), p) for t in range(nt)]
        scores = []
        for row in hist:
            score = 0.
            for t in range(nt):
                score += float(np.float32(weight(dequantize(row[t], scales[t]), p) * pw[t]))
            scores.append(score)
        order.sort(key=lambda i: scores[i])

    def score(left, right):
        l = [dequantize(v, s) for v, s in zip(left, scales, strict=True)]
        r = [dequantize(v, s) for v, s in zip(right, scales, strict=True)]
        lh = rh = 0.
        for a, b in zip(l, r, strict=True):
            lh += a[1]
            rh += b[1]
        lh, rh = lh / nt, rh / nt
        minimum = float(np.float32(p.get("min_child_weight", 1)))
        if not (lh > 0 and rh > 0 and lh >= minimum and rh >= minimum):
            return -math.inf
        value = 0.
        if constrained:
            lw, rw = split_weights(l, r, p, monotone, bounds)
            for a, b, aw, bw in zip(l, r, lw, rw, strict=True):
                value += gain_weight(a, aw, p)
                value += gain_weight(b, bw, p)
        else:
            for a, b in zip(l, r, strict=True):
                value += gain(a, p)
                value += gain(b, p)
        return value

    best = None
    best_gain = -np.finfo(np.float32).max
    limit = max(0, min(n, p.get("max_cat_threshold", 64)) - 1) if kind == 2 else n
    for phase in range(2):
        missing = phase == 0 if kind == 2 else phase == 1
        scan = [(0, 0)] * nt
        for base in range(0, limit, 32):
            chunk = None
            chunk_gain = -math.inf
            for i in range(base, min(base + 32, limit)):
                index = i if kind == 1 or phase == 0 else n - 1 - i
                value = hist[order[index] if kind == 2 else index]
                scan = [add(a, b) for a, b in zip(scan, value, strict=True)]
                child = (value if missing else [sub(a, b) for a, b in zip(total, value, strict=True)]) if kind == 1 else scan
                sibling = [sub(a, b) for a, b in zip(parent, child, strict=True)]
                left, right = (sibling, child) if missing else (child, sibling)
                g = score(left, right)
                if g > chunk_gain:
                    chunk_gain = g
                    cats = [index] if kind == 1 else order[:index + int(missing)] if kind == 2 else None
                    chunk = dict(bin=index, missing=missing, left=left, right=right, cats=cats)
            if not math.isinf(chunk_gain) and np.float32(chunk_gain) > best_gain:
                best_gain = np.float32(chunk_gain)
                best = chunk | {"gain": best_gain}
    return best


def audit_tree(tree, data, cuts, types, grad, hess, p, keep=None,
               selected_feature_only=False, integers=None, value_integers=None):
    """Check all splits, child memberships, gains, weights and allowed leaf stops.

    Reduced-gradient callers can supply independently quantized value gradients
    for the final leaves. Internal base_weights retain the split-target shape.
    """
    grad, hess = grad.reshape(len(data), -1), hess.reshape(len(data), -1)
    nt = grad.shape[1]
    if integers is None:
        columns = [quantize(grad[:, t], hess[:, t], keep) for t in range(nt)]
        pairs = [[columns[t][0][r] for t in range(nt)] for r in range(len(data))]
        scales = [column[1] for column in columns]
    else:
        pairs, scales = integers
    ptr, values = cuts
    constraints = p.get("monotone_constraints", [0] * data.shape[1])
    if isinstance(constraints, str):
        constraints = json.loads(constraints.replace("(", "[").replace(")", "]"))
    constrained = any(constraints)
    interactions = p.get("interaction_constraints")
    if isinstance(interactions, str):
        interactions = json.loads(interactions)
    if interactions is not None:
        included = {f for group in interactions for f in group}
        interactions += [[f] for f in range(data.shape[1]) if f not in included]
    binned = np.full(data.shape, -1, dtype=np.int32)
    for f in range(data.shape[1]):
        present = ~np.isnan(data[:, f])
        cut = values[ptr[f]:ptr[f+1]]
        if len(cut):
            side = "left" if types[f] == "c" else "right"
            binned[present, f] = np.minimum(np.searchsorted(cut, data[present, f], side=side), len(cut)-1)
    cat_map = {node: tree["categories"][start:start+count]
               for node, start, count in zip(tree["categories_nodes"], tree["categories_segments"],
                                             tree["categories_sizes"], strict=True)}
    base_weights = np.array(tree["base_weights"], dtype=np.float32).reshape(-1, nt)
    targets = int(tree["tree_param"]["size_leaf_vector"])
    leaf_weights = np.array(tree["leaf_weights"], dtype=np.float32).reshape(-1, targets)
    eta = np.float32(p.get("learning_rate", p.get("eta", .3)))
    total_leaves = sum(child < 0 for child in tree["left_children"])
    capped = 0 < p.get("max_leaves", 0) <= total_leaves
    depth_limit = p.get("max_depth", 6)
    counts = [0, 0]

    def sums(rows, source, n):
        return [(sum(source[r][t][0] for r in rows), sum(source[r][t][1] for r in rows)) for t in range(n)]

    def visit(node, rows, bounds, path, depth, incoming=None):
        parent = sums(rows, pairs, nt)
        fp = [dequantize(v, s) for v, s in zip(parent, scales, strict=True)]
        pw = [weight(v, p, b) for v, b in zip(fp, bounds, strict=True)]
        if incoming is None:
            incoming = pw
        parent_gain = 0.
        for v, w in zip(fp, pw, strict=True):
            parent_gain += gain_weight(v, w, p)
        left_id, right_id = tree["left_children"][node], tree["right_children"][node]
        expected_weight = pw if left_id >= 0 else np.array(incoming, dtype=np.float32) * eta
        np.testing.assert_array_equal(base_weights[node], expected_weight, err_msg=f"node={node}")
        if left_id < 0:
            counts[1] += 1
            if value_integers is None:
                expected_leaf = expected_weight
            else:
                value_pairs, value_scales = value_integers
                value_sums = sums(rows, value_pairs, targets)
                expected_leaf = [np.float32(weight(dequantize(v, s), p) * eta)
                                 for v, s in zip(value_sums, value_scales, strict=True)]
            np.testing.assert_array_equal(leaf_weights[right_id], expected_leaf, err_msg=f"leaf={node}")
            cover = np.float32(hessian_trace(fp))
            np.testing.assert_array_equal(np.float32(tree["sum_hessian"][node]), cover)
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
            hist = [[(0, 0)] * nt for _ in cut]
            for row in rows:
                b = binned[row, f]
                if b >= 0:
                    hist[b] = [add(a, v) for a, v in zip(hist[b], pairs[row], strict=True)]
            kind = 0 if types[f] != "c" else 1 if len(cut) < p.get("max_cat_to_onehot", 4) else 2
            candidate = evaluate(hist, parent, scales, p, kind, constraints[f], bounds, constrained)
            if candidate is not None and (best is None or candidate["gain"] > best["gain"]):
                best = candidate | {"feature": f, "kind": kind}
        improvement = -math.inf if best is None else np.float32(float(best["gain"]) - parent_gain)
        if left_id < 0:
            assert (improvement <= np.float32(1e-6) or
                    improvement < np.float32(p.get("gamma", p.get("min_split_loss", 0)))), (
                        "Premature vector leaf", node, improvement, best)
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
            i = best["bin"]
            threshold = (-np.inf if i == 0 else cut[i-1]) if best["missing"] else cut[i]
            np.testing.assert_array_equal(np.float32(tree["split_conditions"][node]), np.float32(threshold))
            go_left = data[rows, feature] < threshold
        go_left[np.isnan(data[rows, feature])] = best["missing"]
        lrows, rrows = rows[go_left], rows[~go_left]
        for selected, expected in [(lrows, best["left"]), (rrows, best["right"])]:
            assert sums(selected, pairs, nt) == expected, (node, "integer children")
        np.testing.assert_array_equal(np.float32(tree["loss_changes"][node]), improvement)
        l = [dequantize(v, s) for v, s in zip(best["left"], scales, strict=True)]
        r = [dequantize(v, s) for v, s in zip(best["right"], scales, strict=True)]
        cover = np.float32(hessian_trace(l) + hessian_trace(r))
        np.testing.assert_array_equal(np.float32(tree["sum_hessian"][node]), cover)
        lw, rw = split_weights(l, r, p, constraints[feature], bounds)
        lb, rb = list(bounds), list(bounds)
        if constrained:
            for t, (a, b) in enumerate(zip(lw, rw, strict=True)):
                mid = np.float32(a + np.float32(np.float32(.5) * np.float32(b - a)))
                if constraints[feature] > 0:
                    lb[t], rb[t] = (bounds[t][0], mid), (mid, bounds[t][1])
                elif constraints[feature] < 0:
                    lb[t], rb[t] = (mid, bounds[t][1]), (bounds[t][0], mid)
        counts[0] += 1
        visit(left_id, lrows, lb, path | {feature}, depth+1, lw)
        visit(right_id, rrows, rb, path | {feature}, depth+1, rw)

    limit = np.finfo(np.float32).max if constrained else np.inf
    visit(0, np.arange(len(data)), [(-limit, limit)] * nt, set(), 0)
    return counts


def audit_recorded(model, matrix, data, recorded, p):
    """Audit vector training, including a shared CUDA row sampler over targets."""
    from cuda_uniform_reference import weights
    from cuda_gradient_reference import sample

    data = dense_values(data)
    types = matrix.feature_types or ["q"] * data.shape[1]
    selected_only = any(p.get(key, 1.) < 1. for key in
                        ["colsample_bytree", "colsample_bylevel", "colsample_bynode"])
    counts = [0, 0]
    for iteration, (tree, (g, h)) in enumerate(zip(trees(model), recorded.gradients, strict=True)):
        g, h = g.reshape(len(data), -1), h.reshape(len(data), -1)
        keep, integers = None, None
        fraction = np.float32(p.get("subsample", 1.))
        if fraction < 1.:
            assert not selected_only and p.get("seed_per_iteration") is True
            rng = np.random.RandomState((p.get("seed", 0) * 127 + iteration) & 0xffffffff)
            row_seed = int(rng.randint(0, 1 << 32, dtype=np.uint32))
            if p.get("sampling_method", "uniform") == "uniform":
                keep = (fraction > 0) & (weights(row_seed, len(data)) <= fraction)
            else:
                _, _, _, (qg, qh, ge, he) = sample(g, h, fraction, row_seed, return_quantized=True)
                integers = ([[ (int(qg[r,t]), int(qh[r,t])) for t in range(g.shape[1])]
                             for r in range(len(data))], list(zip(map(int, ge), map(int, he), strict=True)))
        result = audit_tree(tree, data, matrix.get_quantile_cut(), types, g, h, p,
                            keep, selected_only, integers)
        counts = [a + b for a, b in zip(counts, result, strict=True)]
    return counts
