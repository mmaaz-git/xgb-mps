"""Scalar replay of vendored CUDA QuadratureTreeSHAP, without a CUDA runtime.

Authority: predictor/interpretability/shap.cu CompressTree, Descend/HandleReturn,
ExtractQuadrature*Local and SubgroupOps::Sum; quadrature.h for root means/rule.
The recursion is independent of Metal's iterative frame representation. Each
operation rounds to FP32, without contraction. Across trees we choose a serial
order: NVIDIA atomic-add scheduling and NVCC contraction remain unverified.
"""
import json
import math

import numpy as np

F = np.float32


def quadrature():
    def legendre(n, x):
        a, b = 1., x
        if n == 0:
            return a
        for k in range(2, n + 1):
            a, b = b, ((2. * k - 1.) * x * b - (k - 1.) * a) / k
        return b

    def derivative(x, pn):
        return 8. * (x * pn - legendre(7, x)) / (x * x - 1.)

    nodes, weights = np.empty(8, dtype=F), np.empty(8, dtype=F)
    for i in range(8):
        x = math.cos(math.pi * (i + .75) / 8.5)
        for _ in range(64):
            dx = legendre(8, x) / derivative(x, legendre(8, x))
            x -= dx
            if abs(dx) < 1e-15:
                break
        dp = derivative(x, legendre(8, x))
        w = 2. / ((1. - x * x) * dp * dp)
        s = .5 * (x + 1.)
        nodes[7-i], weights[7-i] = F(s * s), F(2. * s * (.5 * w))
    return nodes, weights


NODES, WEIGHTS = quadrature()


def subgroup_sum(values, serial=False):
    if serial:  # Deliberately different association, used to find test witnesses.
        result = F(0)
        for value in values:
            result = F(result + value)
        return result
    values = np.asarray(values, dtype=F).copy()
    for offset in (4, 2, 1):
        previous = values.copy()
        for lane in range(8 - offset):
            values[lane] = F(previous[lane] + previous[lane + offset])
    return values[0]


@np.errstate(all="ignore")
def predict(raw, x, interactions, base_margin=None, tree_end=None, serial=False):
    learner = raw["learner"]
    param = learner["learner_model_param"]
    groups = max(int(param["num_class"]), int(param["num_target"]))
    features = int(param["num_feature"])
    booster = learner["gradient_booster"]
    if booster["name"] == "dart":
        model = booster["gbtree"]["model"]
        weights = np.asarray(booster["weight_drop"], dtype=F)
    else:
        model = booster["model"]
        weights = np.asarray(model.get("weight_drop", np.ones(len(model["trees"]))), dtype=F)
    trees = model["trees"][:tree_end]
    columns = features + 1
    shape = (len(x), groups, columns, columns) if interactions else (len(x), groups, columns)
    result = np.zeros(shape, dtype=F)
    means = np.zeros(groups, dtype=np.float64)
    base = np.asarray(json.loads(param["base_score"]), dtype=F).reshape(-1)

    for t, tree in enumerate(trees):
        count = int(tree["tree_param"]["size_leaf_vector"])
        left, right = tree["left_children"], tree["right_children"]
        covers = np.asarray(tree["sum_hessian"], dtype=F)
        conds = np.asarray(tree["split_conditions"], dtype=F)
        leaf_values = (np.asarray(tree["leaf_weights"], dtype=F).reshape(-1, count)
                       if count > 1 else None)
        categories = {}
        for node, begin, size in zip(tree["categories_nodes"], tree["categories_segments"],
                                     tree["categories_sizes"]):
            categories[node] = set(tree["categories"][begin:begin+size])

        def leaf(node, target):
            return leaf_values[right[node], target] if count > 1 else conds[node]

        def mean_scalar(node):
            if left[node] == -1:
                return float(leaf(node, 0))
            a, b = mean_scalar(left[node]), mean_scalar(right[node])
            if covers[node] == 0:
                return .5 * (a + b)
            return (a * float(covers[left[node]]) + b * float(covers[right[node]])) / float(covers[node])

        tree_means = np.zeros(count, dtype=np.float64)

        def mean_vector(node, probability):
            if left[node] == -1:
                for target in range(count):
                    tree_means[target] += probability * float(leaf(node, target))
                return
            for child in (left[node], right[node]):
                p = (probability * .5 if covers[node] == 0 else
                     probability * float(covers[child]) / float(covers[node]))
                mean_vector(child, p)

        if count > 1:
            mean_vector(0, 1.)
            means += tree_means * float(weights[t])
        else:
            means[model["tree_info"][t]] += mean_scalar(0) * float(weights[t])

        for target in range(count):
            group = target if count > 1 else model["tree_info"][t]
            for row, values in enumerate(x):
                phi = result[row, group]
                path = []

                def visit(node, basis):
                    if left[node] == -1:
                        # CUDA CompressTree applies this FP32 product on host.
                        weighted_leaf = F(leaf(node, target) * weights[t])
                        return np.asarray([F(v * weighted_leaf) for v in basis], dtype=F)
                    feature = tree["split_indices"][node]
                    value = values[feature]
                    if np.isnan(value):
                        goes_left = bool(tree["default_left"][node])
                    elif node in categories:
                        goes_left = value < 0 or value >= 16777216 or int(value) not in categories[node]
                    else:
                        goes_left = value < conds[node]
                    previous = next((q for f, q in reversed(path) if f == feature), F(1))
                    returns = []
                    for index, child in enumerate((left[node], right[node])):
                        weight = (F(.5) if covers[node] <= 0 else
                                  max(F(covers[child] / covers[node]), F(1e-12)))
                        enter = F(previous / weight) if (index == 0) == goes_left else F(0)
                        child_basis = np.empty(8, dtype=F)
                        for lane in range(8):
                            v = F(F(basis[lane] * weight) * F(F(1) + F(F(enter - F(1)) * NODES[lane])))
                            if previous != 1:
                                v = F(v / F(F(1) + F(F(previous - F(1)) * NODES[lane])))
                            child_basis[lane] = v
                        path.append((feature, enter))
                        ret = visit(child, child_basis)
                        local = np.empty(8, dtype=F)
                        for lane in range(8):
                            weighted_ret = F(ret[lane] * WEIGHTS[lane])
                            edge = F(0)
                            if enter != 1:
                                alpha = F(enter - F(1))
                                edge = F(edge + F(alpha / F(F(1) + F(alpha * NODES[lane]))))
                            if previous != 1:
                                alpha = F(previous - F(1))
                                edge = F(edge - F(alpha / F(F(1) + F(alpha * NODES[lane]))))
                            local[lane] = F(weighted_ret * edge)
                        delta = subgroup_sum(local, serial)
                        if interactions:
                            phi[feature, feature] = F(phi[feature, feature] + delta)
                            seen = {feature}
                            for partner, q in reversed(path):
                                if partner in seen:
                                    continue
                                seen.add(partner)
                                pair = np.zeros(8, dtype=F)
                                if q != 1:
                                    alpha = F(q - F(1))
                                    for lane in range(8):
                                        pair[lane] = F(F(alpha * local[lane]) /
                                                       F(F(1) + F(alpha * NODES[lane])))
                                phi[feature, partner] = F(phi[feature, partner] + subgroup_sum(pair, serial))
                        else:
                            phi[feature] = F(phi[feature] + delta)
                        path.pop()
                        returns.append(ret)
                    return np.asarray([F(b + a) for a, b in zip(*returns)], dtype=F)

                visit(0, np.ones(8, dtype=F))

    for row in range(len(x)):
        for group in range(groups):
            margin = base[group if len(base) > 1 else 0] if base_margin is None else base_margin[row, group]
            bias = F(F(means[group]) + F(margin))
            phi = result[row, group]
            if not interactions:
                phi[-1] = F(phi[-1] + bias)
                continue
            phi[-1, -1] = F(phi[-1, -1] + bias)
            for r in range(columns):
                for c in range(r + 1, columns):
                    sym = F(F(.5) * F(phi[r, c] + phi[c, r]))
                    phi[r, c] = phi[c, r] = sym
            for r in range(columns):
                for c in range(columns):
                    if c != r:
                        phi[r, r] = F(phi[r, r] - phi[r, c])
    return result
