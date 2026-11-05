"""CUDA sketch equations with a declared Metal FP32 scan association.

Sorting/pruning/merging/cut queries are independent scalar source oracles. The
vectorized prefix specifies Metal's 256-entry tree association; CUB may choose a
different association. These are not measurements of an NVIDIA GPU. Native
fixtures separately use exactly summable weights, independent of association.
"""
import math
import numpy as np

F = np.float32


def budget(rows, bins):
    if rows == 0:
        return 1
    eps = (1. / min(rows, bins)) / 2.
    level = 1
    while True:
        size = min(rows, math.ceil(level / eps) + 1)
        if (1 << level) * size >= rows:
            return size
        level += 1


def prefix(values, ids):
    """One declared parallel FP32 association, including cross-tile carries."""
    n = len(values)
    if not n:
        return np.empty(0, dtype=F)
    blocks = (n + 255) // 256
    v = np.zeros(blocks * 256, dtype=F)
    k = np.full(blocks * 256, -1, dtype=np.int64)
    v[:n], k[:n] = values, ids
    v, k = v.reshape(-1, 256), k.reshape(-1, 256)
    for step in [1, 2, 4, 8, 16, 32, 64, 128]:
        v[:, step:] = np.where(k[:, step:] == k[:, :-step], v[:, :-step] + v[:, step:], v[:, step:])
    tails = np.minimum(np.arange(1, blocks + 1) * 256, n) - 1
    v = v.ravel()[:n]
    if blocks > 1:
        upper = prefix(v[tails].copy(), ids[tails])
        b = np.arange(n) // 256
        carry = (b > 0) & (ids == ids[tails[np.maximum(0, b - 1)]])
        v[carry] = upper[b[carry] - 1] + v[carry]
    return v


def unique(a):
    if not len(a):
        return a
    return a[np.r_[True, np.diff(a[:, 3]) != F(0)]]


def prune(a, to, categorical, index_unique=False):
    if categorical or len(a) <= to:
        return a.copy()
    if to == 1:
        return a[:1].copy()
    indices = [0]
    middle = a[:, 0] + a[:, 1]
    for k in range(1, to - 1):
        rank = F(F(F(F(k) * F(a[-1, 0] - a[0, 1])) / F(F(to) - F(1))) + a[0, 1]) * F(2)
        if rank < middle[0]:
            index = 0
        elif rank >= middle[-1]:
            index = len(a) - 1
        else:
            # Linear predicate count rather than the shader binary search.
            i = int(np.count_nonzero(~(rank < middle[1:-1])))
            boundary = F(F(a[i, 0] + a[i, 2]) + F(a[i + 1, 1] - a[i + 1, 2]))
            index = i if rank < boundary else i + 1
        indices.append(index)
    indices = np.array(indices + [len(a) - 1])
    if index_unique:
        indices = indices[np.r_[True, indices[1:] != indices[:-1]]]
    return a[indices].copy()


def normalize_input(a):
    for i in range(len(a) - 2, -1, -1):
        l, r = a[i], a[i + 1]
        if F(l[3] - r[3]) == 0 and F(l[0] - r[0]) != 0:
            l[2] = l[2] + r[2]
            l[1] = l[0] + l[2]
    return unique(a)


def merge(a, b):
    out, i, j = [], 0, 0
    if not len(a) or not len(b):
        out = list(b.copy() if not len(a) else a.copy())
    while len(a) and len(b) and (i < len(a) or j < len(b)):
        if i < len(a) and j < len(b) and a[i, 3] == b[j, 3]:
            out.append([a[i, 0] + b[j, 0], a[i, 1] + b[j, 1], a[i, 2] + b[j, 2], a[i, 3]])
            i += 1
        elif i < len(a) and (j == len(b) or a[i, 3] < b[j, 3]):
            lo = F(b[j - 1, 0] + b[j - 1, 2]) if j else F(0)
            hi = (b[-1, 1] if j == len(b) else F(b[j, 1] - b[j, 2])) if len(b) else F(0)
            out.append([a[i, 0] + lo, a[i, 1] + hi, a[i, 2], a[i, 3]])
            i += 1
        else:
            lo = F(a[i - 1, 0] + a[i - 1, 2]) if i else F(0)
            hi = (a[-1, 1] if i == len(a) else F(a[i, 1] - a[i, 2])) if len(a) else F(0)
            out.append([lo + b[j, 0], hi + b[j, 1], b[j, 2], b[j, 3]])
            j += 1
    out = unique(np.array(out, dtype=F).reshape(-1, 4))
    lo, hi = F(0), F(0)
    for e in out:
        if e[0] < lo:
            e[0] = lo
        if e[1] < hi:
            e[1] = hi
        if e[1] < F(e[0] + e[2]):
            e[1] = F(e[0] + e[2])
        lo, hi = e[:2]
    return out


def query(a, bins):
    if not len(a):
        return [F(1e-5)]
    out, last, cursor = [], a[0, 3], 0
    if len(a) <= bins:
        for v in a[1:, 3]:
            if v > last:
                out.append(v)
                last = v
    else:
        for i in range(1, bins):
            rank = 2. * (i * float(a[-1, 1]) / bins)
            while cursor < len(a) - 2 and rank >= float(F(a[cursor + 1, 0] + a[cursor + 1, 1])):
                cursor += 1
            boundary = F(F(a[cursor, 0] + a[cursor, 2]) + F(a[cursor + 1, 1] - a[cursor + 1, 2]))
            c = a[cursor if rank < float(boundary) else cursor + 1, 3]
            if c <= last:
                greater = a[a[:, 3] > last, 3]
                if not len(greater):
                    break
                c = greater[0]
            out.append(c)
            last = c
    v = a[-1, 3]
    out.append(F(v + F(abs(v) + F(1e-5))))
    return out


def cuts(data, bins, hessian=None, page_rows=None, csr=None, adapter=False, return_summary=False):
    csr = data.get_data() if csr is None else csr
    types = data.feature_types or ["q"] * data.num_col()
    weights = data.get_weight()
    groups = data.get_uint_info("group_ptr")
    if len(weights) and len(groups):
        weights = np.repeat(weights, np.diff(groups))
    if hessian is not None:
        weights = np.asarray(hessian, dtype=F) * (weights if len(weights) else F(1))
    columns = [np.empty((0, 4), dtype=F) for _ in types]
    seen = 0
    page_rows = page_rows or [data.num_row()]
    base = 0
    for nrows in page_rows:
        page = csr[base:base + nrows]
        # Adapter windows count raw matrix slots, even when those slots are missing.
        slots = np.repeat(np.arange(nrows, dtype=np.uint64), np.diff(page.indptr)) * data.num_col() + page.indices if adapter else None
        extent = nrows * data.num_col() if adapter else page.nnz
        for slot_begin in range(0, extent, 1 << 26):
            slot_end = min(extent, slot_begin + (1 << 26))
            if adapter:
                rows = max(1, (slot_end - slot_begin + data.num_col() - 1) // data.num_col())
                begin, end = np.searchsorted(slots, [slot_begin, slot_end])
            else:
                begin, end = slot_begin, slot_end
                first = np.searchsorted(page.indptr, begin, side="right") - 1
                last = np.searchsorted(page.indptr, end, side="left")
                rows = max(1, int(last - first))
            seen += rows
            if begin == end:
                continue
            ids, values = page.indices[begin:end], page.data[begin:end]
            order = np.lexsort((values, ids))
            ids, values = ids[order], values[order]
            scan = None
            if len(weights):
                row = np.searchsorted(page.indptr, np.arange(begin, end), side="right") - 1 + base
                scan = prefix(weights[row[order]], ids)
            for f in range(len(types)):
                mask = ids == f
                v = values[mask]
                w = scan[mask] if scan is not None else None
                categorical = types[f] == "c"
                if categorical and len(v):
                    keep = np.r_[True, v[1:] != v[:-1]]
                    v = v[keep]
                    if w is not None:
                        w = w[keep]
                if w is None:
                    lo, hi, weight = np.arange(len(v), dtype=F), np.arange(1, len(v) + 1, dtype=F), np.ones(len(v), dtype=F)
                else:
                    lo = np.r_[F(0), w[:-1]][:len(w)]
                    hi = w
                    weight = hi - lo
                    weight[weight < 0] = F(1e-6)
                incoming = np.column_stack([lo, hi, weight, v]).astype(F)
                incoming = normalize_input(prune(incoming, min(rows, budget(rows, bins)), categorical))
                columns[f] = unique(prune(merge(columns[f], incoming), budget(seen, bins), categorical, index_unique=True))
        base += nrows
    if return_summary:
        return columns, seen
    ptr, values = [0], []
    for f, column in enumerate(columns):
        if types[f] != "c":
            values.append(F(-np.inf))  # Public get_quantile_cut prepends the lower bound.
        values.extend(range(int(column[-1, 3] if len(column) else 0) + 1) if types[f] == "c" else query(column, bins))
        ptr.append(len(values))
    return np.array(ptr, dtype=np.uint64), np.array(values, dtype=F)


def assert_cuts(data, bins, **kwargs):
    expected = cuts(data, bins, **kwargs)
    for actual, oracle in zip(data.get_quantile_cut(), expected):
        np.testing.assert_array_equal(actual.view(np.uint32), oracle.view(np.uint32))


def assert_same_model_cpu(model, data):
    import xgboost as xgb
    loaded = xgb.Booster(model_file=model.save_raw())
    loaded.set_param({"device": "cpu"})
    for margin in [False, True]:
        np.testing.assert_allclose(model.predict(data, output_margin=margin),
                                   loaded.predict(data, output_margin=margin), rtol=2e-5, atol=2e-5)


def fixed_hist_matrix(data, bins):
    """Preserve an equivalence fixture's data/metadata while fixing its cuts.

    Both tree learners must see identical cut candidates for CPU model identity.
    Ordinary Metal ingestion is tested separately against the CUDA source oracle.
    """
    import xgboost as xgb
    rows = data.num_row()
    labels = data.get_label()
    if rows and len(labels):
        labels = labels.reshape(rows, -1)
    margin = data.get_base_margin()
    if rows and len(margin):
        margin = margin.reshape(rows, -1)
    groups = data.get_uint_info("group_ptr")
    return xgb.QuantileDMatrix(data.get_data(), label=labels, weight=data.get_weight(),
                               base_margin=margin if len(margin) else None,
                               group=np.diff(groups) if len(groups) else None,
                               feature_types=data.feature_types, feature_names=data.feature_names,
                               feature_weights=data.get_float_info("feature_weights"),
                               max_bin=bins, nthread=2)
