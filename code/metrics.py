# Implement metrics.

from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from core import apd

def _interval_answers(hist):
    """Return all non-empty contiguous interval sums for a 1-D histogram."""
    prefix = np.concatenate([[0.0], np.cumsum(np.asarray(hist, float))])
    return np.asarray([prefix[hi] - prefix[lo] for lo in range(len(hist)) for hi in range(lo + 1, len(hist) + 1)])

def _rectangle_answers(hist):
    """Return all axis-aligned rectangle sums for a small 2-D histogram."""
    hist = np.asarray(hist, float)
    prefix = np.pad(hist, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    answers = []
    for lo0 in range(hist.shape[0]):
        for hi0 in range(lo0 + 1, hist.shape[0] + 1):
            for lo1 in range(hist.shape[1]):
                for hi1 in range(lo1 + 1, hist.shape[1] + 1):
                    answers.append(prefix[hi0, hi1] - prefix[lo0, hi1] - prefix[hi0, lo1] + prefix[lo0, lo1])
    return np.asarray(answers)

def range_query_errors(real, syn, schema):
    """Errors for fixed bin-aligned numeric range-query workloads.

    The workloads are public and deterministic: all one-dimensional intervals
    on each of the first two numeric attributes and all rectangles on their
    product.  Values are proportions, so results are comparable across output
    sizes.  Bank additionally reports the four public job-hierarchy groups.
    """
    numeric = list(schema.numeric)
    if not numeric:
        return {}
    a, cards = schema.discrete(real)
    b, _ = schema.discrete(syn)
    interval_errors = []
    result = {}
    for position, j in enumerate(numeric[:2]):
        pa = np.bincount(a[:, j], minlength=cards[j]) / len(a)
        pb = np.bincount(b[:, j], minlength=cards[j]) / len(b)
        errors = np.abs(_interval_answers(pa) - _interval_answers(pb))
        key = 'primary' if position == 0 else 'secondary'
        result[f'range_1d_{key}_mae'] = float(errors.mean())
        result[f'range_1d_{key}_p95'] = float(np.quantile(errors, 0.95))
        interval_errors.extend(errors)
    if len(numeric) >= 2:
        j, k = numeric[:2]
        shape = (cards[j], cards[k])
        ia = np.ravel_multi_index(a[:, [j, k]].T, shape)
        ib = np.ravel_multi_index(b[:, [j, k]].T, shape)
        pa = np.bincount(ia, minlength=int(np.prod(shape))).reshape(shape) / len(a)
        pb = np.bincount(ib, minlength=int(np.prod(shape))).reshape(shape) / len(b)
        errors = np.abs(_rectangle_answers(pa) - _rectangle_answers(pb))
        result['range_2d_mae'] = float(errors.mean())
        result['range_2d_p95'] = float(np.quantile(errors, 0.95))
        interval_errors.extend(errors)
    result['range_combined_mae'] = float(np.mean(interval_errors))
    if schema.name == 'bank':
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3])
        real_group = np.bincount(groups[a[:, 3]], minlength=4) / len(a)
        syn_group = np.bincount(groups[b[:, 3]], minlength=4) / len(b)
        result['job_hierarchy_mae'] = float(np.abs(real_group - syn_group).mean())
    return result

def marginal_errors(real, syn, schema):
    a, cards = schema.discrete(real)
    b, _ = schema.discrete(syn)
    result = {}
    for order in [1, 2, 3]:
        values = []
        age = []
        other = []
        for cols in combinations(range(schema.p), order):
            sizes = [cards[j] for j in cols]
            ai = np.ravel_multi_index(a[:, cols].T, sizes)
            bi = np.ravel_multi_index(b[:, cols].T, sizes)
            length = int(np.prod(sizes))
            pa = np.bincount(ai, minlength=length) / len(a)
            pb = np.bincount(bi, minlength=length) / len(b)
            e = float(np.abs(pa - pb).sum())
            values.append(e)
            (age if 0 in cols else other).append(e)
        result[f'l1_{order}way'] = float(np.mean(values))
        result[f'l1_{order}way_age'] = float(np.mean(age))
        result[f'l1_{order}way_other'] = float(np.mean(other)) if other else None
    if schema.name.startswith('xor'):
        parity = np.bitwise_xor.reduce((syn[:, :-1] >= 0.5).astype(int), axis=1)
        result['parity_validity'] = float(np.mean(parity == syn[:, -1]))
        aa = np.column_stack([(real[:, :-1] >= 0.5).astype(int), real[:, -1].astype(int)])
        bb = np.column_stack([(syn[:, :-1] >= 0.5).astype(int), syn[:, -1].astype(int)])
        sizes = [2] * schema.p
        pa = np.bincount(np.ravel_multi_index(aa.T, sizes), minlength=2 ** schema.p) / len(aa)
        pb = np.bincount(np.ravel_multi_index(bb.T, sizes), minlength=2 ** schema.p) / len(bb)
        result['hoerr_bits'] = float(np.abs(pa - pb).sum())
    return result

def evaluate(train, syn, target, schema, seed=0):
    result = marginal_errors(target, syn, schema)
    result.update(range_query_errors(target, syn, schema))
    result.update({'apd': apd(syn, schema), 'unique_fraction': len(np.unique(syn, axis=0)) / len(syn)})
    x = schema.encode(syn)
    xt = schema.encode(target)
    y = syn[:, -1].astype(int)
    yt = target[:, -1].astype(int)
    for name, model in [('lr', LogisticRegression(C=1, max_iter=500)), ('tree', ExtraTreesClassifier(n_estimators=64, max_features=1.0, min_samples_leaf=1, n_jobs=1, random_state=42))]:
        if len(np.unique(y)) < 2:
            prob = np.full(len(target), float(y[0]))
            pred = np.full(len(target), y[0])
        else:
            model.fit(x, y)
            prob = model.predict_proba(xt)[:, 1]
            pred = (prob >= 0.5).astype(int)
        result[f'auc_{name}'] = float(roc_auc_score(yt, prob))
        result[f'f1_{name}'] = float(f1_score(yt, pred, average='macro', zero_division=0))
        result[f'accuracy_{name}'] = float(accuracy_score(yt, pred))
    return result
