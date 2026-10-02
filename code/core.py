# Draft-aligned voting, zCDP calibration, and tabular evolution.

from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from time import perf_counter
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax

@dataclass
class Schema:
    name: str
    names: tuple[str, ...]
    numeric: dict[int, tuple[float, float, float]]
    categorical: dict[int, np.ndarray]
    bins: dict[int, list[float]]

    @property
    def p(self):
        return len(self.names)

    def random(self, n, rng):
        out = np.empty((n, self.p))
        for j in range(self.p):
            if j in self.numeric:
                lo, hi, step = self.numeric[j]
                count = round((hi - lo) / step)
                out[:, j] = rng.integers(count + 1, size=n) / count
            else:
                out[:, j] = rng.integers(len(self.categorical[j]), size=n)
        return out

    def mutate(self, x, rate, rng):
        out = x.copy()
        for j in range(self.p):
            if j in self.numeric:
                lo, hi, step = self.numeric[j]
                count = round((hi - lo) / step)
                out[:, j] = np.round(np.clip(x[:, j] + rng.normal(0, rate, len(x)), 0, 1) * count) / count
            else:
                mask = rng.random(len(x)) < rate
                out[mask, j] = rng.integers(len(self.categorical[j]), size=mask.sum())
        return out

    def distance(self, x, c, squared=False):
        nums = list(self.numeric)
        out = cdist(x[:, nums], c[:, nums], metric='sqeuclidean' if squared else 'cityblock') if nums else np.zeros((len(x), len(c)))
        for j, mat in self.categorical.items():
            out += mat[x[:, j].astype(int)[:, None], c[:, j].astype(int)[None, :]]
        return out / self.p

    def discrete(self, x):
        out = np.empty(x.shape, dtype=int)
        cards = []
        for j in range(self.p):
            if j in self.numeric:
                lo, hi, _ = self.numeric[j]
                out[:, j] = np.digitize(lo + x[:, j] * (hi - lo), self.bins[j])
                cards.append(len(self.bins[j]) + 1)
            else:
                out[:, j] = np.rint(x[:, j]).astype(int)
                cards.append(len(self.categorical[j]))
        return (out, cards)

    def encode(self, x):
        return np.column_stack([x[:, j:j + 1] if j in self.numeric else np.eye(len(self.categorical[j]))[x[:, j].astype(int)] for j in range(self.p - 1)])

    def domain(self, limit=100000):
        values = []
        for j in range(self.p):
            if j in self.numeric:
                lo, hi, step = self.numeric[j]
                values.append(np.linspace(0, 1, round((hi - lo) / step) + 1))
            else:
                values.append(np.arange(len(self.categorical[j])))
        if math.prod(map(len, values)) > limit:
            raise ValueError('Exact public domain exceeds certified enumeration limit')
        return np.asarray(list(product(*values)), dtype=float)

def reach(schema, policy='full', theta=7.0, theta2=None):
    """Actual normalized reach for the draft's single-attribute product edges."""
    p = schema.p
    if policy == 'global':
        return (1.0, [1.0])
    if schema.name.startswith('xor'):
        first = min(max(int(theta), 0), 7) / 7 / p
        terms = [first] + ([1 / 7 / p] * (p - 2) + [1 / p] if policy == 'full' else [0] * (p - 1))
    else:
        lo, hi, step = schema.numeric[0]
        first = min(math.floor(theta / step) * step, hi - lo) / (hi - lo) / p
        if policy == 'age':
            terms = [first] + [0] * (p - 1)
        elif schema.name == 'adult':
            hrs = 8.0 if theta2 is None else theta2
            terms = [first, min(hrs, 98) / 98 / p, 2 / 15 / p, 1 / p, 1 / p]
        else:
            balance = 500.0 if theta2 is None else theta2
            terms = [first, min(balance, 120000) / 120000 / p, 0.5 / p, 0.5 / p, 1 / p]
        if policy == 'two':
            terms = terms[:2] + [0] * (p - 2)
    return (max(terms), terms)

def bounds(r, tau, m):
    if tau <= 0:
        return (math.sqrt(2) if r > 0 else 0.0, 1.0)
    b = math.sqrt(2) * math.tanh(r / (2 * tau))
    u = 1 / math.sqrt(1 + (m - 1) * math.exp(-1 / tau))
    return (b, u)

def calibrate(r, tau, m, adjacency):
    b, u = bounds(r, tau, m)
    return {'B': b, 'U': u, 'BU': max(b, u)}[adjacency]

def votes(x, c, schema, tau, squared=False):
    dist = schema.distance(x, c, squared)
    if tau <= 0:
        return np.eye(len(c))[dist.argmin(1)]
    return softmax(-dist / tau, axis=1)

def histogram(x, c, schema, tau, squared=False, batch=1024):
    hist = np.zeros(len(c))
    entropy = 0.0
    sq = 0.0
    for start in range(0, len(x), batch):
        d = schema.distance(x[start:start + batch], c, squared)
        if tau <= 0:
            hist += np.bincount(d.argmin(1), minlength=len(c))
            sq += len(d)
        else:
            v = softmax(-d / tau, axis=1)
            hist += v.sum(0)
            entropy -= np.sum(v * np.log(np.maximum(v, 1e-300)))
            sq += np.sum(v * v)
    return (hist, entropy / max(len(x) * math.log(max(len(c), 2)), 1), sq / max(len(x), 1))

def exact_sensitivity(c, schema, tau, theta=1, policy='age', adjacency='B'):
    domain = schema.domain()
    v = votes(domain, c, schema, tau)
    b = 0.0
    for j in [0] if policy == 'age' else range(schema.p):
        other = [i for i in range(schema.p) if i != j]
        _, groups = np.unique(domain[:, other], axis=0, return_inverse=True)
        for g in np.unique(groups):
            ids = np.flatnonzero(groups == g)
            vals = domain[ids, j]
            if j in schema.numeric:
                lo, hi, step = schema.numeric[j]
                radius = (theta if j == 0 else step) / (hi - lo)
                allowed = (np.abs(vals[:, None] - vals[None, :]) <= radius + 1e-10) & (vals[:, None] != vals[None, :])
            else:
                allowed = vals[:, None] != vals[None, :]
            if np.any(allowed):
                b = max(b, float(cdist(v[ids], v[ids])[allowed].max()))
    u = float(np.linalg.norm(v, axis=1).max())
    return {'B': b, 'U': u, 'BU': max(b, u)}[adjacency]

def apd(c, schema):
    if len(c) < 2:
        return 0.0
    total = 0.0
    for start in range(0, len(c), 512):
        total += schema.distance(c[start:start + 512], c).sum()
    return float(total / (len(c) * (len(c) - 1)))

def evolve(train, schema, cfg):
    seed = cfg['seed']
    nout = cfg.get('nout', 512)
    rounds = cfg.get('rounds', 8)
    rng_init = np.random.default_rng(seed + 1000)
    rng_noise = np.random.default_rng(seed + 2000)
    rng_select = np.random.default_rng(seed + 3000)
    rng_mutate = np.random.default_rng(seed + 4000)
    c = schema.random(nout, rng_init)
    history = []
    snapshots = {}
    tau = cfg.get('tau', 0.0)
    policy = cfg.get('policy', 'full')
    r, terms = reach(schema, policy, cfg.get('theta', 7), cfg.get('theta2'))
    rho = cfg.get('rho', 0.01)
    adj = cfg.get('adjacency', 'B')
    mode = cfg.get('selection', 'hybrid')
    expansion = cfg.get('expansion', 2)
    squared = cfg.get('squared', False)
    calibration = cfg.get('calibration', 'analytic')
    start_all = perf_counter()
    for t in range(rounds):
        start = perf_counter()
        if calibration == 'exact':
            delta = exact_sensitivity(c, schema, tau, cfg.get('theta', 1), policy, adj)
        elif calibration in ('global', 'matched'):
            delta = calibrate(1.0, 0.0, len(c), adj)
        else:
            delta = calibrate(r, tau, len(c), adj)
        caltime = perf_counter() - start
        sigma = delta * math.sqrt(rounds / (2 * rho)) if rho > 0 else 0.0
        start = perf_counter()
        h, ent, norm2 = histogram(train, c, schema, tau, squared)
        votetime = perf_counter() - start
        noisy = h + rng_noise.normal(0, sigma, len(c))
        clean_top = np.argsort(-h)[:nout]
        start = perf_counter()
        sampling = mode == 'sample' or (mode == 'hybrid' and t < min(2, rounds))
        if sampling:
            clipped = np.maximum(noisy - cfg.get('hist_threshold', 0.0), 0.0)
            probs = clipped / clipped.sum() if clipped.sum() > 0 else np.full(len(c), 1 / len(c))
            ids = rng_select.choice(len(c), size=nout, replace=True, p=probs)
            ess = float(1 / np.sum(probs * probs))
        else:
            ids = np.argsort(-noisy, kind='stable')[:nout]
            ess = None
        selected = c[ids].copy()
        selecttime = perf_counter() - start
        stage = {'round': t, 'm_candidates': len(c), 'sensitivity': delta, 'sigma': sigma, 'reach': r, 'vote_entropy': ent, 'vote_norm_squared': norm2, 'contrast': float(np.std(h)), 'snr': float(np.std(h) / sigma) if sigma > 0 else None, 'clip_fraction': float(np.mean(noisy <= 0)), 'top_overlap': float(len(set(ids) & set(clean_top)) / nout), 'parent_fraction': len(set(ids)) / nout, 'sampling_ess': ess, 'calibration_s': caltime, 'vote_s': votetime, 'selection_s': selecttime}
        if t in {0, rounds // 2, rounds - 1}:
            snapshots[f'round_{t}'] = selected.copy()
        start = perf_counter()
        if t < rounds - 1:
            rate = 0.25 - (0.25 - 0.01) * (t / max(rounds - 1, 1)) ** 0.2
            if sampling:
                c = schema.mutate(selected, rate, rng_mutate)
            else:
                c = np.vstack([selected] + [schema.mutate(selected, rate, rng_mutate) for _ in range(expansion - 1)])
        stage['mutation_s'] = perf_counter() - start
        history.append(stage)
    return (selected, history, snapshots, {'evolution_s': perf_counter() - start_all, 'reach': r, 'reach_terms': terms})
