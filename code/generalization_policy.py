# Explicit single-coordinate product policies for the generalization study.

from __future__ import annotations
import math
import numpy as np
from scipy.spatial.distance import cdist
import core
SEMANTICS = 'single-coordinate-product'
WORKCLASS_GROUPS = [0, 1, 1, 1, 2, 2, 3, 3, 3]

def attribute_specs(schema):
    if schema.name.startswith('xor'):
        return {name: {'attribute': name, 'kind': 'numeric', 'radius': 1.0} for name in schema.names[:-1]} | {'Y': {'attribute': 'Y', 'kind': 'complete'}}
    if schema.name == 'adult':
        return {'age': {'attribute': 'age', 'kind': 'numeric', 'radius': 7.0}, 'hours.per.week': {'attribute': 'hours.per.week', 'kind': 'numeric', 'radius': 8.0}, 'education.num': {'attribute': 'education.num', 'kind': 'numeric', 'radius': 2.0}, 'workclass': {'attribute': 'workclass', 'kind': 'block', 'groups': WORKCLASS_GROUPS}, 'income': {'attribute': 'income', 'kind': 'complete'}}
    if schema.name == 'bank':
        return {'age': {'attribute': 'age', 'kind': 'numeric', 'radius': 7.0}, 'balance': {'attribute': 'balance', 'kind': 'numeric', 'radius': 500.0}, 'education': {'attribute': 'education', 'kind': 'metric-threshold', 'radius': 0.5}, 'job': {'attribute': 'job', 'kind': 'metric-threshold', 'radius': 0.5}, 'y': {'attribute': 'y', 'kind': 'complete'}}
    raise ValueError(schema.name)

def named_policies(schema):
    specs = attribute_specs(schema)

    def make(names):
        return {'semantics': SEMANTICS, 'attributes': [specs[n] for n in names]}
    if schema.name.startswith('xor'):
        return {'x1': make(['X1']), 'x1_x2': make(['X1', 'X2']), 'all_inputs': make(schema.names[:-1]), 'all_columns': make(schema.names)}
    secondary = schema.names[1]
    education = schema.names[2]
    result = {name: make([name]) for name in schema.names}
    result['age_secondary'] = make(['age', secondary])
    result['age_education'] = make(['age', education])
    result['all_predictors'] = make(schema.names[:-1])
    result['all_columns'] = make(schema.names)
    return result

def graph_for_attribute(schema, spec):
    """Return the finite normalized domain and exact allowed-edge matrix."""
    j = schema.names.index(spec['attribute'])
    if j in schema.numeric:
        if spec['kind'] != 'numeric':
            raise ValueError('A numeric graph requires a raw-unit radius')
        lo, hi, step = schema.numeric[j]
        radius = float(spec['radius'])
        if not math.isfinite(radius) or radius < 0:
            raise ValueError('Invalid radius')
        values = np.linspace(0, 1, round((hi - lo) / step) + 1)
        if len(values) > 2048:
            raise ValueError('Dense edge enumeration exceeds the structural-policy limit')
        d = np.abs(values[:, None] - values[None, :])
        allowed = (d <= radius / (hi - lo) + 1e-12) & (d > 0)
        metric = d
    else:
        metric = schema.categorical[j]
        values = np.arange(len(metric), dtype=float)
        if spec['kind'] == 'complete':
            allowed = ~np.eye(len(values), dtype=bool)
        elif spec['kind'] == 'metric-threshold':
            radius = float(spec['radius'])
            if not math.isfinite(radius) or radius < 0:
                raise ValueError('Invalid radius')
            allowed = (metric <= radius + 1e-12) & (metric > 0)
        elif spec['kind'] == 'block':
            groups = np.asarray(spec['groups'])
            if len(groups) != len(values):
                raise ValueError('Block assignment must cover every category')
            allowed = (groups[:, None] == groups[None, :]) & (metric > 0)
        else:
            raise ValueError(spec['kind'])
    return (j, values, allowed, metric)

def validated_specs(schema, policy):
    if not isinstance(policy, dict) or policy.get('semantics') != SEMANTICS:
        raise ValueError('Explicit product-policy specification required')
    seen = set()
    specs = []
    for spec in policy['attributes']:
        if spec['attribute'] in seen:
            raise ValueError('Duplicate protected attribute')
        seen.add(spec['attribute'])
        if spec['attribute'] not in schema.names:
            raise ValueError('Unknown protected attribute')
        specs.append(spec)
    if not specs:
        raise ValueError('At least one active graph is required')
    return specs

def compile_policy(schema, policy):
    return [graph_for_attribute(schema, spec) for spec in validated_specs(schema, policy)]

def policy_reach(schema, policy):
    terms = [0.0] * schema.p
    for spec in validated_specs(schema, policy):
        j = schema.names.index(spec['attribute'])
        if j in schema.numeric:
            if spec['kind'] != 'numeric':
                raise ValueError('Numeric policies require raw-unit radii')
            radius = float(spec['radius'])
            if not math.isfinite(radius) or radius < 0:
                raise ValueError('Invalid radius')
            lo, hi, step = schema.numeric[j]
            terms[j] = min(math.floor(radius / step + 1e-12) * step, hi - lo) / (hi - lo) / schema.p
        else:
            _, _, allowed, metric = graph_for_attribute(schema, spec)
            terms[j] = float(metric[allowed].max() / schema.p) if allowed.any() else 0.0
    return (max(terms), terms)

def dispatch_reach(schema, policy='full', theta=7.0, theta2=None):
    if isinstance(policy, dict):
        return policy_reach(schema, policy)
    return LEGACY_REACH(schema, policy, theta, theta2)
LEGACY_REACH = core.reach

def exact_policy_sensitivity(candidates, schema, tau, policy):
    """Small-domain protocol check, never used as an unverified large-domain shortcut."""
    domain = schema.domain()
    v = core.votes(domain, candidates, schema, tau)
    worst = 0.0
    for j, values, allowed, _ in compile_policy(schema, policy):
        other = [i for i in range(schema.p) if i != j]
        _, groups = np.unique(domain[:, other], axis=0, return_inverse=True)
        for g in np.unique(groups):
            ids = np.flatnonzero(groups == g)
            order = np.argsort(domain[ids, j])
            ids = ids[order]
            if len(ids) != len(values):
                raise AssertionError('Incomplete public fiber')
            if allowed.any():
                worst = max(worst, float(cdist(v[ids], v[ids])[allowed].max()))
    return worst

def hard_crossing_witness(candidates, schema, policy):
    """Certify a protected crossing by an allowed edge or connected fiber path.

    Failure to find a witness is recorded as uncertified, not sensitivity zero.
    Only public candidate populations and policy graphs are inspected.
    """
    for spec in validated_specs(schema, policy):
        j = schema.names.index(spec['attribute'])
        if j in schema.numeric:
            lo, hi, step = schema.numeric[j]
            if spec['kind'] != 'numeric' or float(spec['radius']) < step:
                continue
            points = np.repeat(candidates[:64], 2, axis=0)
            points[::2, j] = 0.0
            points[1::2, j] = 1.0
            distances = schema.distance(points, candidates)
            winners = distances.argmin(1)
            if len(candidates) < 2:
                continue
            top = np.partition(distances, 1, axis=1)[:, :2]
            unique = top[:, 1] - top[:, 0] > 1e-12
            passed = np.flatnonzero((winners[::2] != winners[1::2]) & unique[::2] & unique[1::2])
            if len(passed):
                i = int(passed[0])
                return {'kind': 'connected_numeric_fiber', 'attribute': schema.names[j], 'left': points[2 * i].tolist(), 'right': points[2 * i + 1].tolist(), 'winner_left': int(winners[2 * i]), 'winner_right': int(winners[2 * i + 1]), 'raw_path_step': step, 'raw_policy_radius': float(spec['radius']), 'endpoint_winner_margins': [float(top[2 * i, 1] - top[2 * i, 0]), float(top[2 * i + 1, 1] - top[2 * i + 1, 0])]}
            continue
        j, values, allowed, _ = graph_for_attribute(schema, spec)
        if not allowed.any():
            continue
        for start in range(0, min(len(candidates), 64), 8):
            contexts = candidates[start:start + 8]
            points = np.repeat(contexts, len(values), axis=0)
            points[:, j] = np.tile(values, len(contexts))
            distances = schema.distance(points, candidates)
            winners = distances.argmin(1).reshape(len(contexts), -1)
            if len(candidates) < 2:
                continue
            top = np.partition(distances, 1, axis=1)[:, :2]
            unique = (top[:, 1] - top[:, 0] > 1e-12).reshape(len(contexts), -1)
            for i, row in enumerate(winners):
                pairs = np.argwhere(allowed & (row[:, None] != row[None, :]) & unique[i, :, None] & unique[i, None, :])
                if len(pairs):
                    a, b = map(int, pairs[0])
                    left = points[i * len(values) + a]
                    right = points[i * len(values) + b]
                    distance = float(schema.distance(left[None], right[None])[0, 0])
                    return {'kind': 'allowed_categorical_edge', 'attribute': schema.names[j], 'left': left.tolist(), 'right': right.tolist(), 'winner_left': int(row[a]), 'winner_right': int(row[b]), 'edge_distance': distance}
    return None
