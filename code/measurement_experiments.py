# Measure sensitivity versus temperature, policy reach, and calibration runtime.

from pathlib import Path
from itertools import product
import math, time
import numpy as np
import pandas as pd
from scipy.special import softmax
from core import Schema
OUT = Path('outputs/measurements')

def identity_metric(size):
    return 1 - np.eye(size)

def binary_schema(p):
    return Schema(f'binary_{p}', tuple(map(str, range(p))), {}, {j: identity_metric(2) for j in range(p)}, {})

def binary_domain(p):
    return np.asarray(list(product([0.0, 1.0], repeat=p)))

def hamming_edges(p):
    ids = np.arange(2 ** p)
    left, right = ([], [])
    for bit in range(p):
        target = ids ^ 1 << bit
        mask = ids < target
        left.extend(ids[mask])
        right.extend(target[mask])
    return (np.asarray(left), np.asarray(right))

def candidates(domain, m, geometry, rng):
    if geometry == 'uniform':
        return domain[rng.choice(len(domain), m, replace=m > len(domain))]
    if geometry == 'clustered':
        pool = domain[domain[:, :max(1, domain.shape[1] // 2)].sum(1) <= 1]
        return pool[rng.choice(len(pool), m, replace=m > len(pool))]
    base = domain[rng.choice(len(domain), min(8, len(domain)), replace=False)]
    return base[rng.choice(len(base), m, replace=True)]

def temperature_sensitivity():
    p = 8
    domain = binary_domain(p)
    schema = binary_schema(p)
    left, right = hamming_edges(p)
    temperatures = [0, 0.001, 0.002, 0.004, 0.008, 0.012, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6, 1, 2, 3, 5, 10, 20, 50, 100]
    rows = []
    reach = 1 / p
    for m in [32, 64, 128, 256]:
        for geometry in ['uniform', 'clustered', 'duplicates']:
            for seed in range(5):
                rng = np.random.default_rng(121000 + 1000 * m + 10 * seed)
                candidate = candidates(domain, m, geometry, rng)
                distance = schema.distance(domain, candidate)
                hard = np.eye(m)[distance.argmin(1)]
                hard_exact = float(np.linalg.norm(hard[left] - hard[right], axis=1).max())
                for tau in temperatures:
                    if tau == 0:
                        exact = hard_exact
                        bound = math.sqrt(2)
                    else:
                        vote = softmax(-distance / tau, axis=1)
                        exact = float(np.linalg.norm(vote[left] - vote[right], axis=1).max())
                        bound = math.sqrt(2) * math.tanh(reach / (2 * tau))
                    rows.append({'p': p, 'domain_size': len(domain), 'm': m, 'geometry': geometry, 'seed': seed, 'tau': tau, 'reach': reach, 'exact_soft': exact, 'bound_soft': bound, 'exact_hard': hard_exact, 'bound_hard': math.sqrt(2)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'temperature.csv', index=False)
    print(f'temperature sensitivity: {len(frame)} measurements', flush=True)
    return frame

def exact_ordinal_nxn(n, seed, tau=0.04, chunk=64):
    rng = np.random.default_rng(151000 + seed * 100000 + n)
    domain = np.linspace(0.0, 1.0, n)
    candidate = np.sort(rng.uniform(0.0, 1.0, n))
    maximum = 0.0
    previous = None
    for start in range(0, n, chunk):
        points = domain[start:start + chunk]
        vote = softmax(-np.abs(points[:, None] - candidate[None, :]) / tau, axis=1)
        if previous is not None:
            maximum = max(maximum, float(np.linalg.norm(vote[0] - previous)))
        if len(vote) > 1:
            maximum = max(maximum, float(np.linalg.norm(vote[1:] - vote[:-1], axis=1).max()))
        previous = vote[-1]
    return maximum

def exact_hard_ordinal_nxn(n, seed, chunk=64):
    rng = np.random.default_rng(151000 + seed * 100000 + n)
    domain = np.linspace(0.0, 1.0, n)
    candidate = np.sort(rng.uniform(0.0, 1.0, n))
    maximum = 0.0
    previous = None
    root_two = math.sqrt(2)
    for start in range(0, n, chunk):
        points = domain[start:start + chunk]
        winners = np.abs(points[:, None] - candidate[None, :]).argmin(axis=1)
        if previous is not None and winners[0] != previous:
            maximum = root_two
        if len(winners) > 1 and np.any(winners[1:] != winners[:-1]):
            maximum = root_two
        previous = winners[-1]
    return maximum

def runtime_nxn():
    rows = []
    tau = 0.04
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for n in sizes:
        for seed in range(5):
            reach = 1 / (n - 1)
            analytic_times = []
            for _ in range(5000):
                started = time.perf_counter_ns()
                math.sqrt(2) * math.tanh(reach / (2 * tau))
                analytic_times.append(time.perf_counter_ns() - started)
            if seed % 2 == 0:
                started = time.perf_counter_ns()
                soft_exact = exact_ordinal_nxn(n, seed, tau=tau)
                soft_exact_ns = time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                hard_exact = exact_hard_ordinal_nxn(n, seed)
                hard_exact_ns = time.perf_counter_ns() - started
            else:
                started = time.perf_counter_ns()
                hard_exact = exact_hard_ordinal_nxn(n, seed)
                hard_exact_ns = time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                soft_exact = exact_ordinal_nxn(n, seed, tau=tau)
                soft_exact_ns = time.perf_counter_ns() - started
            rows.append({'N': n, 'seed': seed, 'tau': tau, 'theta_steps': 1, 'exact_sensitivity': soft_exact, 'soft_exact_sensitivity': soft_exact, 'hard_exact_sensitivity': hard_exact, 'analytic_bound': math.sqrt(2) * math.tanh(reach / (2 * tau)), 'analytic_ms': np.median(analytic_times) / 1000000.0, 'exact_ms': soft_exact_ns / 1000000.0, 'soft_exact_ms': soft_exact_ns / 1000000.0, 'hard_exact_ms': hard_exact_ns / 1000000.0, 'distance_evaluations': n * n, 'benchmark_scope': 'N-by-N candidate distances plus full adjacent-edge scan; Hard winner indices or Soft probability vectors'})
        print(f'N-by-N sensitivity runtime: N={n}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'candidates.csv', index=False)
    return frame

def ordinal_votes(domain, candidates, tau):
    distances = np.abs(domain[:, None] - candidates[None, :])
    if tau == 0:
        return np.eye(len(candidates))[distances.argmin(1)]
    return softmax(-distances / tau, axis=1)

def ordinal_edges(size, theta):
    left, right = ([], [])
    for gap in range(1, theta + 1):
        left.extend(range(size - gap))
        right.extend(range(gap, size))
    return (np.asarray(left), np.asarray(right))

def exact_from_votes(votes, left, right):
    return float(np.linalg.norm(votes[left] - votes[right], axis=1).max())

def ordinal_benchmark():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    size = 64
    domain = np.linspace(0, 1, size)
    tau = 0.04
    for seed in range(12):
        rng = np.random.default_rng(9400 + seed)
        candidates = np.sort(rng.choice(domain, 20, replace=False))
        soft = ordinal_votes(domain, candidates, tau)
        hard = ordinal_votes(domain, candidates, 0)
        for theta in [1, 2, 4, 8, 16, 32]:
            left, right = ordinal_edges(size, theta)
            reach = theta / (size - 1)
            rows.append({'study': 'theta', 'seed': seed, 'theta': theta, 'N': len(candidates), 'soft_exact': exact_from_votes(soft, left, right), 'soft_bound': math.sqrt(2) * math.tanh(reach / (2 * tau)), 'hard_exact': exact_from_votes(hard, left, right)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'reach.csv', index=False)
    return frame
