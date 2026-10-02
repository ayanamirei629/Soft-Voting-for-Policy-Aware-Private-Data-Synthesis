# Dense BF-Soft sweet-spot, policy-graph, and workload experiments.

from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import core
from data import ROOT, load, schema_for
from metrics import evaluate
from run import atomic_json, base, identity
HOME = ROOT / 'extended_study'
RESULTS = HOME / 'results'
VERSION = 'draft-v1-extended-sweetspot'
SEEDS = list(range(1200, 1206))
Q_GRID = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97, 0.995]
GRAPH_Q = [0.2, 0.5, 0.8, 0.95]
ADJ_Q = [0.2, 0.5, 0.8, 0.95]
SWEET_BUDGETS = {'adult': [1e-05, 3e-05, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.01, 0.03, 0.1], 'bank': [1e-05, 3e-05, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.01, 0.03, 0.1], 'xor3': [0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.18, 0.3, 0.6, 1.0]}
GRAPH_BUDGETS = {'adult': [0.0001, 0.0006, 0.003], 'bank': [0.0001, 0.0006, 0.003]}
ADJ_BUDGETS = {'adult': [0.0001, 0.0006, 0.003], 'bank': [0.0001, 0.0006, 0.003]}
MATCHED_BUDGET = {'adult': 0.0006, 'bank': 0.0006, 'xor3': 0.18}
DEFAULT_THETA = {'adult': 7.0, 'bank': 7.0, 'xor3': 1.0}
GRAPH_SPECS = {'adult': [dict(graph='age-local', policy='age', theta=7.0, theta2=None), dict(graph='age-hours', policy='two', theta=7.0, theta2=8.0), dict(graph='full-product', policy='full', theta=7.0, theta2=8.0), dict(graph='complete', policy='global', theta=7.0, theta2=None)], 'bank': [dict(graph='age-local', policy='age', theta=7.0, theta2=None), dict(graph='age-balance', policy='two', theta=7.0, theta2=500.0), dict(graph='full-hierarchy', policy='full', theta=7.0, theta2=500.0), dict(graph='complete', policy='global', theta=7.0, theta2=None)]}

def tau_for_q(dataset, policy, theta, theta2, q):
    schema = schema_for(dataset)
    r, _ = core.reach(schema, policy, theta, theta2)
    return r / (2 * math.atanh(q))

def _algorithm_key(cfg):
    ignored = {'roles', 'q_target', 'graph'}
    return json.dumps({k: v for k, v in cfg.items() if k not in ignored}, sort_keys=True)

def experiment_configs():
    configs = {}

    def add(role, dataset, rho, seed, q=1.0, policy='global', theta=7.0, theta2=None, adjacency='B', calibration=None, graph='hard'):
        tau = 0.0 if q >= 1 else tau_for_q(dataset, policy, theta, theta2, q)
        if calibration is None:
            calibration = 'global' if tau == 0 else 'analytic'
        cfg = base(dataset, phase='extended_study', seed=seed, n=0, nout=512, rounds=8, expansion=2, rho=float(rho), tau=float(tau), policy=policy, theta=float(theta), theta2=theta2, adjacency=adjacency, calibration=calibration, selection='hybrid', backend='cuda', q_target=float(q), graph=graph, roles=[role])
        key = _algorithm_key(cfg)
        if key in configs:
            configs[key]['roles'] = sorted(set(configs[key]['roles'] + [role]))
        else:
            configs[key] = cfg
    for dataset, budgets in SWEET_BUDGETS.items():
        theta = DEFAULT_THETA[dataset]
        for seed in SEEDS:
            for rho in budgets:
                add('sweet-hard', dataset, rho, seed)
                for q in Q_GRID:
                    add('sweet-soft', dataset, rho, seed, q, 'age', theta, None, 'B', 'analytic', 'age-local')
            rho = MATCHED_BUDGET[dataset]
            for q in Q_GRID:
                add('matched-soft', dataset, rho, seed, q, 'age', theta, None, 'B', 'matched', 'age-local')
    for dataset, specs in GRAPH_SPECS.items():
        for seed in SEEDS:
            for rho in GRAPH_BUDGETS[dataset]:
                add('graph-hard', dataset, rho, seed)
                for spec in specs:
                    for q in GRAPH_Q:
                        add('graph-soft', dataset, rho, seed, q, spec['policy'], spec['theta'], spec['theta2'], 'B', 'analytic', spec['graph'])
    for dataset, budgets in ADJ_BUDGETS.items():
        theta = DEFAULT_THETA[dataset]
        for seed in SEEDS:
            for rho in budgets:
                for adjacency in ['U', 'BU']:
                    add('adjacency-hard', dataset, rho, seed, 1.0, 'global', theta, None, adjacency, 'global', f'{adjacency}-hard')
                    for q in ADJ_Q:
                        add('adjacency-soft', dataset, rho, seed, q, 'age', theta, None, adjacency, 'analytic', f'age-local-{adjacency}')
    return sorted(configs.values(), key=lambda c: (c['dataset'], c['seed'], c['rho'], c['tau'], c['policy'], c['adjacency'], c['calibration']))

def source_hashes():
    names = ['extended_study.py', 'core.py', 'gpu_backend.py', 'metrics.py', 'data.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}

def freeze_manifest():
    path = HOME / 'manifest.json'
    if path.exists():
        obj = json.loads(path.read_text())
        print('Manifest already frozen:', len(obj['configs']))
        return obj
    HOME.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    configs = experiment_configs()
    obj = dict(version=VERSION, frozen_before_execution=True, seeds=SEEDS, q_grid=Q_GRID, graph_q=GRAPH_Q, adjacency_q=ADJ_Q, configs=configs, sources=source_hashes())
    atomic_json(path, obj)
    print('Frozen configs:', len(configs))
    return obj

def worker(cfg):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rid = identity(cfg)
    path = RESULTS / f'{rid}.json'
    if path.exists():
        return {'id': rid, 'cached': True}
    try:
        import torch
        from gpu_backend import histogram_cuda
        torch.set_num_threads(1)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            core.histogram = histogram_cuda
            schema, train, _, test = load(cfg['dataset'], cfg['n'], cfg.get('data_seed', 0))
            syn, history, snapshots, extra = core.evolve(train, schema, cfg)
            metrics = evaluate(train, syn, test, schema)
            r, terms = core.reach(schema, cfg['policy'], cfg['theta'], cfg.get('theta2'))
            q = math.tanh(r / (2 * cfg['tau'])) if cfg['tau'] > 0 else 1.0
            deltas = [row['sensitivity'] for row in history]
            np.savez_compressed(path.with_suffix('.npz'), synthetic=syn, **snapshots)
            atomic_json(path, dict(id=rid, version=VERSION, config=cfg, train_n=len(train), metrics=metrics, q_actual=q, sensitivity_mean=float(np.mean(deltas)), history=history, wall_s=time.perf_counter() - started, **extra))
        return {'id': rid, 'dataset': cfg['dataset'], 'seconds': round(time.perf_counter() - started, 2)}
    except Exception:
        error = traceback.format_exc()
        atomic_json(path.with_suffix('.failure.json'), dict(config=cfg, error=error))
        return {'id': rid, 'error': error}

def run(workers=2, limit=0):
    obj = freeze_manifest()
    configs = obj['configs'][:limit or None]
    pending = [cfg for cfg in configs if not (RESULTS / f'{identity(cfg)}.json').exists()]
    print('extended study total', len(configs), 'pending', len(pending), 'workers', workers, flush=True)
    started = time.perf_counter()
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, cfg) for cfg in pending]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            failures += int('error' in result)
            if i % 48 == 0 or i == len(futures) or 'error' in result:
                print(json.dumps(dict(completed=i, total=len(futures), elapsed_s=round(time.perf_counter() - started, 1), failures=failures, last=result)), flush=True)
    if failures:
        raise SystemExit(1)

def audit():
    obj = json.loads((HOME / 'manifest.json').read_text())
    rounds = 0
    bad = []
    times = []
    for cfg in obj['configs']:
        path = RESULTS / f'{identity(cfg)}.json'
        if not path.exists():
            bad.append(identity(cfg))
            continue
        row = json.loads(path.read_text())
        history = row['history']
        rounds += len(history)
        spent = sum((h['sensitivity'] ** 2 / (2 * h['sigma'] ** 2) for h in history))
        if not math.isclose(spent, cfg['rho'], rel_tol=1e-10, abs_tol=1e-14):
            bad.append(identity(cfg))
        if not math.isclose(row['q_actual'], cfg['q_target'], rel_tol=1e-10, abs_tol=1e-12):
            bad.append(identity(cfg))
        times.append(row['wall_s'])
    result = dict(configs=len(obj['configs']), rounds=rounds, failures=bad, median_wall_s=float(np.median(times)) if times else None, total_wall_s=float(np.sum(times)) if times else None, manifest_sha256=hashlib.sha256((HOME / 'manifest.json').read_bytes()).hexdigest())
    atomic_json(HOME / 'audit.json', result)
    print(json.dumps(result, indent=2))
    if bad:
        raise SystemExit(1)
