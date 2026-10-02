# Prospective public-pilot forecasts and temperature/reach experiments.

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
import pandas as pd
from scipy.special import erf
from itertools import combinations
from threadpoolctl import threadpool_limits
import core
from data import ROOT, load
from metrics import evaluate, marginal_errors
from benefit_study import certify_hard
from crossover_study import GRIDS
from run import base, identity, atomic_json
HOME = ROOT / 'interaction_transfer'
DATA = {}
TAUS = {d: [0.015, 0.03, 0.045, 0.06, 0.09] for d in ['adult', 'bank']}
TAUS['xor3'] = [0.005, 0.01, 0.02, 0.04, 0.08]
THETAS = {'adult': [2, 4, 8, 16, 28], 'bank': [2, 4, 8, 16, 28], 'xor3': [1, 2, 3, 4]}
BUDGETS = {'adult': [0.0001, 0.0006, 0.003], 'bank': [0.0001, 0.0006, 0.003], 'xor3': [0.03, 0.18, 1.0]}
ISO = {'adult': [(0.015, 2), (0.03, 4), (0.06, 8)], 'bank': [(0.015, 2), (0.03, 4), (0.06, 8)], 'xor3': [(0.01, 1), (0.02, 2), (0.04, 4)]}

def conditions(dataset):
    result = {(float(rho), float(tau), float(theta)) for rho in BUDGETS[dataset] for tau in TAUS[dataset] for theta in THETAS[dataset]}
    result.update(((float(rho), float(tau), float(theta)) for rho in GRIDS[dataset] for tau, theta in ISO[dataset]))
    return sorted(result)

def source_hashes():
    files = ['interaction_transfer.py', 'core.py', 'gpu_backend.py', 'metrics.py', 'data.py', 'INTERACTION_TRANSFER_PROTOCOL.md']
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}

def config(dataset, rho, tau, theta, seed, stage):
    return base(dataset, phase='transfer_' + stage, n=0, nout=512, rounds=8, expansion=2, policy='age', theta=theta, tau=tau, rho=rho, target_rho=rho, seed=seed, data_seed=70 + (seed - 910) % 3 if dataset == 'xor3' and stage == 'target' else 60 if dataset == 'xor3' else 0, calibration='global' if tau == 0 else 'analytic', backend='cuda')

def manifest(stage):
    path = HOME / f'{stage}_manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    seeds = range(910, 916) if stage == 'target' else range(950, 953)
    configs = []
    for dataset in TAUS:
        cells = conditions(dataset)
        for seed in seeds:
            configs += [config(dataset, rho, tau, theta, seed, stage) for rho, tau, theta in cells]
            configs += [config(dataset, rho, 0.0, 1.0, seed, stage) for rho in sorted({c[0] for c in cells})]
    obj = dict(stage=stage, frozen_before_execution=True, configs=configs, sources=source_hashes())
    atomic_json(path, obj)
    return obj

def inputs(dataset, stage, data_seed):
    key = (dataset, stage == 'target', data_seed)
    if key not in DATA:
        schema, train, val, test = load(dataset, 0, data_seed)
        target_n = len(train)
        if stage == 'target':
            DATA[key] = (schema, train, test, target_n)
        elif dataset == 'xor3':
            schema, pilot, reference, _ = load(dataset, 3000, 60)
            DATA[key] = (schema, pilot, reference, target_n)
        else:
            perm = np.random.default_rng(20260908).permutation(len(val))
            DATA[key] = (schema, val[perm[:3000]], val[perm[3000:]], target_n)
    return DATA[key]

def worker(cfg):
    stage = cfg['phase'].replace('transfer_', '')
    folder = HOME / 'results' / stage
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{identity(cfg)}.json'
    if path.exists():
        return {'cached': True}
    try:
        import torch
        from gpu_backend import histogram_cuda
        torch.set_num_threads(1)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            schema, train, reference, target_n = inputs(cfg['dataset'], stage, cfg['data_seed'])
            actual = dict(cfg)
            if stage == 'pilot_scaled':
                actual['rho'] = cfg['target_rho'] * (target_n / len(train)) ** 2
            certs = []

            def histogram(x, c, s, tau, squared=False):
                if tau == 0:
                    certs.append(certify_hard(c, s, cfg))
                return histogram_cuda(x, c, s, tau, squared)
            core.histogram = histogram
            syn, history, snapshots, extra = core.evolve(train, schema, actual)
            metrics = evaluate(train, syn, reference, schema) if stage == 'target' else marginal_errors(reference, syn, schema)
            metric = 'hoerr_bits' if cfg['dataset'] == 'xor3' else 'l1_2way'
            r, _ = core.reach(schema, 'age', cfg['theta'])
            q = math.tanh(r / (2 * cfg['tau'])) if cfg['tau'] > 0 else 1.0
            np.savez_compressed(path.with_suffix('.npz'), synthetic=syn, **snapshots)
            atomic_json(path, dict(id=identity(cfg), config=cfg, actual_rho=actual['rho'], train_n=len(train), target_n=target_n, error=metrics[metric], metrics=metrics, noise_ratio=q, history=history, all_hard_rounds_certified=all(certs) if certs else None, wall_s=time.perf_counter() - started, **extra))
            return {'dataset': cfg['dataset'], 'seconds': round(time.perf_counter() - started, 2)}
    except Exception:
        error = traceback.format_exc()
        atomic_json(path.with_suffix('.failure.json'), dict(config=cfg, error=error))
        return {'error': error}

def folded_normal(bias, sd):
    bias = np.abs(np.asarray(bias, float))
    sd = np.asarray(sd, float)
    ratio = np.divide(bias, sd, out=np.zeros_like(bias), where=sd > 0)
    return np.where(sd > 0, sd * math.sqrt(2 / math.pi) * np.exp(-0.5 * ratio ** 2) + bias * erf(ratio / math.sqrt(2)), bias)

def workload(schema, candidates, reference):
    if schema.name == 'xor3':
        a = np.column_stack([(candidates[:, :-1] >= 0.5).astype(int), candidates[:, -1].astype(int)])
        b = np.column_stack([(reference[:, :-1] >= 0.5).astype(int), reference[:, -1].astype(int)])
        cards = [2] * schema.p
        cols = [tuple(range(schema.p))]
    else:
        a, cards = schema.discrete(candidates)
        b, _ = schema.discrete(reference)
        cols = list(combinations(range(schema.p), 2))
    blocks = []
    for indices in cols:
        sizes = [cards[j] for j in indices]
        size = math.prod(sizes)
        ci = np.ravel_multi_index(a[:, indices].T, sizes)
        ri = np.ravel_multi_index(b[:, indices].T, sizes)
        A = np.eye(size)[ci].T
        truth = np.bincount(ri, minlength=size) / len(reference)
        blocks.append((A, truth))
    return blocks

def marginal_loss(blocks, weights, noise_sd):
    return float(np.mean([folded_normal(A @ weights - truth, noise_sd * np.sqrt(np.sum(A * A, axis=1))).sum() for A, truth in blocks]))

def initial_forecasts():
    path = HOME / 'initial_forecasts.json'
    if path.exists():
        return json.loads(path.read_text())
    import torch
    from gpu_backend import histogram_cuda
    torch.set_num_threads(1)
    rows = []
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for dataset in TAUS:
            schema, pilot, reference, n = inputs(dataset, 'pilot_scaled', 60 if dataset == 'xor3' else 0)
            for seed in range(950, 953):
                candidates = schema.random(512, np.random.default_rng(seed + 1000))
                blocks = workload(schema, candidates, reference)
                hard, _, _ = histogram_cuda(pilot, candidates, schema, 0)
                soft_hist = {tau: histogram_cuda(pilot, candidates, schema, tau)[0] for tau in TAUS[dataset]}
                for rho, tau, theta in conditions(dataset):
                    soft = soft_hist[tau]
                    r, _ = core.reach(schema, 'age', theta)
                    q = math.tanh(r / (2 * tau))
                    sd = math.sqrt(8 / rho) / n
                    eh = marginal_loss(blocks, hard / len(pilot), sd)
                    es = marginal_loss(blocks, soft / len(pilot), q * sd)
                    rows.append(dict(dataset=dataset, rho=rho, tau=tau, theta=theta, seed=seed, hard=eh, soft=es, gain=eh - es, q=q, snr_ratio=float(np.std(soft) / np.std(hard) / q)))
    result = dict(rows=rows, seconds=time.perf_counter() - start, sources=source_hashes())
    atomic_json(path, result)
    return result

def collect(stage):
    rows = []
    for path in (HOME / 'results' / stage).glob('*.json'):
        if '.failure.' in path.name:
            continue
        r = json.loads(path.read_text())
        rows.append(dict(**r['config'], id=r['id'], error=r['error'], wall_s=r['wall_s'], q=r['noise_ratio'], train_n=r['train_n'], target_n=r['target_n'], certified=r['all_hard_rounds_certified']))
    return pd.DataFrame(rows)

def freeze_forecasts():
    path = HOME / 'forecasts.json'
    if path.exists():
        print('Forecasts already frozen:', path)
        return
    assert not list((HOME / 'results/target').glob('*.json')), 'Target labels exist before forecast freeze'
    forecast = {}
    keys = ['dataset', 'target_rho', 'tau', 'theta']
    for stage in ['pilot_scaled', 'pilot_unscaled']:
        f = collect(stage)
        assert len(f) == len(manifest(stage)['configs'])
        hard = f[f.tau == 0].groupby(['dataset', 'target_rho']).error.mean()
        soft = f[f.tau > 0].groupby(keys).error.mean()
        for index, value in soft.items():
            d, rho, tau, theta = index
            key = (d, rho, tau, theta)
            forecast.setdefault(key, dict(dataset=d, rho=rho, tau=tau, theta=theta))
            forecast[key][stage + '_hard'] = float(hard.loc[d, rho])
            forecast[key][stage + '_soft'] = float(value)
            forecast[key][stage + '_gain'] = float(hard.loc[d, rho] - value)
    initial = pd.DataFrame(initial_forecasts()['rows']).groupby(['dataset', 'rho', 'tau', 'theta'])[['hard', 'soft', 'gain', 'q', 'snr_ratio']].mean()
    for key, values in initial.iterrows():
        forecast[key].update({'initial_' + k: float(v) for k, v in values.items()})
    atomic_json(path, dict(frozen_before_target=True, rows=list(forecast.values()), sources=source_hashes(), baseline_rules={'noise_only': 'q <= .5', 'snr': 'initial snr_ratio > 1'}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(HOME / 'forecast_lock.json', dict(sha256=digest, target_manifest_sha256=hashlib.sha256((HOME / 'target_manifest.json').read_bytes()).hexdigest()))
    print('Frozen forecasts', len(forecast), digest)

def run_stage(stage):
    obj = manifest(stage)
    if stage == 'target':
        lock = json.loads((HOME / 'forecast_lock.json').read_text())
        assert hashlib.sha256((HOME / 'forecasts.json').read_bytes()).hexdigest() == lock['sha256']
    pending = [c for c in obj['configs'] if not (HOME / 'results' / stage / f'{identity(c)}.json').exists()]
    print(stage, 'total', len(obj['configs']), 'pending', len(pending), flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(worker, c) for c in pending]
        for i, future in enumerate(as_completed(tasks), 1):
            result = future.result()
            if 'error' in result:
                raise RuntimeError(result)
            if i % 48 == 0 or i == len(tasks):
                print(stage, i, '/', len(tasks), 'elapsed', round(time.perf_counter() - start), flush=True)
