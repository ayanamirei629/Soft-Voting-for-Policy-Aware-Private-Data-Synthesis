# Checkpointed conference experiment driver. Run from the paper folder.

import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse, hashlib, json, platform, sys, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from data import prepare, load, ROOT
from core import evolve
from metrics import evaluate
OUT = ROOT / 'results'
VERSION = 'draft-v1'

def identity(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]

def atomic_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)

def worker(cfg):
    out = OUT / cfg['phase']
    out.mkdir(parents=True, exist_ok=True)
    rid = identity(cfg)
    path = out / f'{rid}.json'
    if path.exists():
        return {'id': rid, 'cached': True}
    t = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            if cfg.get('backend') == 'cuda':
                import core, torch
                from gpu_backend import histogram_cuda
                torch.set_num_threads(1)
                core.histogram = histogram_cuda
            schema, train, val, test = load(cfg['dataset'], cfg['n'], cfg.get('data_seed', 0))
            syn, history, snapshots, extra = evolve(train, schema, cfg)
            validation = evaluate(train, syn, val, schema)
            testing = evaluate(train, syn, test, schema) if cfg['phase'] == 'confirm' else None
            result = {'id': rid, 'version': VERSION, 'config': cfg, 'train_n_actual': len(train), 'validation': validation, 'test': testing, 'history': history, **extra, 'wall_s': time.perf_counter() - t, 'peak_working_set_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)}
            np.savez_compressed(out / f'{rid}.npz', synthetic=syn, **snapshots)
            atomic_json(path, result)
        return {'id': rid, 'seconds': result['wall_s'], 'auc': validation['auc_tree' if cfg['dataset'].startswith('xor') else 'auc_lr'], 'pwerr': validation['l1_2way']}
    except Exception:
        failure = {'config': cfg, 'traceback': traceback.format_exc()}
        atomic_json(out / f'{rid}.failure.json', failure)
        return {'id': rid, 'error': failure['traceback']}

def base(dataset, phase='explore', seed=0, **kwargs):
    c = dict(dataset=dataset, phase=phase, seed=seed, data_seed=0, n=8000, nout=300, rounds=6, expansion=2, tau=0.0, rho=0.01, policy='full', theta=1 if dataset.startswith('xor') else 7.0, theta2=None, adjacency='B', calibration='analytic', selection='hybrid', squared=False, family='budget')
    c.update(kwargs)
    return c

def explore_configs():
    configs = []
    for dataset in ['xor3', 'adult', 'bank']:
        reaches = [1, 2, 4] if dataset.startswith('xor') else [2.0, 7.0, 22.0]
        for seed in range(3):
            for rho in [0.000625, 0.01, 0.16]:
                configs.append(base(dataset, seed=seed, rho=rho, calibration='global'))
                for tau in [0.005, 0.015, 0.05, 0.15, 0.5]:
                    configs.append(base(dataset, seed=seed, rho=rho, tau=tau))
                    for theta in reaches:
                        configs.append(base(dataset, seed=seed, rho=rho, tau=tau, theta=theta, policy='age', family='interaction'))
            for tau in [0.0, 0.015, 0.05, 0.15, 0.5]:
                configs.append(base(dataset, seed=seed, rho=0.0, tau=tau, family='no_noise'))
            for tau in [0.015, 0.05, 0.15, 0.5]:
                configs.append(base(dataset, seed=seed, tau=tau, calibration='matched', family='matched_noise'))
            for adj in ['U', 'BU']:
                for tau in [0.0, 0.05, 0.15, 0.5]:
                    configs.append(base(dataset, seed=seed, tau=tau, adjacency=adj, family='participation'))
            for sel in ['sample', 'rank']:
                for tau in [0.015, 0.15, 0.5]:
                    configs.append(base(dataset, seed=seed, tau=tau, selection=sel, family='selection'))
    return configs

def extended_configs():
    configs = []
    for dataset in ['adult', 'bank', 'xor3']:
        for seed in range(3):
            for rho in [0.0025, 0.04]:
                for tau in [0.0, 0.005, 0.015, 0.05, 0.15, 0.5]:
                    configs.append(base(dataset, phase='extended', seed=seed, rho=rho, tau=tau, calibration='global' if tau == 0 else 'analytic', family='budget'))
    for dataset in ['adult', 'bank', 'xor3']:
        for seed in range(3):
            for tau in [0.0, 0.015, 0.05, 0.15]:
                configs.append(base(dataset, phase='extended', seed=seed, tau=tau, nout=1000, n=0, rounds=8, family='population'))
                configs.append(base(dataset, phase='extended', seed=seed, tau=tau, nout=300, expansion=4, family='expansion'))
            for tau in [0.05, 0.15]:
                configs.append(base(dataset, phase='extended', seed=seed, tau=tau, policy='global', family='global_soft'))
            if dataset == 'xor3':
                for tau in [0.015, 0.05, 0.15]:
                    configs.append(base(dataset, phase='extended', seed=seed, tau=tau, policy='age', calibration='exact', nout=128, rounds=6, family='exact_calibration'))
                    configs.append(base(dataset, phase='extended', seed=seed, tau=tau, policy='age', calibration='analytic', nout=128, rounds=6, family='exact_calibration'))
    for seed in range(3):
        for policy in ['full', 'two']:
            for a in [2.0, 7.0, 22.0]:
                for h in [2.0, 8.0, 35.0]:
                    configs.append(base('adult', phase='extended', seed=seed, tau=0.05, theta=a, theta2=h, policy=policy, family='bottleneck'))
    return configs

def select_confirmation():
    import pandas as pd
    rows = []
    for p in list((OUT / 'explore').glob('*.json')) + list((OUT / 'extended').glob('*.json')):
        if '.failure.' in p.name:
            continue
        r = json.loads(p.read_text())
        rows.append({**r['config'], **r['validation']})
    frame = pd.DataFrame(rows)
    configs = []
    selection = []
    for dataset in ['xor3', 'adult', 'bank']:
        metric = 'hoerr_bits' if dataset.startswith('xor') else 'l1_2way'
        for rho in [0.000625, 0.0025, 0.01, 0.04, 0.16]:
            f = frame[(frame.dataset == dataset) & (frame.rho == rho) & (frame.family == 'budget') & (frame.tau > 0)]
            means = f.groupby('tau')[metric].mean()
            chosen = float(means.idxmin())
            if len(f) != 15:
                raise ValueError(f'Incomplete development grid: {dataset} {rho}: {len(f)}')
            selection.append({'dataset': dataset, 'rho': rho, 'policy': 'full', 'chosen_tau': chosen, 'selection_metric': metric, 'validation_means': {str(k): float(v) for k, v in means.items()}})
            taus = sorted(set([0.0, chosen, 0.05, 0.5]))
            for seed in range(100, 110):
                for tau in taus:
                    configs.append(base(dataset, phase='confirm', seed=seed, rho=rho, tau=tau, n=0, nout=1000, rounds=8, data_seed=(seed - 100) % 3 if dataset.startswith('xor') else 0, calibration='global' if tau == 0 else 'analytic', family='budget_confirmation'))
        for seed in range(100, 110):
            for adj in ['U', 'BU']:
                for tau in [0.0, 0.05, 0.5]:
                    configs.append(base(dataset, phase='confirm', seed=seed, tau=tau, adjacency=adj, n=0, nout=1000, rounds=8, family='participation_confirmation'))
        if dataset != 'xor3':
            for rho in [0.000625, 0.0025, 0.04, 0.16]:
                for seed in range(100, 105):
                    for tau in [0.0, 0.05]:
                        configs.append(base(dataset, phase='confirm', seed=seed, tau=tau, rho=rho, adjacency='U', n=0, nout=1000, rounds=8, family='participation_budget'))
    atomic_json(ROOT / 'confirmation_selection.json', {'selected_before_test': True, 'seeds': list(range(100, 110)), 'selection': selection, 'configs': configs})
    return configs
