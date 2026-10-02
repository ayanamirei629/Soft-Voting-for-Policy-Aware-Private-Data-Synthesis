# Frozen transition study with instrumented, otherwise unchanged draft evolution.

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
from benefit_study import certify_hard
from data import ROOT, load
from metrics import evaluate, marginal_errors
from run import base, identity, atomic_json
HOME = ROOT / 'crossover_study'
DATA = {}
TAUS = {'adult': 0.045, 'bank': 0.065, 'xor3': 0.02}
GRIDS = {'adult': [1e-05, 0.0001, 0.0003, 0.0006, 0.001, 0.0018, 0.003, 0.01, 0.1], 'bank': [1e-05, 0.0001, 0.0003, 0.0006, 0.001, 0.0018, 0.003, 0.01, 0.1], 'xor3': [0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.18, 0.3, 1.0]}

def cfg_for(dataset, seed, rho, arm, selection='hybrid', stage='transition', duplicate=1):
    return base(dataset, phase='crossover_' + stage, seed=seed, data_seed=40 + (seed - 800) % 3 if dataset == 'xor3' else 0, n=0, nout=512, rounds=8, expansion=2, policy='age', theta=1 if dataset == 'xor3' else 7, tau=0.0 if arm == 'hard' else TAUS[dataset], rho=float(rho), calibration='global' if arm == 'hard' else 'matched' if arm == 'matched' else 'analytic', selection=selection, arm=arm, duplicate=duplicate, backend='cuda')

def mse_crossover(hard, soft, rounds, q):
    bias2 = float(np.sum((soft - hard) ** 2))
    return len(hard) * rounds * (1 - q * q) / bias2 if bias2 > 0 else None

def make_manifest(stage):
    path = HOME / f'{stage}_manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    configs = []
    for dataset in TAUS:
        for seed in range(800, 812):
            if stage == 'transition':
                configs += [cfg_for(dataset, seed, rho, arm) for rho in GRIDS[dataset] for arm in ['hard', 'matched', 'soft']]
            elif stage == 'noisefree':
                configs += [cfg_for(dataset, seed, 0, arm, stage=stage) for arm in ['hard', 'soft']]
            elif stage == 'sampling':
                budgets = [0.03, 0.18, 1.0] if dataset == 'xor3' else [0.0001, 0.001, 0.01]
                configs += [cfg_for(dataset, seed, rho, arm, selection='sample', stage=stage) for rho in budgets for arm in ['hard', 'soft']]
            elif stage == 'scaling' and seed < 802:
                rho = 0.03 if dataset == 'xor3' else 0.001
                configs += [cfg_for(dataset, seed, rho / 4, arm, stage=stage, duplicate=2) for arm in ['hard', 'soft']]
    source = ['crossover_study.py', 'core.py', 'data.py', 'metrics.py', 'gpu_backend.py', 'CROSSOVER_PROTOCOL.md']
    result = dict(stage=stage, frozen_before_execution=True, configs=configs, sources={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source})
    atomic_json(path, result)
    return result

def worker(cfg):
    stage = cfg['phase'].replace('crossover_', '')
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
            key = (cfg['dataset'], cfg['data_seed'])
            if key not in DATA:
                DATA[key] = load(cfg['dataset'], 0, cfg['data_seed'])
            schema, original, validation, test = DATA[key]
            train = np.tile(original, (cfg['duplicate'], 1)) if cfg['duplicate'] > 1 else original
            diagnostics = []
            certs = []
            round_index = 0
            r, _ = core.reach(schema, cfg['policy'], cfg['theta'], cfg['theta2'])
            q = math.tanh(r / (2 * cfg['tau'])) if cfg['tau'] > 0 else 1.0

            def histogram(x, c, s, tau, squared=False):
                nonlocal round_index
                h, ent, sq = histogram_cuda(x, c, s, tau, squared)
                if tau == 0:
                    certs.append(certify_hard(c, s, cfg))
                if cfg['arm'] == 'soft' and round_index in [0, 2, 4, 7]:
                    h0, _, _ = histogram_cuda(x, c, s, 0, squared)
                    k = min(cfg['nout'], len(c))
                    hard_order = np.argsort(-h0, kind='stable')
                    soft_order = np.argsort(-h, kind='stable')
                    boundary_gap = float(h[soft_order[k - 1]] - h[soft_order[k]]) if k < len(c) else None
                    clean_overlap = len(set(hard_order[:k]) & set(soft_order[:k])) / k
                    diagnostic = dict(round=round_index, m=len(c), n=len(x), rho_hist=mse_crossover(h0, h, cfg['rounds'], q), normalized_bias2=float(np.sum(((h - h0) / len(x)) ** 2)), hard_contrast=float(np.std(h0)), soft_contrast=float(np.std(h)), contrast_retention=float(np.std(h) / np.std(h0)) if np.std(h0) > 0 else None, clean_rank_overlap=clean_overlap, soft_boundary_gap=boundary_gap)
                    if k < len(c):
                        for label, order in [('hard', hard_order), ('soft', soft_order)]:
                            selected = c[order[:k]]
                            errors = marginal_errors(test, selected, s)
                            diagnostic[label + '_clean_error'] = errors['hoerr_bits' if cfg['dataset'] == 'xor3' else 'l1_2way']
                    diagnostics.append(diagnostic)
                round_index += 1
                return (h, ent, sq)
            core.histogram = histogram
            syn, history, snapshots, extra = core.evolve(train, schema, cfg)
            snapshot_rows = []
            for name, points in snapshots.items():
                snapshot_rows.append(dict(name=name, **marginal_errors(test, points, schema), apd=core.apd(points, schema), unique_fraction=len(np.unique(points, axis=0)) / len(points)))
            result = dict(config=cfg, id=identity(cfg), test=evaluate(train, syn, test, schema), history=history, diagnostics=diagnostics, snapshot_metrics=snapshot_rows, train_n_actual=len(train), hard_certificates=certs, all_hard_rounds_certified=all(certs) if certs else None, noise_sd_ratio=1.0 if cfg['arm'] in ['hard', 'matched'] else q, private_comparison=stage != 'noisefree', wall_s=time.perf_counter() - started, **extra)
            np.savez_compressed(path.with_suffix('.npz'), synthetic=syn, **snapshots)
            atomic_json(path, result)
            return {'seconds': round(result['wall_s'], 2), 'dataset': cfg['dataset']}
    except Exception:
        error = traceback.format_exc()
        atomic_json(path.with_suffix('.failure.json'), dict(config=cfg, error=error))
        return {'error': error}
