# Broad, matched search for simultaneous noise and fidelity benefits.

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
from scipy.stats import t, ttest_1samp
from threadpoolctl import threadpool_limits
import core
from data import ROOT, load
from metrics import evaluate
from run import base, identity, atomic_json
HOME = ROOT / 'benefit_study'
RHOS = [1e-07, 1e-06, 1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
TAUS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.045, 0.065, 0.09, 0.13, 0.2, 0.3]
DATA = {}

def config(dataset, policy, rho, tau, phase='screen', seed=600, calibration=None):
    return base(dataset, phase='benefit_' + phase, seed=seed, n=0, nout=512, rounds=8, policy=policy, theta=1 if dataset == 'xor3' else 7.0, rho=float(rho), tau=float(tau), data_seed=(30 + (seed - 700) % 3 if phase == 'confirm' else 20) if dataset == 'xor3' else 0, calibration=calibration or ('global' if tau == 0 else 'analytic'), family='simultaneous_benefit', backend='cuda')

def certify_hard(candidates, schema, cfg):
    if cfg['policy'] == 'full':
        from hard_certificate import certificate
        return certificate(candidates, schema)
    if cfg['theta'] < schema.numeric[0][2]:
        return False
    seen = {}
    for point in candidates:
        key = tuple(point[1:])
        if key in seen and seen[key] != point[0]:
            return True
        seen[key] = point[0]
    points = candidates[:64].copy()
    ends = np.repeat(points, 2, axis=0)
    ends[::2, 0] = 0
    ends[1::2, 0] = 1
    distances = schema.distance(ends, candidates)
    winners = distances.argmin(axis=1)
    top = np.partition(distances, 1, axis=1)[:, :2]
    unique = top[:, 1] - top[:, 0] > 1e-12
    return bool(np.any((winners[::2] != winners[1::2]) & unique[::2] & unique[1::2]))

def worker(cfg):
    phase = cfg['phase'].replace('benefit_', '')
    folder = HOME / 'results' / phase
    folder.mkdir(parents=True, exist_ok=True)
    rid = identity(cfg)
    path = folder / f'{rid}.json'
    if path.exists():
        return {'id': rid, 'cached': True}
    try:
        import torch
        from gpu_backend import histogram_cuda
        torch.set_num_threads(1)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            key = (cfg['dataset'], cfg['n'], cfg['data_seed'])
            if key not in DATA:
                DATA[key] = load(*key)
            schema, train, validation, testing = DATA[key]
            certs = []

            def histogram(x, candidates, schema, tau, squared=False):
                if tau == 0:
                    certs.append(certify_hard(candidates, schema, cfg))
                return histogram_cuda(x, candidates, schema, tau, squared)
            core.histogram = histogram
            syn, history, snapshots, extra = core.evolve(train, schema, cfg)
            metrics = evaluate(train, syn, validation, schema)
            test = evaluate(train, syn, testing, schema) if phase == 'confirm' else None
            ratio = history[0]['sensitivity'] / math.sqrt(2)
            result = dict(id=rid, config=cfg, validation=metrics, test=test, history=history, noise_sd_ratio=ratio, noise_variance_ratio=ratio ** 2, all_hard_rounds_certified=all(certs) if certs else None, hard_certificates=certs, train_n_actual=len(train), **extra, wall_s=time.perf_counter() - started)
            np.savez_compressed(folder / f'{rid}.npz', synthetic=syn, **snapshots)
            atomic_json(path, result)
            return dict(id=rid, seconds=round(result['wall_s'], 2), dataset=cfg['dataset'], policy=cfg['policy'], rho=cfg['rho'], tau=cfg['tau'], noise_ratio=round(ratio, 3), primary_error=metrics['hoerr_bits' if cfg['dataset'] == 'xor3' else 'l1_2way'])
    except Exception:
        error = traceback.format_exc()
        atomic_json(folder / f'{rid}.failure.json', dict(config=cfg, error=error))
        return dict(id=rid, error=error)

def collect():
    rows = []
    for path in (HOME / 'results').glob('*/*.json'):
        if '.failure.' in path.name:
            continue
        row = json.loads(path.read_text())
        for split in ['validation', 'test']:
            if row.get(split):
                cfg = row['config']
                metric = 'hoerr_bits' if cfg['dataset'] == 'xor3' else 'l1_2way'
                auc = 'auc_tree' if cfg['dataset'] == 'xor3' else 'auc_lr'
                rows.append(dict(**cfg, id=row['id'], split=split, **row[split], primary_error=row[split][metric], primary_auc=row[split][auc], noise_sd_ratio=row['noise_sd_ratio'], all_hard_rounds_certified=row['all_hard_rounds_certified'], wall_s=row['wall_s']))
    return pd.DataFrame(rows)

def comparison(frame):
    f = frame[(frame.split == 'validation') & frame.phase.isin(['benefit_screen', 'benefit_refine'])]
    keys = ['dataset', 'policy', 'rho', 'seed']
    hard = f[f.tau == 0][keys + ['primary_error', 'primary_auc']]
    soft = f[(f.tau > 0) & (f.calibration == 'analytic')]
    c = soft.merge(hard, on=keys, suffixes=('', '_hard'), validate='many_to_one')
    c['gain'] = 1 - c.primary_error / c.primary_error_hard
    return c.groupby(['dataset', 'policy', 'rho', 'tau'], as_index=False).agg(gain=('gain', 'mean'), error=('primary_error', 'mean'), hard_error=('primary_error_hard', 'mean'), auc=('primary_auc', 'mean'), hard_auc=('primary_auc_hard', 'mean'), noise=('noise_sd_ratio', 'mean'), seeds=('seed', 'nunique'))

def choose_refine():
    summary = comparison(collect())
    cases = []
    configs = []
    for (dataset, policy), group in summary.groupby(['dataset', 'policy']):
        eligible = group[group.noise <= 0.8]
        winners = eligible.sort_values('gain', ascending=False).drop_duplicates('rho').head(3)
        for _, row in winners.iterrows():
            ix = TAUS.index(row.tau)
            taus = sorted(set([0.0] + TAUS[max(1, ix - 1):min(len(TAUS), ix + 2)]))
            cases.append(dict(dataset=dataset, policy=policy, rho=row.rho, screening_tau=row.tau, screening_gain=row.gain, refinement_taus=taus))
            for seed in [601, 602]:
                configs += [config(dataset, policy, row.rho, tau, 'refine', seed) for tau in taus]
    return (configs, cases)

def choose_confirmation():
    summary = comparison(collect())
    cases = []
    configs = []
    for (dataset, policy), group in summary.groupby(['dataset', 'policy']):
        qualified = group[(group.seeds >= 3) & (group.noise <= 0.8) & (group.gain >= 0.1)]
        if qualified.empty:
            continue
        winner = qualified.sort_values('gain', ascending=False).iloc[0]
        index = RHOS.index(winner.rho)
        budgets = RHOS[max(0, index - 1):min(len(RHOS), index + 2)]
        ix = TAUS.index(winner.tau)
        neighbors = [value for value in TAUS[max(1, ix - 1):min(len(TAUS), ix + 2)] if value != winner.tau]
        cases.append(dict(dataset=dataset, policy=policy, rho=winner.rho, tau=winner.tau, development_gain=winner.gain, noise_ratio=winner.noise, budgets=budgets, neighboring_taus=neighbors))
        for seed in range(700, 710):
            for rho in budgets:
                configs += [config(dataset, policy, rho, tau, 'confirm', seed) for tau in [0.0, winner.tau]]
            configs.append(config(dataset, policy, winner.rho, winner.tau, 'confirm', seed, 'matched'))
            configs += [config(dataset, policy, winner.rho, tau, 'confirm', seed) for tau in neighbors]
    return (configs, cases)

def run_stage(stage, workers, limit=0):
    HOME.mkdir(exist_ok=True)
    manifest = HOME / f'{stage}_manifest.json'
    if manifest.exists():
        content = json.loads(manifest.read_text())
        configs = content['configs']
    else:
        if stage == 'screen':
            configs = [config(dataset, policy, rho, tau) for dataset in ['adult', 'bank', 'xor3'] for policy in ['full', 'age'] for rho in RHOS for tau in TAUS]
            cases = []
        elif stage == 'refine':
            configs, cases = choose_refine()
        elif stage == 'confirm':
            configs, cases = choose_confirmation()
        else:
            configs = [config(d, p, 0.001, tau, 'smoke', 599) for d in ['adult', 'bank', 'xor3'] for p in ['full', 'age'] for tau in [0.0, 0.03]]
            cases = []
        configs = list({identity(c): c for c in configs}.values())
        source = ['benefit_study.py', 'core.py', 'gpu_backend.py', 'data.py', 'metrics.py']
        atomic_json(manifest, dict(stage=stage, frozen_before_execution=True, cases=cases, configs=configs, source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source}))
    if limit:
        configs = configs[:limit]
    pending = [c for c in configs if not (HOME / 'results' / stage / f'{identity(c)}.json').exists()]
    print(json.dumps(dict(stage=stage, total=len(configs), pending=len(pending), workers=workers)), flush=True)
    start = time.perf_counter()
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(worker, cfg) for cfg in pending]
        for i, future in enumerate(as_completed(tasks), 1):
            result = future.result()
            failures += int('error' in result)
            if i % 10 == 0 or 'error' in result or i == len(tasks):
                print(json.dumps(dict(stage=stage, done=i, pending_total=len(pending), elapsed_s=round(time.perf_counter() - start), failures=failures, last=result)), flush=True)
    if failures:
        raise RuntimeError(f'{failures} failed configurations')
    frame = collect()
    frame.to_csv(HOME / 'metrics.csv', index=False)
    if stage != 'smoke':
        summary = comparison(frame)
        summary.to_csv(HOME / 'development_comparisons.csv', index=False)
        print(summary[summary.noise <= 0.8].sort_values('gain', ascending=False).head(18).to_string(index=False), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke', 'screen', 'refine', 'confirm'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    run_stage(args.stage, args.workers, args.limit)
