# Frozen public-pilot and fresh-seed EDBT policy/n/dimension experiments.

from __future__ import annotations
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import core
from data import ROOT, load, schema_for
from generalization_policy import dispatch_reach, named_policies, policy_reach, hard_crossing_witness
from metrics import evaluate, marginal_errors, range_query_errors
from run import atomic_json, identity
HOME = ROOT / 'generalization_study'
VERSION = 'explicit-product-generalization-v1'
BUDGETS = [0.0001, 0.003, 0.18, 1.0]
PILOT_TAUS = [0.001, 0.003, 0.015, 0.03, 0.06, 0.12, 0.3, 0.6]
PILOT_SEEDS = list(range(1800, 1806))
CONFIRM_SEEDS = list(range(2000, 2012))
Q = 0.2
MAX_N = 100000
MAX_K = 10
INPUT_CACHE = OrderedDict()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def engine_sources():
    names = ['generalization_study.py', 'generalization_policy.py', 'core.py', 'data.py', 'gpu_backend.py', 'metrics.py', 'GENERALIZATION_PROTOCOL.md']
    return {n: digest(ROOT / n) for n in names} | {'data/adult.npz': digest(ROOT / 'data/adult.npz'), 'data/bank.npz': digest(ROOT / 'data/bank.npz')}

def signature(sources):
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]

def primary(dataset):
    return 'hoerr_bits' if dataset.startswith('xor') else 'l1_2way'

def cases():
    result = {}

    def add(dataset, n, m, policy_name, family):
        cid = f'{dataset}_n{n}_m{m}_{policy_name}'
        if cid not in result:
            schema = schema_for(dataset)
            policy = named_policies(schema)[policy_name]
            r, terms = policy_reach(schema, policy)
            result[cid] = dict(case_id=cid, dataset=dataset, n=n, nout=m, policy_name=policy_name, policy=policy, reach=r, reach_terms=terms, families=[])
        if family not in result[cid]['families']:
            result[cid]['families'].append(family)
    for d, n in [('adult', 26048), ('bank', 27126), ('xor3', 12000)]:
        for policy_name in named_policies(schema_for(d)):
            add(d, n, 512, policy_name, 'policy')
    for n in [3000, 12000, 48000, 100000]:
        for name in ['x1', 'all_columns']:
            add('xor3', n, 512, name, 'size')
    for k in [3, 6, 10]:
        for m in [512, 2048]:
            for name in ['all_inputs', 'all_columns']:
                add(f'xor{k}', 12000, m, name, 'dimension')
    return sorted(result.values(), key=lambda c: c['case_id'])

def make_config(case, phase, seed, rho, tau, calibration, sig):
    return dict(version=VERSION, engine_signature=sig, phase=phase, case_id=case['case_id'], dataset=case['dataset'], n=case['n'], nout=case['nout'], rounds=8, expansion=2, policy=case['policy'], policy_name=case['policy_name'], adjacency='B', squared=False, selection='hybrid', backend='cuda', seed=seed, data_seed=seed if case['dataset'].startswith('xor') else 0, public_data_seed=991, rho=float(rho), tau=float(tau), calibration=calibration, data_protocol='nested-ordinal-pool-v1' if case['dataset'].startswith('xor') else 'frozen-uci-split-v1')

def manifest_path(stage, sig):
    return HOME / f'{stage}_manifest_{sig}.json'

def freeze_pilot():
    HOME.mkdir(exist_ok=True)
    sources = engine_sources()
    sig = signature(sources)
    path = manifest_path('pilot', sig)
    if path.exists():
        obj = json.loads(path.read_text())
        if obj['sources'] != sources:
            raise AssertionError('Frozen source mismatch')
        return obj
    configs = []
    all_cases = cases()
    for c in all_cases:
        taus = sorted(set([0.0, *PILOT_TAUS, c['reach'] / (2 * math.atanh(Q))]))
        for rho in BUDGETS:
            for seed in PILOT_SEEDS:
                for tau in taus:
                    configs.append(make_config(c, 'pilot', seed, rho, tau, 'global' if tau == 0 else 'analytic', sig))
    import torch, scipy, pandas, matplotlib
    obj = dict(version=VERSION, engine_signature=sig, frozen_before_execution=True, sources=sources, cases=all_cases, configs=configs, budgets=BUDGETS, pilot_taus=PILOT_TAUS, q_target=Q, pilot_seeds=PILOT_SEEDS, confirmation_seeds=CONFIRM_SEEDS, environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, pandas=pandas.__version__, matplotlib=matplotlib.__version__, device=torch.cuda.get_device_name(0)))
    atomic_json(path, obj)
    atomic_json(HOME / 'active_manifest.json', dict(engine_signature=sig, pilot_manifest=path.name))
    print(json.dumps(dict(frozen='pilot', signature=sig, cases=len(all_cases), configurations=len(configs))), flush=True)
    return obj

def result_path(cfg):
    return HOME / 'results' / cfg['engine_signature'] / cfg['phase'] / (identity(cfg) + '.json')

def compose_xor(raw, k):
    raw = raw[:, :k]
    label = np.bitwise_xor.reduce((raw >= 4).astype(np.int8), axis=1)
    return np.column_stack([raw / 7.0, label])

def inputs(cfg):
    key = (cfg['dataset'], cfg['n'], cfg['phase'] == 'pilot', cfg['data_seed'])
    if key in INPUT_CACHE:
        INPUT_CACHE.move_to_end(key)
        return INPUT_CACHE[key]
    d = cfg['dataset']
    schema = schema_for(d)
    if d.startswith('xor'):
        k = int(d[3:])
        if cfg['phase'] == 'pilot':
            raw = np.random.default_rng(20260907 + 991).integers(0, 8, size=(3000, MAX_K))
            ref = np.random.default_rng(40260907 + 991).integers(0, 8, size=(10000, MAX_K))
            train = compose_xor(raw, k)
        else:
            raw = np.random.default_rng(20260907 + cfg['data_seed']).integers(0, 8, size=(MAX_N, MAX_K))
            ref = np.random.default_rng(40260907 + cfg['data_seed']).integers(0, 8, size=(10000, MAX_K))
            train = compose_xor(raw[:cfg['n']], k)
        reference = compose_xor(ref, k)
    else:
        schema, train, val, test = load(d, 0)
        if cfg['phase'] == 'pilot':
            perm = np.random.default_rng(20260908).permutation(len(val))
            train, reference = (val[perm[:3000]], val[perm[3000:]])
        else:
            reference = test
    answer = (schema, train, reference)
    INPUT_CACHE[key] = answer
    if len(INPUT_CACHE) > 6:
        INPUT_CACHE.popitem(last=False)
    return answer

def oracle_xor_error(syn):
    k = syn.shape[1] - 1
    bits = np.column_stack([(syn[:, :-1] >= 0.5).astype(int), syn[:, -1].astype(int)])
    weights = 2 ** np.arange(k, -1, -1)
    counts = np.bincount(bits @ weights, minlength=2 ** (k + 1)) / len(syn)
    patterns = np.arange(2 ** k)
    parity = np.zeros(len(patterns), dtype=int)
    for j in range(k):
        parity ^= patterns >> j & 1
    truth = np.zeros_like(counts)
    truth[patterns * 2 + parity] = 1 / 2 ** k
    return float(np.abs(counts - truth).sum())

def attribute_metrics(target, syn, schema, policy):
    a, cards = schema.discrete(target)
    b, _ = schema.discrete(syn)
    protected = {schema.names.index(s['attribute']) for s in policy['attributes']}
    result, touched, untouched = ({}, [], [])
    for order in [1, 2]:
        for cols in combinations(range(schema.p), order):
            shape = [cards[j] for j in cols]
            ai = np.ravel_multi_index(a[:, cols].T, shape)
            bi = np.ravel_multi_index(b[:, cols].T, shape)
            length = math.prod(shape)
            pa = np.bincount(ai, minlength=length) / len(a)
            pb = np.bincount(bi, minlength=length) / len(b)
            value = float(np.abs(pa - pb).sum())
            result[f'l1_{order}way__' + '__'.join((schema.names[j] for j in cols))] = value
            if order == 2:
                (touched if any((j in protected for j in cols)) else untouched).append(value)
    result['l1_2way_protected_involving'] = float(np.mean(touched)) if touched else None
    result['l1_2way_unprotected_only'] = float(np.mean(untouched)) if untouched else None
    return result

def compatible(cfg, path):
    if not path.exists():
        return False
    r = json.loads(path.read_text())
    expected_n = 3000 if cfg['phase'] == 'pilot' else cfg['n']
    if r['config'] != cfg or r['train_n_actual'] != expected_n:
        raise AssertionError(f'Incompatible cache {path}')
    if len(r['history']) != 8 or not math.isfinite(r['metrics'][primary(cfg['dataset'])]):
        raise AssertionError(f'Incomplete cache {path}')
    if not path.with_suffix('.npz').exists():
        raise AssertionError(f'Missing saved population {path}')
    return True

def worker(cfg):
    path = result_path(cfg)
    if compatible(cfg, path):
        return dict(id=identity(cfg), cached=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    original_histogram, original_reach = (core.histogram, core.reach)
    try:
        import torch
        from gpu_backend import histogram_cuda
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            schema, train, reference = inputs(cfg)
            actual = dict(cfg)
            if cfg['phase'] == 'pilot':
                actual['rho'] = cfg['rho'] * (cfg['n'] / len(train)) ** 2
            witnesses = []

            def checked_histogram(x, c, s, tau, squared=False):
                if tau == 0:
                    witnesses.append(hard_crossing_witness(c, s, cfg['policy']))
                return histogram_cuda(x, c, s, tau, squared)
            core.histogram, core.reach = (checked_histogram, dispatch_reach)
            syn, history, snapshots, extra = core.evolve(train, schema, actual)
            if cfg['phase'] == 'pilot':
                metrics = marginal_errors(reference, syn, schema)
                metrics.update(range_query_errors(reference, syn, schema))
            else:
                metrics = evaluate(train, syn, reference, schema)
                metrics.update(attribute_metrics(reference, syn, schema, cfg['policy']))
            if cfg['dataset'].startswith('xor'):
                metrics['hoerr_bits_oracle'] = oracle_xor_error(syn)
            reach, terms = policy_reach(schema, cfg['policy'])
            q = math.tanh(reach / (2 * cfg['tau'])) if cfg['tau'] else 1.0
            actual_q = history[0]['sensitivity'] / math.sqrt(2)
            expected_q = q if cfg['calibration'] == 'analytic' else 1.0
            if not math.isclose(actual_q, expected_q, abs_tol=1e-12):
                raise AssertionError('Used noise calibration differs from declared policy')
            if actual['rho'] > 0:
                spent = sum((h['sensitivity'] ** 2 / (2 * h['sigma'] ** 2) for h in history))
                if not math.isclose(spent, actual['rho'], rel_tol=1e-10):
                    raise AssertionError('Privacy schedule mismatch')
            np.savez_compressed(path.with_suffix('.npz'), synthetic=syn, **snapshots)
            record = dict(id=identity(cfg), version=VERSION, config=cfg, train_n_actual=len(train), target_n=cfg['n'], reference_n=len(reference), actual_rho=actual['rho'], metrics=metrics, analytic_q=q, actual_noise_ratio=actual_q, history=history, normalized_sigma=history[0]['sigma'] / len(train), hard_crossing_certificates=witnesses, all_hard_rounds_certified=all((w is not None for w in witnesses)) if witnesses else None, input_sha256=hashlib.sha256(train.tobytes()).hexdigest(), reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(), wall_s=time.perf_counter() - started, **extra)
            atomic_json(path, record)
            return dict(id=record['id'], dataset=cfg['dataset'], phase=cfg['phase'], seconds=round(record['wall_s'], 3))
    except Exception:
        error = traceback.format_exc()
        atomic_json(path.with_suffix('.failure.json'), dict(config=cfg, error=error))
        return dict(id=identity(cfg), error=error)
    finally:
        core.histogram, core.reach = (original_histogram, original_reach)

def execute(configs, workers):
    pending = [c for c in configs if not compatible(c, result_path(c))]
    total = len(configs)
    completed = total - len(pending)
    started = time.perf_counter()
    failures = []
    print(json.dumps(dict(stage=configs[0]['phase'], total=total, cached=completed, pending=len(pending), workers=workers)), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(worker, c): c for c in pending}
        for f in as_completed(jobs):
            result = f.result()
            completed += 1
            if 'error' in result:
                failures.append(result)
            if completed % 48 == 0 or completed == total or 'error' in result:
                progress = dict(stage=configs[0]['phase'], completed=completed, total=total, failures=len(failures), elapsed_s=round(time.perf_counter() - started, 1), last=result)
                atomic_json(HOME / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} failed configurations, saved without exclusion')

def freeze_confirmation(pilot):
    sig = pilot['engine_signature']
    path = manifest_path('confirm', sig)
    if path.exists():
        return json.loads(path.read_text())
    grouped = {}
    pilot_hashes = {}
    for cfg in pilot['configs']:
        p = result_path(cfg)
        if not compatible(cfg, p):
            raise RuntimeError('Public-pilot results must be complete before selection')
        r = json.loads(p.read_text())
        grouped.setdefault((cfg['case_id'], cfg['rho'], cfg['tau']), []).append(r['metrics'][primary(cfg['dataset'])])
        pilot_hashes[r['id']] = digest(p)
    selected = {}
    for c in pilot['cases']:
        for rho in BUDGETS:
            candidates = [(float(np.mean(v)), tau) for (cid, rr, tau), v in grouped.items() if cid == c['case_id'] and rr == rho]
            if not all((len(grouped[c['case_id'], rho, tau]) == 6 for _, tau in candidates)):
                raise AssertionError('Wrong public replicate count')
            candidates.sort(key=lambda x: (x[0], x[1] > 0, x[1]))
            best_soft = min(((v, t) for v, t in candidates if t > 0))
            selected[c['case_id'], rho] = dict(tau=candidates[0][1], public_mean=candidates[0][0], best_soft_tau=best_soft[1], best_soft_public_mean=best_soft[0])
    configs, cells = ({}, [])

    def add(c, seed, rho, tau, calibration, setting, display_rho=None):
        cfg = make_config(c, 'confirm', seed, rho, tau, calibration, sig)
        rid = identity(cfg)
        configs[rid] = cfg
        cells.append(dict(case_id=c['case_id'], seed=seed, rho=rho if display_rho is None else display_rho, actual_target_rho=rho, setting=setting, tau=tau, calibration=calibration, result_id=rid))
    for c in pilot['cases']:
        for rho in BUDGETS:
            settings = {'fixed_tau_0p03': 0.03, 'matched_q_0p2': c['reach'] / (2 * math.atanh(Q)), 'public_selected': selected[c['case_id'], rho]['tau']}
            if 'size' in c['families']:
                settings |= {'fixed_tau_0p015': 0.015, 'fixed_tau_0p06': 0.06}
            for seed in CONFIRM_SEEDS:
                add(c, seed, rho, 0.0, 'global', 'hard')
                for name, tau in settings.items():
                    add(c, seed, rho, tau, 'analytic' if tau else 'global', name)
                    if tau > 0:
                        add(c, seed, rho, tau, 'matched', 'matched_noise__' + name)
                if 'size' in c['families']:
                    scaled_rho = rho * (12000 / c['n']) ** 2
                    add(c, seed, scaled_rho, 0.0, 'global', 'normalized_noise__hard', rho)
                    baseline_case = f"xor3_n12000_m512_{c['policy_name']}"
                    norm_settings = {**settings, 'public_selected': selected[baseline_case, rho]['tau']}
                    for name, tau in norm_settings.items():
                        setting = 'normalized_noise__' + name
                        add(c, seed, scaled_rho, tau, 'analytic' if tau else 'global', setting, rho)
                        if tau > 0:
                            add(c, seed, scaled_rho, tau, 'matched', 'matched_noise__' + setting, rho)
    obj = dict(version=VERSION, engine_signature=sig, frozen_before_target_execution=True, pilot_manifest=manifest_path('pilot', sig).name, pilot_result_sha256=pilot_hashes, sources=pilot['sources'], cases=pilot['cases'], budgets=BUDGETS, seeds=CONFIRM_SEEDS, selection=[dict(case_id=cid, rho=rho, **s) for (cid, rho), s in selected.items()], selection_rule='Lowest public mean including Hard; ties Hard then lower tau', configs=sorted(configs.values(), key=lambda c: (c['case_id'], c['seed'], c['rho'], c['tau'], c['calibration'])), cells=cells)
    atomic_json(path, obj)
    atomic_json(HOME / 'active_manifest.json', dict(engine_signature=sig, pilot_manifest=manifest_path('pilot', sig).name, confirm_manifest=path.name))
    print(json.dumps(dict(frozen='confirm', configurations=len(configs), plotted_cells=len(cells))), flush=True)
    return obj

def audit(manifest):
    counts = dict(total=len(manifest['configs']), completed=0, missing=0, certified_rounds=0, hard_rounds=0, q_checks=0, uncertainty_unit='12 paired seeds for confirmation')
    errors = []
    for c in manifest['configs']:
        p = result_path(c)
        if not compatible(c, p):
            counts['missing'] += 1
            continue
        r = json.loads(p.read_text())
        counts['completed'] += 1
        counts['q_checks'] += int(c['calibration'] == 'analytic')
        for w in r['hard_crossing_certificates']:
            counts['hard_rounds'] += 1
            counts['certified_rounds'] += int(w is not None)
        if not math.isclose(r['reach'], policy_reach(schema_for(c['dataset']), c['policy'])[0], abs_tol=1e-12):
            errors.append(r['id'])
    counts['errors'] = errors
    counts['pass'] = counts['missing'] == 0 and (not errors)
    atomic_json(HOME / f"audit_{manifest['configs'][0]['phase']}_{manifest['engine_signature']}.json", counts)
    print(json.dumps(counts), flush=True)
    return counts
