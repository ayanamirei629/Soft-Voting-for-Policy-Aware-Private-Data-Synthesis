# Prospective forecast evaluation and finite-grid interaction maps.

import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import json
import math
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from interaction_transfer import HOME, TAUS, THETAS, BUDGETS, ISO, collect, manifest
from crossover_study import GRIDS
from analyze_crossover import descending_roots
from run import atomic_json, identity
FIG = HOME / 'figures'
KEYS = ['dataset', 'rho', 'tau', 'theta']
METHODS = ['Scaled pilot', 'Unscaled pilot', 'Initial marginal L1', 'Noise-only', 'Initial SNR', 'Always hard', 'Always soft']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

def adjust(df):
    for _, ids in df.groupby('family').groups.items():
        order = df.loc[ids].sort_values('p_raw').index
        df.loc[order, 'p_holm'] = np.maximum.accumulate([min(1, df.loc[idx, 'p_raw'] * (len(order) - i)) for i, idx in enumerate(order)])
        df.loc[order, 'family_size'] = len(order)
    return df

def paired(v):
    v = np.asarray(v, float)
    margin = t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / math.sqrt(len(v))
    return dict(mean=float(v.mean()), ci_low=float(v.mean() - margin), ci_high=float(v.mean() + margin), n=len(v), p_raw=float(ttest_1samp(v, 0).pvalue) if np.std(v) > 0 else float(v.mean() == 0))

def data():
    f = collect('target').rename(columns={'target_rho': 'target_budget'})
    assert len(f) == len(manifest('target')['configs'])
    hard = f[f.tau == 0][['dataset', 'rho', 'seed', 'error']].rename(columns={'error': 'hard_error'})
    soft = f[f.tau > 0].merge(hard, on=['dataset', 'rho', 'seed'], validate='many_to_one')
    soft['gain'] = soft.hard_error - soft.error
    soft['relative_gain'] = soft.gain / soft.hard_error
    soft['is_map'] = [r.rho in BUDGETS[r.dataset] and r.tau in TAUS[r.dataset] and (r.theta in THETAS[r.dataset]) for _, r in soft.iterrows()]
    forecast = pd.DataFrame(json.loads((HOME / 'forecasts.json').read_text())['rows'])
    rows = []
    for key, g in soft.groupby(KEYS):
        assert len(g) == 6
        rows.append(dict(zip(KEYS, key), **paired(g.gain), hard_error=g.hard_error.mean(), soft_error=g.error.mean(), relative_gain=1 - g.error.mean() / g.hard_error.mean(), is_map=bool(g.is_map.iloc[0]), q=g.q.iloc[0], family='map_fidelity' if g.is_map.iloc[0] else 'extra_curve_fidelity'))
    cells = adjust(pd.DataFrame(rows)).merge(forecast, on=KEYS, validate='one_to_one')
    cells.to_csv(HOME / 'cells.csv', index=False)
    soft.to_csv(HOME / 'target_pairs.csv', index=False)
    return (f, soft, cells)

def choice(df, method):
    if method == 'Scaled pilot':
        return df.pilot_scaled_gain.to_numpy() > 0
    if method == 'Unscaled pilot':
        return df.pilot_unscaled_gain.to_numpy() > 0
    if method == 'Initial marginal L1':
        return df.initial_gain.to_numpy() > 0
    if method == 'Noise-only':
        return df.initial_q.to_numpy() <= 0.5
    if method == 'Initial SNR':
        return df.initial_snr_ratio.to_numpy() > 1
    return np.full(len(df), method == 'Always soft')

def performance(cells, pairs):
    rows = []
    rng = np.random.default_rng(908031)
    for dataset, g in cells[cells.is_map].groupby('dataset'):
        g = g.sort_values(KEYS).reset_index(drop=True)
        pair = pairs[(pairs.dataset == dataset) & pairs.is_map]
        p = pair.pivot(index='seed', columns=['rho', 'tau', 'theta'], values='gain')
        p = p.reindex(columns=pd.MultiIndex.from_frame(g[['rho', 'tau', 'theta']]))
        boot = p.to_numpy()[rng.integers(0, 6, size=(2000, 6))].mean(1) > 0
        truth = g['mean'].to_numpy() > 0
        supported = g.p_holm.to_numpy() < 0.05
        for method in METHODS:
            pick = choice(g, method)
            correct = pick == truth
            balanced = np.mean([correct[truth == v].mean() for v in [False, True] if np.any(truth == v)])
            selected = np.where(pick, g.soft_error, g.hard_error)
            accuracy_draws = (boot == pick).mean(1)
            forecast_column = {'Scaled pilot': 'pilot_scaled_gain', 'Unscaled pilot': 'pilot_unscaled_gain', 'Initial marginal L1': 'initial_gain'}.get(method)
            rows.append(dict(dataset=dataset, method=method, cells=len(g), soft_win_fraction=truth.mean(), choose_soft_fraction=pick.mean(), accuracy=correct.mean(), balanced_accuracy=balanced, accuracy_ci_low=np.quantile(accuracy_draws, 0.025), accuracy_ci_high=np.quantile(accuracy_draws, 0.975), supported_cells=int(supported.sum()), supported_accuracy=correct[supported].mean() if supported.any() else None, gain_mae=float(np.abs(g[forecast_column] - g['mean']).mean()) if forecast_column else None, regret=float(np.mean(selected - np.minimum(g.hard_error, g.soft_error))), mean_error=float(selected.mean()), hard_mean_error=float(g.hard_error.mean())))
    result = pd.DataFrame(rows)
    result.to_csv(HOME / 'predictor_scores.csv', index=False)
    summary = []
    for method, g in result.groupby('method', sort=False):
        summary.append(dict(method=method, cell_weighted_accuracy=np.average(g.accuracy, weights=g.cells), dataset_mean_accuracy=g.accuracy.mean(), dataset_mean_balanced_accuracy=g.balanced_accuracy.mean()))
    pd.DataFrame(summary).to_csv(HOME / 'predictor_summary.csv', index=False)
    return result

def selectors(f, pairs, cells):
    rows = []
    contrasts = []
    choices = []
    for dataset in TAUS:
        selected = {method: [] for method in ['Scaled pilot', 'Unscaled pilot', 'Initial marginal L1', 'Always hard', 'Measured-grid oracle']}
        for (rho, theta), g in cells[(cells.dataset == dataset) & cells.is_map].groupby(['rho', 'theta']):
            actual = f[(f.dataset == dataset) & (f.rho == rho)]
            hard = actual[actual.tau == 0].set_index('seed').error.sort_index()
            for method in selected:
                prefix = {'Scaled pilot': 'pilot_scaled', 'Unscaled pilot': 'pilot_unscaled', 'Initial marginal L1': 'initial'}.get(method)
                if prefix:
                    best = g.sort_values(prefix + '_soft').iloc[0]
                    tau = float(best.tau) if best[prefix + '_soft'] < best[prefix + '_hard'] else 0.0
                elif method == 'Measured-grid oracle':
                    best = g.sort_values('soft_error').iloc[0]
                    tau = float(best.tau) if best.soft_error < best.hard_error else 0.0
                else:
                    tau = 0.0
                values = hard if tau == 0 else actual[(actual.tau == tau) & (actual.theta == theta)].set_index('seed').error.sort_index()
                assert len(values) == 6
                selected[method].append(values.to_numpy())
                choices.append(dict(dataset=dataset, rho=rho, theta=theta, method=method, tau=tau, error=values.mean()))
        mean_by_seed = {m: np.mean(v, axis=0) for m, v in selected.items()}
        oracle = mean_by_seed['Measured-grid oracle']
        for method, v in mean_by_seed.items():
            rows.append(dict(dataset=dataset, method=method, error=v.mean(), regret=v.mean() - oracle.mean(), gain_over_hard=(mean_by_seed['Always hard'] - v).mean()))
            if method not in ['Scaled pilot', 'Measured-grid oracle']:
                contrasts.append(dict(dataset=dataset, reference=method, family='selector_comparisons', **paired(v - mean_by_seed['Scaled pilot'])))
    pd.DataFrame(rows).to_csv(HOME / 'temperature_selector_scores.csv', index=False)
    pd.DataFrame(choices).to_csv(HOME / 'temperature_choices.csv', index=False)
    adjust(pd.DataFrame(contrasts)).to_csv(HOME / 'selector_tests.csv', index=False)
    return pd.DataFrame(rows)

def iso_analysis(pairs, cells):
    tests = []
    root_rows = []
    for dataset in TAUS:
        for rho in GRIDS[dataset]:
            for (ta, wa), (tb, wb) in combinations(ISO[dataset], 2):
                a = pairs[(pairs.dataset == dataset) & (pairs.rho == rho) & (pairs.tau == ta) & (pairs.theta == wa)].set_index('seed').error.sort_index()
                b = pairs[(pairs.dataset == dataset) & (pairs.rho == rho) & (pairs.tau == tb) & (pairs.theta == wb)].set_index('seed').error.sort_index()
                tests.append(dict(dataset=dataset, rho=rho, tau_a=ta, theta_a=wa, tau_b=tb, theta_b=wb, family='equal_ratio_fidelity', **paired(a - b)))
        for tau, theta in ISO[dataset]:
            g = cells[(cells.dataset == dataset) & (cells.tau == tau) & (cells.theta == theta)].sort_values('rho')
            observed = descending_roots(g.rho.to_numpy(), g['mean'].to_numpy())
            for method, col in [('Scaled pilot', 'pilot_scaled_gain'), ('Unscaled pilot', 'pilot_unscaled_gain'), ('Initial marginal L1', 'initial_gain')]:
                predicted = descending_roots(g.rho.to_numpy(), g[col].to_numpy())
                error = abs(math.log10(predicted[0]['root'] / observed[0]['root'])) if observed and predicted else None
                root_rows.append(dict(dataset=dataset, tau=tau, theta=theta, method=method, observed_roots=observed, predicted_roots=predicted, both_cross=bool(observed and predicted), cross_presence_agrees=bool(observed) == bool(predicted), observed_status=root_status(g.rho.to_numpy(), g['mean'].to_numpy()), predicted_status=root_status(g.rho.to_numpy(), g[col].to_numpy()), log10_error=error))
    tests = adjust(pd.DataFrame(tests))
    tests.to_csv(HOME / 'equal_ratio_tests.csv', index=False)
    atomic_json(HOME / 'root_forecasts.json', root_rows)
    return (tests, root_rows)

def root_status(rho, gain):
    roots = descending_roots(rho, gain)
    if len(roots) > 1:
        return 'multiple_upper_crossings'
    if roots:
        return 'upper_crossing'
    if np.all(np.asarray(gain) > 0):
        return 'soft_at_all_tested_budgets'
    if np.all(np.asarray(gain) < 0):
        return 'hard_at_all_tested_budgets'
    return 'other_or_tied'

def grid(g, dataset, column, rho):
    return g[g.rho == rho].pivot(index='theta', columns='tau', values=column).reindex(index=THETAS[dataset], columns=TAUS[dataset]).to_numpy()

def axes_labels(ax, dataset):
    ax.set_xticks(np.log10(TAUS[dataset]), [f'{x:g}' for x in TAUS[dataset]])
    ax.set_yticks(np.log10(THETAS[dataset]), [f'{x:g}' for x in THETAS[dataset]])
    ax.set_xlabel('Temperature tau (log spacing)')
    ax.set_ylabel('Reach theta (ordinal steps)' if dataset == 'xor3' else 'Age-change width theta (years)')

def zero(ax, x, y, z, **kw):
    if np.nanmin(z) < 0 < np.nanmax(z):
        ax.contour(x, y, z, levels=[0], **kw)

def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=210, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

def plot_maps(cells):
    for dataset in TAUS:
        g = cells[(cells.dataset == dataset) & cells.is_map]
        x, y = np.meshgrid(np.log10(TAUS[dataset]), np.log10(THETAS[dataset]))
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), layout='constrained')
        for ax, rho in zip(axes, BUDGETS[dataset]):
            z = grid(g, dataset, 'relative_gain', rho)
            im = ax.contourf(x, y, z, levels=np.linspace(-0.6, 0.6, 25), cmap='RdBu', extend='both')
            zero(ax, x, y, z, colors='black', linewidths=1.6)
            q = grid(g, dataset, 'q', rho)
            cs = ax.contour(x, y, q, levels=[0.2, 0.5, 0.8], colors='#666666', linestyles='dashed', linewidths=0.8)
            ax.clabel(cs, fmt={0.2: '80% SD', 0.5: '50% SD', 0.8: '20% SD'}, fontsize=7)
            ax.scatter(x, y, s=9, c='#333333', alpha=0.45)
            uncertain = (grid(g, dataset, 'ci_low', rho) <= 0) & (grid(g, dataset, 'ci_high', rho) >= 0)
            ax.scatter(x[uncertain], y[uncertain], marker='x', s=32, c='#CA7834', linewidths=1.1)
            axes_labels(ax, dataset)
            ax.set_title(f'rho = {rho:g}')
        fig.colorbar(im, ax=axes, shrink=0.85, label='Relative fidelity gain: (hard - soft) / hard')
        fig.suptitle(f'{dataset.title()}: measured temperature-reach interaction | six paired seeds\nBlack: zero gain; dashed: noise SD saving; x: unadjusted paired interval includes zero')
        save(fig, '01_maps_' + dataset)

def plot_forecasts(cells):
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 10.5), layout='constrained')
    for row, dataset in enumerate(TAUS):
        g = cells[(cells.dataset == dataset) & cells.is_map]
        x, y = np.meshgrid(np.log10(TAUS[dataset]), np.log10(THETAS[dataset]))
        for ax, rho in zip(axes[row], BUDGETS[dataset]):
            z = grid(g, dataset, 'relative_gain', rho)
            pred = grid(g, dataset, 'pilot_scaled_gain', rho)
            im = ax.contourf(x, y, z, levels=np.linspace(-0.6, 0.6, 25), cmap='RdBu', extend='both')
            zero(ax, x, y, z, colors='black', linewidths=1.4)
            zero(ax, x, y, pred, colors='#D58E00', linewidths=1.8, linestyles='dashed')
            ax.scatter(x[(z > 0) != (pred > 0)], y[(z > 0) != (pred > 0)], marker='x', c='#B61D44', s=30)
            axes_labels(ax, dataset)
            ax.set_title(f'{dataset.title()} | rho={rho:g}')
    fig.colorbar(im, ax=axes, shrink=0.55, label='Observed relative fidelity gain')
    fig.suptitle('Prospective prediction: black measured boundary vs gold public-pilot forecast\nForecasts frozen before target runs; x marks a wrong predicted mean sign')
    save(fig, '02_forecast_boundaries')

def plot_iso(f, cells):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4), layout='constrained')
    colors = ['#007E87', '#B25433', '#8769A6']
    for ax, dataset in zip(axes, TAUS):
        h = f[(f.dataset == dataset) & (f.tau == 0)].groupby('rho').error.agg(['mean', 'std', 'count']).sort_index()
        margin = t.ppf(0.975, h['count'] - 1) * h['std'] / np.sqrt(h['count'])
        ax.plot(h.index, h['mean'], 'o-', color='#354044', label='Hard', ms=3)
        ax.fill_between(h.index.to_numpy(), (h['mean'] - margin).to_numpy(), (h['mean'] + margin).to_numpy(), color='#354044', alpha=0.1)
        for (tau, theta), color in zip(ISO[dataset], colors):
            g = f[(f.dataset == dataset) & (f.tau == tau) & (f.theta == theta)].groupby('rho').error.agg(['mean', 'std', 'count']).sort_index()
            margin = t.ppf(0.975, g['count'] - 1) * g['std'] / np.sqrt(g['count'])
            ax.plot(g.index, g['mean'], 'o-', color=color, label=f'tau={tau:g}, theta={theta:g}', ms=3)
            ax.fill_between(g.index.to_numpy(), (g['mean'] - margin).to_numpy(), (g['mean'] + margin).to_numpy(), color=color, alpha=0.12)
        ax.set_xscale('log')
        ax.set_xlabel('Total rho')
        ax.set_ylabel('Primary fidelity error (lower)')
        ax.set_title(dataset.title())
        ax.legend(fontsize=7)
        ax.grid(alpha=0.15)
    fig.suptitle('Equal r/tau gives identical noise ratios, but does it give identical fidelity?\nPolicies protect different widths; this is an equal-noise diagnostic, not equal protection')
    save(fig, '03_equal_noise_ratio')

def plot_crossover_map(cells):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), layout='constrained')
    records = []
    for ax, dataset in zip(axes, TAUS):
        x, y = np.meshgrid(np.log10(TAUS[dataset]), np.log10(THETAS[dataset]))
        values = np.full(x.shape, np.nan)
        labels = np.empty(x.shape, dtype=object)
        for i, theta in enumerate(THETAS[dataset]):
            for j, tau in enumerate(TAUS[dataset]):
                g = cells[(cells.dataset == dataset) & cells.is_map & (cells.theta == theta) & (cells.tau == tau)].sort_values('rho')
                roots = descending_roots(g.rho.to_numpy(), g['mean'].to_numpy())
                status = root_status(g.rho.to_numpy(), g['mean'].to_numpy())
                if len(roots) == 1:
                    values[i, j] = roots[0]['root']
                    labels[i, j] = f"{roots[0]['root']:.1g}"
                else:
                    labels[i, j] = {'soft_at_all_tested_budgets': 'S', 'hard_at_all_tested_budgets': 'H'}.get(status, '?')
                records.append(dict(dataset=dataset, tau=tau, theta=theta, status=status, roots=roots))
        im = ax.pcolormesh(x, y, np.ma.masked_invalid(values), shading='nearest', cmap='viridis', norm=LogNorm(vmin=0.0001, vmax=1))
        ax.set_facecolor('#E9E9E9')
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                dark_text = not np.isfinite(values[i, j]) or values[i, j] > 0.02
                ax.text(x[i, j], y[i, j], labels[i, j], ha='center', va='center', fontsize=7, color='#222222' if dark_text else 'white')
        axes_labels(ax, dataset)
        ax.set_title(dataset.title())
    fig.colorbar(im, ax=axes, shrink=0.8, label='Interpolated crossover rho (log scale)')
    fig.suptitle('Coarse crossover map: only three measured budget planes per cell\nS/H: soft/hard wins throughout the tested planes; ?: other. No invented out-of-range roots.')
    save(fig, '04_coarse_crossover_map')
    atomic_json(HOME / 'coarse_crossover_cells.json', records)

def runtime_audit():
    audit = {'stages': {}, 'rounds': 0, 'hard_target_certified': 0, 'hard_target_runs': 0, 'runtime': []}
    lock = json.loads((HOME / 'forecast_lock.json').read_text())
    assert hashlib.sha256((HOME / 'forecasts.json').read_bytes()).hexdigest() == lock['sha256']
    assert hashlib.sha256((HOME / 'target_manifest.json').read_bytes()).hexdigest() == lock['target_manifest_sha256']
    for filename, digest in manifest('target')['sources'].items():
        assert hashlib.sha256((HOME.parent / filename).read_bytes()).hexdigest() == digest, filename
    audit['frozen_algorithm_sources_verified'] = True
    for stage in ['pilot_scaled', 'pilot_unscaled', 'target']:
        configs = manifest(stage)['configs']
        audit['stages'][stage] = len(configs)
        times = []
        for cfg in configs:
            row = json.loads((HOME / 'results' / stage / f'{identity(cfg)}.json').read_text())
            assert row['config'] == cfg
            rho = cfg['rho'] * (row['target_n'] / row['train_n']) ** 2 if stage == 'pilot_scaled' else cfg['rho']
            assert math.isclose(rho, row['actual_rho'], rel_tol=1e-12)
            assert math.isclose(sum((h['sensitivity'] ** 2 / (2 * h['sigma'] ** 2) for h in row['history'])), rho, rel_tol=1e-12)
            audit['rounds'] += len(row['history'])
            if stage == 'target' and cfg['tau'] == 0:
                audit['hard_target_runs'] += 1
                audit['hard_target_certified'] += bool(row['all_hard_rounds_certified'])
            times.append(dict(dataset=cfg['dataset'], seconds=row['evolution_s']))
        med = pd.DataFrame(times).groupby('dataset').seconds.median()
        audit['runtime'] += [dict(stage=stage, dataset=d, median_evolution_s=float(v)) for d, v in med.items()]
    atomic_json(HOME / 'audit.json', audit)
    return audit
