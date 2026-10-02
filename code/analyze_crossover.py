# Crossover brackets, paired attribution and explicit limits of proxy formulas.

import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from crossover_study import HOME, TAUS, GRIDS, cfg_for
from data import load
from core import apd
from run import identity, atomic_json
FIG = HOME / 'figures'
COLORS = {'hard': '#354044', 'soft': '#007E87', 'matched': '#C15B3B'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

def collect():
    rows = []
    history = []
    diagnostics = []
    snapshots = []
    for path in (HOME / 'results').glob('*/*.json'):
        if '.failure.' in path.name:
            continue
        obj = json.loads(path.read_text())
        c = obj['config']
        d = c['dataset']
        meta = dict(**c, id=obj['id'], train_n=obj['train_n_actual'])
        error = obj['test']['hoerr_bits' if d == 'xor3' else 'l1_2way']
        auc = obj['test']['auc_tree' if d == 'xor3' else 'auc_lr']
        rows.append(dict(**meta, **obj['test'], error=error, auc=auc, noise_ratio=obj['noise_sd_ratio'], certified=obj['all_hard_rounds_certified']))
        history += [dict(**meta, **v) for v in obj['history']]
        diagnostics += [{**meta, **v} for v in obj['diagnostics']]
        snapshots += [dict(**meta, **v) for v in obj['snapshot_metrics']]
    return tuple((pd.DataFrame(v) for v in [rows, history, diagnostics, snapshots]))

def paired(values):
    v = np.asarray(values, float)
    n = len(v)
    margin = t.ppf(0.975, n - 1) * np.std(v, ddof=1) / math.sqrt(n)
    return dict(n=n, mean=float(v.mean()), ci_low=float(v.mean() - margin), ci_high=float(v.mean() + margin), p_raw=float(ttest_1samp(v, 0).pvalue) if np.std(v) > 0 else float(v.mean() == 0))

def adjust(frame):
    for _, ids in frame.groupby('family').groups.items():
        order = frame.loc[ids].sort_values('p_raw').index
        frame.loc[order, 'p_holm'] = np.maximum.accumulate([min(1, frame.loc[idx, 'p_raw'] * (len(order) - i)) for i, idx in enumerate(order)])
        frame.loc[order, 'family_size'] = len(order)
    return frame

def comparisons(frame):
    rows = []
    main = frame[frame.phase == 'crossover_transition']
    for (dataset, rho), group in main.groupby(['dataset', 'rho']):
        for metric in ['error', 'auc']:
            p = group.pivot(index='seed', columns='arm', values=metric)
            assert p.shape == (12, 3) and p.notna().all().all()
            values = {'gain': p.hard - p.soft, 'smoothing_cost': p.matched - p.hard, 'calibration_gain': p.matched - p.soft}
            assert np.allclose(values['gain'], values['calibration_gain'] - values['smoothing_cost'])
            for label, v in values.items():
                rows.append(dict(dataset=dataset, rho=rho, metric=metric, contrast=label, family=label + '_' + metric, **paired(v)))
    sampling = frame[frame.phase == 'crossover_sampling']
    for (dataset, rho), group in sampling.groupby(['dataset', 'rho']):
        p = group.pivot(index='seed', columns='arm', values='error')
        original = main[(main.dataset == dataset) & (main.rho == rho)].pivot(index='seed', columns='arm', values='error')
        for label, v in [('sampling_gain', p.hard - p.soft), ('selection_interaction', p.hard - p.soft - (original.hard - original.soft))]:
            rows.append(dict(dataset=dataset, rho=rho, metric='error', contrast=label, family=label, **paired(v)))
    for dataset, group in frame[frame.phase == 'crossover_noisefree'].groupby('dataset'):
        p = group.pivot(index='seed', columns='arm', values='error')
        rows.append(dict(dataset=dataset, rho=0, metric='error', contrast='noisefree_soft_penalty', family='noisefree', **paired(p.soft - p.hard)))
    result = adjust(pd.DataFrame(rows))
    result.to_csv(HOME / 'paired_contrasts.csv', index=False)
    return result

def descending_roots(rhos, gains):
    roots = []
    for i in range(len(rhos) - 1):
        if gains[i] > 0 and gains[i + 1] <= 0:
            fraction = gains[i] / (gains[i] - gains[i + 1])
            root = math.exp(math.log(rhos[i]) + fraction * math.log(rhos[i + 1] / rhos[i]))
            roots.append(dict(left=float(rhos[i]), right=float(rhos[i + 1]), root=root))
    return roots

def roots(frame, stats, diag):
    result = []
    rng = np.random.default_rng(9082026)
    main = frame[frame.phase == 'crossover_transition']
    for dataset in TAUS:
        g = main[main.dataset == dataset]
        hard = g[g.arm == 'hard'].pivot(index='seed', columns='rho', values='error').reindex(columns=GRIDS[dataset])
        soft = g[g.arm == 'soft'].pivot(index='seed', columns='rho', values='error').reindex(columns=GRIDS[dataset])
        diff = (hard - soft).to_numpy()
        rho = hard.columns.to_numpy()
        mean_roots = descending_roots(rho, diff.mean(0))
        samples = []
        multiple = 0
        missing = 0
        for _ in range(5000):
            indices = rng.integers(0, len(diff), size=len(diff))
            values = descending_roots(rho, diff[indices].mean(0))
            if values:
                samples.append(values[0]['root'])
                multiple += len(values) > 1
            else:
                missing += 1
        sig = stats[(stats.dataset == dataset) & (stats.family == 'gain_error')]
        positive = sig[(sig['mean'] > 0) & (sig.p_holm < 0.05)]
        negative = sig[(sig['mean'] < 0) & (sig.p_holm < 0.05)]
        if mean_roots:
            positive = positive[positive.rho <= mean_roots[0]['left']]
            negative = negative[negative.rho >= mean_roots[0]['right']]
        initial = diag[(diag.dataset == dataset) & (diag.phase == 'crossover_transition') & (diag['round'] == 0)].drop_duplicates('seed')
        n = float(g.train_n.iloc[0])
        m = float(initial.m.iloc[0])
        T = 8
        q = float(g[g.arm == 'soft'].noise_ratio.iloc[0])
        hist_root = m * T * (1 - q * q) / (n * n * initial.normalized_bias2.mean())
        result.append(dict(dataset=dataset, tau=TAUS[dataset], q=q, mean_roots=mean_roots, bootstrap_ci=list(np.quantile(samples, [0.025, 0.975])) if samples else None, bootstrap_missing=missing, bootstrap_multiple=multiple, bootstrap_draws=5000, significant_positive_endpoint=float(positive.rho.max()) if len(positive) else None, significant_negative_endpoint=float(negative.rho.min()) if len(negative) else None, rho_hist_initial=float(hist_root), hist_to_fidelity_ratio=float(hist_root / mean_roots[0]['root']) if mean_roots else None, initial_contrast_retention=float(initial.contrast_retention.mean()), initial_snr_ratio=float(initial.contrast_retention.mean() / q)))
    atomic_json(HOME / 'crossover_estimates.json', result)
    return result

def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=210, bbox_inches='tight')
    plt.close(fig)

def curve(ax, g, metric, arm, label=None, linestyle='-'):
    values = g[g.arm == arm].groupby('rho')[metric].agg(['mean', 'std', 'count']).sort_index()
    margin = t.ppf(0.975, values['count'] - 1) * values['std'] / np.sqrt(values['count'])
    ax.plot(values.index, values['mean'], 'o' + linestyle, color=COLORS[arm], label=label or arm, ms=4)
    ax.fill_between(values.index.to_numpy(), (values['mean'] - margin).to_numpy(), (values['mean'] + margin).to_numpy(), color=COLORS[arm], alpha=0.12)

def plot_main(frame, stats, estimates):
    main = frame[frame.phase == 'crossover_transition']
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), layout='constrained')
    for col, entry in enumerate(estimates):
        dataset = entry['dataset']
        g = main[main.dataset == dataset]
        for arm, label in [('hard', 'Hard'), ('soft', 'Soft + policy noise'), ('matched', 'Soft + hard noise')]:
            curve(axes[0, col], g, 'error', arm, label, '--' if arm == 'matched' else '-')
        s = stats[(stats.dataset == dataset) & (stats.family == 'gain_error')].sort_values('rho')
        axes[1, col].errorbar(s.rho, s['mean'], yerr=[s['mean'] - s.ci_low, s.ci_high - s['mean']], fmt='o-', color=COLORS['soft'], capsize=3)
        axes[1, col].axhline(0, color='black', lw=0.8)
        for ax in axes[:, col]:
            ax.set_xscale('log')
            ax.set_xlabel('Total rho (larger = weaker privacy)')
            ax.grid(alpha=0.15)
            if entry['mean_roots']:
                root = entry['mean_roots'][0]
                ax.axvspan(root['left'], root['right'], color='#B9A170', alpha=0.15)
                ax.axvline(root['root'], color='#766C50', ls=':', lw=1)
        axes[0, col].set_title(f'{dataset.title()} | tau={TAUS[dataset]:g}\nLocal policy, fixed across the curve')
        axes[0, col].set_ylabel('HOErr (lower)' if dataset == 'xor3' else 'PWErr (lower)')
        axes[1, col].set_ylabel('Hard error - soft error\nPositive: soft wins')
        axes[0, col].legend(fontsize=7)
    fig.suptitle('The upper crossover: show both sides, not only the winning budget window\n12 paired seeds; bands/whiskers are unadjusted 95% seed intervals')
    save(fig, '01_crossover_and_uncertainty')

def plot_components(stats):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), layout='constrained')
    for ax, dataset in zip(axes, TAUS):
        for contrast, label, color in [('smoothing_cost', 'C: cost of soft votes', '#C15B3B'), ('calibration_gain', 'N: benefit of lower noise', '#007E87')]:
            s = stats[(stats.dataset == dataset) & (stats.metric == 'error') & (stats.contrast == contrast)].sort_values('rho')
            ax.plot(s.rho, s['mean'], 'o-', label=label, color=color, ms=4)
            ax.fill_between(s.rho.to_numpy(), s.ci_low.to_numpy(), s.ci_high.to_numpy(), color=color, alpha=0.12)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xscale('log')
        ax.grid(alpha=0.15)
        ax.set_title(dataset.title())
        ax.set_xlabel('Total rho')
        ax.set_ylabel('End-to-end error difference')
        ax.legend(fontsize=7)
    fig.suptitle('Measured crossover condition: N(rho) = C(rho)\nExact contrast identity, NOT an independent prediction of final fidelity')
    save(fig, '02_cost_versus_noise_benefit')

def plot_dynamics(frame, history, diag, stats):
    main = frame[frame.phase == 'crossover_transition']
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 9), layout='constrained')
    for col, dataset in enumerate(TAUS):
        g = main[main.dataset == dataset]
        schema, train, _, _ = load(dataset, 0, 40 if dataset == 'xor3' else 0)
        rng = np.random.default_rng(1098)
        points = train[rng.choice(len(train), min(2000, len(train)), replace=False)]
        for arm in ['hard', 'soft']:
            curve(axes[0, col], g, 'apd', arm)
        axes[0, col].axhline(apd(points, schema), color='#88764A', ls=':', label='Real reference sample')
        axes[0, col].set_title(dataset.title())
        axes[0, col].set_ylabel('Final APD (not a fidelity score)')
        axes[0, col].legend(fontsize=7)
        for arm in ['hard', 'soft']:
            final = history[(history.phase == 'crossover_transition') & (history.dataset == dataset) & (history['round'] == 7)]
            curve(axes[1, col], final, 'snr', arm)
        axes[1, col].set_yscale('log')
        axes[1, col].set_ylabel('Final histogram contrast / noise SD')
        for selection, phase, color in [('Hybrid', 'crossover_transition', '#354044'), ('Sampling-only', 'crossover_sampling', '#007E87')]:
            grids = [0.03, 0.18, 1.0] if dataset == 'xor3' else [0.0001, 0.001, 0.01]
            x = frame[(frame.dataset == dataset) & (frame.phase == phase) & frame.rho.isin(grids) & frame.arm.isin(['hard', 'soft'])]
            p = x.pivot(index=['seed', 'rho'], columns='arm', values='error')
            gain = (p.hard - p.soft).groupby('rho').agg(['mean', 'std', 'count'])
            err = t.ppf(0.975, gain['count'] - 1) * gain['std'] / np.sqrt(gain['count'])
            axes[2, col].errorbar(gain.index, gain['mean'], yerr=err, fmt='o-', label=selection, color=color, capsize=3)
        axes[2, col].axhline(0, color='black', lw=0.8)
        axes[2, col].set_ylabel('Hard - soft error\nSelection-rule intervention')
        axes[2, col].legend(fontsize=7)
        for ax in axes[:, col]:
            ax.set_xscale('log')
            ax.set_xlabel('Total rho')
            ax.grid(alpha=0.15)
    fig.suptitle('Diagnose the dynamics and intervene on selection\nAPD and SNR are supporting diagnostics; fidelity remains the primary outcome')
    save(fig, '03_dynamics_and_selection_intervention')

def plot_distributions(frame):
    fig, axes = plt.subplots(2, 5, figsize=(12.6, 5.7), layout='constrained')
    for row, dataset in enumerate(['adult', 'bank']):
        schema, train, _, _ = load(dataset)
        ds, cards = schema.discrete(train)
        configs = [('Real training', None, None), ('Hard | rho=.0001', 'hard', 0.0001), ('Soft | rho=.0001', 'soft', 0.0001), ('Hard | rho=.01', 'hard', 0.01), ('Soft | rho=.01', 'soft', 0.01)]
        matrices = []
        for title, arm, rho in configs:
            arrays = [train] if arm is None else [np.load(HOME / 'results/transition' / f'{rid}.npz')['synthetic'] for rid in frame[(frame.phase == 'crossover_transition') & (frame.dataset == dataset) & (frame.arm == arm) & (frame.rho == rho)].id]
            values = []
            for points in arrays:
                d, _ = schema.discrete(points)
                cells = np.ravel_multi_index(d[:, :2].T, cards[:2])
                values.append(np.bincount(cells, minlength=cards[0] * cards[1]).reshape(cards[:2]) / len(points))
            matrices.append(np.mean(values, axis=0))
        vmax = max((v.max() for v in matrices))
        for col, ((title, _, _), mat) in enumerate(zip(configs, matrices)):
            im = axes[row, col].imshow(mat, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].set_xlabel('Hours bin' if dataset == 'adult' else 'Balance bin')
            axes[row, col].set_ylabel(dataset.title() + ' age bin')
        fig.colorbar(im, ax=axes[row, :], shrink=0.8, label='Joint probability')
    fig.suptitle('How the output distribution changes across the transition\nSynthetic panels average all 12 seeds; fixed bins and shared color scale within each dataset')
    save(fig, '04_joint_distribution_shift')

def audit():
    result = {'stages': {}, 'rounds_checked': 0, 'scaling_replays': []}
    for stage in ['transition', 'noisefree', 'sampling', 'scaling']:
        configs = json.loads((HOME / f'{stage}_manifest.json').read_text())['configs']
        result['stages'][stage] = len(configs)
        for cfg in configs:
            row = json.loads((HOME / 'results' / stage / f'{identity(cfg)}.json').read_text())
            assert row['config'] == cfg
            assert len(row['history']) == 8
            if stage == 'sampling':
                assert 'selection_intervention' in row
                assert [h['m_candidates'] for h in row['history']] == [512, 512, 512, 1024, 1024, 1024, 1024, 1024]
            if cfg['rho'] == 0:
                assert not row['private_comparison'] and all((h['sigma'] == 0 for h in row['history']))
            else:
                total = sum((h['sensitivity'] ** 2 / (2 * h['sigma'] ** 2) for h in row['history']))
                assert math.isclose(total, cfg['rho'], rel_tol=1e-12)
            result['rounds_checked'] += 8
            if stage == 'scaling':
                original = cfg_for(cfg['dataset'], cfg['seed'], cfg['rho'] * 4, cfg['arm'])
                a = np.load(HOME / 'results/transition' / f'{identity(original)}.npz')
                b = np.load(HOME / 'results/scaling' / f'{identity(cfg)}.npz')
                identical = {name: bool(np.array_equal(a[name], b[name])) for name in a.files}
                result['scaling_replays'].append(dict(dataset=cfg['dataset'], seed=cfg['seed'], arm=cfg['arm'], identical_snapshots=identical))
    atomic_json(HOME / 'audit.json', result)
    return result

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])
