# Plot the paper fidelity, sensitivity, runtime, and forecast panels.

from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.signal import find_peaks
from scipy.stats import t
DATASETS = ['adult', 'bank', 'xor3']
NAMES = {'adult': 'Adult', 'bank': 'Bank', 'xor3': 'XOR3'}
PRIMARY = {'adult': 'l1_2way', 'bank': 'l1_2way', 'xor3': 'hoerr_bits'}
COLORS = {'adult': '#007E87', 'bank': '#B45336', 'xor3': '#76589B', 'hard': '#343D40', 'soft': '#007E87', 'matched': '#B45336'}
PLOT_AUDIT = []
MINIMUM_AUDIT = []
ADULT_SWEET_STEMS = {'fig05_sweet_adult_rho_0p0001', 'fig05_sweet_adult_rho_0p0006', 'fig05_sweet_adult_rho_0p003'}
EQUAL_NOISE_STEMS = {f'fig08_equal_noise_{dataset}' for dataset in DATASETS}
MECHANISM_STEMS = {f'fig06_mechanism_{dataset}' for dataset in DATASETS}
SENSITIVITY_STEMS = {'fig03_direct_sensitivity'}
PDF_FIGURE_STEMS = ADULT_SWEET_STEMS | EQUAL_NOISE_STEMS | MECHANISM_STEMS | SENSITIVITY_STEMS
CROSSOVER_STEMS = {f'exp1_crossover_{dataset}' for dataset in DATASETS}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})

def interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    if len(values) < 2:
        return (mean, mean, mean)
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return (mean, mean - margin, mean + margin)

def rho_name(rho):
    return f'{rho:g}'.replace('.', 'p').replace('-', 'm')

def panel_axes(figsize=(4.1, 3.15)):
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.grid(alpha=0.16)
    return (fig, ax)

def common_y_limits(curves, requested, step=0.2):
    """Use requested common limits, expanding to rounded ticks if CIs require it."""
    low = min((float(np.min(item[0])) for item in curves))
    high = max((float(np.max(item[1])) for item in curves))
    return (min(requested[0], math.floor(low / step) * step), max(requested[1], math.ceil(high / step) * step))

def set_common_y(ax, limits, step=0.2):
    ax.set_ylim(*limits)
    ax.set_yticks(np.arange(round(limits[0] / step), round(limits[1] / step) + 1) * step)

def curve_rows(frame, value='error'):
    return sorted(((float(rho), *interval(group[value])) for rho, group in frame.groupby('rho')))

def figure02_budget_panels(outputs, only=None):
    frame = pd.read_csv(RESULTS / 'crossover/runs.csv')
    grouped = {}
    all_bands = []
    for dataset in DATASETS:
        subset = frame[frame.dataset.eq(dataset)]
        curves = [(0.0, 'hard', COLORS['hard']), (0.015, 'BF-Soft $\\tau=0.015$', COLORS['soft']), (0.03, 'BF-Soft $\\tau=0.03$', COLORS['matched']), (0.06, 'BF-Soft $\\tau=0.06$', '#8068A0')]
        grouped[dataset] = []
        for tau, label, color in curves:
            rows = curve_rows(subset[subset.tau.eq(tau)])
            if len(rows) != subset.rho.nunique():
                raise RuntimeError(f'Incomplete crossover curve: {dataset}, {tau}')
            grouped[dataset].append((rows, label, color))
            all_bands.append((np.array([r[2] for r in rows]), np.array([r[3] for r in rows])))
    y_limits = common_y_limits(all_bands, (0, 1.4))
    for dataset in DATASETS:
        stem = f'exp1_crossover_{dataset}'
        if only is not None and stem not in only:
            continue
        fig, ax = panel_axes()
        for rows, label, color in grouped[dataset]:
            x = np.asarray([row[0] for row in rows])
            mean = np.asarray([row[1] for row in rows])
            low = np.asarray([row[2] for row in rows])
            high = np.asarray([row[3] for row in rows])
            ax.plot(x, mean, 'o-', ms=3.0, lw=1.6, color=color, label='Hard' if label == 'hard' else label)
            ax.fill_between(x, low, high, color=color, alpha=0.1)
        ax.set_xscale('log')
        ax.set_xlim(1e-05, 1.0)
        set_common_y(ax, y_limits)
        ax.set_xlabel('Total privacy budget $\\rho$')
        ax.set_ylabel('Primary $L_1$ error (lower better)')
        ax.legend(fontsize=6.8)
        outputs.append(save(fig, stem))

def figure03_direct_sensitivity(outputs):
    frame = pd.read_csv(RESULTS / 'sensitivity/temperature.csv')
    stats = frame.groupby('tau').exact_soft.median().reset_index(name='soft_exact')
    bound = frame.groupby('tau').bound_soft.median().reindex(stats.tau).to_numpy()
    hard = frame.groupby('tau').exact_hard.median().reindex(stats.tau).to_numpy()
    fig, ax = panel_axes((3.5, 2.55))
    ax.plot(stats.tau, hard, color=COLORS['hard'], lw=1.9, label='Hard voting')
    ax.plot(stats.tau, bound, 'o--', ms=3.2, color='#B45336', lw=1.8, label='Soft voting (analytical)')
    ax.plot(stats.tau, stats.soft_exact, 'o-', ms=3.2, color=COLORS['soft'], lw=1.9, label='Soft voting (exact)')
    ax.set_xscale('symlog', linthresh=0.001, linscale=0.7)
    ticks = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    ax.set_xticks(ticks, ['0', '.001', '.003', '.01', '.03', '.1', '.3', '1', '3', '10', '30', '100'])
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlim(-0.00025, 100.0)
    ax.set_ylim(0, 1.52)
    ax.set_xlabel('Temperature $\\tau$')
    ax.set_ylabel('Sensitivity')
    ax.grid(False, axis='x')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.77), frameon=True, fancybox=False, framealpha=0.96, facecolor='white', edgecolor='#6B7476')
    outputs.append(save(fig, 'fig03_direct_sensitivity'))

def figure04_theta_runtime(outputs):
    frame = pd.read_csv(RESULTS / 'sensitivity/reach.csv')
    theta = frame[frame.study == 'theta']
    fig, ax = panel_axes((3.5, 2.55))
    for column, label, color in [('hard_exact', 'Hard exact', COLORS['hard']), ('soft_bound', 'BF-Soft analytical bound', '#B45336'), ('soft_exact', 'BF-Soft exact', COLORS['soft'])]:
        stats = theta.groupby('theta')[column].agg(median='median', low=lambda x: x.quantile(0.25), high=lambda x: x.quantile(0.75)).reset_index()
        ax.plot(stats.theta, stats['median'], 'o-', lw=1.8, color=color, label=label)
        if column != 'soft_bound':
            ax.fill_between(stats.theta, stats.low, stats.high, color=color, alpha=0.12)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Policy reach $\\theta$ (ordinal steps)')
    ax.set_ylabel('Bounded $\\ell_2$ sensitivity')
    ax.legend()
    outputs.append(save(fig, 'fig04_theta_sensitivity'))
    runtime = pd.read_csv(RESULTS / 'runtime/candidates.csv')
    fig, ax = panel_axes((3.5, 2.55))
    for column, label, color, marker, linestyle in [('soft_exact_ms', 'Soft exact', COLORS['soft'], 'o', '-'), ('hard_exact_ms', 'Hard exact', COLORS['hard'], 's', '--'), ('analytic_ms', 'Soft bound', '#B45336', '^', ':')]:
        stats = runtime.groupby('N')[column].agg(median='median', low=lambda x: x.quantile(0.25), high=lambda x: x.quantile(0.75)).reset_index()
        ax.plot(stats.N, stats['median'], marker=marker, linestyle=linestyle, lw=1.8, ms=4, color=color, label=label)
        ax.fill_between(stats.N, stats.low, stats.high, color=color, alpha=0.12)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_ylim(7e-05, 4000)
    ax.set_xticks([64, 256, 1024, 4096, 8192], ['64', '256', '1k', '4k', '8k'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(55, 10000)
    ax.set_xlabel('Candidate population size $N$')
    ax.set_ylabel('Sensitivity-calculation time (ms)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.93), frameon=True, fancybox=False, framealpha=0.96, facecolor='white', edgecolor='#6B7476', fontsize=7.0, handlelength=1.4, handletextpad=0.65, borderpad=0.45, labelspacing=0.35)
    outputs.append(save(fig, 'fig04_runtime_candidates'))

def figure05_sweet_panels(outputs, only=None):
    frame = pd.read_csv(RESULTS / 'temperature_sweep/runs.csv')
    selected = {dataset: [0.0001, 0.0006, 0.003] for dataset in DATASETS}
    requested_y = {'adult': (0.2, 1.0), 'bank': (0.4, 1.2), 'xor3': (0.2, 1.2)}
    y_limits = {}
    for dataset in DATASETS:
        metric = PRIMARY[dataset]
        subset = frame[frame.dataset.eq(dataset) & frame.rho.isin(selected[dataset])]
        bands = []
        for _, group in subset.groupby(['rho', 'tau']):
            _, low, high = interval(group[metric])
            bands.append((np.array([low]), np.array([high])))
        y_limits[dataset] = common_y_limits(bands, requested_y[dataset])
    for dataset in DATASETS:
        metric = PRIMARY[dataset]
        for rho in selected[dataset]:
            stem = f'fig05_sweet_{dataset}_rho_{rho_name(rho)}'
            if only is not None and stem not in only:
                continue
            soft_rows = frame[(frame.dataset == dataset) & frame.rho.eq(rho) & frame.roles.str.contains('common-temperature-soft', na=False) & frame.calibration.eq('analytic')]
            hard_rows = frame[(frame.dataset == dataset) & frame.rho.eq(rho) & frame.tau.eq(0) & frame.adjacency.eq('B')]
            soft = []
            for tau, group in soft_rows.groupby('tau'):
                mean, low, high = interval(group[metric])
                soft.append((tau, mean, low, high))
            soft = sorted(soft)
            hard_mean, hard_low, hard_high = interval(hard_rows.groupby('seed')[metric].mean())
            fig, ax = panel_axes()
            if dataset == 'adult':
                ax.grid(False)
            x = np.asarray([row[0] for row in soft])
            mean = np.asarray([row[1] for row in soft])
            low = np.asarray([row[2] for row in soft])
            high = np.asarray([row[3] for row in soft])
            full_x = np.concatenate([[0.0], x])
            full_mean = np.concatenate([[hard_mean], mean])
            full_low = np.concatenate([[hard_low], low])
            full_high = np.concatenate([[hard_high], high])
            ax.plot(full_x, full_mean, 'o-', ms=3.2, color=COLORS[dataset], label='BF-Soft ($\\tau>0$)')
            ax.fill_between(full_x, full_low, full_high, color=COLORS[dataset], alpha=0.15)
            ax.axhline(hard_mean, color=COLORS['hard'], ls='--', lw=1.5, label='Hard')
            ax.axhspan(hard_low, hard_high, color=COLORS['hard'], alpha=0.08)
            if dataset != 'adult':
                ax.scatter([0], [hard_mean], marker='s', s=30, color=COLORS['hard'], zorder=4)
            best = int(np.argmin(mean))
            if dataset == 'adult':
                MINIMUM_AUDIT.append({'figure': stem, 'method': 'BF-Soft', 'tau': float(x[best]), 'mean_error': float(mean[best]), 'scope': 'lowest mean over positive sampled temperatures'})
            ax.scatter([x[best]], [mean[best]], marker='*', s=95, color=COLORS[dataset] if dataset == 'adult' else '#D28A00', edgecolor='white', zorder=6, label='Lowest sampled Soft mean' if dataset == 'adult' else 'Best tested soft setting')
            if dataset == 'xor3':
                prominence = max(0.01, 0.04 * float(np.ptp(mean)))
                local, _ = find_peaks(-mean, prominence=prominence)
                local = [i for i in local if i != best and mean[i] < hard_mean]
                if local:
                    ax.scatter(x[local], mean[local], marker='D', s=30, facecolors='none', edgecolors='#D28A00', zorder=5, label='Prominent secondary basin')
            ax.set_xscale('symlog', linthresh=0.001, linscale=0.75)
            ticks = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
            ax.set_xticks(ticks, ['0', '.001', '.003', '.01', '.03', '.1', '.3'])
            ax.set_xlim(-0.00012, 0.42)
            set_common_y(ax, y_limits[dataset])
            ax.set_xlabel('Temperature $\\tau$')
            ax.set_ylabel('Primary fidelity error (lower)')
            ax.legend(fontsize=6.4, loc='upper left' if dataset == 'adult' else 'best')
            outputs.append(save(fig, stem))

def figure06_mechanism_panels(outputs):
    frame = pd.read_csv(RESULTS / 'mechanism/paired_contrasts.csv')
    for dataset in DATASETS:
        subset = frame[(frame.dataset == dataset) & (frame.metric == 'error')]
        fig, ax = panel_axes()
        curves = [('smoothing_cost', '$C$: smoothing cost', '#B45336', '-'), ('calibration_gain', '$N$: noise-reduction benefit', '#397CB6', '-'), ('gain', '$G=N-C$: final gain', COLORS['soft'], '--')]
        for contrast, label, color, linestyle in curves:
            group = subset[subset.contrast == contrast].sort_values('rho')
            ax.plot(group.rho, group['mean'], 'o', ms=3.5, color=color, linestyle=linestyle, lw=1.7, label=label)
            ax.fill_between(group.rho, group.ci_low, group.ci_high, color=color, alpha=0.11)
        ax.axhline(0, color='#555', lw=0.8)
        ax.set_xscale('log')
        ax.set_xlim(1e-05, 1.2)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        ax.set_xlabel('Total privacy budget $\\rho$')
        ax.set_ylabel('End-to-end primary-error difference')
        ax.legend(loc='lower left', ncol=1, fontsize=6.4, frameon=False, borderaxespad=0.25, labelspacing=0.25)
        fig.canvas.draw()
        for tick in ax.xaxis.get_major_ticks():
            if np.isclose(tick.get_loc(), 0.0001):
                tick.gridline.set_visible(False)
        outputs.append(save(fig, f'fig06_mechanism_{dataset}'))

def figure07_xor_interaction(outputs):
    runs = pd.read_csv(RESULTS / 'interaction/runs.csv')
    taus = [0.00025, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.04, 0.08]
    thetas = [1.0, 2.0, 3.0, 4.0]
    for rho in [0.03, 0.18, 1.0]:
        subset = runs[runs.rho.eq(rho)].copy()
        hard = subset[subset.tau.eq(0)][['seed', 'error']].rename(columns={'error': 'hard_error'})
        measured_rows = subset[subset.tau.gt(0)].merge(hard, on='seed', how='left')
        measured_rows['relative_gain'] = (measured_rows.hard_error - measured_rows.error) / measured_rows.hard_error
        stats = []
        for (theta, tau), group in measured_rows.groupby(['theta', 'tau']):
            mean, low, high = interval(group.relative_gain)
            stats.append({'theta': theta, 'tau': tau, 'relative_gain': mean, 'ci_low': low, 'ci_high': high})
        stats = pd.DataFrame(stats)
        measured = stats.pivot(index='theta', columns='tau', values='relative_gain').reindex(index=thetas, columns=taus).to_numpy()
        x = np.asarray([0.0, *taus])
        z = np.column_stack([np.zeros(len(thetas)), measured])
        fig, ax = panel_axes((4.35, 3.35))
        levels = np.linspace(-0.6, 0.6, 25)
        image = ax.contourf(x, thetas, z, levels=levels, cmap='RdBu', norm=TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6), extend='both')
        if np.nanmin(z) < 0 < np.nanmax(z):
            ax.contour(x, thetas, z, levels=[0], colors='black', linewidths=1.5)
        ax.axvline(0, color='#5A6668', lw=1.0, zorder=4)
        ax.scatter(np.repeat(taus, len(thetas)), np.tile(thetas, len(taus)), s=10, facecolors='none', edgecolors='#303A3C', linewidths=0.55, alpha=0.78, zorder=5, clip_on=False)
        best_cells = np.argwhere(np.isclose(measured, np.nanmax(measured), rtol=0, atol=1e-12))
        for i, j in best_cells:
            ax.scatter(taus[j], thetas[i], marker='*', s=105, color='#D28A00', edgecolor='white', linewidth=0.9, zorder=7, clip_on=False)
        ax.set_xscale('symlog', linthresh=0.0005, linscale=0.35)
        ax.set_xlim(-4e-05, 0.085)
        ax.set_xticks([0, 0.001, 0.003, 0.01, 0.04, 0.08], ['0', '.001', '.003', '.01', '.04', '.08'])
        ax.set_ylim(1, 4)
        ax.set_yticks(thetas, ['1', '2', '3', '4'])
        ax.set_xlabel('Temperature $\\tau$ (0 is the Hard-vote limit)')
        ax.set_ylabel('Policy reach $\\theta$ (ordinal steps)')
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label('Relative fidelity gain (clipped at +/-0.60)')
        outputs.append(save(fig, f'fig07_xor3_rho_{rho_name(rho)}'))

def prepare_equal_noise_data(dataset):
    restored = dataset == 'xor3'
    data_path = RESULTS / 'equal_noise/xor3_runs.csv' if restored else RESULTS / 'equal_noise/adult_bank_runs.csv'
    manifest_path = RESULTS / 'equal_noise/xor3_settings.json' if restored else RESULTS / 'equal_noise/adult_bank_settings.json'
    frame = pd.read_csv(data_path)
    subset = frame[frame.dataset.eq(dataset)].copy()
    hard = subset[subset.tau.eq(0)][['rho', 'seed', 'error']].rename(columns={'error': 'hard_error'})
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    metadata = manifest if restored else manifest['metadata']
    settings = [(entry['tau'], entry['theta']) for entry in metadata['settings'] if entry['dataset'] == dataset]
    parts = []
    for tau, theta in settings:
        part = subset[subset.arm.eq('soft') & subset.theta.eq(theta)][['rho', 'seed', 'tau', 'theta', 'error', 'q_actual']].copy()
        if not np.allclose(part.tau, tau, rtol=0, atol=1e-12):
            raise RuntimeError(f'Equal-q temperature mismatch: {dataset}, {theta}')
        parts.append(part)
    soft = pd.concat(parts, ignore_index=True)
    return (hard, soft, settings)

def figure08_equal_noise(outputs, only=None):
    grouped = {}
    all_bands = []
    for dataset in DATASETS:
        hard, soft, settings = prepare_equal_noise_data(dataset)
        curves = [(curve_rows(hard, 'hard_error'), 'Hard', COLORS['hard'], None)]
        palette = ['#007E87', '#B45336', '#8068A0']
        for (tau, theta), color in zip(settings, palette):
            subset = soft[soft.theta.eq(theta)]
            q = float(subset.q_actual.median())
            curves.append((curve_rows(subset), f'$\\tau={tau:.4g},\\ \\theta={theta:g}$ ($q={q:.3f}$)', color, (tau, theta)))
        if any((len(curve[0]) != len(curves[0][0]) for curve in curves)):
            raise RuntimeError(f'Incomplete equal-q curves for {dataset}')
        grouped[dataset] = curves
        all_bands.extend(((np.array([r[2] for r in rows]), np.array([r[3] for r in rows])) for rows, _, _, _ in curves))
    y_limits = common_y_limits(all_bands, (0, 1.4))
    for dataset in DATASETS:
        stem = f'fig08_equal_noise_{dataset}'
        if only is not None and stem not in only:
            continue
        fig, ax = panel_axes((4.25, 3.25))
        ax.grid(False)
        for rows, label, color, setting in grouped[dataset]:
            x = np.asarray([row[0] for row in rows])
            mean = np.asarray([row[1] for row in rows])
            low = np.asarray([row[2] for row in rows])
            high = np.asarray([row[3] for row in rows])
            ax.plot(x, mean, 'o-', ms=3, color=color, label=label)
            ax.fill_between(x, low, high, color=color, alpha=0.11)
        ax.set_xscale('log')
        ax.set_xlim(1e-05, 1.0)
        set_common_y(ax, y_limits)
        ax.set_xlabel('Total privacy budget $\\rho$')
        ax.set_ylabel('Primary fidelity error (lower)')
        ax.legend(fontsize=6.5)
        outputs.append(save(fig, stem))

def figure09_forecast_panels(outputs):
    cells = pd.read_csv(RESULTS / 'forecast/cells.csv')
    for rho in [0.0001, 0.0006, 0.003]:
        subset = cells[cells.dataset.eq('adult') & cells.rho.eq(rho)]
        observed = subset.pivot(index='theta', columns='tau', values='relative_gain').sort_index()
        predicted = subset.pivot(index='theta', columns='tau', values='pilot_scaled_gain').reindex(index=observed.index, columns=observed.columns)
        tau_values = observed.columns.to_numpy(float)
        y = observed.index.to_numpy(float)
        observed_values = observed.to_numpy()
        predicted_values = predicted.to_numpy()
        fig, ax = panel_axes((4.15, 3.3))
        image = ax.contourf(tau_values, y, observed_values, levels=np.linspace(-0.6, 0.6, 25), cmap='RdBu', norm=TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6), extend='both')
        for array, color, linestyle in [(observed_values, 'black', '-'), (predicted_values, '#D28A00', '--')]:
            if np.nanmin(array) < 0 < np.nanmax(array):
                ax.contour(tau_values, y, array, levels=[0], colors=[color], linestyles=[linestyle], linewidths=1.5)
        ax.scatter(subset.tau, subset.theta, s=10, facecolors='none', edgecolors='#303A3C', linewidths=0.55, alpha=0.78, clip_on=False, zorder=5)
        ax.axvspan(0, tau_values[0], facecolor='#F3F4F2', edgecolor='#B9C0C2', hatch='////', linewidth=0, zorder=3)
        ax.axvline(tau_values[0], color='#8B9698', lw=0.65, ls=':', zorder=4)
        ax.axvline(0, color='#4D595B', lw=1.15, zorder=6)
        ax.set_xscale('symlog', linthresh=0.0005, linscale=0.8)
        ax.set_yscale('log')
        ax.set_xlim(-4e-05, 0.095)
        ax.set_xticks([0, 0.001, 0.003, 0.015, 0.03, 0.09], ['0', '.001', '.003', '.015', '.03', '.09'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(1, 28)
        ax.set_yticks([1, 2, 4, 8, 16, 28], ['1', '2', '4', '8', '16', '28'])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Temperature $\\tau$ (0 is the Hard-vote limit)')
        ax.set_ylabel('Age reach $\\theta$ (years)')
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label('Mean fidelity gain $(E_H-E_S)/E_H$')
        outputs.append(save(fig, f'fig09_forecast_adult_rho_{rho_name(rho)}'))
RESULTS = Path(__file__).resolve().parents[1] / 'results'
