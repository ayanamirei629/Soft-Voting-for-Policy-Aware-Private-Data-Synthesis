# Analyze paired confirmation results and plot the paper policy and size panels.

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp
from data import ROOT, schema_for
from run import atomic_json, identity
from generalization_study import HOME, BUDGETS, CONFIRM_SEEDS, compose_xor, compatible, result_path, primary, oracle_xor_error
from metrics import marginal_errors
FIG = ROOT / 'figures'
ANALYSIS = HOME / 'analysis'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.8})
COLORS = ['#007E87', '#75589B', '#B45336', '#2A8C61']
TAU_COLORS = {0.0: '#343D40', 0.015: '#75589B', 0.15: '#B45336'}
METHOD_COLORS = {'hard': '#343D40', 'fixed_tau_0p03': '#B45336', 'matched_q_0p2': '#75589B', 'public_selected': '#007E87'}
MARKERS = ['o', 's', '^', 'D']
LINE_STYLES = ['-', '--', ':']

def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(values):
        raise ValueError('All plotted observations must be finite')
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 and np.ptp(values) > 0 else 0.0
    return (mean, mean - margin, mean + margin)

def stem_number(value):
    return f'{value:g}'.replace('.', 'p').replace('-', 'm')

def policy_label(dataset, policy):
    mapping = {'age_secondary': 'Age + hours' if dataset == 'adult' else 'Age + balance', 'age_education': 'Age + education', 'all_predictors': 'All predictors', 'all_columns': 'All Xi + Y' if dataset.startswith('xor') else 'All columns (incl. label)', 'x1': 'X1', 'x1_x2': 'X1 + X2', 'all_inputs': 'All Xi', 'hours.per.week': 'Hours', 'education.num': 'Education', 'workclass': 'Workclass', 'income': 'Income', 'balance': 'Balance', 'education': 'Education', 'job': 'Job', 'y': 'Label y', 'age': 'Age'}
    return mapping[policy]

def axes(figsize=(4.5, 3.55), left=0.17, bottom=0.22):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=left, right=0.97, bottom=bottom, top=0.97)
    ax.grid(alpha=0.14, zorder=0)
    return (fig, ax)

def legend_below(ax, ncol=2):
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=ncol, frameon=False, handlelength=1.5, columnspacing=1.15, handletextpad=0.45, borderaxespad=0.0, labelspacing=0.4)

def paired_cells(manifest, f):
    cells = pd.DataFrame(manifest['cells'])
    expanded = cells.merge(f.drop(columns=['rho', 'tau', 'calibration', 'case_id', 'seed']), left_on='result_id', right_on='id', validate='many_to_one')
    metrics = sorted(set((primary(c['dataset']) for c in manifest['cases'])) | {'auc_lr', 'auc_tree', 'parity_validity', 'hoerr_bits_oracle', 'l1_2way_protected_involving', 'l1_2way_unprotected_only'})
    metrics += [c for c in f if c.startswith('l1_1way__') or c.startswith('l1_2way__')]
    metrics = [m for m in metrics if m in expanded]
    records = []
    for (cid, rho, seed), g in expanded.groupby(['case_id', 'rho', 'seed']):
        for _, soft in g[~g.setting.str.contains('matched_noise__') & ~g.setting.str.endswith('hard')].iterrows():
            norm = soft.setting.startswith('normalized_noise__')
            hard_name = 'normalized_noise__hard' if norm else 'hard'
            hard = g[g.setting == hard_name].iloc[0]
            control_rows = g[g.setting == 'matched_noise__' + soft.setting]
            control = control_rows.iloc[0] if len(control_rows) else hard
            metric = primary(soft.dataset)
            eh, es, em = (float(hard[metric]), float(soft[metric]), float(control[metric]))
            if eh <= 0:
                raise ValueError('Relative primary gain requires positive Hard error')
            record = {k: soft[k] for k in ['case_id', 'dataset', 'policy_name', 'n', 'nout', 'seed', 'rho', 'tau', 'setting', 'reach', 'analytic_q', 'actual_noise_ratio', 'actual_target_rho', 'normalized_sigma']}
            record.update(hard_error=eh, soft_error=es, matched_error=em, hard_result_id=hard.id, soft_result_id=soft.id, matched_noise_result_id=control.id, gain=eh - es, relative_gain=(eh - es) / eh, C=em - eh, N=em - es, G=eh - es, selected_hard=soft.tau == 0, hard_certified=hard.certified)
            for m in metrics:
                if pd.notna(soft[m]) and pd.notna(hard[m]):
                    record['hard__' + m] = float(hard[m])
                    record['soft__' + m] = float(soft[m])
                    record['difference__' + m] = float(soft[m] - hard[m])
            if not math.isclose(record['N'] - record['C'], record['G'], abs_tol=1e-12):
                raise AssertionError('Decomposition identity failed')
            records.append(record)
    return (pd.DataFrame(records), expanded)

def holm(values):
    p = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    adjusted = np.full(len(p), np.nan)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p[index] * (len(order) - rank)))
        adjusted[index] = running
    return adjusted

def summarize(pairs):
    rows = []
    for (cid, setting, rho), g in pairs.groupby(['case_id', 'setting', 'rho']):
        if len(g) != 12 or g.seed.nunique() != 12:
            raise AssertionError('Confirmation cells require twelve paired seeds')
        row = {k: g.iloc[0][k] for k in ['dataset', 'policy_name', 'n', 'nout', 'reach']}
        row.update(case_id=cid, setting=setting, rho=rho, n_pairs=12, selected_hard=int(g.selected_hard.sum()), hard_certified_pairs=int(g.hard_certified.sum()))
        for value in ['relative_gain', 'gain', 'C', 'N', 'G', 'hard_error', 'soft_error']:
            mean, lo, hi = interval(g[value])
            row[value] = mean
            row[value + '_low'] = lo
            row[value + '_high'] = hi
        values = g.relative_gain.to_numpy()
        has_variation = np.ptp(values) > 0
        row['p_raw'] = float(ttest_1samp(values, 0).pvalue) if has_variation else 1.0 if values.mean() == 0 else np.nan
        row['test_status'] = 'Paired t contrast' if has_variation else 'Exact zero differences' if values.mean() == 0 else 'Undefined t statistic: zero sample variance'
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame['p_holm'] = holm(frame.p_raw)
    frame['test_family'] = 'All new prespecified primary confirmation contrasts; no stars in figures'
    return frame

def gain_lines(source, group_key, label_for, ax, x='rho', ylabel='Paired error reduction (%)', value_column='relative_gain'):
    if not np.all(source[x].to_numpy() > 0):
        raise ValueError('Logarithmic curve coordinates must be strictly positive')
    for i, (key, g) in enumerate(source.groupby(group_key, sort=True)):
        rows = []
        for value, h in g.groupby(x):
            rows.append((value, *interval(100 * h[value_column])))
        a = np.asarray(sorted(rows))
        color, marker = (COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)])
        style = LINE_STYLES[i // len(COLORS) % len(LINE_STYLES)]
        ax.plot(a[:, 0], a[:, 1], marker=marker, color=color, linestyle=style, lw=1.5, ms=4, label=label_for(key))
        ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=color, alpha=0.12)
    ax.axhline(0, color='#343D40', lw=0.75)
    ax.set_ylabel(ylabel)

def new_policy_figures(manifest, pairs, datasets=None):
    for d in datasets or ['adult', 'bank', 'xor3']:
        ids = [c['case_id'] for c in manifest['cases'] if c['dataset'] == d and 'policy' in c['families']]
        policy_order = [c['policy_name'] for c in manifest['cases'] if c['case_id'] in ids]
        for setting in {'adult': ['fixed_tau_0p03'], 'bank': ['public_selected']}[d]:
            f = pairs[pairs.case_id.isin(ids) & (pairs.setting == setting)]
            fig, ax = axes((4.9, 4.05), left=0.3, bottom=0.22)
            for j, rho in enumerate(BUDGETS):
                for i, name in enumerate(policy_order):
                    h = f[(f.policy_name == name) & (f.rho == rho)]
                    mean, lo, hi = interval(100 * h.relative_gain)
                    ax.errorbar(mean, i + (j - 1.5) * 0.16, xerr=[[mean - lo], [hi - mean]], fmt=MARKERS[j], color=COLORS[j], markersize=3.8, capsize=2, label=f'rho={rho:g}' if i == 0 else None)
            ax.axvline(0, color='#343D40', lw=0.75)
            ax.set_yticks(range(len(policy_order)), [policy_label(d, p) for p in policy_order])
            ax.invert_yaxis()
            ax.set_xlabel('Paired primary-error reduction (%)')
            ax.grid(axis='y', visible=False)
            legend_below(ax, 2)
            save(fig, f'fig_gen_policy_{d}_{setting}', f, f'{d}, {setting}, training n={f.n.iloc[0]}, modeled columns={schema_for(d).p}. All selected columns remain in the distance and evaluation; only protected graphs vary. 512 outputs, eight rounds, twelve paired confirmation seeds. Means and pointwise 95% t intervals; positive favors Soft. Public selection may choose Hard, yielding an observed zero paired gain. Full-column product is not complete record adjacency.', 'new_policy')

def new_size_figures(manifest, pairs):
    ids = [c['case_id'] for c in manifest['cases'] if 'size' in c['families']]
    settings = ['fixed_tau_0p03']
    for policy in ['x1']:
        for mode in ['fixed_budget']:
            for name in settings:
                setting = ('normalized_noise__' if mode == 'normalized_noise' else '') + name
                f = pairs[pairs.case_id.isin(ids) & (pairs.policy_name == policy) & (pairs.setting == setting)]
                fig, ax = axes()
                gain_lines(f, 'rho', lambda rho: f'rho_ref={rho:g}' if mode == 'normalized_noise' else f'rho={rho:g}', ax, x='n')
                ax.set_xscale('log')
                ax.set_xticks([3000, 12000, 48000, 100000], ['3,000', '12,000', '48,000', '100,000'])
                ax.set_xlabel('Training records n')
                legend_below(ax, 2)
                save(fig, f'fig_gen_xor3_size_{policy}_{name}_{mode}', f, f'XOR3, {policy}, {name}, {mode}. Output m=512, eight rounds, twelve paired independent data/algorithm seeds. Means and pointwise 95% t intervals of joint-error reduction. ' + ('Actual rho(n)=rho_ref*(12000/n)^2; temperatures match the n=12000 reference. Different n have different privacy budgets.' if mode == 'normalized_noise' else 'Actual privacy budget is fixed along each line; n uses nested inputs within each seed.'), 'new_size')
