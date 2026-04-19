#!/usr/bin/env python3
"""Render figures 30-34 for the Phase G.7 bias-correction mini-project.

All figures read existing Phase G.7 CSVs. Per-sample residuals in Fig 31
are read from the checkpoints cached in
``results/v10_bias_correction/checkpoints/`` (canonical seed = 42);
no models are refit.

Outputs (figures/fig{30..34}_bias_correction_*.{png,pdf,txt}):

  Fig 30 -- 2x4 grid, per-regime RMSE (pre, Form A, Form B) with mean
            bootstrap CI envelope across 20 seeds. Hatch when n < 20.
  Fig 31 -- 2x4 grid, residual-vs-predicted scatter. Pre-correction gray;
            shipped form colored by P regime. Canonical seed only.
  Fig 32 -- 2x4 grid, test-set RMSE comparison (pre, A, B) at regime=ALL.
            Shipped bar outlined. Panel subtitle shows vote split.
  Fig 33 -- 2x4 grid, per-seed Form-A vs Form-B ALL-regime delta-RMSE
            scatter with improve/degrade quadrant shading and vote split.
  Fig 34 -- 2-panel heatmap (opx, cpx pipelines). Rows = regimes, cols =
            track x target. Color = fractional improvement of v10_post
            over the best external baseline; cell label = absolute delta.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from config import (FIGURES, LOGS, RESULTS, P_REGIME_BIN_EDGES_KBAR,
                    P_REGIME_LABELS, P_REGIME_MIN_N_FOR_CLAIMS)
from src.io_utils import save_figure
from src.plot_style import apply_style, OKABE_ITO

SHIPPED_P = RESULTS / 'v10_bias_correction_shipped.csv'
SUMMARY_P = RESULTS / 'v10_bias_correction_summary.csv'
PER_SEED_P = RESULTS / 'v10_bias_correction_per_seed.csv'
SCORECARD_P = RESULTS / 'v10_preregistered_scorecard_postcorrection.csv'
CKPT_DIR = RESULTS / 'v10_bias_correction' / 'checkpoints'
LOG_PATH = LOGS / 'v10_phase_g_figures_30_34.log'

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_SHORT = {
    'shallow_crustal': 'shallow',
    'deep_crustal_MASH': 'deep-MASH',
    'lithospheric_mantle': 'litho',
    'deeper_mantle': 'deeper',
    'ALL': 'ALL',
}
REGIME_COLORS = {
    'shallow_crustal': OKABE_ITO['blue'],
    'deep_crustal_MASH': OKABE_ITO['orange'],
    'lithospheric_mantle': OKABE_ITO['green'],
    'deeper_mantle': OKABE_ITO['vermillion'],
    'ALL': OKABE_ITO['gray'],
}

COL_PRE = OKABE_ITO['gray']
COL_A = OKABE_ITO['blue']
COL_B = OKABE_ITO['vermillion']

TRACKS_ORDER = ['opx_liq', 'opx_only', 'cpx_liq', 'cpx_only']
TARGETS_ORDER = ['T_C', 'P_kbar']
TRACK_LABEL = {'opx_liq': 'opx-liq', 'opx_only': 'opx-only',
               'cpx_liq': 'cpx-liq', 'cpx_only': 'cpx-only'}
TARGET_UNIT = {'T_C': 'C', 'P_kbar': 'kbar'}
TARGET_LABEL = {'T_C': 'T [C]', 'P_kbar': 'P [kbar]'}

TRACK_TO_SPLIT = {'opx_liq': 'opx_liq', 'opx_only': 'opx',
                  'cpx_liq': 'cpx_liq', 'cpx_only': 'cpx_only'}
TRACK_TO_PROCESSED = {
    'opx_liq':  Path('data/processed/opx_clean_opx_liq.parquet'),
    'opx_only': Path('data/processed/opx_clean_opx_only.parquet'),
    'cpx_liq':  Path('data/processed/cpx_clean_cpx_liq.parquet'),
    'cpx_only': Path('data/processed/cpx_clean_cpx_only.parquet'),
}


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _ckpt_path(row, seed=42):
    stem = (f"{row['pipeline']}_{row['track']}_{row['target']}"
            f"_{row['model']}_{row['feature_set']}_s{seed}.pkl")
    return CKPT_DIR / stem


def _assign_regime(p_kbar: np.ndarray) -> np.ndarray:
    edges = P_REGIME_BIN_EDGES_KBAR
    labels = np.full(p_kbar.shape, 'unknown', dtype=object)
    for i, lab in enumerate(P_REGIME_LABELS):
        lo, hi = edges[i], edges[i + 1]
        if i < len(P_REGIME_LABELS) - 1:
            mask = (p_kbar >= lo) & (p_kbar < hi)
        else:
            mask = (p_kbar >= lo) & (p_kbar <= hi)
        labels[mask] = lab
    return labels


def fig30_per_regime_rmse(ship, summ, pseed, fh):
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    regimes = REGIME_ORDER
    x = np.arange(len(regimes))
    w = 0.27
    for ri, target in enumerate(TARGETS_ORDER):
        for ci, track in enumerate(TRACKS_ORDER):
            ax = axes[ri, ci]
            win_row = ship[(ship.track == track) & (ship.target == target)]
            winner = win_row['winner'].iloc[0] if not win_row.empty else 'none'
            pre_means, pre_los, pre_his = [], [], []
            a_means, a_los, a_his = [], [], []
            b_means, b_los, b_his = [], [], []
            ns = []
            for reg in regimes:
                pa = pseed[(pseed.track == track) & (pseed.target == target)
                           & (pseed.regime == reg) & (pseed.form == 'A')]
                pb = pseed[(pseed.track == track) & (pseed.target == target)
                           & (pseed.regime == reg) & (pseed.form == 'B')]
                if pa.empty or pb.empty:
                    for lst in (pre_means, pre_los, pre_his, a_means, a_los,
                                a_his, b_means, b_los, b_his):
                        lst.append(np.nan)
                    ns.append(0)
                    continue
                pre_means.append(pa['pre_rmse'].mean())
                pre_los.append(pa['pre_lo'].mean())
                pre_his.append(pa['pre_hi'].mean())
                a_means.append(pa['post_rmse'].mean())
                a_los.append(pa['post_lo'].mean())
                a_his.append(pa['post_hi'].mean())
                b_means.append(pb['post_rmse'].mean())
                b_los.append(pb['post_lo'].mean())
                b_his.append(pb['post_hi'].mean())
                ns.append(int(pa['n'].iloc[0]))
            pre_means = np.asarray(pre_means, float)
            pre_los = np.asarray(pre_los, float)
            pre_his = np.asarray(pre_his, float)
            a_means = np.asarray(a_means, float)
            a_los = np.asarray(a_los, float)
            a_his = np.asarray(a_his, float)
            b_means = np.asarray(b_means, float)
            b_los = np.asarray(b_los, float)
            b_his = np.asarray(b_his, float)
            hatches = ['///' if n < P_REGIME_MIN_N_FOR_CLAIMS and reg != 'ALL'
                       else ''
                       for n, reg in zip(ns, regimes)]
            bars_pre = ax.bar(x - w, pre_means, width=w, color=COL_PRE,
                              edgecolor='k', lw=0.4, label='pre')
            bars_a = ax.bar(x, a_means, width=w, color=COL_A,
                            edgecolor='k', lw=0.4, label='post A')
            bars_b = ax.bar(x + w, b_means, width=w, color=COL_B,
                            edgecolor='k', lw=0.4, label='post B')
            for bars in (bars_pre, bars_a, bars_b):
                for bar, h in zip(bars, hatches):
                    if h:
                        bar.set_hatch(h)
            ax.vlines(x - w, pre_los, pre_his, color='k', lw=0.7)
            ax.vlines(x, a_los, a_his, color='k', lw=0.7)
            ax.vlines(x + w, b_los, b_his, color='k', lw=0.7)
            if winner in ('A', 'B'):
                top = np.nanmax([pre_his, a_his, b_his])
                ax.text(x[-1], top * 1.05, f'ships {winner}',
                        ha='center', va='bottom', fontsize=8, color='k',
                        bbox=dict(facecolor='lemonchiffon', edgecolor='k',
                                  pad=1.5, lw=0.5))
            ax.set_xticks(x)
            ax.set_xticklabels([REGIME_SHORT[r] for r in regimes],
                               rotation=30, ha='right', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'RMSE [{TARGET_UNIT[target]}]')
            ax.set_title(f'{TRACK_LABEL[track]} -- {target}', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
    handles = [
        mpatches.Patch(color=COL_PRE, label='pre-correction'),
        mpatches.Patch(color=COL_A, label='post Form A'),
        mpatches.Patch(color=COL_B, label='post Form B'),
        mpatches.Patch(facecolor='white', edgecolor='k', hatch='///',
                       label=f'n < {P_REGIME_MIN_N_FOR_CLAIMS}'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle('Figure 30. Per-regime test-set RMSE (20-seed mean with '
                 'bootstrap CI envelope)', y=1.06, fontsize=11)
    fig.tight_layout()
    caption = (
        'Fig. 30. Per-regime test-set RMSE before and after bias correction '
        'for the 8 Phase-G.7 cells. Grid rows: T_C (top), P_kbar (bottom). '
        'Columns: opx-liq, opx-only, cpx-liq, cpx-only. Each regime group '
        'shows pre-correction (gray), post Form A (blue), post Form B '
        '(vermillion). Bar heights are 20-seed means; vertical error bars '
        'are the mean of per-seed bootstrap 95% CIs. Hatched bars mark '
        f'regimes with n < {P_REGIME_MIN_N_FOR_CLAIMS} (sample-size '
        'limited). "ships A/B" annotation marks the Phase-G.7 shipping '
        'decision at canonical seed 42. Source CSVs: '
        'results/v10_bias_correction_{per_seed,summary,shipped}.csv.'
    )
    save_figure(fig, 'fig30_bias_correction_per_regime_rmse',
                dir=FIGURES, caption=caption)
    plt.close(fig)
    _log('wrote fig30_bias_correction_per_regime_rmse', fh)


def fig31_residuals(ship, pseed, fh):
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ri, target in enumerate(TARGETS_ORDER):
        for ci, track in enumerate(TRACKS_ORDER):
            ax = axes[ri, ci]
            row = ship[(ship.track == track) & (ship.target == target)]
            if row.empty:
                ax.set_visible(False); continue
            row = row.iloc[0]
            winner = row['winner']
            ck_path = _ckpt_path(row, seed=42)
            if not ck_path.exists():
                ax.text(0.5, 0.5, 'checkpoint missing', transform=ax.transAxes,
                        ha='center', va='center')
                continue
            obj = joblib.load(ck_path)
            split = TRACK_TO_SPLIT[track]
            idx = np.load(f'data/splits/test_indices_{split}.npy')
            df = pd.read_parquet(TRACK_TO_PROCESSED[track]).reset_index(drop=True)
            y_true = df.iloc[idx][target].values.astype(float)
            p_true = df.iloc[idx]['P_kbar'].values.astype(float)
            reg_labels = _assign_regime(p_true)
            y_pre = np.asarray(obj.y_pred_te, float)
            if winner == 'A':
                y_post = np.asarray(obj.y_corr_a_te, float)
            elif winner == 'B':
                y_post = np.asarray(obj.y_corr_b_te, float)
            else:
                y_post = None
            res_pre = y_pre - y_true
            ax.scatter(y_pre, res_pre, s=14, alpha=0.35, c=COL_PRE,
                       edgecolor='none', label='pre')
            if y_post is not None:
                res_post = y_post - y_true
                for reg in REGIME_ORDER[:-1]:
                    mask = reg_labels == reg
                    if mask.sum() == 0:
                        continue
                    ax.scatter(y_post[mask], res_post[mask], s=20, alpha=0.8,
                               color=REGIME_COLORS[reg],
                               edgecolor='k', lw=0.3,
                               label=REGIME_SHORT[reg])
                sigma = float(np.std(res_post))
                ax.axhspan(-sigma, sigma, color='lightgray', alpha=0.35,
                           zorder=0)
            else:
                ax.text(0.05, 0.93, 'no correction ships',
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='k',
                                  lw=0.5, pad=2))
            ax.axhline(0, color='k', lw=0.7)
            ax.set_xlabel(f'Predicted {TARGET_LABEL[target]}')
            if ci == 0:
                ax.set_ylabel(f'Residual [{TARGET_UNIT[target]}]')
            ax.set_title(f'{TRACK_LABEL[track]} -- {target}  '
                         f'(ships {winner})', fontsize=10)
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc='lower right', fontsize=7, ncol=2)
    fig.suptitle('Figure 31. Residual vs predicted at canonical seed 42',
                 y=1.02, fontsize=11)
    fig.tight_layout()
    caption = (
        'Fig. 31. Residual-vs-predicted scatter for the 8 Phase-G.7 cells '
        'at canonical seed 42. Gray markers: pre-correction residuals. '
        'Colored markers: residuals after the shipping form of correction, '
        'grouped by each sample pre-registered P regime (shallow_crustal '
        'blue, deep_crustal_MASH orange, lithospheric_mantle green, '
        'deeper_mantle vermillion). A shaded band marks +/-1 sigma of the '
        'post-correction residuals. Cells whose shipping decision is "none" '
        'show the pre-correction residuals alone and are labelled '
        'accordingly. Per-sample predictions loaded from '
        'results/v10_bias_correction/checkpoints/ (seed 42). Source CSV: '
        'results/v10_bias_correction_shipped.csv.'
    )
    save_figure(fig, 'fig31_bias_correction_residuals',
                dir=FIGURES, caption=caption)
    plt.close(fig)
    _log('wrote fig31_bias_correction_residuals', fh)


def fig32_form_comparison(ship, summ, pseed, fh):
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    labels = ['pre', 'Form A', 'Form B']
    positions = np.array([0, 1, 2])
    colors = [COL_PRE, COL_A, COL_B]
    for ri, target in enumerate(TARGETS_ORDER):
        for ci, track in enumerate(TRACKS_ORDER):
            ax = axes[ri, ci]
            srow = ship[(ship.track == track) & (ship.target == target)]
            if srow.empty:
                ax.set_visible(False); continue
            winner = srow['winner'].iloc[0]
            sa = summ[(summ.track == track) & (summ.target == target)
                      & (summ.form == 'A') & (summ.regime == 'ALL')]
            sb = summ[(summ.track == track) & (summ.target == target)
                      & (summ.form == 'B') & (summ.regime == 'ALL')]
            if sa.empty or sb.empty:
                ax.set_visible(False); continue
            pre = float(sa['pre_rmse_mean'].iloc[0])
            post_a = float(sa['post_rmse_mean'].iloc[0])
            post_b = float(sb['post_rmse_mean'].iloc[0])
            heights = [pre, post_a, post_b]
            bars = ax.bar(positions, heights, color=colors, edgecolor='k',
                          lw=0.5, width=0.6)
            if winner == 'A':
                bars[1].set_edgecolor('black')
                bars[1].set_linewidth(2.5)
            elif winner == 'B':
                bars[2].set_edgecolor('black')
                bars[2].set_linewidth(2.5)
            ps = pseed[(pseed.track == track) & (pseed.target == target)
                       & (pseed.regime == 'ALL')]
            ps_all = ps.drop_duplicates(['seed'])
            n_tot = len(ps_all)
            n_a = int((ps_all.winner == 'A').sum())
            n_b = int((ps_all.winner == 'B').sum())
            n_none = int((~ps_all.winner.isin(['A', 'B'])).sum())
            if winner == 'A':
                subtitle = f'ships A ({n_a}/{n_tot}); B {n_b}, none {n_none}'
            elif winner == 'B':
                subtitle = f'ships B ({n_b}/{n_tot}); A {n_a}, none {n_none}'
            else:
                subtitle = f'ships none (A {n_a}, B {n_b}, none {n_none})'
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f'Test RMSE [{TARGET_UNIT[target]}]')
            ax.set_title(f'{TRACK_LABEL[track]} -- {target}\n{subtitle}',
                         fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)
            for b, h in zip(bars, heights):
                ax.text(b.get_x() + b.get_width() / 2, h, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=8)
    fig.suptitle('Figure 32. Pre- vs Form-A vs Form-B test RMSE at '
                 'regime=ALL (20-seed mean)', y=1.02, fontsize=11)
    fig.tight_layout()
    caption = (
        'Fig. 32. Test-set RMSE at regime=ALL comparing pre-correction with '
        'Form A and Form B post-correction for the 8 Phase-G.7 cells '
        '(20-seed mean). Shipped form (if any) is drawn with a bold '
        'border. Panel subtitle reports the Phase-G.7 shipping decision '
        'and the per-seed vote split across 20 model seeds. Source CSVs: '
        'results/v10_bias_correction_{summary,per_seed,shipped}.csv.'
    )
    save_figure(fig, 'fig32_bias_correction_form_comparison',
                dir=FIGURES, caption=caption)
    plt.close(fig)
    _log('wrote fig32_bias_correction_form_comparison', fh)


def fig33_per_seed_stability(pseed, fh):
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ri, target in enumerate(TARGETS_ORDER):
        for ci, track in enumerate(TRACKS_ORDER):
            ax = axes[ri, ci]
            ps = pseed[(pseed.track == track) & (pseed.target == target)
                       & (pseed.regime == 'ALL')]
            if ps.empty:
                ax.set_visible(False); continue
            a = ps[ps.form == 'A'].sort_values('seed')
            b = ps[ps.form == 'B'].sort_values('seed')
            merged = a[['seed', 'delta_rmse', 'winner']].merge(
                b[['seed', 'delta_rmse']], on='seed', suffixes=('_A', '_B'))
            xs = merged['delta_rmse_A'].values
            ys = merged['delta_rmse_B'].values
            xmin = min(float(xs.min()), 0) - max(0.05 * abs(xs.min()), 0.5)
            xmax = max(float(xs.max()), 0) + max(0.05 * abs(xs.max()), 0.5)
            ymin = min(float(ys.min()), 0) - max(0.05 * abs(ys.min()), 0.5)
            ymax = max(float(ys.max()), 0) + max(0.05 * abs(ys.max()), 0.5)
            ax.axhspan(0, ymax, xmin=0.5, xmax=1.0, color='#DFF2D8',
                       alpha=0.45, zorder=0)
            ax.axhspan(ymin, 0, xmin=0, xmax=0.5, color='#FADBDB',
                       alpha=0.45, zorder=0)
            ax.axhline(0, color='k', lw=0.6)
            ax.axvline(0, color='k', lw=0.6)
            colors = []
            for w in merged['winner']:
                if w == 'A':
                    colors.append(COL_A)
                elif w == 'B':
                    colors.append(COL_B)
                else:
                    colors.append(COL_PRE)
            ax.scatter(xs, ys, c=colors, s=36, edgecolor='k', lw=0.4,
                       zorder=3)
            n_a = int((merged['winner'] == 'A').sum())
            n_b = int((merged['winner'] == 'B').sum())
            n_none = int((~merged['winner'].isin(['A', 'B'])).sum())
            ax.text(0.03, 0.97,
                    f'A: {n_a}/20  B: {n_b}/20  none: {n_none}/20',
                    transform=ax.transAxes, ha='left', va='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='k', lw=0.4,
                              pad=2))
            ax.set_xlabel(f'Form A $\\Delta$RMSE [{TARGET_UNIT[target]}]')
            if ci == 0:
                ax.set_ylabel(f'Form B $\\Delta$RMSE [{TARGET_UNIT[target]}]')
            ax.set_title(f'{TRACK_LABEL[track]} -- {target}', fontsize=10)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)
    handles = [
        mpatches.Patch(color=COL_A, label='winner A'),
        mpatches.Patch(color=COL_B, label='winner B'),
        mpatches.Patch(color=COL_PRE, label='winner none'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle('Figure 33. Per-seed $\\Delta$RMSE, Form A vs Form B '
                 '(regime=ALL)', y=1.06, fontsize=11)
    fig.tight_layout()
    caption = (
        'Fig. 33. Per-seed $\\Delta$RMSE (pre minus post, positive = '
        'improvement) at regime=ALL for Form A (x-axis) vs Form B (y-axis), '
        'one point per model seed across the 20-seed protocol. Points are '
        'colored by the per-seed winner (blue = A, vermillion = B, gray = '
        'none). Light-green shading highlights the upper-right quadrant '
        'where both forms improve test RMSE; light-red shading highlights '
        'the lower-left quadrant where both forms degrade it. Panel '
        'annotation reports the per-seed vote count. Source CSV: '
        'results/v10_bias_correction_per_seed.csv.'
    )
    save_figure(fig, 'fig33_bias_correction_per_seed_stability',
                dir=FIGURES, caption=caption)
    plt.close(fig)
    _log('wrote fig33_bias_correction_per_seed_stability', fh)


def fig34_scorecard_delta(sc, fh):
    apply_style()
    pipelines = [('opx', ['opx_only', 'opx_liq']),
                 ('cpx', ['cpx_only', 'cpx_liq'])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap_div = LinearSegmentedColormap.from_list(
        'v10div',
        [(0.0, '#B22222'), (0.5, '#FFFFFF'), (1.0, '#228B22')])
    for pi, (pipeline, tracks) in enumerate(pipelines):
        ax = axes[pi]
        cols = []
        for track in tracks:
            for target in TARGETS_ORDER:
                cols.append((track, target))
        rows = REGIME_ORDER
        arr = np.full((len(rows), len(cols)), np.nan)
        labels = np.full((len(rows), len(cols)), '', dtype=object)
        for i, reg in enumerate(rows):
            for j, (track, target) in enumerate(cols):
                r = sc[(sc.track == track) & (sc.target == target)
                       & (sc.regime == reg)]
                if r.empty:
                    labels[i, j] = 'N/A'
                    continue
                r = r.iloc[0]
                ext = r['best_external_rmse']
                post = r['v10_post_rmse']
                if not np.isfinite(ext):
                    unit = 'C' if target == 'T_C' else 'kbar'
                    labels[i, j] = f'no ext\npost {post:.2f} {unit}'
                    continue
                pct = (ext - post) / ext
                arr[i, j] = pct
                delta_abs = ext - post
                unit = 'C' if target == 'T_C' else 'kbar'
                labels[i, j] = f'{delta_abs:+.2f} {unit}'
        vmax = np.nanmax(np.abs(arr)) if np.isfinite(
            np.nanmax(np.abs(arr))) else 0.5
        vmax = max(vmax, 0.05)
        im = ax.imshow(arr, cmap=cmap_div, vmin=-vmax, vmax=vmax,
                       aspect='auto')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f'{TRACK_LABEL[t]}\n{tg}' for t, tg in cols],
                           fontsize=9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([REGIME_SHORT[r] for r in rows], fontsize=9)
        for i in range(len(rows)):
            for j in range(len(cols)):
                lbl = labels[i, j]
                if lbl == '':
                    continue
                if np.isfinite(arr[i, j]) and abs(arr[i, j]) > 0.5 * vmax:
                    color = 'white'
                else:
                    color = 'black'
                ax.text(j, i, lbl, ha='center', va='center', fontsize=8,
                        color=color)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('(best_external - v10_post) / best_external',
                       fontsize=9)
        ax.set_title(f'{pipeline} pipeline (green: v10 wins, red: ext wins)',
                     fontsize=10)
    fig.suptitle('Figure 34. Scorecard delta: v10 post-correction '
                 'vs best external per regime', y=1.02, fontsize=11)
    fig.tight_layout()
    caption = (
        'Fig. 34. Scorecard delta between v10 post-correction RMSE and the '
        'best external baseline per pre-registered regime, split by '
        'pipeline (opx left, cpx right). Color encodes fractional '
        'improvement (best_external - v10_post)/best_external on a '
        'diverging scale: green = v10 wins, red = external wins, white = '
        'tie. Cells annotated with the absolute RMSE delta in native '
        'units. Cells without an external baseline list the absolute '
        'v10_post value. External reference methods include Putirka (2008) '
        'thermobarometers, Agreda-Lopez (2024) ML cpx models, and '
        'Jorgenson (2022) ML cpx models. Source CSV: '
        'results/v10_preregistered_scorecard_postcorrection.csv.'
    )
    save_figure(fig, 'fig34_bias_correction_scorecard_delta',
                dir=FIGURES, caption=caption)
    plt.close(fig)
    _log('wrote fig34_bias_correction_scorecard_delta', fh)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        for p in (SHIPPED_P, SUMMARY_P, PER_SEED_P, SCORECARD_P):
            if not p.exists():
                _log(f'ERROR: missing input {p}', fh)
                return 1
        ship = pd.read_csv(SHIPPED_P)
        summ = pd.read_csv(SUMMARY_P)
        pseed = pd.read_csv(PER_SEED_P)
        sc = pd.read_csv(SCORECARD_P)
        fig30_per_regime_rmse(ship, summ, pseed, fh)
        fig31_residuals(ship, pseed, fh)
        fig32_form_comparison(ship, summ, pseed, fh)
        fig33_per_seed_stability(pseed, fh)
        fig34_scorecard_delta(sc, fh)
        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
