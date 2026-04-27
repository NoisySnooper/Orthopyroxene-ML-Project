#!/usr/bin/env python3
"""Core_05: bias correction effect on per-regime RMSE (opx pipelines only).

Four panels (opx_liq T, opx_liq P, opx_only T, opx_only P) each show, per
pre-registered P regime, a triad of bars: pre-correction (gray), post
Form A (blue), post Form B (vermillion). Bars are 20-seed means with
the 20-seed mean of per-seed bootstrap 95% CIs as whiskers. The shipped
form is annotated in the top-right corner of each panel.

Purpose: visualize whether Form A (regime-piecewise OLS) or Form B
(Agreda-Lopez piecewise sigmoid) actually reduce per-regime error and
whether they do so without degrading any regime (the ship criterion).

Source: results/bias_correction_per_seed.csv (for pre/post mean + CI),
results/bias_correction_shipped.csv (for ship decision).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._style import apply_pub_style  # noqa: E402
from scripts.figures._model_palette import OKABE_ITO  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_PRE = OKABE_ITO['black']          # use gray for pre
COL_PRE = '#888888'
COL_A   = OKABE_ITO['blue']
COL_B   = OKABE_ITO['vermillion']

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_SHORT = {
    'shallow_crustal':     'shallow',
    'deep_crustal_MASH':   'deep-MASH',
    'lithospheric_mantle': 'litho',
    'deeper_mantle':       'deeper',
    'ALL':                 'ALL',
}

PANELS = [
    ('opx_liq',  'T_C',    '(a) Opx + Liquid  T (C)',    'C'),
    ('opx_liq',  'P_kbar', '(b) Opx + Liquid  P (kbar)', 'kbar'),
    ('opx_only', 'T_C',    '(c) Opx only  T (C)',        'C'),
    ('opx_only', 'P_kbar', '(d) Opx only  P (kbar)',     'kbar'),
]

P_REGIME_MIN_N_FOR_CLAIMS = 20


def _panel(ax, pseed, ship_final, track, target, unit, title):
    x = np.arange(len(REGIME_ORDER))
    w = 0.27
    win_row = ship_final[(ship_final.pipeline == 'opx')
                         & (ship_final.track == track)
                         & (ship_final.target == target)
                         & (ship_final.model != 'TabPFN')]
    winner = win_row.iloc[0]['winner_final'] if not win_row.empty else 'none'
    model_tag = win_row['model'].iloc[0] if not win_row.empty else ''
    fs_tag = win_row['feature_set'].iloc[0] if not win_row.empty else ''

    pre_m, pre_lo, pre_hi = [], [], []
    a_m, a_lo, a_hi = [], [], []
    b_m, b_lo, b_hi = [], [], []
    ns = []
    for reg in REGIME_ORDER:
        pa = pseed[(pseed.track == track) & (pseed.target == target)
                   & (pseed.regime == reg) & (pseed.form == 'A')]
        pb = pseed[(pseed.track == track) & (pseed.target == target)
                   & (pseed.regime == reg) & (pseed.form == 'B')]
        if pa.empty or pb.empty:
            for lst in (pre_m, pre_lo, pre_hi, a_m, a_lo, a_hi,
                        b_m, b_lo, b_hi):
                lst.append(np.nan)
            ns.append(0); continue
        pre_m.append(pa['pre_rmse'].mean())
        pre_lo.append(pa['pre_lo'].mean())
        pre_hi.append(pa['pre_hi'].mean())
        a_m.append(pa['post_rmse'].mean())
        a_lo.append(pa['post_lo'].mean())
        a_hi.append(pa['post_hi'].mean())
        b_m.append(pb['post_rmse'].mean())
        b_lo.append(pb['post_lo'].mean())
        b_hi.append(pb['post_hi'].mean())
        ns.append(int(pa['n'].iloc[0]))

    pre_m = np.asarray(pre_m, float); pre_lo = np.asarray(pre_lo, float); pre_hi = np.asarray(pre_hi, float)
    a_m = np.asarray(a_m, float); a_lo = np.asarray(a_lo, float); a_hi = np.asarray(a_hi, float)
    b_m = np.asarray(b_m, float); b_lo = np.asarray(b_lo, float); b_hi = np.asarray(b_hi, float)

    hatches = ['///' if n < P_REGIME_MIN_N_FOR_CLAIMS and reg != 'ALL' else ''
               for n, reg in zip(ns, REGIME_ORDER)]

    bars_pre = ax.bar(x - w, pre_m, width=w, color=COL_PRE,
                      edgecolor='k', lw=0.5, label='pre-correction')
    bars_a = ax.bar(x, a_m, width=w, color=COL_A,
                    edgecolor='k', lw=0.5, label='post Form A')
    bars_b = ax.bar(x + w, b_m, width=w, color=COL_B,
                    edgecolor='k', lw=0.5, label='post Form B')
    for bars in (bars_pre, bars_a, bars_b):
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
    ax.vlines(x - w, pre_lo, pre_hi, color='k', lw=0.8)
    ax.vlines(x, a_lo, a_hi, color='k', lw=0.8)
    ax.vlines(x + w, b_lo, b_hi, color='k', lw=0.8)

    top = np.nanmax(np.concatenate([pre_hi, a_hi, b_hi]))
    if np.isfinite(top) and top > 0:
        ax.set_ylim(0, top * 1.14)

    ship_txt_col = {'A': COL_A, 'B': COL_B}.get(winner, '#555555')
    ship_label = {'A': 'A', 'B': 'B'}.get(winner, 'none')
    ship_txt = (f'ships Form {ship_label}' if ship_label != 'none'
                else 'ships none')
    ax.text(0.98, 0.97, ship_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=10, fontweight='bold',
            color=ship_txt_col, linespacing=1.15,
            bbox=dict(facecolor='white', edgecolor=ship_txt_col,
                      boxstyle='round,pad=0.3', alpha=0.95))

    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_SHORT[r] for r in REGIME_ORDER],
                       rotation=20, ha='right')
    ax.set_ylabel(f'RMSE ({unit})')
    subtitle = f'corrected model: {model_tag} + {fs_tag}' if model_tag else ''
    ax.set_title(f'{title}\n{subtitle}', loc='left', pad=6, fontsize=11)
    ax.grid(axis='y')
    ax.set_axisbelow(True)


def main():
    pseed = pd.read_csv('results/bias_correction_per_seed.csv')
    ship_v3_path = Path('results/bias_correction_shipped_v3.csv')
    ship_v2_path = Path('results/bias_correction_shipped_v2.csv')
    ship_v1_path = Path('results/bias_correction_shipped.csv')
    if ship_v3_path.exists():
        ship_final = pd.read_csv(ship_v3_path).rename(
            columns={'winner_v3': 'winner_final'})
    elif ship_v2_path.exists():
        ship_final = pd.read_csv(ship_v2_path).rename(
            columns={'winner_v2': 'winner_final'})
    else:
        ship_final = pd.read_csv(ship_v1_path).rename(
            columns={'winner': 'winner_final'})

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (track, target, title, unit) in zip(axes.ravel(), PANELS):
        _panel(ax, pseed, ship_final, track, target, unit, title)

    handles = [
        mpatches.Patch(color=COL_PRE, label='pre-correction'),
        mpatches.Patch(color=COL_A, label='post Form A (regime-piecewise OLS)'),
        mpatches.Patch(color=COL_B, label='post Form B (Agreda-Lopez sigmoid)'),
        mpatches.Patch(facecolor='white', edgecolor='k', hatch='///',
                       label=f'n < {P_REGIME_MIN_N_FOR_CLAIMS} (sample-limited)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=True,
               framealpha=0.95, edgecolor='0.6')

    fig.suptitle(
        'Bias-correction effect on per-regime RMSE (opx pipelines)',
        y=0.995, fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.94))
    plt.subplots_adjust(hspace=0.40, wspace=0.22, bottom=0.10, top=0.92)

    stem = OUT_DIR / 'Core_05_fig30_bias_correction_per_regime_rmse'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 5. Bias-correction effect on per-regime test-set RMSE for '
        'the four opx track/target combinations. Dataset: ExPetDB 2025-07-21 '
        'opx held-out test partition (citation-grouped 80/20 split, '
        'pre-registered P-regime edges); the ArcPL external holdout is NOT '
        'used in this figure. Each regime group shows three bars: '
        'pre-correction (gray), post Form A (regime-piecewise OLS, blue), '
        'post Form B (Agreda-Lopez piecewise sigmoid, vermillion). Bar '
        'heights are 20-seed means; whiskers are the 20-seed mean of '
        'per-seed bootstrap 95% CIs. Hatched bars mark regimes with n < 20 '
        '(sample-size limited, interpret with care). The shipped correction '
        'form per panel (ships A / ships B / ships none) is annotated in '
        'the top-right. A form ships only if it reduces overall RMSE and '
        'does not degrade any regime beyond the pre-registered tolerance '
        'envelope max(T_ABS, T_REL x pre_rmse), with T_ABS = 10 degC for T '
        'and 1 kbar for P and T_REL = 0.10. How the math works: both forms '
        'post-process the raw ML prediction y_hat using the out-of-fold '
        'residual e = y_hat - y learned on the training partition, then '
        'subtract the learned residual at test time (y_corr = y_hat - '
        'e_hat). Form A fits a separate linear regression e = a + b * y_hat '
        'on each pre-registered P-regime (four regressions stitched '
        'end-to-end at regime boundaries). Form B fits one smooth '
        'piecewise-sigmoid curve e(y_hat) across all regimes with four '
        'learned breakpoints, following Agreda-Lopez et al. (2024). Cpx '
        'pipelines are intentionally excluded; this paper centers the opx '
        'ML thermobarometer. Source CSVs: '
        'results/bias_correction_per_seed.csv, '
        'results/bias_correction_shipped_v3.csv.'
    )
    (OUT_DIR / 'Core_05_fig30_bias_correction_per_regime_rmse.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
