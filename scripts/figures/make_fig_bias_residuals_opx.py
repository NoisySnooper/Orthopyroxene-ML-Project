#!/usr/bin/env python3
"""Core_06: residual-vs-predicted scatter per opx cell (canonical seed 42).

Four panels (opx_liq T, opx_liq P, opx_only T, opx_only P). Pre-correction
residuals drawn as gray; shipped-form residuals drawn colored by each
sample's pre-registered P regime. When no form ships the panel is
labelled accordingly and only shows pre-correction residuals.

The legend is drawn once for the whole figure (regime colors + pre
vs post) so readers can decode every panel, including those that ship
nothing.

Source: results/bias_correction_shipped.csv (winner per cell),
results/bias_correction/checkpoints/ (per-sample predictions at seed 42),
data/processed/opx_clean_*.parquet (for true y and P), data/splits/
test_indices_*.npy (for the held-out row ids).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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

CKPT_DIR = PROJECT_ROOT / 'results' / 'bias_correction' / 'checkpoints'

REGIME_EDGES = [0.0, 5.0, 15.0, 30.0, 100.0]
REGIME_LABELS = ['shallow_crustal', 'deep_crustal_MASH',
                 'lithospheric_mantle', 'deeper_mantle']
REGIME_SHORT = {
    'shallow_crustal':     'shallow (<5 kbar)',
    'deep_crustal_MASH':   'deep-MASH (5-15 kbar)',
    'lithospheric_mantle': 'litho (15-30 kbar)',
    'deeper_mantle':       'deeper (>=30 kbar)',
}
REGIME_COLORS = {
    'shallow_crustal':     OKABE_ITO['blue'],
    'deep_crustal_MASH':   OKABE_ITO['orange'],
    'lithospheric_mantle': OKABE_ITO['green'],
    'deeper_mantle':       OKABE_ITO['vermillion'],
}

COL_PRE = '#4A4A4A'

TRACK_TO_SPLIT = {'opx_liq': 'opx_liq', 'opx_only': 'opx'}
TRACK_TO_PARQUET = {
    'opx_liq':  'data/processed/opx_clean_opx_liq.parquet',
    'opx_only': 'data/processed/opx_clean_opx_only.parquet',
}
TARGET_UNIT = {'T_C': 'C', 'P_kbar': 'kbar'}
TARGET_LABEL = {'T_C': 'T (C)', 'P_kbar': 'P (kbar)'}

PANELS = [
    ('opx_liq',  'T_C',    '(a) Opx + Liquid  T (C)'),
    ('opx_liq',  'P_kbar', '(b) Opx + Liquid  P (kbar)'),
    ('opx_only', 'T_C',    '(c) Opx only  T (C)'),
    ('opx_only', 'P_kbar', '(d) Opx only  P (kbar)'),
]


def assign_regime(p_kbar: np.ndarray) -> np.ndarray:
    labels = np.full(p_kbar.shape, 'unknown', dtype=object)
    for i, lab in enumerate(REGIME_LABELS):
        lo = REGIME_EDGES[i]; hi = REGIME_EDGES[i + 1]
        if i < len(REGIME_LABELS) - 1:
            mask = (p_kbar >= lo) & (p_kbar < hi)
        else:
            mask = (p_kbar >= lo) & (p_kbar <= hi)
        labels[mask] = lab
    return labels


def ckpt_path(row, seed=42) -> Path:
    stem = (f"{row['pipeline']}_{row['track']}_{row['target']}"
            f"_{row['model']}_{row['feature_set']}_s{seed}.pkl")
    return CKPT_DIR / stem


def draw_panel(ax, ship_row, track: str, target: str, title: str):
    ck = ckpt_path(ship_row, seed=42)
    if not ck.exists():
        ax.text(0.5, 0.5, 'checkpoint missing', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title, loc='left', pad=6)
        return

    obj = joblib.load(ck)
    winner = ship_row['winner']
    split_tag = TRACK_TO_SPLIT[track]
    idx = np.load(f'data/splits/test_indices_{split_tag}.npy')
    df = pd.read_parquet(TRACK_TO_PARQUET[track]).reset_index(drop=True)
    y_true = df.iloc[idx][target].values.astype(float)
    p_true = df.iloc[idx]['P_kbar'].values.astype(float)
    reg_labels = assign_regime(p_true)
    y_pre = np.asarray(obj.y_pred_te, float)
    if winner == 'A':
        y_post = np.asarray(obj.y_corr_a_te, float)
    elif winner == 'B':
        y_post = np.asarray(obj.y_corr_b_te, float)
    else:
        y_post = None

    res_pre = y_pre - y_true
    ax.scatter(y_pre, res_pre, s=18, alpha=0.65, c=COL_PRE,
               edgecolor='none', zorder=1, label='pre-correction')
    if y_post is not None:
        res_post = y_post - y_true
        for reg in REGIME_LABELS:
            mask = reg_labels == reg
            if mask.sum() == 0:
                continue
            ax.scatter(y_post[mask], res_post[mask], s=22, alpha=0.85,
                       color=REGIME_COLORS[reg], edgecolor='black',
                       linewidths=0.3, zorder=3,
                       label=REGIME_SHORT[reg])
        sigma = float(np.std(res_post))
        ax.axhspan(-sigma, sigma, color='#d8d8d8', alpha=0.45, zorder=0)
        ship_txt = f'ships Form {winner}'
        ship_col = OKABE_ITO['blue'] if winner == 'A' else OKABE_ITO['vermillion']
    else:
        ship_txt = 'ships none (pre only)'
        ship_col = '#555555'

    ax.axhline(0, color='k', lw=0.8, zorder=2)
    ax.text(0.98, 0.97, ship_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, fontweight='bold',
            color=ship_col,
            bbox=dict(facecolor='white', edgecolor=ship_col,
                      boxstyle='round,pad=0.3', alpha=0.95))

    # Clip axes to 1-99 percentile of the combined pre+post distributions
    # so a handful of outlier residuals do not collapse the body of the
    # scatter into a thin strip. Outliers remain visible at the frame.
    xs = [y_pre]; ys = [res_pre]
    if y_post is not None:
        xs.append(y_post); ys.append(res_post)
    xs = np.concatenate([np.asarray(a, float) for a in xs])
    ys = np.concatenate([np.asarray(a, float) for a in ys])
    if len(xs) and np.isfinite(xs).any():
        x_lo, x_hi = np.nanpercentile(xs, [1, 99])
        y_lo, y_hi = np.nanpercentile(ys, [1, 99])
        pad_x = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 1.0
        pad_y = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
        ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
        ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

    unit = TARGET_UNIT[target]
    ax.set_xlabel(f'Predicted {TARGET_LABEL[target]}')
    ax.set_ylabel(f'Residual ({unit})')
    subtitle = (f'corrected model: {ship_row["model"]} + '
                f'{ship_row["feature_set"]}')
    ax.set_title(f'{title}\n{subtitle}', loc='left', pad=6, fontsize=11)
    ax.grid(True)
    ax.set_axisbelow(True)


def main():
    ship = pd.read_csv('results/bias_correction_shipped.csv')

    # Prefer v3 verdicts so any v3-promoted form actually renders its
    # post-correction scatter here. Map v3 winner back into the row
    # handed to draw_panel via the `winner` column.
    ship_v3_path = Path('results/bias_correction_shipped_v3.csv')
    if ship_v3_path.exists():
        ship_v3 = pd.read_csv(ship_v3_path)
        v3_map = {(r['pipeline'], r['track'], r['target'], r['model']):
                  r['winner_v3']
                  for _, r in ship_v3.iterrows()}
    else:
        v3_map = {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (track, target, title) in zip(axes.ravel(), PANELS):
        row = ship[(ship.pipeline == 'opx') & (ship.track == track)
                   & (ship.target == target)
                   & (ship.model != 'TabPFN')]
        if row.empty:
            ax.set_visible(False); continue
        ship_row = row.iloc[0].copy()
        key = (ship_row['pipeline'], ship_row['track'],
               ship_row['target'], ship_row['model'])
        if key in v3_map:
            ship_row['winner'] = v3_map[key]
        draw_panel(ax, ship_row, track, target, title)

    # Shared figure-level legend.
    handles = [
        mlines.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=COL_PRE, markersize=8, alpha=0.6,
                      label='pre-correction'),
    ]
    for reg in REGIME_LABELS:
        handles.append(
            mlines.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=REGIME_COLORS[reg],
                          markeredgecolor='black', markeredgewidth=0.4,
                          markersize=8, label=REGIME_SHORT[reg]))
    handles.append(
        mpatches.Patch(facecolor='#d8d8d8', edgecolor='none',
                       label='+/- 1 sigma post-correction'))
    fig.legend(handles=handles, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.02), frameon=True,
               framealpha=0.95, edgecolor='0.6',
               title='Marker colors key (post-correction residuals are '
                     'colored by the sample\'s pre-registered P regime)')

    fig.suptitle(
        'Post-correction residual vs predicted (opx pipelines, seed 42)',
        y=1.00, fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    plt.subplots_adjust(hspace=0.30, wspace=0.22, bottom=0.14)

    stem = OUT_DIR / 'Core_06_fig31_bias_correction_residuals'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 6. Post-correction residual-vs-predicted scatter at '
        'canonical seed 42 for the four opx track/target combinations. '
        'Gray markers: pre-correction residuals. Colored markers: '
        'residuals after the shipped form of bias correction, colored by '
        'the sample\'s pre-registered P regime (shallow_crustal blue, '
        'deep_crustal_MASH orange, lithospheric_mantle green, '
        'deeper_mantle vermillion). A gray band marks +/-1 sigma of the '
        'post-correction residuals. The shipped form per cell is '
        'annotated top-right; cells with "ships none" show only the '
        'pre-correction residuals. Shipping is governed by the '
        'pre-registered ship-if-better rule with the tolerance envelope '
        'max(T_ABS, T_REL x pre_rmse) where T_ABS = 10 degC for T, 1 kbar '
        'for P, and T_REL = 0.10. Source: '
        'results/bias_correction_shipped_v3.csv, '
        'results/bias_correction_shipped.csv (for model/feature_set '
        'identity), and per-sample predictions from '
        'results/bias_correction/checkpoints/ (seed 42).'
    )
    (OUT_DIR / 'Core_06_fig31_bias_correction_residuals.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
