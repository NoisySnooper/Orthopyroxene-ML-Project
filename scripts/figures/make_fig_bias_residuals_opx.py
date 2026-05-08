#!/usr/bin/env python3
"""supp_fig_3: residual-vs-predicted scatter per opx cell (canonical seed 42).

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

from scripts.figures._style import apply_pub_style, make_fig, add_grid, resolve_out_dir # noqa: E402
from scripts.figures._model_palette import REGIME_COLORS, MODEL_COLORS  # noqa: E402
from scripts.figures._labels import (REGIME_LABEL, TARGET_UNIT,           # noqa: E402
                                      TARGET_LABEL, panel_header)
from scripts.figures._legend import add_below_legend                      # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

CKPT_DIR = PROJECT_ROOT / 'results' / 'bias_correction' / 'checkpoints'

REGIME_EDGES = [0.0, 5.0, 15.0, 30.0, 100.0]
REGIME_LABELS = ['shallow_crustal', 'deep_crustal_MASH',
                 'lithospheric_mantle', 'deeper_mantle']

COL_PRE = '#888888'

TRACK_TO_SPLIT = {'opx_liq': 'opx_liq', 'opx_only': 'opx'}
TRACK_TO_PARQUET = {
    'opx_liq':  'data/processed/opx_clean_opx_liq.parquet',
    'opx_only': 'data/processed/opx_clean_opx_only.parquet',
}

PANELS = [
    ('opx_liq',  'T_C',    'a'),
    ('opx_liq',  'P_kbar', 'b'),
    ('opx_only', 'T_C',    'c'),
    ('opx_only', 'P_kbar', 'd'),
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


def draw_panel(ax, ship_row, track: str, target: str, idx: str):
    title = panel_header(track, target, idx)
    ck = ckpt_path(ship_row, seed=42)
    if not ck.exists():
        ax.text(0.5, 0.5, 'checkpoint missing', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
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
    ax.scatter(y_pre, res_pre, s=14, alpha=0.55, c=COL_PRE,
               edgecolor='none', zorder=1, label='pre-correction')
    if y_post is not None:
        res_post = y_post - y_true
        for reg in REGIME_LABELS:
            mask = reg_labels == reg
            if mask.sum() == 0:
                continue
            ax.scatter(y_post[mask], res_post[mask], s=18, alpha=0.85,
                       color=REGIME_COLORS[reg], zorder=3,
                       label=REGIME_LABEL[reg])
        sigma = float(np.std(res_post))
        ax.axhspan(-sigma, sigma, color='#d8d8d8', alpha=0.45, zorder=0)
        ship_txt = f'[Form {winner}]'
    else:
        ship_txt = '[no ship]'

    ax.axhline(0, color='#333333', lw=0.8, zorder=2)
    ax.text(0.98, 0.97, ship_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, fontweight='bold',
            color='#333333')

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
    label = TARGET_LABEL[target]
    ax.set_xlabel(f'Predicted {label} ({unit})')
    ax.set_ylabel(f'Residual ({unit})')
    subtitle = f'{ship_row["model"]}/{ship_row["feature_set"]}'
    ax.set_title(f'{title}\n{subtitle}')
    add_grid(ax, axis='both')


def main():
    ship = pd.read_csv('results/bias_correction_shipped.csv')

    # The shipped CSV holds winner_v3 directly; map that into a 'winner'
    # column so draw_panel can pick the form (A / B / none).
    v3_map = {(r['pipeline'], r['track'], r['target'], r['model']):
              r['winner_v3']
              for _, r in ship.iterrows()}

    fig, axes = make_fig('two_col', nrows=2, ncols=2)
    for ax, (track, target, idx) in zip(axes.ravel(), PANELS):
        row = ship[(ship.pipeline == 'opx') & (ship.track == track)
                   & (ship.target == target)
                   & (ship.model != 'TabPFN')]
        if row.empty:
            ax.set_visible(False); continue
        ship_row = row.iloc[0].copy()
        key = (ship_row['pipeline'], ship_row['track'],
               ship_row['target'], ship_row['model'])
        ship_row['winner'] = v3_map.get(key, 'none')
        draw_panel(ax, ship_row, track, target, idx)

    handles = [
        mlines.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=COL_PRE, markersize=7, alpha=0.6,
                      label='pre-correction'),
    ]
    for reg in REGIME_LABELS:
        handles.append(
            mlines.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=REGIME_COLORS[reg],
                          markersize=7, label=REGIME_LABEL[reg]))
    handles.append(
        mpatches.Patch(facecolor='#d8d8d8', edgecolor='none',
                       label='±1σ post-correction'))
    add_below_legend(fig, handles, [h.get_label() for h in handles], ncol=6)

    fig.suptitle(
        'Post-correction residual vs predicted (opx, seed 42)\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190)'
    )

    stem = OUT_DIR / 'supp_fig_3'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Supp. Figure 3. Post-correction residual-vs-predicted scatter at '
        'canonical seed 42 for the four opx (track, target) cells. Gray '
        'markers: pre-correction residuals. Colored markers: residuals '
        'after the shipped correction form, colored by the sample\'s '
        'pre-registered P regime using the canonical regime palette '
        '(shallow_crustal sky-blue, deep_crustal_MASH green, '
        'lithospheric_mantle orange, deeper_mantle vermillion). A gray '
        'band marks ±1σ of the post-correction residuals. The shipped '
        'form per cell is annotated [Form A], [Form B], or [no ship]. '
        'Source: results/bias_correction_shipped(_v3).csv and per-sample '
        'predictions from results/bias_correction/checkpoints/ at seed 42.'
    )
    (OUT_DIR / 'supp_fig_3.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
