#!/usr/bin/env python3
"""main_fig_6: bias-correction effect on per-regime RMSE.

Four panels (opx_liq T, opx_liq P, opx_only T, opx_only P). For each
pre-registered P regime (plus ALL): a triad of bars — pre-correction
(family color, faint+hatched), post Form A (winner family color, solid),
post Form B (Putirka gray, solid for visual distinction from family hue).
Bars are 20-seed means; whiskers are the 20-seed mean of per-seed
bootstrap 95% CIs. The shipped form is named in each panel's corner.

Source: results/bias_correction_per_seed.csv,
        results/bias_correction_shipped.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._style import apply_pub_style, make_fig, add_grid, resolve_out_dir # noqa: E402
from scripts.figures._model_palette import (MODEL_COLORS, role_style,     # noqa: E402
                                             family_from_method)
from scripts.figures._labels import (REGIME_TICK, TARGET_UNIT,            # noqa: E402
                                      panel_header)
from scripts.figures._legend import role_patch, add_below_legend          # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_TICKS = [REGIME_TICK.get(r, r) for r in REGIME_ORDER]

PANELS = [
    ('opx_liq',  'T_C',    'a'),
    ('opx_liq',  'P_kbar', 'b'),
    ('opx_only', 'T_C',    'c'),
    ('opx_only', 'P_kbar', 'd'),
]

P_REGIME_MIN_N_FOR_CLAIMS = 20


def _panel(ax, pseed, ship_final, track, target, idx):
    x = np.arange(len(REGIME_ORDER))
    w = 0.27
    win_row = ship_final[(ship_final.pipeline == 'opx')
                         & (ship_final.track == track)
                         & (ship_final.target == target)
                         & (ship_final.model != 'TabPFN')]
    winner = win_row.iloc[0]['winner_final'] if not win_row.empty else 'none'
    model_tag = win_row['model'].iloc[0] if not win_row.empty else ''
    fs_tag = win_row['feature_set'].iloc[0] if not win_row.empty else ''
    family = family_from_method(model_tag) if model_tag else 'RF'

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
            ns.append(0)
            continue
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

    pre_m = np.asarray(pre_m, float)
    pre_lo = np.asarray(pre_lo, float); pre_hi = np.asarray(pre_hi, float)
    a_m = np.asarray(a_m, float)
    a_lo = np.asarray(a_lo, float); a_hi = np.asarray(a_hi, float)
    b_m = np.asarray(b_m, float)
    b_lo = np.asarray(b_lo, float); b_hi = np.asarray(b_hi, float)

    fam_color, pre_alpha, pre_hatch = role_style('pre', family)
    post_color, _, _ = role_style('post', family)
    putirka_gray = MODEL_COLORS['Putirka']

    bars_pre = ax.bar(x - w, pre_m, width=w, color=fam_color,
                      alpha=pre_alpha, hatch=pre_hatch,
                      label='pre-correction')
    bars_a = ax.bar(x, a_m, width=w, color=post_color,
                    label='post Form A')
    bars_b = ax.bar(x + w, b_m, width=w, color=putirka_gray,
                    label='post Form B')

    # Mark sample-limited regimes (n<20) with extra hatching across the
    # whole triad — overlaying the existing pre-correction hatch.
    sl_hatch = '\\\\\\'
    for bars in (bars_pre, bars_a, bars_b):
        for bar, n, reg in zip(bars, ns, REGIME_ORDER):
            if n < P_REGIME_MIN_N_FOR_CLAIMS and reg != 'ALL':
                bar.set_hatch(sl_hatch)

    ax.vlines(x - w, pre_lo, pre_hi, color='#333333', lw=0.8)
    ax.vlines(x, a_lo, a_hi, color='#333333', lw=0.8)
    ax.vlines(x + w, b_lo, b_hi, color='#333333', lw=0.8)

    top = np.nanmax(np.concatenate([pre_hi, a_hi, b_hi]))
    if np.isfinite(top) and top > 0:
        ax.set_ylim(0, top * 1.14)

    ship_label = {'A': 'A', 'B': 'B'}.get(winner, 'none')
    ship_txt = (f'[Form {ship_label}]' if ship_label != 'none'
                else '[no ship]')
    ax.text(0.98, 0.97, ship_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, fontweight='bold',
            color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_TICKS)
    ax.set_ylabel(f'RMSE ({TARGET_UNIT[target]})')
    ax.set_title(panel_header(track, target, idx))
    if model_tag:
        sub = f'{model_tag.split("/")[0]}/{fs_tag}'
        ax.text(0.02, 0.97, sub, transform=ax.transAxes,
                va='top', ha='left', fontsize=8, color='#444444')
    add_grid(ax)


def main():
    pseed = pd.read_csv('results/bias_correction_per_seed.csv')
    ship_final = pd.read_csv('results/bias_correction_shipped.csv').rename(
        columns={'winner_v3': 'winner_final'})

    fig, axes = make_fig('two_col', nrows=2, ncols=2)
    for ax, (track, target, idx) in zip(axes.ravel(), PANELS):
        _panel(ax, pseed, ship_final, track, target, idx)

    # Use a representative family (RF) for legend swatch coloring; the
    # caption explains that the actual bar inherits the per-cell winner.
    handles = [
        role_patch('pre', family='RF',
                   label='pre-correction (winner family color)'),
        role_patch('post', family='RF',
                   label='post Form A (winner family color)'),
        mpatches.Patch(facecolor=MODEL_COLORS['Putirka'],
                       edgecolor='#333333', label='post Form B'),
        mpatches.Patch(facecolor='white', edgecolor='#333333',
                       hatch='\\\\\\',
                       label=f'n < {P_REGIME_MIN_N_FOR_CLAIMS} regime'),
    ]
    add_below_legend(fig, handles, [h.get_label() for h in handles], ncol=4)

    fig.suptitle(
        'Bias-correction effect on per-regime RMSE (opx)\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190), 20-seed'
    )

    stem = OUT_DIR / 'main_fig_6'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Figure 6. Bias-correction effect on per-regime test-set RMSE for '
        'the four opx (track, target) cells. Each regime cluster shows '
        'three bars: pre-correction (winner family color, faint and hatched), '
        'post Form A (regime-piecewise OLS, winner family color solid), '
        'post Form B (Agreda-Lopez piecewise sigmoid, gray). Bar color '
        'identifies the per-cell post-correction winning family on the '
        'paper-wide locked palette. Bar heights are 20-seed means; whiskers '
        'are the 20-seed mean of per-seed bootstrap 95% CIs. Diagonal '
        'cross-hatching marks regimes with n < 20 (sample-limited; per the '
        'pre-registered honesty bar these regimes are exempt from the '
        'tolerance veto and reported without directional claim). The '
        'shipped form per panel is named in the corner: [Form A], [Form B], '
        'or [no ship]. A form ships only if it reduces aggregate RMSE and '
        'does not degrade any n ≥ 20 regime beyond max(T_ABS, T_REL × '
        'pre_rmse) with T_ABS = 10 °C / 1 kbar and T_REL = 0.10. Sources: '
        'results/bias_correction_per_seed.csv, '
        'results/bias_correction_shipped.csv.'
    )
    (OUT_DIR / 'main_fig_6.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
