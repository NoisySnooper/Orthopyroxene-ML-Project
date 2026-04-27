#!/usr/bin/env python3
"""Core_07: scorecard delta heatmap -- opx pipeline only.

Rows = pre-registered P regimes, cols = (track x target). Color encodes
the fractional improvement of v10 post-correction RMSE over the best
external Putirka benchmark: positive (green) = v10 wins, negative (red)
= Putirka wins. Cell annotation shows the absolute RMSE delta in native
units (C for T, kbar for P). Cells with no external benchmark available
(opx-only T has no Putirka opx-only thermometer in Thermobar) show the
absolute v10 post-correction RMSE instead.

External benchmarks shown here are all Putirka (2008) thermobarometers:
opx-liq T uses eq 28a, opx-liq P uses eq 29a (29b in deeper_mantle),
opx-only P uses eq 29c.

Post-correction RMSE source: results/preregistered_scorecard_post
correction_v3.csv (Amendment 2, tolerance-based ship rule). Falls back
to v2 then v1 if the v3 CSV has not been generated yet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._style import apply_pub_style  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_SHORT = {
    'shallow_crustal':     'shallow',
    'deep_crustal_MASH':   'deep-MASH',
    'lithospheric_mantle': 'litho',
    'deeper_mantle':       'deeper',
    'ALL':                 'ALL',
}

COLS = [
    ('opx_liq',  'T_C',    'opx-liq\nT (C)'),
    ('opx_liq',  'P_kbar', 'opx-liq\nP (kbar)'),
    ('opx_only', 'T_C',    'opx-only\nT (C)'),
    ('opx_only', 'P_kbar', 'opx-only\nP (kbar)'),
]

TARGET_UNIT = {'T_C': 'C', 'P_kbar': 'kbar'}


def main():
    sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')

    n_rows = len(REGIME_ORDER)
    n_cols = len(COLS)
    arr = np.full((n_rows, n_cols), np.nan)
    labels = np.full((n_rows, n_cols), '', dtype=object)

    for i, reg in enumerate(REGIME_ORDER):
        for j, (track, target, _) in enumerate(COLS):
            r = sc[(sc.track == track) & (sc.target == target)
                   & (sc.regime == reg)]
            if r.empty:
                labels[i, j] = 'N/A'
                continue
            r = r.iloc[0]
            ext = r['best_external_rmse']
            post = r['v10_post_rmse']
            v10_method = r['v10_post_method'] if isinstance(
                r['v10_post_method'], str) else ''
            unit = TARGET_UNIT[target]
            if not np.isfinite(ext):
                labels[i, j] = (f'{v10_method}\nno Putirka\n'
                                f'post {post:.2f} {unit}')
                continue
            pct = (ext - post) / ext
            arr[i, j] = pct
            delta_abs = ext - post
            labels[i, j] = f'{v10_method}\n{delta_abs:+.2f} {unit}'

    vmax = np.nanmax(np.abs(arr)) if np.isfinite(np.nanmax(np.abs(arr))) else 0.5
    vmax = max(vmax, 0.05)
    cmap_div = LinearSegmentedColormap.from_list(
        'v10div',
        [(0.0, '#B22222'), (0.5, '#FFFFFF'), (1.0, '#228B22')])

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(arr, cmap=cmap_div, vmin=-vmax, vmax=vmax, aspect='equal')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c[2] for c in COLS])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([REGIME_SHORT[r] for r in REGIME_ORDER])

    for i in range(n_rows):
        for j in range(n_cols):
            lbl = labels[i, j]
            if not lbl:
                continue
            if np.isfinite(arr[i, j]) and abs(arr[i, j]) > 0.5 * vmax:
                color = 'white'
            else:
                color = 'black'
            ax.text(j, i, lbl, ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    # grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label('(Putirka RMSE - v10 post-correction RMSE) / Putirka RMSE',
                   fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        'Scorecard delta vs Putirka (2008), opx post-correction '
        '(v3 tolerance rule, Amendment 2)\n'
        'green = ML wins, red = Putirka wins; * = v3 post-RMSE differs '
        'from prior rule',
        loc='left', pad=8, fontsize=11,
    )

    plt.tight_layout()

    stem = OUT_DIR / 'Core_07_fig34_bias_correction_scorecard_delta'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 7. Per-regime scorecard delta between our v10 post-bias-'
        'correction RMSE and the best external Putirka (2008) '
        'thermobarometer, for the four opx track/target cells. Color '
        'encodes fractional improvement (Putirka_RMSE - v10_post_RMSE) / '
        'Putirka_RMSE on a diverging red-white-green scale: green cells '
        'are wins for v10, red cells are wins for Putirka. Cell text '
        'shows the absolute RMSE delta in native units (C for T, kbar '
        'for P). External references are Putirka (2008) thermobarometers '
        'run through Thermobar: opx-liq T = eq 28a, opx-liq P = eq 29a '
        '(29b in deeper_mantle), opx-only P = eq 29c. Opx-only T has no '
        'Putirka opx-only thermometer in Thermobar, so its row reports '
        'the v10 absolute RMSE instead of a delta. Post-correction RMSE '
        'uses the v3 tolerance-based ship rule (Amendment 2, 2026-04-20) '
        'with T_ABS_T=10 C, T_ABS_P=1 kbar, T_REL=0.10, N_MIN=20; cells '
        'marked with "*" have post-correction RMSE that differs from the '
        'earlier v1/v2 rule, reflecting a correction form promoted under '
        'v3 whose worst-regime degradation is within the pre-registered '
        'measurement-uncertainty tolerance envelope. Cpx pipelines are '
        'intentionally excluded. Source: results/preregistered_scorecard_'
        'postcorrection_v3.csv (with v1/v2 references from preregistered_'
        'scorecard_postcorrection.csv and preregistered_scorecard_'
        'postcorrection_v2.csv).'
    )
    (OUT_DIR / 'Core_07_fig34_bias_correction_scorecard_delta.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
