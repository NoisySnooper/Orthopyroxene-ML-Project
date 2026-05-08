#!/usr/bin/env python3
"""main_fig_7: scorecard delta heatmap, opx post-correction vs Putirka.

Rows = pre-registered P regimes (+ ALL). Cols = (track, target). Cell
color is fractional improvement of post-correction RMSE over the best
external Putirka equation. Cells with no Putirka equivalent show the
post-correction absolute RMSE.

Source: results/preregistered_scorecard_postcorrection.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._style import apply_pub_style, make_fig, resolve_out_dir # noqa: E402
from scripts.figures._labels import REGIME_TICK, TARGET_UNIT  # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_TICKS = [REGIME_TICK.get(r, r) for r in REGIME_ORDER]

COLS = [
    ('opx_liq',  'T_C',    'opx-liq\nT (°C)'),
    ('opx_liq',  'P_kbar', 'opx-liq\nP (kbar)'),
    ('opx_only', 'T_C',    'opx-only\nT (°C)'),
    ('opx_only', 'P_kbar', 'opx-only\nP (kbar)'),
]


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

    fig, ax = make_fig('single')
    im = ax.imshow(arr, cmap=cmap_div, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c[2] for c in COLS])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(REGIME_TICKS)

    for i in range(n_rows):
        for j in range(n_cols):
            lbl = labels[i, j]
            if not lbl:
                continue
            if np.isfinite(arr[i, j]) and abs(arr[i, j]) > 0.5 * vmax:
                color = 'white'
            else:
                color = '#222222'
            ax.text(j, i, lbl, ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label('(Putirka RMSE − post-correction RMSE) / Putirka RMSE')

    ax.set_title(
        'Scorecard delta vs Putirka (opx, post-correction)\n'
        'ExPetDB held-out · opx-liq n=174 · opx-only n=190 · '
        'green = ML wins, red = Putirka',
        loc='center',
    )

    stem = OUT_DIR / 'main_fig_7'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Figure 7. Per-regime scorecard delta between our post-bias-'
        'correction RMSE and the best-available Putirka 2008 equation '
        'for the four opx (track, target) cells. Color encodes fractional '
        'improvement (Putirka RMSE − post-correction RMSE) / Putirka RMSE '
        'on a diverging red–white–green scale: green = ML wins, red = '
        'Putirka wins. Cell text shows the absolute RMSE delta in native '
        'units (°C for T, kbar for P) and the per-cell winning family. '
        'External references are Putirka 2008 thermobarometers run through '
        'Thermobar: opx-liq T = eq 28a, opx-liq P = eq 29a (29b in '
        'deeper_mantle), opx-only P = eq 29c. Opx-only T has no Putirka '
        'opx-only thermometer in Thermobar, so its column reports the '
        'absolute post-correction RMSE instead of a delta. Source: '
        'results/preregistered_scorecard_postcorrection.csv.'
    )
    (OUT_DIR / 'main_fig_7.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
