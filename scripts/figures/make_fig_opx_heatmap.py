#!/usr/bin/env python3
"""main_fig_3: opx performance heatmap.

Rows = 4 opx (track, target) cells.
Cols = 9 ML families in MODEL_ORDER + Putirka.
Cell value = canonical-seed-42 test-set RMSE with bootstrap-on-residuals
95% CI half-width (n_boot=500). The same CI source is used for every
column (ML + Putirka), so deterministic learners like ElasticNet get a
real bootstrap CI rather than a spurious ±0.00 from zero seed-variance.

Source:
  results/bootstrap_rmse_cis_all_cells.csv   (ML + Putirka bootstrap CIs)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._model_palette import MODEL_ORDER  # noqa: E402
from scripts.figures._style import apply_pub_style, make_fig, resolve_out_dir # noqa: E402
from scripts.figures._labels import TARGET_UNIT  # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

ROWS = [
    ('opx_liq',  'T_C',    'opx-liq T (°C)'),
    ('opx_liq',  'P_kbar', 'opx-liq P (kbar)'),
    ('opx_only', 'T_C',    'opx-only T (°C)'),
    ('opx_only', 'P_kbar', 'opx-only P (kbar)'),
]

COLS = list(MODEL_ORDER) + ['Putirka']


def fill_row(boot: pd.DataFrame, sc_all: pd.DataFrame,
             track: str, target: str):
    """Build (rmses, half_widths) for one heatmap row.

    ML families: best feature_set by `rmse_point` at canonical seed 42,
    with bootstrap-on-residuals 95% half-width.
    Putirka: scorecard ALL-row, with half of the scorecard bootstrap CI.
    """
    rmses, halfs = [], []
    sub = boot[(boot['track'] == track) & (boot['target'] == target)]
    for fam in MODEL_ORDER:
        rows = sub[sub['model'] == fam]
        if len(rows) == 0:
            rmses.append(np.nan)
            halfs.append(np.nan)
            continue
        best = rows.loc[rows['rmse_point'].idxmin()]
        rmses.append(float(best['rmse_point']))
        halfs.append(float(best['ci_half_width']))
    pu = sc_all[(sc_all['track'] == track) & (sc_all['target'] == target)]
    if len(pu):
        r = pu.iloc[0]
        if pd.notna(r['best_external_rmse']):
            rmses.append(float(r['best_external_rmse']))
            lo, hi = r['best_external_rmse_lo'], r['best_external_rmse_hi']
            halfs.append(float((hi - lo) / 2.0)
                         if pd.notna(lo) and pd.notna(hi) else np.nan)
        else:
            rmses.append(np.nan)
            halfs.append(np.nan)
    else:
        rmses.append(np.nan)
        halfs.append(np.nan)
    return np.array(rmses), np.array(halfs)


def main():
    boot = pd.read_csv('results/bootstrap_rmse_cis_all_cells.csv')
    sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')
    sc_all = sc[sc['regime'] == 'ALL']

    rmse_grid = np.full((len(ROWS), len(COLS)), np.nan)
    half_grid = np.full((len(ROWS), len(COLS)), np.nan)

    for ri, (track, target, _) in enumerate(ROWS):
        r, h = fill_row(boot, sc_all, track, target)
        rmse_grid[ri, :] = r
        half_grid[ri, :] = h

    normed = np.full_like(rmse_grid, np.nan, dtype=float)
    for ri in range(rmse_grid.shape[0]):
        row = rmse_grid[ri]
        mn = np.nanmin(row)
        mx = np.nanmax(row)
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            normed[ri] = (row - mn) / (mx - mn)
        else:
            normed[ri] = 0.5

    fig, ax = make_fig('wide')
    im = ax.imshow(normed, cmap='viridis_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(COLS)))
    ax.set_xticklabels(COLS, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(ROWS)))
    ax.set_yticklabels([r[2] for r in ROWS])

    for ri in range(rmse_grid.shape[0]):
        unit = TARGET_UNIT[ROWS[ri][1]]
        for ci in range(rmse_grid.shape[1]):
            val = rmse_grid[ri, ci]
            half = half_grid[ri, ci]
            if not np.isfinite(val):
                # Hatched empty cell with mid-gray fill, dark text — readable
                # on every background.
                ax.add_patch(mpatches.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    facecolor='#d8d8d8', edgecolor='#888888',
                    hatch='///', linewidth=0.5, zorder=2))
                ax.text(ci, ri, 'N/A', ha='center', va='center',
                        fontsize=9, color='#333333', fontweight='bold',
                        zorder=3)
                continue
            if np.isfinite(half):
                text = f'{val:.2f}\n±{half:.2f}\n{unit}'
            else:
                text = f'{val:.2f}\n{unit}'
            color = 'white' if normed[ri, ci] > 0.55 else '#222222'
            ax.text(ci, ri, text, ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

        row_vals = rmse_grid[ri]
        finite = np.where(np.isfinite(row_vals))[0]
        if len(finite):
            best_ci = int(finite[np.argmin(row_vals[finite])])
            ax.add_patch(mpatches.Rectangle(
                (best_ci - 0.47, ri - 0.47), 0.94, 0.94,
                fill=False, edgecolor='#D55E00', linewidth=1.6,
                zorder=5,
            ))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which='both', length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Row-normalised RMSE (0 = best in row, 1 = worst)')

    ax.set_title(
        'Test-set RMSE: 4 opx cells × 10 candidates\n'
        'ExPetDB held-out · opx-liq n=174 · opx-only n=190',
        loc='center',
    )

    stem = OUT_DIR / 'main_fig_3'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Figure 3. Overall test-set RMSE heatmap, four opx (track, target) '
        'cells (rows) × ten candidates (columns: nine ML families in '
        'canonical MODEL_ORDER plus best-available Putirka 2008 equation). '
        'Each cell reports the canonical-seed-42 RMSE with the half-width '
        'of the 95% bootstrap-on-residuals confidence interval (n_boot=500, '
        'paired resampling of test residuals). The same CI source is used '
        'for every column so deterministic learners (e.g. ElasticNet) are '
        'reported on the same uncertainty footing as stochastic ones. Cells '
        'are colored on a row-normalised viridis_r colormap so the row '
        'minimum is lightest. Absolute values are printed in each cell in '
        'native units (°C for T, kbar for P). Opx-only T has no Putirka '
        'opx-only thermometer available in Thermobar and is hatched (N/A). '
        'The minimum-RMSE cell in each row is outlined in vermillion.'
    )
    (OUT_DIR / 'main_fig_3.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
