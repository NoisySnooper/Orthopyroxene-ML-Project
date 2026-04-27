#!/usr/bin/env python3
"""Core_04: opx performance heatmap, rows = 4 opx combos, cols = 10 candidates.

Cell value = overall test-set RMSE (ALL regime). Cell label shows the
numeric RMSE plus a +/- 95% CI half-width. Cells colored via viridis
column-normalised so low RMSE per column is lightest.

Rows: (opx_liq, T_C), (opx_liq, P_kbar), (opx_only, T_C), (opx_only, P_kbar)
Cols: ElasticNet, RF, ERT, GB, XGB, LightGBM, CatBoost, MLP, TabPFN, Putirka

Source:
  results/opx_multiseed_summary.csv (8 tuned families, 20-seed)
  results/tabpfn_multiseed_summary.csv (TabPFN, 20-seed)
  results/preregistered_scorecard_postcorrection.csv (Putirka ALL-row)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._model_palette import MODEL_ORDER  # noqa: E402
from scripts.figures._style import apply_pub_style  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS = [
    ('opx_liq',  'T_C',    'opx-liq T (C)'),
    ('opx_liq',  'P_kbar', 'opx-liq P (kbar)'),
    ('opx_only', 'T_C',    'opx-only T (C)'),
    ('opx_only', 'P_kbar', 'opx-only P (kbar)'),
]

COLS = list(MODEL_ORDER) + ['Putirka']


def fill_row(ms: pd.DataFrame, sc_all: pd.DataFrame,
             track: str, target: str):
    rmses, halfs = [], []
    sub = ms[(ms['track'] == track) & (ms['target'] == target)]
    for fam in MODEL_ORDER:
        rows = sub[sub['model'] == fam]
        if len(rows) == 0:
            rmses.append(np.nan)
            halfs.append(np.nan)
            continue
        best = rows.loc[rows['mean'].idxmin()]
        rmses.append(float(best['mean']))
        halfs.append(1.96 * float(best['std']))
    pu = sc_all[(sc_all['track'] == track) & (sc_all['target'] == target)]
    if len(pu):
        r = pu.iloc[0]
        if pd.notna(r['best_external_rmse']):
            rmses.append(float(r['best_external_rmse']))
            lo, hi = r['best_external_rmse_lo'], r['best_external_rmse_hi']
            halfs.append(float((hi - lo) / 2.0) if pd.notna(lo) and pd.notna(hi) else np.nan)
        else:
            rmses.append(np.nan)
            halfs.append(np.nan)
    else:
        rmses.append(np.nan)
        halfs.append(np.nan)
    return np.array(rmses), np.array(halfs)


def main():
    ms_tuned = pd.read_csv('results/opx_multiseed_summary.csv')
    ms_tab = pd.read_csv('results/tabpfn_multiseed_summary.csv')
    ms = pd.concat([ms_tuned, ms_tab], ignore_index=True)
    sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')
    sc_all = sc[sc['regime'] == 'ALL']

    rmse_grid = np.full((len(ROWS), len(COLS)), np.nan)
    half_grid = np.full((len(ROWS), len(COLS)), np.nan)

    for ri, (track, target, _) in enumerate(ROWS):
        r, h = fill_row(ms, sc_all, track, target)
        rmse_grid[ri, :] = r
        half_grid[ri, :] = h

    # Row-normalised viridis (each opx combo has its own unit scale, so
    # normalising per row makes the color scale meaningful).
    normed = np.full_like(rmse_grid, np.nan, dtype=float)
    for ri in range(rmse_grid.shape[0]):
        row = rmse_grid[ri]
        mn = np.nanmin(row)
        mx = np.nanmax(row)
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            normed[ri] = (row - mn) / (mx - mn)
        else:
            normed[ri] = 0.5

    fig, ax = plt.subplots(figsize=(15, 6.5))
    im = ax.imshow(normed, cmap='viridis_r', aspect='equal', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(COLS)))
    ax.set_xticklabels(COLS, fontsize=11, rotation=0, ha='center')
    ax.set_yticks(np.arange(len(ROWS)))
    ax.set_yticklabels([r[2] for r in ROWS], fontsize=11)

    for ri in range(rmse_grid.shape[0]):
        unit = 'C' if ROWS[ri][1] == 'T_C' else 'kbar'
        for ci in range(rmse_grid.shape[1]):
            val = rmse_grid[ri, ci]
            half = half_grid[ri, ci]
            if not np.isfinite(val):
                ax.add_patch(mpatches.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    facecolor='black', edgecolor='none', zorder=2))
                ax.text(ci, ri, 'No Model', ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold',
                        zorder=3)
                continue
            if np.isfinite(half):
                text = f'{val:.2f}\n(\u00b1{half:.2f})\n{unit}'
            else:
                text = f'{val:.2f}\n{unit}'
            color = 'white' if normed[ri, ci] > 0.55 else 'black'
            ax.text(ci, ri, text, ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

        # Highlight the minimum-RMSE cell in this row with a thin red
        # rectangle so the reader immediately sees the best model per
        # opx combo. Use a slightly-inset rectangle so the outline
        # doesn't merge with neighbouring cell edges.
        row_vals = rmse_grid[ri]
        finite = np.where(np.isfinite(row_vals))[0]
        if len(finite):
            best_ci = int(finite[np.argmin(row_vals[finite])])
            ax.add_patch(mpatches.Rectangle(
                (best_ci - 0.47, ri - 0.47), 0.94, 0.94,
                fill=False, edgecolor='red', linewidth=1.6,
                zorder=5,
            ))

    # Seamless heatmap: no grid, no spines.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which='both', length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label('Row-normalised RMSE (0 = best in row, 1 = worst)',
                   fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        'Overall test-set RMSE heatmap -- 4 opx combos x 10 candidates\n'
        '(cell text = mean RMSE \u00b1 half-width of 95% CI; viridis '
        'colormap, row-normalised; thin red outline = best per row)',
        fontsize=12, fontweight='bold', pad=12,
    )

    plt.tight_layout()

    out_stem = OUT_DIR / 'Core_04_fig_nb04_cross_pipeline_heatmap'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 4. Overall test-set RMSE heatmap across the four opx '
        'track/target combinations (rows) versus ten candidates (columns: '
        'the nine ML model families in canonical MODEL_ORDER plus the best '
        'Putirka / Agreda external benchmark). Each cell reports the mean '
        'RMSE of the 20-seed refit (5 seeds for TabPFN) over the held-out '
        'test partition, together with the half-width of the 95% '
        'confidence interval (1.96 * std for the ML families, half the '
        'bootstrap CI for Putirka). Cells are colored via a row-normalised '
        'viridis_r colormap so the row minimum (best) is lightest and the '
        'row maximum (worst) is darkest; absolute values are printed in '
        'each cell in native units (C for T targets, kbar for P targets). '
        'Opx-only T has no Putirka opx-only thermometer available in '
        'Thermobar and is marked N/A. Per-family feature_set choice (raw '
        '/ alr / pwlr) is the 20-seed best per row. The minimum-RMSE cell '
        'in each row is outlined in thin red to highlight the best model '
        'per opx combo.'
    )
    (OUT_DIR / 'Core_04_fig_nb04_cross_pipeline_heatmap.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
