#!/usr/bin/env python3
"""supp_fig_2: per-family overall (ALL-regime) test RMSE, 4 panels.

For each opx (track, target) cell, one bar per model family (9 ML + Putirka,
canonical MODEL_ORDER). Bar height = 20-seed mean RMSE; whiskers = mean ±
1.96 × std. The best feature_set per family is selected automatically.

Source: results/opx_multiseed_summary.csv,
        results/tabpfn_multiseed_summary.csv,
        results/preregistered_scorecard_postcorrection.csv (ALL row)
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

from scripts.figures._model_palette import (MODEL_COLORS, MODEL_ORDER,    # noqa: E402
                                             PUTIRKA_C)
from scripts.figures._style import apply_pub_style, make_fig, add_grid, resolve_out_dir # noqa: E402
from scripts.figures._labels import TARGET_UNIT, panel_header             # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

PANELS = [
    ('opx_liq',  'T_C',    'a'),
    ('opx_liq',  'P_kbar', 'b'),
    ('opx_only', 'T_C',    'c'),
    ('opx_only', 'P_kbar', 'd'),
]

ALL_LABELS = list(MODEL_ORDER) + ['Putirka']


def best_row_per_family(ms: pd.DataFrame, track: str, target: str) -> dict:
    sub = ms[(ms['track'] == track) & (ms['target'] == target)]
    out = {}
    for fam in MODEL_ORDER:
        fam_rows = sub[sub['model'] == fam]
        if len(fam_rows) == 0:
            out[fam] = None
            continue
        out[fam] = fam_rows.loc[fam_rows['mean'].idxmin()]
    return out


def draw_panel(ax, ms_all, sc_all, track, target, idx):
    best = best_row_per_family(ms_all, track, target)
    pu = sc_all[(sc_all['track'] == track) & (sc_all['target'] == target)]
    put_rmse = float(pu['best_external_rmse'].iloc[0]) if len(pu) else np.nan
    put_lo = float(pu['best_external_rmse_lo'].iloc[0]) if len(pu) else np.nan
    put_hi = float(pu['best_external_rmse_hi'].iloc[0]) if len(pu) else np.nan

    x = np.arange(len(ALL_LABELS))
    rmse = np.array(
        [best[f]['mean'] if best[f] is not None else np.nan
         for f in MODEL_ORDER] + [put_rmse], dtype=float)
    ci_lo = np.array(
        [(best[f]['mean'] - 1.96 * best[f]['std'])
         if best[f] is not None else np.nan
         for f in MODEL_ORDER] + [put_lo], dtype=float)
    ci_hi = np.array(
        [(best[f]['mean'] + 1.96 * best[f]['std'])
         if best[f] is not None else np.nan
         for f in MODEL_ORDER] + [put_hi], dtype=float)
    err = np.vstack([np.nan_to_num(rmse - ci_lo, nan=0.0),
                     np.nan_to_num(ci_hi - rmse, nan=0.0)])

    colors = [MODEL_COLORS[f] for f in MODEL_ORDER] + [PUTIRKA_C]
    ax.bar(x, np.nan_to_num(rmse), yerr=err, color=colors)

    top = float(np.nanmax(np.nan_to_num(ci_hi, nan=0.0)))
    if top > 0:
        ax.set_ylim(0, top * 1.10)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_LABELS, rotation=30, ha='right')
    ax.set_ylabel(f'RMSE ({TARGET_UNIT[target]})')
    ax.set_title(panel_header(track, target, idx))
    add_grid(ax)


def main():
    ms_tuned = pd.read_csv('results/opx_multiseed_summary.csv')
    ms_tab = pd.read_csv('results/tabpfn_multiseed_summary.csv')
    ms_all = pd.concat([ms_tuned, ms_tab], ignore_index=True)

    sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')
    sc_all = sc[sc['regime'] == 'ALL']

    fig, axes = make_fig('two_col', nrows=2, ncols=2)
    for ax, (track, target, idx) in zip(axes.ravel(), PANELS):
        draw_panel(ax, ms_all, sc_all, track, target, idx)

    fig.suptitle(
        'Nine families + Putirka, overall test-set RMSE\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190), 20-seed'
    )

    stem = OUT_DIR / 'supp_fig_2'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Supp. Figure 2. Overall test-set RMSE per model family for the '
        'four opx (track, target) cells, aggregated across the full held-'
        'out partition (no regime split). Bars show 20-seed mean RMSE for '
        'every family (TabPFN runs the same 20-seed protocol without '
        'hyperparameter tuning). Whiskers are 95% intervals computed as '
        'mean ± 1.96 × std over seeds. For each family the feature_set '
        '(raw / alr / pwlr) with the lowest mean RMSE is selected. '
        'Putirka shows the best-available 2008 equation per cell. Panel '
        '(c) opx-only T has no Putirka opx-only thermometer in Thermobar.'
    )
    (OUT_DIR / 'supp_fig_2.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
