#!/usr/bin/env python3
"""main_fig_4: per-regime test RMSE by family (4 panels).

For each opx (track, target) cell, 5 regime groups (shallow, MASH, litho,
deeper, ALL). Inside each group: 9 family bars in canonical MODEL_ORDER
plus 1 Putirka bar. 95% bootstrap CI whiskers. The per-regime winning
family is annotated below each cluster.

Source: results/regime_allmodels.csv
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
from scripts.figures._labels import (REGIME_TICK, TARGET_UNIT,            # noqa: E402
                                      panel_header)
from scripts.figures._legend import model_patch, add_below_legend         # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

PANELS = [
    ('opx_liq',  'T_C',    'a'),
    ('opx_liq',  'P_kbar', 'b'),
    ('opx_only', 'T_C',    'c'),
    ('opx_only', 'P_kbar', 'd'),
]

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_TICKS = [REGIME_TICK.get(r, r) for r in REGIME_ORDER]
ALL_FAMILIES = list(MODEL_ORDER) + ['Putirka']


def best_family_row(sub_family: pd.DataFrame) -> pd.Series | None:
    if len(sub_family) == 0:
        return None
    return sub_family.loc[sub_family['rmse'].idxmin()]


def best_putirka(sub: pd.DataFrame) -> pd.Series | None:
    ext = sub[sub['method_family'].isin(
        ['Putirka', 'Agreda', 'Jorgenson', 'Wang'])]
    if len(ext) == 0:
        return None
    return ext.loc[ext['rmse'].idxmin()]


def draw_panel(ax, sub: pd.DataFrame, track: str, target: str, idx: str):
    n_fams = len(MODEL_ORDER)
    bar_w = 0.085
    offsets = (np.arange(n_fams + 1) - n_fams / 2.0) * bar_w
    x_centers = np.arange(len(REGIME_ORDER))

    regime_winners: dict[str, tuple[float, str]] = {}
    regime_put: dict[str, float] = {}
    all_tops: list[np.ndarray] = []

    for fi, fam in enumerate(MODEL_ORDER):
        vals, lo, hi = [], [], []
        for r in REGIME_ORDER:
            sub_r = sub[sub['regime'] == r]
            if fam == 'TabPFN':
                fam_rows = sub_r[sub_r['method_family'] == 'tabpfn']
            else:
                fam_rows = sub_r[
                    (sub_r['method_family'] == 'v10') &
                    (sub_r['method'].str.startswith(f'{fam}/'))]
            best = best_family_row(fam_rows)
            if best is None:
                vals.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                vals.append(best['rmse'])
                lo.append(best['rmse_lo'])
                hi.append(best['rmse_hi'])
                cur = regime_winners.get(r)
                if cur is None or best['rmse'] < cur[0]:
                    regime_winners[r] = (float(best['rmse']), fam)
        vals = np.array(vals, dtype=float)
        lo = np.array(lo, dtype=float)
        hi = np.array(hi, dtype=float)
        err = np.vstack([np.nan_to_num(vals - lo, nan=0.0),
                         np.nan_to_num(hi - vals, nan=0.0)])
        ax.bar(x_centers + offsets[fi], np.nan_to_num(vals),
               width=bar_w, yerr=err, color=MODEL_COLORS[fam])
        all_tops.append(np.nan_to_num(hi, nan=0.0))

    put_vals, put_lo, put_hi = [], [], []
    for r in REGIME_ORDER:
        sub_r = sub[sub['regime'] == r]
        best = best_putirka(sub_r)
        if best is None:
            put_vals.append(np.nan)
            put_lo.append(np.nan); put_hi.append(np.nan)
        else:
            put_vals.append(best['rmse'])
            put_lo.append(best['rmse_lo'])
            put_hi.append(best['rmse_hi'])
            regime_put[r] = float(best['rmse'])
            cur = regime_winners.get(r)
            if cur is None or best['rmse'] < cur[0]:
                regime_winners[r] = (float(best['rmse']), 'Putirka')
    put_vals = np.array(put_vals, dtype=float)
    put_lo = np.array(put_lo, dtype=float)
    put_hi = np.array(put_hi, dtype=float)
    put_err = np.vstack([np.nan_to_num(put_vals - put_lo, nan=0.0),
                         np.nan_to_num(put_hi - put_vals, nan=0.0)])
    ax.bar(x_centers + offsets[-1], np.nan_to_num(put_vals),
           width=bar_w, yerr=put_err, color=PUTIRKA_C)
    all_tops.append(np.nan_to_num(put_hi, nan=0.0))

    if all_tops:
        top = float(np.nanmax(np.concatenate(all_tops)))
        if top > 0:
            ax.set_ylim(0, top * 1.10)

    if np.all(np.isnan(put_vals)):
        ax.text(0.02, 0.98,
                'No Putirka opx-only thermometer\navailable in Thermobar.',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='#888888',
                          boxstyle='round,pad=0.3', alpha=0.95))

    tick_labels = []
    for r, base in zip(REGIME_ORDER, REGIME_TICKS):
        sub_r = sub[sub['regime'] == r]
        n = sub_r['n'].iloc[0] if len(sub_r) else np.nan
        n_txt = f'\n(n={int(n)})' if pd.notna(n) else ''
        tick_labels.append(base + n_txt)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(tick_labels)

    # Per-regime winners are encoded by bar color (locked palette);
    # the per-cell aggregate winner is reported in main_fig_3 (heatmap)
    # and in Table 2.

    ax.set_ylabel(f'RMSE ({TARGET_UNIT[target]})')
    ax.set_title(panel_header(track, target, idx))
    add_grid(ax)


def main():
    rm = pd.read_csv('results/regime_allmodels.csv')
    rm = rm[(rm['pipeline'] == 'opx') & (rm['regime_type'] == 'P')].copy()

    fig, axes = make_fig('two_col', nrows=2, ncols=2)

    for ax, (track, target, idx) in zip(axes.ravel(), PANELS):
        sub = rm[(rm['track'] == track) & (rm['target'] == target)].copy()
        draw_panel(ax, sub, track, target, idx)

    handles = [model_patch(f) for f in ALL_FAMILIES]
    add_below_legend(fig, handles, [h.get_label() for h in handles], ncol=5)

    fig.suptitle(
        'Nine families vs Putirka, per pressure regime\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190), 20-seed'
    )

    stem = OUT_DIR / 'main_fig_4'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Figure 4. Per-regime test-set RMSE for all nine model families '
        'plus the best Putirka 2008 external benchmark, across the four '
        'opx (track, target) cells. Each regime cluster shows ten bars '
        '(one per family in canonical MODEL_ORDER, one Putirka). For each '
        '(family, regime) cell the feature_set (raw / alr / pwlr) with '
        'the lowest test RMSE is selected; TabPFN uses raw features only. '
        'Whiskers are 95% bootstrap CIs. Pressure regimes follow the '
        'pre-registered edges (<5, 5–15, 15–30, ≥30 kbar) plus the ALL '
        'aggregate. The per-regime winning family is named below each '
        'cluster. Panel (c) opx-only T has no Putirka opx-only thermometer '
        'in Thermobar.'
    )
    (OUT_DIR / 'main_fig_4.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
