#!/usr/bin/env python3
"""Core_09a: all 9 model families + Putirka per regime (4 panels).

For each of opx_liq T, opx_liq P, opx_only T, opx_only P, show 5 regime
groups (shallow_crustal, deep_crustal_MASH, lithospheric_mantle,
deeper_mantle, ALL). Inside each group: 9 narrow colored bars (one per
family, feature_set picked per regime by min rmse) plus one gray Putirka
bar when available. 95% bootstrap CI whiskers from rmse_lo/rmse_hi.

Source: results/regime_allmodels.csv (all families + Putirka) and
results/preregistered_scorecard_postcorrection.csv (ALL-row Putirka
agreement check). Data is already computed; this script only plots.
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

from scripts.figures._model_palette import (  # noqa: E402
    MODEL_COLORS, MODEL_ORDER,
)
from scripts.figures._style import apply_pub_style  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PUTIRKA_C = '#555555'

PANELS = [
    ('opx_liq',  'T_C',    '(a) Opx + Liquid  T (C)'),
    ('opx_liq',  'P_kbar', '(b) Opx + Liquid  P (kbar)'),
    ('opx_only', 'T_C',    '(c) Opx only  T (C)'),
    ('opx_only', 'P_kbar', '(d) Opx only  P (kbar)'),
]

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_LABELS = ['shallow\n<5 kbar', 'MASH\n5-15', 'lithos\n15-30',
                 'deep\n>=30', 'ALL']


def best_family_row(sub_family: pd.DataFrame) -> pd.Series | None:
    """Pick the feature_set (or sole TabPFN variant) with the lowest rmse.

    sub_family is already filtered to one (track, target, regime, family).
    """
    if len(sub_family) == 0:
        return None
    return sub_family.loc[sub_family['rmse'].idxmin()]


def best_putirka(sub: pd.DataFrame) -> pd.Series | None:
    """Return the single Putirka / Agreda / etc row with lowest rmse."""
    ext = sub[sub['method_family'].isin(
        ['Putirka', 'Agreda', 'Jorgenson', 'Wang'])]
    if len(ext) == 0:
        return None
    return ext.loc[ext['rmse'].idxmin()]


def main():
    rm = pd.read_csv('results/regime_allmodels.csv')
    rm = rm[(rm['pipeline'] == 'opx') & (rm['regime_type'] == 'P')].copy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.ravel()

    n_fams = len(MODEL_ORDER)
    bar_w = 0.085
    offsets = (np.arange(n_fams + 1) - (n_fams) / 2.0) * bar_w

    for ax, (track, target, title) in zip(axes, PANELS):
        sub = rm[(rm['track'] == track) & (rm['target'] == target)].copy()

        x_centers = np.arange(len(REGIME_ORDER))
        # winner-per-regime tracking: {regime: (rmse, fam_label, fs)}
        regime_winners: dict[str, tuple[float, str, str]] = {}
        # best ML per regime: {regime: (rmse, fam_label, fs)}
        regime_ml_best: dict[str, tuple[float, str, str]] = {}
        # Putirka per regime: {regime: rmse}
        regime_put: dict[str, float] = {}

        # Collect values per family per regime
        all_tops = []
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
                    vals.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                else:
                    vals.append(best['rmse'])
                    lo.append(best['rmse_lo'])
                    hi.append(best['rmse_hi'])
                    # method encodes "family/fs" e.g. "RF/pwlr" for
                    # v10 rows; TabPFN rows carry just "tabpfn" or
                    # similar, so no split happens.
                    method_s = str(best.get('method', ''))
                    if '/' in method_s:
                        fs = method_s.split('/', 1)[1]
                    else:
                        fs = ''
                    cur_w = regime_winners.get(r)
                    if cur_w is None or best['rmse'] < cur_w[0]:
                        regime_winners[r] = (float(best['rmse']), fam, fs)
                    cur_ml = regime_ml_best.get(r)
                    if cur_ml is None or best['rmse'] < cur_ml[0]:
                        regime_ml_best[r] = (float(best['rmse']), fam, fs)
            vals = np.array(vals, dtype=float)
            lo = np.array(lo, dtype=float)
            hi = np.array(hi, dtype=float)
            err = np.vstack([
                np.nan_to_num(vals - lo, nan=0.0),
                np.nan_to_num(hi - vals, nan=0.0),
            ])
            ax.bar(
                x_centers + offsets[fi], np.nan_to_num(vals),
                width=bar_w, yerr=err, capsize=2,
                color=MODEL_COLORS[fam], edgecolor='black', linewidth=0.3,
                label=fam if ax is axes[0] else None,
            )
            all_tops.append(np.nan_to_num(hi, nan=0.0))

        # Putirka bar at the last offset
        put_vals, put_lo, put_hi = [], [], []
        for r in REGIME_ORDER:
            sub_r = sub[sub['regime'] == r]
            best = best_putirka(sub_r)
            if best is None:
                put_vals.append(np.nan)
                put_lo.append(np.nan)
                put_hi.append(np.nan)
            else:
                put_vals.append(best['rmse'])
                put_lo.append(best['rmse_lo'])
                put_hi.append(best['rmse_hi'])
                regime_put[r] = float(best['rmse'])
                # Putirka "fs" slot holds the equation tag (eq28a,
                # eq29c, etc.) so the annotation shows which calibration
                # we're comparing against.
                put_tag = str(best.get('method', '')) if 'method' in best.index else ''
                cur_w = regime_winners.get(r)
                if cur_w is None or best['rmse'] < cur_w[0]:
                    regime_winners[r] = (float(best['rmse']), 'Putirka', put_tag)
        put_vals = np.array(put_vals, dtype=float)
        put_lo = np.array(put_lo, dtype=float)
        put_hi = np.array(put_hi, dtype=float)
        put_err = np.vstack([
            np.nan_to_num(put_vals - put_lo, nan=0.0),
            np.nan_to_num(put_hi - put_vals, nan=0.0),
        ])
        ax.bar(
            x_centers + offsets[-1], np.nan_to_num(put_vals),
            width=bar_w, yerr=put_err, capsize=2,
            color=PUTIRKA_C, edgecolor='black', linewidth=0.3,
            label='Putirka / Agreda' if ax is axes[0] else None,
        )
        all_tops.append(np.nan_to_num(put_hi, nan=0.0))

        top_of_data = float(np.nanmax(np.concatenate(all_tops))) if all_tops else 1.0
        if top_of_data > 0:
            ax.set_ylim(0, top_of_data * 1.08)

        # Annotation if Putirka absent for this panel
        if np.all(np.isnan(put_vals)):
            ax.text(
                0.02, 0.98,
                'No Putirka / Agreda opx-only T\nthermometer available in '
                'Thermobar.',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=9, color='0.25',
                bbox=dict(facecolor='white', edgecolor='0.5',
                          alpha=0.92, pad=4))

        # xticks
        tick_labels = []
        for r, base in zip(REGIME_ORDER, REGIME_LABELS):
            sub_r = sub[sub['regime'] == r]
            n = sub_r['n'].iloc[0] if len(sub_r) else np.nan
            n_txt = f'\n(n={int(n)})' if pd.notna(n) else ''
            tick_labels.append(base + n_txt)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(tick_labels, fontsize=9)

        # per-regime winner label below x-axis (below n). Margin = %
        # improvement vs Putirka; if Putirka wins, margin is negative.
        # fs_w is the transform ("raw" / "alr" / "pwlr" for ML; Putirka
        # equation tag otherwise) -- shown so the reader knows which
        # input representation the winning model was trained on.
        for xi, r in zip(x_centers, REGIME_ORDER):
            w = regime_winners.get(r)
            if w is None:
                continue
            rmse_w, fam_w, fs_w = w
            col = PUTIRKA_C if fam_w == 'Putirka' else MODEL_COLORS.get(
                fam_w, '0.2')
            put_r = regime_put.get(r, np.nan)
            ml_r = regime_ml_best.get(r, (np.nan, None, ''))
            if fam_w == 'Putirka' and np.isfinite(ml_r[0]):
                pct = (rmse_w - ml_r[0]) / rmse_w * 100
                margin = f'\n{pct:+.1f}% vs ML'
            elif np.isfinite(put_r) and put_r > 0:
                pct = (put_r - rmse_w) / put_r * 100
                margin = f'\n{pct:+.1f}% vs Put'
            else:
                margin = ''
            # Each piece on its own line so neighbouring regime labels
            # do not overlap horizontally. For ML winners the second
            # line is "data aug: raw/alr/pwlr"; for Putirka it's
            # "equation: <tag>".
            if fam_w == 'Putirka':
                fs_line = f'equation: {fs_w}' if fs_w else ''
            else:
                fs_line = f'data aug: {fs_w}' if fs_w else ''
            lines = [f'winner: {fam_w}']
            if fs_line:
                lines.append(fs_line)
            lines.append(f'RMSE={rmse_w:.2f}')
            if margin:
                lines.append(margin.lstrip('\n'))
            ax.text(
                xi, -0.14,
                '\n'.join(lines),
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7,
                fontweight='bold', color=col,
                linespacing=1.15,
            )

        unit = 'C' if target == 'T_C' else 'kbar'
        ax.set_ylabel(f'RMSE ({unit})', fontsize=11)
        ax.set_title(title, fontsize=12, loc='left', fontweight='bold',
                     pad=8)
        ax.grid(axis='y', ls=':', alpha=0.4)

    # Shared legend for all families + Putirka
    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS[f], edgecolor='black', label=f)
        for f in MODEL_ORDER
    ]
    legend_handles.append(
        mpatches.Patch(facecolor=PUTIRKA_C, edgecolor='black',
                       label='Putirka / Agreda'))
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=10,
        frameon=True, framealpha=0.95, edgecolor='0.6',
        bbox_to_anchor=(0.5, 0.01), fontsize=10,
        title='9 model families + external benchmark (95% CI whiskers)',
        title_fontsize=10,
    )

    fig.suptitle(
        'All 9 model families versus Putirka / Agreda, per pre-registered '
        'pressure regime  (pre-correction)\n'
        'Benchmark set: ExPetDB opx held-out test partition (citation-grouped '
        'split). Winner %% is vs Putirka; negative = Putirka wins.',
        fontsize=13, fontweight='bold', y=0.99,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.96))
    plt.subplots_adjust(bottom=0.18, hspace=0.45)

    out_stem = OUT_DIR / 'Core_09a_fig_opx_regime_families'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 9a. Per-regime test-set RMSE for all nine model families '
        'versus the best Putirka / Agreda external benchmark, across the '
        'four opx combinations. Each regime group contains ten bars: nine '
        'colored bars (one per family, in the canonical MODEL_ORDER) plus '
        'a gray Putirka / Agreda bar. For each (family, regime) cell the '
        'feature_set (raw / alr / pwlr) with the lowest test RMSE is '
        'selected; whiskers are 95% bootstrap CIs. Pressure regimes are '
        'the pre-registered edges shallow_crustal <5 kbar, '
        'deep_crustal_MASH 5-15 kbar, lithospheric_mantle 15-30 kbar, '
        'deeper_mantle >=30 kbar, plus the ALL aggregate column. Panel '
        '(c) opx-only T has no Putirka opx-only thermometer in Thermobar '
        'and is annotated accordingly. Core_09b is the overall-only '
        'companion; Core_10 collapses to best-of-ours vs Putirka.'
    )
    (OUT_DIR / 'Core_09a_fig_opx_regime_families.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
