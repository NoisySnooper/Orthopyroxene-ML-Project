#!/usr/bin/env python3
"""Core_09b: per-family overall (all-regime) test RMSE, 4 panels.

For each of opx_liq T, opx_liq P, opx_only T, opx_only P, show one bar
per model family (9 total) in canonical order. Bar height is the mean
of the 20-seed refit RMSE; whiskers are 95% CIs (mean +/- 1.96 * std).
For each family the best feature_set by mean is chosen. Putirka / Agreda
overlay (ALL-row from preregistered scorecard) drawn as a horizontal
dashed reference line on each panel, labelled with the method name.

Source: results/opx_multiseed_summary.csv (8 tuned), results/
tabpfn_multiseed_summary.csv (TabPFN), and results/
preregistered_scorecard_postcorrection.csv (ALL-row Putirka benchmark).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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

PUTIRKA_C = '#444444'


def unit_for(target: str) -> str:
    return 'C' if target == 'T_C' else 'kbar'


PANELS = [
    ('opx_liq',  'T_C',    '(a) Opx + Liquid  T (C)'),
    ('opx_liq',  'P_kbar', '(b) Opx + Liquid  P (kbar)'),
    ('opx_only', 'T_C',    '(c) Opx only  T (C)'),
    ('opx_only', 'P_kbar', '(d) Opx only  P (kbar)'),
]


def best_row_per_family(ms: pd.DataFrame, track: str, target: str) -> dict:
    """For each family, pick the feature_set with lowest mean RMSE."""
    sub = ms[(ms['track'] == track) & (ms['target'] == target)]
    out = {}
    for fam in MODEL_ORDER:
        fam_rows = sub[sub['model'] == fam]
        if len(fam_rows) == 0:
            out[fam] = None
            continue
        best = fam_rows.loc[fam_rows['mean'].idxmin()]
        out[fam] = best
    return out


def main():
    ms_tuned = pd.read_csv('results/opx_multiseed_summary.csv')
    ms_tab = pd.read_csv('results/tabpfn_multiseed_summary.csv')
    ms_all = pd.concat([ms_tuned, ms_tab], ignore_index=True)

    sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')
    sc_all = sc[sc['regime'] == 'ALL']

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    axes = axes.ravel()

    for ax, (track, target, title) in zip(axes, PANELS):
        best = best_row_per_family(ms_all, track, target)
        # Putirka as a 10th bar
        pu = sc_all[(sc_all['track'] == track) & (sc_all['target'] == target)]
        put_rmse = float(pu['best_external_rmse'].iloc[0]) if len(pu) else np.nan
        put_lo = float(pu['best_external_rmse_lo'].iloc[0]) if len(pu) else np.nan
        put_hi = float(pu['best_external_rmse_hi'].iloc[0]) if len(pu) else np.nan
        put_method = pu['best_external_method'].iloc[0] if len(pu) else ''

        labels = list(MODEL_ORDER) + ['Putirka']
        x = np.arange(len(labels))

        rmse = [
            best[f]['mean'] if best[f] is not None else np.nan
            for f in MODEL_ORDER
        ] + [put_rmse]
        rmse = np.array(rmse, dtype=float)

        ci_lo = [
            (best[f]['mean'] - 1.96 * best[f]['std'])
            if best[f] is not None else np.nan
            for f in MODEL_ORDER
        ] + [put_lo]
        ci_hi = [
            (best[f]['mean'] + 1.96 * best[f]['std'])
            if best[f] is not None else np.nan
            for f in MODEL_ORDER
        ] + [put_hi]
        ci_lo = np.array(ci_lo, dtype=float)
        ci_hi = np.array(ci_hi, dtype=float)
        err = np.vstack([
            np.nan_to_num(rmse - ci_lo, nan=0.0),
            np.nan_to_num(ci_hi - rmse, nan=0.0),
        ])

        # Under-bar tag: feature_set for ML families, eq number only for Putirka.
        if isinstance(put_method, str) and put_method.lower().startswith('putirka'):
            put_tag = put_method.split(' ', 1)[1]
        else:
            put_tag = put_method if put_method else ''
        fs_labels = [
            best[f]['feature_set'] if best[f] is not None else ''
            for f in MODEL_ORDER
        ] + [put_tag]
        colors = [MODEL_COLORS[f] for f in MODEL_ORDER] + [PUTIRKA_C]

        ax.bar(
            x, np.nan_to_num(rmse), yerr=err,
            capsize=4, color=colors, edgecolor='black', linewidth=0.5,
        )

        top_of_data = float(np.nanmax(np.nan_to_num(ci_hi, nan=0.0)))
        if top_of_data > 0:
            ax.set_ylim(0, top_of_data * 1.08)

        # overall winner (lowest RMSE across all 10 bars), rendered inline
        # with the panel title at the top-right. Margin = % improvement vs
        # Putirka; if Putirka is the winner, margin is negative.
        valid = np.where(np.isfinite(rmse))[0]
        if len(valid):
            win_idx = int(valid[np.argmin(rmse[valid])])
            put_val = rmse[-1]
            unit = unit_for(target)
            win_val = rmse[win_idx]
            # fs tag for winner: for an ML family it's the feature_set
            # (raw/alr/pwlr); for Putirka it's the equation tag.
            if win_idx == len(labels) - 1:
                win_fs = put_tag
            else:
                win_fs = fs_labels[win_idx]
            fs_txt = f' ({win_fs})' if win_fs else ''
            if win_idx == len(labels) - 1:
                ml_vals = rmse[:-1]
                ml_valid = np.where(np.isfinite(ml_vals))[0]
                if len(ml_valid):
                    best_ml = ml_vals[ml_valid][np.argmin(ml_vals[ml_valid])]
                    pct = (win_val - best_ml) / win_val * 100
                    margin_txt = f' ({pct:+.1f}% ML)'
                else:
                    margin_txt = ''
            else:
                if np.isfinite(put_val) and put_val > 0:
                    pct = (put_val - win_val) / put_val * 100
                    margin_txt = f' ({pct:+.1f}% vs Put)'
                else:
                    margin_txt = ''
            ax.text(
                1.0, 1.02,
                f"win: {labels[win_idx]}{fs_txt} {win_val:.2f}{unit}{margin_txt}",
                transform=ax.transAxes, va='bottom', ha='right',
                fontsize=8, fontweight='bold',
                color=colors[win_idx],
                bbox=dict(facecolor='white', edgecolor=colors[win_idx],
                          alpha=0.92, pad=2, boxstyle='round,pad=0.2'),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
        for xi, fs, col in zip(x, fs_labels, colors):
            if not fs:
                continue
            ax.text(xi, -0.13, fs,
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=8,
                    color=col, fontweight='bold')

        unit = 'C' if target == 'T_C' else 'kbar'
        ax.set_ylabel(f'RMSE ({unit})', fontsize=11)
        ax.set_title(title, fontsize=12, loc='left', fontweight='bold',
                     pad=8)
        ax.grid(axis='y', ls=':', alpha=0.4)

    fig.suptitle(
        'All 9 model families + Putirka / Agreda, overall test-set RMSE  '
        '(pre-correction)\n'
        'Benchmark: ExPetDB opx held-out test partition (citation-grouped '
        'split)\n'
        'Winner %% is vs Putirka; negative = Putirka wins.',
        fontsize=12, fontweight='bold', y=0.99,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.subplots_adjust(bottom=0.10, hspace=0.3)

    out_stem = OUT_DIR / 'Core_09b_fig_opx_overall_families'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 9b. Overall test-set RMSE per model family for all four '
        'opx combinations, aggregated across the full held-out partition '
        '(no regime split). Bars show the mean 20-seed RMSE (5-seed for '
        'TabPFN), whiskers are 95% intervals computed as mean +/- 1.96 * '
        'std over seeds; the feature-set winner (raw, alr, or pwlr) for '
        'each family is printed below the bar in the family color. The '
        'dashed horizontal line is the best Putirka / Agreda external '
        'benchmark from the ALL-row of the pre-registered scorecard '
        '(shaded band = 95% bootstrap CI on the benchmark); panel (c) '
        'opx-only T has no Putirka opx-only thermometer in Thermobar and '
        'is annotated accordingly. This figure answers "which family is '
        'best overall?"; the companion Core_09a answers the same question '
        'stratified by pressure regime.'
    )
    (OUT_DIR / 'Core_09b_fig_opx_overall_families.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
