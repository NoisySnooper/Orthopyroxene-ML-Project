#!/usr/bin/env python3
"""Core_10d: Shipped (post-tuned winning) model vs Putirka on ArcPL.

ArcPL companion to Core_10c. For each of opx_liq T, opx_liq P, opx_only
T, opx_only P, shows three bars per pre-registered pressure regime on
the reconstructed ArcPL opx holdout (n=197):

  (1) shipped family pre-correction (peach),
  (2) shipped family post-correction under v3 ship rule (blue),
  (3) Putirka 2008 eq28a / eq29a / eq29c (gray).

95% CIs via 2000 paired bootstraps. The opx-only T panel carries the
standard "no Putirka opx-only thermometer" annotation. Data source:
``results/arcpl_opx_corrected_per_regime.csv`` produced by
``scripts/external_eval/eval_arcpl_opx_corrected.py``.

This is the external-validation twin of Core_10c (internal held-out
test). Both pin the model to the single v3-shipped family per
(track, target) -- the one the paper actually ships -- rather than the
best-of-nine reported in Core_10 / Core_10b.
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

from scripts.figures._model_palette import OKABE_ITO  # noqa: E402
from scripts.figures._style import apply_pub_style  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRE_C = '#F4A582'
POST_C = OKABE_ITO['blue']
PUTIRKA_C = '#999999'

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


def _err(rmse: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.vstack([
        np.nan_to_num(rmse - lo, nan=0.0),
        np.nan_to_num(hi - rmse, nan=0.0),
    ])


def _load_shipped_v3_opx() -> pd.DataFrame:
    df = pd.read_csv('results/bias_correction_shipped_v3.csv')
    return df[(df.pipeline == 'opx') & (df.model != 'TabPFN')].copy()


def main():
    arc = pd.read_csv('results/arcpl_opx_corrected_per_regime.csv')
    ship_v3 = _load_shipped_v3_opx()

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    axes = axes.ravel()

    for ax, (track, target, title) in zip(axes, PANELS):
        sub = arc[(arc['track'] == track) & (arc['target'] == target)].copy()
        piv = sub.pivot_table(
            index='regime', columns='source',
            values=['rmse', 'rmse_lo', 'rmse_hi', 'n'], aggfunc='first'
        ).reindex(REGIME_ORDER)

        x = np.arange(len(REGIME_ORDER))
        width = 0.27

        def _col(metric, source):
            try:
                return piv[metric][source].to_numpy(dtype=float)
            except KeyError:
                return np.full(len(REGIME_ORDER), np.nan, dtype=float)

        pre_r  = _col('rmse',    'ml_pre')
        pre_lo = _col('rmse_lo', 'ml_pre')
        pre_hi = _col('rmse_hi', 'ml_pre')

        post_r  = _col('rmse',    'ml_post')
        post_lo = _col('rmse_lo', 'ml_post')
        post_hi = _col('rmse_hi', 'ml_post')

        put_r  = _col('rmse',    'putirka')
        put_lo = _col('rmse_lo', 'putirka')
        put_hi = _col('rmse_hi', 'putirka')

        n_pre  = _col('n', 'ml_pre')

        ax.bar(x - width, np.nan_to_num(pre_r), width=width,
               yerr=_err(pre_r, pre_lo, pre_hi), capsize=3,
               color=PRE_C, edgecolor='black', linewidth=0.5,
               label='Shipped family pre-correction')
        ax.bar(x, np.nan_to_num(post_r), width=width,
               yerr=_err(post_r, post_lo, post_hi), capsize=3,
               color=POST_C, edgecolor='black', linewidth=0.5,
               label='Shipped family post-correction (v3)')
        ax.bar(x + width, np.nan_to_num(put_r), width=width,
               yerr=_err(put_r, put_lo, put_hi), capsize=3,
               color=PUTIRKA_C, edgecolor='black', linewidth=0.5,
               label='Putirka / Agreda benchmark')

        top_of_data = np.nanmax(np.concatenate([
            np.where(np.isnan(pre_hi),  0, pre_hi),
            np.where(np.isnan(post_hi), 0, post_hi),
            np.where(np.isnan(put_hi),  0, put_hi),
        ])) if len(x) else 1.0
        if top_of_data > 0:
            ax.set_ylim(0, top_of_data * 1.08)

        if np.all(np.isnan(put_r)):
            ax.text(0.02, 0.98,
                    'No Putirka opx-only thermometer\navailable in Thermobar;\n'
                    'gray bars intentionally empty.',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=9, color='0.25',
                    bbox=dict(facecolor='white', edgecolor='0.5',
                              alpha=0.92, pad=4))

        tick_labels = []
        for r, base in zip(REGIME_ORDER, REGIME_LABELS):
            idx = REGIME_ORDER.index(r)
            n_txt = ''
            if pd.notna(n_pre[idx]):
                n_txt = f'\n(n={int(n_pre[idx])})'
            tick_labels.append(base + n_txt)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=9)

        unit = 'C' if target == 'T_C' else 'kbar'
        ax.set_ylabel(f'RMSE ({unit})', fontsize=11)
        ax.set_title(title, fontsize=12, loc='left', fontweight='bold',
                     pad=8)

        ship_row = ship_v3[(ship_v3.track == track)
                           & (ship_v3.target == target)]
        if not ship_row.empty:
            sr = ship_row.iloc[0]
            fam_txt = (f'shipped: {sr["model"]} / {sr["feature_set"]}  '
                       f'(Form {sr["winner_v3"]})')
            ax.set_title(fam_txt, fontsize=9, loc='right',
                         fontweight='bold', color=POST_C, pad=8)
        ax.grid(axis='y', ls=':', alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center', ncol=3,
        frameon=True, framealpha=0.95, edgecolor='0.6',
        bbox_to_anchor=(0.5, 0.01), fontsize=11,
        title='ArcPL opx external corpus (n=197); 95% CI from 2000 paired '
              'bootstraps',
        title_fontsize=10,
    )
    fig.suptitle(
        'Shipped (post-tuned winning) model vs Putirka / Agreda, per '
        'regime\n(ArcPL external holdout, n=197)',
        fontsize=13, fontweight='bold', y=0.99,
    )
    plt.tight_layout(rect=(0, 0.07, 1, 0.96))
    plt.subplots_adjust(bottom=0.13, hspace=0.32)

    out_stem = OUT_DIR / 'Core_10d_fig_shipped_vs_putirka_arcpl'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 10d. Shipped (post-tuned winning) model versus Putirka 2008 '
        'external benchmarks on the reconstructed ArcPL opx holdout (n=197, '
        'Agreda-Lopez 2024 experiments tagged "_notinLEPR" in LEPR Opx-Liq, '
        'filtered through the nb04 Part 3 pipeline: oxide-total 95-102 wt%, '
        'cation sum 3.95-4.05 on 6-O basis, Wo <= 5 mol%, P <= 100 kbar, '
        'Fe-Mg Kd window 0.23-0.35, citation-key deduplication against '
        'ExPetDB opx-liq). Three bars per regime: (1) peach, the shipped '
        'family\'s pre-correction RMSE (raw base-model prediction); (2) '
        'blue, the shipped family\'s post-correction RMSE under the v3 '
        'tolerance ship rule (Form A per-regime piecewise OLS for opx_liq '
        'P, opx_only T, opx_only P; Form B Agreda-Lopez piecewise sigmoid '
        'for opx_liq T); (3) gray, Putirka 2008 (eq28a for opx-liq T, '
        'eq29a for opx-liq P, eq29c for opx-only P). Panel (c) has no '
        'Putirka opx-only thermometer available in Thermobar. Whiskers are '
        '95% percentile intervals from 2000 paired bootstraps. The shipped '
        'family + feature-set + winning form is printed in the upper-right '
        'of each panel. Core_10d is the external-validation twin of '
        'Core_10c (ExPetDB internal test); both pin the model to the '
        'single v3-shipped family per (track, target), answering "does '
        'the model we actually ship beat Putirka on the external ArcPL '
        'holdout?" Source: results/arcpl_opx_corrected_per_regime.csv, '
        'results/bias_correction_shipped_v3.csv.'
    )
    (OUT_DIR / 'Core_10d_fig_shipped_vs_putirka_arcpl.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
