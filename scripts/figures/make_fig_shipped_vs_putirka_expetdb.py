#!/usr/bin/env python3
"""Core_10c: Shipped (post-tuned winning) model vs Putirka on ExPetDB.

Per-regime RMSE, 4 panels (opx_liq T, opx_liq P, opx_only T, opx_only P).
Three bars per regime:
  (1) shipped family pre-correction (peach),
  (2) shipped family post-correction under v3 ship rule (blue),
  (3) Putirka / Agreda external benchmark (gray).

Unlike Core_10 (which picks the best among all nine ML families per
regime), Core_10c fixes the model to the single v3-shipped family per
(track, target) and shows its per-regime behavior. The shipped family +
feature set + winning form is printed in each panel.

95% CIs are the bootstrap whiskers stored in
``results/preregistered_scorecard_postcorrection_v3.csv``.
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
    sc = pd.read_csv('results/preregistered_scorecard_postcorrection_v3.csv')
    ship_v3 = _load_shipped_v3_opx()

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    axes = axes.ravel()

    for ax, (track, target, title) in zip(axes, PANELS):
        sub = sc[(sc['track'] == track) & (sc['target'] == target)].copy()
        sub = sub.set_index('regime').reindex(REGIME_ORDER)

        x = np.arange(len(REGIME_ORDER))
        width = 0.27

        pre_r  = sub['v10_pre_rmse'].to_numpy(dtype=float)
        pre_lo = sub['v10_pre_rmse_lo'].to_numpy(dtype=float)
        pre_hi = sub['v10_pre_rmse_hi'].to_numpy(dtype=float)

        post_r  = sub['v10_post_rmse'].to_numpy(dtype=float)
        post_lo = sub['v10_post_rmse_lo'].to_numpy(dtype=float)
        post_hi = sub['v10_post_rmse_hi'].to_numpy(dtype=float)

        put_r  = sub['best_external_rmse'].to_numpy(dtype=float)
        put_lo = sub['best_external_rmse_lo'].to_numpy(dtype=float)
        put_hi = sub['best_external_rmse_hi'].to_numpy(dtype=float)

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
                    'No Putirka / Agreda thermometer\navailable for opx-only '
                    'composition;\ngray bars intentionally empty.',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=9, color='0.25',
                    bbox=dict(facecolor='white', edgecolor='0.5',
                              alpha=0.92, pad=4))

        tick_labels = []
        for r, base in zip(REGIME_ORDER, REGIME_LABELS):
            n = sub.loc[r, 'n'] if r in sub.index else np.nan
            n_txt = f'\n(n={int(n)})' if pd.notna(n) else ''
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
        title='ExPetDB held-out test (citation-grouped); 95% CI whiskers '
              'from 20-seed bootstrap',
        title_fontsize=10,
    )
    fig.suptitle(
        'Shipped (post-tuned winning) model vs Putirka / Agreda, per '
        'regime\n(ExPetDB held-out test: opx_liq n=174, opx_only n=190)',
        fontsize=13, fontweight='bold', y=0.99,
    )
    plt.tight_layout(rect=(0, 0.07, 1, 0.96))
    plt.subplots_adjust(bottom=0.13, hspace=0.32)

    out_stem = OUT_DIR / 'Core_10c_fig_shipped_vs_putirka_expetdb'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 10c. Shipped (post-tuned winning) model versus Putirka / '
        'Agreda external benchmarks on the ExPetDB held-out test partition '
        '(opx_liq n=174, opx_only n=190), per pre-registered pressure '
        'regime plus the aggregate ALL column. Three bars per regime: (1) '
        'peach, the shipped family\'s pre-correction RMSE (raw base-model '
        'prediction); (2) blue, the shipped family\'s post-correction RMSE '
        'under the v3 tolerance ship rule (Amendment 2, 2026-04-20; '
        'tolerance envelope max(T_ABS, T_REL x pre_rmse) with T_ABS = 10 '
        'degC for T, 1 kbar for P, T_REL = 0.10); (3) gray, the best '
        'external Thermobar / Agreda benchmark from the head-to-head '
        'comparison. The shipped family + feature-set + winning bias-'
        'correction form is printed in the upper-right of each panel. '
        'Whiskers are 95% bootstrap confidence intervals from the 20-seed '
        'refit. Panel (c) opx-only T has no Putirka opx-only thermometer '
        'available in Thermobar and is annotated accordingly. This is the '
        'narrowed companion to Core_10 (best of all nine families); '
        'Core_10c answers "does the single model we actually ship beat '
        'Putirka at each regime on our internal held-out test?" Source: '
        'results/preregistered_scorecard_postcorrection_v3.csv, '
        'results/bias_correction_shipped_v3.csv.'
    )
    (OUT_DIR / 'Core_10c_fig_shipped_vs_putirka_expetdb.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
