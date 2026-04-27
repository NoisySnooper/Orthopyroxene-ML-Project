#!/usr/bin/env python3
"""Core_10: Best-of-our-models vs Putirka (per-regime RMSE, 4 panels).

For each of opx_liq T, opx_liq P, opx_only T, opx_only P, show two bars
per regime: (1) the best among our 9 pre-correction families for that
regime, labelled with the winning model family on top of the bar;
(2) Putirka/Agreda external benchmark. 95% CIs as whiskers. Panels
without a Putirka benchmark (opx_only T) carry an explicit annotation.
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

OURS_C = '#F4A582'
OURS_POST_C = OKABE_ITO['blue']
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


def family_and_fs(method: str) -> tuple[str, str]:
    """Split 'ERT/pwlr' -> ('ERT', 'pwlr'); '' -> ('', '')."""
    if not isinstance(method, str) or method == '':
        return ('', '')
    parts = method.split('/', 1)
    fam = parts[0]
    fs = parts[1] if len(parts) > 1 else ''
    return (fam, fs)


def best_of_ours(row):
    """Return (rmse, lo, hi, label, fs) for best of v10_pre and tabpfn_pre.

    label is family name; fs is the feature-set transform (raw / alr /
    pwlr for v10 families, empty for TabPFN which has no fs choice).
    """
    candidates = []
    v = row.get('v10_pre_rmse')
    if pd.notna(v):
        fam, fs = family_and_fs(row.get('v10_pre_method', ''))
        candidates.append((
            v, row.get('v10_pre_rmse_lo'), row.get('v10_pre_rmse_hi'),
            fam, fs))
    t = row.get('tabpfn_rmse')
    if pd.notna(t):
        candidates.append((
            t, row.get('tabpfn_rmse_lo'), row.get('tabpfn_rmse_hi'),
            'TabPFN', ''))
    if not candidates:
        return (np.nan, np.nan, np.nan, '', '')
    return min(candidates, key=lambda x: x[0])


def best_of_ours_post(row):
    """Return (rmse, lo, hi, label, fs) for best post-correction candidate.

    Compares v10_post and tabpfn_post. Falls back to pre-correction RMSE
    for families whose correction did not ship (post == pre).
    """
    candidates = []
    v = row.get('v10_post_rmse')
    if pd.notna(v):
        fam, fs = family_and_fs(row.get('v10_post_method', ''))
        candidates.append((
            v, row.get('v10_post_rmse_lo'), row.get('v10_post_rmse_hi'),
            fam, fs))
    t = row.get('tabpfn_post_rmse')
    if pd.notna(t):
        candidates.append((
            t, row.get('tabpfn_post_rmse_lo'), row.get('tabpfn_post_rmse_hi'),
            'TabPFN', ''))
    if not candidates:
        return (np.nan, np.nan, np.nan, '', '')
    return min(candidates, key=lambda x: x[0])


def main():
    # Prefer v3 scorecard (tolerance-based ship rule, Amendment 2
    # 2026-04-20). Fall back to v1 if v3 is missing.
    sc_v3_path = Path('results/preregistered_scorecard_postcorrection_v3.csv')
    if sc_v3_path.exists():
        sc = pd.read_csv(sc_v3_path)
    else:
        sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    axes = axes.ravel()

    for ax, (track, target, title) in zip(axes, PANELS):
        sub = sc[(sc['track'] == track) & (sc['target'] == target)].copy()
        sub = sub.set_index('regime').reindex(REGIME_ORDER)

        x = np.arange(len(REGIME_ORDER))
        width = 0.27

        ours = [best_of_ours(sub.loc[r]) for r in REGIME_ORDER]
        ours_rmse = np.array([o[0] for o in ours], dtype=float)
        ours_lo   = np.array([o[1] for o in ours], dtype=float)
        ours_hi   = np.array([o[2] for o in ours], dtype=float)
        ours_lbl  = [o[3] for o in ours]
        ours_fs   = [o[4] for o in ours]

        ours_post = [best_of_ours_post(sub.loc[r]) for r in REGIME_ORDER]
        post_rmse = np.array([o[0] for o in ours_post], dtype=float)
        post_lo   = np.array([o[1] for o in ours_post], dtype=float)
        post_hi   = np.array([o[2] for o in ours_post], dtype=float)
        post_lbl  = [o[3] for o in ours_post]
        post_fs   = [o[4] for o in ours_post]

        put_rmse = sub['best_external_rmse'].values.astype(float)
        put_lo   = sub['best_external_rmse_lo'].values.astype(float)
        put_hi   = sub['best_external_rmse_hi'].values.astype(float)

        ours_err = np.vstack([
            np.nan_to_num(ours_rmse - ours_lo, nan=0.0),
            np.nan_to_num(ours_hi - ours_rmse, nan=0.0),
        ])
        post_err = np.vstack([
            np.nan_to_num(post_rmse - post_lo, nan=0.0),
            np.nan_to_num(post_hi - post_rmse, nan=0.0),
        ])
        put_err = np.vstack([
            np.nan_to_num(put_rmse - put_lo, nan=0.0),
            np.nan_to_num(put_hi - put_rmse, nan=0.0),
        ])

        ax.bar(x - width, np.nan_to_num(ours_rmse), width=width,
               yerr=ours_err, capsize=3,
               color=OURS_C, edgecolor='black', linewidth=0.5,
               label='Best of ours pre-correction')
        ax.bar(x, np.nan_to_num(post_rmse), width=width,
               yerr=post_err, capsize=3,
               color=OURS_POST_C, edgecolor='black', linewidth=0.5,
               label='Best of ours post-correction (v3)')
        ax.bar(x + width, np.nan_to_num(put_rmse), width=width,
               yerr=put_err, capsize=3,
               color=PUTIRKA_C, edgecolor='black', linewidth=0.5,
               label='Putirka / Agreda benchmark')

        top_of_data = np.nanmax(np.concatenate([
            np.where(np.isnan(ours_hi), 0, ours_hi),
            np.where(np.isnan(post_hi), 0, post_hi),
            np.where(np.isnan(put_hi),  0, put_hi),
        ])) if len(x) else 1.0
        if top_of_data > 0:
            ax.set_ylim(0, top_of_data * 1.08)

        # Panels with no external benchmark (opx_only T)
        if np.all(np.isnan(put_rmse)):
            ax.text(0.02, 0.98,
                    'No Putirka / Agreda thermometer\navailable for opx-only '
                    'composition;\ngray bars intentionally empty.',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=9, color='0.25',
                    bbox=dict(facecolor='white', edgecolor='0.5',
                              alpha=0.92, pad=4))

        tick_labels_clean = []
        for r, base in zip(REGIME_ORDER, REGIME_LABELS):
            n = sub.loc[r, 'n'] if r in sub.index else np.nan
            n_txt = f'\n(n={int(n)})' if pd.notna(n) else ''
            tick_labels_clean.append(base + n_txt)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels_clean, fontsize=9)

        # Stack pre-winner (higher) over post-winner (lower) centered on
        # each regime group so the labels do not horizontally collide
        # with each other across the pre/post bar columns.
        for xi, lbl, fs in zip(x, ours_lbl, ours_fs):
            if not lbl:
                continue
            fs_line = f' ({fs})' if fs else ''
            ax.text(xi - width / 2, -0.13, f'pre: {lbl}{fs_line}',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=7.5,
                    fontweight='bold', color=OURS_C)
        for xi, lbl, fs in zip(x, post_lbl, post_fs):
            if not lbl:
                continue
            fs_line = f' ({fs})' if fs else ''
            ax.text(xi - width / 2, -0.20, f'post: {lbl}{fs_line}',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=7.5,
                    fontweight='bold', color=OURS_POST_C)

        unit = 'C' if target == 'T_C' else 'kbar'
        ax.set_ylabel(f'RMSE ({unit})', fontsize=11)
        ax.set_title(title, fontsize=12, loc='left', fontweight='bold',
                     pad=8)
        ax.grid(axis='y', ls=':', alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center', ncol=3,
        frameon=True, framealpha=0.95, edgecolor='0.6',
        bbox_to_anchor=(0.5, 0.01), fontsize=11,
        title='ExPetDB held-out test (citation-grouped); 95% CI whiskers '
              'from 20-seed bootstrap (winning family labeled below bars)',
        title_fontsize=10,
    )
    fig.suptitle(
        'Best of ours vs Putirka / Agreda, per regime\n'
        '(ExPetDB held-out test: opx_liq n=174, opx_only n=190)',
        fontsize=13, fontweight='bold', y=0.99,
    )
    plt.tight_layout(rect=(0, 0.07, 1, 0.96))
    plt.subplots_adjust(bottom=0.15, hspace=0.42)

    out_stem = OUT_DIR / 'Core_10_fig_best_vs_putirka'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 10. Best of our nine model families versus Putirka / Agreda '
        'external benchmarks, per pre-registered pressure regime and '
        'aggregate ALL column, for all four opx combinations. Three bars per '
        'regime: (1) sky blue, best pre-correction RMSE across ElasticNet, '
        'RF, ERT, GB, XGB, LightGBM, CatBoost, MLP, and TabPFN; (2) blue, '
        'best post-correction RMSE across the same nine families using the '
        'shipped bias-correction form; (3) gray, best external '
        'Thermobar / Agreda benchmark from the head-to-head comparison. The '
        'family whose model won that regime is labelled below its bar '
        'together with the input transform ("raw" = native oxide wt%, "alr" '
        '= additive log-ratio, "pwlr" = pairwise log-ratio); TabPFN has no '
        'feature-set choice. Whiskers are 95% bootstrap confidence intervals '
        'from the 20-seed refit (5 seeds for TabPFN). Panel (c) opx-only T '
        'has no Putirka opx-only thermometer available in Thermobar and is '
        'annotated accordingly. This is a simplified companion to Core_09 '
        '(all 9 families shown individually); Core_10 answers the narrower '
        'question "does our best ML candidate beat the best published '
        'calibration at each regime, before and after bias correction?" at '
        'a glance.'
    )
    (OUT_DIR / 'Core_10_fig_best_vs_putirka.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
