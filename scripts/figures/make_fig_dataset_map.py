#!/usr/bin/env python3
"""Core_01: Dataset P-T map across all 4 (pyroxene, pipeline) tracks.

2x2 grid: opx_liq, opx_only, cpx_liq, cpx_only. Points colored by
pre-registered P regime. Train vs test shown via marker alpha.
Marginal histograms on T and P axes. Horizontal dashed lines at
regime boundaries (5, 15, 30 kbar).

Source: data/processed/*_clean_*.parquet; data/splits/*_indices_*.npy.
Caption: regime edges verbatim from p_regime_preregistration.md.
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

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered P regime edges (kbar). Verbatim from
# docs/preregistration/p_regime_preregistration.md.
REGIME_EDGES = [0, 5, 15, 30, 100]
REGIME_NAMES = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle']
REGIME_COLORS = {
    'shallow_crustal':     OKABE_ITO['sky_blue'],
    'deep_crustal_MASH':   OKABE_ITO['green'],
    'lithospheric_mantle': OKABE_ITO['orange'],
    'deeper_mantle':       OKABE_ITO['vermillion'],
}

PANELS = [
    ('opx_liq',  'opx_clean_opx_liq',  'opx_liq',  'Opx + Liquid'),
    ('opx_only', 'opx_clean_opx_only', 'opx',      'Opx only'),
    ('cpx_liq',  'cpx_clean_cpx_liq',  'cpx_liq',  'Cpx + Liquid'),
    ('cpx_only', 'cpx_clean_cpx_only', 'cpx_only', 'Cpx only'),
]


def regime_for(p_kbar: float) -> str:
    for lo, hi, name in zip(REGIME_EDGES[:-1], REGIME_EDGES[1:], REGIME_NAMES):
        if lo <= p_kbar < hi:
            return name
    return REGIME_NAMES[-1]


def load_track(parquet_name: str, split_tag: str):
    df = pd.read_parquet(f'data/processed/{parquet_name}.parquet').copy()
    # Map each row to train/test via splits index arrays when available.
    train_idx = Path(f'data/splits/train_indices_{split_tag}.npy')
    test_idx = Path(f'data/splits/test_indices_{split_tag}.npy')
    if train_idx.exists() and test_idx.exists():
        tr = set(np.load(train_idx).tolist())
        te = set(np.load(test_idx).tolist())
        df['split'] = df.index.map(
            lambda i: 'train' if i in tr else ('test' if i in te else 'unassigned'))
    else:
        df['split'] = 'train'
    df['regime'] = df['P_kbar'].map(regime_for)
    return df


def main():
    fig = plt.figure(figsize=(15, 12), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.30)

    for (track, pq_name, split_tag, title), cell in zip(PANELS, outer):
        df = load_track(pq_name, split_tag)
        inner = cell.subgridspec(2, 2, width_ratios=(4, 1),
                                 height_ratios=(1, 4),
                                 hspace=0.05, wspace=0.05)
        ax_scatter = fig.add_subplot(inner[1, 0])
        ax_histx   = fig.add_subplot(inner[0, 0], sharex=ax_scatter)
        ax_histy   = fig.add_subplot(inner[1, 1], sharey=ax_scatter)

        for regime in REGIME_NAMES:
            sub = df[df['regime'] == regime]
            for split, alpha, edge in [('train', 0.35, 'none'),
                                       ('test',  0.95, 'black')]:
                sel = sub[sub['split'] == split]
                if len(sel) == 0:
                    continue
                ax_scatter.scatter(
                    sel['T_C'], sel['P_kbar'],
                    s=14, c=REGIME_COLORS[regime],
                    edgecolor=edge, linewidths=0.3,
                    alpha=alpha, zorder=2 if split == 'test' else 1,
                )

        # Regime boundary lines with label at right edge
        t_max = df['T_C'].max()
        for edge in REGIME_EDGES[1:-1]:
            ax_scatter.axhline(edge, color='0.4', ls='--', lw=0.6, zorder=3)
            ax_scatter.text(t_max, edge, f' {edge} kbar',
                            fontsize=7, color='0.3', va='center', ha='left')

        ax_scatter.set_xlabel('T (\u00b0C)')
        ax_scatter.set_ylabel('P (kbar)')

        # Prominent track-name title above the joint plot
        ax_histx.set_title(
            f'{title}  |  track = {track}',
            fontsize=13, fontweight='bold', loc='left', pad=6,
        )

        # Supplementary stats block (top-right of scatter)
        n_total = len(df)
        n_train = int((df['split'] == 'train').sum())
        n_test  = int((df['split'] == 'test').sum())
        n_cit   = int(df['Citation'].nunique()) if 'Citation' in df.columns else 0
        t_lo, t_hi = df['T_C'].quantile([0.01, 0.99])
        p_lo, p_hi = df['P_kbar'].quantile([0.01, 0.99])
        stats_lines = [
            f'n total  = {n_total}',
            f'n train  = {n_train}',
            f'n test   = {n_test}',
            f'n citations = {n_cit}',
            f'T range  = {t_lo:.0f}-{t_hi:.0f} \u00b0C',
            f'P range  = {p_lo:.1f}-{p_hi:.1f} kbar',
        ]
        ax_scatter.text(
            0.98, 0.98, '\n'.join(stats_lines),
            transform=ax_scatter.transAxes, va='top', ha='right',
            fontsize=8, family='monospace',
            bbox=dict(facecolor='white', edgecolor='0.7',
                      alpha=0.92, pad=3))

        # Per-regime counts block (bottom-left of scatter) with full names
        regime_lines = ['regime counts:']
        for r in REGIME_NAMES:
            cnt = int((df['regime'] == r).sum())
            regime_lines.append(f'  {r:<20s} {cnt:>4d}')
        ax_scatter.text(
            0.02, 0.02, '\n'.join(regime_lines),
            transform=ax_scatter.transAxes, va='bottom', ha='left',
            fontsize=7.5, family='monospace',
            bbox=dict(facecolor='white', edgecolor='0.7',
                      alpha=0.92, pad=3))

        # Marginal histograms
        ax_histx.hist(df['T_C'].dropna(), bins=40, color='0.6',
                      edgecolor='none')
        ax_histx.set_ylabel('count', fontsize=8)
        ax_histx.tick_params(labelbottom=False)
        ax_histy.hist(df['P_kbar'].dropna(), bins=40, color='0.6',
                      edgecolor='none', orientation='horizontal')
        ax_histy.set_xlabel('count', fontsize=8)
        ax_histy.tick_params(labelleft=False)

    # Shared legend
    handles = []
    for r in REGIME_NAMES:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=REGIME_COLORS[r],
                                   markersize=8, label=r))
    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='0.6', markeredgecolor='black',
                               markersize=8, label='test (dark edge)'))
    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='0.6', alpha=0.35,
                               markersize=8, label='train (faded)'))
    fig.legend(handles=handles, loc='lower center', ncol=6,
               frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(
        'Dataset P-T distribution per pyroxene track, colored by '
        'pre-registered regime',
        fontsize=12, y=0.995,
    )

    out_stem = OUT_DIR / 'Core_01_fig_dataset_map'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 1. Dataset map. P-T distribution of all experiments in the '
        'ExPetDB 2025-07-21 export, partitioned into four pyroxene tracks '
        '(opx-liq, opx-only, cpx-liq, cpx-only). Points are colored by '
        'pre-registered pressure regime (shallow_crustal <5 kbar, '
        'deep_crustal_MASH 5-15 kbar, lithospheric_mantle 15-30 kbar, '
        'deeper_mantle >=30 kbar); regime edges were locked on 2026-04-17 '
        'before any correction fitting. Dashed horizontal lines mark regime '
        'boundaries (labeled at right axis). Marker opacity distinguishes '
        'the 80/20 citation-grouped train/test split: train points faded, '
        'test points with dark edges. Marginal histograms show the '
        'univariate T and P coverage. Each panel is titled with its track '
        'name and displays a supplementary-information block (top-right) '
        'with n_total, n_train, n_test, citation count, and 1st-99th '
        'percentile T and P ranges. Per-regime counts are shown bottom-left '
        'with full regime names.'
    )
    (OUT_DIR / 'Core_01_fig_dataset_map.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
