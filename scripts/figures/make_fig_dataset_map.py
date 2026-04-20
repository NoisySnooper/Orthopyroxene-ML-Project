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
    fig = plt.figure(figsize=(14, 11), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

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

        # Regime boundary lines
        for edge in REGIME_EDGES[1:-1]:
            ax_scatter.axhline(edge, color='0.4', ls='--', lw=0.6, zorder=3)

        ax_scatter.set_xlabel('T (\u00b0C)')
        ax_scatter.set_ylabel('P (kbar)')
        ax_scatter.set_title(f'{title} (n={len(df)})', fontsize=11,
                             loc='left', pad=38)

        # Marginal histograms
        ax_histx.hist(df['T_C'].dropna(), bins=40, color='0.6',
                      edgecolor='none')
        ax_histx.set_ylabel('count', fontsize=8)
        ax_histx.tick_params(labelbottom=False)
        ax_histy.hist(df['P_kbar'].dropna(), bins=40, color='0.6',
                      edgecolor='none', orientation='horizontal')
        ax_histy.set_xlabel('count', fontsize=8)
        ax_histy.tick_params(labelleft=False)

        # Per-regime count annotation
        counts_str = ' | '.join(
            f'{r.split("_")[0][:3]}:{(df["regime"]==r).sum()}'
            for r in REGIME_NAMES)
        ax_scatter.text(0.02, 0.98, counts_str,
                        transform=ax_scatter.transAxes,
                        va='top', ha='left', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='0.7',
                                  alpha=0.85, pad=2))

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
        'ExPetDB 2025-07-21 export, partitioned into four pyroxene tracks: '
        'opx-liq (n=600), opx-only (n=1035), cpx-liq (n=2385), and cpx-only '
        '(n=2897). Points are colored by pre-registered pressure regime '
        '(shallow_crustal <5 kbar, deep_crustal_MASH 5-15 kbar, '
        'lithospheric_mantle 15-30 kbar, deeper_mantle >=30 kbar); regime '
        'edges were locked on 2026-04-17 before any correction fitting. '
        'Dashed horizontal lines mark regime boundaries. Marker opacity '
        'distinguishes the 80/20 citation-grouped train/test split: train '
        'points faded, test points with dark edges. Marginal histograms show '
        'the univariate T and P coverage. Per-regime counts shown top-left '
        'of each panel.'
    )
    (OUT_DIR / 'Core_01_fig_dataset_map.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
