#!/usr/bin/env python3
"""main_fig_1: ExPetDB P-T coverage for opx-liq and opx-only tracks.

Two side-by-side panels (opx_liq, opx_only). Points colored by
pre-registered P regime; train/test split shown via marker alpha and
edge. Marginal T and P histograms + per-panel summary stats.

Source: data/processed/opx_clean_*.parquet, data/splits/*_indices_*.npy
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

from scripts.figures._model_palette import REGIME_COLORS  # noqa: E402
from scripts.figures._style import apply_pub_style, resolve_out_dir, jgr_figsize, jgr_top, jgr_bottom # noqa: E402
from scripts.figures._labels import REGIME_ORDER, TRACK_LABEL  # noqa: E402
from scripts.figures._legend import add_below_legend  # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

REGIME_EDGES = [0, 5, 15, 30, 100]

PANELS = [
    ('opx_liq',  'opx_clean_opx_liq',  'opx_liq',  'a'),
    ('opx_only', 'opx_clean_opx_only', 'opx',      'b'),
]


def regime_for(p_kbar: float) -> str:
    for lo, hi, name in zip(REGIME_EDGES[:-1], REGIME_EDGES[1:],
                            REGIME_ORDER):
        if lo <= p_kbar < hi:
            return name
    return REGIME_ORDER[-1]


def load_track(parquet_name: str, split_tag: str):
    df = pd.read_parquet(f'data/processed/{parquet_name}.parquet').copy()
    train_idx = Path(f'data/splits/train_indices_{split_tag}.npy')
    test_idx = Path(f'data/splits/test_indices_{split_tag}.npy')
    if train_idx.exists() and test_idx.exists():
        tr = set(np.load(train_idx).tolist())
        te = set(np.load(test_idx).tolist())
        df['split'] = df.index.map(
            lambda i: 'train' if i in tr
            else ('test' if i in te else 'unassigned'))
    else:
        df['split'] = 'train'
    df['regime'] = df['P_kbar'].map(regime_for)
    return df


def draw_panel(fig, outer_cell, df, track, idx, *, t_lim, p_lim,
               t_bins, p_bins):
    inner = outer_cell.subgridspec(2, 1, height_ratios=(5, 1.6), hspace=0.18)
    plot_area = inner[0].subgridspec(2, 2, width_ratios=(4, 1),
                                     height_ratios=(1, 4),
                                     hspace=0.05, wspace=0.05)
    ax_scatter = fig.add_subplot(plot_area[1, 0])
    ax_histx = fig.add_subplot(plot_area[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(plot_area[1, 1], sharey=ax_scatter)
    ax_stats = fig.add_subplot(inner[1])
    ax_stats.axis('off')
    ax_scatter.set_xlim(t_lim)
    ax_scatter.set_ylim(p_lim)

    for regime in REGIME_ORDER:
        sub = df[df['regime'] == regime]
        for split, alpha, edge in [('train', 0.35, 'none'),
                                   ('test', 0.95, '#222222')]:
            sel = sub[sub['split'] == split]
            if len(sel) == 0:
                continue
            ax_scatter.scatter(
                sel['T_C'], sel['P_kbar'],
                s=14, c=REGIME_COLORS[regime],
                edgecolor=edge, linewidths=0.3,
                alpha=alpha, zorder=2 if split == 'test' else 1,
            )

    for edge in REGIME_EDGES[1:-1]:
        ax_scatter.axhline(edge, color='#666666', ls='--', lw=0.6, zorder=3)
        ax_scatter.text(
            0.985, edge, f'{edge} kbar',
            transform=ax_scatter.get_yaxis_transform(),
            fontsize=7, color='#444444', va='bottom', ha='right',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.7, pad=1),
            zorder=4,
        )

    ax_scatter.set_xlabel('T (°C)')
    ax_scatter.set_ylabel('P (kbar)')

    ax_histx.set_title(f'({idx}) {TRACK_LABEL[track]}')
    ax_histx.hist(df['T_C'].dropna(), bins=t_bins, color='#bbbbbb',
                  edgecolor='none')
    ax_histx.set_ylabel('count')
    ax_histx.tick_params(labelbottom=False)
    ax_histy.hist(df['P_kbar'].dropna(), bins=p_bins, color='#bbbbbb',
                  edgecolor='none', orientation='horizontal')
    ax_histy.set_xlabel('count')
    ax_histy.tick_params(labelleft=False)

    n_total = len(df)
    n_train = int((df['split'] == 'train').sum())
    n_test = int((df['split'] == 'test').sum())
    n_cit = (int(df['Citation'].nunique())
             if 'Citation' in df.columns else 0)
    t_lo, t_hi = df['T_C'].quantile([0.01, 0.99])
    p_lo, p_hi = df['P_kbar'].quantile([0.01, 0.99])

    left_col = [
        f"n total     = {n_total}",
        f"n train     = {n_train}",
        f"n test      = {n_test}",
        f"n citations = {n_cit}",
        f"T (1–99%)   = {t_lo:.0f}–{t_hi:.0f} °C",
        f"P (1–99%)   = {p_lo:.1f}–{p_hi:.1f} kbar",
    ]
    right_col = ['regime counts:']
    for r in REGIME_ORDER:
        cnt = int((df['regime'] == r).sum())
        right_col.append(f"  {r:<20s} {cnt:>4d}")

    ax_stats.text(0.02, 0.98, '\n'.join(left_col),
                  transform=ax_stats.transAxes, fontsize=8,
                  family='monospace', va='top', ha='left')
    ax_stats.text(0.52, 0.98, '\n'.join(right_col),
                  transform=ax_stats.transAxes, fontsize=8,
                  family='monospace', va='top', ha='left')


def main():
    # Each cell holds scatter + two marginal histograms + a stats block;
    # at print width we need a wider canvas so the stats block doesn't
    # collide with the marginal counts and the suptitle clears the
    # panel headers. Bottom margin is tight so the legend sits flush
    # under the stats blocks.
    # Pre-load both panels so we can compute the union of T/P ranges and
    # use a single axis scale across panels (a) and (b). Histogram bins
    # are also shared so bar widths read identically across panels.
    panel_dfs = []
    for track, pq_name, split_tag, idx in PANELS:
        panel_dfs.append((track, idx, load_track(pq_name, split_tag)))

    all_T = pd.concat([d['T_C'] for _, _, d in panel_dfs]).dropna()
    all_P = pd.concat([d['P_kbar'] for _, _, d in panel_dfs]).dropna()
    t_pad = 0.04 * (float(all_T.max()) - float(all_T.min()))
    p_pad = 0.04 * (float(all_P.max()) - 0.0)
    t_lim = (float(all_T.min()) - t_pad, float(all_T.max()) + t_pad)
    p_lim = (0.0 - p_pad, float(all_P.max()) + p_pad)
    t_bins = np.linspace(t_lim[0], t_lim[1], 41)
    p_bins = np.linspace(p_lim[0], p_lim[1], 41)

    fig = plt.figure(figsize=jgr_figsize((11, 6.5)), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, wspace=0.30,
                             top=jgr_top(0.91), bottom=jgr_bottom(0.10),
                             left=0.07, right=0.97)

    for (track, idx, df), cell in zip(panel_dfs, outer):
        draw_panel(fig, cell, df, track, idx,
                   t_lim=t_lim, p_lim=p_lim, t_bins=t_bins, p_bins=p_bins)

    # Regime legend (with pressure ranges) + train/test marker key.
    regime_legend_label = {
        'shallow_crustal':     'shallow_crustal (<5 kbar)',
        'deep_crustal_MASH':   'deep_crustal_MASH (5–15 kbar)',
        'lithospheric_mantle': 'lithospheric_mantle (15–30 kbar)',
        'deeper_mantle':       'deeper_mantle (≥30 kbar)',
    }
    handles = []
    for r in REGIME_ORDER:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=REGIME_COLORS[r],
                                  markersize=7,
                                  label=regime_legend_label[r]))
    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='#bbbbbb',
                              markeredgecolor='#222222',
                              markersize=7, label='test (dark edge)'))
    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='#bbbbbb', alpha=0.35,
                              markersize=7, label='train (faded)'))
    fig.legend(handles, [h.get_label() for h in handles],
               loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=3)

    fig.suptitle('ExPetDB 2025-07-21: opx P–T distribution')

    stem = OUT_DIR / 'main_fig_1'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{stem}.png', bbox_inches='tight')

    caption = (
        'Figure 1. ExPetDB experimental coverage for the two opx pipelines, '
        'colored by pre-registered pressure regime. Panel (a) opx + liquid '
        '(n = 600, 93 citations); panel (b) opx only (n = 1035, 123 '
        'citations). Pre-registered regime edges are shallow_crustal <5 '
        'kbar, deep_crustal_MASH 5–15 kbar, lithospheric_mantle 15–30 kbar, '
        'deeper_mantle ≥30 kbar (locked 2026-04-17 before any correction '
        'fitting). Dashed horizontal lines mark the regime boundaries. '
        'Marker opacity distinguishes the 80/20 citation-grouped train/'
        'test split: train points faded, test points with dark edges. '
        'Marginal histograms show univariate T and P coverage. The summary '
        'block below each scatter reports n_total, n_train, n_test, '
        'citation count, 1st–99th percentile ranges, and per-regime counts.'
    )
    (OUT_DIR / 'main_fig_1.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
