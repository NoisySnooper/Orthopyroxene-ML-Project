#!/usr/bin/env python3
"""Core_01b: ExPetDB opx corpora vs ArcPL opx external holdout.

Top row (ExPetDB, train + internal test): P-T scatter colored by pre-
registered pressure regime with Core_01's train/test marker alpha
pattern (train faded, test dark-edge).
  (a) ExPetDB opx_liq
  (b) ExPetDB opx_only

Bottom row (ArcPL external held-out, no train/test split): reconstructed
from LEPR Opx-Liq sheet filtered to Citation_x `_notinLEPR` and cleaned
by the nb04 Part 3 pipeline (n ~= 197). Colored by regime, with the
matching ExPetDB corpus (train + test) underlaid in gray so the reader
can see how the ArcPL distribution sits inside the training space.
  (c) ArcPL opx_liq held-out over ExPetDB opx_liq gray underlay
  (d) ArcPL opx_only held-out over ExPetDB opx_only gray underlay
      (same underlying ArcPL corpus as (c); the ArcPL opx-only
      evaluation track uses the same rows but is predicted from opx
      composition alone.)

All four scatter axes share a single (T, P) axis range scaled to the
union of all four data sources, so regime position is visually
comparable across panels.

Source:
  data/processed/opx_clean_opx_liq.parquet
  data/processed/opx_clean_opx_only.parquet
  data/raw/external/LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx
    (Opx-Liq sheet, Citation_x contains `_notinLEPR`)
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
from src.external.arcpl_opx import load_arcpl_opx_liq  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_EDGES = [0, 5, 15, 30, 100]
REGIME_NAMES = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle']
REGIME_COLORS = {
    'shallow_crustal':     OKABE_ITO['sky_blue'],
    'deep_crustal_MASH':   OKABE_ITO['green'],
    'lithospheric_mantle': OKABE_ITO['orange'],
    'deeper_mantle':       OKABE_ITO['vermillion'],
}


def regime_for(p_kbar: float) -> str:
    for lo, hi, name in zip(REGIME_EDGES[:-1], REGIME_EDGES[1:], REGIME_NAMES):
        if lo <= p_kbar < hi:
            return name
    return REGIME_NAMES[-1]


def load_expetdb(parquet_name: str, split_tag: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(f'data/processed/{parquet_name}.parquet').copy()
    df['regime'] = df['P_kbar'].map(regime_for)
    if split_tag is not None:
        tr_path = Path(f'data/splits/train_indices_{split_tag}.npy')
        te_path = Path(f'data/splits/test_indices_{split_tag}.npy')
        if tr_path.exists() and te_path.exists():
            tr = set(np.load(tr_path).tolist())
            te = set(np.load(te_path).tolist())
            df['split'] = df.index.map(
                lambda i: 'train' if i in tr else ('test' if i in te else 'unassigned'))
        else:
            df['split'] = 'train'
    else:
        df['split'] = 'holdout'
    return df


def _nice_range(vmin: float, vmax: float, pad: float = 0.04) -> tuple[float, float]:
    span = vmax - vmin
    return (vmin - pad * span, vmax + pad * span)


def scatter_with_marginals(fig, gs_cell, df, title,
                           *, t_lim: tuple[float, float],
                           p_lim: tuple[float, float],
                           t_bins: np.ndarray,
                           p_bins: np.ndarray,
                           t_count_max: float,
                           p_count_max: float,
                           color_by_regime: bool = True,
                           overlay_hull: pd.DataFrame | None = None,
                           arcpl_color: str | None = None,
                           stats_kind: str = 'expetdb'):
    """Render a P-T scatter + marginal histograms + stats block inside gs_cell.

    The scatter shares the gridspec cell with its marginal histograms so
    the top/side histograms run the full length of the scatter axes.
    Below the scatter, a per-panel 2-column summary mirrors Core_01
    (n_total, n_train/n_test or holdout flag, citations, T/P 1-99%
    percentile ranges, and per-regime counts).
    """
    outer_inner = gs_cell.subgridspec(
        2, 1, height_ratios=(5, 1.6), hspace=0.18,
    )
    plot_area = outer_inner[0].subgridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        hspace=0.05, wspace=0.05,
    )
    ax_scatter = fig.add_subplot(plot_area[1, 0])
    ax_histx = fig.add_subplot(plot_area[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(plot_area[1, 1], sharey=ax_scatter)
    ax_stats = fig.add_subplot(outer_inner[1])
    ax_stats.axis('off')

    if overlay_hull is not None:
        ex_T = overlay_hull['T_C'].dropna()
        ex_P = overlay_hull['P_kbar'].dropna()
        ax_scatter.scatter(
            ex_T, ex_P, s=8, c='0.50', alpha=0.45,
            edgecolor='none', zorder=0,
            label='ExPetDB (train+test, gray underlay)',
        )

    if color_by_regime:
        # For ExPetDB panels (split column contains 'train' and 'test'),
        # adopt Core_01's alpha pattern: train faded, test dark-edge on
        # top. For ArcPL panels the split is uniformly 'holdout' so we
        # fall through to a single pass with the default styling.
        has_split = (stats_kind == 'expetdb'
                     and 'split' in df.columns
                     and set(df['split'].unique()) & {'train', 'test'})
        for regime in REGIME_NAMES:
            sub = df[df['regime'] == regime]
            if len(sub) == 0:
                continue
            if has_split:
                for split, alpha, edge, zorder in [
                        ('train', 0.35, 'none',  1),
                        ('test',  0.95, 'black', 2)]:
                    sel = sub[sub['split'] == split]
                    if len(sel) == 0:
                        continue
                    ax_scatter.scatter(
                        sel['T_C'], sel['P_kbar'],
                        s=14, c=REGIME_COLORS[regime],
                        edgecolor=edge, linewidths=0.3,
                        alpha=alpha, zorder=zorder,
                    )
            else:
                ax_scatter.scatter(
                    sub['T_C'], sub['P_kbar'],
                    s=14, c=REGIME_COLORS[regime],
                    edgecolor='black', linewidths=0.3,
                    alpha=0.85, zorder=2,
                )
    else:
        ax_scatter.scatter(
            df['T_C'], df['P_kbar'], s=14,
            c=arcpl_color or OKABE_ITO['vermillion'], edgecolor='black',
            linewidths=0.3, alpha=0.85, zorder=2,
            label=f'ArcPL opx_liq (n={len(df)})',
        )

    for edge in REGIME_EDGES[1:-1]:
        ax_scatter.axhline(edge, color='0.4', ls='--', lw=0.6, zorder=3)
        ax_scatter.text(
            0.985, edge, f'{edge} kbar',
            transform=ax_scatter.get_yaxis_transform(),
            fontsize=7, color='0.3', va='bottom', ha='right',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.7, pad=1),
            zorder=4,
        )

    ax_scatter.set_xlim(*t_lim)
    ax_scatter.set_ylim(*p_lim)
    ax_scatter.set_xlabel('T (\u00b0C)')
    ax_scatter.set_ylabel('P (kbar)')
    if overlay_hull is not None:
        ax_scatter.legend(fontsize=8, loc='upper left', framealpha=0.9)

    ax_histx.hist(df['T_C'].dropna(), bins=t_bins, color='0.6',
                  edgecolor='none')
    ax_histx.set_ylabel('count', fontsize=8)
    ax_histx.set_ylim(0, t_count_max)
    ax_histx.tick_params(labelbottom=False)
    ax_histx.set_title(title, fontsize=12, fontweight='bold',
                       loc='left', pad=6)

    ax_histy.hist(df['P_kbar'].dropna(), bins=p_bins, color='0.6',
                  edgecolor='none', orientation='horizontal')
    ax_histy.set_xlabel('count', fontsize=8)
    ax_histy.set_xlim(0, p_count_max)
    ax_histy.tick_params(labelleft=False)

    # Per-panel 2-column stats block directly below the scatter.
    n_total = len(df)
    n_cit = int(df['Citation'].nunique()) if 'Citation' in df.columns else 0
    t_lo, t_hi = df['T_C'].quantile([0.01, 0.99])
    p_lo, p_hi = df['P_kbar'].quantile([0.01, 0.99])

    left_col = [f"n total       = {n_total}"]
    if stats_kind == 'expetdb' and 'split' in df.columns:
        n_train = int((df['split'] == 'train').sum())
        n_test  = int((df['split'] == 'test').sum())
        left_col += [
            f"n train       = {n_train}",
            f"n test        = {n_test}",
        ]
    else:
        left_col += [
            f"partition     = external holdout",
            f"(no train/test split)",
        ]
    left_col += [
        f"n citations   = {n_cit}",
        f"T (1-99%)     = {t_lo:.0f}-{t_hi:.0f} \u00b0C",
        f"P (1-99%)     = {p_lo:.1f}-{p_hi:.1f} kbar",
    ]
    right_col = ['regime counts:']
    for r in REGIME_NAMES:
        cnt = int((df['regime'] == r).sum())
        right_col.append(f"  {r:<20s} {cnt:>4d}")

    ax_stats.text(
        0.02, 0.98, '\n'.join(left_col),
        transform=ax_stats.transAxes,
        fontsize=9, family='monospace',
        va='top', ha='left',
    )
    ax_stats.text(
        0.52, 0.98, '\n'.join(right_col),
        transform=ax_stats.transAxes,
        fontsize=9, family='monospace',
        va='top', ha='left',
    )


def main():
    expetdb_opx_liq  = load_expetdb('opx_clean_opx_liq',  split_tag='opx_liq')
    expetdb_opx_only = load_expetdb('opx_clean_opx_only', split_tag='opx')
    arcpl_opx = load_arcpl_opx_liq()
    arcpl_opx['regime'] = arcpl_opx['P_kbar'].map(regime_for)
    arcpl_opx['split'] = 'holdout'

    # Shared axis range: union over all four data sources, so regime
    # position is visually comparable across panels.
    all_T = pd.concat([
        expetdb_opx_liq['T_C'], expetdb_opx_only['T_C'], arcpl_opx['T_C']
    ]).dropna()
    all_P = pd.concat([
        expetdb_opx_liq['P_kbar'], expetdb_opx_only['P_kbar'],
        arcpl_opx['P_kbar']
    ]).dropna()
    t_lim = _nice_range(float(all_T.min()), float(all_T.max()))
    p_lim = _nice_range(float(max(0.0, all_P.min())), float(all_P.max()))

    # Shared bin edges across all panels so histogram bars have the same
    # width everywhere. Binning spans the shared axis range, not each
    # panel's data range, which is what was producing different bar
    # thicknesses between ExPetDB (wide range) and ArcPL (narrow range).
    t_bins = np.linspace(t_lim[0], t_lim[1], 41)
    p_bins = np.linspace(p_lim[0], p_lim[1], 31)

    # Shared count-axis scale across all panels. Computed from the
    # largest dataset (ExPetDB opx_only) so ArcPL histograms read as
    # visibly smaller against the same vertical scale, which is the
    # intended "see how much smaller the external holdout is" story.
    panels_for_counts = [expetdb_opx_liq, expetdb_opx_only,
                         arcpl_opx, arcpl_opx]
    t_count_max = max(
        int(np.histogram(p['T_C'].dropna(), bins=t_bins)[0].max())
        for p in panels_for_counts
    )
    p_count_max = max(
        int(np.histogram(p['P_kbar'].dropna(), bins=p_bins)[0].max())
        for p in panels_for_counts
    )
    # Small headroom so the tallest bar does not touch the frame.
    t_count_max = int(np.ceil(t_count_max * 1.05))
    p_count_max = int(np.ceil(p_count_max * 1.05))

    fig = plt.figure(figsize=(14, 16), constrained_layout=False)
    outer = fig.add_gridspec(
        2, 2, hspace=0.02, wspace=0.22,
        top=0.955, bottom=0.07, left=0.06, right=0.97,
    )

    scatter_with_marginals(
        fig, outer[0, 0], expetdb_opx_liq,
        f'(a) ExPetDB opx_liq  n={len(expetdb_opx_liq)}',
        t_lim=t_lim, p_lim=p_lim, t_bins=t_bins, p_bins=p_bins,
        t_count_max=t_count_max, p_count_max=p_count_max,
        stats_kind='expetdb')
    scatter_with_marginals(
        fig, outer[0, 1], expetdb_opx_only,
        f'(b) ExPetDB opx_only  n={len(expetdb_opx_only)}',
        t_lim=t_lim, p_lim=p_lim, t_bins=t_bins, p_bins=p_bins,
        t_count_max=t_count_max, p_count_max=p_count_max,
        stats_kind='expetdb')
    scatter_with_marginals(
        fig, outer[1, 0], arcpl_opx,
        f'(c) ArcPL opx_liq held-out  n={len(arcpl_opx)}',
        t_lim=t_lim, p_lim=p_lim, t_bins=t_bins, p_bins=p_bins,
        t_count_max=t_count_max, p_count_max=p_count_max,
        stats_kind='arcpl', overlay_hull=expetdb_opx_liq)
    scatter_with_marginals(
        fig, outer[1, 1], arcpl_opx,
        f'(d) ArcPL opx_only held-out  n={len(arcpl_opx)}',
        t_lim=t_lim, p_lim=p_lim, t_bins=t_bins, p_bins=p_bins,
        t_count_max=t_count_max, p_count_max=p_count_max,
        stats_kind='arcpl', overlay_hull=expetdb_opx_only)

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
    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='0.50', alpha=0.60,
                               markersize=8,
                               label='ExPetDB underlay (panels c, d)'))
    fig.legend(handles=handles, loc='lower center', ncol=7,
               frameon=True, framealpha=0.95, edgecolor='0.6',
               bbox_to_anchor=(0.5, 0.025), fontsize=10,
               title='Regime colors; train/test markers for ExPetDB '
                     'panels (a, b); gray ExPetDB underlay under ArcPL '
                     'points in panels (c, d)',
               title_fontsize=10)

    fig.suptitle(
        'ExPetDB (training + internal test) vs ArcPL opx '
        '(external held-out) data distribution',
        fontsize=13, fontweight='bold', y=0.985,
    )

    out_stem = OUT_DIR / 'Core_01b_fig_dataset_map_holdout'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 1b. Distribution comparison between the ExPetDB opx '
        'corpora (top row) and the ArcPL external opx held-out dataset '
        '(bottom row). Top row: ExPetDB opx_liq (panel a) and ExPetDB '
        'opx_only (panel b) P-T scatter, colored by pre-registered '
        'pressure regime with marginal T and P histograms, using Core_01\'s '
        'train/test marker alpha pattern (train faded, test dark-edge on '
        'top). Bottom row: ArcPL opx held-out (panel c, opx_liq evaluation '
        'track; panel d, opx_only evaluation track) with the matching '
        'ExPetDB corpus (train + test combined) underlaid in gray so the '
        'reader can see how the ArcPL distribution sits inside the '
        f'training space -- the two ArcPL panels share the same underlying '
        f'n={len(arcpl_opx)} reconstructed corpus because the ArcPL '
        'opx_only evaluation re-uses the opx rows from the LEPR Opx-Liq '
        'sheet but predicts from the opx composition alone. The ArcPL opx corpus is reconstructed from the LEPR '
        'Opx-Liq sheet by filtering Citation_x to the `_notinLEPR` tag '
        '(the ArcPL-sourced subset inside LEPR) and applying the nb04 '
        'Part 3 cleaning pipeline: rename to the ExPetDB flat schema, '
        'H2O non-negativity, oxide-total and cation-sum QC on the '
        '6-oxygen basis, Wo <= 5 mol%% pigeonite filter, P <= 100 kbar '
        'ceiling, Fe-Mg Kd equilibrium window (0.23-0.35, Putirka 2008), '
        'and citation-key overlap removal against ExPetDB. All four '
        'scatter axes share a single (T, P) axis range computed as the '
        'union of all four data sources, so regime position is visually '
        'comparable across panels. Regime boundary dashed lines are '
        'labeled at the right edge of each plot using axis-fraction '
        'coordinates so labels always stay inside the axes. Source: '
        'data/processed/{opx_clean_opx_liq, opx_clean_opx_only}.parquet '
        'and data/raw/external/'
        'LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx (sheet Opx-Liq, '
        'Citation_x contains `_notinLEPR`).'
    )
    (OUT_DIR / 'Core_01b_fig_dataset_map_holdout.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
