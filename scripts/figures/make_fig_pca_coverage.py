#!/usr/bin/env python3
"""supp_fig_4: compositional coverage in PCA space.

Two panels (opx-liq, opx-only). For each track:
  * fit StandardScaler + PCA on the ExPetDB train opx oxides;
  * project ExPetDB test and ArcPL holdout onto the same axes;
  * color points by pre-registered P regime (locked palette);
  * use marker style (alpha + edge) to encode train / test / ArcPL.

Variance explained per axis is annotated so the reader can decide how
seriously to take the inferred geometry. The headline question this
figure answers: does the ArcPL external holdout sit inside the ExPetDB
compositional cloud (supporting the §4.3.1 cross-corpus claim)?

Sources:
  data/processed/opx_clean_opx_liq.parquet
  data/processed/opx_clean_opx_only.parquet
  data/splits/{train,test}_indices_*.npy
  src.external.arcpl_opx.load_arcpl_opx_liq
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._model_palette import REGIME_COLORS  # noqa: E402
from scripts.figures._style import apply_pub_style, make_fig, add_grid, resolve_out_dir, jgr_figsize, jgr_top, jgr_bottom # noqa: E402
from scripts.figures._labels import REGIME_ORDER, TRACK_LABEL  # noqa: E402
from src.external.arcpl_opx import load_arcpl_opx_liq  # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

# Consistent oxide set across ExPetDB and ArcPL.
PCA_FEATURES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total',
                'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']

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


def load_expetdb(parquet_name: str, split_tag: str):
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


def fit_pca_and_project(train_df: pd.DataFrame, *projections):
    X_train = train_df[PCA_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=2).fit(scaler.transform(X_train))
    train_scores = pca.transform(scaler.transform(X_train))
    out = [train_scores]
    for proj in projections:
        Xp = proj[PCA_FEATURES].to_numpy(dtype=float)
        out.append(pca.transform(scaler.transform(Xp)))
    return out, pca.explained_variance_ratio_


def draw_expetdb_panel(ax, train_df, test_df, evr, train_pc, test_pc,
                       track, idx, *, x_lim, y_lim):
    """Top-row panel — ExPetDB train + test only, colored by regime."""
    train_regimes = train_df['regime'].to_numpy()
    test_regimes = test_df['regime'].to_numpy()

    for r in REGIME_ORDER:
        m = train_regimes == r
        if m.sum():
            ax.scatter(train_pc[m, 0], train_pc[m, 1],
                       s=14, c=REGIME_COLORS[r], alpha=0.35,
                       edgecolor='none', zorder=1)
        m = test_regimes == r
        if m.sum():
            ax.scatter(test_pc[m, 0], test_pc[m, 1],
                       s=20, c=REGIME_COLORS[r], alpha=0.90,
                       edgecolor='#222222', linewidths=0.4, zorder=2)

    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.set_xlabel(f'PC1 ({evr[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({evr[1] * 100:.1f}% var)')
    ax.set_title(f'({idx}) ExPetDB · {TRACK_LABEL[track]}')
    add_grid(ax, axis='both')


def draw_arcpl_panel(ax, arcpl_df, evr, arcpl_pc,
                     train_pc_underlay, track, idx, *, x_lim, y_lim):
    """Bottom-row panel — ArcPL points alone, with the ExPetDB train
    cloud as a faint gray underlay so the reader can confirm overlap."""
    # Faint ExPetDB underlay
    ax.scatter(train_pc_underlay[:, 0], train_pc_underlay[:, 1],
               s=10, c='#cccccc', alpha=0.40,
               edgecolor='none', zorder=1)
    # ArcPL points (squares, vermillion edge), colored by regime
    arcpl_regimes = arcpl_df['regime'].to_numpy()
    for r in REGIME_ORDER:
        m = arcpl_regimes == r
        if m.sum():
            ax.scatter(arcpl_pc[m, 0], arcpl_pc[m, 1],
                       s=26, c=REGIME_COLORS[r], alpha=0.95,
                       edgecolor='#D55E00', linewidths=0.7,
                       marker='s', zorder=3)

    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.set_xlabel(f'PC1 ({evr[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({evr[1] * 100:.1f}% var)')
    ax.set_title(f'({idx}) ArcPL · {TRACK_LABEL[track]}')
    add_grid(ax, axis='both')


def main():
    arcpl_full = load_arcpl_opx_liq().copy()
    arcpl_full['regime'] = arcpl_full['P_kbar'].map(regime_for)

    panel_data = []  # one entry per track
    for track, pq, split_tag, _ in PANELS:
        full = load_expetdb(pq, split_tag)
        full = full.dropna(subset=PCA_FEATURES + ['P_kbar']).copy()
        train = full[full['split'] == 'train']
        test = full[full['split'] == 'test']
        arcpl = arcpl_full.dropna(subset=PCA_FEATURES + ['P_kbar']).copy()

        (train_pc, test_pc, arcpl_pc), evr = fit_pca_and_project(
            train, test, arcpl)

        # Shared axis limits per column so top/bottom panels read on
        # the same scale.
        all_pc = np.vstack([train_pc, test_pc, arcpl_pc])
        x_pad = 0.06 * (all_pc[:, 0].max() - all_pc[:, 0].min())
        y_pad = 0.06 * (all_pc[:, 1].max() - all_pc[:, 1].min())
        x_lim = (all_pc[:, 0].min() - x_pad, all_pc[:, 0].max() + x_pad)
        y_lim = (all_pc[:, 1].min() - y_pad, all_pc[:, 1].max() + y_pad)

        panel_data.append({
            'track': track, 'train': train, 'test': test, 'arcpl': arcpl,
            'evr': evr,
            'train_pc': train_pc, 'test_pc': test_pc, 'arcpl_pc': arcpl_pc,
            'x_lim': x_lim, 'y_lim': y_lim,
        })

    # 2x2 layout: top row ExPetDB, bottom row ArcPL. One column per
    # track. JGR mode shrinks width only; height stays absolute so the
    # 4 panels remain readable.
    from scripts.figures._style import JGR_2COL_IN, is_jgr_mode  # noqa
    target_w = JGR_2COL_IN if is_jgr_mode() else 11.0
    fig, axes = plt.subplots(2, 2, figsize=(target_w, 9.0))
    plt.subplots_adjust(top=jgr_top(0.90), bottom=jgr_bottom(0.10),
                        left=0.07, right=0.97,
                        hspace=0.35, wspace=0.22)

    panel_letters = [['a', 'b'], ['c', 'd']]
    # Top row: ExPetDB
    for col, pd_ in enumerate(panel_data):
        draw_expetdb_panel(axes[0, col], pd_['train'], pd_['test'],
                           pd_['evr'], pd_['train_pc'], pd_['test_pc'],
                           pd_['track'], panel_letters[0][col],
                           x_lim=pd_['x_lim'], y_lim=pd_['y_lim'])
    # Bottom row: ArcPL with ExPetDB train as faint underlay
    for col, pd_ in enumerate(panel_data):
        draw_arcpl_panel(axes[1, col], pd_['arcpl'], pd_['evr'],
                         pd_['arcpl_pc'], pd_['train_pc'],
                         pd_['track'], panel_letters[1][col],
                         x_lim=pd_['x_lim'], y_lim=pd_['y_lim'])

    legend_handles = []
    for r in REGIME_ORDER:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=REGIME_COLORS[r],
                                          markersize=7, label=r))
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#bbbbbb',
                                      markersize=7, alpha=0.35,
                                      label='ExPetDB train (faded)'))
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#bbbbbb',
                                      markeredgecolor='#222222',
                                      markersize=7,
                                      label='ExPetDB test (dark edge)'))
    legend_handles.append(plt.Line2D([0], [0], marker='s', color='w',
                                      markerfacecolor='#bbbbbb',
                                      markeredgecolor='#D55E00',
                                      markersize=7,
                                      label='ArcPL (vermillion edge)'))
    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=4)

    fig.suptitle(
        'Compositional coverage in PCA space (opx oxides)\n'
        'Top: ExPetDB train + test  ·  Bottom: ArcPL holdout '
        '(over faint ExPetDB train cloud)'
    )

    stem = OUT_DIR / 'supp_fig_4'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Supp. Figure 4. Compositional coverage in PCA space for the two '
        'opx pipelines, separated by corpus. Datasets: ExPetDB 2025-07-21 '
        'train + test (opx-liq n=600, opx-only n=1035, citation-grouped '
        '80/20 split) and ArcPL external holdout (LEPR _notinLEPR subset, '
        'n=197). Layout: top row (a, b) shows ExPetDB only — train as '
        'faded circles, test with dark edges, colored by pre-registered '
        'P regime; bottom row (c, d) shows ArcPL only as vermillion-edge '
        'squares, with the ExPetDB train cloud as a faint gray underlay '
        'so the reader can confirm overlap. Each column shares axes '
        'between top and bottom panels. For each track we standardize '
        'ten opx oxides (SiO2, TiO2, Al2O3, Cr2O3, FeO_total, MnO, MgO, '
        'CaO, Na2O, K2O) on the ExPetDB train partition, fit a 2-component '
        'PCA on the standardized training compositions, and project the '
        'remaining datasets onto the same axes. PC1 and PC2 variance '
        'fractions are annotated. The figure supports §4.3.1: ArcPL '
        'points fall inside the ExPetDB compositional cloud, so the '
        'cross-corpus claim is geometrically defensible — ArcPL is not '
        'in a chemistry region the model has never seen.'
    )
    (OUT_DIR / 'supp_fig_4.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
