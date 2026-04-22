#!/usr/bin/env python3
"""Phase 2 P2.2 figure: feature concordance heatmaps.

Produces an 8-panel figure (2 rows x 4 cols, one panel per cell) showing
the pairwise Spearman rho across 8 tuned families per cell.

Output: figures/core/Core_15_fig_feature_concordance.pdf (+ .png)
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

SPR_CSV = PROJECT_ROOT / 'results' / 'feature_concordance_spearman_matrix.csv'
FIG_STEM = PROJECT_ROOT / 'figures' / 'core' / 'Core_15_fig_feature_concordance'

FAMILIES = ['RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM',
            'ElasticNet', 'MLP']
CELLS = [
    ('opx', 'opx_liq', 'T_C'),
    ('opx', 'opx_liq', 'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
    ('cpx', 'cpx_liq', 'T_C'),
    ('cpx', 'cpx_liq', 'P_kbar'),
    ('cpx', 'cpx_only', 'T_C'),
    ('cpx', 'cpx_only', 'P_kbar'),
]


def build_matrix(df, track, target):
    sub = df[(df.track == track) & (df.target == target)]
    M = np.full((len(FAMILIES), len(FAMILIES)), np.nan)
    for _, r in sub.iterrows():
        if r['family_a'] in FAMILIES and r['family_b'] in FAMILIES:
            i = FAMILIES.index(r['family_a'])
            j = FAMILIES.index(r['family_b'])
            M[i, j] = r['spearman_rho']
            M[j, i] = r['spearman_rho']
    np.fill_diagonal(M, 1.0)
    return M


def main():
    df = pd.read_csv(SPR_CSV)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    vmin, vmax = -0.2, 1.0
    for ax, (pipe, track, target) in zip(axes.ravel(), CELLS):
        M = build_matrix(df, track, target)
        im = ax.imshow(M, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES, rotation=45, fontsize=7, ha='right')
        ax.set_yticklabels(FAMILIES, fontsize=7)
        unit = '°C' if target == 'T_C' else 'kbar'
        off = np.nanmedian(M[np.triu_indices_from(M, k=1)])
        ax.set_title(f'{track}/{target} ({unit})\nmed rho={off:.2f}',
                     fontsize=9, loc='left')
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=6,
                            color='white' if abs(v) > 0.6 else 'black')
    fig.suptitle('Feature concordance across 8 tuned families '
                 '(pairwise Spearman rho of permutation-importance ranks)',
                 fontsize=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                        orientation='vertical', pad=0.02)
    cbar.set_label('Spearman rho', fontsize=9)
    fig.savefig(f'{FIG_STEM}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{FIG_STEM}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {FIG_STEM}.pdf')


if __name__ == '__main__':
    main()
