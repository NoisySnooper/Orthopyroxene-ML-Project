#!/usr/bin/env python3
"""supp_fig_5: feature concordance heatmaps for the 4 opx (track, target) cells.

Pairwise Spearman rho across the 8 tuned families per cell. TabPFN is
omitted (no fitted sklearn estimator exposed to permutation_importance).

Source: results/feature_concordance_spearman_matrix.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._style import apply_pub_style, make_fig, resolve_out_dir # noqa: E402
from scripts.figures._labels import TARGET_UNIT, panel_header  # noqa: E402

apply_pub_style()

SPR_CSV = PROJECT_ROOT / 'results' / 'feature_concordance_spearman_matrix.csv'
OUT_DIR = resolve_out_dir(PROJECT_ROOT)
FIG_STEM = OUT_DIR / 'supp_fig_5'

FAMILIES = ['RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM',
            'ElasticNet', 'MLP']
CELLS = [
    ('opx', 'opx_liq',  'T_C',    'a'),
    ('opx', 'opx_liq',  'P_kbar', 'b'),
    ('opx', 'opx_only', 'T_C',    'c'),
    ('opx', 'opx_only', 'P_kbar', 'd'),
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
    fig, axes = make_fig('two_col', nrows=2, ncols=2)
    vmin, vmax = -0.2, 1.0
    for ax, (pipe, track, target, idx) in zip(axes.ravel(), CELLS):
        M = build_matrix(df, track, target)
        im = ax.imshow(M, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES, rotation=45, fontsize=8, ha='right')
        ax.set_yticklabels(FAMILIES, fontsize=8)
        off = np.nanmedian(M[np.triu_indices_from(M, k=1)])
        title = panel_header(track, target, idx)
        ax.set_title(f'{title}\nmed ρ={off:.2f}')
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=6,
                            color='white' if abs(v) > 0.6 else '#222222')
    fig.suptitle(
        'Feature concordance across 8 tuned families\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190), perm. importance seed 42'
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                 orientation='vertical', pad=0.02, label='Spearman ρ')
    fig.savefig(f'{FIG_STEM}.pdf')
    fig.savefig(f'{FIG_STEM}.png')

    caption = (
        'Supp. Figure 5. Feature concordance heatmaps for the four opx '
        '(track, target) cells. Each cell shows the pairwise Spearman ρ '
        'across the 8 tuned families on permutation-importance ranks '
        '(20 repeats, seed 42). High off-diagonal ρ means families agree '
        'on which features matter; low ρ means disagreement (often '
        'driven by feature_set differences narrowing the shared feature '
        'set). TabPFN is omitted from the matrix (no fitted sklearn '
        'estimator exposed to permutation_importance). Source: '
        'results/feature_concordance_spearman_matrix.csv.'
    )
    (OUT_DIR / 'supp_fig_5.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {FIG_STEM}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
