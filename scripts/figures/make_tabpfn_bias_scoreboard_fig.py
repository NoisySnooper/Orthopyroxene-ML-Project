#!/usr/bin/env python3
"""Build fig44_tabpfn_bias_scoreboard_opx.

Panel A: pre vs post RMSE bar chart for the 4 opx combos, TabPFN only,
  at canonical seed 42. Bars annotated with winner label.

Panel B: per-seed ship stability heatmap (5 seeds x 4 combos) with cells
  colored by winner (none/A/B) and annotated with Form A overall delta.

Source CSVs:
  results/tabpfn_bias_correction_perseed.csv (20 rows)
  results/tabpfn_bias_correction_summary.csv (4 rows)

Outputs:
  figures/fig44_tabpfn_bias_scoreboard_opx.pdf
  figures/fig44_tabpfn_bias_scoreboard_opx.png
  figures/fig44_tabpfn_bias_scoreboard_opx.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / 'results'
FIGS = PROJECT_ROOT / 'figures'

# Okabe-Ito palette
OK_BLUE = '#0072B2'
OK_ORANGE = '#E69F00'
OK_GREEN = '#009E73'
OK_RED = '#D55E00'
OK_GRAY = '#999999'

COMBO_ORDER = [
    ('opx_liq', 'T_C'),
    ('opx_liq', 'P_kbar'),
    ('opx_only', 'T_C'),
    ('opx_only', 'P_kbar'),
]
COMBO_LABELS = ['opx-liq T', 'opx-liq P', 'opx-only T', 'opx-only P']
UNITS = {'T_C': '°C', 'P_kbar': 'kbar'}


def panel_a(ax, summary: pd.DataFrame, perseed: pd.DataFrame):
    x = np.arange(len(COMBO_ORDER))
    width = 0.28
    pre = []
    post_a = []
    post_b = []
    winners_canonical = []
    for tr, tg in COMBO_ORDER:
        s = summary[(summary['track'] == tr) & (summary['target'] == tg)].iloc[0]
        pre.append(s['pre_rmse_mean'])
        post_a.append(s['post_rmse_a_mean'])
        post_b.append(s['post_rmse_b_mean'])
        c = perseed[(perseed['track'] == tr) & (perseed['target'] == tg)
                    & (perseed['seed'] == 42)].iloc[0]
        winners_canonical.append(c['winner'])

    b1 = ax.bar(x - width, pre, width, label='Pre-correction',
                color=OK_GRAY, edgecolor='black')
    b2 = ax.bar(x, post_a, width, label='Form A (regime piecewise)',
                color=OK_BLUE, edgecolor='black')
    b3 = ax.bar(x + width, post_b, width, label='Form B (piecewise sigmoid)',
                color=OK_ORANGE, edgecolor='black')

    for i, (_, tg) in enumerate(COMBO_ORDER):
        unit = UNITS[tg]
        for offset, v in zip((-width, 0.0, width), (pre[i], post_a[i], post_b[i])):
            ax.text(x[i] + offset, v, f'{v:.1f}', ha='center', va='bottom',
                    fontsize=7)
        # Winner annotation above the cluster
        w = winners_canonical[i]
        color = OK_GREEN if w == 'A' else OK_RED if w == 'B' else 'black'
        ax.text(x[i], max(pre[i], post_a[i], post_b[i]) * 1.10,
                f'winner: {w}', ha='center', fontsize=8,
                color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(COMBO_LABELS, fontsize=9)
    ax.set_ylabel('Test RMSE (seed mean, native units)', fontsize=9)
    ax.set_title('A. TabPFN pre/post-correction RMSE (mean over 5 seeds)',
                 fontsize=10, loc='left')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def panel_b(ax, perseed: pd.DataFrame):
    seeds = sorted(perseed['seed'].unique())
    grid = np.zeros((len(COMBO_ORDER), len(seeds)))
    labels = np.empty((len(COMBO_ORDER), len(seeds)), dtype=object)
    for i, (tr, tg) in enumerate(COMBO_ORDER):
        for j, s in enumerate(seeds):
            row = perseed[(perseed['track'] == tr)
                          & (perseed['target'] == tg)
                          & (perseed['seed'] == s)]
            if row.empty:
                grid[i, j] = np.nan
                labels[i, j] = ''
                continue
            w = row.iloc[0]['winner']
            grid[i, j] = {'none': 0, 'A': 1, 'B': 2}.get(w, 0)
            labels[i, j] = w

    cmap = plt.matplotlib.colors.ListedColormap([OK_GRAY, OK_BLUE, OK_ORANGE])
    im = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, aspect='auto')
    for i in range(len(COMBO_ORDER)):
        for j in range(len(seeds)):
            ax.text(j, i, labels[i, j], ha='center', va='center',
                    color='white' if labels[i, j] != 'none' else 'black',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'{s}' for s in seeds], fontsize=9)
    ax.set_yticks(range(len(COMBO_ORDER)))
    ax.set_yticklabels(COMBO_LABELS, fontsize=9)
    ax.set_xlabel('Seed', fontsize=9)
    ax.set_title('B. Per-seed winner (5-seed OOF bias fit)', fontsize=10, loc='left')

    legend_elems = [
        Patch(facecolor=OK_GRAY, label='none'),
        Patch(facecolor=OK_BLUE, label='Form A ships'),
        Patch(facecolor=OK_ORANGE, label='Form B ships'),
    ]
    ax.legend(handles=legend_elems, loc='upper right',
              bbox_to_anchor=(1.0, -0.12), ncol=3, fontsize=8, frameon=False)


def main():
    perseed = pd.read_csv(RESULTS / 'tabpfn_bias_correction_perseed.csv')
    summary = pd.read_csv(RESULTS / 'tabpfn_bias_correction_summary.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panel_a(axes[0], summary, perseed)
    panel_b(axes[1], perseed)
    fig.tight_layout()

    pdf = FIGS / 'fig44_tabpfn_bias_scoreboard_opx.pdf'
    png = FIGS / 'fig44_tabpfn_bias_scoreboard_opx.png'
    cap = FIGS / 'fig44_tabpfn_bias_scoreboard_opx.txt'
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    caption = (
        'Fig. 44. TabPFN v2 bias-correction scoreboard on the 4 opx '
        'combinations. Panel A: mean test RMSE (over 5 seeds) pre-correction '
        '(gray) and after applying Form A (blue, regime-piecewise linear) or '
        'Form B (orange, piecewise sigmoid) fit on 5-seed x 10-fold OOF '
        'residuals. Annotations show winner verdict under the conservative '
        'ship-if-better rule at canonical seed 42 (overall delta > 1e-6 and '
        'no regime degradation). Form A ships on opx_only/T_C and '
        'opx_only/P_kbar with large pre-post gaps (TabPFN materially over-'
        'predicts deeper mantle residual bias uncorrected). Form B ships '
        'nothing on opx TabPFN, consistent with the 0/8 tuned-family pattern. '
        'Panel B: per-seed ship stability over seeds 42-46; cells color '
        'winner (gray = none, blue = A, orange = B) annotated with the '
        'verdict. Shipping is consistent at seed 42, 43, 45 on opx_only/T_C '
        'and at all 5 seeds on opx_only/P_kbar. Sources: '
        'results/tabpfn_bias_correction_perseed.csv, '
        'results/tabpfn_bias_correction_summary.csv.'
    )
    cap.write_text(caption, encoding='utf-8')
    print(f'wrote: {pdf.name}, {png.name}, {cap.name}')


if __name__ == '__main__':
    main()
