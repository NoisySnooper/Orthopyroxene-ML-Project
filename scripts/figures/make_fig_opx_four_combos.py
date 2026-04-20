#!/usr/bin/env python3
"""Core_09: opx 4-panel companion to Core_08 headline.

Per-regime test RMSE across 5 candidates for all four opx combinations
(opx_liq T, opx_liq P, opx_only T, opx_only P). Honest complement to the
opx_only P headline: three of four panels are null or marginal; only
panel (d) opx_only P_kbar shows a shippable post-correction win.
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

from scripts.figures._model_palette import (  # noqa: E402
    CANDIDATE_COLORS, CANDIDATE_LABELS,
)

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [
    ('opx_liq',  'T_C',    '(a) Opx+Liq T (C)'),
    ('opx_liq',  'P_kbar', '(b) Opx+Liq P (kbar)'),
    ('opx_only', 'T_C',    '(c) Opx only T (C)'),
    ('opx_only', 'P_kbar', '(d) Opx only P (kbar)'),
]

CANDIDATES = [
    ('tuned_pre',   'v10_pre_rmse'),
    ('tuned_post',  'v10_post_rmse'),
    ('putirka',     'best_external_rmse'),
    ('tabpfn_pre',  'tabpfn_rmse'),
    ('tabpfn_post', 'tabpfn_post_rmse'),
]

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_LABELS = ['shallow', 'MASH', 'lithos', 'deep', 'ALL']


def main():
    sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')
    shipped = pd.read_csv('results/bias_correction_shipped.csv')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for ax, (track, target, title) in zip(axes, PANELS):
        sub = sc[(sc['track'] == track) & (sc['target'] == target)].copy()
        sub = sub.set_index('regime').reindex(REGIME_ORDER)

        x = np.arange(len(REGIME_ORDER))
        width = 0.16
        for i, (key, col) in enumerate(CANDIDATES):
            vals = sub[col].values
            offsets = x + (i - 2) * width
            ax.bar(offsets, vals, width=width,
                   color=CANDIDATE_COLORS[key],
                   label=CANDIDATE_LABELS[key],
                   edgecolor='black', linewidth=0.4)

        # Winner column annotation (per regime, from 'winner' field)
        for xi, w in zip(x, sub['winner'].values):
            if pd.isna(w):
                continue
            abbrev = {'v10_pre': 'Tp', 'v10_corrected': 'TP',
                      'external': 'PU', 'tabpfn': 'TFp',
                      'tabpfn_corrected': 'TFP'}.get(w, '?')
            ax.text(xi, ax.get_ylim()[1] * 0.96 if ax.get_ylim()[1] else 1,
                    abbrev, ha='center', va='top', fontsize=7,
                    color='0.2')

        # Ship verdict box
        ship_rows = shipped[(shipped['track'] == track)
                            & (shipped['target'] == target)]
        tuned_ship = tabpfn_ship = False
        for _, r in ship_rows.iterrows():
            ship_a = str(r.get('ship_a', 'False')) == 'True'
            ship_b = str(r.get('ship_b', 'False')) == 'True'
            if r['model'] == 'TabPFN':
                tabpfn_ship = ship_a or ship_b
            else:
                tuned_ship = tuned_ship or ship_a or ship_b
        verdict = (f'Tuned ships: {"YES" if tuned_ship else "NO"}    '
                   f'TabPFN ships: {"YES" if tabpfn_ship else "NO"}')
        ax.text(0.02, 0.98, verdict,
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8.5, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='0.5',
                          alpha=0.92, pad=3))

        ax.set_xticks(x)
        ax.set_xticklabels(REGIME_LABELS, fontsize=8)
        unit = 'C' if target == 'T_C' else 'kbar'
        ax.set_ylabel(f'RMSE ({unit})')
        ax.set_title(title, fontsize=11, loc='left')
        ax.grid(axis='y', ls=':', alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               fontsize=9)
    fig.suptitle(
        'Four opx combinations: only opx-only P (d) shows shippable '
        'post-correction dominance',
        fontsize=12, y=1.00,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.98))

    out_stem = OUT_DIR / 'Core_09_fig_opx_four_combos'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 9. Per-regime test RMSE across the four opx combinations '
        '(companion to Core_08 headline). Bars show five candidates: '
        'tuned pre-correction, tuned post-correction, best Putirka/Agreda '
        'external benchmark, TabPFN pre-correction, TabPFN post-correction. '
        'Top-left annotation per panel reports whether either correction '
        'track ships under the conservative-acceptance rule (overall_delta '
        '> 1e-6 AND max_regime_degradation <= 1e-6). Only panel (d) '
        'opx_only P_kbar shows genuine post-correction dominance via the '
        'TabPFN Form A route; panels (a) and (b) show no shippable '
        'correction for opx_liq, and panel (c) opx_only T shows only '
        'a marginal TabPFN deeper_mantle win. Winner abbreviations: '
        'Tp=tuned pre, TP=tuned post, PU=Putirka/external, TFp=TabPFN pre, '
        'TFP=TabPFN post. This figure is the honest complement to the '
        'opx_only P_kbar headline (Core_08): the headline is the exception, '
        'not the rule.'
    )
    (OUT_DIR / 'Core_09_fig_opx_four_combos.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
