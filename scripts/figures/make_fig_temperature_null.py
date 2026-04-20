#!/usr/bin/env python3
"""Core_10: Temperature null result (2-panel opx-only: opx_liq T, opx_only T).

Per-regime RMSE across 5 candidates (tuned pre, tuned post, Putirka best,
TabPFN pre, TabPFN post). Annotated with shipping verdict per panel.
Honest complement to Core_08 opx-only P headline.
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
    ('opx_liq',  'T_C', 'Opx + Liquid T (C)'),
    ('opx_only', 'T_C', 'Opx only T (C)'),
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
REGIME_LABELS = ['shallow\n<5 kbar', 'MASH\n5-15', 'lithos\n15-30',
                 'deep\n>=30', 'ALL']


def main():
    sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')
    shipped = pd.read_csv('results/bias_correction_shipped.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

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

        # Ship verdict annotation
        ship_rows = shipped[(shipped['track'] == track)
                            & (shipped['target'] == target)]
        tuned_ship = False
        tabpfn_ship = False
        if len(ship_rows):
            for _, r in ship_rows.iterrows():
                ship_a = str(r.get('ship_a', 'False')) == 'True'
                ship_b = str(r.get('ship_b', 'False')) == 'True'
                if r['model'] == 'TabPFN':
                    tabpfn_ship = ship_a or ship_b
                else:
                    tuned_ship = ship_a or ship_b
        verdict_lines = []
        verdict_lines.append(
            f'Tuned correction ships: {"YES" if tuned_ship else "NO"}')
        verdict_lines.append(
            f'TabPFN correction ships: {"YES" if tabpfn_ship else "NO"}')
        ax.text(0.02, 0.98, '\n'.join(verdict_lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='0.5',
                          alpha=0.92, pad=3))

        ax.set_xticks(x)
        ax.set_xticklabels(REGIME_LABELS, fontsize=8)
        ax.set_ylabel('RMSE (C)')
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', ls=':', alpha=0.4)

        # Annotate n per regime above bars
        for xi, n in zip(x, sub['n'].values):
            ax.text(xi, ax.get_ylim()[1] * 0.02, f'n={int(n)}',
                    ha='center', va='bottom', fontsize=7, color='0.35')

    axes[0].legend(loc='upper center', bbox_to_anchor=(1.08, -0.15),
                   ncol=5, frameon=False, fontsize=9)
    fig.suptitle(
        'Temperature null result: no correction ships on opx T at the '
        'conservative-acceptance threshold',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    out_stem = OUT_DIR / 'Core_10_fig_temperature_null'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 10. Temperature null result (opx pipeline). Per-regime '
        'test RMSE in Celsius across five candidates (tuned pre-correction, '
        'tuned post-correction, best Putirka/Agreda external benchmark, '
        'TabPFN pre-correction, TabPFN post-correction) for opx_liq T (left) '
        'and opx_only T (right) at canonical seed 42. Shipping verdict per '
        'panel annotated top-left. Neither tuned nor TabPFN corrections '
        'clear the conservative-acceptance rule (overall_delta > 1e-6 AND '
        'max_regime_degradation <= 1e-6) for opx_liq T; for opx_only T, '
        'the tuned post-correction also fails, though TabPFN Form A ships '
        'on the deeper_mantle regime with marginal aggregate gain. The '
        'opx thermometer as a category is the honest complement to the '
        'opx_only P_kbar headline result (Core_08): where the headline '
        'reports a genuine post-correction improvement, here we report '
        'no shippable correction. Companion to Core_08; both should be '
        'read together to avoid selective reporting.'
    )
    (OUT_DIR / 'Core_10_fig_temperature_null.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
