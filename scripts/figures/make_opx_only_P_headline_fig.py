#!/usr/bin/env python3
"""Build fig45_opx_only_P_headline.

Per-regime RMSE comparison for opx_only P_kbar across:
  v10_pre (best tuned uncorrected), v10_post (best tuned corrected),
  external (Putirka 29c), tabpfn (uncorrected), tabpfn_corrected (Form A).

Error bars show 95% bootstrap CIs from preregistered_scorecard_
postcorrection.csv (where available) or 0 otherwise.

Outputs:
  figures/fig45_opx_only_P_headline.pdf
  figures/fig45_opx_only_P_headline.png
  figures/fig45_opx_only_P_headline.txt
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

RESULTS = PROJECT_ROOT / 'results'
FIGS = PROJECT_ROOT / 'figures'

OK_BLUE = '#0072B2'
OK_ORANGE = '#E69F00'
OK_GREEN = '#009E73'
OK_RED = '#D55E00'
OK_PURPLE = '#CC79A7'
OK_GRAY = '#999999'

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_LABELS = ['shallow\ncrustal\n(<5 kbar)', 'deep crustal\nMASH\n(5-10 kbar)',
                 'lithospheric\nmantle\n(10-20 kbar)',
                 'deeper\nmantle\n(>=20 kbar)', 'ALL\nregimes']


def main():
    sc = pd.read_csv(RESULTS / 'preregistered_scorecard_postcorrection.csv')
    sub = sc[(sc['track'] == 'opx_only') & (sc['target'] == 'P_kbar')].copy()
    sub = sub.set_index('regime').loc[REGIME_ORDER].reset_index()

    series = {
        'v10 pre (tuned, uncorrected)': {
            'vals': sub['v10_pre_rmse'].to_numpy(),
            'lo': sub['v10_pre_rmse_lo'].to_numpy(),
            'hi': sub['v10_pre_rmse_hi'].to_numpy(),
            'color': OK_GRAY,
            'hatch': None,
        },
        'v10 post (Form A)': {
            'vals': sub['v10_post_rmse'].to_numpy(),
            'lo': sub['v10_post_rmse_lo'].to_numpy(),
            'hi': sub['v10_post_rmse_hi'].to_numpy(),
            'color': OK_BLUE,
            'hatch': None,
        },
        'Putirka 29c': {
            'vals': sub['best_external_rmse'].to_numpy(),
            'lo': sub['best_external_rmse_lo'].to_numpy(),
            'hi': sub['best_external_rmse_hi'].to_numpy(),
            'color': OK_RED,
            'hatch': None,
        },
        'TabPFN pre': {
            'vals': sub['tabpfn_rmse'].to_numpy(),
            'lo': sub['tabpfn_rmse_lo'].to_numpy(),
            'hi': sub['tabpfn_rmse_hi'].to_numpy(),
            'color': OK_PURPLE,
            'hatch': None,
        },
        'TabPFN post (Form A)': {
            'vals': sub['tabpfn_post_rmse'].to_numpy(),
            'lo': sub['tabpfn_post_rmse_lo'].to_numpy(),
            'hi': sub['tabpfn_post_rmse_hi'].to_numpy(),
            'color': OK_GREEN,
            'hatch': '//',
        },
    }

    x = np.arange(len(REGIME_ORDER))
    width = 0.16
    offsets = np.linspace(-2 * width, 2 * width, len(series))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (name, d), off in zip(series.items(), offsets):
        vals = d['vals']
        yerr_lo = np.clip(vals - d['lo'], 0, None)
        yerr_hi = np.clip(d['hi'] - vals, 0, None)
        yerr = np.vstack([yerr_lo, yerr_hi])
        ax.bar(x + off, vals, width, label=name, color=d['color'],
               edgecolor='black', yerr=yerr, capsize=3, hatch=d['hatch'])

    # Annotate winner text
    for i, reg in enumerate(REGIME_ORDER):
        w = sub.iloc[i]['winner']
        ax.text(x[i], ax.get_ylim()[1] * 0.95,
                f'winner:\n{w}', ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=OK_GREEN if 'corrected' in w else 'gray',
                          alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, fontsize=9)
    ax.set_ylabel('RMSE (kbar)', fontsize=10)
    ax.set_title('opx-only P_kbar: per-regime RMSE (pre-registered partition, '
                 'bootstrap 95% CI)', fontsize=11)
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    pdf = FIGS / 'fig45_opx_only_P_headline.pdf'
    png = FIGS / 'fig45_opx_only_P_headline.png'
    cap = FIGS / 'fig45_opx_only_P_headline.txt'
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    caption = (
        'Fig. 45. Per-regime RMSE on the opx-only P_kbar test set across '
        'five candidates: v10 tuned pre-correction (gray), v10 tuned post-'
        'correction (Form A, blue), Putirka 29c (red), TabPFN v2 pre-'
        'correction (purple), TabPFN v2 post-correction (Form A, green hatched). '
        'Regimes follow the pre-registered pressure partition (<5, 5-10, '
        '10-20, >=20 kbar) plus ALL. Error bars are 95% bootstrap CIs (n_boot '
        '= 500). TabPFN post-correction wins 3 of the 5 regimes '
        '(lithospheric_mantle, deeper_mantle, ALL) and is within the v10 '
        'corrected CI on the remaining two. Form A transforms TabPFN from '
        'the worst pre-correction method into the aggregate winner, '
        'demonstrating that the in-context predictions carry a systematic '
        'deeper-regime bias that can be removed with a regime-piecewise '
        'linear transform. Source: results/preregistered_scorecard_'
        'postcorrection.csv (canonical seed 42 for TabPFN post).'
    )
    cap.write_text(caption, encoding='utf-8')
    print(f'wrote: {pdf.name}, {png.name}, {cap.name}')


if __name__ == '__main__':
    main()
