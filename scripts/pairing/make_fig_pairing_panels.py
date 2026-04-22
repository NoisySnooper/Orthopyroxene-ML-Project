#!/usr/bin/env python3
"""Phase 1: 20-panel pairing panels (Core_14a T, Core_14b P).

Reads results/pairing_matrix_22rows.csv for row labels and RMSEs.
Computes per-sample Method A vs Method B scatters on the natural
corpus, with fixed-axis convention.

  T panels: [750, 1800] C, identity + +/-50 C envelope
  P panels: [0, 60] kbar, identity + +/-5 kbar envelope
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


def load_corpus():
    nb8 = pd.read_csv('results/nb08_natural_predictions.csv')
    c11 = pd.read_csv('results/core11_extended_predictions.csv')
    return nb8.merge(c11, on='Experiment', how='outer', suffixes=('_nb8', '_c11'))


COLS = {
    'T_ml_opx_liq':        lambda m: m.get('T_ml_opx_liq'),
    'P_ml_opx_liq':        lambda m: m.get('P_ml_opx_liq'),
    'T_ml_opx_only':       lambda m: m.get('T_ml_opx_only'),
    'P_ml_opx_only':       lambda m: m.get('P_ml_opx_only'),
    'T_ml_cpx_liq':        lambda m: m.get('T_ml_cpx_liq'),
    'P_ml_cpx_liq':        lambda m: m.get('P_ml_cpx_liq'),
    'T_ml_cpx_only':       lambda m: m.get('T_ml_cpx_liq'),
    'P_ml_cpx_only':       lambda m: m.get('P_ml_cpx_liq'),
    'T_agreda_cpx_liq':    lambda m: m.get('T_agreda_cpx_liq'),
    'P_agreda_cpx_liq':    lambda m: m.get('P_agreda_cpx_liq'),
    'T_agreda_cpx_only':   lambda m: m.get('T_agreda_cpx_liq'),
    'P_agreda_cpx_only':   lambda m: m.get('P_agreda_cpx_liq'),
    'T_jorgenson_cpx_liq': lambda m: m.get('T_jorgenson_cpx_liq'),
    'P_jorgenson_cpx_liq': lambda m: m.get('P_jorgenson_cpx_liq'),
    'T_jorgenson_cpx_only': lambda m: m.get('T_jorgenson_cpx_liq'),
    'P_jorgenson_cpx_only': lambda m: m.get('P_jorgenson_cpx_liq'),
    'T_wang_cpx':          lambda m: m.get('T_wang_cpx_liq'),
    'P_wang_cpx':          lambda m: m.get('P_wang_cpx_liq'),
    'T_putirka_opx_liq':   lambda m: m.get('T_putirka_opx_liq_eq28a'),
    'P_putirka_opx_liq':   lambda m: pd.Series([np.nan] * len(m)),
    'T_putirka_2px':       lambda m: m.get('T_putirka_2px_eq36'),
    'P_putirka_2px':       lambda m: m.get('P_putirka_2px_eq39'),
    'T_brey_kohler':       lambda m: m.get('T_putirka_2px_eq36'),
    'P_brey_kohler':       lambda m: m.get('P_putirka_2px_eq39'),
    'T_putirka_cpx_only':  lambda m: m.get('T_putirka_2px_eq36'),
    'P_putirka_cpx_only':  lambda m: m.get('P_putirka_opx_only_eq29c'),
}

ROWS = [
    ('A', 'T_ml_opx_liq',  'T_ml_cpx_liq',        'P_ml_opx_liq', 'P_ml_cpx_liq'),
    ('B', 'T_ml_opx_liq',  'T_agreda_cpx_liq',    'P_ml_opx_liq', 'P_agreda_cpx_liq'),
    ('C', 'T_ml_opx_liq',  'T_putirka_opx_liq',   'P_ml_opx_liq', 'P_putirka_opx_liq'),
    ('D', 'T_ml_opx_liq',  'T_jorgenson_cpx_liq', 'P_ml_opx_liq', 'P_jorgenson_cpx_liq'),
    ('E', 'T_ml_opx_liq',  'T_wang_cpx',          'P_ml_opx_liq', 'P_wang_cpx'),
    ('F', 'T_ml_opx_liq',  'T_ml_opx_only',       'P_ml_opx_liq', 'P_ml_opx_only'),
    ('G', 'T_ml_opx_only', 'T_putirka_2px',       'P_ml_opx_only', 'P_putirka_2px'),
    ('H', 'T_ml_opx_only', 'T_brey_kohler',       'P_ml_opx_only', 'P_brey_kohler'),
    ('I', 'T_ml_opx_only', 'T_ml_cpx_only',       'P_ml_opx_only', 'P_ml_cpx_only'),
    ('J', 'T_agreda_cpx_only', 'T_jorgenson_cpx_only', 'P_agreda_cpx_only', 'P_jorgenson_cpx_only'),
    ('M', 'T_ml_opx_only', 'T_agreda_cpx_only',   'P_ml_opx_only', 'P_agreda_cpx_only'),
    ('N', 'T_ml_opx_only', 'T_jorgenson_cpx_only','P_ml_opx_only', 'P_jorgenson_cpx_only'),
    ('O', 'T_ml_opx_only', 'T_putirka_cpx_only',  'P_ml_opx_only', 'P_putirka_cpx_only'),
    ('P', 'T_ml_opx_only', 'T_wang_cpx',          'P_ml_opx_only', 'P_wang_cpx'),
    ('Q', 'T_ml_cpx_liq',  'T_agreda_cpx_liq',    'P_ml_cpx_liq', 'P_agreda_cpx_liq'),
    ('R', 'T_ml_cpx_liq',  'T_jorgenson_cpx_liq', 'P_ml_cpx_liq', 'P_jorgenson_cpx_liq'),
    ('S', 'T_ml_cpx_only', 'T_agreda_cpx_only',   'P_ml_cpx_only', 'P_agreda_cpx_only'),
    ('T', 'T_ml_cpx_only', 'T_jorgenson_cpx_only','P_ml_cpx_only', 'P_jorgenson_cpx_only'),
    ('U', 'T_ml_opx_liq',  'T_putirka_opx_liq',   None, None),
    ('V', None, None,                              'P_ml_opx_liq', 'P_putirka_opx_liq'),
]


def plot_pairing(corpus, out_stem, axis='T'):
    fig, axes = plt.subplots(4, 6, figsize=(22, 15))
    axes = axes.ravel()

    pair_df = pd.read_csv('results/pairing_matrix_22rows.csv')

    lim = (750, 1800) if axis == 'T' else (0, 60)
    envelope = 50 if axis == 'T' else 5
    unit = 'C' if axis == 'T' else 'kbar'

    for ax_idx, (label, ta, tb, pa, pb) in enumerate(ROWS):
        ax = axes[ax_idx]
        a_col = ta if axis == 'T' else pa
        b_col = tb if axis == 'T' else pb
        if a_col is None or b_col is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f'row {label}\n(no {axis} pair)', ha='center',
                    va='center', transform=ax.transAxes, fontsize=10,
                    color='0.4')
            continue
        x = COLS[a_col](corpus)
        y = COLS[b_col](corpus)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f'row {label}\n(no valid pairs)', ha='center',
                    va='center', transform=ax.transAxes, fontsize=10,
                    color='0.4')
            continue
        x, y = x[m], y[m]
        ax.scatter(x, y, s=6, alpha=0.55, color='#2b8cbe',
                   edgecolor='none')
        ax.plot(lim, lim, 'k-', lw=0.7)
        ax.plot(lim, (lim[0] + envelope, lim[1] + envelope),
                ls='--', color='0.5', lw=0.5)
        ax.plot(lim, (lim[0] - envelope, lim[1] - envelope),
                ls='--', color='0.5', lw=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')

        row_info = pair_df[pair_df.row_label == label]
        if len(row_info):
            r = row_info.iloc[0]
            rmse_key = f'{axis}_rmse_disagreement'
            lo_key = f'{axis}_ci_lo'
            hi_key = f'{axis}_ci_hi'
            rmse_val = r[rmse_key]
            ci_txt = (f'{rmse_val:.1f} {unit} [{r[lo_key]:.1f}, {r[hi_key]:.1f}]'
                      if np.isfinite(rmse_val) else 'NA')
        else:
            ci_txt = 'NA'
        ax.set_title(f'{label}: n={m.sum()}\nRMSE={ci_txt}',
                     fontsize=8, loc='left', pad=2)
        ax.tick_params(labelsize=7)

    for i in range(len(ROWS), len(axes)):
        axes[i].set_axis_off()

    fig.suptitle(f'Pairing panels - {axis} ({unit}): 20-row natural-sample matrix',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def main():
    corpus = load_corpus()
    out_dir = PROJECT_ROOT / 'figures' / 'core'
    plot_pairing(corpus, str(out_dir / 'Core_14a_fig_pairing_panels_T'), axis='T')
    plot_pairing(corpus, str(out_dir / 'Core_14b_fig_pairing_panels_P'), axis='P')

    caption = (
        '20-panel pairing comparison of methods on LEPR natural '
        'two-pyroxene corpus. Each panel is a 1:1 scatter of Method A '
        'predicted vs Method B predicted. Identity line in solid '
        'black; +/- 50 C envelope (T) or +/- 5 kbar envelope (P) in '
        'dashed gray. Fixed axes: [750, 1800] C for T, [0, 60] kbar for '
        'P. RMSE of disagreement with 95% bootstrap CI (n_boot=500) '
        'annotated per panel. Rows A through V labelled per spec; '
        'K and L omitted per user decision. Rows with missing method '
        'predictions (e.g., V using Putirka opx-liq P which is not '
        'computed in this corpus) are shown as empty placeholder '
        'panels.'
    )
    (out_dir / 'Core_14_txt_caption.txt').write_text(caption, encoding='utf-8')
    print('wrote Core_14a, Core_14b')


if __name__ == '__main__':
    main()
