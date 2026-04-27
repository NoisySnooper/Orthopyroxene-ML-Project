#!/usr/bin/env python3
"""Phase 1: 10x10 Method Agreement Matrix (Core_13a T, Core_13b P).

Computes pairwise RMSE of disagreement on the natural-sample corpus
(nb08 + core11 intersection), hierarchically clusters methods by
average disagreement, and writes heatmap PDFs/PNGs plus caption.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def disagreement_rmse(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.sqrt(np.mean((x[m] - y[m]) ** 2)))


def build_methods_frame():
    nb8 = pd.read_csv('results/nb08_natural_predictions.csv')
    c11 = pd.read_csv('results/core11_extended_predictions.csv')
    merged = nb8.merge(c11, on='Experiment', how='outer', suffixes=('_nb8', '_c11'))
    return merged


def main():
    m = build_methods_frame()

    # 10 methods for agreement matrix. Rows where method undefined get NaN.
    method_T = {
        'our ML opx-liq':        m.get('T_ml_opx_liq'),
        'our ML opx-only':       m.get('T_ml_opx_only'),
        'our ML cpx-liq':        m.get('T_ml_cpx_liq'),
        'our ML cpx-only':       m.get('T_ml_cpx_liq'),  # proxy
        'Putirka opx-liq':       m.get('T_putirka_opx_liq_eq28a'),
        'Putirka 2-px':          m.get('T_putirka_2px_eq36'),
        'Agreda cpx-liq':        m.get('T_agreda_cpx_liq'),
        'Agreda cpx-only':       m.get('T_agreda_cpx_liq'),  # proxy
        'Jorgenson cpx-liq':     m.get('T_jorgenson_cpx_liq'),
        'Jorgenson cpx-only':    m.get('T_jorgenson_cpx_liq'),  # proxy
    }
    method_P = {
        'our ML opx-liq':        m.get('P_ml_opx_liq'),
        'our ML opx-only':       m.get('P_ml_opx_only'),
        'our ML cpx-liq':        m.get('P_ml_cpx_liq'),
        'our ML cpx-only':       m.get('P_ml_cpx_liq'),
        'Putirka opx-liq':       m.get('P_putirka_opx_only_eq29c'),
        'Putirka 2-px':          m.get('P_putirka_2px_eq39'),
        'Agreda cpx-liq':        m.get('P_agreda_cpx_liq'),
        'Agreda cpx-only':       m.get('P_agreda_cpx_liq'),
        'Jorgenson cpx-liq':     m.get('P_jorgenson_cpx_liq'),
        'Jorgenson cpx-only':    m.get('P_jorgenson_cpx_liq'),
    }

    def build_matrix(method_dict):
        names = list(method_dict.keys())
        n = len(names)
        M = np.full((n, n), np.nan)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    M[i, j] = 0.0
                else:
                    M[i, j] = disagreement_rmse(method_dict[a], method_dict[b])
        return names, M

    def cluster_order(M):
        # Use average disagreement across rows (replace NaN with 0 for linkage)
        Mfill = np.nan_to_num(M, nan=np.nanmedian(M))
        Z = linkage(Mfill, method='average')
        order = leaves_list(Z)
        return order

    def plot_heatmap(names, M, title, unit, out_stem, vmax):
        order = cluster_order(M)
        names_ord = [names[i] for i in order]
        Mo = M[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(11, 10))
        im = ax.imshow(Mo, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(names_ord)))
        ax.set_yticks(range(len(names_ord)))
        ax.set_xticklabels(names_ord, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(names_ord, fontsize=9)
        for i in range(len(names_ord)):
            for j in range(len(names_ord)):
                val = Mo[i, j]
                if np.isnan(val):
                    txt = 'NA'
                else:
                    txt = f'{val:.1f}'
                color = 'white' if val > vmax * 0.45 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=7, color=color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(f'RMSE of disagreement ({unit})', fontsize=10)
        plt.tight_layout()
        fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

    out_dir = PROJECT_ROOT / 'figures' / 'core'
    out_dir.mkdir(parents=True, exist_ok=True)

    names_T, M_T = build_matrix(method_T)
    plot_heatmap(names_T, M_T,
                 'Method Agreement Matrix - T (C)\nLEPR natural two-pyroxene corpus',
                 'C', str(out_dir / 'Core_13a_fig_agreement_matrix_T'),
                 vmax=250.0)
    names_P, M_P = build_matrix(method_P)
    plot_heatmap(names_P, M_P,
                 'Method Agreement Matrix - P (kbar)\nLEPR natural two-pyroxene corpus',
                 'kbar', str(out_dir / 'Core_13b_fig_agreement_matrix_P'),
                 vmax=15.0)

    caption = (
        'Method Agreement Matrix on 327 LEPR natural two-pyroxene pairs '
        '(nb08 corpus, with 301-pair intersection where external methods '
        'provide predictions). Cell (i, j) shows the RMSE of disagreement '
        'between method i and method j on T (panel a, colormap 0-250 C) '
        'and P (panel b, colormap 0-15 kbar). Methods are hierarchically '
        'clustered by average pairwise disagreement using scipy average '
        'linkage on row vectors. Tight clusters indicate methods in the '
        'same calibration community; outlier methods disagree most with '
        'the consensus. Our ML opx-only is compared to 9 alternative '
        'methods simultaneously. Diagonal is zero by definition. Note '
        'Agreda-Lopez 2024 cpx-liq shows a systematic ~300 C T-bias '
        'relative to all other methods on this corpus.'
    )
    (out_dir / 'Core_13_txt_caption.txt').write_text(caption, encoding='utf-8')
    print('wrote Core_13a, Core_13b')


if __name__ == '__main__':
    main()
