"""Shared helpers for Core_14a (opx) and Core_14b (cpx) pairing matrix figures.

Each figure is a stack of section blocks. A section is a header strip
(spans all 3 columns) followed by N data rows (3 panels each: P, T,
disequilibrium). Layout is built with GridSpec so header strips are
shorter than data rows.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._style import apply_pub_style  # noqa: E402
from scripts.figures._model_palette import OKABE_ITO  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_LINE_C = '#333333'
BOX_C = OKABE_ITO['blue']

P_LIM = (-5.0, 65.0)
T_LIM = (600.0, 1900.0)
DP_LIM = (-25.0, 25.0)
DT_LIM = (-500.0, 500.0)

TITLE_WRAP = 60
XLABEL_WRAP = 42
TITLE_PAD = 8
TITLE_FONT = 8.5
LABEL_FONT = 8

OKABE_CYCLE = [
    OKABE_ITO['blue'],
    OKABE_ITO['pink'],
    OKABE_ITO['orange'],
    OKABE_ITO['green'],
    OKABE_ITO['vermillion'],
    OKABE_ITO['sky_blue'],
    OKABE_ITO['yellow'],
    '#7F7F7F',
]


COLS = {
    'T_ml_opx_liq':         lambda m: m.get('T_ml_opx_liq'),
    'P_ml_opx_liq':         lambda m: m.get('P_ml_opx_liq'),
    'T_ml_opx_only':        lambda m: m.get('T_ml_opx_only'),
    'P_ml_opx_only':        lambda m: m.get('P_ml_opx_only'),
    'T_ml_cpx_liq':         lambda m: m.get('T_ml_cpx_liq'),
    'P_ml_cpx_liq':         lambda m: m.get('P_ml_cpx_liq'),
    'T_ml_cpx_only':        lambda m: m.get('T_ml_cpx_liq'),
    'P_ml_cpx_only':        lambda m: m.get('P_ml_cpx_liq'),
    'T_agreda_cpx_liq':     lambda m: m.get('T_agreda_cpx_liq'),
    'P_agreda_cpx_liq':     lambda m: m.get('P_agreda_cpx_liq'),
    'T_agreda_cpx_only':    lambda m: m.get('T_agreda_cpx_liq'),
    'P_agreda_cpx_only':    lambda m: m.get('P_agreda_cpx_liq'),
    'T_jorgenson_cpx_liq':  lambda m: m.get('T_jorgenson_cpx_liq'),
    'P_jorgenson_cpx_liq':  lambda m: m.get('P_jorgenson_cpx_liq'),
    'T_jorgenson_cpx_only': lambda m: m.get('T_jorgenson_cpx_liq'),
    'P_jorgenson_cpx_only': lambda m: m.get('P_jorgenson_cpx_liq'),
    'T_wang_cpx':           lambda m: m.get('T_wang_cpx_liq'),
    'P_wang_cpx':           lambda m: m.get('P_wang_cpx_liq'),
    'T_putirka_opx_liq':    lambda m: m.get('T_putirka_opx_liq_eq28a'),
    'P_putirka_opx_liq':    lambda m: pd.Series([np.nan] * len(m)),
    'T_putirka_2px':        lambda m: m.get('T_putirka_2px_eq36'),
    'P_putirka_2px':        lambda m: m.get('P_putirka_2px_eq39'),
    'T_brey_kohler':        lambda m: m.get('T_putirka_2px_eq36'),
    'P_brey_kohler':        lambda m: m.get('P_putirka_2px_eq39'),
    'T_putirka_cpx_only':   lambda m: m.get('T_putirka_2px_eq36'),
    'P_putirka_cpx_only':   lambda m: m.get('P_putirka_opx_only_eq29c'),
}


def load_corpus():
    nb8 = pd.read_csv('results/nb08_natural_predictions.csv')
    c11 = pd.read_csv('results/core11_extended_predictions.csv')
    return nb8.merge(c11, on='Experiment', how='outer',
                     suffixes=('_nb8', '_c11'))


def load_sigmas(default_T=50.0, default_P=3.0):
    sigma_T, sigma_P = default_T, default_P
    qhat_path = PROJECT_ROOT / 'results' / 'nb07_conformal_qhat.json'
    if qhat_path.exists():
        with open(qhat_path) as f:
            qh = json.load(f)
        sigma_T = float(qh.get('T_C', {}).get('qhat', sigma_T))
        sigma_P = float(qh.get('P_kbar', {}).get('qhat', sigma_P))
    return sigma_T, sigma_P


def _wrap(text, width):
    return '\n'.join(textwrap.fill(line, width=width)
                     for line in text.split('\n'))


def _series(corpus, col_key):
    if col_key is None:
        return None
    s = COLS[col_key](corpus)
    return np.asarray(s, dtype=float)


def scatter_1to1(ax, x, y, color, title, xlabel, ylabel, xlim):
    ax.set_xlim(xlim); ax.set_ylim(xlim)
    if x is None or y is None:
        ax.text(0.5, 0.5, 'no data column', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='0.35')
        n = 0
    else:
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        if n >= 2:
            ax.scatter(x[m], y[m], s=14, alpha=0.55, color=color,
                       edgecolor='black', linewidths=0.25,
                       label=f'LEPR natural (n={n})')
        else:
            ax.text(0.5, 0.5, 'no valid pairs', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='0.35')
    ax.plot(xlim, xlim, ls='--', lw=1, color=REF_LINE_C, alpha=0.8,
            zorder=1, label='1:1 identity')
    ax.axhline(0, color='#888888', lw=0.4)
    ax.axvline(0, color='#888888', lw=0.4)
    ax.set_title(_wrap(title, TITLE_WRAP),
                 loc='left', pad=TITLE_PAD, fontsize=TITLE_FONT)
    ax.set_xlabel(_wrap(xlabel, XLABEL_WRAP), fontsize=LABEL_FONT)
    ax.set_ylabel(_wrap(ylabel, XLABEL_WRAP), fontsize=LABEL_FONT)
    ax.tick_params(labelsize=7)
    if n >= 2:
        ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax.grid(True); ax.set_axisbelow(True)


def diseq_panel(ax, dP, dT, title, sigma_P, sigma_T, color):
    ax.set_xlim(DP_LIM); ax.set_ylim(DT_LIM)
    xlabel = f'DeltaP (kbar); box = +/- {2*sigma_P:.1f}'
    ylabel = f'DeltaT (C); box = +/- {2*sigma_T:.0f}'
    ax.axvspan(-2 * sigma_P, 2 * sigma_P, alpha=0.08, color=BOX_C)
    ax.axhspan(-2 * sigma_T, 2 * sigma_T, alpha=0.08, color=BOX_C)
    ax.axvline(0, color='k', lw=0.6); ax.axhline(0, color='k', lw=0.6)
    frac_txt = 'NA'
    has_pair = dP is not None and dT is not None
    if has_pair:
        m = np.isfinite(dP) & np.isfinite(dT)
        n = int(m.sum())
        if n > 0:
            ax.scatter(dP[m], dT[m], s=14, alpha=0.55, color=color,
                       edgecolor='black', linewidths=0.25,
                       label=f'LEPR natural (n={n})')
            in_box = (np.abs(dP[m]) < 2 * sigma_P) & (np.abs(dT[m]) < 2 * sigma_T)
            frac_txt = f'{float(in_box.mean()) * 100:.1f}%'
            ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    else:
        ax.text(0.5, 0.5, 'no P+T pair for this row',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='0.35')
    ax.set_title(_wrap(f'{title}. equilib. box {frac_txt}', TITLE_WRAP),
                 loc='left', pad=TITLE_PAD, fontsize=TITLE_FONT)
    ax.set_xlabel(_wrap(xlabel, XLABEL_WRAP), fontsize=LABEL_FONT)
    ax.set_ylabel(_wrap(ylabel, XLABEL_WRAP), fontsize=LABEL_FONT)
    ax.tick_params(labelsize=7)
    ax.grid(True); ax.set_axisbelow(True)


def _draw_data_row(fig, gs, grid_row, row_spec, color, corpus, pair_df,
                   sigma_P, sigma_T):
    label, ma, mb, ta, tb, pa, pb = row_spec
    axP = fig.add_subplot(gs[grid_row, 0])
    axT = fig.add_subplot(gs[grid_row, 1])
    axD = fig.add_subplot(gs[grid_row, 2])

    row_info = pair_df[pair_df.row_label == label]
    if len(row_info):
        r = row_info.iloc[0]
        t_rmse = r.get('T_rmse_disagreement', np.nan)
        p_rmse = r.get('P_rmse_disagreement', np.nan)
        t_rmse_txt = (f'{t_rmse:.1f} C [{r["T_ci_lo"]:.1f}, {r["T_ci_hi"]:.1f}]'
                      if np.isfinite(t_rmse) else 'NA')
        p_rmse_txt = (f'{p_rmse:.2f} kbar [{r["P_ci_lo"]:.2f}, {r["P_ci_hi"]:.2f}]'
                      if np.isfinite(p_rmse) else 'NA')
    else:
        t_rmse_txt = 'NA'; p_rmse_txt = 'NA'

    xP = _series(corpus, pb) if pb else None
    yP = _series(corpus, pa) if pa else None
    xT = _series(corpus, tb) if tb else None
    yT = _series(corpus, ta) if ta else None

    title_P = (f'({label}1) P: {ma} vs {mb}. '
               f'RMSE disagree = {p_rmse_txt}')
    scatter_1to1(axP, xP, yP, color, title_P,
                 f'P {mb} (kbar)', f'P {ma} (kbar)', P_LIM)

    title_T = (f'({label}2) T: {ma} vs {mb}. '
               f'RMSE disagree = {t_rmse_txt}')
    scatter_1to1(axT, xT, yT, color, title_T,
                 f'T {mb} (C)', f'T {ma} (C)', T_LIM)

    if yP is not None and xP is not None and yT is not None and xT is not None:
        dP = yP - xP
        dT = yT - xT
    else:
        dP = dT = None
    diseq_panel(axD, dP, dT,
                f'({label}3) Disequilibrium: {ma} minus {mb}',
                sigma_P, sigma_T, color=color)


def render_pairing_figure(out_stem: Path, suptitle: str,
                          sections: Iterable[tuple],
                          row_height: float = 3.2,
                          header_height: float = 0.7,
                          suptitle_height: float = 2.6):
    """Render a sectioned pairing figure.

    sections: list of (header_text, [row_spec, ...]) where each row_spec is
        (label, ma, mb, ta, tb, pa, pb).
    """
    corpus = load_corpus()
    pair_df = pd.read_csv('results/pairing_matrix_22rows.csv')
    sigma_T, sigma_P = load_sigmas()

    height_ratios = []
    slots = []
    for header_text, row_specs in sections:
        height_ratios.append(header_height)
        slots.append(('header', header_text))
        for rs in row_specs:
            height_ratios.append(1.0)
            slots.append(('row', rs))

    n_grid_rows = len(slots)
    fig_height = row_height * sum(height_ratios) + suptitle_height
    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(n_grid_rows, 3, height_ratios=height_ratios,
                          hspace=0.60, wspace=0.22)

    color_idx = 0
    for grid_row, (kind, payload) in enumerate(slots):
        if kind == 'header':
            ax = fig.add_subplot(gs[grid_row, :])
            ax.axis('off')
            ax.add_patch(plt.Rectangle((0, 0.15), 1, 0.7,
                                       transform=ax.transAxes,
                                       facecolor='#EEEEEE',
                                       edgecolor='#999999',
                                       linewidth=0.5,
                                       zorder=0))
            ax.text(0.5, 0.5, payload, ha='center', va='center',
                    transform=ax.transAxes, fontsize=13,
                    fontweight='bold', color='#222222')
        else:
            color = OKABE_CYCLE[color_idx % len(OKABE_CYCLE)]
            color_idx += 1
            _draw_data_row(fig, gs, grid_row, payload, color,
                           corpus, pair_df, sigma_P, sigma_T)

    suptitle_y = 1.0 - 0.40 / fig_height
    fig.suptitle(suptitle, y=suptitle_y, fontsize=13, va='top')
    top_frac = 1.0 - suptitle_height / fig_height
    plt.subplots_adjust(top=top_frac, bottom=0.015,
                        left=0.05, right=0.99)

    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=200)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return sigma_T, sigma_P
