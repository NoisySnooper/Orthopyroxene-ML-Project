"""Shared matplotlib style and plotting helpers.

Defines the Paul Tol bright palette, model color assignments, a global
rcParams preset (`apply_style`), and common axis helpers (panel labels,
stats boxes, 1:1 reference line, regression line). `save_figure` and
`load_winning_config` are re-exported from `src.io_utils` and
`src.data` respectively so notebooks can import plotting-and-config
helpers from a single module.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

from src.data import (
    canonical_model_filename,
    load_canonical_model,
    load_winning_config,
)
from src.io_utils import save_figure

TOL_BRIGHT = {
    'blue':   '#4477AA',
    'cyan':   '#66CCEE',
    'green':  '#228833',
    'yellow': '#CCBB44',
    'red':    '#EE6677',
    'purple': '#AA3377',
    'gray':   '#BBBBBB',
}

# Okabe-Ito colorblind-safe palette (v5 canonical).
# Reference: Okabe & Ito 2008 "Color Universal Design" sample set.
OKABE_ITO = {
    'black':      '#000000',
    'orange':     '#E69F00',
    'sky_blue':   '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'reddish':    '#CC79A7',
    'gray':       '#999999',
}
OKABE_ITO_CYCLE = [
    OKABE_ITO['blue'], OKABE_ITO['orange'], OKABE_ITO['green'],
    OKABE_ITO['vermillion'], OKABE_ITO['sky_blue'], OKABE_ITO['reddish'],
    OKABE_ITO['yellow'], OKABE_ITO['gray'],
]

MODEL_COLORS = {
    'RF':  OKABE_ITO['blue'],
    'ERT': OKABE_ITO['green'],
    'XGB': OKABE_ITO['vermillion'],
    'GB':  OKABE_ITO['reddish'],
}

PUTIRKA_COLOR = OKABE_ITO['orange']
ML_COLOR = OKABE_ITO['blue']


def apply_style():
    """Set rcParams for manuscript-ready figures. Okabe-Ito palette is the
    default prop_cycle so every line/scatter without an explicit color is
    colorblind-safe by default."""
    from cycler import cycler
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': cycler(color=OKABE_ITO_CYCLE),
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'figure.titlesize': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def save_both(fig, path_no_ext, dpi=300):
    """Save a matplotlib figure as both PNG and PDF at 300 DPI.
    `path_no_ext` is a Path or string without extension."""
    from pathlib import Path as _Path
    p = _Path(str(path_no_ext))
    png = p.with_suffix('.png')
    pdf = p.with_suffix('.pdf')
    fig.savefig(png, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf,          bbox_inches='tight')
    return png, pdf


def panel_label(ax, label, x=0.02, y=0.98, fontsize=12):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))


def stats_box(ax, rmse=None, rmse_std=None, r2=None, slope=None, intercept=None,
              n=None, x=0.97, y=0.03, unit=''):
    lines = []
    if rmse is not None:
        if rmse_std is not None:
            lines.append(f'RMSE = {rmse:.2f} +/- {rmse_std:.2f} {unit}'.strip())
        else:
            lines.append(f'RMSE = {rmse:.2f} {unit}'.strip())
    if r2 is not None:
        lines.append(f'R2 = {r2:.3f}')
    if slope is not None:
        lines.append(f'slope = {slope:.2f}')
    if intercept is not None:
        lines.append(f'int = {intercept:.1f}')
    if n is not None:
        lines.append(f'n = {n}')
    text = '\n'.join(lines)
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8, va='bottom', ha='right', family='monospace',
            bbox=dict(facecolor='white', edgecolor='gray',
                      linewidth=0.5, alpha=0.9, pad=3))


def one_to_one(ax, color='black', lw=1.0, alpha=0.6, label='1:1'):
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], '-', color=color, lw=lw, alpha=alpha,
            zorder=0, label=label if label else None)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')


def regression_line(ax, x, y, color='gray', lw=0.8, ls='--'):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None, None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    xx = np.array([x[mask].min(), x[mask].max()])
    ax.plot(xx, slope * xx + intercept, color=color, lw=lw, ls=ls, zorder=1)
    return slope, intercept


def fmt_value(mean, std=None, unit='', decimals=2):
    if std is not None:
        return f"{mean:.{decimals}f} +/- {std:.{decimals}f} {unit}".strip()
    return f"{mean:.{decimals}f} {unit}".strip()


__all__ = [
    'TOL_BRIGHT', 'OKABE_ITO', 'OKABE_ITO_CYCLE',
    'MODEL_COLORS', 'PUTIRKA_COLOR', 'ML_COLOR',
    'apply_style', 'save_both',
    'panel_label', 'stats_box', 'one_to_one',
    'regression_line', 'fmt_value',
    'save_figure', 'load_winning_config',
    'canonical_model_filename', 'load_canonical_model',
]
