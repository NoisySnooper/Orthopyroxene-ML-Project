"""Publication-consistent matplotlib style for all core figures.

Call `apply_pub_style()` once at module import time before any plotting.
Keeps font face, sizes, weights, and grid behaviour consistent across
the 12 Core_* figures so the JGR ML & Computation submission reads
as a single matched set.
"""
from __future__ import annotations

import matplotlib as mpl


def apply_pub_style() -> None:
    """Apply the canonical publication style (DejaVu Sans at consistent sizes)."""
    mpl.rcParams.update({
        'font.family':         'sans-serif',
        'font.sans-serif':     ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'font.size':           10,
        'axes.titlesize':      11,
        'axes.titleweight':    'bold',
        'axes.labelsize':      10,
        'axes.labelweight':    'normal',
        'axes.linewidth':      0.8,
        'axes.edgecolor':      '#333333',
        'xtick.labelsize':     9,
        'ytick.labelsize':     9,
        'xtick.color':         '#333333',
        'ytick.color':         '#333333',
        'legend.fontsize':     9,
        'legend.frameon':      True,
        'legend.framealpha':   0.92,
        'legend.edgecolor':    '#888888',
        'legend.borderpad':    0.5,
        'figure.titlesize':    13,
        'figure.titleweight':  'bold',
        'figure.dpi':          100,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'lines.linewidth':     1.2,
        'patch.linewidth':     0.5,
        'grid.linewidth':      0.5,
        'grid.alpha':          0.35,
        'grid.color':          '#bbbbbb',
    })
