"""Publication-consistent matplotlib style and layout helpers.

`apply_pub_style()` sets the locked rcParams (font face/sizes, axes weight,
grid, legend frame, savefig DPI). Builders call it once at module import.

`make_fig(kind, ...)` returns (fig, axes) at JGR ML & Computation column
sizes with the canonical spacing recipe baked in. Builders never call
`plt.subplots(figsize=...)` or `plt.subplots_adjust(...)` directly.

`add_grid(ax, axis='y')` is the canonical way to enable the grid; it relies
on the rcParams style and the global axisbelow=True so bars always render
on top of grid lines.

JGR submission mode
-------------------
Setting the env var `JGR_PRINT=1` switches the renderer to AGU column widths
(2-col max = 17.5 cm = 6.89 in) so figures land at submission-ready scale
with absolute font sizes preserved. The output directory also redirects to
`figures/opx_only/jgr/` so preview and print versions live side-by-side.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


# AGU column widths (inches).
JGR_2COL_IN = 6.89
JGR_1COL_IN = 3.39


def is_jgr_mode() -> bool:
    """Return True if the JGR_PRINT env var is set to '1'."""
    return os.environ.get('JGR_PRINT') == '1'


def apply_pub_style() -> None:
    """Apply the canonical publication style. Call once at import time.

    In JGR mode we shrink absolute font sizes slightly so a 6.89-inch
    canvas keeps body text at AGU's preferred 8 pt and headings at 10 pt.
    """
    base = {
        'font.family':           'sans-serif',
        'font.sans-serif':       ['DejaVu Sans', 'Arial', 'Helvetica',
                                  'Liberation Sans'],
        'font.size':             10,
        'axes.titlesize':        11,
        'axes.titleweight':      'bold',
        'axes.titlelocation':    'left',
        'axes.titlepad':         6,
        'axes.labelsize':        10,
        'axes.labelweight':      'normal',
        'axes.linewidth':        0.8,
        'axes.edgecolor':        '#333333',
        'axes.axisbelow':        True,
        'xtick.labelsize':       9,
        'ytick.labelsize':       9,
        'xtick.color':           '#333333',
        'ytick.color':           '#333333',
        'legend.fontsize':       9,
        'legend.title_fontsize': 9,
        'legend.frameon':        True,
        'legend.framealpha':     0.95,
        'legend.edgecolor':      '#888888',
        'legend.borderpad':      0.5,
        'legend.handletextpad':  0.6,
        'legend.columnspacing':  1.4,
        'figure.titlesize':      13,
        'figure.titleweight':    'bold',
        'figure.dpi':            100,
        'savefig.dpi':           300,
        'savefig.bbox':          'tight',
        'lines.linewidth':       1.2,
        'patch.linewidth':       0.5,
        'patch.edgecolor':       '#333333',
        'grid.linewidth':        0.5,
        'grid.alpha':            0.35,
        'grid.color':            '#bbbbbb',
        'grid.linestyle':        ':',
        'errorbar.capsize':      3.5,
    }
    if is_jgr_mode():
        # Tighter type at JGR print sizes — keeps tick labels at ~7 pt and
        # body text at ~8 pt when the canvas is rendered at 6.89 in wide.
        base.update({
            'font.size':             8,
            'axes.titlesize':        9,
            'axes.labelsize':        8,
            'xtick.labelsize':       7,
            'ytick.labelsize':       7,
            'legend.fontsize':       7,
            'legend.title_fontsize': 7,
            'figure.titlesize':      10,
            'savefig.dpi':           600,
        })
    mpl.rcParams.update(base)


# Layout dimensions are authored at preview width; `_jgr_scale_figsize`
# applies a uniform shrink in JGR mode so absolute font sizes (set in
# `apply_pub_style`) land at AGU's print column widths.
LAYOUTS: dict[str, dict[str, Any]] = {
    'single':  dict(figsize=(7.0, 5.0),
                    rect=(0, 0.05, 1, 0.94)),
    'wide':    dict(figsize=(7.5, 4.5),
                    rect=(0, 0.06, 1, 0.94)),
    'tall':    dict(figsize=(7.0, 8.0),
                    rect=(0, 0.05, 1, 0.95)),
    'two_col': dict(figsize=(11.0, 9.5),
                    subplots=dict(top=0.88, bottom=0.13,
                                  left=0.08, right=0.97,
                                  hspace=0.45, wspace=0.22)),
    'two_row': dict(figsize=(7.5, 5.5),
                    rect=(0, 0.10, 1, 0.94),
                    hspace=0.30, wspace=0.25),
    'wide_2x4': dict(figsize=(7.5, 5.0),
                     rect=(0, 0.10, 1, 0.92),
                     hspace=0.45, wspace=0.30),
}


def jgr_figsize(figsize: tuple[float, float], *,
                keep_size: bool = False) -> tuple[float, float]:
    """Shrink (w, h) so width fits AGU's 2-column maximum (6.89 in) when
    JGR mode is active. No-op otherwise. Aspect ratio is preserved.

    Pass ``keep_size=True`` for figures whose internal coordinates are
    canvas-dependent (e.g. the methods-flowchart schematic with absolute
    box positions in [0, 14]). Such figures stay at their authored size;
    AGU will scale them to column width at typesetting time.
    """
    if not is_jgr_mode() or keep_size:
        return figsize
    w, h = figsize
    if w <= JGR_2COL_IN:
        return figsize
    scale = JGR_2COL_IN / w
    return (w * scale, h * scale)


def resolve_out_dir(project_root: Path) -> Path:
    """Return the figures output dir, redirecting to `jgr/` in JGR mode.

    Builders should use this instead of hardcoding `figures/opx_only/`.
    """
    base = project_root / 'figures' / 'opx_only'
    if is_jgr_mode():
        base = base / 'jgr'
    base.mkdir(parents=True, exist_ok=True)
    return base


def _scale_subplots_for_jgr(adjust: dict[str, Any]) -> dict[str, Any]:
    """In JGR mode, give multi-line suptitles more headroom by lowering
    the figure 'top' fraction, and bump the inter-panel gap so panel
    titles in the lower row clear tick labels of the row above. The
    fractional spacing that worked at the 11" preview canvas crowds the
    suptitle and inter-row gutter when scaled down to 6.89"."""
    if not is_jgr_mode():
        return adjust
    out = dict(adjust)
    if 'top' in out:
        out['top'] = max(0.0, out['top'] - 0.04)
    if 'bottom' in out:
        out['bottom'] = max(0.0, out['bottom'] + 0.02)
    if 'hspace' in out:
        # 25% extra vertical gutter — the same fractional gutter is a
        # smaller absolute gap on the shrunk JGR canvas.
        out['hspace'] = out['hspace'] * 1.25
    return out


def make_fig(kind: str, *, nrows: int = 1, ncols: int = 1):
    """Return (fig, axes) sized for JGR with locked spacing.

    For multi-row layouts we set explicit top/bottom/left/right via
    subplots_adjust so multi-line panel titles never clip into the
    previous row's tick labels (which tight_layout reflows imperfectly).
    """
    cfg = LAYOUTS[kind]
    figsize = jgr_figsize(cfg['figsize'])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if 'subplots' in cfg:
        plt.subplots_adjust(**_scale_subplots_for_jgr(cfg['subplots']))
    else:
        adjust = {k: cfg[k] for k in ('hspace', 'wspace') if k in cfg}
        if adjust:
            plt.subplots_adjust(**adjust)
        rect = cfg['rect']
        if is_jgr_mode():
            # Lift the rect top so suptitle headroom survives the shrink.
            rect = (rect[0], rect[1] + 0.02, rect[2], max(0.0, rect[3] - 0.04))
        fig.tight_layout(rect=rect)
    return fig, axes


def add_grid(ax, axis: str = 'y') -> None:
    """Enable grid the canonical way. axis ∈ {'x', 'y', 'both'}."""
    ax.grid(True, axis=axis)


def jgr_top(top: float) -> float:
    """If JGR mode is active, lower the suptitle headroom fraction so a
    2-line suptitle clears the panel headers on the smaller canvas.
    Apply to a hand-coded subplots_adjust(top=...) or gridspec(top=...)."""
    if not is_jgr_mode():
        return top
    return max(0.0, top - 0.04)


def jgr_bottom(bottom: float) -> float:
    """Companion to jgr_top — bumps the bottom margin slightly so a
    legend below the panels does not collide with tick labels."""
    if not is_jgr_mode():
        return bottom
    return min(1.0, bottom + 0.02)
