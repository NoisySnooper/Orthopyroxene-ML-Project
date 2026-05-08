"""Legend factory. One y-anchor, one frame style, one ncol policy.

Every legend in the paper goes through `add_below_legend` (figure-wide)
or uses `model_patch` / `role_patch` for handle construction inside an
axes-local legend. Builders never set frameon, framealpha, edgecolor,
fontsize, or bbox_to_anchor by hand.
"""
from __future__ import annotations

import matplotlib.patches as mpatches

from scripts.figures._model_palette import MODEL_COLORS, role_style


_PATCH_EDGE = '#333333'   # locked patch border for every legend handle


def add_below_legend(fig, handles, labels, *, ncol: int | None = None,
                     title: str | None = None):
    """Place a framed legend below the panels at the canonical y-anchor.

    ncol defaults to min(len(handles), 5) — caps row width so the legend
    never sprawls. Pass an explicit ncol if you need a specific layout.
    """
    if ncol is None:
        ncol = min(len(handles), 5)
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.005),
        ncol=ncol,
        title=title,
    )


def model_patch(family: str, *, label: str | None = None) -> mpatches.Patch:
    """Solid color swatch for a family, used in family-keyed legends."""
    return mpatches.Patch(
        facecolor=MODEL_COLORS[family],
        edgecolor=_PATCH_EDGE,
        label=label or family,
    )


def role_patch(role: str, *, family: str | None = None,
               label: str) -> mpatches.Patch:
    """Color + alpha + hatch swatch for a candidate role.

    role ∈ {'pre', 'post', 'putirka'}. For 'pre' / 'post' a family is
    required and the swatch inherits the family color with the canonical
    alpha+hatch encoding for correction state.
    """
    color, alpha, hatch = role_style(role, family)
    return mpatches.Patch(
        facecolor=color,
        alpha=alpha,
        hatch=hatch,
        edgecolor=_PATCH_EDGE,
        label=label,
    )
