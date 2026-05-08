"""Canonical model order, color palette, and role-style helper.

One color per entity, locked across every figure. Pre-correction state is
encoded by alpha + hatch, never by hue.

- ML families use Okabe-Ito where possible. MLP gets brown (tab10 9th slot)
  so gray is reserved exclusively for Putirka.
- TabPFN is a family, not a separate track; vermillion is its locked hue.
- Putirka is the comparator and is always neutral gray.

Imported by every figure builder. Builders never define their own
PUTIRKA_C / PRE_C / POST_C / OK_RED constants.
"""
from __future__ import annotations

import matplotlib.patches as mpatches

# Family display order, used wherever bars or columns are laid out by family.
MODEL_ORDER = [
    'ElasticNet',
    'RF',
    'ERT',
    'GB',
    'XGB',
    'LightGBM',
    'CatBoost',
    'MLP',
    'TabPFN',
]

# Okabe-Ito 8-color qualitative palette (colorblind-safe).
# Reference: Okabe & Ito (2008) https://jfly.uni-koeln.de/color/
OKABE_ITO = {
    'black':      '#000000',
    'orange':     '#E69F00',
    'sky_blue':   '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'pink':       '#CC79A7',
}

# Locked family color assignments. Reused everywhere a family is named.
MODEL_COLORS = {
    'ElasticNet': OKABE_ITO['black'],
    'RF':         OKABE_ITO['blue'],
    'ERT':        OKABE_ITO['sky_blue'],
    'GB':         OKABE_ITO['green'],
    'XGB':        OKABE_ITO['yellow'],
    'LightGBM':   OKABE_ITO['pink'],
    'CatBoost':   OKABE_ITO['orange'],
    'MLP':        '#8C564B',                # tab10 brown; not OI; reserves gray for Putirka
    'TabPFN':     OKABE_ITO['vermillion'],
    'Putirka':    '#666666',                # neutral gray; comparator
}

PUTIRKA_C = MODEL_COLORS['Putirka']

# Locked regime palette (matches dataset_map convention; bias_residuals is
# updated to follow). Used wherever regime is a color encoding.
REGIME_COLORS = {
    'shallow_crustal':     OKABE_ITO['sky_blue'],
    'deep_crustal_MASH':   OKABE_ITO['green'],
    'lithospheric_mantle': OKABE_ITO['orange'],
    'deeper_mantle':       OKABE_ITO['vermillion'],
}


def color_for(model: str, default: str = '#999999') -> str:
    return MODEL_COLORS.get(model, default)


def role_style(role: str, family: str | None = None
               ) -> tuple[str, float, str | None]:
    """Return (color, alpha, hatch) for any candidate role.

    role ∈ {'pre', 'post', 'putirka'}. For 'pre' / 'post', a family must be
    supplied; the bar inherits the family color. Pre-correction state uses
    alpha 0.45 + '///' hatch so the correction-state distinction is
    grayscale-safe and orthogonal to the hue-encoded family identity.
    """
    if role == 'putirka':
        return (MODEL_COLORS['Putirka'], 1.0, None)
    if family is None:
        raise ValueError(f"role {role!r} requires a family")
    color = MODEL_COLORS.get(family, '#999999')
    if role == 'pre':
        return (color, 0.45, '///')
    if role == 'post':
        return (color, 1.0, None)
    raise ValueError(f"unknown role: {role!r}")


def family_from_method(method: str) -> str:
    """Parse 'ElasticNet/raw' or 'TabPFN' into a MODEL_ORDER family."""
    if not method or method == 'nan':
        return ''
    if '/' in method:
        return method.split('/')[0]
    return method
