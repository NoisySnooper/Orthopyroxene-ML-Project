"""Canonical model order + Okabe-Ito color palette for all 9 families.

Every figure that plots per-model bars or lines imports from this module
so the color + order story is consistent across the manuscript.

TabPFN gets the emphasis color (vermillion) because it is the new
foundation-model entrant contrasted against the 8 tuned families, not
because it is a separate track.
"""
from __future__ import annotations

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

MODEL_COLORS = {
    'ElasticNet': OKABE_ITO['black'],       # linear baseline, neutral
    'RF':         OKABE_ITO['blue'],
    'ERT':        OKABE_ITO['sky_blue'],
    'GB':         OKABE_ITO['green'],
    'XGB':        OKABE_ITO['yellow'],
    'LightGBM':   OKABE_ITO['pink'],
    'CatBoost':   OKABE_ITO['orange'],
    'MLP':        '#999999',                # gray (9th slot; non-OI)
    'TabPFN':     OKABE_ITO['vermillion'],  # emphasis color
}

# Candidate-level palette for scorecard-style figures.
# Extends MODEL_COLORS with the comparison baselines.
CANDIDATE_COLORS = {
    'tuned_pre':     '#1f77b4',
    'tuned_post':    '#0072B2',
    'putirka':       '#666666',
    'tabpfn_pre':    '#F4A582',
    'tabpfn_post':   OKABE_ITO['vermillion'],
    'jorgenson':     OKABE_ITO['sky_blue'],
    'agreda':        OKABE_ITO['pink'],
    'wang':          OKABE_ITO['green'],
}

CANDIDATE_LABELS = {
    'tuned_pre':   'Tuned (pre)',
    'tuned_post':  'Tuned (post)',
    'putirka':     'Putirka',
    'tabpfn_pre':  'TabPFN (pre)',
    'tabpfn_post': 'TabPFN (post)',
    'jorgenson':   'Jorgenson cpx',
    'agreda':      'Agreda-Lopez',
    'wang':        'Wang',
}


def color_for(model: str, default: str = '#999999') -> str:
    return MODEL_COLORS.get(model, default)


def color_for_candidate(key: str, default: str = '#999999') -> str:
    return CANDIDATE_COLORS.get(key, default)
