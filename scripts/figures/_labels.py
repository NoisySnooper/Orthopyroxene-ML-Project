"""Locked terminology for figure labels and panel headers.

One canonical string per concept. Builders import from here so the same
regime is named the same way in every figure, and the same panel header
appears in identical typography across the paper.
"""
from __future__ import annotations

# Track display strings (figure labels). Caption prose may use different
# casing; the figure-side string is locked here.
TRACK_LABEL = {
    'opx_liq':  'Opx + Liquid',
    'opx_only': 'Opx only',
}

# Regime display strings. Long-form for axis/legend, short-form for
# tick labels. Use the canonical Unicode characters (≤, ≥, –, °).
REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle']

REGIME_LABEL = {
    'shallow_crustal':     'shallow (<5 kbar)',
    'deep_crustal_MASH':   'deep-MASH (5–15 kbar)',
    'lithospheric_mantle': 'litho (15–30 kbar)',
    'deeper_mantle':       'deeper (≥30 kbar)',
}

# Compact tick-label form (two-line); used where horizontal space is tight.
REGIME_TICK = {
    'shallow_crustal':     'shallow\n<5 kbar',
    'deep_crustal_MASH':   'MASH\n5–15',
    'lithospheric_mantle': 'litho\n15–30',
    'deeper_mantle':       'deeper\n≥30',
}

# Target unit strings.
TARGET_UNIT = {
    'T_C':    '°C',
    'P_kbar': 'kbar',
}

TARGET_LABEL = {
    'T_C':    'T',
    'P_kbar': 'P',
}


def panel_header(track: str, target: str, idx: str | None = None) -> str:
    """Canonical panel header e.g. '(a) Opx + Liquid · T (°C)'.

    idx is the panel letter ('a', 'b', 'c', 'd'); pass None to omit.
    """
    track_s = TRACK_LABEL[track]
    tgt = TARGET_LABEL[target]
    unit = TARGET_UNIT[target]
    body = f'{track_s} · {tgt} ({unit})'
    if idx:
        return f'({idx}) {body}'
    return body
