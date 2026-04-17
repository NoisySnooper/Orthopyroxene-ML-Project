"""Universal pyroxene thermobarometer features (v10 Phase F).

45-dim fixed-length input (plan doc Section 1.2 lists this as "~48"
but the actual tally is 45: 42 phase features + 3 presence bits):
  9 opx oxides + 7 opx engineered
  + 9 cpx oxides + 7 cpx engineered
  + 8 liquid oxides + 2 liquid engineered (liq_Mg_num, H2O_Liq)
  + 3 presence bits (opx_present, cpx_present, liq_present)
  = 42 + 3 = 45

Feature set: `universal_raw` only (one set). Log-ratios are disabled
because zero-filled absent phases would produce infinities.

Public API:
  UNIVERSAL_FEATURE_COLUMNS   -> list of 48 column names
  build_universal_matrix(df)  -> (X, names)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPX_OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO',
              'MgO', 'CaO', 'Na2O']
CPX_OXIDES = OPX_OXIDES
LIQ_OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O']

OPX_ENG = ['Mg_num_opx', 'Al_IV_opx', 'Al_VI_opx',
           'En_opx', 'Fs_opx', 'Wo_opx', 'MgTs_opx']
CPX_ENG = ['Mg_num_cpx', 'Jd', 'Di', 'Aeg', 'CaTs', 'En_cpx', 'Fs_cpx']
LIQ_ENG = ['liq_Mg_num', 'H2O_Liq']
PRESENCE_BITS = ('opx_present', 'cpx_present', 'liq_present')


def _universal_columns():
    cols = []
    for ox in OPX_OXIDES:
        cols.append(f'{ox}_opx')
    cols.extend(OPX_ENG)
    for ox in CPX_OXIDES:
        cols.append(f'{ox}_cpx')
    cols.extend(CPX_ENG)
    for ox in LIQ_OXIDES:
        cols.append(f'liq_{ox}')
    cols.extend(LIQ_ENG)
    cols.extend(PRESENCE_BITS)
    return cols


UNIVERSAL_FEATURE_COLUMNS = _universal_columns()
assert len(UNIVERSAL_FEATURE_COLUMNS) == 45, \
    f'universal vector should be 45-dim, got {len(UNIVERSAL_FEATURE_COLUMNS)}'


def build_universal_matrix(df: pd.DataFrame):
    """Extract the 48 universal feature columns from df.

    Any missing column is treated as a zero-filled slot.
    """
    n = len(df)
    X = np.zeros((n, len(UNIVERSAL_FEATURE_COLUMNS)), dtype=float)
    for j, col in enumerate(UNIVERSAL_FEATURE_COLUMNS):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values
            X[:, j] = v
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, list(UNIVERSAL_FEATURE_COLUMNS)
