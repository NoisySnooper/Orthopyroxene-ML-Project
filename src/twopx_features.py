"""Feature engineering for two-pyroxene (opx + cpx) training frames.

Assumes a merged DataFrame where opx oxides carry `_opx` suffix, cpx
oxides carry `_cpx` suffix, and cation columns likewise. The prep script
`scripts/v10_phase_e_twopx_prep.py` produces this shape.

Feature sets:
  raw              : 9 opx + 9 cpx oxides + per-mineral engineered
  alr              : log-ratio normalized per mineral (SiO2 denom)
  pwlr             : pairwise log-ratios within each mineral + cross-mineral
  twopx_components : structural + KD features only

Public API: build_twopx_feature_matrix(df, feature_set) -> (X, names)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import OXIDE_MASSES

OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO',
          'MgO', 'CaO', 'Na2O']

ENGINEERED_OPX = ['Mg_num_opx', 'Al_IV_opx', 'Al_VI_opx',
                  'En_opx', 'Fs_opx', 'Wo_opx']
ENGINEERED_CPX = ['Mg_num_cpx', 'Jd', 'Di', 'Aeg', 'CaTs', 'En_cpx', 'Fs_cpx']
COMPONENT_FEATURES = [
    'KD_FeMg_opx_cpx',
    'Mg_num_opx', 'Mg_num_cpx', 'Delta_Mg_num',
    'Ca_ratio', 'Al_ratio', 'Na_ratio',
    'En_opx', 'Fs_opx', 'Wo_opx',
    'En_cpx', 'Fs_cpx', 'Di', 'Jd',
]


def compute_kd_femg(df: pd.DataFrame) -> pd.Series:
    """Putirka 2008 KD_Fe-Mg(opx-cpx) = (Fe/Mg)_opx / (Fe/Mg)_cpx.

    Uses mole ratios from FeO_total/MgO weight fractions in each mineral.
    Nominal equilibrium value 1.09 +/- 0.14 (Brey & Kohler 1990).
    """
    fe_opx = pd.to_numeric(df['FeO_total_opx'], errors='coerce') / OXIDE_MASSES['FeO_total']
    mg_opx = pd.to_numeric(df['MgO_opx'], errors='coerce') / OXIDE_MASSES['MgO']
    fe_cpx = pd.to_numeric(df['FeO_total_cpx'], errors='coerce') / OXIDE_MASSES['FeO_total']
    mg_cpx = pd.to_numeric(df['MgO_cpx'], errors='coerce') / OXIDE_MASSES['MgO']
    femg_opx = fe_opx / mg_opx.replace(0, np.nan)
    femg_cpx = fe_cpx / mg_cpx.replace(0, np.nan)
    return femg_opx / femg_cpx.replace(0, np.nan)


def add_twopx_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add Delta_Mg_num, Ca/Al/Na ratios, KD_FeMg_opx_cpx.

    Expects Mg_num_opx and Mg_num_cpx present (propagated by data-prep).
    """
    out = df.copy()
    out['Delta_Mg_num'] = out['Mg_num_cpx'] - out['Mg_num_opx']
    out['Ca_ratio'] = out['CaO_opx'] / out['CaO_cpx'].replace(0, np.nan)
    out['Al_ratio'] = out['Al2O3_opx'] / out['Al2O3_cpx'].replace(0, np.nan)
    out['Na_ratio'] = out['Na2O_opx'] / out['Na2O_cpx'].replace(0, np.nan)
    out['KD_FeMg_opx_cpx'] = compute_kd_femg(out)
    for c in ['Delta_Mg_num', 'Ca_ratio', 'Al_ratio', 'Na_ratio', 'KD_FeMg_opx_cpx']:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _make_raw_twopx(df):
    X_list, names = [], []
    for ox in OXIDES:
        for suf in ('_opx', '_cpx'):
            col = f'{ox}{suf}'
            if col in df.columns:
                X_list.append(df[col].fillna(0).values)
                names.append(f'raw_{col}')
    for eng in ENGINEERED_OPX + ENGINEERED_CPX:
        if eng in df.columns:
            X_list.append(df[eng].fillna(0).values)
            names.append(eng)
    X = np.column_stack(X_list)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _make_alr_twopx(df, eps=1e-3):
    X_list, names = [], []
    for mineral, suf in [('opx', '_opx'), ('cpx', '_cpx')]:
        denom_col = f'SiO2{suf}'
        denom = df[denom_col].replace(0, np.nan).fillna(df[denom_col].mean())
        for ox in OXIDES:
            if ox == 'SiO2':
                continue
            col = f'{ox}{suf}'
            if col not in df.columns:
                continue
            numer = df[col].replace(0, np.nan)
            fill = eps * max(numer.mean(skipna=True), 1e-6)
            numer = numer.fillna(fill)
            X_list.append(np.log(numer / denom).values)
            names.append(f'alr_{col}')
    for eng in ENGINEERED_OPX + ENGINEERED_CPX:
        if eng in df.columns:
            X_list.append(df[eng].fillna(0).values)
            names.append(eng)
    X = np.column_stack(X_list)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _make_pwlr_twopx(df, eps=1e-3):
    def safe(col):
        s = df[col].replace(0, np.nan)
        fill = eps * max(s.mean(skipna=True), 1e-6)
        return s.fillna(fill).values

    X_list, names = [], []
    for suf in ('_opx', '_cpx'):
        present = [ox for ox in OXIDES if f'{ox}{suf}' in df.columns]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                X_list.append(np.log(safe(f'{a}{suf}') / safe(f'{b}{suf}')))
                names.append(f'pwlr_{a}{suf}_{b}{suf}')
    # Cross-mineral ox_opx / ox_cpx.
    for ox in OXIDES:
        a = f'{ox}_opx'
        b = f'{ox}_cpx'
        if a in df.columns and b in df.columns:
            X_list.append(np.log(safe(a) / safe(b)))
            names.append(f'pwlr_cross_{ox}')
    for eng in ENGINEERED_OPX + ENGINEERED_CPX:
        if eng in df.columns:
            X_list.append(df[eng].fillna(0).values)
            names.append(eng)
    X = np.column_stack(X_list)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _make_components_twopx(df):
    X_list, names = [], []
    for c in COMPONENT_FEATURES:
        if c in df.columns:
            X_list.append(df[c].fillna(0).values)
            names.append(c)
    X = np.column_stack(X_list)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


TWOPX_FEATURE_FUNCS = {
    'raw':              _make_raw_twopx,
    'alr':              _make_alr_twopx,
    'pwlr':             _make_pwlr_twopx,
    'twopx_components': _make_components_twopx,
}


def build_twopx_feature_matrix(df, feature_set):
    if feature_set not in TWOPX_FEATURE_FUNCS:
        raise ValueError(f'unknown feature_set: {feature_set!r}')
    return TWOPX_FEATURE_FUNCS[feature_set](df)
