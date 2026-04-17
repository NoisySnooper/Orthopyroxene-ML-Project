"""Feature engineering for clinopyroxene compositions (v10 Phase D).

Mirrors `src/features.py` for opx with cpx-specific engineered columns:
  Mg_num_cpx, Jd (jadeite), Di (diopside), Aeg (aegirine), CaTs,
  En_cpx (enstatite component), Fs_cpx (ferrosilite component).

Raw oxide list mirrors opx (9 majors) since ExPetDB schema is the same.
Cation recalculation is 6-oxygen basis, same as opx.

Public API:
  build_cpx_feature_matrix(df, feature_set, use_liq) -> (X, names)

`feature_set` is one of: 'raw', 'alr', 'pwlr'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    OXIDES_LIQ, OXIDE_MASSES, cation_recalc_6oxy,
)

OXIDES_CPX = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO',
              'MgO', 'CaO', 'Na2O']
ENGINEERED_CPX = ['Mg_num_cpx', 'Jd', 'Di', 'Aeg', 'CaTs', 'En_cpx', 'Fs_cpx']


def add_cpx_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cpx component fractions using cation moles on 6-oxygen basis.

    Requires `{ox}_cat` columns from `cation_recalc_6oxy(df, oxides=OXIDES_CPX)`.

    Component estimates (simplified Lindsley-style projection):
      Jd   = min(Na_cat, Al_VI)                    # NaAlSi2O6
      Aeg  = max(Na_cat - Al_VI, 0)                # NaFe3+Si2O6; proxied via Na_cat leftover
      CaTs = Al_IV_leftover after MgTs             # CaAl2SiO6
      Di   = Ca_cat - CaTs  (or - (CaTs + Aeg))    # CaMgSi2O6
      En_cpx = Mg_cat - Di_Mg                      # residual MgSiO3 (enstatite)
      Fs_cpx = Fe_cat  residual                    # residual FeSiO3 (ferrosilite)

    These are approximate component fractions, not thermodynamic end-member
    proportions. They provide the same kind of compositional summary as
    En/Fs/Wo does for opx: correlated features that help ML models.
    """
    out = df.copy()

    out['Mg_num_cpx'] = (
        out['MgO'] / OXIDE_MASSES['MgO']
    ) / (
        out['MgO'] / OXIDE_MASSES['MgO']
        + out['FeO_total'] / OXIDE_MASSES['FeO_total']
    )

    Si = out.get('SiO2_cat', 0)
    Al = out.get('Al2O3_cat', 0)
    Na = out.get('Na2O_cat', 0)
    Ca = out.get('CaO_cat', 0)
    Mg = out.get('MgO_cat', 0)
    Fe = out.get('FeO_total_cat', 0)
    Cr = out.get('Cr2O3_cat', 0)

    Al_IV = np.maximum(2.0 - Si, 0)
    Al_VI = np.maximum(Al - Al_IV, 0)

    out['Jd']   = np.minimum(Na, Al_VI)
    out['Aeg']  = np.maximum(Na - out['Jd'], 0)
    out['CaTs'] = np.minimum(Al_IV, Al_VI - out['Jd'] + 1e-9).clip(lower=0)
    # Diopside = Ca left over after CaTs (each CaTs consumes 1 Ca).
    out['Di']   = np.maximum(Ca - out['CaTs'], 0)

    # En/Fs cpx residual: Mg / Fe not accounted for by Di.
    # Di stoich: 1 Ca + 1 Mg + 2 Si per unit. Assume Mg balances Ca 1:1 for Di.
    Mg_Di = np.minimum(Mg, out['Di'])
    out['En_cpx'] = np.maximum(Mg - Mg_Di, 0)
    out['Fs_cpx'] = np.maximum(Fe, 0)  # simple pass-through; Fe not in Di
    return out


def lepr_to_cpx_training_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename LEPR `_Cpx` columns to unsuffixed training schema.

    Mirrors lepr_to_training_schema in src/features.py but for cpx.
    Assumes _Liq columns already renamed to liq_ (or not present).
    """
    rename = {}
    for col in df.columns:
        if col == 'FeOt_Cpx':
            target = 'FeO_total'
        elif col.endswith('_Cpx'):
            target = col[: -len('_Cpx')]
        else:
            continue
        if target not in df.columns and target not in rename.values():
            rename[col] = target
    return df.rename(columns=rename)


def expetdb_cpx_to_training(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw cpx schema -> training schema + engineered cols.

    Input df is expected to have unsuffixed oxide columns (SiO2, Al2O3, etc.)
    at the output of data-prep merging. Adds cation_recalc + engineered.
    """
    with_cats = cation_recalc_6oxy(df, oxides=OXIDES_CPX)
    return add_cpx_engineered_features(with_cats)


# ---------------------------------------------------------------------------
# Feature-matrix builders
# ---------------------------------------------------------------------------

def _make_raw_cpx(df, use_liq=False):
    X_list, names = [], []
    for ox in OXIDES_CPX:
        if ox in df.columns:
            X_list.append(df[ox].fillna(0).values)
            names.append(f'raw_{ox}')
    for eng in ENGINEERED_CPX:
        if eng in df.columns:
            X_list.append(df[eng].fillna(0).values)
            names.append(eng)
    if use_liq:
        for ox in OXIDES_LIQ:
            col = f'liq_{ox}'
            if col in df.columns:
                X_list.append(df[col].fillna(0).values)
                names.append(f'raw_liq_{ox}')
        if 'liq_Mg_num' in df.columns:
            X_list.append(df['liq_Mg_num'].fillna(0).values)
            names.append('liq_Mg_num')
    X = np.column_stack(X_list)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _make_alr_cpx(df, use_liq=False, eps=1e-3):
    oxides = [ox for ox in OXIDES_CPX if ox != 'SiO2']
    denom = df['SiO2'].replace(0, np.nan).fillna(df['SiO2'].mean())
    X_list, names = [], []
    for ox in oxides:
        if ox not in df.columns:
            continue
        numer = df[ox].replace(0, np.nan)
        fill = eps * max(numer.mean(skipna=True), 1e-6)
        numer = numer.fillna(fill)
        X_list.append(np.log(numer / denom).values)
        names.append(f'alr_{ox}')
    for eng in ENGINEERED_CPX:
        if eng in df.columns:
            X_list.append(df[eng].fillna(0).values)
            names.append(eng)
    if use_liq:
        oxides_liq = [ox for ox in OXIDES_LIQ if ox != 'SiO2']
        liq_denom = df['liq_SiO2'].replace(0, np.nan).fillna(df['liq_SiO2'].mean())
        for ox in oxides_liq:
            col = f'liq_{ox}'
            if col not in df.columns:
                continue
            numer = df[col].replace(0, np.nan)
            fill = eps * max(numer.mean(skipna=True), 1e-6)
            numer = numer.fillna(fill)
            X_list.append(np.log(numer / liq_denom).values)
            names.append(f'alr_liq_{ox}')
        if 'liq_Mg_num' in df.columns:
            X_list.append(df['liq_Mg_num'].fillna(0).values)
            names.append('liq_Mg_num')
    X = np.column_stack(X_list)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _make_pwlr_cpx(df, use_liq=False, eps=1e-3):
    def safe(frame, col, fallback_mean):
        s = frame[col].replace(0, np.nan)
        return s.fillna(eps * max(fallback_mean, 1e-6)).values

    cpx_present = [ox for ox in OXIDES_CPX if ox in df.columns]
    cpx_means = {ox: df[ox].mean(skipna=True) for ox in cpx_present}
    X_list, names = [], []
    for i, a in enumerate(cpx_present):
        for b in cpx_present[i + 1:]:
            av = safe(df, a, cpx_means[a])
            bv = safe(df, b, cpx_means[b])
            X_list.append(np.log(av / bv))
            names.append(f'pwlr_{a}_{b}')
    for eng in ENGINEERED_CPX:
        if eng in df.columns:
            X_list.append(df[eng].fillna(0).values)
            names.append(eng)
    if use_liq:
        liq_present = [ox for ox in OXIDES_LIQ if f'liq_{ox}' in df.columns]
        liq_means = {ox: df[f'liq_{ox}'].mean(skipna=True) for ox in liq_present}
        for i, a in enumerate(liq_present):
            for b in liq_present[i + 1:]:
                av = safe(df, f'liq_{a}', liq_means[a])
                bv = safe(df, f'liq_{b}', liq_means[b])
                X_list.append(np.log(av / bv))
                names.append(f'pwlr_liq_{a}_{b}')
        if 'liq_Mg_num' in df.columns:
            X_list.append(df['liq_Mg_num'].fillna(0).values)
            names.append('liq_Mg_num')
    X = np.column_stack(X_list)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


CPX_FEATURE_FUNCS = {
    'raw':  _make_raw_cpx,
    'alr':  _make_alr_cpx,
    'pwlr': _make_pwlr_cpx,
}


def build_cpx_feature_matrix(df, feature_set, use_liq=True):
    """Dispatcher parallel to build_feature_matrix in src/features.py."""
    if feature_set not in CPX_FEATURE_FUNCS:
        raise ValueError(f'unknown feature_set: {feature_set!r}')
    return CPX_FEATURE_FUNCS[feature_set](df, use_liq=use_liq)
