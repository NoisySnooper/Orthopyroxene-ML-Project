"""Feature engineering for orthopyroxene compositions.

Three representations are supported:

raw
    Raw oxide weight percentages, plus engineered ratios carried through
    from cleaning (Mg#, Al_IV, Al_VI, En/Fs/Wo fractions, MgTs).
alr
    Additive log-ratio with SiO2 as the denominator. Multiplicative zero
    replacement (Martin-Fernandez et al. 2011).
pwlr
    Pairwise log-ratio. Every unique oxide pair contributes a log-ratio
    feature. This is the canonical winning representation per the NB03
    feature-set benchmark.

`augment_dataframe` applies EPMA-realistic Gaussian noise to raw oxide
columns (3% relative on majors, 8% on minors) and returns a concatenation
of clean plus n_aug-1 noised copies.

`build_feature_matrix(df, feature_set, use_liq)` is the dispatching entry
point used by downstream notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OXIDES_OPX = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO', 'MgO', 'CaO', 'Na2O']
OXIDES_LIQ = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O']
ENGINEERED = ['Mg_num', 'Al_IV', 'Al_VI', 'En_frac', 'Fs_frac', 'Wo_frac', 'MgTs']

MAJOR_OXIDES = {'SiO2', 'Al2O3', 'FeO_total', 'FeO', 'MgO', 'CaO'}
MINOR_OXIDES = {'TiO2', 'Cr2O3', 'MnO', 'Na2O', 'K2O'}


OXIDE_MASSES = {
    'SiO2': 60.084, 'TiO2': 79.865, 'Al2O3': 101.961, 'Cr2O3': 151.989,
    'FeO': 71.844, 'FeO_total': 71.844, 'MnO': 70.937, 'MgO': 40.304,
    'CaO': 56.077, 'Na2O': 61.979, 'K2O': 94.196, 'P2O5': 141.943,
}
_CATION_PER_OXIDE = {
    'SiO2': 1, 'TiO2': 1, 'Al2O3': 2, 'Cr2O3': 2, 'FeO': 1, 'FeO_total': 1,
    'MnO': 1, 'MgO': 1, 'CaO': 1, 'Na2O': 2, 'K2O': 2, 'P2O5': 2,
}
_OXY_PER_OXIDE = {
    'SiO2': 2, 'TiO2': 2, 'Al2O3': 3, 'Cr2O3': 3, 'FeO': 1, 'FeO_total': 1,
    'MnO': 1, 'MgO': 1, 'CaO': 1, 'Na2O': 1, 'K2O': 1, 'P2O5': 5,
}


def cation_recalc_6oxy(df, oxides=OXIDES_OPX):
    """Cation recalculation on a 6-oxygen basis (for opx/pyx M-site math).

    Mirrors the NB01 cleaning step; adds `{ox}_cat` columns plus `cation_sum`.
    """
    out = df.copy()
    cat_factors = {}
    for ox in oxides:
        if ox not in df.columns:
            continue
        cat_factors[ox] = (df[ox] / OXIDE_MASSES[ox]) * _CATION_PER_OXIDE[ox]
    moles_oxy = pd.DataFrame()
    for ox in oxides:
        if ox not in df.columns:
            continue
        moles_oxy[ox] = (df[ox] / OXIDE_MASSES[ox]) * _OXY_PER_OXIDE[ox]
    moles_oxy = moles_oxy.fillna(0)
    sum_oxy = moles_oxy.sum(axis=1)
    factor = 6.0 / sum_oxy.replace(0, np.nan)
    for ox in oxides:
        if ox not in df.columns:
            continue
        out[f'{ox}_cat'] = cat_factors[ox] * factor
    cat_cols = [f'{ox}_cat' for ox in oxides if ox in df.columns]
    if cat_cols:
        out['cation_sum'] = out[cat_cols].sum(axis=1)
    return out


def add_engineered_features(df):
    """Add Mg_num, En/Fs/Wo, Al_IV/VI, MgTs, liq_Mg_num to a training-schema df.

    Requires unsuffixed opx oxides (SiO2, Al2O3, MgO, CaO, FeO_total) + the
    `{ox}_cat` columns from `cation_recalc_6oxy`. Safe to call when some
    engineered columns already exist (they are overwritten).
    """
    out = df.copy()
    out['Mg_num'] = (out['MgO'] / OXIDE_MASSES['MgO']) / (
        out['MgO'] / OXIDE_MASSES['MgO']
        + out['FeO_total'] / OXIDE_MASSES['FeO_total']
    )
    Mg = out.get('MgO_cat', 0)
    Fe = out.get('FeO_total_cat', 0)
    Ca = out.get('CaO_cat', 0)
    tot = Mg + Fe + Ca
    out['En_frac'] = Mg / tot
    out['Fs_frac'] = Fe / tot
    out['Wo_frac'] = Ca / tot
    Si_cat = out.get('SiO2_cat', 0)
    Al_cat = out.get('Al2O3_cat', 0)
    out['Al_IV'] = np.maximum(2.0 - Si_cat, 0)
    out['Al_VI'] = np.maximum(Al_cat - out['Al_IV'], 0)
    out['MgTs'] = np.minimum(out['Al_IV'], out['Al_VI'])
    if 'liq_MgO' in out.columns and 'liq_FeO' in out.columns:
        out['liq_Mg_num'] = (out['liq_MgO'] / OXIDE_MASSES['MgO']) / (
            out['liq_MgO'] / OXIDE_MASSES['MgO']
            + out['liq_FeO'] / OXIDE_MASSES['FeO']
        )
    return out


def lepr_to_training_features(df):
    """Full pipeline: LEPR suffixed schema -> training-schema + engineered cols.

    Equivalent to `add_engineered_features(cation_recalc_6oxy(lepr_to_training_schema(df)))`.
    Use this before `build_feature_matrix` on raw LEPR DataFrames.
    """
    renamed = lepr_to_training_schema(df)
    with_cations = cation_recalc_6oxy(renamed)
    return add_engineered_features(with_cations)


def lepr_to_training_schema(df):
    """Rename LEPR `_Opx` / `_Liq` columns to the ExPetDB training schema
    expected by `build_feature_matrix`.

    LEPR uses suffixed oxide names (SiO2_Opx, FeOt_Opx, SiO2_Liq, FeOt_Liq).
    The training schema uses unsuffixed opx oxides (SiO2, FeO_total) and
    `liq_`-prefixed liquid oxides (liq_SiO2, liq_FeO). `_Cpx` columns are
    left untouched so cpx external models keep their own schema on the
    same DataFrame (call on .copy()).

    Renames only when the target name does not already exist.
    """
    rename = {}
    for col in df.columns:
        if col == 'FeOt_Opx':
            target = 'FeO_total'
        elif col == 'FeOt_Liq':
            target = 'liq_FeO'
        elif col.endswith('_Opx') and not col.endswith('_Cpx'):
            target = col[: -len('_Opx')]
        elif col.endswith('_Liq'):
            target = 'liq_' + col[: -len('_Liq')]
        else:
            continue
        if target not in df.columns and target not in rename.values():
            rename[col] = target
    return df.rename(columns=rename)


def make_raw_features(df, use_liq=False):
    """Raw oxide weight percentages plus engineered ratios."""
    X_list, names = [], []
    for ox in OXIDES_OPX:
        if ox in df.columns:
            X_list.append(df[ox].fillna(0).values)
            names.append(f'raw_{ox}')
    for eng in ENGINEERED:
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
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def make_alr_features(df, use_liq=False, eps=1e-3):
    """Additive log-ratio with SiO2 denominator. Multiplicative zero replacement."""
    oxides_opx = [ox for ox in OXIDES_OPX if ox != 'SiO2']
    oxides_liq = [ox for ox in OXIDES_LIQ if ox != 'SiO2']
    X_list, names = [], []
    opx_denom = df['SiO2'].replace(0, np.nan).fillna(df['SiO2'].mean())
    for ox in oxides_opx:
        if ox not in df.columns:
            continue
        numer = df[ox].replace(0, np.nan)
        fill = eps * max(numer.mean(skipna=True), 1e-6)
        numer = numer.fillna(fill)
        X_list.append(np.log(numer / opx_denom).values)
        names.append(f'alr_{ox}')
    for eng in ENGINEERED:
        if eng in df.columns:
            X_list.append(df[eng].fillna(0).values)
            names.append(eng)
    if use_liq:
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
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def make_pwlr_features(df, use_liq=False, eps=1e-3):
    """Pairwise log-ratio. Every unique oxide pair becomes one feature."""
    def safe_col(frame, col, fallback_mean):
        s = frame[col].replace(0, np.nan)
        return s.fillna(eps * max(fallback_mean, 1e-6)).values

    opx_present = [ox for ox in OXIDES_OPX if ox in df.columns]
    opx_means = {ox: df[ox].mean(skipna=True) for ox in opx_present}
    X_list, names = [], []
    for i, ox_a in enumerate(opx_present):
        for ox_b in opx_present[i + 1:]:
            a = safe_col(df, ox_a, opx_means[ox_a])
            b = safe_col(df, ox_b, opx_means[ox_b])
            X_list.append(np.log(a / b))
            names.append(f'pwlr_{ox_a}_{ox_b}')
    for eng in ENGINEERED:
        if eng in df.columns:
            X_list.append(df[eng].fillna(0).values)
            names.append(eng)
    if use_liq:
        liq_present = [ox for ox in OXIDES_LIQ if f'liq_{ox}' in df.columns]
        liq_means = {ox: df[f'liq_{ox}'].mean(skipna=True) for ox in liq_present}
        for i, ox_a in enumerate(liq_present):
            for ox_b in liq_present[i + 1:]:
                a = safe_col(df, f'liq_{ox_a}', liq_means[ox_a])
                b = safe_col(df, f'liq_{ox_b}', liq_means[ox_b])
                X_list.append(np.log(a / b))
                names.append(f'pwlr_liq_{ox_a}_{ox_b}')
        if 'liq_Mg_num' in df.columns:
            X_list.append(df['liq_Mg_num'].fillna(0).values)
            names.append('liq_Mg_num')
    X = np.column_stack(X_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


FEATURE_FUNCS = {
    'raw': make_raw_features,
    'raw_aug': make_raw_features,
    'alr': make_alr_features,
    'alr_aug': make_alr_features,
    'pwlr': make_pwlr_features,
    'pwlr_aug': make_pwlr_features,
}


def build_feature_matrix(df, feature_set, use_liq=True):
    """Dispatch to the appropriate feature engineer. `_aug` variants are
    identical to their clean counterparts; augmentation is applied externally
    via `augment_dataframe` before this call."""
    if feature_set not in FEATURE_FUNCS:
        raise ValueError(f'unknown feature_set: {feature_set!r}. '
                         f'Choices: {sorted(FEATURE_FUNCS)}.')
    return FEATURE_FUNCS[feature_set](df, use_liq=use_liq)


def augment_dataframe(df, n_aug=5, seed=42):
    """Augment raw oxide columns BEFORE feature engineering.

    Applies Gaussian noise matching EPMA measurement error: 3% relative
    standard deviation on major oxides, 8% on minor oxides. Metadata,
    targets, and engineered features pass through unchanged. Returns a
    concatenation of 1 clean plus (n_aug - 1) noised copies.

    Noise model follows Agreda-Lopez et al. (2024) Table 2.
    """
    if n_aug <= 1:
        return df.copy()

    major_cols = [c for c in df.columns
                  if c in MAJOR_OXIDES or c.replace('liq_', '') in MAJOR_OXIDES]
    minor_cols = [c for c in df.columns
                  if c in MINOR_OXIDES or c.replace('liq_', '') in MINOR_OXIDES]

    rng = np.random.default_rng(seed)
    aug_list = [df.copy()]
    for _ in range(n_aug - 1):
        df_copy = df.copy()
        for col in major_cols:
            noise = rng.normal(0, 0.03, size=len(df)) * np.abs(df[col].values)
            df_copy[col] = df[col].values + noise
        for col in minor_cols:
            noise = rng.normal(0, 0.08, size=len(df)) * np.abs(df[col].values)
            df_copy[col] = df[col].values + noise
        aug_list.append(df_copy)
    return pd.concat(aug_list, ignore_index=True)
