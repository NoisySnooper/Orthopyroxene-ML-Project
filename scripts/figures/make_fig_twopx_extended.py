#!/usr/bin/env python3
"""Core_11 (expanded): multi-category cross-mineral / same-mineral check.

Three row categories, each with (P 1:1) + (T 1:1) + (disequilibrium map):

  Row A. ML opx-liq  vs  ML cpx-liq                 (our ML, cross-mineral)
  Row B. ML opx-liq  vs  Agreda-Lopez cpx-liq       (external cpx, cross-mineral)
  Row C. ML opx-only vs  Putirka opx-only / opx-liq (external opx, same-mineral)

Panels per row:
  col 1: P (ours-y axis) vs external/other-x axis, 1:1 dashed reference
  col 2: T same treatment
  col 3: (DeltaP, DeltaT) scatter with 2-sigma conformal box shaded

Data source: natural paired-pyroxene LEPR dataset used by nb08 (n=327).
Predictions cached to results/core11_extended_predictions.csv.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from scripts.figures._style import apply_pub_style  # noqa: E402
from scripts.figures._model_palette import OKABE_ITO  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE = PROJECT_ROOT / 'results' / 'core11_extended_predictions.csv'
LEPR_XLSX = PROJECT_ROOT / 'data' / 'raw' / 'external' / (
    'LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx')

ML_OPX_LIQ  = OKABE_ITO['blue']          # row A marker
AGREDA_C    = OKABE_ITO['pink']          # row B marker
PUTIRKA_C   = OKABE_ITO['orange']        # row C marker
JORG_C      = OKABE_ITO['green']         # row D marker (Jorgenson 2022)
WANG_C      = OKABE_ITO['vermillion']    # row E marker (Wang 2021)
DIS_C       = OKABE_ITO['green']         # diseq map marker
REF_LINE_C  = '#333333'


def _num(df, suf):
    for c in df.columns:
        if c.endswith(suf):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    return df


def load_lepr():
    xls = pd.ExcelFile(LEPR_XLSX)
    cpx = pd.read_excel(xls, sheet_name='Cpx').drop_duplicates('Experiment')
    opx = pd.read_excel(xls, sheet_name='Opx').drop_duplicates('Experiment')
    liq = pd.read_excel(xls, sheet_name='Liq').drop_duplicates('Experiment')
    cpx = _num(cpx, '_Cpx'); opx = _num(opx, '_Opx'); liq = _num(liq, '_Liq')
    lepr = cpx.merge(opx, on='Experiment', how='inner', suffixes=('_cpx', '_opx'))
    lepr = lepr.merge(liq, on='Experiment', how='inner')
    for tgt in ['T_K', 'P_kbar']:
        for src in [tgt, f'{tgt}_cpx', f'{tgt}_opx']:
            if src in lepr.columns and tgt not in lepr.columns:
                lepr[tgt] = pd.to_numeric(lepr[src], errors='coerce')
    lepr['T_C'] = lepr['T_K'] - 273.15
    lepr = lepr[np.isfinite(lepr['T_C']) & np.isfinite(lepr['P_kbar'])
                & lepr['T_C'].between(400, 1900)
                & lepr['P_kbar'].between(-2, 100)].reset_index(drop=True)
    return lepr


def _build_opx_with_liq(lepr: pd.DataFrame) -> pd.DataFrame:
    """LEPR merged frame -> training-schema opx + liq columns + engineered."""
    from src.features import lepr_to_training_features
    return lepr_to_training_features(lepr.copy())


def _build_cpx_with_liq(lepr: pd.DataFrame) -> pd.DataFrame:
    """Extract cpx cols, rename unsuffixed, add cpx engineered; merge liq."""
    from src.cpx_features import expetdb_cpx_to_training
    cpx_rename = {
        'SiO2_Cpx':  'SiO2',  'TiO2_Cpx':  'TiO2',  'Al2O3_Cpx': 'Al2O3',
        'Cr2O3_Cpx': 'Cr2O3', 'FeOt_Cpx':  'FeO_total',
        'MnO_Cpx':   'MnO',   'MgO_Cpx':   'MgO',   'CaO_Cpx': 'CaO',
        'Na2O_Cpx':  'Na2O',
    }
    liq_rename = {
        'SiO2_Liq':  'liq_SiO2',  'TiO2_Liq':  'liq_TiO2',
        'Al2O3_Liq': 'liq_Al2O3', 'FeOt_Liq':  'liq_FeO',
        'MgO_Liq':   'liq_MgO',   'CaO_Liq':   'liq_CaO',
        'Na2O_Liq':  'liq_Na2O',  'K2O_Liq':   'liq_K2O',
    }
    cpx_only = lepr[[c for c in cpx_rename if c in lepr.columns]].rename(
        columns=cpx_rename)
    liq_only = lepr[[c for c in liq_rename if c in lepr.columns]].rename(
        columns=liq_rename)
    cpx_feat = expetdb_cpx_to_training(cpx_only)
    # add liq_Mg_num
    if 'liq_MgO' in liq_only.columns and 'liq_FeO' in liq_only.columns:
        from src.features import OXIDE_MASSES
        mg = liq_only['liq_MgO'] / OXIDE_MASSES['MgO']
        fe = liq_only['liq_FeO'] / OXIDE_MASSES['FeO']
        liq_only['liq_Mg_num'] = mg / (mg + fe).replace(0, np.nan)
    return pd.concat([cpx_feat.reset_index(drop=True),
                      liq_only.reset_index(drop=True)], axis=1)


def compute_predictions(lepr: pd.DataFrame) -> pd.DataFrame:
    from src.features import build_feature_matrix
    from src.cpx_features import build_cpx_feature_matrix

    opx_with_liq = _build_opx_with_liq(lepr)
    cpx_with_liq = _build_cpx_with_liq(lepr)

    # ML opx-only (use opx-only slice of opx_with_liq; use_liq=False ignores liq_*)
    m_opx_only_T = joblib.load(PROJECT_ROOT /
        'models/canonical/opx/base_RF_T_C_opx_only_pwlr.joblib')
    m_opx_only_P = joblib.load(PROJECT_ROOT /
        'models/canonical/opx/base_RF_P_kbar_opx_only_pwlr.joblib')
    X, _ = build_feature_matrix(opx_with_liq, 'pwlr', use_liq=False)
    T_ml_opx_only = m_opx_only_T.predict(X)
    P_ml_opx_only = m_opx_only_P.predict(X)

    # ML opx-liq: RF pwlr for T, RF alr for P
    m_opx_liq_T = joblib.load(PROJECT_ROOT /
        'models/canonical/opx/base_RF_T_C_opx_liq_pwlr.joblib')
    m_opx_liq_P = joblib.load(PROJECT_ROOT /
        'models/canonical/opx/base_RF_P_kbar_opx_liq_alr.joblib')
    X_pwlr, _ = build_feature_matrix(opx_with_liq, 'pwlr', use_liq=True)
    T_ml_opx_liq = m_opx_liq_T.predict(X_pwlr)
    X_alr, _ = build_feature_matrix(opx_with_liq, 'alr', use_liq=True)
    P_ml_opx_liq = m_opx_liq_P.predict(X_alr)

    # ML cpx-liq: ERT pwlr for T, LightGBM pwlr for P
    m_cpx_liq_T = joblib.load(PROJECT_ROOT /
        'models/canonical/cpx/base_ERT_T_C_cpx_liq_pwlr.joblib')
    m_cpx_liq_P = joblib.load(PROJECT_ROOT /
        'models/canonical/cpx/base_LightGBM_P_kbar_cpx_liq_pwlr.joblib')
    Xc, _ = build_cpx_feature_matrix(cpx_with_liq, 'pwlr', use_liq=True)
    T_ml_cpx_liq = m_cpx_liq_T.predict(Xc)
    P_ml_cpx_liq = m_cpx_liq_P.predict(Xc)

    # Agreda-Lopez cpx_liq T/P (expects suffixed columns)
    from src.external_models import (
        predict_agreda_from_df, predict_jorgenson, predict_wang,
    )
    agreda_cols = [c for c in lepr.columns
                   if c.endswith('_Cpx') or c.endswith('_Liq')]
    agreda_df = lepr[agreda_cols].copy()
    ag_T = predict_agreda_from_df(agreda_df, PROJECT_ROOT / 'models' / 'external',
                                  phase='cpx_liq', target='T', n_perturb=15)
    ag_P = predict_agreda_from_df(agreda_df, PROJECT_ROOT / 'models' / 'external',
                                  phase='cpx_liq', target='P', n_perturb=15)
    T_agreda_cpx = np.asarray(ag_T['median'], float)
    P_agreda_cpx = np.asarray(ag_P['median'], float)
    if float(np.nanmean(T_agreda_cpx)) > 500:
        T_agreda_cpx = T_agreda_cpx - 273.15

    # Jorgenson 2022 + Wang 2021 need a wider liq oxide set than Agreda.
    # Pad missing liq cols with safe defaults so Thermobar does not raise.
    ext_df = agreda_df.copy()
    for col, default in [('Sample_ID_Liq', ''),
                         ('Fe3Fet_Liq', 0.0), ('H2O_Liq', 0.0),
                         ('P2O5_Liq', 0.0), ('F_Liq', 0.0),
                         ('Cl_Liq', 0.0), ('NiO_Liq', 0.0),
                         ('CoO_Liq', 0.0), ('CO2_Liq', 0.0)]:
        if col not in ext_df.columns:
            ext_df[col] = default
    try:
        T_jorg_cpx = np.asarray(predict_jorgenson(
            ext_df, 'T', phase='cpx_liq',
            P_kbar=lepr['P_kbar'].values), float)
    except Exception as e:
        print(f'Jorgenson T failed: {e}')
        T_jorg_cpx = np.full(len(lepr), np.nan)
    try:
        P_jorg_cpx = np.asarray(predict_jorgenson(
            ext_df, 'P', phase='cpx_liq',
            T_K=lepr['T_C'].values + 273.15), float)
    except Exception as e:
        print(f'Jorgenson P failed: {e}')
        P_jorg_cpx = np.full(len(lepr), np.nan)

    # Wang 2021 cpx_liq T/P (uses ground-truth P and T as seeds)
    try:
        T_wang_cpx = np.asarray(predict_wang(
            ext_df, 'T', P_kbar=lepr['P_kbar'].values), float)
    except Exception as e:
        print(f'Wang T failed: {e}')
        T_wang_cpx = np.full(len(lepr), np.nan)
    try:
        P_wang_cpx = np.asarray(predict_wang(
            ext_df, 'P', T_K=lepr['T_C'].values + 273.15), float)
    except Exception as e:
        print(f'Wang P failed: {e}')
        P_wang_cpx = np.full(len(lepr), np.nan)

    # Putirka opx: eq 29c opx-only P, eq 28a opx-liq T
    import Thermobar as pt
    from config import THERMOBAR_T_RETURNS_KELVIN
    opx_comps = pd.DataFrame({c: lepr[c] for c in lepr.columns
                              if c.endswith('_Opx')})
    liq_comps = pd.DataFrame({c: lepr[c] for c in lepr.columns
                              if c.endswith('_Liq')})
    # Thermobar expects Fe3/FeT and H2O columns; default to anhydrous + FeT.
    for col, default in [('Fe3Fet_Liq', 0.0), ('H2O_Liq', 0.0),
                         ('P2O5_Liq', 0.0), ('F_Liq', 0.0),
                         ('Cl_Liq', 0.0), ('NiO_Liq', 0.0)]:
        if col not in liq_comps.columns:
            liq_comps[col] = default
    try:
        P_put_opx_only = pt.calculate_opx_only_press(
            opx_comps=opx_comps, equationP='P_Put2008_eq29c',
            T=lepr['T_C'].values + 273.15,
        )
        if hasattr(P_put_opx_only, 'values'):
            P_put_opx_only = np.asarray(P_put_opx_only.values, float)
    except Exception as e:
        print(f'Putirka eq29c failed: {e}')
        P_put_opx_only = np.full(len(lepr), np.nan)
    try:
        T_put_opx_liq = pt.calculate_opx_liq_temp(
            equationT='T_Put2008_eq28a',
            opx_comps=opx_comps, liq_comps=liq_comps,
            P=lepr['P_kbar'].values,
        )
        if hasattr(T_put_opx_liq, 'values'):
            T_put_opx_liq = np.asarray(T_put_opx_liq.values, float)
        if THERMOBAR_T_RETURNS_KELVIN:
            T_put_opx_liq = T_put_opx_liq - 273.15
    except Exception as e:
        print(f'Putirka eq28a failed: {e}')
        T_put_opx_liq = np.full(len(lepr), np.nan)

    out = pd.DataFrame({
        'Experiment': lepr['Experiment'].astype(str).values,
        'T_C_true':   lepr['T_C'].values,
        'P_kbar_true': lepr['P_kbar'].values,
        'T_ml_opx_only':  T_ml_opx_only,
        'P_ml_opx_only':  P_ml_opx_only,
        'T_ml_opx_liq':   T_ml_opx_liq,
        'P_ml_opx_liq':   P_ml_opx_liq,
        'T_ml_cpx_liq':   T_ml_cpx_liq,
        'P_ml_cpx_liq':   P_ml_cpx_liq,
        'T_agreda_cpx_liq': T_agreda_cpx,
        'P_agreda_cpx_liq': P_agreda_cpx,
        'T_jorgenson_cpx_liq': T_jorg_cpx,
        'P_jorgenson_cpx_liq': P_jorg_cpx,
        'T_wang_cpx_liq': T_wang_cpx,
        'P_wang_cpx_liq': P_wang_cpx,
        'T_putirka_opx_liq_eq28a': T_put_opx_liq,
        'P_putirka_opx_only_eq29c': P_put_opx_only,
    })
    return out


def scatter_1to1(ax, x, y, color, label, unit, title, xlim=None):
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    ax.scatter(x[m], y[m], s=18, alpha=0.55, color=color,
               edgecolor='black', linewidths=0.3,
               label=f'LEPR (n={n})')
    if n == 0:
        ax.text(0.5, 0.5, 'no valid predictions', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='0.35')
        ax.set_title(title, loc='left', pad=4, fontsize=10)
        ax.grid(True); ax.set_axisbelow(True)
        return
    if xlim is None:
        lo = float(np.nanmin(np.concatenate([x[m], y[m]])))
        hi = float(np.nanmax(np.concatenate([x[m], y[m]])))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        xlim = (lo - pad, hi + pad)
    ax.plot(xlim, xlim, ls='--', lw=1, color=REF_LINE_C, alpha=0.8,
            zorder=1)
    ax.axhline(0, color='#888888', lw=0.5)
    ax.axvline(0, color='#888888', lw=0.5)
    ax.set_xlim(xlim); ax.set_ylim(xlim)
    ax.set_title(title, loc='left', pad=4, fontsize=10)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax.grid(True); ax.set_axisbelow(True)


def diseq_panel(ax, dP, dT, title, sigma_P, sigma_T, color=DIS_C):
    m = np.isfinite(dP) & np.isfinite(dT)
    n = int(m.sum())
    ax.scatter(dP[m], dT[m], s=18, alpha=0.55, color=color,
               edgecolor='black', linewidths=0.3,
               label=f'LEPR (n={n})')
    ax.axvline(0, color='k', lw=0.7); ax.axhline(0, color='k', lw=0.7)
    ax.axvspan(-2 * sigma_P, 2 * sigma_P, alpha=0.08, color=OKABE_ITO['blue'])
    ax.axhspan(-2 * sigma_T, 2 * sigma_T, alpha=0.08, color=OKABE_ITO['blue'])
    in_box = (np.abs(dP[m]) < 2 * sigma_P) & (np.abs(dT[m]) < 2 * sigma_T)
    frac = float(in_box.mean()) if n > 0 else float('nan')
    ax.set_title(f'{title}\nequilib. box {frac*100:.1f}%',
                 loc='left', pad=4, fontsize=10)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax.grid(True); ax.set_axisbelow(True)


def main():
    # Load or compute cache.
    if CACHE.exists():
        pred = pd.read_csv(CACHE)
        print(f'using cached {CACHE.name} n={len(pred)}')
    else:
        print('computing extended predictions (5-10 min)...')
        lepr = load_lepr()
        pred = compute_predictions(lepr)
        pred.to_csv(CACHE, index=False)
        print(f'wrote {CACHE.name} n={len(pred)}')

    # Pull conformal sigmas from nb07 if present.
    sigma_T = 50.0; sigma_P = 3.0
    qhat_path = PROJECT_ROOT / 'results' / 'nb07_conformal_qhat.json'
    if qhat_path.exists():
        with open(qhat_path) as f:
            qh = json.load(f)
        sigma_T = float(qh.get('T_C', {}).get('qhat', sigma_T))
        sigma_P = float(qh.get('P_kbar', {}).get('qhat', sigma_P))

    fig, axes = plt.subplots(5, 3, figsize=(15, 22))

    # Row A. ML opx-liq (RF pwlr T, RF alr P) vs ML cpx-liq (ERT pwlr T, LightGBM pwlr P)
    axA_P, axA_T, axA_D = axes[0]
    scatter_1to1(axA_P, pred['P_ml_cpx_liq'].values, pred['P_ml_opx_liq'].values,
                 ML_OPX_LIQ, 'ours', 'kbar',
                 '(A1) Cross-mineral P: ML opx-liq (RF alr)\n'
                 'vs ML cpx-liq (LightGBM pwlr)')
    axA_P.set_xlabel('P ML cpx-liq (kbar)'); axA_P.set_ylabel('P ML opx-liq (kbar)')
    scatter_1to1(axA_T, pred['T_ml_cpx_liq'].values, pred['T_ml_opx_liq'].values,
                 ML_OPX_LIQ, 'ours', 'C',
                 '(A2) Cross-mineral T: ML opx-liq (RF pwlr)\n'
                 'vs ML cpx-liq (ERT pwlr)')
    axA_T.set_xlabel('T ML cpx-liq (C)'); axA_T.set_ylabel('T ML opx-liq (C)')
    dP_A = pred['P_ml_opx_liq'].values - pred['P_ml_cpx_liq'].values
    dT_A = pred['T_ml_opx_liq'].values - pred['T_ml_cpx_liq'].values
    diseq_panel(axA_D, dP_A, dT_A,
                '(A3) Disequilibrium: ML opx-liq - ML cpx-liq',
                sigma_P, sigma_T, color=ML_OPX_LIQ)
    axA_D.set_xlabel(f'DeltaP (kbar); box = +/- {2*sigma_P:.1f}')
    axA_D.set_ylabel(f'DeltaT (C); box = +/- {2*sigma_T:.0f}')

    # Row B. ML opx-liq (RF pwlr T, RF alr P) vs Agreda-Lopez cpx-liq
    axB_P, axB_T, axB_D = axes[1]
    scatter_1to1(axB_P, pred['P_agreda_cpx_liq'].values, pred['P_ml_opx_liq'].values,
                 AGREDA_C, 'ours', 'kbar',
                 '(B1) P: ML opx-liq (RF alr)\n'
                 'vs Agreda-Lopez cpx-liq')
    axB_P.set_xlabel('P Agreda-Lopez cpx-liq (kbar)')
    axB_P.set_ylabel('P ML opx-liq (kbar)')
    scatter_1to1(axB_T, pred['T_agreda_cpx_liq'].values, pred['T_ml_opx_liq'].values,
                 AGREDA_C, 'ours', 'C',
                 '(B2) T: ML opx-liq (RF pwlr)\n'
                 'vs Agreda-Lopez cpx-liq')
    axB_T.set_xlabel('T Agreda-Lopez cpx-liq (C)')
    axB_T.set_ylabel('T ML opx-liq (C)')
    dP_B = pred['P_ml_opx_liq'].values - pred['P_agreda_cpx_liq'].values
    dT_B = pred['T_ml_opx_liq'].values - pred['T_agreda_cpx_liq'].values
    diseq_panel(axB_D, dP_B, dT_B,
                '(B3) Disequilibrium: ML opx-liq - Agreda-Lopez cpx-liq',
                sigma_P, sigma_T, color=AGREDA_C)
    axB_D.set_xlabel(f'DeltaP (kbar); box = +/- {2*sigma_P:.1f}')
    axB_D.set_ylabel(f'DeltaT (C); box = +/- {2*sigma_T:.0f}')

    # Row C. ML opx vs Putirka opx (P: ML opx-only RF pwlr vs eq29c; T: ML opx-liq RF pwlr vs eq28a)
    axC_P, axC_T, axC_D = axes[2]
    scatter_1to1(axC_P, pred['P_putirka_opx_only_eq29c'].values,
                 pred['P_ml_opx_only'].values, PUTIRKA_C, 'ours', 'kbar',
                 '(C1) P: ML opx-only (RF pwlr)\n'
                 'vs Putirka eq29c')
    axC_P.set_xlabel('P Putirka eq29c (kbar)'); axC_P.set_ylabel('P ML opx-only (kbar)')
    scatter_1to1(axC_T, pred['T_putirka_opx_liq_eq28a'].values,
                 pred['T_ml_opx_liq'].values, PUTIRKA_C, 'ours', 'C',
                 '(C2) T: ML opx-liq (RF pwlr)\n'
                 'vs Putirka eq28a')
    axC_T.set_xlabel('T Putirka eq28a (C)'); axC_T.set_ylabel('T ML opx-liq (C)')
    dP_C = pred['P_ml_opx_only'].values - pred['P_putirka_opx_only_eq29c'].values
    dT_C = pred['T_ml_opx_liq'].values  - pred['T_putirka_opx_liq_eq28a'].values
    diseq_panel(axC_D, dP_C, dT_C,
                '(C3) Disequilibrium: ML opx - Putirka opx',
                sigma_P, sigma_T, color=PUTIRKA_C)
    axC_D.set_xlabel(f'DeltaP (kbar); box = +/- {2*sigma_P:.1f}')
    axC_D.set_ylabel(f'DeltaT (C); box = +/- {2*sigma_T:.0f}')

    # Row D. ML opx-liq vs Jorgenson (2022) cpx-liq
    axD_P, axD_T, axD_D = axes[3]
    scatter_1to1(axD_P, pred['P_jorgenson_cpx_liq'].values,
                 pred['P_ml_opx_liq'].values, JORG_C, 'ours', 'kbar',
                 '(D1) P: ML opx-liq (RF alr)\n'
                 'vs Jorgenson 2022 cpx-liq')
    axD_P.set_xlabel('P Jorgenson 2022 cpx-liq (kbar)')
    axD_P.set_ylabel('P ML opx-liq (kbar)')
    scatter_1to1(axD_T, pred['T_jorgenson_cpx_liq'].values,
                 pred['T_ml_opx_liq'].values, JORG_C, 'ours', 'C',
                 '(D2) T: ML opx-liq (RF pwlr)\n'
                 'vs Jorgenson 2022 cpx-liq')
    axD_T.set_xlabel('T Jorgenson 2022 cpx-liq (C)')
    axD_T.set_ylabel('T ML opx-liq (C)')
    dP_D = pred['P_ml_opx_liq'].values - pred['P_jorgenson_cpx_liq'].values
    dT_D = pred['T_ml_opx_liq'].values - pred['T_jorgenson_cpx_liq'].values
    diseq_panel(axD_D, dP_D, dT_D,
                '(D3) Disequilibrium: ML opx-liq - Jorgenson 2022 cpx-liq',
                sigma_P, sigma_T, color=JORG_C)
    axD_D.set_xlabel(f'DeltaP (kbar); box = +/- {2*sigma_P:.1f}')
    axD_D.set_ylabel(f'DeltaT (C); box = +/- {2*sigma_T:.0f}')

    # Row E. ML opx-liq vs Wang (2021) cpx-liq
    axE_P, axE_T, axE_D = axes[4]
    scatter_1to1(axE_P, pred['P_wang_cpx_liq'].values,
                 pred['P_ml_opx_liq'].values, WANG_C, 'ours', 'kbar',
                 '(E1) P: ML opx-liq (RF alr)\n'
                 'vs Wang 2021 cpx-liq')
    axE_P.set_xlabel('P Wang 2021 cpx-liq (kbar)')
    axE_P.set_ylabel('P ML opx-liq (kbar)')
    scatter_1to1(axE_T, pred['T_wang_cpx_liq'].values,
                 pred['T_ml_opx_liq'].values, WANG_C, 'ours', 'C',
                 '(E2) T: ML opx-liq (RF pwlr)\n'
                 'vs Wang 2021 cpx-liq')
    axE_T.set_xlabel('T Wang 2021 cpx-liq (C)')
    axE_T.set_ylabel('T ML opx-liq (C)')
    dP_E = pred['P_ml_opx_liq'].values - pred['P_wang_cpx_liq'].values
    dT_E = pred['T_ml_opx_liq'].values - pred['T_wang_cpx_liq'].values
    diseq_panel(axE_D, dP_E, dT_E,
                '(E3) Disequilibrium: ML opx-liq - Wang 2021 cpx-liq',
                sigma_P, sigma_T, color=WANG_C)
    axE_D.set_xlabel(f'DeltaP (kbar); box = +/- {2*sigma_P:.1f}')
    axE_D.set_ylabel(f'DeltaT (C); box = +/- {2*sigma_T:.0f}')

    fig.suptitle(
        'Natural paired-pyroxene (opx + cpx) cross-check on LEPR  '
        '(ML bars = pre-correction)\n'
        'rows: A=ML vs ML, B=ML vs Agreda-Lopez 2024, C=ML vs '
        'Putirka 2008, D=ML vs Jorgenson 2022, E=ML vs Wang 2021\n'
        'cols: P 1:1, T 1:1, disequilibrium map',
        y=0.985, fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.subplots_adjust(hspace=0.50, wspace=0.30, top=0.935)

    stem = OUT_DIR / 'Core_11_fig_nb08_twopx_1to1'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 11. Natural paired-pyroxene sanity check on LEPR samples '
        'that contain both opx and cpx (inner-join on Experiment). Five '
        'row categories, each presented as (P 1:1, T 1:1, disequilibrium '
        'map). Row A compares our ML opx-liq (RF pwlr T, RF alr P) with '
        'our ML cpx-liq (ERT pwlr T, LightGBM pwlr P) -- cross-mineral '
        'agreement between two independently trained ML models. Row B '
        'compares our ML opx-liq with the external Agreda-Lopez (2024) '
        'cpx-liq model. Row C compares our ML opx predictions with the '
        'classical Putirka (2008) opx thermobarometers (opx-only P via '
        'eq 29c, opx-liq T via eq 28a; same-mineral benchmark). Row D '
        'compares our ML opx-liq with the Jorgenson (2022) ET-based '
        'cpx-liq model. Row E compares our ML opx-liq with the Wang '
        '(2021) XGB-based cpx-liq model. Disequilibrium maps plot '
        'DeltaP vs DeltaT; the shaded box spans +/- 2 sigma_conformal '
        '(from nb07 calibration, used as a petrological equilibrium '
        'flag). Axes are auto-scaled so no samples are clipped. Source '
        'data: LEPR_Wet_Stitched (April 2023 dump); intermediate '
        'predictions cached in results/core11_extended_predictions.csv.'
    )
    (OUT_DIR / 'Core_11_fig_nb08_twopx_1to1.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
