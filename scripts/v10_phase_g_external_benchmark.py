#!/usr/bin/env python3
"""v10 Phase G.1b: external model benchmark on v10 fixed test splits.

For each v10 pipeline track, run the classical / pre-trained external
thermobarometers on the v10 Citation-grouped test split, compute RMSE /
MAE / R2 vs the true T_C and P_kbar, and emit a side-by-side comparison
against the v10 best base / ensemble.

Methods covered (Thermobar where possible):
  opx_liq:  Putirka 2008 eq 28a (T, true P) / eq 29a (P, true T)
  opx_only: Putirka 2008 eq 29c (P, true T)
  cpx_liq:  Putirka 2008 eq 33 (T) / eq 30 (P)
            Jorgenson 2022 cpx-liq
            Petrelli 2020 cpx-liq
            Wang 2021 cpx-liq
            Agreda-Lopez 2024 cpx-liq (if ONNX available)
  cpx_only: Putirka 2008 eq 32d (T) / eq 32a (P)
            Jorgenson 2022 cpx-only
            Agreda-Lopez 2024 cpx-only (if ONNX available)
  twopx:    Putirka 2008 eq 28b_opx_sat + cpx-opx eq39 where applicable

Output: results/v10_external_benchmark.csv + log.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import RESULTS, LOGS, MODELS
from src.data import (load_opx_liq, load_opx_only, load_cpx_liq,
                      load_cpx_only, load_twopx, load_splits)

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_external_benchmark.log'
OUT_CSV = RESULTS / 'v10_external_benchmark.csv'

FE2O3_TO_FEO = 0.8998  # molar mass ratio FeO / Fe2O3 * 2


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def score(y_true, y_pred, method, target, track, n=None, notes=''):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(y_t) == 0:
        return {'method': method, 'target': target, 'track': track,
                'n': 0, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
                'notes': notes + ' | empty'}
    return {
        'method': method, 'target': target, 'track': track,
        'n': int(len(y_t)),
        'rmse': float(np.sqrt(mean_squared_error(y_t, y_p))),
        'mae':  float(mean_absolute_error(y_t, y_p)),
        'r2':   float(r2_score(y_t, y_p)),
        'notes': notes,
    }


def score_filtered(y_true, y_pred, method, target, track, notes=''):
    """Drop predictions outside a physically plausible range for target."""
    if target == 'T_C':
        lo, hi = 500, 2000
    else:
        lo, hi = -5, 80
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask = (np.isfinite(y_true) & np.isfinite(y_pred) &
            (y_pred >= lo) & (y_pred <= hi))
    n_kept = int(mask.sum())
    n_total = int(np.isfinite(y_true).sum())
    dropped = n_total - n_kept
    note = f'{notes} | filtered plausible {target}, dropped={dropped}/{n_total}'
    return score(y_true[mask], y_pred[mask],
                 f'{method} [filtered]', target, track, notes=note.strip(' |'))


# ---------------------------------------------------------------------------
# Column renamers per pipeline - produce Thermobar-style frames
# ---------------------------------------------------------------------------

OXIDES = ['SiO2','TiO2','Al2O3','Cr2O3','MnO','MgO','CaO','Na2O','K2O']


def _opx_phase_frame(df):
    """opx oxide columns live in 'SiO2 value', 'FeO value', 'Fe2O3 value'."""
    out = pd.DataFrame(index=df.index)
    for o in OXIDES:
        col = f'{o} value'
        out[f'{o}_Opx'] = df[col] if col in df.columns else 0.0
    feo = df.get('FeO value', pd.Series(0.0, index=df.index)).fillna(0.0)
    fe2o3 = df.get('Fe2O3 value', pd.Series(0.0, index=df.index)).fillna(0.0)
    out['FeOt_Opx'] = feo + fe2o3 * FE2O3_TO_FEO
    out['P2O5_Opx'] = 0.0
    return out


def _liq_phase_frame_from_opx_liq(df):
    """Liquid columns live in 'liq_SiO2', etc."""
    out = pd.DataFrame(index=df.index)
    mapping = {'SiO2':'liq_SiO2','TiO2':'liq_TiO2','Al2O3':'liq_Al2O3',
               'FeO':'liq_FeO','MgO':'liq_MgO','CaO':'liq_CaO',
               'Na2O':'liq_Na2O','K2O':'liq_K2O'}
    for key, src in mapping.items():
        out[f'{key}_Liq'] = df[src] if src in df.columns else 0.0
    # Defaults for oxides missing in liq
    out['Cr2O3_Liq'] = 0.0
    out['MnO_Liq'] = 0.0
    out['P2O5_Liq'] = 0.0
    out['NiO_Liq'] = 0.0
    out['CoO_Liq'] = 0.0
    out['CO2_Liq'] = 0.0
    out['FeOt_Liq'] = out['FeO_Liq']
    h2o = df.get('H2O_Liq', pd.Series(0.0, index=df.index)).fillna(0.0)
    out['H2O_Liq'] = h2o
    out['Fe3Fet_Liq'] = 0.15
    out['Sample_ID_Liq'] = df.index.astype(str)
    return out


def _cpx_phase_frame(df):
    """cpx_only / cpx_liq: cpx oxides unprefixed lowercase."""
    out = pd.DataFrame(index=df.index)
    for o in OXIDES:
        if o in df.columns:
            out[f'{o}_Cpx'] = df[o]
        else:
            out[f'{o}_Cpx'] = 0.0
    # FeO_total -> FeOt
    feot = df.get('FeO_total', pd.Series(0.0, index=df.index)).fillna(0.0)
    out['FeOt_Cpx'] = feot
    out['P2O5_Cpx'] = 0.0
    return out


def _liq_phase_frame_from_cpx_liq(df):
    out = pd.DataFrame(index=df.index)
    mapping = {'SiO2':'liq_SiO2','TiO2':'liq_TiO2','Al2O3':'liq_Al2O3',
               'FeO':'liq_FeO','MgO':'liq_MgO','CaO':'liq_CaO',
               'Na2O':'liq_Na2O','K2O':'liq_K2O'}
    for key, src in mapping.items():
        out[f'{key}_Liq'] = df[src] if src in df.columns else 0.0
    out['Cr2O3_Liq'] = 0.0
    out['MnO_Liq'] = 0.0
    out['P2O5_Liq'] = 0.0
    out['NiO_Liq'] = 0.0
    out['CoO_Liq'] = 0.0
    out['CO2_Liq'] = 0.0
    out['FeOt_Liq'] = df.get('liq_FeO_total', out['FeO_Liq']).fillna(out['FeO_Liq'])
    out['H2O_Liq'] = 0.0
    out['Fe3Fet_Liq'] = 0.15
    out['Sample_ID_Liq'] = df.index.astype(str)
    return out


# ---------------------------------------------------------------------------
# Thermobar caller helpers
# ---------------------------------------------------------------------------

def _tb_extract(out, want_celsius):
    """Pull 1-D float array from a Thermobar return (DataFrame or array)."""
    if isinstance(out, pd.DataFrame):
        # take the first numeric column
        for c in out.columns:
            if out[c].dtype.kind in 'fiu':
                arr = out[c].to_numpy(dtype=float)
                break
        else:
            arr = out.iloc[:, 0].to_numpy(dtype=float)
    else:
        arr = np.asarray(out, dtype=float).ravel()
    if want_celsius:
        # Heuristic: if >400, assume Kelvin
        if np.nanmean(arr) > 400:
            arr = arr - 273.15
    return arr


def putirka_opx_liq(opx_df, liq_df, target, true_T_C=None, true_P_kbar=None):
    import Thermobar as pt
    if target == 'T':
        P = true_P_kbar
        out = pt.calculate_opx_liq_temp(
            equationT='T_Put2008_eq28a', opx_comps=opx_df, liq_comps=liq_df,
            P=P)
        return _tb_extract(out, want_celsius=True)
    T_K = true_T_C + 273.15
    out = pt.calculate_opx_liq_press(
        equationP='P_Put2008_eq29a', opx_comps=opx_df, liq_comps=liq_df,
        T=T_K)
    return _tb_extract(out, want_celsius=False)


def putirka_opx_only_P(opx_df, true_T_C):
    import Thermobar as pt
    T_K = true_T_C + 273.15
    out = pt.calculate_opx_only_press(
        equationP='P_Put2008_eq29c', opx_comps=opx_df, T=T_K)
    return _tb_extract(out, want_celsius=False)


def putirka_cpx_liq(cpx_df, liq_df, target, true_T_C=None, true_P_kbar=None):
    import Thermobar as pt
    if target == 'T':
        out = pt.calculate_cpx_liq_temp(
            equationT='T_Put2008_eq33', cpx_comps=cpx_df, liq_comps=liq_df,
            P=true_P_kbar)
        return _tb_extract(out, want_celsius=True)
    T_K = true_T_C + 273.15
    out = pt.calculate_cpx_liq_press(
        equationP='P_Put2008_eq30', cpx_comps=cpx_df, liq_comps=liq_df,
        T=T_K)
    return _tb_extract(out, want_celsius=False)


def putirka_cpx_only(cpx_df, target, true_T_C=None, true_P_kbar=None):
    import Thermobar as pt
    if target == 'T':
        out = pt.calculate_cpx_only_temp(
            equationT='T_Put2008_eq32d', cpx_comps=cpx_df, P=true_P_kbar)
        return _tb_extract(out, want_celsius=True)
    T_K = true_T_C + 273.15
    out = pt.calculate_cpx_only_press(
        equationP='P_Put2008_eq32a', cpx_comps=cpx_df, T=T_K)
    return _tb_extract(out, want_celsius=False)


def jorgenson_cpx_liq(cpx_df, liq_df, target, true_T_C=None, true_P_kbar=None):
    import Thermobar as pt
    if target == 'T':
        out = pt.calculate_cpx_liq_temp(
            equationT='T_Jorgenson2022_Cpx_Liq', cpx_comps=cpx_df,
            liq_comps=liq_df, P=true_P_kbar)
        return _tb_extract(out, want_celsius=True)
    T_K = true_T_C + 273.15
    out = pt.calculate_cpx_liq_press(
        equationP='P_Jorgenson2022_Cpx_Liq', cpx_comps=cpx_df,
        liq_comps=liq_df, T=T_K)
    return _tb_extract(out, want_celsius=False)


def jorgenson_cpx_only(cpx_df, target, true_T_C=None, true_P_kbar=None):
    import Thermobar as pt
    if target == 'T':
        out = pt.calculate_cpx_only_temp(
            equationT='T_Jorgenson2022_Cpx_only', cpx_comps=cpx_df,
            P=true_P_kbar)
        return _tb_extract(out, want_celsius=True)
    T_K = true_T_C + 273.15
    out = pt.calculate_cpx_only_press(
        equationP='P_Jorgenson2022_Cpx_only', cpx_comps=cpx_df, T=T_K)
    return _tb_extract(out, want_celsius=False)


def petrelli_cpx_liq(cpx_df, liq_df, target, true_T_C=None, true_P_kbar=None):
    import Thermobar as pt
    if target == 'T':
        out = pt.calculate_cpx_liq_temp(
            equationT='T_Petrelli2020_Cpx_Liq', cpx_comps=cpx_df,
            liq_comps=liq_df, P=true_P_kbar)
        return _tb_extract(out, want_celsius=True)
    T_K = true_T_C + 273.15
    out = pt.calculate_cpx_liq_press(
        equationP='P_Petrelli2020_Cpx_Liq', cpx_comps=cpx_df,
        liq_comps=liq_df, T=T_K)
    return _tb_extract(out, want_celsius=False)


def wang_cpx_liq(cpx_df, liq_df, target, true_T_C=None, true_P_kbar=None):
    import Thermobar as pt
    if target == 'T':
        out = pt.calculate_cpx_liq_temp(
            equationT='T_Wang2021_eq2', cpx_comps=cpx_df, liq_comps=liq_df,
            P=true_P_kbar)
        return _tb_extract(out, want_celsius=True)
    T_K = true_T_C + 273.15
    out = pt.calculate_cpx_liq_press(
        equationP='P_Wang2021_eq1', cpx_comps=cpx_df, liq_comps=liq_df,
        T=T_K)
    return _tb_extract(out, want_celsius=False)


# ---------------------------------------------------------------------------
# Pipeline drivers
# ---------------------------------------------------------------------------

def bench_opx_liq(rows, fh):
    df = load_opx_liq()
    _, te = load_splits('opx_liq')
    df_te = df.iloc[te].reset_index(drop=True)
    opx = _opx_phase_frame(df_te)
    liq = _liq_phase_frame_from_opx_liq(df_te)
    y_T = df_te['T_C'].to_numpy(float)
    y_P = df_te['P_kbar'].to_numpy(float)
    try:
        y_hat = putirka_opx_liq(opx, liq, 'T', true_P_kbar=y_P)
        rows.append(score(y_T, y_hat, 'Putirka 28a (true P)', 'T_C', 'opx_liq'))
        rows.append(score_filtered(y_T, y_hat, 'Putirka 28a (true P)', 'T_C', 'opx_liq'))
    except Exception as e:
        _log(f'opx_liq Put28a T FAIL {e}', fh)
    try:
        y_hat = putirka_opx_liq(opx, liq, 'P', true_T_C=y_T)
        rows.append(score(y_P, y_hat, 'Putirka 29a (true T)', 'P_kbar', 'opx_liq'))
        rows.append(score_filtered(y_P, y_hat, 'Putirka 29a (true T)', 'P_kbar', 'opx_liq'))
    except Exception as e:
        _log(f'opx_liq Put29a P FAIL {e}', fh)


def bench_opx_only(rows, fh):
    df = load_opx_only()
    _, te = load_splits('opx_only')
    df_te = df.iloc[te].reset_index(drop=True)
    opx = _opx_phase_frame(df_te)
    y_T = df_te['T_C'].to_numpy(float)
    y_P = df_te['P_kbar'].to_numpy(float)
    try:
        y_hat = putirka_opx_only_P(opx, y_T)
        rows.append(score(y_P, y_hat, 'Putirka 29c (true T)', 'P_kbar', 'opx_only'))
        rows.append(score_filtered(y_P, y_hat, 'Putirka 29c (true T)', 'P_kbar', 'opx_only'))
    except Exception as e:
        _log(f'opx_only Put29c P FAIL {e}', fh)


def bench_cpx_liq(rows, fh):
    df = load_cpx_liq()
    _, te = load_splits('cpx_liq')
    df_te = df.iloc[te].reset_index(drop=True)
    cpx = _cpx_phase_frame(df_te)
    liq = _liq_phase_frame_from_cpx_liq(df_te)
    y_T = df_te['T_C'].to_numpy(float)
    y_P = df_te['P_kbar'].to_numpy(float)

    for name, fn in [('Putirka 33/30', putirka_cpx_liq),
                     ('Jorgenson 2022', jorgenson_cpx_liq),
                     ('Petrelli 2020', petrelli_cpx_liq),
                     ('Wang 2021', wang_cpx_liq)]:
        try:
            y_hat = fn(cpx, liq, 'T', true_P_kbar=y_P)
            rows.append(score(y_T, y_hat, f'{name} (true P)', 'T_C', 'cpx_liq'))
            rows.append(score_filtered(y_T, y_hat, f'{name} (true P)', 'T_C', 'cpx_liq'))
        except Exception as e:
            _log(f'cpx_liq {name} T FAIL {e}', fh)
        try:
            y_hat = fn(cpx, liq, 'P', true_T_C=y_T)
            rows.append(score(y_P, y_hat, f'{name} (true T)', 'P_kbar', 'cpx_liq'))
            rows.append(score_filtered(y_P, y_hat, f'{name} (true T)', 'P_kbar', 'cpx_liq'))
        except Exception as e:
            _log(f'cpx_liq {name} P FAIL {e}', fh)

    # Agreda 2024 cpx_liq via existing src/external_models.py
    try:
        from src.external_models import predict_agreda_from_df
        df_ag = df_te.rename(columns=_agreda_cpx_liq_rename(df_te)).copy()
        df_ag['MnO_Liq'] = 0.0  # v10 cpx_liq parquet lacks liq_MnO
        out_T = predict_agreda_from_df(df_ag, MODELS / 'external', 'cpx_liq', 'T')
        rows.append(score(y_T, out_T['median'], 'Agreda-Lopez 2024', 'T_C', 'cpx_liq'))
        out_P = predict_agreda_from_df(df_ag, MODELS / 'external', 'cpx_liq', 'P')
        rows.append(score(y_P, out_P['median'], 'Agreda-Lopez 2024', 'P_kbar', 'cpx_liq'))
        # Restrict to Agreda's calibration range (P<20 kbar). Crustal+upper-mantle
        # conditions — a fair apples-to-apples comparison to their paper numbers.
        mask20 = y_P < 20
        rows.append(score(y_T[mask20], out_T['median'][mask20],
                          'Agreda-Lopez 2024 [P<20 kbar]', 'T_C', 'cpx_liq',
                          notes='restricted to Agreda calibration window'))
        rows.append(score(y_P[mask20], out_P['median'][mask20],
                          'Agreda-Lopez 2024 [P<20 kbar]', 'P_kbar', 'cpx_liq',
                          notes='restricted to Agreda calibration window'))
    except Exception as e:
        import traceback; traceback.print_exc()
        _log(f'cpx_liq Agreda FAIL {e}', fh)


def _agreda_cpx_liq_rename(df):
    """Map v10 cpx_liq column names to Agreda expected names.

    Agreda also requires MnO_Liq, which v10 cpx_liq lacks — caller must
    add MnO_Liq=0 after renaming.
    """
    return {
        'SiO2':'SiO2_Cpx', 'TiO2':'TiO2_Cpx', 'Al2O3':'Al2O3_Cpx',
        'FeO_total':'FeOt_Cpx', 'MgO':'MgO_Cpx', 'MnO':'MnO_Cpx',
        'CaO':'CaO_Cpx', 'Na2O':'Na2O_Cpx', 'Cr2O3':'Cr2O3_Cpx',
        'liq_SiO2':'SiO2_Liq', 'liq_TiO2':'TiO2_Liq', 'liq_Al2O3':'Al2O3_Liq',
        'liq_FeO':'FeOt_Liq', 'liq_MgO':'MgO_Liq', 'liq_CaO':'CaO_Liq',
        'liq_Na2O':'Na2O_Liq', 'liq_K2O':'K2O_Liq',
    }


def _agreda_cpx_only_rename():
    return {
        'SiO2':'SiO2_Cpx', 'TiO2':'TiO2_Cpx', 'Al2O3':'Al2O3_Cpx',
        'FeO_total':'FeOt_Cpx', 'MgO':'MgO_Cpx', 'MnO':'MnO_Cpx',
        'CaO':'CaO_Cpx', 'Na2O':'Na2O_Cpx', 'Cr2O3':'Cr2O3_Cpx',
    }


def bench_cpx_only(rows, fh):
    df = load_cpx_only()
    _, te = load_splits('cpx_only')
    df_te = df.iloc[te].reset_index(drop=True)
    cpx = _cpx_phase_frame(df_te)
    y_T = df_te['T_C'].to_numpy(float)
    y_P = df_te['P_kbar'].to_numpy(float)
    for name, fn in [('Putirka 32d/32a', putirka_cpx_only),
                     ('Jorgenson 2022', jorgenson_cpx_only)]:
        try:
            y_hat = fn(cpx, 'T', true_P_kbar=y_P)
            rows.append(score(y_T, y_hat, f'{name} (true P)', 'T_C', 'cpx_only'))
            rows.append(score_filtered(y_T, y_hat, f'{name} (true P)', 'T_C', 'cpx_only'))
        except Exception as e:
            _log(f'cpx_only {name} T FAIL {e}', fh)
        try:
            y_hat = fn(cpx, 'P', true_T_C=y_T)
            rows.append(score(y_P, y_hat, f'{name} (true T)', 'P_kbar', 'cpx_only'))
            rows.append(score_filtered(y_P, y_hat, f'{name} (true T)', 'P_kbar', 'cpx_only'))
        except Exception as e:
            _log(f'cpx_only {name} P FAIL {e}', fh)
    # Agreda cpx_only
    try:
        from src.external_models import predict_agreda_from_df
        df_ren = df_te.rename(columns=_agreda_cpx_only_rename())
        out_T = predict_agreda_from_df(df_ren, MODELS / 'external', 'cpx_only', 'T')
        rows.append(score(y_T, out_T['median'], 'Agreda-Lopez 2024', 'T_C', 'cpx_only'))
        out_P = predict_agreda_from_df(df_ren, MODELS / 'external', 'cpx_only', 'P')
        rows.append(score(y_P, out_P['median'], 'Agreda-Lopez 2024', 'P_kbar', 'cpx_only'))
        mask20 = y_P < 20
        rows.append(score(y_T[mask20], out_T['median'][mask20],
                          'Agreda-Lopez 2024 [P<20 kbar]', 'T_C', 'cpx_only',
                          notes='restricted to Agreda calibration window'))
        rows.append(score(y_P[mask20], out_P['median'][mask20],
                          'Agreda-Lopez 2024 [P<20 kbar]', 'P_kbar', 'cpx_only',
                          notes='restricted to Agreda calibration window'))
    except Exception as e:
        _log(f'cpx_only Agreda FAIL {e}', fh)


# ---------------------------------------------------------------------------
# v10 ML reference rows
# ---------------------------------------------------------------------------

def add_v10_reference(rows, fh):
    for pipeline, per_cell_csv in [('opx', 'results/v10_opx_per_cell_results.csv'),
                                    ('cpx', 'results/v10_cpx_per_cell_results.csv'),
                                    ('twopx', 'results/v10_twopx_per_cell_results.csv')]:
        if not Path(per_cell_csv).exists():
            continue
        df = pd.read_csv(per_cell_csv)
        best = df.loc[df.groupby(['target','track'])['test_rmse'].idxmin()]
        for _, r in best.iterrows():
            rows.append({
                'method': f'v10 best base ({r["model"]} {r["feature_set"]})',
                'target': r['target'], 'track': r['track'],
                'n': None, 'rmse': r['test_rmse'], 'mae': None, 'r2': r['test_r2'],
                'notes': 'v10 base model, fixed Citation split',
            })
        # Ensemble
        ens_csv = per_cell_csv.replace('per_cell', 'ensemble')
        if Path(ens_csv).exists():
            ens = pd.read_csv(ens_csv)
            best_ens = ens.loc[ens.groupby(['target','track'])['test_rmse'].idxmin()]
            for _, r in best_ens.iterrows():
                rows.append({
                    'method': f'v10 best ensemble ({r["method"]} {r["feature_set"]})',
                    'target': r['target'], 'track': r['track'],
                    'n': None, 'rmse': r['test_rmse'], 'mae': None, 'r2': None,
                    'notes': 'v10 ensemble',
                })


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')
    rows = []
    t0 = time.time()
    try:
        _log('START external benchmark', fh)
        for name, fn in [('opx_liq', bench_opx_liq),
                         ('opx_only', bench_opx_only),
                         ('cpx_liq', bench_cpx_liq),
                         ('cpx_only', bench_cpx_only)]:
            _log(f'[{name}] running', fh)
            try:
                fn(rows, fh)
            except Exception as e:
                import traceback
                _log(f'[{name}] TRACK FAIL {e}', fh)
                _log(traceback.format_exc(), fh)
        add_v10_reference(rows, fh)
        df_out = pd.DataFrame(rows)
        df_out = df_out.sort_values(['track','target','rmse'], na_position='last')
        df_out.to_csv(OUT_CSV, index=False)
        _log(f'wrote {OUT_CSV}  rows={len(df_out)}  elapsed={time.time()-t0:.1f}s', fh)
    finally:
        fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
