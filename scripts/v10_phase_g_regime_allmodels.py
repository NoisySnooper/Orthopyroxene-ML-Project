#!/usr/bin/env python3
"""v10 Phase G.1c-extended: comprehensive regime-stratified benchmark
across ALL Optuna cells (8 models x 3 feature sets per target/track)
vs ALL external baselines (Putirka 2008, Jorgenson 2022, Wang 2021,
Agreda-Lopez 2024) with bootstrap 95% CIs on RMSE.

Bins by BOTH:
  P regime  (true P_kbar bins): P<5, 5-10, 10-20, 20-40, P>=40, P<20, ALL
  T regime  (true T_C bins):    T<800, 800-1000, 1000-1200, 1200-1400,
                                T>=1400, ALL

Output: results/v10_regime_allmodels.csv  (long format)
Columns:
    pipeline, track, target, regime_type ('P' or 'T'), regime,
    regime_lo, regime_hi, n,
    method_family, method, method_label,
    rmse, rmse_lo, rmse_hi, mae, mae_lo, mae_hi, n_used

Bootstrap: 500 resamples with replacement on (y_true, y_pred) pairs
within regime; 2.5 / 97.5 percentiles reported as CI.
Externals are filtered to physical ranges (T 500-2000 C; P -5-80 kbar)
before scoring (matches v10_phase_g_regime_benchmark.py contract).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('V10_MAX_JOBS', '2')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import (
    RESULTS, LOGS, MODELS,
    P_REGIME_BIN_EDGES_KBAR, P_REGIME_LABELS,
    P_REGIME_REGISTERED_DATE,
)
from src.data import (load_opx_liq, load_opx_only, load_cpx_liq,
                      load_cpx_only, load_splits)
from src.models import build_model

warnings.filterwarnings('ignore')

OUT_CSV = RESULTS / 'v10_regime_allmodels.csv'
LOG_PATH = LOGS / 'v10_phase_g_regime_allmodels.log'

FE2O3_TO_FEO = 0.8998
N_BOOT = 500
BOOT_SEED = 20260418

# Pre-registered P bins (authoritative, registered 2026-04-17).
# Source: config.P_REGIME_BIN_EDGES_KBAR = [0, 5, 15, 30, 100].
# Labels:  shallow_crustal / deep_crustal_MASH / lithospheric_mantle /
#          deeper_mantle. Any modification requires a dated entry in
#          docs/v10_p_regime_preregistration.md. Upper ceiling is the
#          ExPetDB training ceiling; widen to 1000 here only so any
#          stray out-of-range rows are captured rather than silently
#          dropped (physical guard rails handled elsewhere).
_PREREG_EDGES = P_REGIME_BIN_EDGES_KBAR  # [0, 5, 15, 30, 100]
PREREG_P_REGIMES = [
    (P_REGIME_LABELS[0], _PREREG_EDGES[0], _PREREG_EDGES[1]),   # 0-5
    (P_REGIME_LABELS[1], _PREREG_EDGES[1], _PREREG_EDGES[2]),   # 5-15
    (P_REGIME_LABELS[2], _PREREG_EDGES[2], _PREREG_EDGES[3]),   # 15-30
    (P_REGIME_LABELS[3], _PREREG_EDGES[3], 1000.0),             # >=30
]

# Exploratory finer-grained bins retained for secondary inspection only.
# These are NOT the authoritative bins; the pre-registered four-bin set
# above is the one the manuscript cites.
EXPLORATORY_P_REGIMES = [
    ('P<5',                  0,   5),
    ('5<=P<10',              5,  10),
    ('10<=P<20',            10,  20),
    ('20<=P<40',            20,  40),
    ('P>=40',               40, 1000),
    ('P<20 (Agreda range)',  0,  20),
]

# Composite list used by the benchmark loop. Pre-registered bins first so
# they sort to the top of any downstream grouping; 'ALL' closes the list.
P_REGIMES = (
    PREREG_P_REGIMES
    + EXPLORATORY_P_REGIMES
    + [('ALL', 0, 1000)]
)

# Physically-motivated T bins (C). Coverage spans typical magmatic range.
T_REGIMES = [
    ('T<800',         -1000,  800),
    ('800<=T<1000',     800, 1000),
    ('1000<=T<1200',   1000, 1200),
    ('1200<=T<1400',   1200, 1400),
    ('T>=1400',        1400, 5000),
    ('ALL',           -1000, 5000),
]


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n'); fh.flush()


def bootstrap_metrics(y_true, y_pred, mask, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap RMSE and MAE within `mask`. Returns dict with point
    estimate and 2.5/97.5 percentile CIs for each; `n_used` counts
    finite pairs."""
    y_t = y_true[mask]; y_p = y_pred[mask]
    good = np.isfinite(y_t) & np.isfinite(y_p)
    y_t, y_p = y_t[good], y_p[good]
    n = len(y_t)
    out = {'rmse': np.nan, 'rmse_lo': np.nan, 'rmse_hi': np.nan,
           'mae': np.nan, 'mae_lo': np.nan, 'mae_hi': np.nan, 'n_used': n}
    if n < 1:
        return out
    out['rmse'] = float(np.sqrt(mean_squared_error(y_t, y_p)))
    out['mae']  = float(mean_absolute_error(y_t, y_p))
    if n < 3:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    yt_b = y_t[idx]; yp_b = y_p[idx]
    rmse_b = np.sqrt(np.mean((yt_b - yp_b)**2, axis=1))
    mae_b  = np.mean(np.abs(yt_b - yp_b), axis=1)
    out['rmse_lo'], out['rmse_hi'] = [float(v) for v in np.percentile(rmse_b, [2.5, 97.5])]
    out['mae_lo'],  out['mae_hi']  = [float(v) for v in np.percentile(mae_b,  [2.5, 97.5])]
    return out


# --- Thermobar frame builders ---

def _opx_phase_frame(df):
    OX = ['SiO2','TiO2','Al2O3','Cr2O3','MnO','MgO','CaO','Na2O','K2O']
    out = pd.DataFrame({f'{o}_Opx': df.get(f'{o} value', 0) for o in OX})
    out['FeOt_Opx'] = (df.get('FeO value',  0).fillna(0)
                       + df.get('Fe2O3 value',0).fillna(0) * FE2O3_TO_FEO)
    out['P2O5_Opx'] = 0.0
    return out


def _liq_phase_frame_from_opx_liq(df):
    out = pd.DataFrame({'SiO2_Liq':df['liq_SiO2'],'TiO2_Liq':df['liq_TiO2'],
        'Al2O3_Liq':df['liq_Al2O3'],'FeO_Liq':df['liq_FeO'],'MgO_Liq':df['liq_MgO'],
        'CaO_Liq':df['liq_CaO'],'Na2O_Liq':df['liq_Na2O'],'K2O_Liq':df['liq_K2O']})
    out['FeOt_Liq']=out['FeO_Liq']; out['Cr2O3_Liq']=0; out['MnO_Liq']=0
    out['P2O5_Liq']=0; out['NiO_Liq']=0; out['CoO_Liq']=0; out['CO2_Liq']=0
    out['H2O_Liq']=df.get('H2O_Liq', pd.Series(0,index=df.index)).fillna(0)
    out['Fe3Fet_Liq']=0.15; out['Sample_ID_Liq']=df.index.astype(str)
    return out


def _cpx_phase_frame(df):
    OX = ['SiO2','TiO2','Al2O3','Cr2O3','MnO','MgO','CaO','Na2O','K2O']
    out = pd.DataFrame({f'{o}_Cpx': df.get(o, 0) for o in OX})
    out['FeOt_Cpx'] = df.get('FeO_total', pd.Series(0,index=df.index)).fillna(0)
    out['P2O5_Cpx']=0.0
    return out


def _liq_phase_frame_from_cpx_liq(df):
    out = pd.DataFrame({'SiO2_Liq':df['liq_SiO2'],'TiO2_Liq':df['liq_TiO2'],
        'Al2O3_Liq':df['liq_Al2O3'],'FeO_Liq':df['liq_FeO'],'MgO_Liq':df['liq_MgO'],
        'CaO_Liq':df['liq_CaO'],'Na2O_Liq':df['liq_Na2O'],'K2O_Liq':df['liq_K2O']})
    out['FeOt_Liq']=df.get('liq_FeO_total', out['FeO_Liq']).fillna(out['FeO_Liq'])
    out['Cr2O3_Liq']=0; out['MnO_Liq']=0; out['P2O5_Liq']=0
    out['NiO_Liq']=0; out['CoO_Liq']=0; out['CO2_Liq']=0
    out['H2O_Liq']=0; out['Fe3Fet_Liq']=0.15
    out['Sample_ID_Liq']=df.index.astype(str)
    return out


def _agreda_cpx_liq_rename():
    return {
        'SiO2':'SiO2_Cpx','TiO2':'TiO2_Cpx','Al2O3':'Al2O3_Cpx','FeO_total':'FeOt_Cpx',
        'MgO':'MgO_Cpx','MnO':'MnO_Cpx','CaO':'CaO_Cpx','Na2O':'Na2O_Cpx','Cr2O3':'Cr2O3_Cpx',
        'liq_SiO2':'SiO2_Liq','liq_TiO2':'TiO2_Liq','liq_Al2O3':'Al2O3_Liq',
        'liq_FeO':'FeOt_Liq','liq_MgO':'MgO_Liq','liq_CaO':'CaO_Liq',
        'liq_Na2O':'Na2O_Liq','liq_K2O':'K2O_Liq'}


def _agreda_cpx_only_rename():
    return {
        'SiO2':'SiO2_Cpx','TiO2':'TiO2_Cpx','Al2O3':'Al2O3_Cpx','FeO_total':'FeOt_Cpx',
        'MgO':'MgO_Cpx','MnO':'MnO_Cpx','CaO':'CaO_Cpx','Na2O':'Na2O_Cpx','Cr2O3':'Cr2O3_Cpx'}


def load_all_cells(pipeline):
    """Return every (model, target, track, feature_set, best_params)
    tuple from the pipeline's Optuna best_params JSON."""
    with open(RESULTS / f'v10_optuna_best_params_{pipeline}.json') as f:
        payload = json.load(f)
    return [(r['model'], r['target'], r['track'], r['feature_set'],
             r['best_params']) for r in payload['results']]


def build_Xy(pipeline, track, feat, df_tr, df_te):
    if pipeline == 'opx':
        from src.features import build_feature_matrix
        use_liq = (track == 'opx_liq')
        X_tr, _ = build_feature_matrix(df_tr, feat, use_liq=use_liq)
        X_te, _ = build_feature_matrix(df_te, feat, use_liq=use_liq)
    elif pipeline == 'cpx':
        from src.cpx_features import build_cpx_feature_matrix
        use_liq = (track == 'cpx_liq')
        X_tr, _ = build_cpx_feature_matrix(df_tr, feat, use_liq=use_liq)
        X_te, _ = build_cpx_feature_matrix(df_te, feat, use_liq=use_liq)
    elif pipeline == 'twopx':
        from src.twopx_features import build_twopx_feature_matrix
        X_tr, _ = build_twopx_feature_matrix(df_tr, feat)
        X_te, _ = build_twopx_feature_matrix(df_te, feat)
    elif pipeline == 'universal':
        from src.universal_features import build_universal_matrix
        X_tr, _ = build_universal_matrix(df_tr)
        X_te, _ = build_universal_matrix(df_te)
    else:
        raise ValueError(pipeline)
    return np.asarray(X_tr, float), np.asarray(X_te, float)


def compute_external_preds(track, df_te, y_T, y_P, fh):
    """Return dict {(method, target): yhat} with Putirka eqns (multiple
    per track), Jorgenson (cpx tracks), Wang (cpx_liq), Agreda (cpx)."""
    preds = {}
    try:
        import Thermobar as pt
        from src.external_models import (predict_agreda_from_df,
                                         predict_jorgenson, predict_wang)
    except Exception as e:
        _log(f'  EXTERNAL import FAIL: {e}', fh)
        return preds

    def _safe(label, fn):
        try:
            arr = np.asarray(fn(), float)
            if arr.ndim > 1:
                arr = arr.ravel()
            preds[label] = arr
        except Exception as e:
            _log(f'  ext fail {label}: {e}', fh)

    if track == 'opx_liq':
        opx = _opx_phase_frame(df_te); liq = _liq_phase_frame_from_opx_liq(df_te)
        _safe(('Putirka 28a','T_C'), lambda: pt.calculate_opx_liq_temp(
            equationT='T_Put2008_eq28a', opx_comps=opx, liq_comps=liq, P=y_P) - 273.15)
        _safe(('Putirka 28b','T_C'), lambda: pt.calculate_opx_liq_temp(
            equationT='T_Put2008_eq28b_opx_sat', opx_comps=opx, liq_comps=liq, P=y_P) - 273.15)
        _safe(('Putirka 29a','P_kbar'), lambda: pt.calculate_opx_liq_press(
            equationP='P_Put2008_eq29a', opx_comps=opx, liq_comps=liq, T=y_T+273.15))
        _safe(('Putirka 29b','P_kbar'), lambda: pt.calculate_opx_liq_press(
            equationP='P_Put2008_eq29b', opx_comps=opx, liq_comps=liq, T=y_T+273.15))
    elif track == 'opx_only':
        opx = _opx_phase_frame(df_te)
        _safe(('Putirka 29c','P_kbar'), lambda: pt.calculate_opx_only_press(
            equationP='P_Put2008_eq29c', opx_comps=opx, T=y_T+273.15))
    elif track == 'cpx_liq':
        cpx = _cpx_phase_frame(df_te); liq = _liq_phase_frame_from_cpx_liq(df_te)
        _safe(('Putirka 33','T_C'), lambda: pt.calculate_cpx_liq_temp(
            equationT='T_Put2008_eq33', cpx_comps=cpx, liq_comps=liq, P=y_P) - 273.15)
        _safe(('Putirka 34','T_C'), lambda: pt.calculate_cpx_liq_temp(
            equationT='T_Put2008_eq34_cpx_sat', cpx_comps=cpx, liq_comps=liq, P=y_P) - 273.15)
        _safe(('Putirka 30','P_kbar'), lambda: pt.calculate_cpx_liq_press(
            equationP='P_Put2008_eq30', cpx_comps=cpx, liq_comps=liq, T=y_T+273.15))
        _safe(('Putirka 31','P_kbar'), lambda: pt.calculate_cpx_liq_press(
            equationP='P_Put2008_eq31', cpx_comps=cpx, liq_comps=liq, T=y_T+273.15))
        # Agreda
        df_ag = df_te.rename(columns=_agreda_cpx_liq_rename()).copy()
        df_ag['MnO_Liq'] = 0.0
        _safe(('Agreda 2024','T_C'), lambda: predict_agreda_from_df(
            df_ag, MODELS/'external','cpx_liq','T')['median'])
        _safe(('Agreda 2024','P_kbar'), lambda: predict_agreda_from_df(
            df_ag, MODELS/'external','cpx_liq','P')['median'])
        # Jorgenson + Wang require _Cpx / _Liq suffixed frames
        df_jw = df_te.rename(columns=_agreda_cpx_liq_rename()).copy()
        df_jw['MnO_Liq'] = 0.0
        _safe(('Jorgenson 2022','T_C'), lambda: predict_jorgenson(
            df_jw, 'T', phase='cpx_liq', P_kbar=y_P))
        _safe(('Jorgenson 2022','P_kbar'), lambda: predict_jorgenson(
            df_jw, 'P', phase='cpx_liq', T_K=y_T+273.15))
        _safe(('Wang 2021','T_C'), lambda: predict_wang(
            df_jw, 'T', P_kbar=y_P))
        _safe(('Wang 2021','P_kbar'), lambda: predict_wang(
            df_jw, 'P', T_K=y_T+273.15))
    elif track == 'cpx_only':
        cpx = _cpx_phase_frame(df_te)
        _safe(('Putirka 32d','T_C'), lambda: pt.calculate_cpx_only_temp(
            equationT='T_Put2008_eq32d', cpx_comps=cpx, P=y_P) - 273.15)
        _safe(('Putirka 32a','P_kbar'), lambda: pt.calculate_cpx_only_press(
            equationP='P_Put2008_eq32a', cpx_comps=cpx, T=y_T+273.15))
        _safe(('Putirka 32b','P_kbar'), lambda: pt.calculate_cpx_only_press(
            equationP='P_Put2008_eq32b', cpx_comps=cpx, T=y_T+273.15))
        # Agreda cpx_only
        df_ag = df_te.rename(columns=_agreda_cpx_only_rename()).copy()
        _safe(('Agreda 2024','T_C'), lambda: predict_agreda_from_df(
            df_ag, MODELS/'external','cpx_only','T')['median'])
        _safe(('Agreda 2024','P_kbar'), lambda: predict_agreda_from_df(
            df_ag, MODELS/'external','cpx_only','P')['median'])
        # Jorgenson cpx_only
        df_jw = df_te.rename(columns=_agreda_cpx_only_rename()).copy()
        _safe(('Jorgenson 2022','T_C'), lambda: predict_jorgenson(
            df_jw, 'T', phase='cpx_only', P_kbar=y_P))
        _safe(('Jorgenson 2022','P_kbar'), lambda: predict_jorgenson(
            df_jw, 'P', phase='cpx_only', T_K=y_T+273.15))
    elif track == 'twopx':
        # Two-pyroxene Putirka 2008 equations. Requires both opx + cpx
        # composition frames on the same row.
        try:
            opx = _opx_phase_frame_from_twopx(df_te)
            cpx = _cpx_phase_frame_from_twopx(df_te)
            _safe(('Putirka 36','T_C'), lambda: pt.calculate_cpx_opx_temp(
                equationT='T_Put2008_eq36', opx_comps=opx, cpx_comps=cpx,
                P=y_P) - 273.15)
            _safe(('Putirka 37','T_C'), lambda: pt.calculate_cpx_opx_temp(
                equationT='T_Put2008_eq37', opx_comps=opx, cpx_comps=cpx,
                P=y_P) - 273.15)
            _safe(('Putirka 38','P_kbar'), lambda: pt.calculate_cpx_opx_press(
                equationP='P_Put2008_eq38', opx_comps=opx, cpx_comps=cpx,
                T=y_T+273.15))
            _safe(('Putirka 39','P_kbar'), lambda: pt.calculate_cpx_opx_press(
                equationP='P_Put2008_eq39', opx_comps=opx, cpx_comps=cpx,
                T=y_T+273.15))
        except Exception as e:
            _log(f'  twopx external skipped: {e}', fh)
    # universal: no equivalent external model; v10-only.
    return preds


def _opx_phase_frame_from_twopx(df):
    """Twopx dataset uses `opx_*` prefixed columns."""
    OX = ['SiO2','TiO2','Al2O3','Cr2O3','MnO','MgO','CaO','Na2O','K2O']
    out = pd.DataFrame({f'{o}_Opx': df.get(f'opx_{o}', 0) for o in OX})
    feo_col = df.get('opx_FeOt', df.get('opx_FeO', pd.Series(0, index=df.index)))
    out['FeOt_Opx'] = pd.to_numeric(feo_col, errors='coerce').fillna(0)
    out['P2O5_Opx'] = 0.0
    return out


def _cpx_phase_frame_from_twopx(df):
    OX = ['SiO2','TiO2','Al2O3','Cr2O3','MnO','MgO','CaO','Na2O','K2O']
    out = pd.DataFrame({f'{o}_Cpx': df.get(f'cpx_{o}', 0) for o in OX})
    feo_col = df.get('cpx_FeOt', df.get('cpx_FeO_total',
                 df.get('cpx_FeO', pd.Series(0, index=df.index))))
    out['FeOt_Cpx'] = pd.to_numeric(feo_col, errors='coerce').fillna(0)
    out['P2O5_Cpx'] = 0.0
    return out


def _family_for(method_name):
    if method_name.startswith('Putirka'):
        return 'Putirka'
    if method_name.startswith('Agreda'):
        return 'Agreda'
    if method_name.startswith('Jorgenson'):
        return 'Jorgenson'
    if method_name.startswith('Wang'):
        return 'Wang'
    return 'External'


def run_track(pipeline, track, df_tr, df_te, fh):
    """Iterate all cells for (pipeline, track) and both targets, plus
    external baselines. Returns list of rows."""
    rows = []
    cells = [c for c in load_all_cells(pipeline) if c[2] == track]
    _log(f'[{pipeline}/{track}] cells={len(cells)}', fh)
    y_T = df_te['T_C'].to_numpy(float)
    y_P = df_te['P_kbar'].to_numpy(float)

    # v10: fit every cell, store predictions.
    v10_preds = {}  # (model, target, feat) -> yhat
    for i, (m, tg, tr, fs, bp) in enumerate(cells, 1):
        try:
            X_tr, X_te = build_Xy(pipeline, track, fs, df_tr, df_te)
            est = build_model(m, bp, seed=42)
            est.fit(X_tr, df_tr[tg].to_numpy(float))
            yhat = est.predict(X_te)
            v10_preds[(m, tg, fs)] = np.asarray(yhat, float)
            if i % 8 == 0 or i == len(cells):
                _log(f'  [{i}/{len(cells)}] fit {m}/{tg}/{fs}', fh)
        except Exception as e:
            _log(f'  FAIL cell {m}/{tg}/{fs}: {e}', fh)

    # External baselines
    ext_preds = compute_external_preds(track, df_te, y_T, y_P, fh)

    def _emit(family, method, method_label, tg, yhat, bin_vec,
              regime_label, lo, hi, regime_type, phys_lo, phys_hi):
        in_regime = (bin_vec >= lo) & (bin_vec < hi)
        y_true = y_T if tg == 'T_C' else y_P
        ok = np.isfinite(yhat) & (yhat >= phys_lo) & (yhat <= phys_hi)
        mask = in_regime & ok
        n_reg = int(in_regime.sum())
        if n_reg == 0:
            return None
        metrics = bootstrap_metrics(y_true, yhat, mask)
        row = {
            'pipeline': pipeline, 'track': track, 'target': tg,
            'regime_type': regime_type,
            'regime': regime_label, 'regime_lo': lo, 'regime_hi': hi,
            'n': n_reg,
            'method_family': family,
            'method': method, 'method_label': method_label,
        }
        row.update(metrics)
        return row

    # Emit for P regimes (bin_vec = y_P) and T regimes (bin_vec = y_T)
    for regime_type, bins, bin_vec in [('P', P_REGIMES, y_P),
                                        ('T', T_REGIMES, y_T)]:
        for regime_label, lo, hi in bins:
            if ((bin_vec >= lo) & (bin_vec < hi)).sum() == 0:
                continue
            for (m, tg, fs), yhat in v10_preds.items():
                phys_lo, phys_hi = (500, 2000) if tg == 'T_C' else (-5, 80)
                r = _emit('v10', f'{m}/{fs}', f'v10 {m} {fs}',
                         tg, yhat, bin_vec, regime_label, lo, hi,
                         regime_type, phys_lo, phys_hi)
                if r: rows.append(r)
            for (em, tg), yhat in ext_preds.items():
                phys_lo, phys_hi = (500, 2000) if tg == 'T_C' else (-5, 80)
                r = _emit(_family_for(em), em, em,
                         tg, yhat, bin_vec, regime_label, lo, hi,
                         regime_type, phys_lo, phys_hi)
                if r: rows.append(r)
    return rows


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')
    rows = []
    try:
        _log(f'START regime_allmodels bootstrap n_boot={N_BOOT}', fh)
        from src.data import load_twopx, load_universal
        plan = [
            ('opx',       'opx_liq',   load_opx_liq),
            ('opx',       'opx_only',  load_opx_only),
            ('cpx',       'cpx_liq',   load_cpx_liq),
            ('cpx',       'cpx_only',  load_cpx_only),
            ('twopx',     'twopx',     load_twopx),
            ('universal', 'universal', load_universal),
        ]
        for pipeline, track, loader in plan:
            df = loader()
            tr_idx, te_idx = load_splits(track)
            df_tr = df.iloc[tr_idx].reset_index(drop=True)
            df_te = df.iloc[te_idx].reset_index(drop=True)
            _log(f'[{track}] train n={len(df_tr)} test n={len(df_te)}', fh)
            rows.extend(run_track(pipeline, track, df_tr, df_te, fh))
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)  # checkpoint
            _log(f'[{track}] checkpoint rows={len(rows)}', fh)
        _log(f'DONE total rows={len(rows)} -> {OUT_CSV}', fh)
    finally:
        fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
