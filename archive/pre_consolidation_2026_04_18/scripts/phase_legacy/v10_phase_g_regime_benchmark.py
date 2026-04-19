#!/usr/bin/env python3
"""v10 Phase G.1c: benchmark v10 best base vs external models stratified
by true-P regime on each pipeline's fixed test split.

Output: results/v10_regime_benchmark.csv

Each row has columns:
  track, target, regime (label), n,
  v10_method, v10_rmse,
  put_method, put_rmse,
  ag_method, ag_rmse (NaN where Agreda does not apply).

The regime bins are physically motivated:
  P<5, 5-10, 10-20, 20-40, P>=40, P<20 (Agreda calibration window), ALL
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
from sklearn.metrics import mean_squared_error

from config import RESULTS, LOGS, MODELS
from src.data import (load_opx_liq, load_opx_only, load_cpx_liq,
                      load_cpx_only, load_splits)
from src.models import build_model

warnings.filterwarnings('ignore')

OUT_CSV = RESULTS / 'v10_regime_benchmark.csv'
LOG_PATH = LOGS / 'v10_phase_g_regime_benchmark.log'

FE2O3_TO_FEO = 0.8998


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n'); fh.flush()


def rmse(y_true, y_pred, mask):
    y_t = y_true[mask]; y_p = y_pred[mask]
    m = np.isfinite(y_t) & np.isfinite(y_p)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_t[m], y_p[m])))


REGIMES = [
    ('P<5',                  0,   5),
    ('5<=P<10',              5,  10),
    ('10<=P<20',            10,  20),
    ('20<=P<40',            20,  40),
    ('P>=40',               40, 1000),
    ('P<20 (Agreda range)',  0,  20),
    ('ALL',                  0, 1000),
]


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


def v10_best_cell(pipeline, target, track):
    pc = pd.read_csv(RESULTS / f'v10_{pipeline}_per_cell_results.csv')
    sub = pc[(pc.target==target) & (pc.track==track)]
    r = sub.loc[sub['test_rmse'].idxmin()]
    with open(RESULTS / f'v10_optuna_best_params_{pipeline}.json') as f:
        opts = json.load(f)['results']
    bp = next((o['best_params'] for o in opts
               if o['target']==target and o['track']==track
               and o['model']==r['model'] and o['feature_set']==r['feature_set']), None)
    return r['model'], r['feature_set'], bp


def fit_v10(pipeline, track, target, model, feat, bp, df_tr, df_te):
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
    est = build_model(model, bp, seed=42)
    est.fit(np.asarray(X_tr,float), df_tr[target].to_numpy(float))
    return est.predict(np.asarray(X_te,float))


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')
    rows = []
    try:
        _log('START regime benchmark', fh)
        import Thermobar as pt
        from src.external_models import predict_agreda_from_df

        plan = [
            ('opx', 'opx_liq',   load_opx_liq),
            ('opx', 'opx_only',  load_opx_only),
            ('cpx', 'cpx_liq',   load_cpx_liq),
            ('cpx', 'cpx_only',  load_cpx_only),
        ]
        for pipeline, track, loader in plan:
            df = loader()
            tr_idx, te_idx = load_splits(track)
            df_tr = df.iloc[tr_idx].reset_index(drop=True)
            df_te = df.iloc[te_idx].reset_index(drop=True)
            y_T = df_te['T_C'].to_numpy(float)
            y_P = df_te['P_kbar'].to_numpy(float)
            _log(f'[{track}] test n={len(df_te)}', fh)

            # v10 predictions for each target
            v10_preds = {}
            for target in ['T_C', 'P_kbar']:
                m, f, bp = v10_best_cell(pipeline, target, track)
                _log(f'  v10 best {target}: {m}/{f}', fh)
                yhat = fit_v10(pipeline, track, target, m, f, bp, df_tr, df_te)
                v10_preds[target] = (f'v10 {m} {f}', yhat)

            # External model predictions (track-specific)
            ext_T, ext_P = {}, {}  # method-name -> yhat
            ag_T, ag_P = None, None  # Agreda (cpx only)
            if track == 'opx_liq':
                opx = _opx_phase_frame(df_te); liq = _liq_phase_frame_from_opx_liq(df_te)
                ext_T['Putirka 28a'] = np.asarray(pt.calculate_opx_liq_temp(
                    equationT='T_Put2008_eq28a', opx_comps=opx, liq_comps=liq, P=y_P), float) - 273.15
                ext_P['Putirka 29a'] = np.asarray(pt.calculate_opx_liq_press(
                    equationP='P_Put2008_eq29a', opx_comps=opx, liq_comps=liq, T=y_T+273.15), float)
            elif track == 'opx_only':
                opx = _opx_phase_frame(df_te)
                ext_P['Putirka 29c'] = np.asarray(pt.calculate_opx_only_press(
                    equationP='P_Put2008_eq29c', opx_comps=opx, T=y_T+273.15), float)
            elif track == 'cpx_liq':
                cpx = _cpx_phase_frame(df_te); liq = _liq_phase_frame_from_cpx_liq(df_te)
                ext_T['Putirka 33'] = np.asarray(pt.calculate_cpx_liq_temp(
                    equationT='T_Put2008_eq33', cpx_comps=cpx, liq_comps=liq, P=y_P), float) - 273.15
                ext_P['Putirka 30'] = np.asarray(pt.calculate_cpx_liq_press(
                    equationP='P_Put2008_eq30', cpx_comps=cpx, liq_comps=liq, T=y_T+273.15), float)
                df_ag = df_te.rename(columns=_agreda_cpx_liq_rename()).copy()
                df_ag['MnO_Liq'] = 0.0
                ag_T = predict_agreda_from_df(df_ag, MODELS/'external','cpx_liq','T')['median']
                ag_P = predict_agreda_from_df(df_ag, MODELS/'external','cpx_liq','P')['median']
            elif track == 'cpx_only':
                cpx = _cpx_phase_frame(df_te)
                ext_T['Putirka 32d'] = np.asarray(pt.calculate_cpx_only_temp(
                    equationT='T_Put2008_eq32d', cpx_comps=cpx, P=y_P), float) - 273.15
                ext_P['Putirka 32a'] = np.asarray(pt.calculate_cpx_only_press(
                    equationP='P_Put2008_eq32a', cpx_comps=cpx, T=y_T+273.15), float)
                df_ag = df_te.rename(columns=_agreda_cpx_only_rename()).copy()
                ag_T = predict_agreda_from_df(df_ag, MODELS/'external','cpx_only','T')['median']
                ag_P = predict_agreda_from_df(df_ag, MODELS/'external','cpx_only','P')['median']

            for regime_label, lo, hi in REGIMES:
                m_all = (y_P >= lo) & (y_P < hi)
                if m_all.sum() == 0:
                    continue
                for target, y_true in [('T_C', y_T), ('P_kbar', y_P)]:
                    v10_method, v10_yhat = v10_preds[target]
                    ext_map = ext_T if target == 'T_C' else ext_P
                    ag_yhat = ag_T if target == 'T_C' else ag_P
                    row = {
                        'track': track, 'target': target,
                        'regime': regime_label, 'regime_lo': lo, 'regime_hi': hi,
                        'n': int(m_all.sum()),
                        'v10_method': v10_method,
                        'v10_rmse': rmse(y_true, v10_yhat, m_all),
                    }
                    for ext_name, ext_yhat in ext_map.items():
                        # Filter implausibles before scoring
                        if target == 'T_C':
                            good = np.isfinite(ext_yhat) & (ext_yhat > 500) & (ext_yhat < 2000)
                        else:
                            good = np.isfinite(ext_yhat) & (ext_yhat > -5) & (ext_yhat < 80)
                        row['ext_method'] = ext_name
                        row['ext_rmse'] = rmse(y_true, ext_yhat, m_all & good)
                        row['ext_n_used'] = int((m_all & good).sum())
                    if ag_yhat is not None:
                        row['agreda_rmse'] = rmse(y_true, np.asarray(ag_yhat), m_all)
                    rows.append(row)
        out = pd.DataFrame(rows)
        out.to_csv(OUT_CSV, index=False)
        _log(f'wrote {OUT_CSV} rows={len(out)}', fh)
    finally:
        fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
