#!/usr/bin/env python3
"""Phase H.3b (pre-staged): canonical cpx inference on natural samples.

Status (2026-04-18): PRE-STAGED. Runs only after H.1b (GEOROC cpx pull)
populates `data/natural/natural_cpx_with_coords.csv`. Until then this
script errors out cleanly with an informative message, by design.

Canonical cells per `results/v10_canonical_cells_h0b.{csv,json}`
(aggregate 20-seed winners):
  * cpx_only  T_C    : ERT      / pwlr
  * cpx_only  P_kbar : MLP      / alr
  * cpx_liq   T_C    : ERT      / pwlr
  * cpx_liq   P_kbar : LightGBM / pwlr

The cpx_liq track also needs a natural glass/liquid pair file from H.1c.
If that file is absent, cpx_liq inference is skipped and only cpx_only
predictions are written. This makes the script partial-runnable the
moment H.1b lands, without waiting for H.1c.

Inputs:
  data/natural/natural_cpx_with_coords.csv      (H.1b; required)
  data/natural/natural_cpx_liq_pairs.csv        (H.1d; optional)
  data/processed/cpx_clean_cpx_only.parquet     (training set for OOD)
  data/processed/cpx_clean_cpx_liq.parquet      (training set for OOD)
  models/canonical/cpx/base_ERT_T_C_cpx_only_pwlr.joblib
  models/canonical/cpx/base_MLP_P_kbar_cpx_only_alr.joblib
  models/canonical/cpx/base_ERT_T_C_cpx_liq_pwlr.joblib
  models/canonical/cpx/base_LightGBM_P_kbar_cpx_liq_pwlr.joblib

Outputs:
  results/v10_natural_cpx_cpx_only_predictions.csv
  results/v10_natural_cpx_cpx_liq_predictions.csv   (only if pairs file)
    - original cols
    - pred_T_C_<track>
    - pred_P_kbar_<track>
    - ood_isoforest_score_<track>
    - ood_flag_<track>

No MC uncertainty; canonical artifacts are single-seed. Per Phase G
collision 4, no per-regime RMSE claims on natural samples.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import DATA_NATURAL, DATA_PROC, MODELS, LOGS, RESULTS
from src.features import (add_engineered_features, build_feature_matrix,
                          cation_recalc_6oxy)

LOG_PATH = LOGS / 'v10_phase_h3b_cpx_inference.log'

NATURAL_CPX_CSV = DATA_NATURAL / 'natural_cpx_with_coords.csv'
NATURAL_CPX_LIQ_CSV = DATA_NATURAL / 'natural_cpx_liq_pairs.csv'

TRAIN_CPX_ONLY = DATA_PROC / 'cpx_clean_cpx_only.parquet'
TRAIN_CPX_LIQ = DATA_PROC / 'cpx_clean_cpx_liq.parquet'

CANONICAL = {
    'cpx_only': {
        'T_C':    MODELS / 'canonical' / 'cpx' / 'base_ERT_T_C_cpx_only_pwlr.joblib',
        'P_kbar': MODELS / 'canonical' / 'cpx' / 'base_MLP_P_kbar_cpx_only_alr.joblib',
        'T_feat': 'pwlr',
        'P_feat': 'alr',
        'use_liq': False,
        'train': TRAIN_CPX_ONLY,
        'natural': NATURAL_CPX_CSV,
        'out': RESULTS / 'v10_natural_cpx_cpx_only_predictions.csv',
    },
    'cpx_liq': {
        'T_C':    MODELS / 'canonical' / 'cpx' / 'base_ERT_T_C_cpx_liq_pwlr.joblib',
        'P_kbar': MODELS / 'canonical' / 'cpx' / 'base_LightGBM_P_kbar_cpx_liq_pwlr.joblib',
        'T_feat': 'pwlr',
        'P_feat': 'pwlr',
        'use_liq': True,
        'train': TRAIN_CPX_LIQ,
        'natural': NATURAL_CPX_LIQ_CSV,
        'out': RESULTS / 'v10_natural_cpx_cpx_liq_predictions.csv',
    },
}

OOD_COLS_BASE = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total',
                 'MnO', 'MgO', 'CaO', 'Na2O', 'Mg_num']


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def prepare_features(df):
    return add_engineered_features(cation_recalc_6oxy(df))


def run_track(track, cfg, fh):
    natural_path = cfg['natural']
    if not natural_path.exists():
        _log(f'SKIP {track}: natural input missing -> {natural_path}', fh)
        return False

    nat = pd.read_csv(natural_path, low_memory=False)
    _log(f'{track}: natural loaded {nat.shape}', fh)

    tr = pd.read_parquet(cfg['train'])
    _log(f'{track}: training loaded {tr.shape}', fh)

    nat_e = prepare_features(nat)
    tr_e = prepare_features(tr)

    X_nat_T, _ = build_feature_matrix(nat_e, feature_set=cfg['T_feat'],
                                      use_liq=cfg['use_liq'])
    X_tr_T, _ = build_feature_matrix(tr_e, feature_set=cfg['T_feat'],
                                     use_liq=cfg['use_liq'])
    X_nat_P, _ = build_feature_matrix(nat_e, feature_set=cfg['P_feat'],
                                      use_liq=cfg['use_liq'])
    X_tr_P, _ = build_feature_matrix(tr_e, feature_set=cfg['P_feat'],
                                     use_liq=cfg['use_liq'])
    _log(f'{track}: T features {X_nat_T.shape} P features {X_nat_P.shape}', fh)

    t_model = joblib.load(cfg['T_C'])
    p_model = joblib.load(cfg['P_kbar'])
    _log(f'{track}: T={type(t_model).__name__} P={type(p_model).__name__}', fh)

    t_pred = t_model.predict(X_nat_T)
    p_pred = p_model.predict(X_nat_P)
    _log(f'{track}: T mean={t_pred.mean():.1f} min={t_pred.min():.1f} '
         f'max={t_pred.max():.1f}', fh)
    _log(f'{track}: P mean={p_pred.mean():.2f} min={p_pred.min():.2f} '
         f'max={p_pred.max():.2f}', fh)

    ood_cols = [c for c in OOD_COLS_BASE
                if c in tr_e.columns and c in nat_e.columns]
    iso = IsolationForest(n_estimators=200, random_state=42,
                          contamination='auto')
    iso.fit(tr_e[ood_cols].fillna(0).values)
    train_scores = iso.score_samples(tr_e[ood_cols].fillna(0).values)
    nat_scores = iso.score_samples(nat_e[ood_cols].fillna(0).values)
    ood_threshold = np.percentile(train_scores, 1.0)
    ood_flag = nat_scores < ood_threshold
    _log(f'{track}: OOD threshold={ood_threshold:.4f}, flagged '
         f'{ood_flag.sum()}/{len(ood_flag)} ({ood_flag.mean()*100:.1f}%)', fh)

    out = nat.copy()
    out[f'pred_T_C_{track}'] = t_pred
    out[f'pred_P_kbar_{track}'] = p_pred
    out[f'ood_isoforest_score_{track}'] = nat_scores
    out[f'ood_flag_{track}'] = ood_flag
    out.to_csv(cfg['out'], index=False)
    _log(f'{track}: wrote {cfg["out"]} shape={out.shape}', fh)

    if 'TECTONIC SETTING' in out.columns:
        summary = (out.groupby('TECTONIC SETTING')
                      .agg(n=(f'pred_T_C_{track}', 'size'),
                           T_mean=(f'pred_T_C_{track}', 'mean'),
                           P_mean=(f'pred_P_kbar_{track}', 'mean'),
                           ood_pct=(f'ood_flag_{track}',
                                    lambda s: 100 * s.mean()))
                      .sort_values('n', ascending=False))
        _log(f'{track}: tectonic summary:\n' + summary.to_string(), fh)
    return True


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START H.3b cpx inference (pre-staged)', fh)
        ran = []
        for track, cfg in CANONICAL.items():
            ok = run_track(track, cfg, fh)
            if ok:
                ran.append(track)
        if not ran:
            _log('no cpx tracks executed: awaiting H.1b (and H.1d for '
                 'cpx_liq). This is expected until GEOROC cpx lands.', fh)
            _log('DONE (no-op)', fh)
            return 2
        _log(f'DONE; tracks executed: {ran}', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
