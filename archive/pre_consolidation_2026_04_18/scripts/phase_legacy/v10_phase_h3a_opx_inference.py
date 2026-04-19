#!/usr/bin/env python3
"""Phase H.3a (partial): canonical opx_only inference on natural samples.

Scope note (2026-04-18): opx_liq natural inference is blocked pending
GEOROC glass pull (H.1c). This script covers opx_only only.

Canonical cells per results/v10_opx_multiseed_summary.csv (aggregate
20-seed winners):
  * opx_only T_C    : LightGBM / alr
  * opx_only P_kbar : RF / pwlr

Inputs:
  data/natural/natural_opx_with_coords.csv      (H.1a output, 53050 rows)
  data/processed/opx_clean_opx_only.parquet     (training set for OOD)
  models/canonical/opx/base_LightGBM_T_C_opx_only_alr.joblib
  models/canonical/opx/base_RF_P_kbar_opx_only_pwlr.joblib

Outputs:
  results/v10_natural_opx_opx_only_predictions.csv
    - original cols (metadata + oxides + lat/lon)
    - pred_T_C_opx_only
    - pred_P_kbar_opx_only
    - ood_isoforest_score   (IsolationForest anomaly score; lower = more
                              out-of-distribution)
    - ood_flag              (bool; score below training 1st percentile)

No MC uncertainty this pass; canonical artifacts are single-seed. Adding
20-seed spread requires retraining or per-tree prediction aggregation
and is deferred as a follow-up.
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

LOG_PATH = LOGS / 'v10_phase_h3a_opx_inference.log'
OUT_PATH = RESULTS / 'v10_natural_opx_opx_only_predictions.csv'

NATURAL_CSV = DATA_NATURAL / 'natural_opx_with_coords.csv'
TRAIN_PARQUET = DATA_PROC / 'opx_clean_opx_only.parquet'

T_MODEL_PATH = MODELS / 'canonical' / 'opx' / 'base_LightGBM_T_C_opx_only_alr.joblib'
P_MODEL_PATH = MODELS / 'canonical' / 'opx' / 'base_RF_P_kbar_opx_only_pwlr.joblib'


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def prepare_features(df):
    """Apply cation recalc + engineered features. Returns a copy."""
    return add_engineered_features(cation_recalc_6oxy(df))


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START H.3a (opx_only natural inference)', fh)

        # ---- load natural ----
        nat = pd.read_csv(NATURAL_CSV, low_memory=False)
        _log(f'natural samples loaded: {nat.shape}', fh)

        # ---- load training set (for OOD IsolationForest) ----
        tr = pd.read_parquet(TRAIN_PARQUET)
        _log(f'training set loaded: {tr.shape}', fh)

        # ---- feature engineering on both ----
        nat_e = prepare_features(nat)
        tr_e = prepare_features(tr)
        _log('engineered features added', fh)

        # ---- alr features for T_C (LightGBM) ----
        X_nat_alr, _ = build_feature_matrix(nat_e, feature_set='alr',
                                            use_liq=False)
        X_tr_alr, _ = build_feature_matrix(tr_e, feature_set='alr',
                                            use_liq=False)
        _log(f'alr feature shapes: train={X_tr_alr.shape} '
             f'nat={X_nat_alr.shape}', fh)

        # ---- pwlr features for P_kbar (RF) ----
        X_nat_pwlr, _ = build_feature_matrix(nat_e, feature_set='pwlr',
                                              use_liq=False)
        X_tr_pwlr, _ = build_feature_matrix(tr_e, feature_set='pwlr',
                                              use_liq=False)
        _log(f'pwlr feature shapes: train={X_tr_pwlr.shape} '
             f'nat={X_nat_pwlr.shape}', fh)

        # ---- predictions ----
        t_model = joblib.load(T_MODEL_PATH)
        p_model = joblib.load(P_MODEL_PATH)
        _log(f'models loaded: T={type(t_model).__name__} '
             f'P={type(p_model).__name__}', fh)

        t_pred = t_model.predict(X_nat_alr)
        p_pred = p_model.predict(X_nat_pwlr)
        _log(f'T_C pred summary: mean={t_pred.mean():.1f} '
             f'std={t_pred.std():.1f} min={t_pred.min():.1f} '
             f'max={t_pred.max():.1f}', fh)
        _log(f'P_kbar pred summary: mean={p_pred.mean():.2f} '
             f'std={p_pred.std():.2f} min={p_pred.min():.2f} '
             f'max={p_pred.max():.2f}', fh)

        # ---- OOD: IsolationForest on raw oxide inputs ----
        # Feature-set-agnostic: use the 9 training oxides + engineered Mg_num.
        # Training bootstrap sample has ~1k rows; IsolationForest default is
        # contamination='auto' which uses a fixed offset.
        ood_cols = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total',
                    'MnO', 'MgO', 'CaO', 'Na2O', 'Mg_num']
        ood_cols = [c for c in ood_cols if c in tr_e.columns
                    and c in nat_e.columns]
        _log(f'OOD feature cols: {ood_cols}', fh)

        iso = IsolationForest(n_estimators=200, random_state=42,
                              contamination='auto')
        iso.fit(tr_e[ood_cols].fillna(0).values)
        train_scores = iso.score_samples(tr_e[ood_cols].fillna(0).values)
        nat_scores = iso.score_samples(nat_e[ood_cols].fillna(0).values)
        ood_threshold = np.percentile(train_scores, 1.0)
        ood_flag = nat_scores < ood_threshold
        _log(f'OOD threshold (1st pct of train): {ood_threshold:.4f}', fh)
        _log(f'OOD flagged natural samples: {ood_flag.sum()}/{len(ood_flag)} '
             f'({ood_flag.mean()*100:.1f}%)', fh)

        # ---- assemble output ----
        out = nat.copy()
        out['pred_T_C_opx_only'] = t_pred
        out['pred_P_kbar_opx_only'] = p_pred
        out['ood_isoforest_score'] = nat_scores
        out['ood_flag'] = ood_flag

        out.to_csv(OUT_PATH, index=False)
        _log(f'wrote {OUT_PATH} shape={out.shape}', fh)

        # ---- quick sanity summary by tectonic setting ----
        if 'TECTONIC SETTING' in out.columns:
            summary = (out.groupby('TECTONIC SETTING')
                          .agg(n=('pred_T_C_opx_only', 'size'),
                               T_mean=('pred_T_C_opx_only', 'mean'),
                               T_std=('pred_T_C_opx_only', 'std'),
                               P_mean=('pred_P_kbar_opx_only', 'mean'),
                               P_std=('pred_P_kbar_opx_only', 'std'),
                               ood_pct=('ood_flag',
                                        lambda s: 100 * s.mean()))
                          .sort_values('n', ascending=False))
            _log('per-tectonic-setting summary:\n' + summary.to_string(), fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
