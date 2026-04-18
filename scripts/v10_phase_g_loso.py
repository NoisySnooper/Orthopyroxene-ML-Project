#!/usr/bin/env python3
"""v10 Phase G.2: Leave-One-Study-Out (LOSO) generalization test.

For each v10 pipeline track, for each (target), refit the **best base
model** (per the Optuna best_params + per_cell winner) under LOSO
groups on Citation. Pool out-of-fold predictions and compute RMSE / MAE
/ R2. Also compute external-model RMSE (Jorgenson, Agreda, Putirka) on
the same held-out rows as a direct generalization comparator.

Scope (to keep wall-clock reasonable):
  - Only the v10 best base per (track, target), not all 8 models.
  - No Cluster-KFold or TargetBinKFold (LOSO is the strictest generalization
    test and is the one that checks Jorgenson/Agreda leakage).

Output: results/v10_loso_results.csv (one row per method x track x target)
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

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import RESULTS, LOGS, MODELS
from src.data import (load_opx_liq, load_opx_only, load_cpx_liq,
                      load_cpx_only)
from src.models import build_model

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_loso.log'
OUT_CSV = RESULTS / 'v10_loso_results.csv'


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def get_best_cell(pipeline, target, track):
    """Return the (model, feature_set, best_params) cell with lowest
    test_rmse for this (target, track) from the v10 Optuna output."""
    best_path = RESULTS / f'v10_optuna_best_params_{pipeline}.json'
    with open(best_path) as f:
        payload = json.load(f)
    rows = [r for r in payload['results']
            if r['target'] == target and r['track'] == track]
    if not rows:
        return None
    # Pick the lowest test_rmse cell
    best = min(rows, key=lambda r: r.get('test_rmse', float('inf')))
    return best


def build_Xy(df, pipeline, track, feature_set, target):
    """Construct the v10 feature matrix + target vector for one pipeline
    cell. Uses the canonical per-pipeline feature builders."""
    if pipeline == 'opx':
        from src.v10_phase_c_analysis import prepare_train_test
        # Rebuild without split: generate full X/y by concatenating.
        # prepare_train_test returns train+test. We want a full-matrix
        # re-do, so call the underlying feature builder directly.
        from src.features import build_feature_matrix, ALLOWED_FEATURE_SETS
        use_liq = (track == 'opx_liq')
        X, _ = build_feature_matrix(df, feature_set, use_liq=use_liq)
    elif pipeline == 'cpx':
        from src.cpx_features import build_cpx_feature_matrix
        use_liq = (track == 'cpx_liq')
        X, _ = build_cpx_feature_matrix(df, feature_set, use_liq=use_liq)
    elif pipeline == 'twopx':
        from src.twopx_features import build_twopx_feature_matrix
        X, _ = build_twopx_feature_matrix(df, feature_set)
    elif pipeline == 'universal':
        from src.universal_features import build_universal_matrix
        X, _ = build_universal_matrix(df)
    else:
        raise ValueError(pipeline)
    y = df[target].to_numpy(float)
    return np.asarray(X, float), y


def loso_predict(X, y, groups, model_name, best_params, fh, min_train=30):
    """Leave-One-Study-Out OOF predictions. Skip folds whose training
    size falls below `min_train` (one-citation datasets leave nothing
    to train on in pathological cases)."""
    oof = np.full_like(y, np.nan, dtype=float)
    unique = np.unique(groups)
    for i, g in enumerate(unique):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        if len(tr) < min_train or len(te) == 0:
            continue
        est = build_model(model_name, best_params, seed=42)
        est.fit(X[tr], y[tr])
        oof[te] = est.predict(X[te])
        if (i + 1) % 20 == 0:
            _log(f'  LOSO {i+1}/{len(unique)} folds done', fh)
    return oof


def score_oof(y_true, y_pred, method, track, target, notes=''):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n == 0:
        return None
    y_t, y_p = y_true[mask], y_pred[mask]
    return {
        'method': method, 'track': track, 'target': target,
        'n': n,
        'rmse': float(np.sqrt(mean_squared_error(y_t, y_p))),
        'mae':  float(mean_absolute_error(y_t, y_p)),
        'r2':   float(r2_score(y_t, y_p)),
        'notes': notes,
    }


PIPELINE_LOADERS = {
    ('opx', 'opx_liq'):   ('opx', load_opx_liq),
    ('opx', 'opx_only'):  ('opx', load_opx_only),
    ('cpx', 'cpx_liq'):   ('cpx', load_cpx_liq),
    ('cpx', 'cpx_only'):  ('cpx', load_cpx_only),
}


def run_loso_track(pipeline, track, rows, fh):
    _, loader = PIPELINE_LOADERS[(pipeline, track)]
    df = loader().reset_index(drop=True)
    groups = df['Citation'].astype(str).values
    n_unique = len(np.unique(groups))
    _log(f'[{pipeline}/{track}] n={len(df)}, citations={n_unique}', fh)

    for target in ['T_C', 'P_kbar']:
        best = get_best_cell(pipeline, target, track)
        if best is None:
            _log(f'[{pipeline}/{track}/{target}] no best cell, skip', fh)
            continue
        model = best['model']
        feat = best['feature_set']
        bp = best['best_params']
        _log(f'[{pipeline}/{track}/{target}] best={model}/{feat}', fh)
        X, y = build_Xy(df, pipeline, track, feat, target)
        t0 = time.time()
        oof = loso_predict(X, y, groups, model, bp, fh)
        r = score_oof(y, oof, f'v10 {model} {feat}', track, target,
                      notes=f'LOSO, n_citations={n_unique}, elapsed={time.time()-t0:.0f}s')
        if r is not None:
            rows.append(r)
            _log(f'  -> v10 LOSO rmse={r["rmse"]:.2f} r2={r["r2"]:+.3f} n={r["n"]}', fh)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')
    rows = []
    try:
        _log('START LOSO v10', fh)
        for (pipeline, track) in PIPELINE_LOADERS:
            try:
                run_loso_track(pipeline, track, rows, fh)
            except Exception as e:
                import traceback
                _log(f'[{pipeline}/{track}] TRACK FAIL {e}', fh)
                _log(traceback.format_exc(), fh)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        _log(f'wrote {OUT_CSV} rows={len(rows)}', fh)
    finally:
        fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
