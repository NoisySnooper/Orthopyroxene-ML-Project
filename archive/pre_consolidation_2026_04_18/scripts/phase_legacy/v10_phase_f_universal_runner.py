#!/usr/bin/env python3
"""v10 Phase F universal runner: refit + ensembles + T01/T02/T11/T12/T13/T14.

Consumes results/v10_optuna_best_params_universal.json. Single track
(universal), 1 feature set (universal_raw), 2 targets = 2 cells. For
each cell: refit all 8 bases, build OOF, fit 3 ensembles, evaluate
test, log T01/T02/T11/T12 plus universal-specific T13 (graceful
degradation across phase-scope subsets) and T14 (all-phase outperforms
single-phase subsets).

Outputs:
  results/universal/v10_universal_per_cell_results.csv
  results/universal/v10_universal_ensemble_results.csv
  results/universal/v10_universal_scope_rmse.csv  (per phase_scope subset)
  results/universal/v10_nb03_test_log.csv  (T01-T14 rows appended)
  models/canonical/universal/base_{model}_{target}_{track}_{feature_set}.joblib
  models/canonical/universal/ens_{method}_{target}_{track}_{feature_set}.joblib
  logs/v10_phase_f_universal_runner.log

Author: NQTa (with Claude)
Date: 2026-04-16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from config import RESULTS, MODELS, LOGS
from src.data import load_universal, load_splits
from src.universal_features import build_universal_matrix
from src.opx_tb_analysis import (
    V10_BASE_ORDER,
    load_best_params, build_oof_matrix,
    evaluate_all_bases, fit_internal_ensembles,
    ensemble_predict_on_test, ensemble_rmses,
    test_T01_boosted_primary, test_T02_ensemble_beats_base,
    test_T11_catlgb_beats_xgb, test_T12_mlp_enet_beats_trees,
)
from scripts.v10_test_protocol import log_test_result

warnings.filterwarnings('ignore')

TRACK = 'universal'
FEATURE_SET = 'universal_raw'
V10_TARGETS = ('T_C', 'P_kbar')

BEST_JSON = RESULTS / 'v10_optuna_best_params_universal.json'
UNIV_DIR = RESULTS / 'universal'
CELL_CSV = UNIV_DIR / 'v10_universal_per_cell_results.csv'
ENS_CSV = UNIV_DIR / 'v10_universal_ensemble_results.csv'
SCOPE_CSV = UNIV_DIR / 'v10_universal_scope_rmse.csv'
CANON_DIR = MODELS / 'canonical' / 'universal'
LOG_PATH = LOGS / 'v10_phase_f_universal_runner.log'


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def prepare_train_test(target):
    df = load_universal()
    tr_idx, te_idx = load_splits(TRACK)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    X_tr, feat_names = build_universal_matrix(df_tr)
    X_te, _          = build_universal_matrix(df_te)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    groups_tr = df_tr['Citation'].to_numpy()
    groups_te = df_te['Citation'].to_numpy()
    return {
        'X_tr': np.asarray(X_tr, float), 'y_tr': y_tr, 'groups_tr': groups_tr,
        'X_te': np.asarray(X_te, float), 'y_te': y_te, 'groups_te': groups_te,
        'feat_names': feat_names,
        'df_te': df_te,
    }


def _scope_rmses(y_te_hat, y_te, df_te):
    """RMSE per phase_scope subset on test set."""
    out = {}
    for scope, sub in df_te.groupby('phase_scope'):
        mask = df_te['phase_scope'].values == scope
        if mask.sum() < 3:
            continue
        out[scope] = {
            'n':    int(mask.sum()),
            'rmse': float(np.sqrt(mean_squared_error(y_te[mask], y_te_hat[mask]))),
        }
    return out


def test_T13_graceful_degradation(scope_rmse_map):
    """Pass if RMSE on larger-phase-count subsets is lower than smaller.

    Phase scopes in increasing information content:
      opx_only, cpx_only    (1 phase)
      opx_liq, cpx_liq, twopx (2 phases)
      twopx_liq             (3 phases)

    Test: median RMSE of 3-phase subset < median of 2-phase < median of 1-phase.
    """
    tiers = {
        1: ['opx_only', 'cpx_only'],
        2: ['opx_liq', 'cpx_liq', 'twopx'],
        3: ['twopx_liq'],
    }
    tier_med = {}
    for tier, scopes in tiers.items():
        rmses = [scope_rmse_map[s]['rmse']
                 for s in scopes if s in scope_rmse_map]
        if rmses:
            tier_med[tier] = float(np.median(rmses))
    ok = (tier_med.get(3, np.inf) <= tier_med.get(2, np.inf)
          <= tier_med.get(1, np.inf))
    return {
        'passed': bool(ok),
        'value':  float(tier_med.get(1, np.nan) - tier_med.get(3, np.nan)),
        'details': {'tier_medians': tier_med, 'scopes': scope_rmse_map},
    }


def test_T14_all_phase_best(scope_rmse_map):
    """Pass if the all-phase subset (twopx_liq) has the lowest RMSE."""
    if not scope_rmse_map:
        return {'passed': False, 'value': 0.0, 'details': {}}
    rmses = {s: v['rmse'] for s, v in scope_rmse_map.items()}
    best_scope = min(rmses, key=rmses.get)
    ok = (best_scope == 'twopx_liq')
    return {
        'passed': bool(ok),
        'value':  float(min(rmses.values())),
        'details': {'best_scope': best_scope, 'rmses': rmses},
    }


def run_one_cell(target, bp, fh):
    cell_key = f'{target}/{TRACK}/{FEATURE_SET}'
    _log(f'CELL {cell_key} START', fh)
    t0 = time.time()

    splits = prepare_train_test(target)
    _log(f'  X_tr={splits["X_tr"].shape}  X_te={splits["X_te"].shape}', fh)

    base_df, base_preds = evaluate_all_bases(
        bp, splits, target=target, track=TRACK, feature_set=FEATURE_SET)
    base_rmse_map = dict(zip(base_df['model'], base_df['test_rmse']))
    _log(f'  bases: {base_rmse_map}', fh)

    oof = build_oof_matrix(bp, splits['X_tr'], splits['y_tr'],
                           splits['groups_tr'],
                           target=target, track=TRACK, feature_set=FEATURE_SET)
    ensembles = fit_internal_ensembles(oof, splits['y_tr'], splits['groups_tr'])

    CANON_DIR.mkdir(parents=True, exist_ok=True)
    for m in V10_BASE_ORDER:
        params = bp[(m, target, TRACK, FEATURE_SET)]['best_params']
        from src.models import build_model
        est = build_model(m, params)
        est.fit(splits['X_tr'], splits['y_tr'])
        joblib.dump(est, CANON_DIR /
                    f'base_{m}_{target}_{TRACK}_{FEATURE_SET}.joblib')
    for method, obj in ensembles.items():
        joblib.dump(obj, CANON_DIR /
                    f'ens_{method}_{target}_{TRACK}_{FEATURE_SET}.joblib')

    ens_preds = ensemble_predict_on_test(ensembles, base_preds)
    ens_rmse = ensemble_rmses(ens_preds, splits['y_te'])
    _log(f'  ensembles: {ens_rmse}', fh)

    # Pick the winning ensemble for scope-stratified evaluation.
    win_ens = min(ens_rmse, key=ens_rmse.get)
    win_pred = ens_preds[win_ens]
    scope_rmse = _scope_rmses(win_pred, splits['y_te'], splits['df_te'])
    _log(f'  winner={win_ens}  scope_rmse={scope_rmse}', fh)

    # Log tests.
    t01 = test_T01_boosted_primary(base_rmse_map)
    log_test_result('T01', pipeline='universal', target=target, track=TRACK,
                    passed=t01['passed'], value=t01['value'],
                    details={'feature_set': FEATURE_SET, **t01['details']})
    t02 = test_T02_ensemble_beats_base(ens_rmse, base_rmse_map)
    log_test_result('T02', pipeline='universal', target=target, track=TRACK,
                    passed=t02['passed'], value=t02['value'],
                    details={'feature_set': FEATURE_SET, **t02['details']})
    t11 = test_T11_catlgb_beats_xgb(base_rmse_map)
    log_test_result('T11', pipeline='universal', target=target, track=TRACK,
                    passed=t11['passed'], value=t11['value'],
                    details={'feature_set': FEATURE_SET, **t11['details']})
    t12 = test_T12_mlp_enet_beats_trees(base_rmse_map)
    log_test_result('T12', pipeline='universal', target=target, track=TRACK,
                    passed=t12['passed'], value=t12['value'],
                    details={'feature_set': FEATURE_SET, **t12['details']})
    t13 = test_T13_graceful_degradation(scope_rmse)
    log_test_result('T13', pipeline='universal', target=target, track=TRACK,
                    passed=t13['passed'], value=t13['value'],
                    details={'feature_set': FEATURE_SET, **t13['details']})
    t14 = test_T14_all_phase_best(scope_rmse)
    log_test_result('T14', pipeline='universal', target=target, track=TRACK,
                    passed=t14['passed'], value=t14['value'],
                    details={'feature_set': FEATURE_SET, **t14['details']})

    cell_base_rows = base_df.to_dict('records')
    cell_ens_rows = [
        {'method': k, 'target': target, 'track': TRACK,
         'feature_set': FEATURE_SET, 'test_rmse': v}
        for k, v in ens_rmse.items()
    ]
    scope_rows = [
        {'target': target, 'track': TRACK, 'feature_set': FEATURE_SET,
         'phase_scope': s, 'n': rec['n'], 'rmse': rec['rmse'],
         'ensemble': win_ens}
        for s, rec in scope_rmse.items()
    ]
    _log(f'CELL {cell_key} DONE  elapsed={time.time()-t0:.1f}s', fh)
    return cell_base_rows, cell_ens_rows, scope_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=list(V10_TARGETS))
    args = ap.parse_args()

    if not BEST_JSON.exists():
        print(f'ERROR: {BEST_JSON} not found. Run v10_phase_f_universal_driver.py first.')
        return 1

    bp = load_best_params(BEST_JSON)
    UNIV_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')

    try:
        all_bases, all_ens, all_scope = [], [], []
        _log(f'START universal runner  targets={args.targets}', fh)

        for t in args.targets:
            try:
                br, er, sr = run_one_cell(t, bp, fh)
                all_bases.extend(br); all_ens.extend(er); all_scope.extend(sr)
            except Exception as e:
                import traceback
                _log(f'CELL FAILED {t}: {e}', fh)
                _log(traceback.format_exc(), fh)

        if all_bases:
            pd.DataFrame(all_bases).to_csv(CELL_CSV, index=False)
            _log(f'wrote {CELL_CSV}', fh)
        if all_ens:
            pd.DataFrame(all_ens).to_csv(ENS_CSV, index=False)
            _log(f'wrote {ENS_CSV}', fh)
        if all_scope:
            pd.DataFrame(all_scope).to_csv(SCOPE_CSV, index=False)
            _log(f'wrote {SCOPE_CSV}', fh)
        _log(f'DONE  n_base={len(all_bases)}  n_ens={len(all_ens)}  '
             f'n_scope={len(all_scope)}', fh)
    finally:
        fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
