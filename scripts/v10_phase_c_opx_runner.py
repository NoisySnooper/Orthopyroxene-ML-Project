#!/usr/bin/env python3
"""v10 Phase C opx runner: consume Optuna studies, fit finals, run T01-T12 subset.

Reads `results/v10_optuna_best_params_opx.json` (from v10_phase_c_driver.py)
and produces, for each (target, track, feature_set) cell in the grid:

  - Final 8 base models fitted on the full training split
  - OOF prediction matrix for internal ensemble training
  - 3 internal ensembles (Ridge stack, two-level, Caruana greedy)
  - Test-set RMSE/R2 table per base model and per ensemble
  - T01, T02, T11, T12 test results (the tests computable from refit alone)

Outputs:
  results/v10_opx_per_cell_results.csv
  results/v10_opx_ensemble_results.csv
  results/v10_opx_test_log.csv  (rows appended via test_protocol logger)
  models/canonical/opx/base_{model}_{target}_{track}_{feature_set}.joblib
  models/canonical/opx/ens_{method}_{target}_{track}_{feature_set}.joblib
  logs/v10_phase_c_opx_runner.log

Note: T03-T10 tests depend on v9 mechanics (resampling, bias correction,
stacking-variant comparisons) that require cells beyond pure refit. Those
run in a follow-up notebook/script once refit-level results are in hand.

Author: NQTa (with Claude)
Date: 2026-04-16
"""
from __future__ import annotations

import argparse
import json
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

from config import RESULTS, MODELS, LOGS
from src.opx_tb_analysis import (
    V10_BASE_ORDER, V10_TARGETS, V10_TRACKS, V10_FEATURE_SETS,
    prepare_train_test, load_best_params, build_oof_matrix,
    evaluate_all_bases, fit_internal_ensembles,
    ensemble_predict_on_test, ensemble_rmses,
    test_T01_boosted_primary, test_T02_ensemble_beats_base,
    test_T11_catlgb_beats_xgb, test_T12_mlp_enet_beats_trees,
)
from scripts.v10_test_protocol import log_test_result

warnings.filterwarnings('ignore')

BEST_JSON = RESULTS / 'v10_optuna_best_params_opx.json'
CELL_CSV = RESULTS / 'v10_opx_per_cell_results.csv'
ENS_CSV = RESULTS / 'v10_opx_ensemble_results.csv'
CANON_DIR = MODELS / 'canonical' / 'opx'
LOG_PATH = LOGS / 'v10_phase_c_opx_runner.log'


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def run_one_cell(target, track, feature_set, bp, fh):
    """Process one (target, track, feature_set) cell end to end."""
    cell_key = f'{target}/{track}/{feature_set}'
    _log(f'CELL {cell_key} START', fh)
    t0 = time.time()

    splits = prepare_train_test(track, target, feature_set)

    # Base models: refit + test eval.
    base_df, base_preds = evaluate_all_bases(
        bp, splits, target=target, track=track, feature_set=feature_set)
    base_rmse_map = dict(zip(base_df['model'], base_df['test_rmse']))
    _log(f'  bases: {base_rmse_map}', fh)

    # OOF matrix + internal ensembles.
    oof = build_oof_matrix(bp, splits['X_tr'], splits['y_tr'],
                           splits['groups_tr'],
                           target=target, track=track, feature_set=feature_set)
    ensembles = fit_internal_ensembles(oof, splits['y_tr'], splits['groups_tr'])

    # Save fitted ensembles + refitted bases.
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    for m in V10_BASE_ORDER:
        params = bp[(m, target, track, feature_set)]['best_params']
        from src.models import build_model
        est = build_model(m, params)
        est.fit(splits['X_tr'], splits['y_tr'])
        fname = f'base_{m}_{target}_{track}_{feature_set}.joblib'
        joblib.dump(est, CANON_DIR / fname)
    for method, obj in ensembles.items():
        fname = f'ens_{method}_{target}_{track}_{feature_set}.joblib'
        joblib.dump(obj, CANON_DIR / fname)

    # Ensemble test-set RMSEs.
    ens_preds = ensemble_predict_on_test(ensembles, base_preds)
    ens_rmse = ensemble_rmses(ens_preds, splits['y_te'])
    _log(f'  ensembles: {ens_rmse}', fh)

    # T01, T02, T11, T12 logging.
    for tid, tfn, tkw in [
        ('T01', test_T01_boosted_primary, {'base_rmses': base_rmse_map}),
        ('T02', test_T02_ensemble_beats_base,
            {'ensemble_rmse': ens_rmse, 'base_rmses': base_rmse_map}),
        ('T11', test_T11_catlgb_beats_xgb, {'base_rmses': base_rmse_map}),
        ('T12', test_T12_mlp_enet_beats_trees, {'base_rmses': base_rmse_map}),
    ]:
        r = tfn(**tkw)
        log_test_result(tid, pipeline='opx', target=target, track=track,
                        passed=r['passed'], value=r['value'],
                        details={'feature_set': feature_set, **r['details']})

    cell_base_rows = base_df.to_dict('records')
    cell_ens_rows = [
        {'method': k, 'target': target, 'track': track,
         'feature_set': feature_set, 'test_rmse': v}
        for k, v in ens_rmse.items()
    ]
    _log(f'CELL {cell_key} DONE  elapsed={time.time()-t0:.1f}s', fh)
    return cell_base_rows, cell_ens_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=list(V10_TARGETS))
    ap.add_argument('--tracks', nargs='+', default=list(V10_TRACKS))
    ap.add_argument('--feature-sets', nargs='+', default=list(V10_FEATURE_SETS))
    args = ap.parse_args()

    if not BEST_JSON.exists():
        print(f'ERROR: {BEST_JSON} not found. Run v10_phase_c_driver.py first.')
        return 1

    bp = load_best_params(BEST_JSON)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')

    try:
        all_bases, all_ens = [], []
        cells = [(t, tr, fs)
                 for t in args.targets for tr in args.tracks
                 for fs in args.feature_sets]
        _log(f'START opx runner  cells={len(cells)}', fh)

        for i, (t, tr, fs) in enumerate(cells, 1):
            _log(f'[{i}/{len(cells)}]', fh)
            try:
                br, er = run_one_cell(t, tr, fs, bp, fh)
                all_bases.extend(br)
                all_ens.extend(er)
            except Exception as e:
                import traceback
                _log(f'CELL FAILED {t}/{tr}/{fs}: {e}', fh)
                _log(traceback.format_exc(), fh)

        if all_bases:
            pd.DataFrame(all_bases).to_csv(CELL_CSV, index=False)
            _log(f'wrote {CELL_CSV}', fh)
        if all_ens:
            pd.DataFrame(all_ens).to_csv(ENS_CSV, index=False)
            _log(f'wrote {ENS_CSV}', fh)
        _log(f'DONE  n_base_rows={len(all_bases)}  n_ens_rows={len(all_ens)}', fh)
    finally:
        fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
