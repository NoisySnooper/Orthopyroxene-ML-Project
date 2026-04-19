#!/usr/bin/env python3
"""v10 Phase C driver: 96 Optuna TPE studies for opx pipeline.

Grid: 8 models x 2 targets x 2 tracks x 3 feature sets = 96 studies.

Each study uses GroupKFold(n_splits=3) by Citation, neg-RMSE scoring,
TPESampler(multivariate=True, seed=OPTUNA_SEED), MedianPruner.

Outputs:
  results/v10_optuna_studies/opx/study_{model}_{target}_{track}_{fs}.joblib
  results/v10_optuna_best_params_opx.json              (final)
  results/v10_optuna_best_params_opx_partial.json      (incremental)
  logs/v10_phase_c_driver.log                           (progress + errors)

Companion to: docs/master_plan.md Section 6 (Phase C).
Author: NQTa (with Claude)
Date: 2026-04-16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import (
    OPTUNA_SEED,
    OPTUNA_N_JOBS_INNER,
    RESULTS,
    LOGS,
)
from src.data import load_opx_liq, load_opx_only, load_splits
from src.features import build_feature_matrix
from src.optuna_search import optuna_search

MODELS = ('RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM', 'ElasticNet', 'MLP')
TARGETS = ('T_C', 'P_kbar')
TRACKS = ('opx_only', 'opx_liq')
FEATURE_SETS = ('raw', 'alr', 'pwlr')

STUDIES_DIR = RESULTS / 'v10_optuna_studies' / 'opx'
BEST_JSON = RESULTS / 'v10_optuna_best_params_opx.json'
PARTIAL_JSON = RESULTS / 'v10_optuna_best_params_opx_partial.json'
LOG_PATH = LOGS / 'v10_phase_c_driver.log'


def log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def load_track_frame(track):
    return load_opx_liq() if track == 'opx_liq' else load_opx_only()


def prepare_training(track, target, feature_set):
    """Build X_tr, y_tr, groups_tr for one (track, target, feature_set)."""
    df = load_track_frame(track)
    tr_idx, _ = load_splits(track)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)

    use_liq = (track == 'opx_liq')
    X, feat_names = build_feature_matrix(df_tr, feature_set, use_liq=use_liq)
    y = df_tr[target].to_numpy(dtype=float)
    groups = df_tr['Citation'].to_numpy()
    return np.asarray(X, dtype=float), y, groups


def run_one(model, target, track, feature_set, n_trials, fh):
    key = f'{model}__{target}__{track}__{feature_set}'
    save_path = STUDIES_DIR / f'study_{model}_{target}_{track}_{feature_set}.joblib'

    log(f'START {key} (n_trials={n_trials})', fh)
    t0 = time.time()

    X, y, groups = prepare_training(track, target, feature_set)
    log(f'  data shape X={X.shape} y={y.shape} n_groups={len(set(groups))}', fh)

    result = optuna_search(
        model_name=model,
        X=X, y=y, groups=groups,
        n_trials=n_trials,
        seed=OPTUNA_SEED,
        n_jobs_inner=OPTUNA_N_JOBS_INNER,
        study_save_path=save_path,
        study_name=key,
    )
    elapsed = time.time() - t0
    log(f'DONE  {key}  best_rmse={result["best_score"]:.4f}  '
        f'elapsed={elapsed:.1f}s', fh)

    return {
        'model': model,
        'target': target,
        'track': track,
        'feature_set': feature_set,
        'best_params': result['best_params'],
        'best_score': result['best_score'],
        'best_trial': result['best_trial'],
        'elapsed_s': elapsed,
    }


def save_partial(all_results):
    PARTIAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_JSON, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-trials', type=int, default=50,
                    help='Trials per study (default 50; smoke test uses 3).')
    ap.add_argument('--models', nargs='+', default=list(MODELS),
                    help='Subset of base models to run.')
    ap.add_argument('--targets', nargs='+', default=list(TARGETS))
    ap.add_argument('--tracks', nargs='+', default=list(TRACKS))
    ap.add_argument('--feature-sets', nargs='+', default=list(FEATURE_SETS))
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip studies whose .joblib already exists.')
    args = ap.parse_args()

    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fh = open(LOG_PATH, 'a', encoding='utf-8')
    try:
        grid = [(m, t, tr, fs)
                for m in args.models
                for t in args.targets
                for tr in args.tracks
                for fs in args.feature_sets]
        log(f'Phase C driver START  total={len(grid)} studies '
            f'n_trials={args.n_trials}', fh)

        all_results = []
        if PARTIAL_JSON.exists():
            try:
                with open(PARTIAL_JSON) as f:
                    all_results = json.load(f)
                log(f'Resumed partial: {len(all_results)} prior results', fh)
            except Exception:
                all_results = []

        done_keys = {(r['model'], r['target'], r['track'], r['feature_set'])
                     for r in all_results}

        failures = []
        for i, (m, t, tr, fs) in enumerate(grid, 1):
            save_path = STUDIES_DIR / f'study_{m}_{t}_{tr}_{fs}.joblib'
            if args.skip_existing and (m, t, tr, fs) in done_keys and save_path.exists():
                log(f'[{i}/{len(grid)}] SKIP {m}/{t}/{tr}/{fs} (cached)', fh)
                continue

            log(f'[{i}/{len(grid)}]', fh)
            try:
                r = run_one(m, t, tr, fs, args.n_trials, fh)
                # Replace any stale partial entry for this key.
                all_results = [x for x in all_results
                               if (x['model'], x['target'], x['track'],
                                   x['feature_set']) != (m, t, tr, fs)]
                all_results.append(r)
                save_partial(all_results)
            except Exception as e:
                msg = f'FAIL  {m}/{t}/{tr}/{fs}  {type(e).__name__}: {e}'
                log(msg, fh)
                log(traceback.format_exc(), fh)
                failures.append({'model': m, 'target': t, 'track': tr,
                                 'feature_set': fs, 'error': str(e)})

        # Final manifest.
        with open(BEST_JSON, 'w') as f:
            json.dump({
                'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'n_trials': args.n_trials,
                'results': all_results,
                'failures': failures,
            }, f, indent=2, default=str)
        log(f'Phase C driver DONE  completed={len(all_results)}  '
            f'failed={len(failures)}', fh)
        return 0 if not failures else 2
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
