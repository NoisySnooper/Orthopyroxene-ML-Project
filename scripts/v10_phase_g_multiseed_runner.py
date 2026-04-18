#!/usr/bin/env python3
"""v10 Phase G multi-seed refit protocol.

For each pipeline in {opx, cpx, twopx, universal}, for each cell
(model, target, track, feature_set) with Optuna best_params already
on disk, refit the base model at every seed in config.SPLIT_SEEDS
(20 seeds, 42..61) and evaluate on the fixed test split. The fixed
train/test split is unchanged (v10 uses Citation-grouped splits on
disk); seed varies only the model's internal stochasticity (tree
splits, bagging, MLP init, etc.).

Writes, per pipeline:
  results/v10_{pipeline}_multiseed_results.csv   long format, 1 row per
                                                 (cell, seed)
  results/v10_{pipeline}_multiseed_summary.csv   per-cell mean/std/
                                                 min/max/count

Log: logs/v10_phase_g_multiseed.log

Incremental: each cell's 20 rows are appended to the results CSV as
soon as they are computed, so a crash loses at most one cell.
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

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from config import RESULTS, LOGS, SPLIT_SEEDS
from src.models import build_model

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_multiseed.log'


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


# ---------------------------------------------------------------------------
# Pipeline-specific prepare_train_test adapters
# ---------------------------------------------------------------------------

def _prep_opx(track, target, feature_set):
    from src.v10_phase_c_analysis import prepare_train_test
    return prepare_train_test(track, target, feature_set)


def _prep_cpx(track, target, feature_set):
    from src.cpx_features import build_cpx_feature_matrix
    from src.data import load_cpx_liq, load_cpx_only, load_splits
    df = load_cpx_liq() if track == 'cpx_liq' else load_cpx_only()
    tr_idx, te_idx = load_splits(track)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    use_liq = (track == 'cpx_liq')
    X_tr, feat_names = build_cpx_feature_matrix(df_tr, feature_set, use_liq=use_liq)
    X_te, _          = build_cpx_feature_matrix(df_te, feature_set, use_liq=use_liq)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    return {'X_tr': np.asarray(X_tr, float), 'y_tr': y_tr,
            'X_te': np.asarray(X_te, float), 'y_te': y_te}


def _prep_twopx(track, target, feature_set):
    from src.twopx_features import build_twopx_feature_matrix
    from src.data import load_twopx, load_splits
    df = load_twopx()
    tr_idx, te_idx = load_splits(track)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    X_tr, feat_names = build_twopx_feature_matrix(df_tr, feature_set)
    X_te, _          = build_twopx_feature_matrix(df_te, feature_set)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    return {'X_tr': np.asarray(X_tr, float), 'y_tr': y_tr,
            'X_te': np.asarray(X_te, float), 'y_te': y_te}


def _prep_universal(track, target, feature_set):
    from src.universal_features import build_universal_matrix
    from src.data import load_universal, load_splits
    df = load_universal()
    tr_idx, te_idx = load_splits('universal')
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    X_tr, feat_names = build_universal_matrix(df_tr)
    X_te, _          = build_universal_matrix(df_te)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    return {'X_tr': np.asarray(X_tr, float), 'y_tr': y_tr,
            'X_te': np.asarray(X_te, float), 'y_te': y_te}


PIPELINES = {
    'opx':        {'best_json': 'v10_optuna_best_params_opx.json',
                   'prep': _prep_opx},
    'cpx':        {'best_json': 'v10_optuna_best_params_cpx.json',
                   'prep': _prep_cpx},
    'twopx':      {'best_json': 'v10_optuna_best_params_twopx.json',
                   'prep': _prep_twopx},
    'universal':  {'best_json': 'v10_optuna_best_params_universal.json',
                   'prep': _prep_universal},
}


def load_cells(best_json_path):
    import json
    with open(best_json_path) as f:
        payload = json.load(f)
    rows = payload['results']
    # One cell per (model, target, track, feature_set); keep best_params.
    return [(r['model'], r['target'], r['track'], r['feature_set'],
             r['best_params']) for r in rows]


def run_cell_multiseed(pipeline, model, target, track, feature_set,
                       best_params, prep_fn, seeds, fh):
    """Refit `model` at each seed, eval on test split, return rows."""
    splits = prep_fn(track, target, feature_set)
    X_tr, y_tr = splits['X_tr'], splits['y_tr']
    X_te, y_te = splits['X_te'], splits['y_te']
    rows = []
    for s in seeds:
        est = build_model(model, best_params, seed=int(s))
        est.fit(X_tr, y_tr)
        y_hat = est.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
        rows.append({
            'pipeline':    pipeline,
            'model':       model,
            'target':      target,
            'track':       track,
            'feature_set': feature_set,
            'seed':        int(s),
            'test_rmse':   rmse,
        })
    return rows


def summarise(results_df):
    g = (results_df.groupby(['pipeline', 'model', 'target', 'track', 'feature_set'])
                    ['test_rmse']
                    .agg(['mean', 'std', 'min', 'max', 'count'])
                    .reset_index())
    return g


def run_pipeline(pipeline, seeds, skip_done, fh):
    cfg = PIPELINES[pipeline]
    best_json = RESULTS / cfg['best_json']
    if not best_json.exists():
        _log(f'[{pipeline}] SKIP (no best_params at {best_json})', fh)
        return
    cells = load_cells(best_json)
    _log(f'[{pipeline}] n_cells={len(cells)} seeds={len(seeds)}', fh)

    results_csv = RESULTS / f'v10_{pipeline}_multiseed_results.csv'
    summary_csv = RESULTS / f'v10_{pipeline}_multiseed_summary.csv'

    # Resume support: if results CSV exists, identify completed cells.
    completed = set()
    if skip_done and results_csv.exists():
        try:
            prev = pd.read_csv(results_csv)
            if len(prev) > 0:
                got = (prev.groupby(['model', 'target', 'track', 'feature_set'])
                            ['seed'].nunique())
                completed = {k for k, v in got.items() if v >= len(seeds)}
                _log(f'[{pipeline}] resume: {len(completed)} cells '
                     f'already at {len(seeds)}+ seeds', fh)
        except Exception as e:
            _log(f'[{pipeline}] resume read failed ({e}), starting fresh', fh)

    # Prep writer. Append mode; header only if file is missing or empty.
    header_needed = (not results_csv.exists()) or (results_csv.stat().st_size == 0)

    t0 = time.time()
    for i, (m, tg, tr, fs, bp) in enumerate(cells, 1):
        if (m, tg, tr, fs) in completed:
            _log(f'[{pipeline}] [{i}/{len(cells)}] {m}/{tg}/{tr}/{fs}  SKIP (done)', fh)
            continue
        tc = time.time()
        try:
            rows = run_cell_multiseed(pipeline, m, tg, tr, fs, bp,
                                      cfg['prep'], seeds, fh)
        except Exception as e:
            import traceback
            _log(f'[{pipeline}] [{i}/{len(cells)}] {m}/{tg}/{tr}/{fs}  FAIL {e}', fh)
            _log(traceback.format_exc(), fh)
            continue
        df = pd.DataFrame(rows)
        df.to_csv(results_csv, index=False,
                  mode='a', header=header_needed)
        header_needed = False
        mean = df['test_rmse'].mean()
        std = df['test_rmse'].std()
        _log(f'[{pipeline}] [{i}/{len(cells)}] {m}/{tg}/{tr}/{fs}  '
             f'mean={mean:.3f} std={std:.3f} elapsed={time.time()-tc:.1f}s', fh)

    # Write summary from whatever is on disk now.
    if results_csv.exists():
        full = pd.read_csv(results_csv)
        s = summarise(full)
        s.to_csv(summary_csv, index=False)
        _log(f'[{pipeline}] wrote {summary_csv}  rows={len(s)}', fh)

    _log(f'[{pipeline}] DONE  elapsed={time.time()-t0:.1f}s', fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipelines', nargs='+',
                    default=['opx', 'cpx', 'twopx', 'universal'],
                    choices=list(PIPELINES))
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Override seeds; default uses config.SPLIT_SEEDS')
    ap.add_argument('--skip-done', action='store_true',
                    help='Skip cells already at full seed count in CSV')
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else list(SPLIT_SEEDS)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')
    try:
        _log(f'START pipelines={args.pipelines} seeds={seeds}', fh)
        for p in args.pipelines:
            run_pipeline(p, seeds, args.skip_done, fh)
        _log('ALL PIPELINES DONE', fh)
    finally:
        fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
