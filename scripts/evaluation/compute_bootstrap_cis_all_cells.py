#!/usr/bin/env python3
"""Phase 1 primary fix: compute bootstrap RMSE CIs for every (pipeline,
model, feature_set, track, target) cell using canonical seed=42 models
and the canonical held-out test set.

Output: results/bootstrap_rmse_cis_all_cells.csv

Strategy
--------
The 20-seed multiseed runner is archived and regenerating per-seed
predictions for 8 families x 8 cells x 3 feature sets x 20 seeds is out
of scope for Phase 1. This script computes bootstrap CIs from the
canonical seed=42 fit, which is the fit reported in the manuscript
Table 2 headline. For TabPFN, per-seed sample-level predictions are
already saved in `results/tabpfn_predictions.csv`; we bootstrap seed=42
TabPFN predictions to match the rest of the grid. ci_source column
records provenance.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import bootstrap_rmse_ci  # noqa: E402
from src.prepare_train_test import prepare_train_test  # noqa: E402

MODELS_DIR = PROJECT_ROOT / 'models' / 'canonical'

OPX_CELLS = [
    ('opx', 'opx_liq', 'T_C'),
    ('opx', 'opx_liq', 'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
]
CPX_CELLS = [
    ('cpx', 'cpx_liq', 'T_C'),
    ('cpx', 'cpx_liq', 'P_kbar'),
    ('cpx', 'cpx_only', 'T_C'),
    ('cpx', 'cpx_only', 'P_kbar'),
]
ALL_CELLS = OPX_CELLS + CPX_CELLS
TUNED_FAMILIES = ['RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM',
                  'ElasticNet', 'MLP']
FEATURE_SETS = ['raw', 'alr', 'pwlr']


def canonical_path(pipeline, model, target, track, feature_set):
    return (MODELS_DIR / pipeline
            / f'base_{model}_{target}_{track}_{feature_set}.joblib')


def load_summary_mean_std(pipeline, model, feature_set, track, target):
    csv = f'results/{pipeline}_multiseed_summary.csv'
    df = pd.read_csv(csv)
    m = df[(df.model == model) & (df.feature_set == feature_set)
           & (df.track == track) & (df.target == target)]
    if len(m) == 0:
        return np.nan, np.nan
    return float(m['mean'].iloc[0]), float(m['std'].iloc[0])


def load_tabpfn_seed42(track, target):
    df = pd.read_csv('results/tabpfn_predictions.csv')
    m = df[(df.track == track) & (df.target == target) & (df.seed == 42)]
    if len(m) == 0:
        return None, None
    return m['y_true'].to_numpy(dtype=float), m['y_pred'].to_numpy(dtype=float)


def load_tabpfn_summary(track, target):
    df = pd.read_csv('results/tabpfn_multiseed_summary.csv')
    m = df[(df.track == track) & (df.target == target)]
    if len(m) == 0:
        return np.nan, np.nan
    return float(m['mean'].iloc[0]), float(m['std'].iloc[0])


def main():
    rows = []
    for pipeline, track, target in ALL_CELLS:
        prepared = None
        for fs in FEATURE_SETS:
            if prepared is None or prepared.get('fs') != fs:
                d = prepare_train_test(pipeline, track, target, fs)
                prepared = {'fs': fs, 'd': d}
            d = prepared['d']
            for model in TUNED_FAMILIES:
                mp = canonical_path(pipeline, model, target, track, fs)
                if not mp.exists():
                    print(f'MISSING: {mp.name}')
                    continue
                try:
                    est = joblib.load(mp)
                    y_pred = np.asarray(est.predict(d['X_te']), dtype=float)
                except Exception as e:
                    print(f'predict failed for {mp.name}: {e}')
                    continue
                y_true = d['y_te']
                rmse_point, ci_lo, ci_hi = bootstrap_rmse_ci(
                    y_true, y_pred, n_boot=500, random_state=42)
                seed_mean, seed_std = load_summary_mean_std(
                    pipeline, model, fs, track, target)
                rows.append({
                    'pipeline':       pipeline,
                    'model':          model,
                    'feature_set':    fs,
                    'track':          track,
                    'target':         target,
                    'rmse_point':     seed_mean if not np.isnan(seed_mean) else rmse_point,
                    'rmse_seed42':    rmse_point,
                    'rmse_seed_std':  seed_std,
                    'rmse_ci_lo':     ci_lo,
                    'rmse_ci_hi':     ci_hi,
                    'ci_half_width':  (ci_hi - ci_lo) / 2.0,
                    'n_test':         int(len(y_true)),
                    'ci_source':      'bootstrap_n500_seed42',
                })
        # TabPFN
        yt, yp = load_tabpfn_seed42(track, target)
        if yt is not None:
            rmse_point, ci_lo, ci_hi = bootstrap_rmse_ci(
                yt, yp, n_boot=500, random_state=42)
            tab_mean, tab_std = load_tabpfn_summary(track, target)
            rows.append({
                'pipeline':       pipeline,
                'model':          'TabPFN',
                'feature_set':    'raw',
                'track':          track,
                'target':         target,
                'rmse_point':     tab_mean if not np.isnan(tab_mean) else rmse_point,
                'rmse_seed42':    rmse_point,
                'rmse_seed_std':  tab_std,
                'rmse_ci_lo':     ci_lo,
                'rmse_ci_hi':     ci_hi,
                'ci_half_width':  (ci_hi - ci_lo) / 2.0,
                'n_test':         int(len(yt)),
                'ci_source':      'bootstrap_n500_seed42_tabpfn',
            })

    out = pd.DataFrame(rows)
    outp = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
    out.to_csv(outp, index=False)
    print(f'wrote {outp}')
    print(f'  rows: {len(out)}')
    print(f'  n cells covered: {out.groupby(["pipeline","track","target"]).ngroups}')
    n_families_per_cell = out.groupby(
        ['pipeline', 'track', 'target']).size()
    print(f'  families per cell: {sorted(n_families_per_cell.unique())}')
    zero_ci = out[(out.ci_half_width == 0.0)]
    if len(zero_ci):
        print(f'  WARNING: {len(zero_ci)} rows have zero CI width')


if __name__ == '__main__':
    main()
