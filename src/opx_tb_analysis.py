"""v10 Phase C analysis: refit, OOF preds, ensembles, tests for opx pipeline.

Consumes `results/v10_optuna_best_params_opx.json` (produced by
`scripts/v10_phase_c_driver.py`) and generates:
  - Final fitted base models on the full training set
  - OOF prediction matrix (8 bases x N_train)
  - 3 internal ensembles (Ridge stack, two-level stack, Caruana greedy)
    plus AutoGluon is handled separately via a driver script.
  - Test-set RMSE / R2 per (model, target, track, feature_set)
  - T01, T02, T11, T12 test outcomes via src/test_protocol logger

The module is intentionally function-first so the notebook cell that
calls it stays short and the logic remains unit-testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.data import load_opx_liq, load_opx_only, load_splits
from src.ensembles import fit_greedy_ensemble, fit_two_level_stack
from src.features import build_feature_matrix
from src.models import build_model
from src.stacking import fit_ridge_meta_model

BASE_ORDER = ('RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM', 'ElasticNet', 'MLP')
TARGETS = ('T_C', 'P_kbar')
TRACKS = ('opx_only', 'opx_liq')
FEATURE_SETS = ('raw', 'alr', 'pwlr')

V10_BASE_ORDER = BASE_ORDER
V10_TARGETS = TARGETS
V10_TRACKS = TRACKS
V10_FEATURE_SETS = FEATURE_SETS


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_track(track):
    return load_opx_liq() if track == 'opx_liq' else load_opx_only()


def prepare_train_test(track, target, feature_set):
    df = load_track(track)
    tr_idx, te_idx = load_splits(track)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    use_liq = (track == 'opx_liq')
    X_tr, feat_names = build_feature_matrix(df_tr, feature_set, use_liq=use_liq)
    X_te, _          = build_feature_matrix(df_te, feature_set, use_liq=use_liq)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    groups_tr = df_tr['Citation'].to_numpy()
    groups_te = df_te['Citation'].to_numpy()
    return {
        'X_tr': np.asarray(X_tr, float), 'y_tr': y_tr, 'groups_tr': groups_tr,
        'X_te': np.asarray(X_te, float), 'y_te': y_te, 'groups_te': groups_te,
        'feat_names': feat_names,
    }


# ---------------------------------------------------------------------------
# Best-params manifest
# ---------------------------------------------------------------------------

def load_best_params(manifest_path):
    """Return a dict keyed by (model, target, track, feature_set)."""
    with open(manifest_path) as f:
        payload = json.load(f)
    out = {}
    for r in payload['results']:
        key = (r['model'], r['target'], r['track'], r['feature_set'])
        out[key] = {
            'best_params': r['best_params'],
            'best_score':  r['best_score'],
        }
    return out


# ---------------------------------------------------------------------------
# Refit + OOF preds
# ---------------------------------------------------------------------------

def refit_model(model_name, best_params, X_tr, y_tr):
    est = build_model(model_name, best_params)
    est.fit(X_tr, y_tr)
    return est


def generate_oof_vector(model_name, best_params, X_tr, y_tr, groups_tr,
                        n_splits=3, seed=42):
    """OOF predictions via GroupKFold. Returns an (N_tr,) array."""
    cv = GroupKFold(n_splits=n_splits)
    n = len(y_tr)
    oof = np.full(n, np.nan, dtype=float)
    for tr, va in cv.split(X_tr, y_tr, groups_tr):
        est = build_model(model_name, best_params, seed=seed)
        est.fit(X_tr[tr], y_tr[tr])
        oof[va] = est.predict(X_tr[va])
    if np.isnan(oof).any():
        raise RuntimeError('OOF vector has unvisited rows.')
    return oof


def build_oof_matrix(best_params_block, X_tr, y_tr, groups_tr,
                     base_order=BASE_ORDER, target=None, track=None,
                     feature_set=None):
    """Stack OOF vectors column-wise for all 8 bases.

    best_params_block is the dict returned by load_best_params(), keyed
    by (model, target, track, feature_set). We select entries matching
    (target, track, feature_set).
    """
    cols = []
    for m in base_order:
        key = (m, target, track, feature_set)
        if key not in best_params_block:
            raise KeyError(f'missing best_params for {key}')
        params = best_params_block[key]['best_params']
        cols.append(generate_oof_vector(m, params, X_tr, y_tr, groups_tr))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_model(est, X_te, y_te):
    y_hat = est.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
    r2 = float(r2_score(y_te, y_hat))
    return {'rmse': rmse, 'r2': r2, 'y_hat': y_hat}


def evaluate_all_bases(best_params_block, splits, base_order=BASE_ORDER,
                       target=None, track=None, feature_set=None):
    rows = []
    preds = {}
    for m in base_order:
        key = (m, target, track, feature_set)
        if key not in best_params_block:
            continue
        params = best_params_block[key]['best_params']
        est = refit_model(m, params, splits['X_tr'], splits['y_tr'])
        ev = evaluate_model(est, splits['X_te'], splits['y_te'])
        preds[m] = ev['y_hat']
        rows.append({
            'model': m, 'target': target, 'track': track,
            'feature_set': feature_set,
            'test_rmse': ev['rmse'], 'test_r2': ev['r2'],
            'cv_rmse': best_params_block[key]['best_score'],
        })
    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def fit_internal_ensembles(oof_matrix, y_tr, groups_tr,
                           base_order=BASE_ORDER, seed=42):
    """Fit Ridge stack, two-level stack, Caruana greedy on OOF matrix."""
    ridge_meta = fit_ridge_meta_model(oof_matrix, y_tr)
    two_level  = fit_two_level_stack(oof_matrix, y_tr, groups_tr, base_order,
                                     seed=seed)
    greedy     = fit_greedy_ensemble(oof_matrix, y_tr, base_order, n_iters=50)
    return {'ridge': ridge_meta, 'two_level': two_level, 'greedy': greedy}


def ensemble_predict_on_test(ensembles, base_test_preds,
                             base_order=BASE_ORDER):
    """Run each ensemble on the test-set base preds dict."""
    # Ridge meta takes an (N, K) matrix in base_order.
    X_meta = np.column_stack([base_test_preds[k] for k in base_order])
    out = {
        'ridge':     ensembles['ridge'].predict(X_meta),
        'two_level': ensembles['two_level'].predict(base_test_preds),
        'greedy':    ensembles['greedy'].predict(base_test_preds),
    }
    return out


def ensemble_rmses(ensemble_preds, y_te):
    return {name: float(np.sqrt(mean_squared_error(y_te, yhat)))
            for name, yhat in ensemble_preds.items()}


# ---------------------------------------------------------------------------
# Tests (T01, T02, T11, T12)
# ---------------------------------------------------------------------------

def test_T01_boosted_primary(base_rmses: Mapping[str, float]) -> dict:
    """Pass if boosted family (XGB, GB, CatBoost, LightGBM) best RMSE
    beats forest family (RF, ERT) best RMSE."""
    forest = min(base_rmses.get(m, np.inf) for m in ('RF', 'ERT'))
    boosted = min(base_rmses.get(m, np.inf)
                  for m in ('XGB', 'GB', 'CatBoost', 'LightGBM'))
    return {
        'passed': bool(boosted < forest),
        'value':  float(forest - boosted),
        'details': {'forest_best': forest, 'boosted_best': boosted},
    }


def test_T02_ensemble_beats_base(ensemble_rmse: Mapping[str, float],
                                 base_rmses: Mapping[str, float]) -> dict:
    best_base = min(base_rmses.values())
    best_ens  = min(ensemble_rmse.values())
    winner = min(ensemble_rmse, key=ensemble_rmse.get)
    return {
        'passed': bool(best_ens < best_base),
        'value':  float(best_base - best_ens),
        'details': {'best_base_rmse': best_base,
                    'best_ensemble':  winner,
                    'ensemble_rmses': dict(ensemble_rmse)},
    }


def test_T11_catlgb_beats_xgb(base_rmses: Mapping[str, float]) -> dict:
    xgb = base_rmses.get('XGB', np.inf)
    cat = base_rmses.get('CatBoost', np.inf)
    lgb = base_rmses.get('LightGBM', np.inf)
    best_new = min(cat, lgb)
    return {
        'passed': bool(best_new < xgb),
        'value':  float(xgb - best_new),
        'details': {'XGB': xgb, 'CatBoost': cat, 'LightGBM': lgb},
    }


def test_T12_mlp_enet_beats_trees(base_rmses: Mapping[str, float]) -> dict:
    tree_best = min(base_rmses.get(m, np.inf)
                    for m in ('RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM'))
    mlp = base_rmses.get('MLP', np.inf)
    enet = base_rmses.get('ElasticNet', np.inf)
    best_nn = min(mlp, enet)
    return {
        'passed': bool(best_nn < tree_best),
        'value':  float(tree_best - best_nn),
        'details': {'tree_best': tree_best, 'MLP': mlp, 'ElasticNet': enet},
    }
