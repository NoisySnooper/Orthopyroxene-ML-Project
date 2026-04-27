#!/usr/bin/env python3
"""Compute SHAP feature attributions for the 8 shipped-winner models.

One winning ML family per (pipeline, track, target) cell, as recorded in
`results/bias_correction_shipped.csv`. Per user direction 2026-04-20:
SHAP is restricted to winners only (no full per-family expansion) to keep
runtime within the one-shot Phase 6 budget.

Explainer policy (fast-path, matches agreed scope):
  - Tree models (RF, ERT, LightGBM): `shap.TreeExplainer` — exact, fast.
  - Linear / ElasticNet-in-pipeline: `shap.LinearExplainer` on the
    inner `enet` step, using scaled features.
  - MLP: `sklearn.inspection.permutation_importance` as a fast surrogate
    for |SHAP| feature ranking (KernelExplainer would be ~30+ min per
    cell; permutation is seconds).

Outputs (append-mode):
  - results/shap_values_winners.npz — one array per cell, keyed
    'pipeline__track__target__model__fs' (shape: n_test x n_features)
  - results/shap_importance_winners.csv — long-format importance table
    with columns (pipeline, track, target, model, feature_set, feature,
    mean_abs_shap, rank). Ranks are per-cell.

Notes
-----
- Features are the model's training-space features (e.g. pwlr_* or
  alr_* or raw_*). Feature names come from
  `src.prepare_train_test.prepare_train_test`.
- Background set is a 50-row sample of the training set (for
  LinearExplainer) or unused for TreeExplainer. Sample seed = 42.
- All runs use `SEED_MODEL = 42` canonical seed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model  # noqa: E402
from src.prepare_train_test import prepare_train_test  # noqa: E402


SEED = 42
BACKGROUND_N = 50
MLP_PERMUTATION_N_REPEATS = 20


def _load_best_params(pipeline: str, track: str, target: str,
                      model: str, feature_set: str) -> dict:
    """Look up best_params from the canonical optuna JSON for this cell."""
    path = Path(f'results/optuna_best_params_{pipeline}.json')
    with open(path) as f:
        data = json.load(f)
    for entry in data['results']:
        if (entry['model'] == model and entry['track'] == track
                and entry['target'] == target
                and entry['feature_set'] == feature_set):
            return entry['best_params']
    raise LookupError(f'best_params not found for '
                      f'{pipeline}/{track}/{target} {model}/{feature_set}')


def _shap_tree(est, X_te: np.ndarray, feat_names: list[str]) -> np.ndarray:
    expl = shap.TreeExplainer(est)
    sv = expl.shap_values(X_te)
    return np.asarray(sv, dtype=np.float32)


def _shap_linear(pipeline_est, X_tr: np.ndarray, X_te: np.ndarray,
                 feat_names: list[str]) -> np.ndarray:
    """Explain the ElasticNet inside a (scaler, enet) sklearn Pipeline."""
    inner = pipeline_est.named_steps['enet']
    X_tr_s = pipeline_est.named_steps['scaler'].transform(X_tr)
    X_te_s = pipeline_est.named_steps['scaler'].transform(X_te)
    rng = np.random.default_rng(SEED)
    idx_bg = rng.choice(len(X_tr_s),
                        size=min(BACKGROUND_N, len(X_tr_s)),
                        replace=False)
    bg = X_tr_s[idx_bg]
    expl = shap.LinearExplainer(inner, bg)
    sv = expl.shap_values(X_te_s)
    return np.asarray(sv, dtype=np.float32)


def _perm_importance_as_shap_proxy(est, X_te: np.ndarray, y_te: np.ndarray,
                                   feat_names: list[str]) -> np.ndarray:
    """Permutation importance gives mean |importance| per feature, not
    per-sample SHAP. We broadcast the per-feature importance across
    test rows so downstream code that consumes an (n_te x n_feat)
    array still works; the per-row detail is lost (same signed value
    per row) and documented in the caption.
    """
    r = permutation_importance(est, X_te, y_te, n_repeats=MLP_PERMUTATION_N_REPEATS,
                               random_state=SEED, n_jobs=-1)
    # importances_mean can be negative; SHAP-style magnitude uses abs,
    # but we preserve sign for parity with SHAP convention.
    imp = r.importances_mean
    sv = np.tile(imp, (len(X_te), 1)).astype(np.float32)
    return sv


def run_cell(pipeline: str, track: str, target: str,
             model: str, feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Prepare data, fit model at seed=42, compute SHAP or surrogate."""
    data = prepare_train_test(pipeline=pipeline, track=track,
                              target=target, feature_set=feature_set)
    X_tr, y_tr = data['X_tr'], data['y_tr']
    X_te, y_te = data['X_te'], data['y_te']
    feat_names = data['feat_names']

    best_params = _load_best_params(pipeline, track, target, model, feature_set)
    est = build_model(model, best_params, seed=SEED)
    est.fit(X_tr, y_tr)

    if model in ('RF', 'ERT', 'LightGBM', 'XGB', 'CatBoost'):
        # GB (HistGradientBoosting) lacks a fully-supported TreeExplainer
        # path in shap 0.51; if it surfaces here, fall back to
        # permutation.
        try:
            sv = _shap_tree(est, X_te, feat_names)
        except Exception:
            sv = _perm_importance_as_shap_proxy(est, X_te, y_te, feat_names)
    elif model == 'ElasticNet':
        sv = _shap_linear(est, X_tr, X_te, feat_names)
    elif model == 'MLP':
        sv = _perm_importance_as_shap_proxy(est, X_te, y_te, feat_names)
    else:
        sv = _perm_importance_as_shap_proxy(est, X_te, y_te, feat_names)
    return sv, feat_names


def main():
    shipped = pd.read_csv('results/bias_correction_shipped.csv')
    shipped = shipped[shipped.model != 'TabPFN'].reset_index(drop=True)

    out_npz = Path('results/shap_values_winners.npz')
    out_csv = Path('results/shap_importance_winners.csv')

    sv_dict: dict[str, np.ndarray] = {}
    imp_rows: list[dict] = []
    for i, r in shipped.iterrows():
        pipe, track, target, model, fs = (r.pipeline, r.track, r.target,
                                          r.model, r.feature_set)
        key = f'{pipe}__{track}__{target}__{model}__{fs}'
        print(f'[{i+1}/{len(shipped)}] {key} ... ', end='', flush=True)
        try:
            sv, feat_names = run_cell(pipe, track, target, model, fs)
        except Exception as e:
            print(f'FAILED: {e}')
            continue
        sv_dict[key] = sv
        mean_abs = np.mean(np.abs(sv), axis=0)
        ranks = np.argsort(-mean_abs)
        for rank_i, j in enumerate(ranks):
            imp_rows.append({
                'pipeline': pipe, 'track': track, 'target': target,
                'model': model, 'feature_set': fs,
                'feature': feat_names[j],
                'mean_abs_shap': float(mean_abs[j]),
                'rank': int(rank_i + 1),
            })
        print(f'done (n_test={sv.shape[0]}, n_feat={sv.shape[1]})')

    np.savez_compressed(out_npz, **sv_dict)
    pd.DataFrame(imp_rows).to_csv(out_csv, index=False)
    print(f'\nwrote {out_npz} ({len(sv_dict)} cells)')
    print(f'wrote {out_csv} ({len(imp_rows)} rows)')


if __name__ == '__main__':
    main()
