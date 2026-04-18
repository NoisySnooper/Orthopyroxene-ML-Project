#!/usr/bin/env python3
"""Phase H.3d (partial): RF per-tree spread as uncertainty proxy for
opx_only P_kbar predictions on natural samples.

Canonical P_kbar model for opx_only is RandomForestRegressor/pwlr with
700 trees. This script queries each tree individually, then reports
per-sample std/p10/p90 across trees as an uncertainty estimate. Cheap
relative to retraining and does not require multiseed artifacts.

Important caveats:
  * RF per-tree spread quantifies model variance under bootstrapped
    training sets; it does NOT capture epistemic uncertainty from
    dataset shift (GEOROC vs training calibration domain).
  * Use alongside the ood_flag column from H.3a for context: an OOD-
    flagged sample with tight per-tree spread is still extrapolation.
  * T_C model is LightGBM (single gradient-boosted tree), so per-tree
    spread does not translate. T uncertainty stays deferred.

Inputs:
  results/v10_natural_opx_opx_only_predictions.csv   (from H.3a)
  models/canonical/opx/base_RF_P_kbar_opx_only_pwlr.joblib

Output:
  Updates the H.3a CSV in place, adding:
    - pred_P_kbar_opx_only_tree_std
    - pred_P_kbar_opx_only_tree_p10
    - pred_P_kbar_opx_only_tree_p90
    - pred_P_kbar_opx_only_tree_iqr   (p75 - p25; narrower, more robust)
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

from config import DATA_NATURAL, MODELS, LOGS, RESULTS
from src.features import (add_engineered_features, build_feature_matrix,
                          cation_recalc_6oxy)

LOG_PATH = LOGS / 'v10_phase_h3d_opx_rf_uncertainty.log'
PRED_CSV = RESULTS / 'v10_natural_opx_opx_only_predictions.csv'
NATURAL_CSV = DATA_NATURAL / 'natural_opx_with_coords.csv'
P_MODEL_PATH = MODELS / 'canonical' / 'opx' / 'base_RF_P_kbar_opx_only_pwlr.joblib'


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START H.3d RF per-tree uncertainty (opx_only P_kbar)', fh)
        nat = pd.read_csv(NATURAL_CSV, low_memory=False)
        pred = pd.read_csv(PRED_CSV, low_memory=False)
        if len(nat) != len(pred):
            raise RuntimeError(
                f'row count mismatch: natural={len(nat)} pred={len(pred)}')
        _log(f'loaded {len(nat)} natural rows', fh)

        nat_e = add_engineered_features(cation_recalc_6oxy(nat))
        X, _ = build_feature_matrix(nat_e, feature_set='pwlr', use_liq=False)
        _log(f'pwlr feature matrix: {X.shape}', fh)

        m = joblib.load(P_MODEL_PATH)
        n_trees = len(m.estimators_)
        _log(f'RF loaded: n_trees={n_trees}', fh)

        # Stream per-tree predictions to avoid a single big allocation.
        per_tree = np.empty((n_trees, len(X)), dtype=np.float32)
        for i, est in enumerate(m.estimators_):
            per_tree[i] = est.predict(X).astype(np.float32)
            if (i + 1) % 100 == 0:
                _log(f'  {i + 1}/{n_trees} trees', fh)

        ensemble_mean = per_tree.mean(axis=0)
        ensemble_std = per_tree.std(axis=0, ddof=0)
        p10 = np.percentile(per_tree, 10, axis=0)
        p25 = np.percentile(per_tree, 25, axis=0)
        p75 = np.percentile(per_tree, 75, axis=0)
        p90 = np.percentile(per_tree, 90, axis=0)
        iqr = p75 - p25

        # Sanity: ensemble mean should equal the canonical point prediction.
        delta = np.max(np.abs(ensemble_mean - pred['pred_P_kbar_opx_only'].values))
        _log(f'max |ensemble_mean - canonical pred| = {delta:.6f} kbar', fh)

        _log(f'tree_std   : mean={ensemble_std.mean():.2f} '
             f'median={np.median(ensemble_std):.2f} '
             f'p90={np.percentile(ensemble_std, 90):.2f}', fh)
        _log(f'tree_iqr   : mean={iqr.mean():.2f} '
             f'median={np.median(iqr):.2f}', fh)
        _log(f'tree_p10   : mean={p10.mean():.2f}', fh)
        _log(f'tree_p90   : mean={p90.mean():.2f}', fh)

        pred['pred_P_kbar_opx_only_tree_std'] = ensemble_std
        pred['pred_P_kbar_opx_only_tree_p10'] = p10
        pred['pred_P_kbar_opx_only_tree_p90'] = p90
        pred['pred_P_kbar_opx_only_tree_iqr'] = iqr

        # Spread vs OOD cross-tab for sanity
        if 'ood_flag' in pred.columns:
            ood = pred['ood_flag'].astype(bool).values
            _log(f'tree_std by OOD: '
                 f'OOD mean={ensemble_std[ood].mean():.2f} '
                 f'non-OOD mean={ensemble_std[~ood].mean():.2f}', fh)

        pred.to_csv(PRED_CSV, index=False)
        _log(f'updated {PRED_CSV} shape={pred.shape}', fh)
        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
