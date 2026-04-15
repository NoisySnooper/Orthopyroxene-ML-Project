"""Evaluation metrics and cross-validation utilities.

`compute_metrics` gives RMSE, MAE, R2, and bias. `residual_by_bin`
returns per-bin residual statistics (used for the pressure-range bias
diagnostic). `stratify_labels` produces quintile bins for stratified
grouped CV. `oof_rf` and `oof_qrf` do 10-fold StratifiedGroupKFold OOF
prediction. `loso_splits` and `cluster_kfold_splits` provide iterators
for grouped validation.

v7 Part E additions:
- `resolve_columns` / `COLUMN_ALIASES`: non-mutating alias shim for the
  drift in CSV column naming across results files (M3).
- `qcut_with_warning`: wraps `pd.qcut(..., duplicates='drop')` and logs
  when the realized bin count drops below the requested q (L6).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)

from src.models import predict_median


COLUMN_ALIASES = {
    'T_true_C':     ['T_C_true', 'y_T_true', 'obs_T_C', 'T_true', 'T_C_obs'],
    'P_true_kbar':  ['P_kbar_true', 'y_P_true', 'obs_P_kbar', 'P_true', 'P_kbar_obs'],
    'T_pred_C':     ['T_pred', 'ml_pred_T_C', 'T_C_pred', 'pred_T_C'],
    'P_pred_kbar':  ['P_pred', 'ml_pred_P_kbar', 'P_kbar_pred', 'pred_P_kbar'],
}


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with canonical column aliases present alongside
    the original names. Non-mutating: original CSV column names remain."""
    out = df.copy()
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out[canonical] = out[alias]
                break
    return out


def qcut_with_warning(y, q=5, **kwargs):
    """Wrap pd.qcut so that duplicate-edge bin collapses are logged.

    Use this anywhere the codebase currently calls
    `pd.qcut(..., duplicates='drop')` and the realized bin count matters
    for downstream stratification (NB05 TargetBinKFold, NB07 residual
    diagnostics)."""
    kwargs.setdefault('duplicates', 'drop')
    result, bin_edges = pd.qcut(y, q=q, retbins=True, **kwargs)
    realized = len(bin_edges) - 1
    if realized < q:
        warnings.warn(
            f"qcut requested q={q} bins but produced {realized} after "
            f"dropping duplicate edges (edges={bin_edges.tolist()})",
            RuntimeWarning,
            stacklevel=2,
        )
    return result, bin_edges


def compute_metrics(y_true, y_pred):
    """Standard regression diagnostics plus mean bias."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'r2':   float(r2_score(y_true, y_pred)),
        'bias': float(np.mean(y_pred - y_true)),
        'n':    int(len(y_true)),
    }


def residual_by_bin(y_true, y_pred, bin_edges):
    """Per-bin RMSE, MAE, mean bias, and count. `bin_edges` divides the
    predicted-value axis into contiguous bins. Used to document the
    pressure-range bias (see NB07 and NB09)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            rows.append({'bin_lo': lo, 'bin_hi': hi, 'n': 0,
                         'rmse': np.nan, 'mae': np.nan, 'bias': np.nan})
            continue
        rows.append({
            'bin_lo': lo, 'bin_hi': hi,
            'n': int(mask.sum()),
            'rmse': float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
            'mae':  float(mean_absolute_error(y_true[mask], y_pred[mask])),
            'bias': float(np.mean(y_pred[mask] - y_true[mask])),
        })
    return pd.DataFrame(rows)


def stratify_labels(y, n_bins=5):
    """Return quintile bin labels for stratified-grouped KFold."""
    return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')


def coverage(lo, hi, y):
    """Empirical coverage fraction of an interval (lo, hi)."""
    lo = np.asarray(lo); hi = np.asarray(hi); y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def loso_splits(X, y, groups):
    """Leave-one-study-out split iterator. Returns a list of (train, test)."""
    logo = LeaveOneGroupOut()
    return list(logo.split(X, y, groups=groups))


def cluster_kfold_splits(X, y, clusters):
    """Leave-one-cluster-out via GroupKFold. Number of folds equals the
    number of unique cluster labels."""
    n_clusters = int(pd.Series(clusters).nunique())
    gkf = GroupKFold(n_splits=n_clusters)
    return list(gkf.split(X, y, groups=clusters))


def oof_rf(X, y, groups, params, seed, n_folds=10):
    """Out-of-fold RandomForest predictions on the full training set using
    10-fold StratifiedGroupKFold. Per-fold predictions use `predict_median`
    to match the canonical median-of-trees inference."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_strat = stratify_labels(y)
    oof = np.zeros_like(np.asarray(y, dtype=float))
    for tr, va in sgkf.split(X, y_strat, groups):
        rf = RandomForestRegressor(**params, random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        oof[va] = predict_median(rf, X[va])
    return oof


def oof_qrf(X, y, groups, params, seed, n_folds=10,
            quantiles=(0.16, 0.5, 0.84)):
    """10-fold OOF quantile predictions via quantile_forest. Returns
    (lo, median, hi) arrays aligned with y."""
    from quantile_forest import RandomForestQuantileRegressor
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_strat = stratify_labels(y)
    lo = np.zeros_like(np.asarray(y, dtype=float))
    md = np.zeros_like(np.asarray(y, dtype=float))
    hi = np.zeros_like(np.asarray(y, dtype=float))
    for tr, va in sgkf.split(X, y_strat, groups):
        qrf = RandomForestQuantileRegressor(**params, random_state=seed, n_jobs=-1)
        qrf.fit(X[tr], y[tr])
        q = qrf.predict(X[va], quantiles=list(quantiles))
        lo[va] = q[:, 0]
        md[va] = q[:, 1]
        hi[va] = q[:, 2]
    return lo, md, hi
