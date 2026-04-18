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

from config import (
    P_REGIME_BIN_EDGES_KBAR,
    P_REGIME_LABELS,
    P_REGIME_MIN_N_FOR_CLAIMS,
    SEED_BOOTSTRAP,
)


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


STUDY_TYPE_PATTERNS = [
    ('MORB',            r'mid.?ocean ridge|morb\b'),
    ('mantle_melting',  r'peridotite|harzburgite|lherzolite|mantle melt'),
    ('primitive_mafic', r'boninite|komatiite|picrite'),
    ('arc_silicic',     r'andesite|dacite|rhyolite|high.silica|\barc\b|subduction'),
    ('basalt',          r'basalt|tholeiite|alkali'),
    ('partitioning',    r'chromite|chromium|cr partition|partitioning between'),
    ('metamorphic',     r'dehydration|amphibol|gneiss|granulite|eclogite'),
]


def infer_study_type(citations):
    """Return a study-type label array inferred from Citation strings.

    Used as the 'region' proxy for LeaveOneRegionOut: experimental petrology
    citations rarely name a specific geography (the corpus is mostly mantle
    melting / phase equilibria calibrations), so we categorize by the
    petrologic study *type* — mantle_melting, MORB, arc_silicic, basalt,
    primitive_mafic, partitioning, metamorphic, or `other` for strings that
    match no keyword.
    """
    import re
    labels = []
    for c in citations:
        s = str(c).lower()
        matched = None
        for label, pattern in STUDY_TYPE_PATTERNS:
            if re.search(pattern, s, re.IGNORECASE):
                matched = label
                break
        labels.append(matched if matched else 'other')
    return np.array(labels, dtype=object)


def leave_one_region_out_splits(X, y, regions, min_train_fold=50):
    """LeaveOneRegionOut iterator. Folds with training set smaller than
    `min_train_fold` are skipped so each yielded split has enough data to
    refit. Returns a list of (train_idx, test_idx, region_label) tuples —
    the region label lets callers attribute per-fold metrics.
    """
    regions = np.asarray(regions)
    splits = []
    for region in sorted(pd.Series(regions).unique()):
        test_mask = (regions == region)
        train_mask = ~test_mask
        if train_mask.sum() < min_train_fold:
            continue
        if test_mask.sum() < 1:
            continue
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0], region))
    return splits


def target_bin_kfold_splits(X, y, bin_labels, min_train_fold=50):
    """Leave-one-target-bin-out iterator. `bin_labels` is a pre-computed
    categorical label per sample (e.g. from `assign_p_regime`). Same skip-
    rule as LORO — bins with training set below `min_train_fold` are
    dropped. Returns (train_idx, test_idx, bin_label) tuples.
    """
    bin_labels = np.asarray(bin_labels)
    splits = []
    for bin_ in sorted(pd.Series(bin_labels).unique()):
        test_mask = (bin_labels == bin_)
        train_mask = ~test_mask
        if train_mask.sum() < min_train_fold:
            continue
        if test_mask.sum() < 1:
            continue
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0], bin_))
    return splits


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


# --- Pre-registered P-regime analysis ---------------------------------------
# Registration: docs/v10_p_regime_preregistration.md (locked 2026-04-17).
# Bin edges and labels come from config.py; do not hardcode here.

def assign_p_regime(p_kbar):
    """Return regime label array for each pressure value.

    Uses the pre-registered right-open bin structure from config:
    P_REGIME_BIN_EDGES_KBAR = [0, 5, 15, 30, 100]. P = 5.0 is placed in
    'deep_crustal_MASH' (not 'shallow_crustal'). Negatives are clipped to 0;
    values at or above the ceiling fall into 'deeper_mantle'. NaN pressures
    return the literal string 'unassigned'.
    """
    p = np.asarray(p_kbar, dtype=float)
    labels = np.array(['unassigned'] * len(p), dtype=object)
    finite = np.isfinite(p)
    if not finite.any():
        return labels
    p_valid = p[finite]
    p_clipped = np.clip(p_valid, 0.0, P_REGIME_BIN_EDGES_KBAR[-1] - 1e-9)
    bin_idx = np.digitize(p_clipped, P_REGIME_BIN_EDGES_KBAR[1:-1], right=False)
    labels[finite] = np.array([P_REGIME_LABELS[i] for i in bin_idx], dtype=object)
    return labels


def _bootstrap_stat(y_true, y_pred, stat_fn, n_bootstrap=1000,
                    seed=SEED_BOOTSTRAP):
    """Return (point, ci_low, ci_high) for stat_fn(y_true, y_pred) via
    bootstrap on paired (y_true, y_pred) rows with a fixed seed."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    point = float(stat_fn(y_true, y_pred))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boots = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        boots[b] = stat_fn(y_true[idx[b]], y_pred[idx[b]])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (point, float(lo), float(hi))


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))


def compute_per_regime_metrics(y_true, y_pred, p_kbar_true, metric='rmse',
                               n_bootstrap=1000, seed=SEED_BOOTSTRAP,
                               y_pred_lo=None, y_pred_hi=None):
    """Return per-regime metric table with bootstrap 95% CIs.

    Columns: regime, n, metric, metric_ci_low, metric_ci_high,
             sample_size_limited.
    metric options: 'rmse', 't_bias', 'p_bias', 'coverage_90'.
    'coverage_90' requires y_pred_lo and y_pred_hi (prediction-interval
    bounds covering the central 90%). Empty bins return NaN metrics with
    n=0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    p_true = np.asarray(p_kbar_true, dtype=float)
    regimes = assign_p_regime(p_true)

    if metric == 'rmse':
        stat_fn = _rmse
    elif metric in ('t_bias', 'p_bias', 'bias'):
        stat_fn = _bias
    elif metric == 'coverage_90':
        if y_pred_lo is None or y_pred_hi is None:
            raise ValueError("coverage_90 requires y_pred_lo and y_pred_hi")
        y_pred_lo = np.asarray(y_pred_lo, dtype=float)
        y_pred_hi = np.asarray(y_pred_hi, dtype=float)
        stat_fn = None
    else:
        raise ValueError(f"Unknown metric: {metric}")

    rows = []
    for label in P_REGIME_LABELS:
        mask = (regimes == label)
        n = int(mask.sum())
        limited = n < P_REGIME_MIN_N_FOR_CLAIMS

        if n == 0:
            rows.append({
                'regime': label, 'n': 0, 'metric': metric,
                'metric_value': np.nan,
                'metric_ci_low': np.nan, 'metric_ci_high': np.nan,
                'sample_size_limited': True,
            })
            continue

        if metric == 'coverage_90':
            yt = y_true[mask]; lo = y_pred_lo[mask]; hi = y_pred_hi[mask]
            cov_fn = lambda _t, _p, _l=lo, _h=hi: float(np.mean((_t >= _l) & (_t <= _h)))
            # Bootstrap on paired indices; use y_true twice as dummy y_pred.
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n, size=(n_bootstrap, n))
            point = float(np.mean((yt >= lo) & (yt <= hi)))
            boots = np.empty(n_bootstrap, dtype=float)
            for b in range(n_bootstrap):
                bi = idx[b]
                boots[b] = float(np.mean((yt[bi] >= lo[bi]) & (yt[bi] <= hi[bi])))
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            rows.append({
                'regime': label, 'n': n, 'metric': metric,
                'metric_value': point,
                'metric_ci_low': float(ci_lo), 'metric_ci_high': float(ci_hi),
                'sample_size_limited': limited,
            })
        else:
            point, lo, hi = _bootstrap_stat(
                y_true[mask], y_pred[mask], stat_fn,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            rows.append({
                'regime': label, 'n': n, 'metric': metric,
                'metric_value': point,
                'metric_ci_low': lo, 'metric_ci_high': hi,
                'sample_size_limited': limited,
            })
    return pd.DataFrame(rows)


def per_regime_benchmark(predictions_dict, y_true, p_kbar_true,
                          metrics=('rmse', 't_bias', 'p_bias'),
                          n_bootstrap=1000, seed=SEED_BOOTSTRAP):
    """Long-format benchmark table across multiple methods.

    predictions_dict: mapping method name -> either {'y_pred': array} or
    {'y_pred': array, 'y_pred_lo': array, 'y_pred_hi': array} for methods
    that report prediction intervals (enables 'coverage_90' metric).

    Returns DataFrame with columns: method, regime, n, metric,
    metric_value, metric_ci_low, metric_ci_high, sample_size_limited.
    """
    frames = []
    for method, preds in predictions_dict.items():
        y_pred = preds.get('y_pred') if isinstance(preds, dict) else preds
        lo = preds.get('y_pred_lo') if isinstance(preds, dict) else None
        hi = preds.get('y_pred_hi') if isinstance(preds, dict) else None
        for metric in metrics:
            if metric == 'coverage_90' and (lo is None or hi is None):
                continue
            df = compute_per_regime_metrics(
                y_true, y_pred, p_kbar_true, metric=metric,
                n_bootstrap=n_bootstrap, seed=seed,
                y_pred_lo=lo, y_pred_hi=hi,
            )
            df.insert(0, 'method', method)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=[
            'method', 'regime', 'n', 'metric', 'metric_value',
            'metric_ci_low', 'metric_ci_high', 'sample_size_limited',
        ])
    return pd.concat(frames, ignore_index=True)


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
