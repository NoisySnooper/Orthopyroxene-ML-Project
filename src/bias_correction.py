"""Phase G.7 bias correction: Form A (regime-OLS) and Form B (piecewise).

Factored out 2026-04-18 to generalise the single-cell opx_liq pilot
(scripts/v10_phase_g_nb07_bias.py) to the full 8-combination v10 scope
(opx-liq, opx-only, cpx-liq, cpx-only × T_C, P_kbar), run it at every
seed in config.SPLIT_SEEDS, and add a ship-if-better decision so the
bias correction is only shipped to ExPetDB/GEOROC inference when the
OOF-fit correction demonstrably improves held-out test RMSE.

Two correction forms
--------------------

Form A -- per-regime linear. For each pre-registered P regime r, fit
    y_corr = a_r * y_pred_oof + b_r via OLS on training-set OOF
    predictions. At inference, each test sample gets the correction of
    its assigned regime (regime from *true* P_kbar, matching the
    protocol used for the per-regime RMSE tables).

Form B -- Ágreda-style piecewise (eq S1 in Ágreda-López 2024):
        bias(y) = s_L * (y - a_L)  if y < a_L
                = s_R * (y - a_R)  if y > a_R
                = 0                otherwise
        y_corr = y_pred - bias(y_pred)

    a_L and a_R are absolute thresholds chosen via a quantile grid: the
    grid runs over normalised quantiles alpha_L in [0.05, 0.45] and
    alpha_R in [0.55, 0.95] (11 candidates each), and the actual
    thresholds are `np.quantile(y_pred_oof, alpha)`. Width constraint:
    `a_R - a_L >= 0.25 * (q95 - q5)` of y_pred_oof, so the middle (no-
    correction) region is at least a quarter of the IQR span. For each
    (a_L, a_R) the slopes s_L and s_R are fit by OLS with no intercept
    in their respective regions; candidates with |s| > 2 are discarded
    (pathological). The (a_L, a_R) with minimum OOF MSE of the
    corrected prediction wins.

Ship-if-better
--------------
After both forms are fit on training OOF, we compute pre/post RMSE on
the held-out test split (per-regime + overall). A correction is shipped
only if:
    (overall pre_rmse - overall post_rmse) > SHIP_TOL
    AND
    max over regimes of (post_rmse_r - pre_rmse_r) <= SHIP_TOL
i.e. overall improves by more than the numerical tolerance and no
regime degrades by more than the numerical tolerance. If both Form A
and Form B ship, the one with larger overall improvement wins.

Per-seed semantics matches the existing v10 multi-seed protocol
(scripts/v10_phase_g_multiseed_runner.py): the train/test split on disk
is fixed, and `seed` varies the model's internal stochasticity plus the
OOF CV fold assignment. Nothing about the test set changes across
seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    P_REGIME_LABELS,
    SEED_BOOTSTRAP,
)
from src.evaluation import (
    _bootstrap_stat,
    _rmse,
    assign_p_regime,
    stratify_labels,
)
from src.models import build_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHIP_TOL = 1e-6  # numerical ship tolerance (RMSE, native units)

# Form B quantile grid (user-specified 2026-04-18).
FORM_B_ALPHA_L_GRID = np.linspace(0.05, 0.45, 11)
FORM_B_ALPHA_R_GRID = np.linspace(0.55, 0.95, 11)
FORM_B_WIDTH_MIN_FRAC = 0.25  # (a_R - a_L) >= 0.25 * (q95 - q5)
FORM_B_SLOPE_ABS_MAX = 2.0    # pathology guard


# ---------------------------------------------------------------------------
# OOF prediction
# ---------------------------------------------------------------------------

def oof_predict(model_name: str, best_params: dict,
                X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                seed: int = 42, n_folds: int = 10) -> np.ndarray:
    """10-fold StratifiedGroupKFold OOF predictions stratified on y
    quintiles with Citation (or equivalent) grouping.

    The same `seed` drives both the fold assignment and the model's
    internal stochasticity, so varying seed perturbs both (matching the
    multiseed_runner protocol).
    """
    y_strat = stratify_labels(y)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                random_state=seed)
    oof = np.full_like(y, fill_value=np.nan, dtype=float)
    for tr, va in sgkf.split(X, y_strat, groups=groups):
        est = build_model(model_name, best_params, seed=seed)
        est.fit(X[tr], y[tr])
        oof[va] = est.predict(X[va])
    return oof


# ---------------------------------------------------------------------------
# Form A -- per-regime OLS
# ---------------------------------------------------------------------------

def fit_form_a(y_true: np.ndarray, y_pred: np.ndarray,
               regimes: np.ndarray) -> dict:
    """Fit per-regime OLS y = a*y_pred + b. Returns {regime: (a, b)}.
    Regimes with fewer than 5 finite samples fall back to the identity
    (a=1, b=0).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    regimes = np.asarray(regimes)
    params: dict[str, tuple[float, float]] = {}
    for r in np.unique(regimes):
        mask = (regimes == r) & np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 5:
            params[str(r)] = (1.0, 0.0)
            continue
        lr = LinearRegression()
        lr.fit(y_pred[mask].reshape(-1, 1), y_true[mask])
        params[str(r)] = (float(lr.coef_[0]), float(lr.intercept_))
    return params


def apply_form_a(y_pred: np.ndarray, regimes: np.ndarray,
                 params: dict) -> np.ndarray:
    """Apply per-regime linear correction. Samples whose regime label is
    not in `params` pass through untouched."""
    y_pred = np.asarray(y_pred, dtype=float)
    regimes = np.asarray(regimes)
    y_corr = y_pred.copy()
    for r, (a, b) in params.items():
        mask = (regimes == r)
        if not mask.any():
            continue
        y_corr[mask] = a * y_pred[mask] + b
    return y_corr


# ---------------------------------------------------------------------------
# Form B -- Agreda-style piecewise
# ---------------------------------------------------------------------------

@dataclass
class FormBParams:
    alpha_L: float
    alpha_R: float
    a_L: float
    a_R: float
    s_L: float
    s_R: float
    oof_mse: float

    def to_dict(self) -> dict:
        return {
            'alpha_L': self.alpha_L, 'alpha_R': self.alpha_R,
            'a_L': self.a_L, 'a_R': self.a_R,
            's_L': self.s_L, 's_R': self.s_R,
            'oof_mse': self.oof_mse,
        }


def _fit_slope_no_intercept(dy: np.ndarray, dx: np.ndarray) -> float:
    """Least-squares slope for `dy = s * dx` (no intercept)."""
    num = float(np.sum(dx * dy))
    den = float(np.sum(dx * dx))
    if den <= 0.0:
        return 0.0
    return num / den


def _form_b_apply_slopes(y_pred: np.ndarray,
                         a_L: float, a_R: float,
                         s_L: float, s_R: float) -> np.ndarray:
    bias = np.zeros_like(y_pred, dtype=float)
    left = y_pred < a_L
    right = y_pred > a_R
    bias[left] = s_L * (y_pred[left] - a_L)
    bias[right] = s_R * (y_pred[right] - a_R)
    return y_pred - bias


def fit_form_b(y_true: np.ndarray, y_pred: np.ndarray,
               alpha_L_grid: np.ndarray = FORM_B_ALPHA_L_GRID,
               alpha_R_grid: np.ndarray = FORM_B_ALPHA_R_GRID,
               width_min_frac: float = FORM_B_WIDTH_MIN_FRAC,
               slope_abs_max: float = FORM_B_SLOPE_ABS_MAX,
               ) -> Optional[FormBParams]:
    """Fit Ágreda-style piecewise correction on OOF predictions.

    Grid-search (alpha_L, alpha_R) quantile pairs. For each pair,
    compute thresholds as quantiles of y_pred, check the width
    constraint, fit slopes by OLS-no-intercept in each tail, reject if
    |slope| > slope_abs_max, and score total MSE of the corrected OOF
    prediction. Return the best FormBParams or None if no candidate
    satisfies all constraints.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 20:
        return None
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    q5, q95 = np.quantile(y_pred, [0.05, 0.95])
    width_min = width_min_frac * (q95 - q5)
    if not np.isfinite(width_min) or width_min <= 0:
        return None

    best: Optional[FormBParams] = None
    for alpha_L in alpha_L_grid:
        a_L = float(np.quantile(y_pred, alpha_L))
        for alpha_R in alpha_R_grid:
            a_R = float(np.quantile(y_pred, alpha_R))
            if (a_R - a_L) < width_min:
                continue
            left_mask = y_pred < a_L
            right_mask = y_pred > a_R
            if left_mask.sum() < 5 or right_mask.sum() < 5:
                continue
            resid = y_pred - y_true
            s_L = _fit_slope_no_intercept(
                resid[left_mask], y_pred[left_mask] - a_L)
            s_R = _fit_slope_no_intercept(
                resid[right_mask], y_pred[right_mask] - a_R)
            if abs(s_L) > slope_abs_max or abs(s_R) > slope_abs_max:
                continue
            y_corr = _form_b_apply_slopes(y_pred, a_L, a_R, s_L, s_R)
            mse = float(np.mean((y_corr - y_true) ** 2))
            if best is None or mse < best.oof_mse:
                best = FormBParams(
                    alpha_L=float(alpha_L), alpha_R=float(alpha_R),
                    a_L=a_L, a_R=a_R, s_L=s_L, s_R=s_R, oof_mse=mse)
    return best


def apply_form_b(y_pred: np.ndarray, params: FormBParams) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=float)
    return _form_b_apply_slopes(y_pred, params.a_L, params.a_R,
                                params.s_L, params.s_R)


# ---------------------------------------------------------------------------
# Paired bootstrap Delta-RMSE
# ---------------------------------------------------------------------------

def paired_bootstrap_delta_rmse(y_true: np.ndarray,
                                y_pred_pre: np.ndarray,
                                y_pred_post: np.ndarray,
                                n_boot: int = 2000,
                                seed: int = SEED_BOOTSTRAP,
                                ) -> tuple[float, float, float]:
    """Paired bootstrap for Delta-RMSE = pre_rmse - post_rmse.

    Positive Delta = correction helped. Returns (point, lo95, hi95).
    Each bootstrap draws one index array and scores both pre and post
    on the *same* resampled indices (paired), so the variance in their
    difference is what we estimate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pre = np.asarray(y_pred_pre, dtype=float)
    y_post = np.asarray(y_pred_post, dtype=float)
    n = len(y_true)
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    pre0 = _rmse(y_true, y_pre)
    post0 = _rmse(y_true, y_post)
    point = float(pre0 - post0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = (_rmse(y_true[idx], y_pre[idx])
                     - _rmse(y_true[idx], y_post[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return (point, float(lo), float(hi))


# ---------------------------------------------------------------------------
# Ship-if-better decision
# ---------------------------------------------------------------------------

@dataclass
class ShipDecision:
    """Outcome of applying one correction form to the test split."""
    form: str                        # 'A', 'B', or 'none'
    ships: bool
    overall_delta: float             # pre_rmse - post_rmse (positive=better)
    max_regime_degradation: float    # max over regimes of (post-pre); <=tol to ship
    reason: str                      # short human-readable reason

    def to_dict(self) -> dict:
        return {
            'form': self.form,
            'ships': bool(self.ships),
            'overall_delta': self.overall_delta,
            'max_regime_degradation': self.max_regime_degradation,
            'reason': self.reason,
        }


def ship_decision(form: str,
                  pre_overall: float, post_overall: float,
                  per_regime_pre: dict, per_regime_post: dict,
                  tol: float = SHIP_TOL) -> ShipDecision:
    """Apply the two-part ship rule:
       1. overall_delta > tol
       2. max_regime_degradation <= tol
    """
    overall_delta = float(pre_overall - post_overall)
    degradations = []
    for r, post_r in per_regime_post.items():
        pre_r = per_regime_pre.get(r, np.nan)
        if not (np.isfinite(pre_r) and np.isfinite(post_r)):
            continue
        degradations.append(float(post_r - pre_r))
    max_degr = max(degradations) if degradations else float('nan')

    if not np.isfinite(overall_delta):
        return ShipDecision(form=form, ships=False,
                            overall_delta=overall_delta,
                            max_regime_degradation=max_degr,
                            reason='non-finite overall RMSE')
    if overall_delta <= tol:
        return ShipDecision(form=form, ships=False,
                            overall_delta=overall_delta,
                            max_regime_degradation=max_degr,
                            reason=f'overall delta {overall_delta:+.4g} '
                                   f'<= tol {tol:g}')
    if np.isfinite(max_degr) and max_degr > tol:
        return ShipDecision(form=form, ships=False,
                            overall_delta=overall_delta,
                            max_regime_degradation=max_degr,
                            reason=f'regime worst degradation {max_degr:+.4g} '
                                   f'> tol {tol:g}')
    return ShipDecision(form=form, ships=True,
                        overall_delta=overall_delta,
                        max_regime_degradation=max_degr,
                        reason='ships: overall improves, no regime degrades')


def choose_winner(decision_a: ShipDecision,
                  decision_b: ShipDecision) -> str:
    """Return 'A', 'B', or 'none'. If both ship, pick larger overall_delta;
    ties break in favour of Form A (simpler, more interpretable)."""
    a_ships = decision_a.ships
    b_ships = decision_b.ships
    if not a_ships and not b_ships:
        return 'none'
    if a_ships and not b_ships:
        return 'A'
    if b_ships and not a_ships:
        return 'B'
    if decision_a.overall_delta >= decision_b.overall_delta:
        return 'A'
    return 'B'


# ---------------------------------------------------------------------------
# Per-regime + overall metrics with bootstrap CIs
# ---------------------------------------------------------------------------

def _regime_rmse_dict(y_true: np.ndarray, y_pred: np.ndarray,
                      regimes: np.ndarray,
                      min_n: int = 5) -> dict:
    """Return {regime: point_rmse} for each regime with >= min_n samples."""
    out: dict[str, float] = {}
    for r in np.unique(regimes):
        mask = (regimes == r)
        if mask.sum() < min_n:
            out[str(r)] = float('nan')
            continue
        out[str(r)] = _rmse(y_true[mask], y_pred[mask])
    return out


def evaluate_pre_post(y_true_te: np.ndarray,
                      y_pred_pre_te: np.ndarray,
                      y_pred_post_te: np.ndarray,
                      regimes_te: np.ndarray,
                      n_boot_ci: int = 2000,
                      bootstrap_seed: int = SEED_BOOTSTRAP,
                      ) -> list[dict]:
    """Compute per-regime + ALL pre/post RMSE with bootstrap 95% CIs and
    paired-bootstrap Delta-RMSE CIs. Returns a list of row-dicts ready
    for a DataFrame."""
    rows = []
    regs = [r for r in P_REGIME_LABELS if (regimes_te == r).any()]
    regs.append('ALL')
    for r in regs:
        if r == 'ALL':
            mask = np.ones(len(y_true_te), dtype=bool)
        else:
            mask = (regimes_te == r)
        n_r = int(mask.sum())
        if n_r < 5:
            rows.append(dict(regime=r, n=n_r,
                             pre_rmse=np.nan, pre_lo=np.nan, pre_hi=np.nan,
                             post_rmse=np.nan, post_lo=np.nan, post_hi=np.nan,
                             delta_rmse=np.nan, delta_lo=np.nan, delta_hi=np.nan))
            continue
        pre, pre_lo, pre_hi = _bootstrap_stat(
            y_true_te[mask], y_pred_pre_te[mask], _rmse,
            n_bootstrap=n_boot_ci, seed=bootstrap_seed)
        post, post_lo, post_hi = _bootstrap_stat(
            y_true_te[mask], y_pred_post_te[mask], _rmse,
            n_bootstrap=n_boot_ci, seed=bootstrap_seed)
        delta, delta_lo, delta_hi = paired_bootstrap_delta_rmse(
            y_true_te[mask], y_pred_pre_te[mask], y_pred_post_te[mask],
            n_boot=n_boot_ci, seed=bootstrap_seed)
        rows.append(dict(regime=r, n=n_r,
                         pre_rmse=pre, pre_lo=pre_lo, pre_hi=pre_hi,
                         post_rmse=post, post_lo=post_lo, post_hi=post_hi,
                         delta_rmse=delta, delta_lo=delta_lo, delta_hi=delta_hi))
    return rows


# ---------------------------------------------------------------------------
# End-to-end: one (cell, seed)
# ---------------------------------------------------------------------------

@dataclass
class BiasCorrectionResult:
    """All artefacts from fitting + evaluating one (cell, seed)."""
    pipeline: str
    track: str
    target: str
    feature_set: str
    model: str
    seed: int
    form_a_params: dict
    form_b_params: Optional[dict]     # None if no valid Form B fit
    pre_rows: list[dict]              # per-regime+ALL, pre only
    form_a_rows: list[dict]           # per-regime+ALL pre/post for Form A
    form_b_rows: list[dict]           # per-regime+ALL pre/post for Form B
    ship_a: dict
    ship_b: dict
    winner: str                       # 'A', 'B', or 'none'

    def to_summary_rows(self) -> list[dict]:
        """Flatten to a long-format list of dicts, one per
        (regime, form) combination, suitable for CSV writing."""
        out = []
        base = {
            'pipeline': self.pipeline, 'track': self.track,
            'target': self.target, 'feature_set': self.feature_set,
            'model': self.model, 'seed': self.seed,
            'winner': self.winner,
        }
        for form, rows in (('A', self.form_a_rows), ('B', self.form_b_rows)):
            for r in rows:
                out.append({**base, 'form': form, **r})
        return out


def run_bias_correction_for_cell(
    pipeline: str,
    track: str,
    target: str,
    feature_set: str,
    model_name: str,
    best_params: dict,
    X_tr: np.ndarray, y_tr: np.ndarray, groups_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    p_true_tr: np.ndarray,              # true P_kbar for regime on train
    p_true_te: np.ndarray,              # true P_kbar for regime on test
    seed: int,
    n_folds: int = 10,
    n_boot_ci: int = 2000,
    ship_tol: float = SHIP_TOL,
) -> BiasCorrectionResult:
    """Run the full Form-A / Form-B / ship-if-better pipeline for one
    (pipeline, track, target, feature_set, model, seed)."""
    regimes_tr = assign_p_regime(p_true_tr)
    regimes_te = assign_p_regime(p_true_te)

    # Step 1: OOF predictions on training set.
    oof_tr = oof_predict(model_name, best_params, X_tr, y_tr,
                         groups_tr, seed=seed, n_folds=n_folds)
    mask_fin = np.isfinite(oof_tr)

    # Step 2: fit both forms on OOF.
    params_a = fit_form_a(y_tr[mask_fin], oof_tr[mask_fin],
                          regimes_tr[mask_fin])
    params_b = fit_form_b(y_tr[mask_fin], oof_tr[mask_fin])

    # Step 3: fit final model on full train, predict test.
    est = build_model(model_name, best_params, seed=seed)
    est.fit(X_tr, y_tr)
    y_pred_te = est.predict(X_te)

    y_corr_a_te = apply_form_a(y_pred_te, regimes_te, params_a)
    if params_b is None:
        y_corr_b_te = y_pred_te.copy()  # identity fallback
    else:
        y_corr_b_te = apply_form_b(y_pred_te, params_b)

    # Step 4: evaluate pre vs post on test, per regime + ALL.
    form_a_rows = evaluate_pre_post(y_te, y_pred_te, y_corr_a_te,
                                    regimes_te, n_boot_ci=n_boot_ci,
                                    bootstrap_seed=SEED_BOOTSTRAP)
    form_b_rows = evaluate_pre_post(y_te, y_pred_te, y_corr_b_te,
                                    regimes_te, n_boot_ci=n_boot_ci,
                                    bootstrap_seed=SEED_BOOTSTRAP)

    # Step 5: ship-if-better decisions.
    pre_regime = {r['regime']: r['pre_rmse'] for r in form_a_rows
                  if r['regime'] != 'ALL'}
    post_regime_a = {r['regime']: r['post_rmse'] for r in form_a_rows
                     if r['regime'] != 'ALL'}
    post_regime_b = {r['regime']: r['post_rmse'] for r in form_b_rows
                     if r['regime'] != 'ALL'}
    pre_all = next(r['pre_rmse'] for r in form_a_rows if r['regime'] == 'ALL')
    post_all_a = next(r['post_rmse'] for r in form_a_rows if r['regime'] == 'ALL')
    post_all_b = next(r['post_rmse'] for r in form_b_rows if r['regime'] == 'ALL')

    ship_a = ship_decision('A', pre_all, post_all_a,
                           pre_regime, post_regime_a, tol=ship_tol)
    if params_b is None:
        ship_b = ShipDecision(form='B', ships=False,
                              overall_delta=0.0,
                              max_regime_degradation=0.0,
                              reason='no valid Form B fit on OOF')
    else:
        ship_b = ship_decision('B', pre_all, post_all_b,
                               pre_regime, post_regime_b, tol=ship_tol)
    winner = choose_winner(ship_a, ship_b)

    pre_rows = [dict(regime=r['regime'], n=r['n'],
                     pre_rmse=r['pre_rmse'],
                     pre_lo=r['pre_lo'], pre_hi=r['pre_hi'])
                for r in form_a_rows]

    return BiasCorrectionResult(
        pipeline=pipeline, track=track, target=target,
        feature_set=feature_set, model=model_name, seed=int(seed),
        form_a_params={r: {'a': a, 'b': b} for r, (a, b) in params_a.items()},
        form_b_params=(params_b.to_dict() if params_b is not None else None),
        pre_rows=pre_rows,
        form_a_rows=form_a_rows,
        form_b_rows=form_b_rows,
        ship_a=ship_a.to_dict(),
        ship_b=ship_b.to_dict(),
        winner=winner,
    )
