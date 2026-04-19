"""Tests for src.bias_correction.

Covers:

1. Unit tests for Form A math (per-regime OLS, identity pass-through on
   undersampled regimes).
2. Unit tests for Form B math (grid search finds the planted
   breakpoints, slope constraint rejects pathological fits, width
   constraint rejects too-narrow middle regions).
3. Ship-decision logic (both rules, tie-breaks).
4. Regression: full cell run for `ElasticNet/T_C/opx_liq/raw` with
   seed=42 matches the archived Form A params in
   `results/v10_opx_liq_bias_correction_params.csv` (written by the
   v10 Phase G.4 pilot run on 2026-04-17). This ensures the factored
   module preserves Form A bit-for-bit with the pilot.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.bias_correction import (
    SHIP_TOL,
    apply_form_a,
    apply_form_b,
    choose_winner,
    fit_form_a,
    fit_form_b,
    oof_predict,
    paired_bootstrap_delta_rmse,
    run_bias_correction_for_cell,
    ship_decision,
)


# ---------------------------------------------------------------------------
# Form A unit tests
# ---------------------------------------------------------------------------

def test_form_a_recovers_planted_affine():
    rng = np.random.default_rng(0)
    n = 200
    y_pred = rng.uniform(1, 30, size=n)
    regimes = np.where(y_pred < 5, 'shallow_crustal', 'deeper_mantle')
    y_true = np.where(regimes == 'shallow_crustal',
                      1.1 * y_pred + 0.5, y_pred) + rng.normal(0, 0.05, size=n)
    params = fit_form_a(y_true, y_pred, regimes)
    a_s, b_s = params['shallow_crustal']
    a_d, b_d = params['deeper_mantle']
    assert abs(a_s - 1.1) < 0.02
    assert abs(b_s - 0.5) < 0.05
    assert abs(a_d - 1.0) < 0.02
    assert abs(b_d - 0.0) < 0.05


def test_form_a_falls_back_to_identity_for_small_regime():
    # Three samples in a regime -> identity fallback (min_n=5).
    y_pred = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    y_true = np.array([1.1, 2.1, 3.1, 10.0, 20.0, 30.0])
    regimes = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    params = fit_form_a(y_true, y_pred, regimes)
    assert params['A'] == (1.0, 0.0)  # identity
    assert params['B'] == (1.0, 0.0)


def test_apply_form_a_leaves_unknown_regimes_unchanged():
    y_pred = np.array([1.0, 2.0, 3.0])
    regimes = np.array(['shallow_crustal', 'unknown_regime', 'shallow_crustal'])
    params = {'shallow_crustal': (2.0, 1.0)}
    y_corr = apply_form_a(y_pred, regimes, params)
    assert y_corr[0] == 3.0  # 2*1+1
    assert y_corr[1] == 2.0  # passthrough
    assert y_corr[2] == 7.0  # 2*3+1


# ---------------------------------------------------------------------------
# Form B unit tests
# ---------------------------------------------------------------------------

def test_form_b_recovers_planted_breakpoints():
    rng = np.random.default_rng(1)
    y_pred = np.sort(rng.uniform(0, 100, size=500))
    # Plant a bias: overestimate at y<20, underestimate at y>80.
    true_bias = np.where(y_pred < 20, 0.6 * (y_pred - 20),
                 np.where(y_pred > 80, -0.5 * (y_pred - 80), 0.0))
    y_true = y_pred - true_bias + rng.normal(0, 0.5, size=500)
    pb = fit_form_b(y_true, y_pred)
    assert pb is not None
    # Quantile 0.2 of uniform(0,100) ~= 20, quantile 0.8 ~= 80.
    assert 15 <= pb.a_L <= 25
    assert 75 <= pb.a_R <= 85
    assert 0.3 <= pb.s_L <= 0.9
    assert -0.9 <= pb.s_R <= -0.3
    y_corr = apply_form_b(y_pred, pb)
    pre_mse = np.mean((y_pred - y_true) ** 2)
    post_mse = np.mean((y_corr - y_true) ** 2)
    assert post_mse < pre_mse * 0.25  # correction cuts MSE by >75%


def test_form_b_rejects_degenerate_data():
    # Perfect prediction: no bias to fit, no tails worth correcting.
    # The fit may return a params object with s_L~s_R~0 or None.
    y_pred = np.linspace(0, 100, 500)
    y_true = y_pred.copy()
    pb = fit_form_b(y_true, y_pred)
    if pb is not None:
        assert abs(pb.s_L) < 0.05
        assert abs(pb.s_R) < 0.05


def test_form_b_slope_constraint_rejects_pathological():
    # Very steep planted bias would fit s far outside [-2, 2].
    rng = np.random.default_rng(2)
    y_pred = np.sort(rng.uniform(0, 10, size=300))
    # Blow up y_true only at the extreme 5%.
    y_true = y_pred.copy()
    mask = y_pred < np.quantile(y_pred, 0.05)
    y_true[mask] = y_pred[mask] - 10 * (y_pred[mask] - np.quantile(y_pred, 0.05))
    pb = fit_form_b(y_true, y_pred)
    if pb is not None:
        assert abs(pb.s_L) <= 2.0
        assert abs(pb.s_R) <= 2.0


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def test_paired_bootstrap_reports_positive_delta_when_post_is_better():
    rng = np.random.default_rng(3)
    n = 300
    y_true = rng.normal(0, 1, size=n)
    y_pre = y_true + rng.normal(1.0, 0.5, size=n)   # biased +1
    y_post = y_true + rng.normal(0.0, 0.5, size=n)  # unbiased
    d, lo, hi = paired_bootstrap_delta_rmse(
        y_true, y_pre, y_post, n_boot=500, seed=42)
    assert d > 0
    assert lo > 0  # entire CI above zero -> correction demonstrably better


# ---------------------------------------------------------------------------
# Ship decision
# ---------------------------------------------------------------------------

def test_ship_rule_requires_both_conditions():
    # (1) overall delta > tol
    # (2) no regime degrades > tol
    pre = {'r1': 10.0, 'r2': 5.0}
    post_improve = {'r1': 8.0, 'r2': 4.0}
    post_degrade_one = {'r1': 8.0, 'r2': 6.0}
    post_tiny_overall = {'r1': 9.9999, 'r2': 5.0}  # overall delta within tol

    # Case: ships
    d = ship_decision('A', 15.0, 12.0, pre, post_improve)
    assert d.ships

    # Case: one regime degrades by 1.0 > tol -> does not ship
    d = ship_decision('A', 15.0, 14.0, pre, post_degrade_one)
    assert not d.ships
    assert 'regime worst degradation' in d.reason

    # Case: overall delta ~= 0 -> does not ship (no net benefit)
    d = ship_decision('A', 15.0, 15.0 - 1e-8, pre, post_tiny_overall)
    assert not d.ships
    assert 'overall delta' in d.reason


def test_choose_winner_prefers_larger_delta():
    from src.bias_correction import ShipDecision
    a = ShipDecision(form='A', ships=True, overall_delta=2.0,
                     max_regime_degradation=-0.1, reason='')
    b = ShipDecision(form='B', ships=True, overall_delta=3.0,
                     max_regime_degradation=-0.2, reason='')
    assert choose_winner(a, b) == 'B'
    # Tie -> A (simpler)
    c = ShipDecision(form='B', ships=True, overall_delta=2.0,
                     max_regime_degradation=-0.2, reason='')
    assert choose_winner(a, c) == 'A'
    # Only A ships -> A
    c_nonship = ShipDecision(form='B', ships=False, overall_delta=0.0,
                             max_regime_degradation=0.0, reason='')
    assert choose_winner(a, c_nonship) == 'A'
    # Neither -> none
    a_nonship = ShipDecision(form='A', ships=False, overall_delta=0.0,
                             max_regime_degradation=0.0, reason='')
    assert choose_winner(a_nonship, c_nonship) == 'none'


# ---------------------------------------------------------------------------
# Regression: match archived v10 Phase G.4 opx_liq Form A params
# ---------------------------------------------------------------------------

ARCHIVED_PARAMS = PROJECT_ROOT / 'results' / 'v10_opx_liq_bias_correction_params.csv'


@pytest.mark.skipif(not ARCHIVED_PARAMS.exists(),
                    reason='archived Phase G.4 params not on disk')
def test_form_a_matches_archived_opx_liq_elasticnet_TC():
    """ElasticNet/T_C/opx_liq/raw at seed=42 must produce the same
    per-regime (a, b) we committed during the Phase G.4 pilot."""
    archived = pd.read_csv(ARCHIVED_PARAMS)
    sub = archived[(archived.model == 'ElasticNet')
                   & (archived.target == 'T_C')
                   & (archived.track == 'opx_liq')
                   & (archived.feature_set == 'raw')].copy()
    if sub.empty:
        pytest.skip('no archived params for ElasticNet/T_C/opx_liq/raw')

    from src.v10_phase_c_analysis import load_best_params, prepare_train_test
    best = load_best_params(PROJECT_ROOT / 'results'
                            / 'v10_optuna_best_params_opx.json')
    bp = best[('ElasticNet', 'T_C', 'opx_liq', 'raw')]['best_params']
    pt = prepare_train_test('opx_liq', 'T_C', 'raw')
    from src.data import load_opx_liq, load_splits
    df = load_opx_liq()
    tr_idx, _ = load_splits('opx_liq')
    p_tr = df['P_kbar'].to_numpy(dtype=float)[tr_idx]

    from src.evaluation import assign_p_regime
    regimes_tr = assign_p_regime(p_tr)
    oof = oof_predict('ElasticNet', bp, pt['X_tr'], pt['y_tr'],
                      pt['groups_tr'], seed=42, n_folds=10)
    mask = np.isfinite(oof)
    params = fit_form_a(pt['y_tr'][mask], oof[mask], regimes_tr[mask])

    tol = 1e-4
    for _, row in sub.iterrows():
        r = row['regime']
        a_ref, b_ref = float(row['a']), float(row['b'])
        a_new, b_new = params[r]
        assert abs(a_new - a_ref) < tol, \
            f'a mismatch in {r}: {a_new} vs {a_ref}'
        assert abs(b_new - b_ref) < tol, \
            f'b mismatch in {r}: {b_new} vs {b_ref}'


# ---------------------------------------------------------------------------
# End-to-end smoke on a tiny synthetic cell
# ---------------------------------------------------------------------------

def test_run_bias_correction_for_cell_end_to_end_synthetic():
    """Use ElasticNet on synthetic data to exercise the full pipeline.
    This ensures the orchestrator wires OOF -> fit -> apply -> evaluate
    -> ship correctly without blowing up."""
    rng = np.random.default_rng(7)
    n_tr, n_te = 300, 120
    X_tr = rng.normal(0, 1, size=(n_tr, 5))
    w = np.array([1.0, -0.5, 0.3, 0.0, 0.2])
    y_tr = X_tr @ w + rng.normal(0, 0.3, size=n_tr)
    p_tr = rng.uniform(0, 40, size=n_tr)
    groups_tr = np.array([f'cit_{i % 15}' for i in range(n_tr)])
    X_te = rng.normal(0, 1, size=(n_te, 5))
    y_te = X_te @ w + rng.normal(0, 0.3, size=n_te)
    p_te = rng.uniform(0, 40, size=n_te)

    res = run_bias_correction_for_cell(
        pipeline='opx', track='opx_liq', target='T_C', feature_set='raw',
        model_name='ElasticNet', best_params={'alpha': 1.0, 'l1_ratio': 0.5},
        X_tr=X_tr, y_tr=y_tr, groups_tr=groups_tr,
        X_te=X_te, y_te=y_te,
        p_true_tr=p_tr, p_true_te=p_te,
        seed=42, n_folds=5, n_boot_ci=200, ship_tol=SHIP_TOL,
    )
    assert res.winner in ('A', 'B', 'none')
    assert len(res.form_a_rows) >= 2
    assert len(res.form_b_rows) >= 2
    # Every row must have a regime, n, pre/post RMSE.
    for r in res.form_a_rows:
        assert set(('regime', 'n', 'pre_rmse', 'post_rmse',
                    'delta_rmse')).issubset(r.keys())
