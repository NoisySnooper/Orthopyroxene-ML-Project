"""Unit tests for pre-registered P-regime helpers in src/evaluation.py.

Registration: docs/v10_p_regime_preregistration.md (locked 2026-04-17).
These tests pin:
  1. Exact boundary behavior at P = 0, 5, 15, 30, 100 kbar (right-open).
  2. Out-of-range handling (negatives clipped, > ceiling clamped).
  3. NaN handling (returns 'unassigned').
  4. n < 20 sample-size-limited flag per regime.
  5. Bootstrap CI reproducibility (fixed seed).
  6. End-to-end per_regime_benchmark returns expected long-form schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    P_REGIME_BIN_EDGES_KBAR,
    P_REGIME_LABELS,
    P_REGIME_MIN_N_FOR_CLAIMS,
)
from src.evaluation import (  # noqa: E402
    assign_p_regime,
    compute_per_regime_metrics,
    per_regime_benchmark,
)


# --- 1. Boundary behavior ---------------------------------------------------

def test_bin_edges_locked():
    assert P_REGIME_BIN_EDGES_KBAR == [0.0, 5.0, 15.0, 30.0, 100.0]
    assert P_REGIME_LABELS == [
        'shallow_crustal', 'deep_crustal_MASH',
        'lithospheric_mantle', 'deeper_mantle',
    ]


def test_right_open_convention():
    """P = 5.0 belongs to deep_crustal_MASH, not shallow_crustal."""
    p = np.array([0.0, 4.999, 5.0, 14.999, 15.0, 29.999, 30.0, 99.999])
    out = assign_p_regime(p)
    expected = [
        'shallow_crustal',     # 0.0
        'shallow_crustal',     # 4.999
        'deep_crustal_MASH',   # 5.0 (right-open: 5 is in next bin)
        'deep_crustal_MASH',   # 14.999
        'lithospheric_mantle', # 15.0
        'lithospheric_mantle', # 29.999
        'deeper_mantle',       # 30.0
        'deeper_mantle',       # 99.999
    ]
    assert list(out) == expected


def test_negative_and_ceiling_handling():
    p = np.array([-1.0, -0.5, 100.0, 150.0])
    out = assign_p_regime(p)
    # Negatives clip to 0 -> shallow; >= ceiling land in deeper_mantle.
    assert out[0] == 'shallow_crustal'
    assert out[1] == 'shallow_crustal'
    assert out[2] == 'deeper_mantle'
    assert out[3] == 'deeper_mantle'


def test_nan_becomes_unassigned():
    p = np.array([np.nan, 2.0, np.nan, 20.0])
    out = assign_p_regime(p)
    assert out[0] == 'unassigned'
    assert out[2] == 'unassigned'
    assert out[1] == 'shallow_crustal'
    assert out[3] == 'lithospheric_mantle'


# --- 2. Empty + sample-size-limited ----------------------------------------

def test_empty_bin_marked_limited():
    """A regime with zero samples returns n=0 and sample_size_limited=True."""
    rng = np.random.default_rng(0)
    # All samples in shallow_crustal only.
    p = rng.uniform(0, 5, size=50)
    y_true = rng.normal(1000, 50, size=50)
    y_pred = y_true + rng.normal(0, 10, size=50)

    df = compute_per_regime_metrics(y_true, y_pred, p, metric='rmse')
    for label in P_REGIME_LABELS:
        row = df[df.regime == label].iloc[0]
        if label == 'shallow_crustal':
            assert row.n == 50
            assert not row.sample_size_limited
            assert np.isfinite(row.metric_value)
        else:
            assert row.n == 0
            assert row.sample_size_limited
            assert not np.isfinite(row.metric_value)


def test_small_bin_flagged_limited():
    """A regime with n below P_REGIME_MIN_N_FOR_CLAIMS is flagged."""
    assert P_REGIME_MIN_N_FOR_CLAIMS == 20
    n_small = 5
    rng = np.random.default_rng(1)
    p = np.concatenate([
        rng.uniform(0, 5, 30),      # shallow_crustal: 30
        rng.uniform(20, 25, n_small), # lithospheric_mantle: 5 (limited)
    ])
    y_true = rng.normal(1000, 50, size=len(p))
    y_pred = y_true + rng.normal(0, 10, size=len(p))

    df = compute_per_regime_metrics(y_true, y_pred, p, metric='rmse')
    shallow = df[df.regime == 'shallow_crustal'].iloc[0]
    litho = df[df.regime == 'lithospheric_mantle'].iloc[0]
    assert shallow.n == 30 and not shallow.sample_size_limited
    assert litho.n == n_small and litho.sample_size_limited


# --- 3. Bootstrap reproducibility ------------------------------------------

def test_bootstrap_ci_reproducible_with_seed():
    rng = np.random.default_rng(2)
    n = 100
    p = rng.uniform(0, 30, size=n)
    y_true = rng.normal(1000, 50, size=n)
    y_pred = y_true + rng.normal(0, 15, size=n)

    a = compute_per_regime_metrics(y_true, y_pred, p, metric='rmse',
                                    n_bootstrap=200, seed=123)
    b = compute_per_regime_metrics(y_true, y_pred, p, metric='rmse',
                                    n_bootstrap=200, seed=123)
    pd.testing.assert_frame_equal(a, b)

    c = compute_per_regime_metrics(y_true, y_pred, p, metric='rmse',
                                    n_bootstrap=200, seed=999)
    # Different seed should change CI bounds (point estimate unchanged).
    non_empty = a.n > 0
    assert np.allclose(a.metric_value[non_empty], c.metric_value[non_empty])
    # At least one CI bound differs for seeds 123 vs 999.
    any_differ = False
    for col in ('metric_ci_low', 'metric_ci_high'):
        for ai, ci in zip(a[col][non_empty], c[col][non_empty]):
            if np.isfinite(ai) and np.isfinite(ci) and not np.isclose(ai, ci):
                any_differ = True; break
    assert any_differ


def test_rmse_point_estimate_matches_manual():
    rng = np.random.default_rng(3)
    n = 60
    p = rng.uniform(0, 5, size=n)  # all in shallow_crustal
    y_true = rng.normal(1000, 50, size=n)
    y_pred = y_true + rng.normal(0, 12, size=n)
    expected = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    df = compute_per_regime_metrics(y_true, y_pred, p, metric='rmse',
                                     n_bootstrap=50, seed=0)
    got = df[df.regime == 'shallow_crustal'].iloc[0].metric_value
    assert np.isclose(got, expected)


# --- 4. Benchmark long-format schema ---------------------------------------

def test_per_regime_benchmark_schema():
    rng = np.random.default_rng(4)
    n = 80
    p = rng.uniform(0, 30, size=n)
    y_true = rng.normal(1000, 50, size=n)
    preds = {
        'method_A': {'y_pred': y_true + rng.normal(0, 8, size=n)},
        'method_B': {'y_pred': y_true + rng.normal(3, 8, size=n)},
    }
    df = per_regime_benchmark(preds, y_true, p,
                               metrics=('rmse', 't_bias'),
                               n_bootstrap=50)
    assert set(df.columns) >= {
        'method', 'regime', 'n', 'metric',
        'metric_value', 'metric_ci_low', 'metric_ci_high',
        'sample_size_limited',
    }
    assert set(df.method.unique()) == {'method_A', 'method_B'}
    assert set(df.metric.unique()) == {'rmse', 't_bias'}
    # 4 regimes x 2 metrics x 2 methods = 16 rows.
    assert len(df) == 16


def test_coverage_metric_requires_bounds():
    rng = np.random.default_rng(5)
    n = 40
    p = rng.uniform(0, 5, n)
    y_true = rng.normal(1000, 50, n)
    preds = {
        'm_with_ci': {
            'y_pred':    y_true + rng.normal(0, 10, n),
            'y_pred_lo': y_true - 20,
            'y_pred_hi': y_true + 20,
        },
        'm_no_ci': {'y_pred': y_true + rng.normal(0, 10, n)},
    }
    df = per_regime_benchmark(preds, y_true, p,
                               metrics=('rmse', 'coverage_90'),
                               n_bootstrap=40)
    # coverage_90 only reported for the method with prediction-interval bounds.
    cov = df[df.metric == 'coverage_90']
    assert set(cov.method.unique()) == {'m_with_ci'}
    rmse = df[df.metric == 'rmse']
    assert set(rmse.method.unique()) == {'m_with_ci', 'm_no_ci'}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
