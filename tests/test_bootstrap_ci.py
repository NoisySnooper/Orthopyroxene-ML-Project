"""Bootstrap RMSE CI tests (Phase 1, Lee revisions).

Pins:
  1. bootstrap_rmse_ci returns positive-width CI for a deterministic fit
     with nonzero residuals (patches the ElasticNet zero-std artifact
     documented in results/seed_variance_patch_log.md).
  2. bootstrap_rmse_ci is deterministic given a fixed seed.
  3. results/bootstrap_rmse_cis_all_cells.csv exists, covers all 8
     (track, target) cells, contains 25 rows per cell, and has no zero-
     width CIs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.evaluation import bootstrap_rmse_ci  # noqa: E402

BOOT_CSV = ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'


def test_bootstrap_ci_nontrivial_for_deterministic_fit():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 10, size=200)
    y_pred = y_true + rng.normal(0, 2, size=200)
    rmse, lo, hi = bootstrap_rmse_ci(y_true, y_pred, n_boot=200,
                                     random_state=42)
    assert np.isfinite(rmse) and rmse > 0
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi > lo
    assert lo <= rmse <= hi
    assert (hi - lo) > 1e-3


def test_bootstrap_ci_deterministic_given_seed():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 10, size=200)
    y_pred = y_true + rng.normal(0, 2, size=200)
    r1 = bootstrap_rmse_ci(y_true, y_pred, n_boot=200, random_state=42)
    r2 = bootstrap_rmse_ci(y_true, y_pred, n_boot=200, random_state=42)
    assert r1 == r2
    r3 = bootstrap_rmse_ci(y_true, y_pred, n_boot=200, random_state=7)
    assert r1 != r3


def test_bootstrap_rmse_cis_all_cells_exists_and_complete():
    assert BOOT_CSV.exists(), f'missing {BOOT_CSV}'
    df = pd.read_csv(BOOT_CSV)
    required = {'pipeline', 'track', 'target', 'model', 'feature_set',
                'rmse_point', 'rmse_ci_lo', 'rmse_ci_hi'}
    assert required.issubset(df.columns)
    cells = df.groupby(['track', 'target']).size()
    assert len(cells) == 8, f'expected 8 (track,target) cells, got {len(cells)}'
    assert (cells == 25).all(), f'expected 25 rows/cell, got\n{cells}'
    widths = df['rmse_ci_hi'] - df['rmse_ci_lo']
    assert (widths > 0).all(), 'zero-width CI detected'
    assert (df['rmse_ci_lo'] <= df['rmse_point']).all()
    assert (df['rmse_point'] <= df['rmse_ci_hi']).all()
