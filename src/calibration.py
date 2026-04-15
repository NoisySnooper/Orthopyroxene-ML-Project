"""Split conformal prediction utilities.

Implements the classical split-conformal regression procedure:
fit a base regressor on a training split; compute absolute residuals on
a held-out calibration split; take the (1 - alpha) empirical quantile
of those residuals with a finite-sample correction; symmetric intervals
of width q_hat around the test-set predictions cover the true target
with at least 1 - alpha probability under exchangeability.

References
----------
Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
Lei et al. (2018). Distribution-free predictive inference for regression.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.10) -> float:
    """Finite-sample corrected (1 - alpha) quantile of |residuals|.

    Parameters
    ----------
    residuals : array-like
        Signed residuals y_cal - y_pred on the calibration split.
    alpha : float
        Miscoverage rate. Target coverage is 1 - alpha.

    Returns
    -------
    q_hat : float
        Half-width of the symmetric prediction interval.
    """
    r = np.abs(np.asarray(residuals, dtype=float))
    n = r.size
    if n == 0:
        raise ValueError("residuals must be non-empty")
    # Finite-sample correction: use ceil((n+1) * (1-alpha)) / n quantile.
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(r, level, method="higher"))


def conformal_intervals(
    predictions: np.ndarray, q_hat: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric prediction intervals of half-width q_hat."""
    p = np.asarray(predictions, dtype=float)
    return p - q_hat, p + q_hat


def compute_coverage(
    y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    """Empirical coverage: fraction of y_true that lies in [lo, hi]."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))
