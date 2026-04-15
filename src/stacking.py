"""Ridge-regression stacking meta-model.

Combines 4 base learners (RF, ERT, XGB, GB) via RidgeCV on out-of-fold
predictions. Ground-truth target is the outer fold's y. See
`docs/stacking_strategy.md` for design.

Public API:
- `generate_oof_predictions(model_ctor, X, y, groups, cv_splitter, seed)`
- `fit_ridge_meta_model(oof_matrix, y_train, alphas=None, cv=5)`
- `stacking_predict(meta_model, base_predictions_dict,
                    base_order=('RF','ERT','XGB','GB'))`
- `compute_oof_correlation_matrix(oof_dict, base_order=...)`

The callable returned by `load_stacked_model` in src/data.py uses these
primitives end to end. Callers in NB03 Phase 3.10 call them directly so
the logging / sanity checks stay in the notebook.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

BASE_ORDER = ('RF', 'ERT', 'XGB', 'GB')
DEFAULT_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def generate_oof_predictions(model_ctor: Callable,
                             X: np.ndarray,
                             y: np.ndarray,
                             groups: np.ndarray,
                             cv_splitter,
                             seed: int = 42) -> np.ndarray:
    """Run one base model through a CV splitter and return length-N OOF vector.

    `model_ctor(seed)` must return a fresh estimator. Predictions are
    written at val indices; the final array is aligned with `y`.
    """
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    for tr, va in cv_splitter.split(X, y, groups):
        est = model_ctor(seed)
        est.fit(X[tr], y[tr])
        oof[va] = est.predict(X[va])
    if np.isnan(oof).any():
        # CV splitter should have visited every index; surface otherwise.
        n_missing = int(np.isnan(oof).sum())
        raise RuntimeError(
            f'generate_oof_predictions: {n_missing} rows never served as validation.'
        )
    return oof


def fit_ridge_meta_model(oof_matrix: np.ndarray,
                         y_train: np.ndarray,
                         alphas: Iterable[float] | None = None,
                         cv: int = 5) -> RidgeCV:
    """Fit RidgeCV on the (N, n_bases) OOF matrix against y_train."""
    a = tuple(alphas) if alphas is not None else DEFAULT_ALPHAS
    meta = RidgeCV(alphas=a, cv=cv)
    meta.fit(oof_matrix, y_train)
    return meta


def stacking_predict(meta_model: RidgeCV,
                     base_predictions: Mapping[str, np.ndarray],
                     base_order: Sequence[str] = BASE_ORDER) -> np.ndarray:
    """Predict from a fitted RidgeCV and a dict of base-model prediction arrays."""
    cols = [np.asarray(base_predictions[k], dtype=float) for k in base_order]
    X_meta = np.column_stack(cols)
    return meta_model.predict(X_meta)


def compute_oof_correlation_matrix(oof_dict: Mapping[str, np.ndarray],
                                   base_order: Sequence[str] = BASE_ORDER) -> pd.DataFrame:
    """Pearson correlation matrix of OOF vectors, ordered by `base_order`."""
    df = pd.DataFrame({k: np.asarray(oof_dict[k], dtype=float) for k in base_order})
    return df.corr(method='pearson')
