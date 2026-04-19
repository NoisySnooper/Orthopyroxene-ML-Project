"""v10 ensemble strategies beyond Ridge stacking.

Four ensemble methods compete in v10 Phase C (see docs/master_plan.md
Section 3):

1. Ridge stacking -- existing, via src/stacking.py (kept as baseline).
2. Two-level stacking -- level-0 OOF preds from 8 bases, level-1 preds
   from {Ridge, ElasticNet, HistGB} meta-learners fitted on OOF, level-2
   Ridge blend across level-1 OOF preds.
3. Greedy selection with replacement (Caruana 2004) -- iteratively add
   bases to a simple average; final weights = count / total.
4. AutoGluon tabular stacker -- handled in a separate driver since it
   owns its own tuning loop.

All ensembles consume a (N, K) OOF matrix produced upstream by
src/stacking.generate_oof_predictions over K base models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import GroupKFold

DEFAULT_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


# ---------------------------------------------------------------------------
# (2) Two-level stacking
# ---------------------------------------------------------------------------

@dataclass
class TwoLevelStack:
    """Level-1 meta-learners (fit on OOF) and a level-2 blender.

    Attributes
    ----------
    level1_models : dict[str, estimator]
        Meta-learners fit on level-0 OOF matrix (X_meta) vs y_train.
    blender : RidgeCV
        Level-2 blender fit on level-1 OOF predictions vs y_train.
    base_order : tuple[str, ...]
        Order of level-0 base columns in the OOF matrix.
    level1_order : tuple[str, ...]
        Order of level-1 meta-learner columns in the blender input.
    """
    level1_models: dict
    blender: RidgeCV
    base_order: tuple
    level1_order: tuple

    def predict(self, base_predictions: Mapping[str, np.ndarray]) -> np.ndarray:
        X_meta = np.column_stack([
            np.asarray(base_predictions[k], dtype=float) for k in self.base_order
        ])
        level1_preds = np.column_stack([
            self.level1_models[k].predict(X_meta) for k in self.level1_order
        ])
        return self.blender.predict(level1_preds)


def fit_two_level_stack(oof_matrix: np.ndarray,
                        y_train: np.ndarray,
                        groups: np.ndarray,
                        base_order: Sequence[str],
                        cv_splitter=None,
                        seed: int = 42) -> TwoLevelStack:
    """Fit 3 level-1 meta-learners on OOF, then a Ridge level-2 blender.

    Level-1 OOF is regenerated via `cv_splitter` (default: GroupKFold=3)
    so the blender is trained on genuinely out-of-sample level-1 preds.
    """
    oof_matrix = np.asarray(oof_matrix)
    y_train = np.asarray(y_train)
    groups = np.asarray(groups)
    cv = cv_splitter or GroupKFold(n_splits=3)

    level1_ctors = {
        'ridge':  lambda: RidgeCV(alphas=DEFAULT_ALPHAS, cv=5),
        'enet':   lambda: ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5,
                                        max_iter=10000, random_state=seed),
        'hgb':    lambda: HistGradientBoostingRegressor(
                              random_state=seed, max_iter=200, max_depth=3,
                              learning_rate=0.05, early_stopping=False),
    }
    level1_order = ('ridge', 'enet', 'hgb')

    # Level-1 OOF via inner CV.
    n = len(y_train)
    level1_oof = np.full((n, len(level1_order)), np.nan, dtype=float)
    for tr, va in cv.split(oof_matrix, y_train, groups):
        for j, name in enumerate(level1_order):
            m = level1_ctors[name]()
            m.fit(oof_matrix[tr], y_train[tr])
            level1_oof[va, j] = m.predict(oof_matrix[va])
    if np.isnan(level1_oof).any():
        raise RuntimeError('two-level stack: level-1 OOF has missing rows.')

    # Refit level-1 on full OOF matrix for inference-time predictions.
    level1_models = {}
    for name in level1_order:
        m = level1_ctors[name]()
        m.fit(oof_matrix, y_train)
        level1_models[name] = m

    # Level-2 blender on level-1 OOF.
    blender = RidgeCV(alphas=DEFAULT_ALPHAS, cv=5)
    blender.fit(level1_oof, y_train)

    return TwoLevelStack(
        level1_models=level1_models,
        blender=blender,
        base_order=tuple(base_order),
        level1_order=level1_order,
    )


# ---------------------------------------------------------------------------
# (3) Greedy selection with replacement (Caruana 2004)
# ---------------------------------------------------------------------------

@dataclass
class GreedyEnsemble:
    """Weighted average of base models; weights produced by greedy search.

    Attributes
    ----------
    weights : np.ndarray, shape (K,)
        Non-negative weights summing to 1.
    base_order : tuple[str, ...]
        Column order matching `weights`.
    trace : list[str]
        Names of bases picked at each iteration (for audit).
    val_rmse : float
        Final RMSE on the training OOF used for selection.
    """
    weights: np.ndarray
    base_order: tuple
    trace: list
    val_rmse: float

    def predict(self, base_predictions: Mapping[str, np.ndarray]) -> np.ndarray:
        cols = np.column_stack([
            np.asarray(base_predictions[k], dtype=float) for k in self.base_order
        ])
        return cols @ self.weights


def fit_greedy_ensemble(oof_matrix: np.ndarray,
                        y_train: np.ndarray,
                        base_order: Sequence[str],
                        n_iters: int = 50) -> GreedyEnsemble:
    """Caruana 2004 greedy ensemble selection with replacement.

    Starts from the best single base; at each iter picks the base that,
    when averaged into the running prediction, most reduces RMSE. Runs
    with replacement, so weights are counts / n_iters.
    """
    P = np.asarray(oof_matrix, dtype=float)   # (N, K)
    y = np.asarray(y_train, dtype=float)
    n, K = P.shape
    if K != len(base_order):
        raise ValueError('base_order length must equal OOF columns.')

    # Seed with the single-best base.
    per_base_rmse = np.sqrt(((P - y[:, None]) ** 2).mean(axis=0))
    first = int(np.argmin(per_base_rmse))

    counts = np.zeros(K, dtype=int)
    counts[first] += 1
    trace = [base_order[first]]
    running_sum = P[:, first].copy()

    for _ in range(1, n_iters):
        m = counts.sum()
        # Candidate predictions: (running_sum + P[:, k]) / (m + 1)
        cand = (running_sum[:, None] + P) / (m + 1)
        rmse = np.sqrt(((cand - y[:, None]) ** 2).mean(axis=0))
        k = int(np.argmin(rmse))
        counts[k] += 1
        trace.append(base_order[k])
        running_sum = running_sum + P[:, k]

    weights = counts / counts.sum()
    final_pred = P @ weights
    val_rmse = float(np.sqrt(((final_pred - y) ** 2).mean()))

    return GreedyEnsemble(
        weights=weights,
        base_order=tuple(base_order),
        trace=trace,
        val_rmse=val_rmse,
    )
