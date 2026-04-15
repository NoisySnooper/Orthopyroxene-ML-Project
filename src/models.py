"""Model factories and prediction helpers.

Four base learners: random forest, extremely randomized trees, XGBoost,
and HistGradientBoosting. All share the same seed from config.py. For
tree ensembles with accessible per-tree predictions, `predict_median`
returns the median across the forest (Jorgenson 2022). `predict_iqr`
additionally returns the 16th and 84th percentile across trees, giving
an IQR uncertainty band that approximates 68% nominal coverage.
"""
from __future__ import annotations

import ast
import json

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor

from config import SEED_MODEL

SEARCH_NJOBS = -1
ESTIMATOR_NJOBS = 1


BASE_MODELS = {
    'RF':  lambda: RandomForestRegressor(random_state=SEED_MODEL, n_jobs=ESTIMATOR_NJOBS),
    'ERT': lambda: ExtraTreesRegressor(random_state=SEED_MODEL, n_jobs=ESTIMATOR_NJOBS),
    'XGB': lambda: XGBRegressor(random_state=SEED_MODEL, n_jobs=ESTIMATOR_NJOBS,
                                verbosity=0, tree_method='hist'),
    'GB':  lambda: HistGradientBoostingRegressor(random_state=SEED_MODEL,
                                                 early_stopping=True,
                                                 validation_fraction=0.15,
                                                 n_iter_no_change=20),
}


PARAM_GRIDS = {
    'RF': {
        'n_estimators': [200, 500, 800],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [0.33, 0.5, 0.66, 'sqrt'],
    },
    'ERT': {
        'n_estimators': [200, 500, 800],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [0.33, 0.5, 0.66, 'sqrt'],
    },
    'XGB': {
        'n_estimators': [200, 500, 800],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [1, 5, 10],
    },
    'GB': {
        'max_iter': [200, 500, 800],
        'max_depth': [3, 5, 7, None],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_leaf': [10, 20, 40],
        'l2_regularization': [0.0, 0.1, 1.0],
        'max_leaf_nodes': [15, 31, 63],
    },
}


MODEL_CLASSES = {
    'RF':  RandomForestRegressor,
    'ERT': ExtraTreesRegressor,
    'XGB': XGBRegressor,
    'GB':  HistGradientBoostingRegressor,
}


def parse_params(s):
    """Parse a best_params payload stored as JSON, Python literal, or dict."""
    if isinstance(s, dict):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return {}


def build_model(model_name, params, seed=SEED_MODEL):
    """Construct a fresh estimator with the given hyperparameters and seed.

    Mirrors BASE_MODELS sklearn/xgboost constructor signatures so that
    `seed` is always honored and njobs flags remain consistent.
    """
    p = dict(params)
    if model_name == 'GB':
        return HistGradientBoostingRegressor(**p, random_state=seed)
    if model_name == 'XGB':
        return XGBRegressor(**p, random_state=seed, n_jobs=-1, verbosity=0)
    if model_name not in MODEL_CLASSES:
        raise KeyError(f'unknown model: {model_name!r}')
    return MODEL_CLASSES[model_name](**p, random_state=seed, n_jobs=-1)


def clone_with_params(model_name, params):
    """Clone the base estimator and set hyperparameters. Preserves the
    base factory configuration (early stopping, tree_method, etc.)."""
    est = clone(BASE_MODELS[model_name]())
    if params:
        est.set_params(**params)
    return est


def predict_median(model, X):
    """Median across trees for RF/ERT. Default `predict` for XGB/GB."""
    if hasattr(model, 'estimators_'):
        try:
            per_tree = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
            return np.median(per_tree, axis=0)
        except Exception:
            return model.predict(X)
    return model.predict(X)


def predict_iqr(model, X):
    """Return (median, q16, q84) across trees for RF/ERT. Collapses to
    a degenerate interval for XGB/GB since they have no tree ensemble
    with directly accessible predictions."""
    if hasattr(model, 'estimators_'):
        try:
            per_tree = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
            return (
                np.median(per_tree, axis=0),
                np.percentile(per_tree, 16, axis=0),
                np.percentile(per_tree, 84, axis=0),
            )
        except Exception:
            p = model.predict(X)
            return p, p, p
    p = model.predict(X)
    return p, p, p
