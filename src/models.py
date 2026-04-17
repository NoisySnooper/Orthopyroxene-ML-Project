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
import os

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from config import SEED_MODEL

SEARCH_NJOBS = -1
ESTIMATOR_NJOBS = 1


def _make_elasticnet():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(random_state=SEED_MODEL, max_iter=10000)),
    ])


def _make_mlp():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=SEED_MODEL,
                             early_stopping=True,
                             validation_fraction=0.15,
                             n_iter_no_change=20,
                             max_iter=500)),
    ])


BASE_MODELS = {
    'RF':  lambda: RandomForestRegressor(random_state=SEED_MODEL, n_jobs=ESTIMATOR_NJOBS),
    'ERT': lambda: ExtraTreesRegressor(random_state=SEED_MODEL, n_jobs=ESTIMATOR_NJOBS),
    'XGB': lambda: XGBRegressor(random_state=SEED_MODEL, n_jobs=ESTIMATOR_NJOBS,
                                verbosity=0, tree_method='hist'),
    'GB':  lambda: HistGradientBoostingRegressor(random_state=SEED_MODEL,
                                                 early_stopping=True,
                                                 validation_fraction=0.15,
                                                 n_iter_no_change=20),
    'CatBoost': lambda: CatBoostRegressor(random_seed=SEED_MODEL,
                                          verbose=False,
                                          allow_writing_files=False,
                                          thread_count=ESTIMATOR_NJOBS),
    'LightGBM': lambda: LGBMRegressor(random_state=SEED_MODEL,
                                      n_jobs=ESTIMATOR_NJOBS,
                                      verbose=-1),
    'ElasticNet': _make_elasticnet,
    'MLP': _make_mlp,
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
    'CatBoost': CatBoostRegressor,
    'LightGBM': LGBMRegressor,
    'ElasticNet': ElasticNet,
    'MLP': MLPRegressor,
}


PIPELINE_MODELS = {'ElasticNet', 'MLP'}


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
    `seed` is always honored and njobs flags remain consistent. For
    pipeline models (ElasticNet, MLP), params target the inner step.

    Env var V10_MAX_JOBS caps n_jobs/thread_count per model (useful when
    running multiple runners in parallel). Default -1 (all cores).
    """
    nj = int(os.environ.get('V10_MAX_JOBS', '-1'))
    p = dict(params)
    if model_name == 'GB':
        return HistGradientBoostingRegressor(**p, random_state=seed)
    if model_name == 'XGB':
        return XGBRegressor(**p, random_state=seed, n_jobs=nj, verbosity=0)
    if model_name == 'CatBoost':
        # CatBoost thread_count: -1 means all cores; pass cap when set.
        tc = {} if nj == -1 else {'thread_count': nj}
        return CatBoostRegressor(**p, **tc, random_seed=seed, verbose=False,
                                 allow_writing_files=False)
    if model_name == 'LightGBM':
        return LGBMRegressor(**p, random_state=seed, n_jobs=nj, verbose=-1)
    if model_name == 'ElasticNet':
        est = _make_elasticnet()
        if p:
            # Accept either prefixed ('enet__alpha') or bare ('alpha') keys.
            p_out = {k if k.startswith('enet__') else f'enet__{k}': v
                     for k, v in p.items()}
            est.set_params(**p_out)
        est.named_steps['enet'].set_params(random_state=seed)
        return est
    if model_name == 'MLP':
        est = _make_mlp()
        if p:
            # Translate Optuna trial keys (n_layers, layer_size) to sklearn's
            # hidden_layer_sizes tuple. Mirror logic in optuna_search._suggest_mlp_params.
            if 'n_layers' in p or 'layer_size' in p:
                n_layers = int(p.pop('n_layers', 1))
                layer_size = int(p.pop('layer_size', 64))
                p['hidden_layer_sizes'] = ((layer_size,) if n_layers == 1
                                           else (layer_size, layer_size // 2))
            p_out = {k if k.startswith('mlp__') else f'mlp__{k}': v
                     for k, v in p.items()}
            est.set_params(**p_out)
        est.named_steps['mlp'].set_params(random_state=seed)
        return est
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
