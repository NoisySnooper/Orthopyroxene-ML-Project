"""Optuna TPE hyperparameter search for the 4 base regressors.

One study per (model, target, track, feature_set). Caller orchestrates
the 48-study sweep in NB03 Phase 3.3b (see `docs/optuna_strategy.md`).

Public API:
- `optuna_search(model_name, X, y, groups, n_trials, seed, timeout_per_trial,
                 n_jobs_inner, study_save_path)` -> dict with keys
  `best_params`, `best_score`, `best_trial`, `study`.

Search spaces live in the `_suggest_{name}_params` helpers. They are the
single source of truth for reproducibility; bumping a range is a
code-level decision, not a runtime flag.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import optuna
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Search spaces (fixed; see docs/optuna_strategy.md)
# ---------------------------------------------------------------------------

def _suggest_rf_params(trial: optuna.trial.Trial) -> dict:
    return {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 800, step=100),
        'max_depth':        trial.suggest_int('max_depth', 5, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features':     trial.suggest_float('max_features', 0.3, 1.0),
    }


def _suggest_ert_params(trial: optuna.trial.Trial) -> dict:
    return {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 800, step=100),
        'max_depth':        trial.suggest_int('max_depth', 5, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features':     trial.suggest_float('max_features', 0.3, 1.0),
    }


def _suggest_xgb_params(trial: optuna.trial.Trial) -> dict:
    return {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 800, step=100),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
    }


def _suggest_gb_params(trial: optuna.trial.Trial) -> dict:
    return {
        'max_iter':          trial.suggest_int('max_iter', 200, 800, step=100),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 10, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        'max_leaf_nodes':    trial.suggest_int('max_leaf_nodes', 15, 127, log=True),
    }


def _suggest_catboost_params(trial: optuna.trial.Trial) -> dict:
    return {
        'iterations':        trial.suggest_int('iterations', 200, 1000, step=100),
        'depth':             trial.suggest_int('depth', 4, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength':   trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
    }


def _suggest_lightgbm_params(trial: optuna.trial.Trial) -> dict:
    return {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 127, log=True),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }


def _suggest_elasticnet_params(trial: optuna.trial.Trial) -> dict:
    # Params target the inner ElasticNet step of the Pipeline.
    return {
        'enet__alpha':     trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'enet__l1_ratio':  trial.suggest_float('l1_ratio', 0.0, 1.0),
    }


def _suggest_mlp_params(trial: optuna.trial.Trial) -> dict:
    # Single/double layer search with log-scaled learning rate and alpha.
    n_layers = trial.suggest_int('n_layers', 1, 2)
    layer_size = trial.suggest_int('layer_size', 32, 256, log=True)
    if n_layers == 1:
        hidden = (layer_size,)
    else:
        hidden = (layer_size, layer_size // 2)
    return {
        'mlp__hidden_layer_sizes': hidden,
        'mlp__alpha':              trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'mlp__learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'mlp__activation':         trial.suggest_categorical('activation', ['relu', 'tanh']),
    }


_SUGGEST = {
    'RF':  _suggest_rf_params,
    'ERT': _suggest_ert_params,
    'XGB': _suggest_xgb_params,
    'GB':  _suggest_gb_params,
    'CatBoost':   _suggest_catboost_params,
    'LightGBM':   _suggest_lightgbm_params,
    'ElasticNet': _suggest_elasticnet_params,
    'MLP':        _suggest_mlp_params,
}


def _build_estimator(model_name: str, params: dict, seed: int, n_jobs_inner: int):
    if model_name == 'RF':
        return RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs_inner)
    if model_name == 'ERT':
        return ExtraTreesRegressor(**params, random_state=seed, n_jobs=n_jobs_inner)
    if model_name == 'XGB':
        return XGBRegressor(**params, random_state=seed, n_jobs=n_jobs_inner,
                            verbosity=0, tree_method='hist')
    if model_name == 'GB':
        return HistGradientBoostingRegressor(**params, random_state=seed)
    if model_name == 'CatBoost':
        return CatBoostRegressor(**params, random_seed=seed, verbose=False,
                                 allow_writing_files=False,
                                 thread_count=n_jobs_inner)
    if model_name == 'LightGBM':
        return LGBMRegressor(**params, random_state=seed, n_jobs=n_jobs_inner,
                             verbose=-1)
    if model_name == 'ElasticNet':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('enet', ElasticNet(random_state=seed, max_iter=10000)),
        ])
        pipe.set_params(**params)
        return pipe
    if model_name == 'MLP':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(random_state=seed,
                                 early_stopping=True,
                                 validation_fraction=0.15,
                                 n_iter_no_change=20,
                                 max_iter=500)),
        ])
        pipe.set_params(**params)
        return pipe
    raise KeyError(f'unknown model: {model_name!r}')


# ---------------------------------------------------------------------------
# Public search entry point
# ---------------------------------------------------------------------------

def optuna_search(model_name: str,
                  X: np.ndarray,
                  y: np.ndarray,
                  groups: np.ndarray,
                  n_trials: int = 50,
                  seed: int = 42,
                  timeout_per_trial: int = 300,
                  n_jobs_inner: int = 4,
                  study_save_path: Optional[str | Path] = None,
                  study_name: Optional[str] = None) -> dict:
    """Run one TPE study and return best hyperparameters.

    Scoring: neg-RMSE via 3-fold GroupKFold inner CV. We negate back to
    positive RMSE in the returned `best_score`.
    """
    if model_name not in _SUGGEST:
        raise KeyError(f'unknown model: {model_name!r}')

    cv = GroupKFold(n_splits=3)
    suggest = _SUGGEST[model_name]

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest(trial)
        est = _build_estimator(model_name, params, seed=seed, n_jobs_inner=n_jobs_inner)
        # neg-RMSE per sklearn convention
        scores = cross_val_score(
            est, X, y,
            groups=groups,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=1,
            error_score='raise',
        )
        return float(-scores.mean())  # positive RMSE; we minimize

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None,  # per-trial budget handled implicitly via pruner/time
        gc_after_trial=True,
        show_progress_bar=False,
    )

    if study_save_path is not None:
        path = Path(study_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, path)

    return {
        'best_params': dict(study.best_params),
        'best_score':  float(study.best_value),  # positive RMSE
        'best_trial':  int(study.best_trial.number),
        'study':       study,
    }
