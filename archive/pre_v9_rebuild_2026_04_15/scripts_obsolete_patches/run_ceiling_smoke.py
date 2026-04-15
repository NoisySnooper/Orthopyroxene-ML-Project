"""Run Phase 3R.8 ceiling cells in isolation to confirm they execute cleanly
before we papermill the full NB03.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROC, DATA_SPLITS, FIGURES, RESULTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from src.features import build_feature_matrix

with open(RESULTS / 'nb03_winning_configurations.json') as f:
    config = json.load(f)
WIN_FEAT_CEIL = config['global_feature_set']
print(f'Ceiling analysis feature set: {WIN_FEAT_CEIL}')

df_liq_c = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
idx_tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
idx_te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')
df_tr_c = df_liq_c.loc[idx_tr].copy()
df_te_c = df_liq_c.loc[idx_te].copy()
X_tr_c, _ = build_feature_matrix(df_tr_c, WIN_FEAT_CEIL, use_liq=True)
X_te_c, _ = build_feature_matrix(df_te_c, WIN_FEAT_CEIL, use_liq=True)
yT_tr, yP_tr = df_tr_c['T_C'].values, df_tr_c['P_kbar'].values
yT_te, yP_te = df_te_c['T_C'].values, df_te_c['P_kbar'].values


def _parse_params(s):
    if isinstance(s, dict):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return json.loads(s)


multi_c = pd.read_csv(RESULTS / 'nb03_multi_seed_results.csv')
rf_liq_c = multi_c[(multi_c.track == 'opx_liq') & (multi_c.model_name == 'RF')
                   & (multi_c.feature_set == WIN_FEAT_CEIL)
                   & (multi_c.split_seed == 42)]
rf_params_T = _parse_params(rf_liq_c[rf_liq_c.target == 'T_C'].iloc[0]['best_params'])
rf_params_P = _parse_params(rf_liq_c[rf_liq_c.target == 'P_kbar'].iloc[0]['best_params'])
print(f'train n={len(df_tr_c)}  test n={len(df_te_c)}  n_features={X_tr_c.shape[1]}')

SEED_CEIL = 42

def _mk_rf(params):
    return RandomForestRegressor(**params, random_state=SEED_CEIL, n_jobs=-1)
def _mk_et():
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                               max_features='sqrt', random_state=SEED_CEIL, n_jobs=-1)
def _mk_xgb():
    return xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0,
                            tree_method='hist', random_state=SEED_CEIL,
                            n_jobs=-1, verbosity=0)
def _mk_mlp(hidden):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=hidden, activation='relu',
                             solver='adam', alpha=1e-3, learning_rate_init=1e-3,
                             max_iter=800, early_stopping=True,
                             validation_fraction=0.1, random_state=SEED_CEIL)),
    ])
def _mk_ridge():
    return Pipeline([('scaler', StandardScaler()),
                     ('ridge', Ridge(alpha=1.0, random_state=SEED_CEIL))])

FAMILIES_CEIL = [
    ('RF_baseline',    lambda: _mk_rf(rf_params_T), lambda: _mk_rf(rf_params_P)),
    ('ExtraTrees',     _mk_et,     _mk_et),
    ('XGBoost_tuned',  _mk_xgb,    _mk_xgb),
    ('MLP_32_64_32',   lambda: _mk_mlp((32, 64, 32)),  lambda: _mk_mlp((32, 64, 32))),
    ('MLP_64_128_64',  lambda: _mk_mlp((64, 128, 64)), lambda: _mk_mlp((64, 128, 64))),
    ('Ridge',          _mk_ridge,  _mk_ridge),
]

def _eval_one(name, factory, X_tr, y_tr, X_te, y_te):
    m = factory()
    m.fit(X_tr, y_tr)
    pred_tr = m.predict(X_tr)
    pred_te = m.predict(X_te)
    return {
        'family': name,
        'rmse_train': float(np.sqrt(mean_squared_error(y_tr, pred_tr))),
        'rmse_test':  float(np.sqrt(mean_squared_error(y_te, pred_te))),
        'mae_test':   float(mean_absolute_error(y_te, pred_te)),
        'r2_test':    float(r2_score(y_te, pred_te)),
    }

rows = []
for name, fT, fP in FAMILIES_CEIL:
    print(f'  fitting {name} ...', flush=True)
    rT = _eval_one(name, fT, X_tr_c, yT_tr, X_te_c, yT_te); rT['target'] = 'T_C'
    rP = _eval_one(name, fP, X_tr_c, yP_tr, X_te_c, yP_te); rP['target'] = 'P_kbar'
    rows += [rT, rP]

df_fam = pd.DataFrame(rows)[['target', 'family', 'rmse_train',
                             'rmse_test', 'mae_test', 'r2_test']]
df_fam.to_csv(RESULTS / 'nb11_model_family_ceiling.csv', index=False)

for tgt in ['T_C', 'P_kbar']:
    sub = df_fam[df_fam.target == tgt].sort_values('rmse_test')
    print(f'\n{tgt}:')
    print(sub[['family', 'rmse_test', 'mae_test', 'r2_test']].round(3).to_string(index=False))
    rng = sub['rmse_test'].max() - sub['rmse_test'].min()
    unit = 'C' if tgt == 'T_C' else 'kbar'
    print(f'  spread: {rng:.2f} {unit}')

COLORS_CEIL = {
    'RF_baseline': '#0072B2', 'ExtraTrees': '#56B4E9',
    'XGBoost_tuned': '#D55E00', 'MLP_32_64_32': '#009E73',
    'MLP_64_128_64': '#117755', 'Ridge': '#CC79A7',
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for ax, tgt, units in zip(axes, ['T_C', 'P_kbar'], ['C', 'kbar']):
    sub = df_fam[df_fam.target == tgt].sort_values('rmse_test')
    colors = [COLORS_CEIL.get(f, '#777777') for f in sub['family']]
    ax.barh(sub['family'], sub['rmse_test'], color=colors, edgecolor='black')
    rf_rmse = float(sub.loc[sub.family == 'RF_baseline', 'rmse_test'].iloc[0])
    ax.axvline(rf_rmse, color='#0072B2', linestyle='--', linewidth=1.2,
               alpha=0.6, label=f'RF baseline = {rf_rmse:.2f} {units}')
    ax.set_xlabel(f'Test RMSE ({units})')
    ax.set_title(f'{tgt}: ceiling (n_test={len(df_te_c)})')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
fig.suptitle('Model-family ceiling')
fig.savefig(FIGURES / 'fig_nb11_model_family_ceiling.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES / 'fig_nb11_model_family_ceiling.pdf', bbox_inches='tight')
plt.close(fig)
print('DONE')
