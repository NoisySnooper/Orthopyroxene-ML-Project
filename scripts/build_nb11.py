"""Build notebooks/nb11_model_family_ceiling.ipynb from scratch.

Purpose:
    Test whether RandomForest performance is a ceiling effect of the feature
    set or a limitation of the model family. Evaluate six families on the
    canonical opx-liq split (same X_train / X_test / WIN_FEAT as NB03 / NB07)
    and compare test RMSE: RF baseline, ExtraTrees, XGBoost_tuned,
    MLP_32_64_32, MLP_64_128_64, Ridge. If all six families cluster within
    ~0.5 RMSE of each other, performance is feature-set limited (ceiling),
    not model-limited.

Inputs:
    - data/processed/opx_clean_opx_liq.parquet
    - data/splits/{train,test}_indices_opx_liq.npy
    - results/nb03_winning_configurations.json  (global WIN_FEAT)
    - results/nb03_multi_seed_results.csv  (RF hyperparams for seed-42)

Outputs:
    - results/nb11_model_family_ceiling.csv (test RMSE/MAE/R2 per family x target)
    - figures/fig_nb11_model_family_ceiling.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebooks' / 'nb11_model_family_ceiling.ipynb'


HEADER_MD = """# NB11: Model-family ceiling analysis

**Purpose:** Test whether the RandomForest test RMSE is a property of the feature
set (ceiling) rather than a property of the RF family itself. Evaluate six
model families on the same canonical opx-liq split + feature set used in NB03
and NB07, and compare test RMSE.

**Interpretation:** If the six families cluster within ~0.5 RMSE of each other,
the model is feature-limited and performance is at or near the information
ceiling for this input set (supports Part 6's H2O engineered-feature work).

**Inputs:** `opx_clean_opx_liq.parquet`, split indices, `nb03_winning_configurations.json`,
`nb03_multi_seed_results.csv`.

**Outputs:** `results/nb11_model_family_ceiling.csv`,
`figures/fig_nb11_model_family_ceiling.{png,pdf}`.
"""


IMPORTS_CODE = """import sys
import json
import ast
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))
from config import (
    ROOT, DATA_PROC, DATA_SPLITS, MODELS, FIGURES, RESULTS,
    SEED_SPLIT, SEED_MODEL,
)
from src.features import build_feature_matrix
from src.plot_style import load_winning_config
from src.models import predict_median

warnings.filterwarnings('ignore')

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
"""


LOAD_CODE = """# Load canonical data, split, and winning feature set (same as NB03 / NB07)
config_3r = load_winning_config(RESULTS)
WIN_FEAT = config_3r['global_feature_set']
print(f'Global winning feature set: {WIN_FEAT}')

df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
idx_tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
idx_te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')

df_train = df_liq.loc[idx_tr].copy()
df_test  = df_liq.loc[idx_te].copy()

X_train, feat_names = build_feature_matrix(df_train, WIN_FEAT, use_liq=True)
X_test,  _          = build_feature_matrix(df_test,  WIN_FEAT, use_liq=True)
y_train_T = df_train['T_C'].values
y_train_P = df_train['P_kbar'].values
y_test_T  = df_test['T_C'].values
y_test_P  = df_test['P_kbar'].values

# RF hyperparameters from NB03 multi-seed results (seed-42, winning feature set)
def parse_params(s):
    if isinstance(s, dict): return s
    try:    return ast.literal_eval(s)
    except:
        try:    return json.loads(s)
        except: return {}

multi = pd.read_csv(RESULTS / 'nb03_multi_seed_results.csv')
rf_liq = multi[(multi.track == 'opx_liq') & (multi.model_name == 'RF') &
               (multi.feature_set == WIN_FEAT) & (multi.split_seed == 42)]
rf_params_T = parse_params(rf_liq[rf_liq.target == 'T_C'].iloc[0]['best_params'])
rf_params_P = parse_params(rf_liq[rf_liq.target == 'P_kbar'].iloc[0]['best_params'])

print(f'train n={len(df_train)}  test n={len(df_test)}  n_features={X_train.shape[1]}')
print(f'RF T params: {rf_params_T}')
print(f'RF P params: {rf_params_P}')
"""


FAMILIES_CODE = """# Build six model-family configurations. Each returns (name, factory_T, factory_P)
# where the factory is a callable producing a fresh fitted estimator.

def mk_rf_T():
    return RandomForestRegressor(**rf_params_T, random_state=SEED_MODEL, n_jobs=-1)
def mk_rf_P():
    return RandomForestRegressor(**rf_params_P, random_state=SEED_MODEL, n_jobs=-1)

def mk_et():
    return ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, max_features='sqrt',
        random_state=SEED_MODEL, n_jobs=-1,
    )

def mk_xgb():
    return xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist', random_state=SEED_MODEL, n_jobs=-1,
        verbosity=0,
    )

def mk_mlp(hidden):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=hidden, activation='relu',
                             solver='adam', alpha=1e-3, learning_rate_init=1e-3,
                             max_iter=800, early_stopping=True,
                             validation_fraction=0.1,
                             random_state=SEED_MODEL)),
    ])

def mk_ridge():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(alpha=1.0, random_state=SEED_MODEL)),
    ])

FAMILIES = [
    ('RF_baseline',     mk_rf_T,   mk_rf_P),
    ('ExtraTrees',      mk_et,     mk_et),
    ('XGBoost_tuned',   mk_xgb,    mk_xgb),
    ('MLP_32_64_32',    lambda: mk_mlp((32, 64, 32)), lambda: mk_mlp((32, 64, 32))),
    ('MLP_64_128_64',   lambda: mk_mlp((64, 128, 64)), lambda: mk_mlp((64, 128, 64))),
    ('Ridge',           mk_ridge,  mk_ridge),
]
print(f'Testing {len(FAMILIES)} model families.')
"""


EVAL_CODE = """# Fit each family on X_train and evaluate on X_test.

def eval_one(family_name, factory, X_tr, y_tr, X_te, y_te):
    model = factory()
    model.fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    return {
        'family':      family_name,
        'rmse_train':  float(np.sqrt(mean_squared_error(y_tr, pred_tr))),
        'rmse_test':   float(np.sqrt(mean_squared_error(y_te, pred_te))),
        'mae_test':    float(mean_absolute_error(y_te, pred_te)),
        'r2_test':     float(r2_score(y_te, pred_te)),
    }


rows = []
for (name, fac_T, fac_P) in FAMILIES:
    print(f'  fitting {name} ...', flush=True)
    r_T = eval_one(name, fac_T, X_train, y_train_T, X_test, y_test_T)
    r_T['target'] = 'T_C'
    rows.append(r_T)
    r_P = eval_one(name, fac_P, X_train, y_train_P, X_test, y_test_P)
    r_P['target'] = 'P_kbar'
    rows.append(r_P)

df_fam = pd.DataFrame(rows)[['target', 'family', 'rmse_train', 'rmse_test',
                             'mae_test', 'r2_test']]
df_fam.to_csv(RESULTS / 'nb11_model_family_ceiling.csv', index=False)

print('\\n== Model-family test RMSE ==')
for tgt in ['T_C', 'P_kbar']:
    sub = df_fam[df_fam.target == tgt].sort_values('rmse_test')
    print(f'\\n{tgt}:')
    print(sub[['family', 'rmse_test', 'mae_test', 'r2_test']].round(3).to_string(index=False))

# Ceiling check: spread across families
for tgt in ['T_C', 'P_kbar']:
    sub = df_fam[df_fam.target == tgt]
    rmse_min = sub['rmse_test'].min()
    rmse_max = sub['rmse_test'].max()
    rmse_rng = rmse_max - rmse_min
    unit = 'C' if tgt == 'T_C' else 'kbar'
    print(f'\\n{tgt}: family RMSE range = {rmse_rng:.2f} {unit} '
          f'({rmse_min:.2f} - {rmse_max:.2f}).')
    if tgt == 'P_kbar' and rmse_rng < 0.5:
        print('  -> feature-set ceiling: families cluster within 0.5 kbar')
    elif tgt == 'T_C' and rmse_rng < 20.0:
        print('  -> feature-set ceiling: families cluster within 20 C')
    else:
        print('  -> spread suggests residual model-family gain possible')
"""


FIG_CODE = """# Figure: horizontal bar chart of test RMSE by family, one panel per target.
# Colorblind-safe Okabe-Ito palette: #0072B2 (RF/baseline), #E69F00 (boosting),
# #009E73 (MLP), #56B4E9 (extra trees), #CC79A7 (Ridge), #D55E00 (XGB).
COLORS = {
    'RF_baseline':    '#0072B2',
    'ExtraTrees':     '#56B4E9',
    'XGBoost_tuned':  '#D55E00',
    'MLP_32_64_32':   '#009E73',
    'MLP_64_128_64':  '#117755',
    'Ridge':          '#CC79A7',
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for ax, tgt, units in zip(axes, ['T_C', 'P_kbar'], ['C', 'kbar']):
    sub = df_fam[df_fam.target == tgt].sort_values('rmse_test')
    colors = [COLORS.get(f, '#777777') for f in sub['family']]
    bars = ax.barh(sub['family'], sub['rmse_test'], color=colors, edgecolor='black')
    # RF baseline reference line
    rf_rmse = float(sub.loc[sub.family == 'RF_baseline', 'rmse_test'].iloc[0])
    ax.axvline(rf_rmse, color='#0072B2', linestyle='--', linewidth=1.2,
               alpha=0.6, label=f'RF baseline = {rf_rmse:.2f} {units}')
    for b, v in zip(bars, sub['rmse_test']):
        ax.text(v + 0.01 * v, b.get_y() + b.get_height()/2,
                f'{v:.2f}', va='center', fontsize=9)
    ax.set_xlabel(f'Test RMSE ({units})')
    ax.set_title(f'{tgt}: model-family ceiling (n_test={len(df_test)})')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)

fig.suptitle('Model-family ceiling analysis: six families on canonical feature set',
             fontsize=11)
fig_path_png = FIGURES / 'fig_nb11_model_family_ceiling.png'
fig_path_pdf = FIGURES / 'fig_nb11_model_family_ceiling.pdf'
fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
fig.savefig(fig_path_pdf, bbox_inches='tight')
plt.show()
print(f'Saved {fig_path_png.name} and {fig_path_pdf.name}')
"""


VERIFY_CODE = """# Verification assertions
assert (RESULTS / 'nb11_model_family_ceiling.csv').exists()
assert (FIGURES / 'fig_nb11_model_family_ceiling.png').exists()
assert (FIGURES / 'fig_nb11_model_family_ceiling.pdf').exists()

chk = pd.read_csv(RESULTS / 'nb11_model_family_ceiling.csv')
assert set(chk['target']) == {'T_C', 'P_kbar'}
expected = {'RF_baseline', 'ExtraTrees', 'XGBoost_tuned',
            'MLP_32_64_32', 'MLP_64_128_64', 'Ridge'}
assert set(chk['family']) == expected
assert {'rmse_test', 'mae_test', 'r2_test'}.issubset(chk.columns)
print('=== NB11 COMPLETE ===')
print(f'  families tested: {sorted(expected)}')
print(f'  rows: {len(chk)}')
"""


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(HEADER_MD,          metadata={'id': 'nb11-header'}),
        nbformat.v4.new_code_cell    (IMPORTS_CODE,       metadata={'id': 'nb11-imports'}),
        nbformat.v4.new_code_cell    (LOAD_CODE,          metadata={'id': 'nb11-load'}),
        nbformat.v4.new_code_cell    (FAMILIES_CODE,      metadata={'id': 'nb11-families'}),
        nbformat.v4.new_code_cell    (EVAL_CODE,          metadata={'id': 'nb11-eval'}),
        nbformat.v4.new_code_cell    (FIG_CODE,           metadata={'id': 'nb11-fig'}),
        nbformat.v4.new_code_cell    (VERIFY_CODE,        metadata={'id': 'nb11-verify'}),
    ]
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }
    NB_PATH.write_text(nbformat.writes(nb), encoding='utf-8')
    print(f'wrote {NB_PATH}')


if __name__ == '__main__':
    main()
