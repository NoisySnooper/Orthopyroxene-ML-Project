import os
os.chdir(r"C:\Users\NQTa\Documents\MLCourse\Final Project\notebooks")
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from config import (
    ROOT, DATA_RAW, DATA_EXTERNAL, DATA_PROC, DATA_SPLITS, DATA_NATURAL,
    MODELS, FIGURES, RESULTS, LOGS,
    EXPETDB, LEPR_XLSX, LIN2023_NATURAL,
    FE3_FET_RATIO, KD_FEMG_MIN, KD_FEMG_MAX, WO_MAX_MOL_PCT,
    P_CEILING_KBAR, CATION_SUM_MIN, CATION_SUM_MAX,
    OXIDE_TOTAL_MIN, OXIDE_TOTAL_MAX,
    SEED_SPLIT, SEED_MODEL, SEED_NOISE_AUG, SEED_KMEANS,
    OPX_RAW_OXIDES, OPX_FULL_OXIDES, LIQ_OXIDES,
)
from src.plot_style import load_winning_config
import warnings
warnings.filterwarnings('ignore')

import ast
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
# Canonical features and prediction helpers from src/ (one source of truth).
from src.features import (
    build_feature_matrix,
    make_raw_features,
    make_alr_features,
    make_pwlr_features,
    augment_dataframe,
)
from src.models import predict_median, predict_iqr

import ast
import json
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Dynamic setup: load data and canonical features from Phase 3R winner
df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')

config_3r = load_winning_config(RESULTS)
WIN_FEAT = config_3r['global_feature_set']
print(f'Phase 3R global winning feature set: {WIN_FEAT}')

# Validation uses the UNAUGMENTED feature set (no noise injection during CV)
X, feature_names = build_feature_matrix(df_liq, WIN_FEAT, use_liq=True)
groups_study = df_liq['Citation'].values
clusters = df_liq['chemical_cluster'].values
y_T = df_liq['T_C'].values
y_P = df_liq['P_kbar'].values

print(f'opx-liq rows: {len(df_liq)}, studies: {df_liq["Citation"].nunique()}, clusters: {df_liq["chemical_cluster"].nunique()}, X.shape: {X.shape}')

# Reconstruct best_params_by_model from the seed-42 rows of the canonical
# feature set in the multi-seed results CSV
ms = pd.read_csv(RESULTS / 'nb03_multi_seed_results.csv')
ms_canonical = ms[(ms['split_seed'] == 42) &
                  (ms['feature_set'] == WIN_FEAT) &
                  (ms['track'] == 'opx_liq')]

def _parse_params(s):
    if isinstance(s, dict):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return {}

best_params_by_model = {}
for _, row in ms_canonical.iterrows():
    bp = _parse_params(row['best_params'])
    best_params_by_model.setdefault(row['model_name'], {})[row['target']] = bp

print('Best params reconstructed for models:', sorted(best_params_by_model.keys()))

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

MODEL_CLASSES = {
    'RF': RandomForestRegressor,
    'ERT': ExtraTreesRegressor,
    'XGB': XGBRegressor,
    'GB': HistGradientBoostingRegressor,
}

def build_model(model_name, params):
    p = dict(params)
    if model_name == 'GB':
        return HistGradientBoostingRegressor(**p, random_state=SEED_MODEL)
    if model_name == 'XGB':
        return XGBRegressor(**p, random_state=SEED_MODEL, n_jobs=-1, verbosity=0)
    return MODEL_CLASSES[model_name](**p, random_state=SEED_MODEL, n_jobs=-1)