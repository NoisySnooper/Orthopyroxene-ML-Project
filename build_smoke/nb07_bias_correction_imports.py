import os
os.chdir(r"C:\Users\NQTa\Documents\MLCourse\Final Project\notebooks")
import sys
import json
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
    CONFORMAL_ALPHA,
)
from src.plot_style import load_winning_config, canonical_model_filename
import warnings
warnings.filterwarnings('ignore')

import ast
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
# Canonical features and prediction helpers from src/ (one source of truth).
from src.features import (
    build_feature_matrix,
    make_raw_features,
    make_alr_features,
    make_pwlr_features,
    augment_dataframe,
)
from src.models import predict_median, predict_iqr

# Phase 7R setup: load data, splits, canonical winning feature set, and RF models
config_3r = load_winning_config(RESULTS)
WIN_FEAT = config_3r['global_feature_set']
print(f'Phase 3R global winning feature set: {WIN_FEAT}')

df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
idx_tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
idx_te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')

df_train = df_liq.loc[idx_tr].copy()
df_test = df_liq.loc[idx_te].copy()

def parse_params(s):
    import ast, json
    if isinstance(s, dict): return s
    try: return ast.literal_eval(s)
    except:
        try: return json.loads(s)
        except: return {}

# Reconstruct RF opx-liq hyperparameters for the winning feature set from the
# multi-seed results table (seed-42 rows are the canonical reference).
multi = pd.read_csv(RESULTS / 'nb03_multi_seed_results.csv')
rf_liq = multi[(multi.track == 'opx_liq') &
               (multi.model_name == 'RF') &
               (multi.feature_set == WIN_FEAT) &
               (multi.split_seed == 42)]

params_T = parse_params(rf_liq[rf_liq.target == 'T_C'].iloc[0]['best_params'])
params_P = parse_params(rf_liq[rf_liq.target == 'P_kbar'].iloc[0]['best_params'])

# Load canonical RF models (same feature set for T and P under dynamic selection)
model_T = joblib.load(MODELS / canonical_model_filename('RF', 'T_C', 'opx_liq', RESULTS))
model_P = joblib.load(MODELS / canonical_model_filename('RF', 'P_kbar', 'opx_liq', RESULTS))

# Build feature matrices using the standalone function
X_train, feat_names = build_feature_matrix(df_train, WIN_FEAT, use_liq=True)
X_test, _ = build_feature_matrix(df_test, WIN_FEAT, use_liq=True)

y_train_T = df_train['T_C'].values
y_train_P = df_train['P_kbar'].values
y_test_T = df_test['T_C'].values
y_test_P = df_test['P_kbar'].values

# Groups for StratifiedGroupKFold (used in OOF tracks)
groups_train = df_train['Citation'].astype(str).values
print(f'train n={len(df_train)} test n={len(df_test)} features={len(feat_names)}')