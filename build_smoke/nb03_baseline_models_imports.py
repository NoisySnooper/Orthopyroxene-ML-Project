import os
os.chdir(r"C:\Users\NQTa\Documents\MLCourse\Final Project\notebooks")
import sys, os
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
import warnings
warnings.filterwarnings('ignore')

import json
import logging
import shutil
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, HalvingRandomSearchCV,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Local vs Colab pathing
if os.path.exists('/content'):
    TEMP_DIR = Path('/content')
else:
    TEMP_DIR = Path.cwd()

FIGS = FIGURES  # alias
# Canonical features and prediction helpers from src/ (one source of truth).
from src.features import (
    build_feature_matrix,
    make_raw_features,
    make_alr_features,
    make_pwlr_features,
    augment_dataframe,
)
from src.models import predict_median, predict_iqr

# Phase 3R.8 setup: load canonical split + RF params from NB03 multi-seed table.
import ast
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

from src.features import build_feature_matrix

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
    if isinstance(s, dict): return s
    try: return ast.literal_eval(s)
    except Exception:
        try: return json.loads(s)
        except Exception: return {}

multi_c = pd.read_csv(RESULTS / 'nb03_multi_seed_results.csv')
rf_liq_c = multi_c[(multi_c.track == 'opx_liq') & (multi_c.model_name == 'RF') &
                   (multi_c.feature_set == WIN_FEAT_CEIL) & (multi_c.split_seed == 42)]
rf_params_T = _parse_params(rf_liq_c[rf_liq_c.target == 'T_C'].iloc[0]['best_params'])
rf_params_P = _parse_params(rf_liq_c[rf_liq_c.target == 'P_kbar'].iloc[0]['best_params'])
print(f'train n={len(df_tr_c)}  test n={len(df_te_c)}  n_features={X_tr_c.shape[1]}')
print(f'RF T params: {rf_params_T}')
print(f'RF P params: {rf_params_P}')
