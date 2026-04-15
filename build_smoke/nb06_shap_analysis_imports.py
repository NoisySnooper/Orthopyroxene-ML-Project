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
from src.plot_style import load_winning_config, canonical_model_filename
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
# Canonical features and prediction helpers from src/ (one source of truth).
from src.features import (
    build_feature_matrix,
    make_raw_features,
    make_alr_features,
    make_pwlr_features,
    augment_dataframe,
)
from src.models import predict_median, predict_iqr

# Load the canonical train + test feature matrices for robustness testing.
train_idx = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
df_train = df_liq.loc[train_idx].copy()

X_train, _ = build_feature_matrix(df_train, WIN_FEAT, use_liq=True)
y_train_P = df_train['P_kbar'].values
y_train_T = df_train['T_C'].values
y_test_P = df_test['P_kbar'].values
y_test_T = df_test['T_C'].values

# X_train/X_test are numpy arrays; wrap in DataFrames so we can drop columns by
# feature name in the ablation test below.
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score

# Refit the canonical models once and record original RMSE.
model_P_fit = clone(model_P); model_P_fit.fit(X_train_df.values, y_train_P)
model_T_fit = clone(model_T); model_T_fit.fit(X_train_df.values, y_train_T)
original_rmse_P = float(np.sqrt(mean_squared_error(
    y_test_P, predict_median(model_P_fit, X_test_df.values))))
original_rmse_T = float(np.sqrt(mean_squared_error(
    y_test_T, predict_median(model_T_fit, X_test_df.values))))
print(f'Baseline RMSE P = {original_rmse_P:.3f} kbar | T = {original_rmse_T:.2f} C')
