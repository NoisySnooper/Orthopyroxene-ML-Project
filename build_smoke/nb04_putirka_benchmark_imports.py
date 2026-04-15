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
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import Thermobar as pt
# Canonical features and prediction helpers from src/ (one source of truth).
from src.features import (
    build_feature_matrix,
    make_raw_features,
    make_alr_features,
    make_pwlr_features,
    augment_dataframe,
)
from src.models import predict_median, predict_iqr

# Force RF rows for fair comparison regardless of overall winner
import ast
from sklearn.ensemble import RandomForestRegressor

results_all = pd.read_csv(RESULTS / 'nb03_multi_seed_summary.csv')
rf_liq = results_all[(results_all.track=='opx_liq') & (results_all.model_name=='RF')]

rf_T = rf_liq[rf_liq.target=='T_C'].sort_values('rmse_test_mean').iloc[0]
rf_P = rf_liq[rf_liq.target=='P_kbar'].sort_values('rmse_test_mean').iloc[0]

fs_T = rf_T['feature_set']
fs_P = rf_P['feature_set']
print(f'Fair-comparison RF opx-liq: T uses {fs_T}, P uses {fs_P}')


def get_or_train_rf(target, feat_set):
    """Load the saved RF-per-target model, or retrain it from the canonical
    seed-42 best_params if NB03 did not persist that specific combo."""
    fname = f'model_RF_{target}_opx_liq_{feat_set}.joblib'
    fpath = MODELS / fname
    if fpath.exists():
        return joblib.load(fpath)

    multi = pd.read_csv(RESULTS / 'nb03_multi_seed_results.csv')
    row = multi[(multi.track == 'opx_liq') &
                (multi.model_name == 'RF') &
                (multi.feature_set == feat_set) &
                (multi.target == target) &
                (multi.split_seed == 42)].iloc[0]
    try:
        params = ast.literal_eval(row['best_params'])
    except Exception:
        import json as _json
        params = _json.loads(row['best_params'])

    df_full = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
    tr_idx = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
    df_tr = df_full.loc[tr_idx]
    X_tr, _ = build_feature_matrix(df_tr, feat_set, use_liq=True)
    y_tr = df_tr[target].values

    mdl = RandomForestRegressor(**params, random_state=SEED_MODEL, n_jobs=-1)
    mdl.fit(X_tr, y_tr)
    joblib.dump(mdl, fpath)
    print(f'Re-trained and cached {fname}')
    return mdl


model_T_ml = get_or_train_rf('T_C', fs_T)
model_P_ml = get_or_train_rf('P_kbar', fs_P)
