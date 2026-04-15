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
)
from src.plot_style import load_winning_config
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

config_3r = load_winning_config(RESULTS)
WIN_FEAT = config_3r['global_feature_set']
print(f'Phase 3R winning feature set: {WIN_FEAT}')
# Phase 9R.EXT setup: extra imports + canonical model/data loads for the
# absorbed NB10 analyses. NB09's earlier imports supply ROOT, DATA_PROC,
# DATA_SPLITS, MODELS, FIGURES, RESULTS, WIN_FEAT, numpy as np, pandas as pd.

from src.features import build_feature_matrix
from src.models import predict_median, predict_iqr
from src.evaluation import compute_metrics as metrics
from src.plot_style import canonical_model_filename, apply_style
apply_style()

import joblib
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

feat_fn = lambda df, use_liq: build_feature_matrix(df, WIN_FEAT, use_liq=use_liq)
model_T = joblib.load(MODELS / canonical_model_filename('RF', 'T_C', 'opx_liq', RESULTS))
model_P = joblib.load(MODELS / canonical_model_filename('RF', 'P_kbar', 'opx_liq', RESULTS))
qrf_T_path = MODELS / 'model_QRF_T_C_opx_liq.joblib'
qrf_P_path = MODELS / 'model_QRF_P_kbar_opx_liq.joblib'
qrf_T = joblib.load(qrf_T_path) if qrf_T_path.exists() else None
qrf_P = joblib.load(qrf_P_path) if qrf_P_path.exists() else None

df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
idx_tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
idx_te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')
df_train = df_liq.loc[idx_tr].copy()
df_test  = df_liq.loc[idx_te].copy()
X_train, feat_names = feat_fn(df_train, use_liq=True)
X_test,  _          = feat_fn(df_test,  use_liq=True)

arcpl_path = RESULTS / 'nb04b_arcpl_predictions.csv'
if not arcpl_path.exists():
    raise FileNotFoundError(f'Missing {arcpl_path}. Run nb04b first.')
df_arcpl = pd.read_csv(arcpl_path)
print(f'Ext setup OK: train n={len(df_train)} test n={len(df_test)} ArcPL n={len(df_arcpl)}')

# Phase 10.1: Two-pyroxene benchmark via Thermobar on the opx-liq test set
# Evaluates Putirka (2008) Eq. 36/39 two-pyroxene-only T (and P where available)
# as a non-ML baseline that can be directly compared against our opx-liq RF.

try:
    import Thermobar as pt
    HAVE_THERMOBAR = True
except Exception as e:
    HAVE_THERMOBAR = False
    print(f'Thermobar unavailable: {e}')

two_px_rows = []

if HAVE_THERMOBAR and {'T_C', 'P_kbar'}.issubset(df_test.columns):
    # Build Thermobar-style input frames. The two-pyroxene thermometer needs
    # both Opx and Cpx compositions; samples without Cpx are skipped.
    opx_cols = ['SiO2', 'TiO2', 'Al2O3', 'FeO_total', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    cpx_prefix = 'cpx_'
    has_cpx = any(c.startswith(cpx_prefix) for c in df_test.columns)
    if has_cpx:
        mask = df_test[[f'{cpx_prefix}SiO2']].notna().all(axis=1).values
        sub = df_test[mask].copy()
        opx_df = pd.DataFrame({f'{c}_Opx': sub[c].values for c in opx_cols if c in sub.columns})
        cpx_df = pd.DataFrame({f'{c}_Cpx': sub[f'{cpx_prefix}{c}'].values
                               for c in opx_cols if f'{cpx_prefix}{c}' in sub.columns})

        try:
            out = pt.calculate_cpx_opx_press_temp(
                cpx_comps=cpx_df, opx_comps=opx_df,
                equationT='T_Put2008_eq36', equationP='P_Put2008_eq39')
            t_pred = out['T_K_calc'].values - 273.15
            p_pred = out['P_kbar_calc'].values if 'P_kbar_calc' in out.columns else np.full(len(sub), np.nan)
            t_obs = sub['T_C'].values
            p_obs = sub['P_kbar'].values

            two_px_rows.append({'target': 'T_C', **metrics(t_obs[np.isfinite(t_pred)],
                                                           t_pred[np.isfinite(t_pred)])})
            mask_p = np.isfinite(p_pred)
            if mask_p.sum() > 0:
                two_px_rows.append({'target': 'P_kbar', **metrics(p_obs[mask_p], p_pred[mask_p])})
        except Exception as e:
            print(f'Thermobar call failed: {e}')
    else:
        print('Test set has no Cpx columns; two-pyroxene benchmark skipped.')
else:
    print('Skipped two-pyroxene benchmark (no Thermobar or missing cols).')

two_px_df = pd.DataFrame(two_px_rows)
two_px_df.to_csv(RESULTS / 'nb10_two_pyroxene_benchmark.csv', index=False)
if len(two_px_df):
    print(two_px_df.round(3).to_string(index=False))