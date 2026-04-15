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
from src.plot_style import load_winning_config, canonical_model_filename
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Canonical features and prediction helpers from src/ (one source of truth).
from src.features import (
    build_feature_matrix,
    make_raw_features,
    make_alr_features,
    make_pwlr_features,
    augment_dataframe,
)
from src.models import predict_median, predict_iqr
from src.evaluation import compute_metrics as metrics
from src.plot_style import apply_style
apply_style()  # Okabe-Ito colorblind-safe palette, 300 dpi PNG+PDF defaults

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