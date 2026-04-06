"""
Notebook 03: Baseline Model Training
Opx ML Thermobarometer
Author: [Your name]
Date: 2026-04-04

Input:  opx_clean_core.csv, ExPetDB xlsx (for liquid data)
Output: model_*.joblib (trained models)
        nb03_results.csv, nb03_results_all.csv
        fig03_pred_vs_obs_combined.png
        nb03_test_predictions.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy, joblib, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================
CORE_CSV = 'opx_clean_core.csv'
EXPETDB = 'ExPetDB_download_ExPetDB-2025-07-21.xlsx'  # adjust path

OPX_FEAT = ['SiO2','Al2O3','FeO_total','MgO','CaO',
            'Mg_num','Al_IV','Al_VI','En_frac','Fs_frac','Wo_frac','MgTs']
LIQ_OXIDES = ['SiO2','TiO2','Al2O3','FeO','MgO','CaO','Na2O','K2O']

# Fixed hyperparameters (Jorgenson 2022 showed tuning has <3°C / <0.2 kbar effect)
CONFIGS = {
    'RF':  RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1),
    'ERT': ExtraTreesRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1),
    'XGB': XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0),
    'GB':  GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                                      subsample=0.8, random_state=42),
}


def train_evaluate(X_train, X_test, y_train, y_test, configs, target_name, model_type):
    """Train all models, return results list."""
    unit = '°C' if target_name == 'T_C' else 'kbar'
    results = []
    for name in configs:
        model = copy.deepcopy(configs[name])
        model.fit(X_train, y_train)
        p_tr = model.predict(X_train)
        p_te = model.predict(X_test)
        rmse_tr = np.sqrt(mean_squared_error(y_train, p_tr))
        rmse_te = np.sqrt(mean_squared_error(y_test, p_te))
        mae_te = mean_absolute_error(y_test, p_te)
        r2_te = r2_score(y_test, p_te)
        print(f"  {name:4s}: RMSE={rmse_te:6.1f}{unit}, R²={r2_te:.4f}, "
              f"MAE={mae_te:5.1f}{unit}, train/test={rmse_tr/rmse_te:.3f}")
        results.append(dict(target=target_name, model=name, model_type=model_type,
                           rmse_train=rmse_tr, rmse_test=rmse_te,
                           mae_test=mae_te, r2_test=r2_te))
        joblib.dump(model, f'model_{name}_{target_name}_{model_type}.joblib')
    return results


# ============================================================
# PART 1: OPX-ONLY MODELS
# ============================================================
print("=" * 60)
print("PART 1: OPX-ONLY MODELS")
print("=" * 60)

df = pd.read_csv(CORE_CSV)
X = df[OPX_FEAT].values
y_T, y_P = df['T_C'].values, df['P_kbar'].values
p_bins = pd.qcut(y_P, q=5, labels=False, duplicates='drop')

Xtr, Xte, yTtr, yTte, yPtr, yPte, idx_tr, idx_te = \
    train_test_split(X, y_T, y_P, df.index.values, test_size=0.2, random_state=42, stratify=p_bins)

print(f"Train: {len(Xtr)}, Test: {len(Xte)}\n")

all_results = []
print("--- Temperature ---")
all_results += train_evaluate(Xtr, Xte, yTtr, yTte, CONFIGS, 'T_C', 'opx_only')
print("\n--- Pressure ---")
all_results += train_evaluate(Xtr, Xte, yPtr, yPte, CONFIGS, 'P_kbar', 'opx_only')

# ============================================================
# PART 2: OPX-LIQUID MODELS
# ============================================================
print("\n" + "=" * 60)
print("PART 2: OPX-LIQUID MODELS")
print("=" * 60)

liq_raw = pd.read_excel(EXPETDB, sheet_name='Liquid')
liq = liq_raw[['Experiment'] + [f'{o} value' for o in LIQ_OXIDES]].copy()
liq.columns = ['Experiment'] + [f'liq_{o}' for o in LIQ_OXIDES]
for c in [f'liq_{o}' for o in LIQ_OXIDES]:
    liq[c] = pd.to_numeric(liq[c], errors='coerce')
liq = liq.drop_duplicates(subset='Experiment', keep='first')

df_liq = df.merge(liq, on='Experiment', how='inner')
mw_MgO, mw_FeO = 40.304, 71.844
df_liq['liq_Mg_num'] = (df_liq['liq_MgO']/mw_MgO) / \
                        (df_liq['liq_MgO']/mw_MgO + df_liq['liq_FeO']/mw_FeO)
df_liq = df_liq.dropna(subset=['liq_SiO2','liq_Al2O3','liq_FeO','liq_MgO','liq_CaO','liq_Mg_num'])

liq_feat = [f'liq_{o}' for o in LIQ_OXIDES] + ['liq_Mg_num']
ALL_FEAT = OPX_FEAT + liq_feat
X2 = df_liq[ALL_FEAT].fillna(0).values
y2T, y2P = df_liq['T_C'].values, df_liq['P_kbar'].values
p2_bins = pd.qcut(y2P, q=5, labels=False, duplicates='drop')

X2tr, X2te, y2Ttr, y2Tte, y2Ptr, y2Pte = \
    train_test_split(X2, y2T, y2P, test_size=0.2, random_state=42, stratify=p2_bins)

print(f"Opx-liq pairs: {len(df_liq)}, Train: {len(X2tr)}, Test: {len(X2te)}\n")

print("--- Temperature ---")
all_results += train_evaluate(X2tr, X2te, y2Ttr, y2Tte, CONFIGS, 'T_C', 'opx_liq')
print("\n--- Pressure ---")
all_results += train_evaluate(X2tr, X2te, y2Ptr, y2Pte, CONFIGS, 'P_kbar', 'opx_liq')

# ============================================================
# SAVE ALL RESULTS
# ============================================================
pd.DataFrame(all_results).to_csv('nb03_results_all.csv', index=False)
np.save('test_indices.npy', idx_te)
np.save('train_indices.npy', idx_tr)
print("\nAll results saved.")
