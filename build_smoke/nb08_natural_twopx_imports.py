import os
os.chdir(r"C:\Users\NQTa\Documents\MLCourse\Final Project\notebooks")
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import warnings
warnings.filterwarnings('ignore')

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ROOT, MODELS, RESULTS, FIGURES, LEPR_XLSX, SEED_MODEL
from src.features import build_feature_matrix
from src.plot_style import (
    apply_style, OKABE_ITO, PUTIRKA_COLOR, ML_COLOR, load_winning_config,
    canonical_model_filename,
)
from src.external_models import predict_jorgenson

apply_style()
print('NB08 v6 imports OK')

# Section 3. Cross-mineral agreement + disequilibrium index.
# Primary quantity: DeltaP_(opx - cpx) in kbar.
from scipy import stats

# Sigma from conformal calibration (v5 NB07 JSON)
with open(RESULTS / 'nb07_conformal_qhat.json') as f:
    qhat = json.load(f)
sigma_T = qhat.get('T_C',    {}).get('qhat', 50.0)
sigma_P = qhat.get('P_kbar', {}).get('qhat',  3.0)
print(f'Conformal half-widths: sigma_T={sigma_T:.1f} C, sigma_P={sigma_P:.2f} kbar')

delta_T_lepr = T_ml_lepr - T_jor_lepr
delta_P_lepr = P_ml_lepr - P_jor_lepr
equilib_mask_lepr = (np.abs(delta_T_lepr) < 2 * sigma_T) & (np.abs(delta_P_lepr) < 2 * sigma_P)
print(f'LEPR cross-mineral "equilibrium pair" fraction '
      f'(|dT|<2s_T AND |dP|<2s_P): {equilib_mask_lepr.mean():.3f}')

rows = []
for name, vT, vP, yT, yP in [
    ('ML opx-only (ours)',  T_ml_lepr,  P_ml_lepr,  lepr['T_C'].values, lepr['P_kbar'].values),
    ('Jorgenson cpx-only',  T_jor_lepr, P_jor_lepr, lepr['T_C'].values, lepr['P_kbar'].values),
    ('Putirka 2-px eq36/39',T_put_lepr, P_put_lepr, lepr['T_C'].values, lepr['P_kbar'].values),
]:
    mT = np.isfinite(yT) & np.isfinite(vT)
    mP = np.isfinite(yP) & np.isfinite(vP)
    rows.append({
        'Method': name,
        'n_T':    int(mT.sum()),
        'T_RMSE': float(np.sqrt(np.mean((yT[mT] - vT[mT]) ** 2))),
        'T_bias': float(np.mean(vT[mT] - yT[mT])),
        'n_P':    int(mP.sum()),
        'P_RMSE': float(np.sqrt(np.mean((yP[mP] - vP[mP]) ** 2))),
        'P_bias': float(np.mean(vP[mP] - yP[mP])),
    })
cross = pd.DataFrame(rows)
cross.to_csv(RESULTS / 'nb08_cross_mineral_agreement.csv', index=False)
print('\nLEPR vs experimental ground truth (cross-mineral agreement):')
print(cross.round(2).to_string(index=False))

# Save per-sample predictions.
pred = pd.DataFrame({
    'Experiment':          lepr['Experiment'].astype(str).values,
    'T_C_true':            lepr['T_C'].values,
    'P_kbar_true':         lepr['P_kbar'].values,
    'T_ml_opx':            T_ml_lepr,
    'P_ml_opx':            P_ml_lepr,
    'T_jorgenson_cpx':     T_jor_lepr,
    'P_jorgenson_cpx':     P_jor_lepr,
    'T_putirka_2px_eq36':  T_put_lepr,
    'P_putirka_2px_eq39':  P_put_lepr,
    'delta_T_opx_minus_cpx': delta_T_lepr,
    'delta_P_opx_minus_cpx': delta_P_lepr,
    'equilibrium_pair_flag': equilib_mask_lepr,
})
pred.to_csv(RESULTS / 'nb08_natural_predictions.csv', index=False)
print(f'\nSaved nb08_natural_predictions.csv (n={len(pred)})')
