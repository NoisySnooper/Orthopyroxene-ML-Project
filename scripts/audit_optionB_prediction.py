"""Option B pre-flight prediction.

Apply Thermobar's built-in Putirka 2008 Kd Fe-Mg equilibrium filter
(via calculate_opx_liq_press_temp(..., eq_tests=True)) to the current
nb04 Part 3 ArcPL dataset and compute ML vs Putirka RMSE on the
Kd-equilibrated subset."""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import Thermobar as pt
from sklearn.metrics import mean_squared_error
from src.features import cation_recalc_6oxy, add_engineered_features
from config import (LEPR_XLSX, OXIDE_TOTAL_MIN, OXIDE_TOTAL_MAX, CATION_SUM_MIN,
                    CATION_SUM_MAX, WO_MAX_MOL_PCT, P_CEILING_KBAR, FE3_FET_RATIO,
                    KD_FEMG_MIN, KD_FEMG_MAX)

# Rebuild nb04 Part 3 ArcPL n=197
lepr = pd.read_excel(LEPR_XLSX, sheet_name='Opx-Liq')
arcpl = lepr[lepr['Citation_x'].astype(str).str.contains('_notinLEPR', na=False)].copy()
rm = {'Citation_x':'Citation','Experiment_x':'Experiment','P_kbar_x':'P_kbar',
      'SiO2_Opx':'SiO2','TiO2_Opx':'TiO2','Al2O3_Opx':'Al2O3','FeOt_Opx':'FeO_total',
      'MnO_Opx':'MnO','MgO_Opx':'MgO','CaO_Opx':'CaO','Na2O_Opx':'Na2O','K2O_Opx':'K2O',
      'Cr2O3_Opx':'Cr2O3','P2O5_Opx':'P2O5',
      'SiO2_Liq':'liq_SiO2','TiO2_Liq':'liq_TiO2','Al2O3_Liq':'liq_Al2O3','FeOt_Liq':'liq_FeO',
      'MnO_Liq':'liq_MnO','MgO_Liq':'liq_MgO','CaO_Liq':'liq_CaO','Na2O_Liq':'liq_Na2O',
      'K2O_Liq':'liq_K2O','Cr2O3_Liq':'liq_Cr2O3','H2O_Liq':'H2O_Liq'}
arcpl = arcpl.rename(columns=rm)
if 'T_K_x' in arcpl.columns:
    arcpl['T_C'] = arcpl['T_K_x'] - 273.15
arcpl = arcpl.drop(columns=[c for c in arcpl.columns if c.endswith('_y')], errors='ignore')
arcpl = arcpl[arcpl.get('H2O_Liq', pd.Series(np.zeros(len(arcpl)))) >= 0].copy()
all_ox = ['SiO2','TiO2','Al2O3','FeO_total','MnO','MgO','CaO','Na2O','K2O','Cr2O3','P2O5']
for o in all_ox:
    if o in arcpl.columns:
        arcpl[o] = pd.to_numeric(arcpl[o], errors='coerce')
present = [o for o in all_ox if o in arcpl.columns]
arcpl['oxide_total'] = arcpl[present].sum(axis=1, min_count=5)
arcpl = arcpl[arcpl['oxide_total'].between(OXIDE_TOTAL_MIN, OXIDE_TOTAL_MAX)].copy()
arcpl = cation_recalc_6oxy(arcpl, oxides=all_ox)
arcpl = arcpl[arcpl['cation_sum'].between(CATION_SUM_MIN, CATION_SUM_MAX)].copy()
arcpl = arcpl.dropna(subset=['SiO2','Al2O3','FeO_total','MgO','CaO']).copy()
arcpl = add_engineered_features(arcpl)
arcpl['Wo'] = arcpl['Wo_frac']*100.0
arcpl = arcpl[arcpl['Wo']<=WO_MAX_MOL_PCT].copy()
arcpl = arcpl[arcpl['P_kbar']<=P_CEILING_KBAR].copy()
fe_opx = arcpl['FeO_total']/71.844; mg_opx=arcpl['MgO']/40.304
fe_liq = (arcpl['liq_FeO']*(1-FE3_FET_RATIO))/71.844; mg_liq = arcpl['liq_MgO']/40.304
kd_legacy = (fe_opx/mg_opx)/(fe_liq/mg_liq)
arcpl = arcpl[(kd_legacy>=KD_FEMG_MIN) & (kd_legacy<=KD_FEMG_MAX)].copy().reset_index(drop=True)
N_full = len(arcpl)
print(f'nb04 Part 3 ArcPL n_full = {N_full}')

# Build Thermobar frames
def g(col, n=N_full, d=0.0):
    if col in arcpl.columns:
        return pd.to_numeric(arcpl[col], errors='coerce').fillna(d).values.astype(float)
    return np.full(n, d)

opx = pd.DataFrame({'SiO2_Opx':g('SiO2'),'TiO2_Opx':g('TiO2'),'Al2O3_Opx':g('Al2O3'),
    'FeOt_Opx':g('FeO_total'),'MgO_Opx':g('MgO'),'CaO_Opx':g('CaO'),'MnO_Opx':g('MnO'),
    'Cr2O3_Opx':g('Cr2O3'),'Na2O_Opx':g('Na2O')})
liq = pd.DataFrame({'SiO2_Liq':g('liq_SiO2'),'TiO2_Liq':g('liq_TiO2'),'Al2O3_Liq':g('liq_Al2O3'),
    'FeOt_Liq':g('liq_FeO'),'MgO_Liq':g('liq_MgO'),'CaO_Liq':g('liq_CaO'),'MnO_Liq':g('liq_MnO'),
    'K2O_Liq':g('liq_K2O'),'Na2O_Liq':g('liq_Na2O'),'Cr2O3_Liq':g('liq_Cr2O3'),
    'P2O5_Liq':g('P2O5_Liq'),'H2O_Liq':g('H2O_Liq')})
liq['Fe3Fet_Liq'] = 0.0

out = pt.calculate_opx_liq_press_temp(
    opx_comps=opx, liq_comps=liq,
    equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a',
    eq_tests=True,
)

# Thermobar's built-in eq flag: 'Kd Eq (Put2008+-0.06)' -> 'Y'/'N'
eq_col = 'Kd Eq (Put2008+-0.06)'
eq_yn = out[eq_col].astype(str).values
eq_pass = np.array([v.strip().upper().startswith('Y') for v in eq_yn])
T_put = out['T_K_calc'].values - 273.15
P_put = out['P_kbar_calc'].values
conv = np.isfinite(T_put) & np.isfinite(P_put)

kd_fet = out['Kd_Fe_Mg_Fet'].values
ideal = out['Ideal_Kd'].values

print(f'Thermobar Kd Eq (Put2008+-0.06): pass={eq_pass.sum()}, fail={(~eq_pass).sum()}')
print(f'Putirka solver convergence:     converged={conv.sum()}, failed={(~conv).sum()}')
print(f'Eq AND converged (Option B scope): {(eq_pass & conv).sum()}')

# Also report legacy [0.23, 0.35] filter for comparison
legacy_pass = (kd_fet >= 0.23) & (kd_fet <= 0.35)
print(f'Legacy KD_FEMG [0.23,0.35]: pass={legacy_pass.sum()}')

# Load ML forest predictions (from Part 3)
ml = pd.read_csv('results/nb04_arcpl_opx_liq_predictions_forest.csv')
if len(ml) != N_full:
    print(f'WARNING: ML predictions file has n={len(ml)}, ArcPL has n={N_full}')
    common = min(len(ml), N_full)
else:
    common = N_full

ml_T = ml['T_pred'].values[:common]
ml_P = ml['P_pred'].values[:common]
T_obs = arcpl['T_C'].values[:common]
P_obs = arcpl['P_kbar'].values[:common]

def rmse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(mean_squared_error(y[m], yhat[m]))) if m.sum() else float('nan')

eqconv = eq_pass & conv
print()
print('=' * 78)
print('RMSE TABLE')
print('=' * 78)
scopes = [
    ('Full ArcPL (n=all, ML only)', np.ones(common, dtype=bool)),
    ('Eq-filter pass  (Kd Eq Put2008+-0.06 == Y)', eq_pass[:common]),
    ('Solver converged (T/P finite)', conv[:common]),
    ('Eq AND converged (Option B scope)', eqconv[:common]),
]
for label, mask in scopes:
    n = int(mask.sum())
    ml_T_rmse = rmse(T_obs[mask], ml_T[mask])
    ml_P_rmse = rmse(P_obs[mask], ml_P[mask])
    put_T_rmse = rmse(T_obs[mask], T_put[:common][mask]) if n else float('nan')
    put_P_rmse = rmse(P_obs[mask], P_put[:common][mask]) if n else float('nan')
    print(f'  {label}: n={n}')
    print(f'     ML forest:  T RMSE={ml_T_rmse:6.2f}  P RMSE={ml_P_rmse:5.2f}')
    print(f'     Putirka:    T RMSE={put_T_rmse:6.2f}  P RMSE={put_P_rmse:5.2f}')

# Save metrics
pd.DataFrame({
    'scope': ['full', 'eq_pass', 'converged', 'eq_and_conv'],
    'n': [
        int(np.ones(common, dtype=bool).sum()),
        int(eq_pass[:common].sum()),
        int(conv[:common].sum()),
        int(eqconv[:common].sum()),
    ],
    'ML_T_rmse_C': [
        rmse(T_obs, ml_T),
        rmse(T_obs[eq_pass[:common]], ml_T[eq_pass[:common]]),
        rmse(T_obs[conv[:common]], ml_T[conv[:common]]),
        rmse(T_obs[eqconv[:common]], ml_T[eqconv[:common]]),
    ],
    'ML_P_rmse_kbar': [
        rmse(P_obs, ml_P),
        rmse(P_obs[eq_pass[:common]], ml_P[eq_pass[:common]]),
        rmse(P_obs[conv[:common]], ml_P[conv[:common]]),
        rmse(P_obs[eqconv[:common]], ml_P[eqconv[:common]]),
    ],
    'Putirka_T_rmse_C': [
        float('nan'),
        rmse(T_obs[eq_pass[:common]], T_put[:common][eq_pass[:common]]),
        rmse(T_obs[conv[:common]], T_put[:common][conv[:common]]),
        rmse(T_obs[eqconv[:common]], T_put[:common][eqconv[:common]]),
    ],
    'Putirka_P_rmse_kbar': [
        float('nan'),
        rmse(P_obs[eq_pass[:common]], P_put[:common][eq_pass[:common]]),
        rmse(P_obs[conv[:common]], P_put[:common][conv[:common]]),
        rmse(P_obs[eqconv[:common]], P_put[:common][eqconv[:common]]),
    ],
}).to_csv('results/optionB_preflight_metrics.csv', index=False)
print()
print('Saved results/optionB_preflight_metrics.csv')
