"""Side-by-side comparison: Path A (nb04b cell 19) vs Path B (nb04 cell 27)
Putirka 28a/29a code paths on the same ArcPL samples."""
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

print(f"Thermobar version: {pt.__version__}")

lepr = pd.read_excel(LEPR_XLSX, sheet_name='Opx-Liq')
arcpl = lepr[lepr['Citation_x'].astype(str).str.contains('_notinLEPR', na=False)].copy()
rename_map = {
    'Citation_x':'Citation','Experiment_x':'Experiment','P_kbar_x':'P_kbar',
    'SiO2_Opx':'SiO2','TiO2_Opx':'TiO2','Al2O3_Opx':'Al2O3','FeOt_Opx':'FeO_total',
    'MnO_Opx':'MnO','MgO_Opx':'MgO','CaO_Opx':'CaO','Na2O_Opx':'Na2O','K2O_Opx':'K2O',
    'Cr2O3_Opx':'Cr2O3','P2O5_Opx':'P2O5',
    'SiO2_Liq':'liq_SiO2','TiO2_Liq':'liq_TiO2','Al2O3_Liq':'liq_Al2O3','FeOt_Liq':'liq_FeO',
    'MnO_Liq':'liq_MnO','MgO_Liq':'liq_MgO','CaO_Liq':'liq_CaO','Na2O_Liq':'liq_Na2O',
    'K2O_Liq':'liq_K2O','Cr2O3_Liq':'liq_Cr2O3','H2O_Liq':'H2O_Liq',
}
arcpl = arcpl.rename(columns=rename_map)
if 'T_K_x' in arcpl.columns:
    arcpl['T_C'] = arcpl['T_K_x'] - 273.15
arcpl = arcpl.drop(columns=[c for c in arcpl.columns if c.endswith('_y')], errors='ignore')
arcpl = arcpl[arcpl.get('H2O_Liq', pd.Series(np.zeros(len(arcpl)))) >= 0].copy()
all_ox = ['SiO2','TiO2','Al2O3','FeO_total','MnO','MgO','CaO','Na2O','K2O','Cr2O3','P2O5']
for _ox in all_ox:
    if _ox in arcpl.columns:
        arcpl[_ox] = pd.to_numeric(arcpl[_ox], errors='coerce')
present = [o for o in all_ox if o in arcpl.columns]
arcpl['oxide_total'] = arcpl[present].sum(axis=1, min_count=5)
arcpl = arcpl[arcpl['oxide_total'].between(OXIDE_TOTAL_MIN, OXIDE_TOTAL_MAX)].copy()
arcpl = cation_recalc_6oxy(arcpl, oxides=all_ox)
arcpl = arcpl[arcpl['cation_sum'].between(CATION_SUM_MIN, CATION_SUM_MAX)].copy()
arcpl = arcpl.dropna(subset=['SiO2','Al2O3','FeO_total','MgO','CaO']).copy()
arcpl = add_engineered_features(arcpl)
arcpl['Wo'] = arcpl['Wo_frac'] * 100.0
arcpl = arcpl[arcpl['Wo'] <= WO_MAX_MOL_PCT].copy()
arcpl = arcpl[arcpl['P_kbar'] <= P_CEILING_KBAR].copy()
fe_opx = arcpl['FeO_total']/71.844
mg_opx = arcpl['MgO']/40.304
fe_liq = (arcpl['liq_FeO']*(1-FE3_FET_RATIO))/71.844
mg_liq = arcpl['liq_MgO']/40.304
kd = (fe_opx/mg_opx)/(fe_liq/mg_liq)
arcpl = arcpl[(kd >= KD_FEMG_MIN) & (kd <= KD_FEMG_MAX)].copy().reset_index(drop=True)
print(f"Full ArcPL n = {len(arcpl)}")

rng = np.random.default_rng(42)
idx = rng.choice(len(arcpl), 5, replace=False)
samp = arcpl.iloc[idx].reset_index(drop=True)
print(f"Sampled row indices: {list(idx)}")


def build_frames(df):
    def g(col, default=0.0):
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').fillna(default).values.astype(float)
        return np.full(len(df), default, dtype=float)
    opx = pd.DataFrame({
        'SiO2_Opx': g('SiO2'), 'TiO2_Opx': g('TiO2'), 'Al2O3_Opx': g('Al2O3'),
        'FeOt_Opx': g('FeO_total'), 'MgO_Opx': g('MgO'), 'CaO_Opx': g('CaO'),
        'MnO_Opx': g('MnO'), 'Cr2O3_Opx': g('Cr2O3'), 'Na2O_Opx': g('Na2O'),
    })
    liq = pd.DataFrame({
        'SiO2_Liq': g('liq_SiO2'), 'TiO2_Liq': g('liq_TiO2'), 'Al2O3_Liq': g('liq_Al2O3'),
        'FeOt_Liq': g('liq_FeO'), 'MgO_Liq': g('liq_MgO'), 'CaO_Liq': g('liq_CaO'),
        'MnO_Liq': g('liq_MnO'), 'K2O_Liq': g('liq_K2O'), 'Na2O_Liq': g('liq_Na2O'),
        'Cr2O3_Liq': g('liq_Cr2O3'), 'P2O5_Liq': g('P2O5_Liq'), 'H2O_Liq': g('H2O_Liq'),
    })
    return opx, liq


def clip_arr(a, lo, hi):
    a = np.asarray(a, dtype=float).copy()
    a[(a < lo) | (a > hi)] = np.nan
    return a


opx5, liq5 = build_frames(samp)
yT, yP = samp['T_C'].values, samp['P_kbar'].values
h2o = samp['H2O_Liq'].fillna(0).values.astype(float)

# Path A: nb04b cell 19
fe3_A = np.full(5, 0.15)
tA = pt.calculate_opx_liq_temp(equationT='T_Put2008_eq28a', opx_comps=opx5,
                                liq_comps=liq5, P=yP, Fe3Fet_Liq=fe3_A, H2O_Liq=h2o)
tA = np.asarray(tA).astype(float) - 273.15
tA_clip = clip_arr(tA, 0, 2000)
pA = pt.calculate_opx_liq_press(equationP='P_Put2008_eq29a', opx_comps=opx5,
                                 liq_comps=liq5, T=yT+273.15, Fe3Fet_Liq=fe3_A,
                                 H2O_Liq=h2o)
pA = np.asarray(pA).astype(float)
pA_clip = clip_arr(pA, -10, 100)

# Path B: nb04 cell 27
liq5_B = liq5.copy()
liq5_B['Fe3Fet_Liq'] = 0.0
out = pt.calculate_opx_liq_press_temp(opx_comps=opx5, liq_comps=liq5_B,
                                       equationP='P_Put2008_eq29a',
                                       equationT='T_Put2008_eq28a')
tB = np.asarray(out['T_K_calc']).astype(float) - 273.15
pB = np.asarray(out['P_kbar_calc']).astype(float)

print()
print('=' * 100)
print(f'{"Row":>3} | {"T_obs":>7} {"P_obs":>7} | {"T_A_raw":>8} {"T_A_clip":>9} {"P_A_raw":>8} {"P_A_clip":>9} | {"T_B":>7} {"P_B":>7}')
print('-' * 100)
for i in range(5):
    print(f'{i:>3} | {yT[i]:>7.1f} {yP[i]:>7.2f} | '
          f'{tA[i]:>8.1f} {tA_clip[i]:>9.1f} {pA[i]:>8.2f} {pA_clip[i]:>9.2f} | '
          f'{tB[i]:>7.1f} {pB[i]:>7.2f}')

# Same rows but Fe3=0.0 under Path A to isolate Fe3 effect
fe3_0 = np.full(5, 0.0)
tAs = pt.calculate_opx_liq_temp(equationT='T_Put2008_eq28a', opx_comps=opx5,
                                 liq_comps=liq5, P=yP, Fe3Fet_Liq=fe3_0, H2O_Liq=h2o)
tAs = np.asarray(tAs).astype(float) - 273.15
pAs = pt.calculate_opx_liq_press(equationP='P_Put2008_eq29a', opx_comps=opx5,
                                  liq_comps=liq5, T=yT+273.15, Fe3Fet_Liq=fe3_0,
                                  H2O_Liq=h2o)
pAs = np.asarray(pAs).astype(float)

print()
print(f'--- Path A* (Path A recipe but Fe3Fet=0.0 and no clipping) vs Path B ---')
print(f'{"Row":>3} | {"T_A*(Fe3=0)":>12} {"T_B":>7} {"dT":>7} | {"P_A*(Fe3=0)":>12} {"P_B":>7} {"dP":>7}')
for i in range(5):
    print(f'{i:>3} | {tAs[i]:>12.2f} {tB[i]:>7.2f} {tAs[i]-tB[i]:>7.2f} | '
          f'{pAs[i]:>12.3f} {pB[i]:>7.3f} {pAs[i]-pB[i]:>7.3f}')

# Full-set convergence + RMSE under each recipe
print()
print('=' * 80)
print('FULL ArcPL (n=197) convergence + RMSE under each recipe')
print('=' * 80)
opx_full, liq_full = build_frames(arcpl)
h2o_full = arcpl['H2O_Liq'].fillna(0).values.astype(float)

fe3_full = np.full(len(arcpl), 0.15)
tA_full = np.asarray(pt.calculate_opx_liq_temp(equationT='T_Put2008_eq28a',
    opx_comps=opx_full, liq_comps=liq_full, P=arcpl['P_kbar'].values,
    Fe3Fet_Liq=fe3_full, H2O_Liq=h2o_full)).astype(float) - 273.15
pA_full = np.asarray(pt.calculate_opx_liq_press(equationP='P_Put2008_eq29a',
    opx_comps=opx_full, liq_comps=liq_full, T=arcpl['T_C'].values+273.15,
    Fe3Fet_Liq=fe3_full, H2O_Liq=h2o_full)).astype(float)
tA_full_clip = clip_arr(tA_full, 0, 2000)
pA_full_clip = clip_arr(pA_full, -10, 100)
okA = np.isfinite(tA_full_clip) & np.isfinite(pA_full_clip)
rmseT_A = np.sqrt(mean_squared_error(arcpl['T_C'].values[okA], tA_full_clip[okA]))
rmseP_A = np.sqrt(mean_squared_error(arcpl['P_kbar'].values[okA], pA_full_clip[okA]))
print(f'PATH A (1-sided, Fe3=0.15, clip): fair n={okA.sum()}/{len(arcpl)} '
      f'({100*okA.sum()/len(arcpl):.1f}%)  T RMSE={rmseT_A:.1f} C  P RMSE={rmseP_A:.2f} kbar')

liq_full_B = liq_full.copy()
liq_full_B['Fe3Fet_Liq'] = 0.0
out_full = pt.calculate_opx_liq_press_temp(opx_comps=opx_full, liq_comps=liq_full_B,
                                            equationP='P_Put2008_eq29a',
                                            equationT='T_Put2008_eq28a')
tB_full = np.asarray(out_full['T_K_calc']).astype(float) - 273.15
pB_full = np.asarray(out_full['P_kbar_calc']).astype(float)
okB = np.isfinite(tB_full) & np.isfinite(pB_full)
rmseT_B = np.sqrt(mean_squared_error(arcpl['T_C'].values[okB], tB_full[okB]))
rmseP_B = np.sqrt(mean_squared_error(arcpl['P_kbar'].values[okB], pB_full[okB]))
print(f'PATH B (iterative, Fe3=0.0, no clip):   fair n={okB.sum()}/{len(arcpl)} '
      f'({100*okB.sum()/len(arcpl):.1f}%)  T RMSE={rmseT_B:.1f} C  P RMSE={rmseP_B:.2f} kbar')

both = okA & okB
rmseTA_i = np.sqrt(mean_squared_error(arcpl['T_C'].values[both], tA_full_clip[both]))
rmsePA_i = np.sqrt(mean_squared_error(arcpl['P_kbar'].values[both], pA_full_clip[both]))
rmseTB_i = np.sqrt(mean_squared_error(arcpl['T_C'].values[both], tB_full[both]))
rmsePB_i = np.sqrt(mean_squared_error(arcpl['P_kbar'].values[both], pB_full[both]))
print()
print(f'Intersection (both converge): n={both.sum()}')
print(f'  A on intersection: T RMSE={rmseTA_i:.1f}   P RMSE={rmsePA_i:.2f}')
print(f'  B on intersection: T RMSE={rmseTB_i:.1f}   P RMSE={rmsePB_i:.2f}')
print(f'  A converges exclusively (not B): n={(okA & ~okB).sum()}')
print(f'  B converges exclusively (not A): n={(okB & ~okA).sum()}')

# save metrics for report
pd.DataFrame({
    'metric': ['fair_n', 'fair_pct', 'T_rmse_fair', 'P_rmse_fair',
               'T_rmse_intersection', 'P_rmse_intersection'],
    'path_A_1sided_Fe3_0.15_clip': [okA.sum(), 100*okA.sum()/len(arcpl),
                                     rmseT_A, rmseP_A, rmseTA_i, rmsePA_i],
    'path_B_iter_Fe3_0.0_noclip':  [okB.sum(), 100*okB.sum()/len(arcpl),
                                     rmseT_B, rmseP_B, rmseTB_i, rmsePB_i],
}).to_csv('results/putirka_path_A_vs_B_metrics.csv', index=False)
print('\nSaved results/putirka_path_A_vs_B_metrics.csv')
