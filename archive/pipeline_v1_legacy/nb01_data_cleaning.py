"""
Notebook 01: Data Cleaning & Feature Engineering
Opx ML Thermobarometer
Author: [Your name]
Date: 2026-04-04

Input:  ExPetDB_download_ExPetDB-2025-07-21.xlsx
Output: opx_clean_core.csv  (5-oxide model, ~1,176 rows)
        opx_clean_full.csv  (9-oxide model, ~526 rows)
        cleaning_log.txt    (provenance of every drop)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
INFILE = PROJECT_DIR / 'ExPetDB_download_ExPetDB-2025-07-21.xlsx'
OUTDIR = PROJECT_DIR

OXIDE_TOTAL_MIN = 95.0
OXIDE_TOTAL_MAX = 102.0
CATION_SUM_MIN = 3.95
CATION_SUM_MAX = 4.05
WO_MAX_MOL_PCT = 5.0  # pigeonite filter

# Molecular weights, cations per formula unit, oxygens per formula unit
OXIDE_INFO: dict[str, tuple[float, int, int]] = {
    'SiO2':      (60.084,  1, 2),
    'TiO2':      (79.866,  1, 2),
    'Al2O3':     (101.961, 2, 3),
    'Cr2O3':     (151.990, 2, 3),
    'FeO_total': (71.844,  1, 1),
    'MnO':       (70.937,  1, 1),
    'MgO':       (40.304,  1, 1),
    'CaO':       (56.077,  1, 1),
    'Na2O':      (61.979,  2, 1),
}

CATION_NAMES: dict[str, str] = {
    'SiO2': 'Si', 'TiO2': 'Ti', 'Al2O3': 'Al', 'Cr2O3': 'Cr',
    'FeO_total': 'Fe', 'MnO': 'Mn', 'MgO': 'Mg', 'CaO': 'Ca', 'Na2O': 'Na',
}

RAW_OXIDE_CORE = ['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO']
RAW_OXIDE_FULL = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO', 'MgO', 'CaO', 'Na2O']

log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg)
    log_lines.append(msg)


# ============================================================
# STEP 1: EXTRACT AND MERGE
# ============================================================
log("STEP 1: Extract and merge")

exp = pd.read_excel(INFILE, sheet_name='Experiment')
opx = pd.read_excel(INFILE, sheet_name='Orthopyroxene')

exp_dedup = exp.drop_duplicates(subset='Experiment', keep='first')
merge_cols = ['Experiment', 'T (C)', 'P (GPa)', 'Device', 'Container', 'Rock type', 'Type']
df = opx.merge(exp_dedup[merge_cols], on='Experiment', how='left')

df['P_kbar'] = df['P (GPa)'] * 10.0
df['T_C'] = df['T (C)']

n0 = len(df)
df = df.dropna(subset=['T_C', 'P_kbar'])
log(f"  Raw opx rows: {n0}")
log(f"  After dropping missing P/T: {len(df)} (dropped {n0 - len(df)})")

# ============================================================
# STEP 2: EXTRACT OXIDES, HANDLE IRON
# ============================================================
log("\nSTEP 2: Extract oxides, handle iron")

oxide_value_cols = {
    'SiO2': 'SiO2 value', 'TiO2': 'TiO2 value', 'Al2O3': 'Al2O3 value',
    'Cr2O3': 'Cr2O3 value', 'FeO': 'FeO value', 'Fe2O3': 'Fe2O3 value',
    'MnO': 'MnO value', 'MgO': 'MgO value', 'CaO': 'CaO value', 'Na2O': 'Na2O value',
}
for short, full in oxide_value_cols.items():
    df[short] = pd.to_numeric(df[full], errors='coerce')

# FeO_total = FeO + Fe2O3 * 0.8998
fe2o3_n = df['Fe2O3'].notna().sum()
df['FeO_total'] = df['FeO'].fillna(0) + df['Fe2O3'].fillna(0) * 0.8998
both_fe_missing = df['FeO'].isna() & df['Fe2O3'].isna()
df.loc[both_fe_missing, 'FeO_total'] = np.nan
log(f"  Fe2O3 reported in {fe2o3_n} rows; all Fe treated as FeO_total")

# ============================================================
# STEP 3: OXIDE TOTAL FILTER
# ============================================================
log(f"\nSTEP 3: Oxide total filter ({OXIDE_TOTAL_MIN}-{OXIDE_TOTAL_MAX} wt%)")

df['oxide_total'] = df[RAW_OXIDE_FULL].sum(axis=1, min_count=3)
n_before = len(df)
mask_total = (df['oxide_total'] >= OXIDE_TOTAL_MIN) & (df['oxide_total'] <= OXIDE_TOTAL_MAX)
dropped_total = df[~mask_total & df['oxide_total'].notna()]
df = df[mask_total].copy()
log(f"  Dropped {n_before - len(df)} rows (below {OXIDE_TOTAL_MIN}: {(dropped_total['oxide_total'] < OXIDE_TOTAL_MIN).sum()}, "
    f"above {OXIDE_TOTAL_MAX}: {(dropped_total['oxide_total'] > OXIDE_TOTAL_MAX).sum()}, "
    f"null: {(~mask_total & df['oxide_total'].isna() if 'oxide_total' in df.columns else pd.Series(dtype=bool)).sum()})")
log(f"  Remaining: {len(df)}")

# ============================================================
# STEP 4: CATION RECALCULATION (6-oxygen basis, Morimoto 1988)
# ============================================================
log("\nSTEP 4: Cation recalculation (6-oxygen basis)")

for ox, (mw, ncat, nox) in OXIDE_INFO.items():
    df[f'{ox}_moles'] = df[ox].fillna(0) / mw
    df[f'{ox}_oxy'] = df[f'{ox}_moles'] * nox

df['total_oxy'] = sum(df[f'{ox}_oxy'] for ox in OXIDE_INFO)
df['norm_factor'] = 6.0 / df['total_oxy']

for ox, (mw, ncat, nox) in OXIDE_INFO.items():
    cat = CATION_NAMES[ox]
    df[f'cat_{cat}'] = df[f'{ox}_moles'] * ncat * df['norm_factor']

cat_cols = [f'cat_{c}' for c in CATION_NAMES.values()]
df['cation_sum'] = df[cat_cols].sum(axis=1)
log(f"  Cation sum: mean={df['cation_sum'].mean():.4f}, "
    f"std={df['cation_sum'].std():.4f}, "
    f"range=[{df['cation_sum'].min():.4f}, {df['cation_sum'].max():.4f}]")

# ============================================================
# STEP 5: REQUIRE CORE OXIDES + CATION SUM FILTER
# ============================================================
log(f"\nSTEP 5: Core oxide requirement + cation sum filter ({CATION_SUM_MIN}-{CATION_SUM_MAX})")

n_before = len(df)
core_present = df[['SiO2', 'MgO', 'FeO_total']].notna().all(axis=1)
df = df[core_present].copy()
log(f"  After requiring SiO2+MgO+FeO: {len(df)} (dropped {n_before - len(df)})")

n_before = len(df)
df = df[(df['cation_sum'] >= CATION_SUM_MIN) & (df['cation_sum'] <= CATION_SUM_MAX)].copy()
log(f"  After cation sum filter: {len(df)} (dropped {n_before - len(df)})")

# ============================================================
# STEP 6: PIGEONITE FILTER
# ============================================================
log(f"\nSTEP 6: Pigeonite filter (Wo <= {WO_MAX_MOL_PCT} mol%)")

df['sum_MgFeCa'] = df['cat_Mg'] + df['cat_Fe'] + df['cat_Ca']
df['En'] = df['cat_Mg'] / df['sum_MgFeCa'] * 100
df['Fs'] = df['cat_Fe'] / df['sum_MgFeCa'] * 100
df['Wo'] = df['cat_Ca'] / df['sum_MgFeCa'] * 100

n_pigeonite = (df['Wo'] > WO_MAX_MOL_PCT).sum()
n_before = len(df)
df = df[df['Wo'] <= WO_MAX_MOL_PCT].copy()
log(f"  Dropped {n_before - len(df)} analyses with Wo > {WO_MAX_MOL_PCT}%")
log(f"  Remaining: {len(df)}")

# ============================================================
# STEP 7: FEATURE ENGINEERING
# ============================================================
log("\nSTEP 7: Feature engineering")

df['Mg_num'] = df['cat_Mg'] / (df['cat_Mg'] + df['cat_Fe'])
df['Al_IV'] = np.clip(2.0 - df['cat_Si'], 0, df['cat_Al'])
df['Al_VI'] = df['cat_Al'] - df['Al_IV']
df['MgTs'] = df['Al_IV']
df['En_frac'] = df['cat_Mg'] / df['sum_MgFeCa']
df['Fs_frac'] = df['cat_Fe'] / df['sum_MgFeCa']
df['Wo_frac'] = df['cat_Ca'] / df['sum_MgFeCa']

log(f"  Mg#:   [{df['Mg_num'].min():.3f}, {df['Mg_num'].max():.3f}], median={df['Mg_num'].median():.3f}")
log(f"  Al_IV: [{df['Al_IV'].min():.4f}, {df['Al_IV'].max():.4f}], median={df['Al_IV'].median():.4f}")
log(f"  Al_VI: [{df['Al_VI'].min():.4f}, {df['Al_VI'].max():.4f}], median={df['Al_VI'].median():.4f}")

# ============================================================
# STEP 8: SPLIT INTO CORE AND FULL DATASETS, SAVE
# ============================================================
log("\nSTEP 8: Save datasets")

prov_cols = ['Experiment', 'Citation', 'DOI']
if 'DOI' not in df.columns:
    df['DOI'] = np.nan
target_cols = ['T_C', 'P_kbar']
cation_feature_cols = ['cat_Si', 'cat_Ti', 'cat_Al', 'cat_Cr', 'cat_Fe', 'cat_Mn', 'cat_Mg', 'cat_Ca', 'cat_Na']
eng_cols = ['Mg_num', 'Al_IV', 'Al_VI', 'En_frac', 'Fs_frac', 'Wo_frac', 'MgTs']
meta_cols = ['oxide_total', 'cation_sum', 'Device', 'Rock type']

output_cols = prov_cols + target_cols + RAW_OXIDE_FULL + cation_feature_cols + eng_cols + meta_cols
output_cols = [c for c in output_cols if c in df.columns]

# Core: 5 oxides complete
df_core = df[df[RAW_OXIDE_CORE].notna().all(axis=1)].copy()
# Full: all 9 oxides complete
df_full = df[df[RAW_OXIDE_FULL].notna().all(axis=1)].copy()

df_core[output_cols].to_csv(OUTDIR / 'opx_clean_core.csv', index=False)
df_full[output_cols].to_csv(OUTDIR / 'opx_clean_full.csv', index=False)

log(f"  Core dataset: {len(df_core)} rows, {df_core['Experiment'].nunique()} experiments, {df_core['Citation'].nunique()} citations")
log(f"  Full dataset: {len(df_full)} rows, {df_full['Experiment'].nunique()} experiments, {df_full['Citation'].nunique()} citations")
log(f"  Core P range: {df_core['P_kbar'].min():.1f}-{df_core['P_kbar'].max():.1f} kbar")
log(f"  Core T range: {df_core['T_C'].min():.0f}-{df_core['T_C'].max():.0f} C")

# Save cleaning log
with open(OUTDIR / 'cleaning_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))

log("\nDone. Files saved to " + str(OUTDIR))
