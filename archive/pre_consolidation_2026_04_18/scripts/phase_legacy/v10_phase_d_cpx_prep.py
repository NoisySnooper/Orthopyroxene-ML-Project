#!/usr/bin/env python3
"""v10 Phase D data prep: build cpx_only and cpx_liq training frames + splits.

Reads ExPetDB raw xlsx, extracts Clinopyroxene + Experiment (+ Liquid),
applies opx-parallel cleaning filters, computes training-schema DataFrames,
and writes parquet + split .npy files.

Outputs:
  data/processed/cpx_clean_cpx_only.parquet
  data/processed/cpx_clean_cpx_liq.parquet
  data/splits/train_indices_cpx_only.npy
  data/splits/test_indices_cpx_only.npy
  data/splits/train_indices_cpx_liq.npy
  data/splits/test_indices_cpx_liq.npy
  logs/v10_phase_d_cpx_prep.log

Author: NQTa (with Claude)
Date: 2026-04-16
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import (
    EXPETDB, DATA_PROC, DATA_SPLITS, LOGS,
    FE3_FET_RATIO, CATION_SUM_MIN, CATION_SUM_MAX,
    OXIDE_TOTAL_MIN, OXIDE_TOTAL_MAX, SEED_SPLIT,
)
from src.cpx_features import OXIDES_CPX, expetdb_cpx_to_training

LOG_PATH = LOGS / 'v10_phase_d_cpx_prep.log'
CPX_ONLY_PARQUET = DATA_PROC / 'cpx_clean_cpx_only.parquet'
CPX_LIQ_PARQUET = DATA_PROC / 'cpx_clean_cpx_liq.parquet'


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def _oxide_values(sheet_df, oxides, prefix=''):
    """Extract `{ox} value` columns and rename to `{prefix}{ox}`.

    Missing columns are filled with 0. Fe2O3 is folded into FeO_total
    via FeO_total = FeO + 0.899*Fe2O3 (or inferred via FE3_FET_RATIO
    if only total iron is given).
    """
    out = pd.DataFrame(index=sheet_df.index)
    for ox in oxides:
        col = f'{ox} value'
        if col in sheet_df.columns:
            out[f'{prefix}{ox}'] = pd.to_numeric(sheet_df[col], errors='coerce')
        else:
            out[f'{prefix}{ox}'] = 0.0
    # Compose FeO_total from Fe2O3 + FeO if both present.
    feo_col = f'{prefix}FeO'
    fe2o3_col = 'Fe2O3 value'
    fet_col = f'{prefix}FeO_total'
    feo = pd.to_numeric(sheet_df.get('FeO value', 0), errors='coerce').fillna(0)
    fe2o3 = pd.to_numeric(sheet_df.get('Fe2O3 value', 0), errors='coerce').fillna(0)
    feot_from_fe2o3 = feo + 0.899 * fe2o3
    feo_any = feo > 0
    fe2o3_any = fe2o3 > 0
    # If Fe2O3 reported, use combined; otherwise assume all reported iron is FeO.
    out[fet_col] = np.where(feo_any | fe2o3_any, feot_from_fe2o3, 0)
    return out


def load_expetdb_tables():
    xl = pd.ExcelFile(EXPETDB)
    exp = pd.read_excel(xl, 'Experiment')
    cpx = pd.read_excel(xl, 'Clinopyroxene')
    liq = pd.read_excel(xl, 'Liquid')
    return exp, cpx, liq


def build_cpx_only(exp, cpx, fh):
    _log(f'cpx rows: {len(cpx)}  experiments: {cpx["Experiment"].nunique()}', fh)
    oxide_df = _oxide_values(cpx, OXIDES_CPX)
    oxide_df['Index'] = cpx['Index'].values
    oxide_df['Experiment'] = cpx['Experiment'].values
    oxide_df['Citation'] = cpx['Citation'].values

    # Join experimental T, P.
    exp_keys = exp[['Index', 'Experiment', 'Citation', 'T (C)', 'P (GPa)']].rename(
        columns={'T (C)': 'T_C_raw', 'P (GPa)': 'P_GPa'})
    merged = oxide_df.merge(exp_keys, on=['Index', 'Experiment', 'Citation'], how='left')

    merged['T_C'] = pd.to_numeric(merged['T_C_raw'], errors='coerce')
    merged['P_kbar'] = pd.to_numeric(merged['P_GPa'], errors='coerce') * 10.0
    merged = merged.drop(columns=['T_C_raw', 'P_GPa'])

    n0 = len(merged)
    merged = merged.dropna(subset=['T_C', 'P_kbar'])
    _log(f'  after T/P dropna: {len(merged)}  (-{n0 - len(merged)})', fh)

    n1 = len(merged)
    merged = merged[(merged['T_C'] > 0) & (merged['P_kbar'] > 0)]
    _log(f'  after T>0, P>0: {len(merged)}  (-{n1 - len(merged)})', fh)

    # Oxide total filter.
    oxide_sum = merged[OXIDES_CPX].sum(axis=1)
    n2 = len(merged)
    merged = merged[(oxide_sum >= OXIDE_TOTAL_MIN) & (oxide_sum <= OXIDE_TOTAL_MAX)]
    _log(f'  after oxide_sum in [{OXIDE_TOTAL_MIN},{OXIDE_TOTAL_MAX}]: '
         f'{len(merged)}  (-{n2 - len(merged)})', fh)

    # Enrich with cation recalc + engineered features.
    merged = expetdb_cpx_to_training(merged)

    # Cation sum filter (6-oxygen basis -> expect ~4 cations).
    n3 = len(merged)
    merged = merged[(merged['cation_sum'] >= CATION_SUM_MIN) &
                    (merged['cation_sum'] <= CATION_SUM_MAX)]
    _log(f'  after cation_sum in [{CATION_SUM_MIN},{CATION_SUM_MAX}]: '
         f'{len(merged)}  (-{n3 - len(merged)})', fh)

    merged = merged.reset_index(drop=True)
    _log(f'  cpx_only final n={len(merged)}  n_citations={merged["Citation"].nunique()}', fh)
    return merged


def build_cpx_liq(cpx_only_df, liq, fh):
    """Inner-join cpx_only rows to Liquid rows on (Experiment, Citation).

    ExPetDB may have multiple Liquid rows per experiment (multiple analyses,
    glasses vs quenched). Average them to one row per (Experiment, Citation)
    before joining to prevent row explosion.
    """
    liq_oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O']
    liq_df = _oxide_values(liq, liq_oxides, prefix='liq_')
    liq_df['Experiment'] = liq['Experiment'].values
    liq_df['Citation'] = liq['Citation'].values

    liq_agg = (liq_df
               .groupby(['Experiment', 'Citation'], as_index=False)
               .mean(numeric_only=True))
    _log(f'  liq deduped: {len(liq_df)} -> {len(liq_agg)} rows per (Exp, Cit)', fh)

    merged = cpx_only_df.merge(liq_agg, on=['Experiment', 'Citation'], how='inner')
    _log(f'  cpx_liq after Liquid join: {len(merged)}', fh)

    # liq_Mg_num (needed by feature matrix builders).
    from src.features import OXIDE_MASSES
    mgo = pd.to_numeric(merged['liq_MgO'], errors='coerce').fillna(0)
    feo = pd.to_numeric(merged['liq_FeO'], errors='coerce').fillna(0)
    denom = mgo / OXIDE_MASSES['MgO'] + feo / OXIDE_MASSES['FeO']
    merged['liq_Mg_num'] = np.where(denom > 0,
                                     (mgo / OXIDE_MASSES['MgO']) / denom, np.nan)
    merged = merged.reset_index(drop=True)
    _log(f'  cpx_liq final n={len(merged)}  n_citations={merged["Citation"].nunique()}', fh)
    return merged


def make_splits(df, track_name, fh, test_frac=0.30, seed=SEED_SPLIT):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df['Citation'].values))
    tr_path = DATA_SPLITS / f'train_indices_{track_name}.npy'
    te_path = DATA_SPLITS / f'test_indices_{track_name}.npy'
    np.save(tr_path, tr_idx)
    np.save(te_path, te_idx)
    _log(f'  {track_name} split: train={len(tr_idx)} test={len(te_idx)} '
         f'(train pub={df.iloc[tr_idx]["Citation"].nunique()} / '
         f'test pub={df.iloc[te_idx]["Citation"].nunique()})', fh)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('v10 Phase D cpx data prep START', fh)
        exp, cpx, liq = load_expetdb_tables()
        _log(f'ExPetDB sheets loaded: Experiment={len(exp)}, '
             f'Clinopyroxene={len(cpx)}, Liquid={len(liq)}', fh)

        _log('[1] Build cpx_only frame...', fh)
        cpx_only = build_cpx_only(exp, cpx, fh)
        cpx_only.to_parquet(CPX_ONLY_PARQUET, index=False)
        _log(f'  wrote {CPX_ONLY_PARQUET}', fh)

        _log('[2] Build cpx_liq frame (inner-join to Liquid)...', fh)
        cpx_liq = build_cpx_liq(cpx_only, liq, fh)
        cpx_liq.to_parquet(CPX_LIQ_PARQUET, index=False)
        _log(f'  wrote {CPX_LIQ_PARQUET}', fh)

        _log('[3] Generate train/test splits by Citation...', fh)
        make_splits(cpx_only, 'cpx_only', fh)
        make_splits(cpx_liq, 'cpx_liq', fh)

        _log('v10 Phase D cpx data prep DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
