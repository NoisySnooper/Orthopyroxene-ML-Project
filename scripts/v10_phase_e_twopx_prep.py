#!/usr/bin/env python3
"""v10 Phase E data prep: build twopx training frame + split.

Inner-joins already-cleaned opx_clean_opx_only.parquet and
cpx_clean_cpx_only.parquet on (Experiment, Citation). Adds twopx
engineered features (KD_FeMg_opx_cpx, Delta_Mg_num, ratios) and filters
on Fe-Mg exchange coefficient (Putirka 2008 / Brey-Kohler 1990:
1.09 +/- 0.14, taken here as [0.95, 1.23]).

Outputs:
  data/processed/twopx_clean.parquet
  data/splits/train_indices_twopx.npy
  data/splits/test_indices_twopx.npy
  logs/v10_phase_e_twopx_prep.log

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

from config import DATA_PROC, DATA_SPLITS, LOGS, SEED_SPLIT
from src.data import load_opx_only, load_cpx_only
from src.twopx_features import add_twopx_engineered, compute_kd_femg

LOG_PATH = LOGS / 'v10_phase_e_twopx_prep.log'
TWOPX_PARQUET = DATA_PROC / 'twopx_clean.parquet'

# Fe-Mg equilibrium window for opx-cpx pair.
# Putirka 2008 reports KD_FeMg(opx-cpx) = 1.09 +/- 0.14 (Brey & Kohler 1990).
TWOPX_KD_FEMG_MIN = 0.95
TWOPX_KD_FEMG_MAX = 1.23

OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO',
          'MgO', 'CaO', 'Na2O']


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def _rename_opx(df):
    """Project opx parquet to the columns we want in the twopx frame.

    Renames oxides + engineered to `_opx` suffix and standardizes the
    engineered column names so that Mg_num_opx, En_opx, Fs_opx, Wo_opx,
    Al_IV_opx, Al_VI_opx exist alongside the raw oxides.
    """
    keep = ['Experiment', 'Citation', 'T_C', 'P_kbar']
    for ox in OXIDES:
        if ox in df.columns:
            keep.append(ox)
    # opx-side engineered columns use unsuffixed names in opx_clean_opx_only:
    # Mg_num, Al_IV, Al_VI, En_frac, Fs_frac, Wo_frac.
    rename_map = {
        'Mg_num': 'Mg_num_opx',
        'Al_IV':  'Al_IV_opx',
        'Al_VI':  'Al_VI_opx',
        'En_frac': 'En_opx',
        'Fs_frac': 'Fs_opx',
        'Wo_frac': 'Wo_opx',
    }
    for src_col in rename_map:
        if src_col in df.columns:
            keep.append(src_col)

    out = df[keep].rename(columns=rename_map).copy()

    suffix_map = {ox: f'{ox}_opx' for ox in OXIDES if ox in out.columns}
    out = out.rename(columns=suffix_map)
    return out


def _rename_cpx(df):
    """Project cpx parquet to `_cpx`-suffixed oxides + engineered.

    cpx parquet already has Mg_num_cpx, En_cpx, Fs_cpx, Jd, Di, Aeg, CaTs.
    """
    keep = ['Experiment', 'Citation']
    for ox in OXIDES:
        if ox in df.columns:
            keep.append(ox)
    for eng in ['Mg_num_cpx', 'Jd', 'Di', 'Aeg', 'CaTs', 'En_cpx', 'Fs_cpx']:
        if eng in df.columns:
            keep.append(eng)
    out = df[keep].copy()
    suffix_map = {ox: f'{ox}_cpx' for ox in OXIDES if ox in out.columns}
    out = out.rename(columns=suffix_map)
    return out


def build_twopx(opx_df, cpx_df, fh):
    opx = _rename_opx(opx_df)
    cpx = _rename_cpx(cpx_df)
    _log(f'  opx projected: {opx.shape}  cpx projected: {cpx.shape}', fh)

    merged = opx.merge(cpx, on=['Experiment', 'Citation'], how='inner')
    _log(f'  after inner-join on (Experiment, Citation): n={len(merged)}', fh)

    merged = add_twopx_engineered(merged)
    kd = merged['KD_FeMg_opx_cpx']
    n0 = len(merged)
    merged = merged[(kd >= TWOPX_KD_FEMG_MIN) & (kd <= TWOPX_KD_FEMG_MAX)]
    _log(f'  KD_FeMg filter [{TWOPX_KD_FEMG_MIN},{TWOPX_KD_FEMG_MAX}]: '
         f'{len(merged)} / {n0}', fh)

    merged = merged.reset_index(drop=True)
    _log(f'  twopx final n={len(merged)}  n_citations={merged["Citation"].nunique()}', fh)
    return merged


def make_splits(df, fh, test_frac=0.30, seed=SEED_SPLIT):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df['Citation'].values))
    tr_path = DATA_SPLITS / 'train_indices_twopx.npy'
    te_path = DATA_SPLITS / 'test_indices_twopx.npy'
    np.save(tr_path, tr_idx)
    np.save(te_path, te_idx)
    _log(f'  twopx split: train={len(tr_idx)} test={len(te_idx)} '
         f'(train pub={df.iloc[tr_idx]["Citation"].nunique()} / '
         f'test pub={df.iloc[te_idx]["Citation"].nunique()})', fh)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('v10 Phase E twopx data prep START', fh)
        opx = load_opx_only()
        cpx = load_cpx_only()
        _log(f'opx_clean_opx_only: {opx.shape}  cpx_clean_cpx_only: {cpx.shape}', fh)

        twopx = build_twopx(opx, cpx, fh)
        twopx.to_parquet(TWOPX_PARQUET, index=False)
        _log(f'  wrote {TWOPX_PARQUET}', fh)

        make_splits(twopx, fh)
        _log('v10 Phase E twopx data prep DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
