#!/usr/bin/env python3
"""v10 Phase F data prep: build universal training frame + split.

Union of (Experiment, Citation) keys across opx_only, cpx_only,
opx_liq, cpx_liq parquets. For each unique experiment, fill opx slots
from opx_only, cpx slots from cpx_only, liquid slots from whichever
_liq parquet has the row. Missing phases -> zero-filled; presence bits
flip 0/1 accordingly.

T_C / P_kbar come from whichever source is available (they are
experiment-level, not phase-level, so they must match; we pick the
first non-null).

Outputs:
  data/processed/universal_clean.parquet
  data/splits/train_indices_universal.npy
  data/splits/test_indices_universal.npy
  logs/v10_phase_f_universal_prep.log

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
from src.data import load_opx_only, load_opx_liq, load_cpx_only, load_cpx_liq
from src.universal_features import (
    OPX_OXIDES, CPX_OXIDES, LIQ_OXIDES,
    OPX_ENG, CPX_ENG, LIQ_ENG,
)

LOG_PATH = LOGS / 'v10_phase_f_universal_prep.log'
UNIVERSAL_PARQUET = DATA_PROC / 'universal_clean.parquet'


def _log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def _project_opx(opx_df):
    """Pull opx oxides + engineered into `_opx`-suffixed columns. Average
    duplicate rows per (Experiment, Citation) to guarantee one row per key."""
    rename = {}
    for ox in OPX_OXIDES:
        if ox in opx_df.columns:
            rename[ox] = f'{ox}_opx'
    rename.update({
        'Mg_num': 'Mg_num_opx',
        'Al_IV':  'Al_IV_opx',
        'Al_VI':  'Al_VI_opx',
        'En_frac': 'En_opx',
        'Fs_frac': 'Fs_opx',
        'Wo_frac': 'Wo_opx',
        'MgTs':   'MgTs_opx',
    })
    keep = ['Experiment', 'Citation', 'T_C', 'P_kbar'] + list(rename.keys())
    keep = [c for c in keep if c in opx_df.columns]
    out = opx_df[keep].rename(columns=rename).copy()
    return (out.groupby(['Experiment', 'Citation'], as_index=False)
               .mean(numeric_only=True))


def _project_cpx(cpx_df):
    """Pull cpx oxides + engineered into `_cpx`-suffixed columns. Average
    duplicate rows per (Experiment, Citation) to guarantee one row per key."""
    rename = {ox: f'{ox}_cpx' for ox in CPX_OXIDES if ox in cpx_df.columns}
    eng_cols = ['Mg_num_cpx', 'Jd', 'Di', 'Aeg', 'CaTs', 'En_cpx', 'Fs_cpx']
    keep = ['Experiment', 'Citation', 'T_C', 'P_kbar'] + list(rename.keys()) + eng_cols
    keep = [c for c in keep if c in cpx_df.columns]
    out = cpx_df[keep].rename(columns=rename).copy()
    return (out.groupby(['Experiment', 'Citation'], as_index=False)
               .mean(numeric_only=True))


def _project_liq(opx_liq_df, cpx_liq_df):
    """Liquid data lives in both opx_liq and cpx_liq parquets. Stack and
    deduplicate on (Experiment, Citation)."""
    liq_cols = [f'liq_{ox}' for ox in LIQ_OXIDES] + LIQ_ENG
    frames = []
    for src in (opx_liq_df, cpx_liq_df):
        keep = ['Experiment', 'Citation'] + [c for c in liq_cols if c in src.columns]
        frames.append(src[keep].copy())
    stacked = pd.concat(frames, ignore_index=True)
    deduped = (stacked
               .drop_duplicates(subset=['Experiment', 'Citation'])
               .reset_index(drop=True))
    return deduped


def build_universal(opx_only, cpx_only, opx_liq, cpx_liq, fh):
    opx_proj = _project_opx(opx_only)
    cpx_proj = _project_cpx(cpx_only)
    liq_proj = _project_liq(opx_liq, cpx_liq)
    _log(f'  opx_proj={opx_proj.shape}  cpx_proj={cpx_proj.shape}  '
         f'liq_proj={liq_proj.shape}', fh)

    # All unique (Experiment, Citation) keys.
    all_keys = (pd.concat([opx_proj[['Experiment', 'Citation']],
                            cpx_proj[['Experiment', 'Citation']],
                            liq_proj[['Experiment', 'Citation']]],
                           ignore_index=True)
                .drop_duplicates()
                .reset_index(drop=True))
    _log(f'  union unique experiments: {len(all_keys)}', fh)

    merged = all_keys.merge(opx_proj, on=['Experiment', 'Citation'], how='left')
    merged = merged.merge(cpx_proj, on=['Experiment', 'Citation'],
                          how='left', suffixes=('', '_cpx_src'))

    # T_C / P_kbar may appear in both frames. Coalesce.
    for col in ('T_C', 'P_kbar'):
        cpx_col = f'{col}_cpx_src'
        if cpx_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[cpx_col])
            merged = merged.drop(columns=[cpx_col])

    merged = merged.merge(liq_proj, on=['Experiment', 'Citation'], how='left')

    # Presence bits: opx_present if any opx oxide non-null; same for cpx; liq
    # likewise via liq_SiO2 non-null.
    opx_col = 'SiO2_opx'
    cpx_col = 'SiO2_cpx'
    liq_col = 'liq_SiO2'
    merged['opx_present'] = merged[opx_col].notna().astype(int) if opx_col in merged.columns else 0
    merged['cpx_present'] = merged[cpx_col].notna().astype(int) if cpx_col in merged.columns else 0
    merged['liq_present'] = merged[liq_col].notna().astype(int) if liq_col in merged.columns else 0

    # Zero-fill all feature columns.
    feat_cols = [f'{ox}_opx' for ox in OPX_OXIDES] + OPX_ENG \
                + [f'{ox}_cpx' for ox in CPX_OXIDES] + CPX_ENG \
                + [f'liq_{ox}' for ox in LIQ_OXIDES] + LIQ_ENG
    for c in feat_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        else:
            merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0.0)

    # Rows missing T_C or P_kbar can't be targets; drop.
    n0 = len(merged)
    merged = merged.dropna(subset=['T_C', 'P_kbar']).reset_index(drop=True)
    _log(f'  after T_C/P_kbar dropna: {len(merged)} (-{n0 - len(merged)})', fh)

    # Derive phase-scope label for T13/T14 stratified evaluation.
    def scope(row):
        bits = (row['opx_present'], row['cpx_present'], row['liq_present'])
        return {
            (1, 0, 0): 'opx_only',
            (1, 0, 1): 'opx_liq',
            (0, 1, 0): 'cpx_only',
            (0, 1, 1): 'cpx_liq',
            (1, 1, 0): 'twopx',
            (1, 1, 1): 'twopx_liq',
            (0, 0, 1): 'liq_only',
        }.get(bits, 'other')
    merged['phase_scope'] = merged.apply(scope, axis=1)

    scope_counts = merged['phase_scope'].value_counts().to_dict()
    _log(f'  phase_scope counts: {scope_counts}', fh)
    _log(f'  universal final n={len(merged)}  n_citations={merged["Citation"].nunique()}', fh)
    return merged


def make_splits(df, fh, test_frac=0.30, seed=SEED_SPLIT):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df['Citation'].values))
    tr_path = DATA_SPLITS / 'train_indices_universal.npy'
    te_path = DATA_SPLITS / 'test_indices_universal.npy'
    np.save(tr_path, tr_idx)
    np.save(te_path, te_idx)
    _log(f'  universal split: train={len(tr_idx)} test={len(te_idx)} '
         f'(train pub={df.iloc[tr_idx]["Citation"].nunique()} / '
         f'test pub={df.iloc[te_idx]["Citation"].nunique()})', fh)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('v10 Phase F universal data prep START', fh)
        opx_only = load_opx_only()
        cpx_only = load_cpx_only()
        opx_liq  = load_opx_liq()
        cpx_liq  = load_cpx_liq()
        _log(f'  opx_only={opx_only.shape}  cpx_only={cpx_only.shape}  '
             f'opx_liq={opx_liq.shape}  cpx_liq={cpx_liq.shape}', fh)

        universal = build_universal(opx_only, cpx_only, opx_liq, cpx_liq, fh)
        universal.to_parquet(UNIVERSAL_PARQUET, index=False)
        _log(f'  wrote {UNIVERSAL_PARQUET}', fh)

        make_splits(universal, fh)
        _log('v10 Phase F universal data prep DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
