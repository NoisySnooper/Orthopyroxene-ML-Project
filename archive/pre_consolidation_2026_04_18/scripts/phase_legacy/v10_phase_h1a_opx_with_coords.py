#!/usr/bin/env python3
"""Phase H.1a: re-merge lat/lon + K2O into cleaned natural opx.

Original `data/natural/natural_sample_prep_script.py` dropped lat/lon
and K2O. This rebuild keeps both, re-applies the same oxide-dropna and
clip-negative-microprobe logic, and writes
`data/natural/natural_opx_with_coords.csv`.

Schema (13 oxide + 8 metadata + 4 geo + 1 derived = 26 columns):
  CITATION, SAMPLE NAME, TECTONIC SETTING, LOCATION, LOCATION COMMENT,
  ROCK NAME, ROCK TEXTURE, LAND/SEA (SAMPLING),
  LATITUDE (MIN.), LATITUDE (MAX.), LONGITUDE (MIN.), LONGITUDE (MAX.),
  lat, lon  (midpoints; NaN if either endpoint missing),
  SiO2, TiO2, Al2O3, Cr2O3, FeO_total, MnO, MgO, CaO, Na2O, K2O,
  FeO, Fe2O3  (raw split where available, for downstream Fe speciation)

Expected row count ~53,050 (parity with existing cleaned file).
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

from config import LOGS

RAW = Path('data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv')
OUT = Path('data/natural/natural_opx_with_coords.csv')
LOG_PATH = LOGS / 'v10_phase_h1a_opx_with_coords.log'

META_COLS = [
    'CITATION', 'SAMPLE NAME', 'TECTONIC SETTING', 'LOCATION',
    'LOCATION COMMENT', 'ROCK NAME', 'ROCK TEXTURE', 'LAND/SEA (SAMPLING)',
    'LATITUDE (MIN.)', 'LATITUDE (MAX.)',
    'LONGITUDE (MIN.)', 'LONGITUDE (MAX.)',
]

OXIDE_RENAME = {
    'SIO2(WT%)': 'SiO2', 'TIO2(WT%)': 'TiO2', 'AL2O3(WT%)': 'Al2O3',
    'CR2O3(WT%)': 'Cr2O3', 'FEOT(WT%)': 'FeO_total',
    'MNO(WT%)': 'MnO', 'MGO(WT%)': 'MgO', 'CAO(WT%)': 'CaO',
    'NA2O(WT%)': 'Na2O', 'K2O(WT%)': 'K2O',
    'FEO(WT%)': 'FeO', 'FE2O3(WT%)': 'Fe2O3',
}

REQUIRED_OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total',
                   'MnO', 'MgO', 'CaO', 'Na2O']


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log(f'START H.1a; read {RAW}', fh)
        df = pd.read_csv(RAW, low_memory=False, encoding='latin1')
        _log(f'raw shape: {df.shape}', fh)

        mask = df['MINERAL'].astype(str).str.contains(
            'ORTHOPYROXENE', case=False, na=False)
        df = df[mask].copy()
        _log(f'after ORTHOPYROXENE filter: {df.shape}', fh)

        keep = [c for c in META_COLS if c in df.columns]
        keep += [c for c in OXIDE_RENAME if c in df.columns]
        df = df[keep].rename(columns=OXIDE_RENAME)

        oxide_cols = [c for c in OXIDE_RENAME.values() if c in df.columns]
        for c in oxide_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df[oxide_cols] = df[oxide_cols].clip(lower=0.0001)
        df = df.dropna(subset=[c for c in REQUIRED_OXIDES if c in df.columns])
        _log(f'after numeric + dropna(required oxides): {df.shape}', fh)

        # Midpoint lat/lon (NaN if either endpoint missing)
        for axis, lo, hi in [('lat', 'LATITUDE (MIN.)', 'LATITUDE (MAX.)'),
                             ('lon', 'LONGITUDE (MIN.)', 'LONGITUDE (MAX.)')]:
            a = pd.to_numeric(df.get(lo), errors='coerce')
            b = pd.to_numeric(df.get(hi), errors='coerce')
            df[axis] = (a + b) / 2.0

        _log(f'lat/lon coverage: lat={df.lat.notna().sum()} '
             f'lon={df.lon.notna().sum()}', fh)

        # Diagnostic: compare to existing cleaned file
        prev = pd.read_csv('data/natural/natural_opx_cleaned.csv',
                           low_memory=False)
        _log(f'prior cleaned rows={len(prev)} new rows={len(df)} '
             f'delta={len(df)-len(prev)}', fh)

        tecs = (df['TECTONIC SETTING'].fillna('UNKNOWN')
                .value_counts().head(15))
        _log(f'top-15 tectonic settings:\n{tecs.to_string()}', fh)

        df.to_csv(OUT, index=False)
        _log(f'wrote {OUT} shape={df.shape}', fh)
        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
