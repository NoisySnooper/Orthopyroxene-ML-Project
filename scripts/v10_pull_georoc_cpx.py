#!/usr/bin/env python3
"""H.1b: Pull GEOROC cpx 2024-12 release via the GRO.data Dataverse Native API.

Per docs/natural_worldwide_plan.md Section 2.2, the H.1b instructions, and
the user-supplied resolution to the original halt #1.

The GEOROC compilation is published on the Goettingen Research Online
(GRO.data) Dataverse instance under DOI 10.25625/SGFTFN (no extra path
component). The Dataverse Native API at /api/datasets/:persistentId/
returns the file list, and /api/access/datafile/{file_id} streams the
raw bytes. CC BY-SA 4.0 license; no API key, no guestbook required for
public datasets.

Pipeline:
  1. Look up the SGFTFN dataset metadata.
  2. Find the file labelled '2024-12-SGFTFN_CLINOPYROXENES.csv' (~338 MB).
  3. Stream-download to data/natural/2024-12-GEOROC_CLINOPYROXENES.csv.
  4. Filter rows where MINERAL contains 'CLINOPYROXENE' (the raw file
     mixes mineral types).
  5. Apply Wo > 20 (pyroxene quad cut), Mg# in [0.5, 0.95], oxide total
     in [99, 101], drop rows missing any of 9 required oxides.
  6. Re-merge lat/lon and metadata; write
     data/natural/natural_cpx_with_coords.csv.

Halt condition: if the raw file count is < 30,000 rows, write
results/HALT_REPORT_PHASE_H.md and exit 1 (halt #1).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATAVERSE_BASE = 'https://data.goettingen-research-online.de'
DATASET_PID = 'doi:10.25625/SGFTFN'
TARGET_LABEL_SUBSTR = 'CLINOPYROXENES'
RAW_OUT = PROJECT_ROOT / 'data' / 'natural' / '2024-12-GEOROC_CLINOPYROXENES.csv'
CLEAN_OUT = PROJECT_ROOT / 'data' / 'natural' / 'natural_cpx_with_coords.csv'
LOG_FILE = PROJECT_ROOT / 'results' / 'PHASE_H_RUN_LOG.md'

# Cleaning parameters (mirrored from natural_sample_prep_script.py + opx
# pipeline).
REQUIRED_OXIDES = ['SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'CR2O3(WT%)',
                   'FEOT(WT%)', 'MNO(WT%)', 'MGO(WT%)', 'CAO(WT%)',
                   'NA2O(WT%)']
META_COLS = ['CITATION', 'SAMPLE NAME', 'TECTONIC SETTING', 'LOCATION',
             'LOCATION COMMENT', 'ROCK NAME', 'ROCK TEXTURE',
             'LAND/SEA (SAMPLING)',
             'LATITUDE (MIN.)', 'LATITUDE (MAX.)',
             'LONGITUDE (MIN.)', 'LONGITUDE (MAX.)']

# Halt thresholds
HALT_RAW_THRESHOLD = 30_000          # rows in raw cpx CSV
HALT_FINAL_FLOOR = 40_000            # cleaned rows must hit this
HALT_FINAL_CEIL = 80_000             # cleaned rows must not exceed this


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = f'- {ts} (H.1b) {msg}'
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def fetch_dataset_metadata() -> dict:
    url = (f'{DATAVERSE_BASE}/api/datasets/:persistentId/'
           f'?persistentId={DATASET_PID}')
    _log(f'caveman: GET {url}')
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def find_target_file(meta: dict) -> dict:
    files = meta['data']['latestVersion']['files']
    for f in files:
        df = f.get('dataFile', {})
        label = (f.get('label') or df.get('filename') or '').upper()
        if TARGET_LABEL_SUBSTR in label:
            return f
    raise RuntimeError(
        f'no file with {TARGET_LABEL_SUBSTR!r} substring in dataset '
        f'{DATASET_PID}; got {len(files)} files')


def download_file(file_id: int, expected_mb: float) -> None:
    url = f'{DATAVERSE_BASE}/api/access/datafile/{file_id}'
    _log(f'caveman: stream-GET {url} (expect ~{expected_mb:.1f} MB)')
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    n_bytes = 0
    chunk_count = 0
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(RAW_OUT, 'wb') as fh:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                sha.update(chunk)
                n_bytes += len(chunk)
                chunk_count += 1
    _log(f'caveman: download done. {n_bytes/1024/1024:.1f} MB in '
         f'{chunk_count} chunks. SHA256[:12]={sha.hexdigest()[:12]}')


def write_halt(reason: str, n_raw: int) -> None:
    halt_path = PROJECT_ROOT / 'results' / 'HALT_REPORT_PHASE_H.md'
    sha = ''
    try:
        import subprocess
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        pass
    halt_path.write_text(
        f'# HALT REPORT — Phase H (re-fired)\n\n'
        f'Halt condition: #1 (raw cpx count below threshold)\n'
        f'UTC: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}\n'
        f'Branch: phase_h_natural_worldwide\n'
        f'SHA: {sha}\n\n'
        f'Reason: {reason}\nRaw rows: {n_raw}\n'
        f'Threshold: {HALT_RAW_THRESHOLD}\n',
        encoding='utf-8')


def clean_cpx_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard pyroxene cleaning rules to the raw GEOROC pull.

    Mirrors the opx natural_sample_prep_script.py logic but for
    clinopyroxene. Returns a cleaned DataFrame with training-schema
    column names.
    """
    n0 = len(df)

    # 1. Filter to clinopyroxene rows
    df = df[df['MINERAL'].astype(str).str.contains(
        'CLINOPYROXENE', case=False, na=False)].copy()
    n_cpx = len(df)
    _log(f'caveman: rows after CLINOPYROXENE filter: {n_cpx} (from {n0})')

    # 2. Coerce required oxides to numeric
    for col in REQUIRED_OXIDES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            _log(f'caveman: missing required oxide column {col!r}; bail')
            raise RuntimeError(f'missing required column {col}')

    # 3. Drop rows missing any required oxide
    df = df.dropna(subset=REQUIRED_OXIDES).copy()
    n_complete = len(df)
    _log(f'caveman: rows with all 9 oxides: {n_complete}')

    # 4. Oxide total in [99, 101]
    total = df[REQUIRED_OXIDES].sum(axis=1)
    df = df[(total >= 99.0) & (total <= 101.0)].copy()
    n_total = len(df)
    _log(f'caveman: rows after oxide total in [99,101]: {n_total}')

    # 5. Mg# in [0.5, 0.95]
    fe = df['FEOT(WT%)']; mg = df['MGO(WT%)']
    # Convert wt% to molar via molar masses (FeO=71.85, MgO=40.30)
    fe_mol = fe / 71.85
    mg_mol = mg / 40.30
    mg_num = mg_mol / (mg_mol + fe_mol)
    df = df[(mg_num >= 0.5) & (mg_num <= 0.95)].copy()
    n_mgnum = len(df)
    _log(f'caveman: rows after Mg# in [0.5,0.95]: {n_mgnum}')

    # 6. Wo > 20 (cpx pyroxene-quad: cpx is high-Ca, so Wo > ~20)
    # Wo (mol fraction of CaSiO3 wollastonite) ~ Ca / (Ca + Mg + Fe)
    ca = df['CAO(WT%)']
    ca_mol = ca / 56.08
    fe_mol2 = df['FEOT(WT%)'] / 71.85
    mg_mol2 = df['MGO(WT%)'] / 40.30
    wo = 100 * ca_mol / (ca_mol + mg_mol2 + fe_mol2)
    df = df[wo > 20.0].copy()
    n_wo = len(df)
    _log(f'caveman: rows after Wo > 20: {n_wo}')

    # 7. Project metadata columns + oxides to training schema
    keep_cols = [c for c in META_COLS + REQUIRED_OXIDES if c in df.columns]
    df = df[keep_cols].copy()

    # Rename oxides to training-schema (no WT% suffix, normal-case)
    rename_map = {
        'SIO2(WT%)': 'SiO2', 'TIO2(WT%)': 'TiO2', 'AL2O3(WT%)': 'Al2O3',
        'CR2O3(WT%)': 'Cr2O3', 'FEOT(WT%)': 'FeO_total', 'MNO(WT%)': 'MnO',
        'MGO(WT%)': 'MgO', 'CAO(WT%)': 'CaO', 'NA2O(WT%)': 'Na2O',
    }
    df = df.rename(columns=rename_map)

    # Add lat/lon convenience columns (mid of min/max)
    if {'LATITUDE (MIN.)', 'LATITUDE (MAX.)'} <= set(df.columns):
        df['lat'] = pd.to_numeric(df['LATITUDE (MIN.)'], errors='coerce')
        lat_max = pd.to_numeric(df['LATITUDE (MAX.)'], errors='coerce')
        df['lat'] = df['lat'].fillna(lat_max)
    if {'LONGITUDE (MIN.)', 'LONGITUDE (MAX.)'} <= set(df.columns):
        df['lon'] = pd.to_numeric(df['LONGITUDE (MIN.)'], errors='coerce')
        lon_max = pd.to_numeric(df['LONGITUDE (MAX.)'], errors='coerce')
        df['lon'] = df['lon'].fillna(lon_max)

    return df


def main() -> int:
    _log('caveman: H.1b start (Dataverse Native API). target dataset='
         f'{DATASET_PID}')

    if RAW_OUT.exists() and RAW_OUT.stat().st_size > 50_000_000:
        _log(f'caveman: existing raw at {RAW_OUT.name} '
             f'({RAW_OUT.stat().st_size/1024/1024:.1f} MB), skip download')
    else:
        meta = fetch_dataset_metadata()
        target = find_target_file(meta)
        df_info = target['dataFile']
        file_id = df_info['id']
        size_mb = df_info.get('filesize', 0) / 1024 / 1024
        _log(f'caveman: target file_id={file_id} '
             f'name={df_info.get("filename")!r} size={size_mb:.1f} MB')
        download_file(file_id, size_mb)

    if RAW_OUT.stat().st_size < 50_000_000:
        write_halt(
            f'Downloaded file too small: {RAW_OUT.stat().st_size} bytes',
            n_raw=0)
        return 1

    _log(f'caveman: read raw CSV ({RAW_OUT.stat().st_size/1024/1024:.1f} MB)')
    df = pd.read_csv(RAW_OUT, low_memory=False, encoding='latin1')
    n_raw = len(df)
    _log(f'caveman: raw row count = {n_raw}')

    if n_raw < HALT_RAW_THRESHOLD:
        write_halt(
            f'raw row count {n_raw} < threshold {HALT_RAW_THRESHOLD}',
            n_raw=n_raw)
        return 1

    cleaned = clean_cpx_dataframe(df)
    n_clean = len(cleaned)
    _log(f'caveman: cleaned row count = {n_clean}')

    cleaned.to_csv(CLEAN_OUT, index=False, encoding='utf-8')
    sha = hashlib.sha256(CLEAN_OUT.read_bytes()).hexdigest()
    _log(f'caveman: wrote {CLEAN_OUT.name} ({n_clean} rows, '
         f'SHA256[:12]={sha[:12]})')

    if not (HALT_FINAL_FLOOR <= n_clean <= HALT_FINAL_CEIL):
        _log(f'caveman: cleaned n {n_clean} outside expected band '
             f'[{HALT_FINAL_FLOOR}, {HALT_FINAL_CEIL}]; soft warning, '
             'not a halt')

    return 0


if __name__ == '__main__':
    sys.exit(main())
