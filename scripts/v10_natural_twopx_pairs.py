#!/usr/bin/env python3
"""H.1d: Build natural two-pyroxene pairs by sample-name + citation match.

Per docs/natural_worldwide_plan.md Section 2.3 and the activation
prompt H.1d step:

Inner-join data/natural/natural_opx_with_coords.csv and
data/natural/natural_cpx_with_coords.csv on:
  - Same CITATION
  - Same SAMPLE NAME (case-insensitive, whitespace-stripped)
  - Within 0.01 degree lat/lon tolerance
  - Same TECTONIC SETTING
  - Consistent ROCK NAME (case-insensitive substring match either way)

Output: data/natural/natural_twopx_pairs.csv with both mineral
compositions side by side (suffixes _Opx and _Cpx) and one row per
matched pair.

Halt condition #2 fires if the join produces < 1,000 pairs.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

OPX_PATH = PROJECT_ROOT / 'data' / 'natural' / 'natural_opx_with_coords.csv'
CPX_PATH = PROJECT_ROOT / 'data' / 'natural' / 'natural_cpx_with_coords.csv'
OUT_PATH = PROJECT_ROOT / 'data' / 'natural' / 'natural_twopx_pairs.csv'
LOG_FILE = PROJECT_ROOT / 'results' / 'PHASE_H_RUN_LOG.md'

OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO',
          'MgO', 'CaO', 'Na2O']
META = ['CITATION', 'SAMPLE NAME', 'TECTONIC SETTING', 'LOCATION',
        'ROCK NAME', 'lat', 'lon']

LATLON_TOL = 0.01  # degrees


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = f'- {ts} (H.1d) {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def _normalize_join_key(s: pd.Series) -> pd.Series:
    """Lowercase + strip, replace NaN with empty string for hash-key use."""
    return (s.astype(str).str.strip().str.lower()
              .replace({'nan': '', 'none': ''}))


def _consistent_rock(a: str, b: str) -> bool:
    """Two rock names are consistent if either is a substring of the other
    after lowercase-strip, OR both are empty/NA."""
    a = (a or '').strip().lower()
    b = (b or '').strip().lower()
    if not a or not b:
        return True
    return a in b or b in a


def main() -> int:
    _log('caveman: H.1d start. join opx + cpx natural sets')

    opx = pd.read_csv(OPX_PATH, low_memory=False)
    cpx = pd.read_csv(CPX_PATH, low_memory=False)
    _log(f'caveman: loaded opx={len(opx)} cpx={len(cpx)} (grain level)')

    # Normalize join keys
    for fr in (opx, cpx):
        fr['_cit_key'] = _normalize_join_key(fr['CITATION'])
        fr['_smp_key'] = _normalize_join_key(fr['SAMPLE NAME'])
        fr['_tec_key'] = _normalize_join_key(fr['TECTONIC SETTING'])

    # Drop rows with empty sample name OR citation
    opx = opx[(opx._cit_key != '') & (opx._smp_key != '')].copy()
    cpx = cpx[(cpx._cit_key != '') & (cpx._smp_key != '')].copy()

    # Aggregate grain-level oxide analyses to per-sample medians.
    # Plan section 2.3 expects "samples where both opx and cpx are
    # analyzed" -> per-sample, not per-grain. Per-grain would Cartesian
    # to ~1.5M rows which is not what the plan intended.
    group_keys = ['_cit_key', '_smp_key', '_tec_key']
    agg_oxides = {ox: 'median' for ox in OXIDES if ox in opx.columns}
    agg_meta = {c: 'first' for c in
                ['CITATION', 'SAMPLE NAME', 'TECTONIC SETTING',
                 'LOCATION', 'ROCK NAME', 'lat', 'lon']
                if c in opx.columns}
    opx_keyed = (opx.groupby(group_keys, as_index=False)
                    .agg({**agg_meta, **agg_oxides}))
    agg_oxides_c = {ox: 'median' for ox in OXIDES if ox in cpx.columns}
    agg_meta_c = {c: 'first' for c in
                  ['CITATION', 'SAMPLE NAME', 'TECTONIC SETTING',
                   'LOCATION', 'ROCK NAME', 'lat', 'lon']
                  if c in cpx.columns}
    cpx_keyed = (cpx.groupby(group_keys, as_index=False)
                    .agg({**agg_meta_c, **agg_oxides_c}))
    _log(f'caveman: keyed (per-sample medians) opx={len(opx_keyed)} '
         f'cpx={len(cpx_keyed)}')

    # Inner join on (cit, smp, tec). Tectonic-setting is part of key so
    # mismatches drop out immediately.
    merged = opx_keyed.merge(
        cpx_keyed,
        on=['_cit_key', '_smp_key', '_tec_key'],
        suffixes=('_Opx', '_Cpx'),
    )
    _log(f'caveman: post-merge n_pre_filter={len(merged)}')

    # Lat/lon tolerance filter
    if 'lat_Opx' in merged.columns and 'lat_Cpx' in merged.columns:
        lat_close = (np.abs(merged['lat_Opx'] - merged['lat_Cpx'])
                     <= LATLON_TOL) | merged['lat_Opx'].isna() \
            | merged['lat_Cpx'].isna()
        lon_close = (np.abs(merged['lon_Opx'] - merged['lon_Cpx'])
                     <= LATLON_TOL) | merged['lon_Opx'].isna() \
            | merged['lon_Cpx'].isna()
        merged = merged[lat_close & lon_close].copy()
        _log(f'caveman: after lat/lon filter ({LATLON_TOL} deg): '
             f'{len(merged)}')

    # Rock-name consistency check
    rock_ok = [_consistent_rock(a, b) for a, b in zip(
        merged['ROCK NAME_Opx'].fillna(''),
        merged['ROCK NAME_Cpx'].fillna(''))]
    merged = merged[pd.Series(rock_ok, index=merged.index)].copy()
    _log(f'caveman: after rock-name consistency: {len(merged)}')

    # Halt #2: <1000 pairs
    if len(merged) < 1_000:
        halt_path = PROJECT_ROOT / 'results' / 'HALT_REPORT_PHASE_H.md'
        with open(halt_path, 'a', encoding='utf-8') as f:
            f.write(
                '\n\n## Halt #2 (H.1d, 2026-04-27)\n\n'
                f'twopx pair join produced only {len(merged)} pairs '
                '(< 1000 threshold). Likely sample-name collision '
                'failure; check normalization rules.\n')
        _log(f'caveman: HALT #2 fires at {len(merged)} pairs')
        return 1

    # Build the wide-format output. Suffix oxides _Opx / _Cpx for use by
    # external_models.py (which slices by suffix).
    opx_renamed = {ox: f'{ox}_Opx' for ox in OXIDES}
    cpx_renamed = {ox: f'{ox}_Cpx' for ox in OXIDES}
    # The merged frame already has _Opx / _Cpx suffixes from pandas merge,
    # but only for columns that COLLIDED (every column except the join
    # keys). Oxide columns collided because both opx and cpx had them.

    # Sanity: confirm we have all 9 oxides under both _Opx and _Cpx.
    for ox in OXIDES:
        for suf in ('_Opx', '_Cpx'):
            col = f'{ox}{suf}'
            if col not in merged.columns:
                _log(f'caveman: WARN missing {col} after merge')

    # Build clean output schema. Use Opx-side metadata as the canonical
    # locality + tectonic setting (they're equal by join-key construction
    # for citation / sample / tectonic, and within 0.01 deg for lat/lon).
    out_cols = []
    out_cols += ['CITATION_Opx', 'SAMPLE NAME_Opx',
                 'TECTONIC SETTING_Opx', 'LOCATION_Opx',
                 'ROCK NAME_Opx', 'lat_Opx', 'lon_Opx']
    out_cols += [f'{ox}_Opx' for ox in OXIDES]
    out_cols += [f'{ox}_Cpx' for ox in OXIDES]
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols].copy()
    out = out.rename(columns={
        'CITATION_Opx': 'CITATION',
        'SAMPLE NAME_Opx': 'SAMPLE NAME',
        'TECTONIC SETTING_Opx': 'TECTONIC SETTING',
        'LOCATION_Opx': 'LOCATION',
        'ROCK NAME_Opx': 'ROCK NAME',
        'lat_Opx': 'lat',
        'lon_Opx': 'lon',
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding='utf-8')
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    _log(f'caveman: wrote {OUT_PATH.name} ({len(out)} pairs, '
         f'SHA256[:12]={sha[:12]})')

    # Tectonic setting distribution for the run-log audit
    tec = out['TECTONIC SETTING'].value_counts(dropna=False).head(8)
    _log(f'caveman: top tectonic settings in pairs:\n  '
         + '\n  '.join(f'{k}: {v}' for k, v in tec.items()))

    return 0


if __name__ == '__main__':
    sys.exit(main())
