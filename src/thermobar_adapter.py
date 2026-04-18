"""Adapter to convert v10-native processed frames into Thermobar schema
(`_Opx`, `_Liq`, `_Cpx` suffixed columns with FeOt_*, Fe3Fet_Liq, etc.).

Phase H.0a (2026-04-18). The phase-frame builders were duplicated across
`scripts/v10_phase_g_external_benchmark.py` and Phase H inference code
that did not yet exist. This module centralizes them so the Thermobar
wrappers in `src/external_models.py` work uniformly on:
  * ExPetDB test splits (Phase G benchmarks)
  * GEOROC natural samples (Phase H)
  * Curated-locality CSVs (Phase H.6)

Canonical column conventions:
  * `_Opx` / `_Liq` / `_Cpx` (capital phase suffix) is Thermobar native
  * v10 opx training frames: `SiO2 value`, `FeO value`, `Fe2O3 value`,
    liquid columns as `liq_SiO2`, etc.
  * v10 cpx training frames: `SiO2` (unprefixed), `FeO_total`, liquid
    columns as `liq_SiO2`, etc.
  * v10 twopx frames: `SiO2_opx` / `SiO2_cpx` (lowercase phase suffix)
    with `FeO_total_opx`, `FeO_total_cpx`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FE2O3_TO_FEO = 0.8998  # mass ratio 2*M_FeO / M_Fe2O3

OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']


def opx_frame_from_opx_training(df: pd.DataFrame) -> pd.DataFrame:
    """v10 opx training/test parquet -> Thermobar opx frame.

    Source columns: `SiO2 value`, `TiO2 value`, ..., `FeO value`,
    `Fe2O3 value`. Produces `{Oxide}_Opx` + `FeOt_Opx` + `P2O5_Opx`.
    """
    out = pd.DataFrame(index=df.index)
    for o in OXIDES:
        col = f'{o} value'
        out[f'{o}_Opx'] = df[col] if col in df.columns else 0.0
    feo = df.get('FeO value', pd.Series(0.0, index=df.index)).fillna(0.0)
    fe2o3 = df.get('Fe2O3 value', pd.Series(0.0, index=df.index)).fillna(0.0)
    out['FeOt_Opx'] = feo + fe2o3 * FE2O3_TO_FEO
    out['P2O5_Opx'] = 0.0
    return out


def liq_frame_from_opx_liq(df: pd.DataFrame) -> pd.DataFrame:
    """v10 opx_liq parquet -> Thermobar liq frame (`liq_SiO2` -> `SiO2_Liq`)."""
    out = pd.DataFrame(index=df.index)
    mapping = {'SiO2': 'liq_SiO2', 'TiO2': 'liq_TiO2', 'Al2O3': 'liq_Al2O3',
               'FeO':  'liq_FeO',  'MgO':  'liq_MgO',
               'CaO':  'liq_CaO',  'Na2O': 'liq_Na2O', 'K2O': 'liq_K2O'}
    for key, src in mapping.items():
        out[f'{key}_Liq'] = df[src] if src in df.columns else 0.0
    out['Cr2O3_Liq'] = 0.0
    out['MnO_Liq']   = 0.0
    out['P2O5_Liq']  = 0.0
    out['NiO_Liq']   = 0.0
    out['CoO_Liq']   = 0.0
    out['CO2_Liq']   = 0.0
    out['FeOt_Liq']  = out['FeO_Liq']
    h2o = df.get('H2O_Liq', pd.Series(0.0, index=df.index)).fillna(0.0)
    out['H2O_Liq'] = h2o
    out['Fe3Fet_Liq'] = 0.15
    out['Sample_ID_Liq'] = df.index.astype(str)
    return out


def cpx_frame_from_cpx_training(df: pd.DataFrame) -> pd.DataFrame:
    """v10 cpx training/test parquet -> Thermobar cpx frame."""
    out = pd.DataFrame(index=df.index)
    for o in OXIDES:
        out[f'{o}_Cpx'] = df[o] if o in df.columns else 0.0
    feot = df.get('FeO_total', pd.Series(0.0, index=df.index)).fillna(0.0)
    out['FeOt_Cpx'] = feot
    out['P2O5_Cpx'] = 0.0
    return out


def liq_frame_from_cpx_liq(df: pd.DataFrame) -> pd.DataFrame:
    """v10 cpx_liq parquet -> Thermobar liq frame."""
    out = pd.DataFrame(index=df.index)
    mapping = {'SiO2': 'liq_SiO2', 'TiO2': 'liq_TiO2', 'Al2O3': 'liq_Al2O3',
               'FeO':  'liq_FeO',  'MgO':  'liq_MgO',
               'CaO':  'liq_CaO',  'Na2O': 'liq_Na2O', 'K2O': 'liq_K2O'}
    for key, src in mapping.items():
        out[f'{key}_Liq'] = df[src] if src in df.columns else 0.0
    out['Cr2O3_Liq'] = 0.0
    out['MnO_Liq']   = 0.0
    out['P2O5_Liq']  = 0.0
    out['NiO_Liq']   = 0.0
    out['CoO_Liq']   = 0.0
    out['CO2_Liq']   = 0.0
    out['FeOt_Liq']  = df.get(
        'liq_FeO_total', out['FeO_Liq']).fillna(out['FeO_Liq'])
    out['H2O_Liq']   = 0.0
    out['Fe3Fet_Liq'] = 0.15
    out['Sample_ID_Liq'] = df.index.astype(str)
    return out


def twopx_frame_from_twopx_training(df: pd.DataFrame) -> pd.DataFrame:
    """v10 twopx parquet (`SiO2_opx`, `SiO2_cpx` lowercase) -> Thermobar
    frame (`SiO2_Opx`, `SiO2_Cpx` capital).

    Expected source columns: `{Oxide}_opx`, `{Oxide}_cpx`,
    `FeO_total_opx`, `FeO_total_cpx`.
    """
    out = pd.DataFrame(index=df.index)
    for o in OXIDES:
        out[f'{o}_Opx'] = df[f'{o}_opx'] if f'{o}_opx' in df.columns else 0.0
        out[f'{o}_Cpx'] = df[f'{o}_cpx'] if f'{o}_cpx' in df.columns else 0.0
    out['FeOt_Opx'] = df.get(
        'FeO_total_opx', pd.Series(0.0, index=df.index)).fillna(0.0)
    out['FeOt_Cpx'] = df.get(
        'FeO_total_cpx', pd.Series(0.0, index=df.index)).fillna(0.0)
    out['P2O5_Opx'] = 0.0
    out['P2O5_Cpx'] = 0.0
    return out


def natural_opx_frame(df: pd.DataFrame, source: str = 'georoc') -> pd.DataFrame:
    """GEOROC/natural opx dataset -> Thermobar frame.

    GEOROC SGFTFN natural_opx_with_coords uses `SiO2`, `FeO`, `Fe2O3`
    (unsuffixed) per `scripts/natural_sample_prep_script.py`. Falls back
    to ` value` suffix if `source='expetdb'`.
    """
    out = pd.DataFrame(index=df.index)
    suffix = ' value' if source == 'expetdb' else ''
    for o in OXIDES:
        col = f'{o}{suffix}'
        out[f'{o}_Opx'] = df[col] if col in df.columns else 0.0
    feo = df.get(f'FeO{suffix}', pd.Series(0.0, index=df.index)).fillna(0.0)
    fe2o3 = df.get(f'Fe2O3{suffix}',
                   pd.Series(0.0, index=df.index)).fillna(0.0)
    out['FeOt_Opx'] = feo + fe2o3 * FE2O3_TO_FEO
    out['P2O5_Opx'] = 0.0
    return out
