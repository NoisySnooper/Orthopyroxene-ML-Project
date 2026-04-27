"""Opx-liq ArcPL external corpus loader.

ArcPL (Agreda-Lopez 2024) ships as a cpx-only spreadsheet in the paper's
datasets directory. The project-side opx-liq ArcPL corpus is reconstructed
from the LEPR Opx-Liq sheet by filtering to rows whose Citation_x carries
the `_notinLEPR` tag that marks the ArcPL-sourced subset. The loader
replicates the nb04 Part 3 pipeline: rename to ExPetDB flat schema, drop
stray `_y` columns from the LEPR merge, H2O non-negativity, oxide-total
and cation-sum QC, pigeonite / Wo filter, pressure ceiling, Fe-Mg Kd
equilibrium window, and citation-key overlap removal against ExPetDB.

Consumers: scripts/figures/make_fig_dataset_map_holdout.py (Core_01b
panels c/d) and scripts/external_eval/eval_arcpl_opx_corrected.py
(Core_10b).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CATION_SUM_MAX,
    CATION_SUM_MIN,
    DATA_PROC,
    FE3_FET_RATIO,
    KD_FEMG_MAX,
    KD_FEMG_MIN,
    LEPR_XLSX,
    OXIDE_TOTAL_MAX,
    OXIDE_TOTAL_MIN,
    P_CEILING_KBAR,
    WO_MAX_MOL_PCT,
)
from src.features import add_engineered_features, cation_recalc_6oxy

_RENAME = {
    'Citation_x': 'Citation', 'Experiment_x': 'Experiment',
    'P_kbar_x': 'P_kbar',
    'SiO2_Opx': 'SiO2', 'TiO2_Opx': 'TiO2', 'Al2O3_Opx': 'Al2O3',
    'FeOt_Opx': 'FeO_total', 'MnO_Opx': 'MnO', 'MgO_Opx': 'MgO',
    'CaO_Opx': 'CaO', 'Na2O_Opx': 'Na2O', 'K2O_Opx': 'K2O',
    'Cr2O3_Opx': 'Cr2O3', 'P2O5_Opx': 'P2O5',
    'SiO2_Liq': 'liq_SiO2', 'TiO2_Liq': 'liq_TiO2', 'Al2O3_Liq': 'liq_Al2O3',
    'FeOt_Liq': 'liq_FeO', 'MnO_Liq': 'liq_MnO', 'MgO_Liq': 'liq_MgO',
    'CaO_Liq': 'liq_CaO', 'Na2O_Liq': 'liq_Na2O', 'K2O_Liq': 'liq_K2O',
    'Cr2O3_Liq': 'liq_Cr2O3', 'H2O_Liq': 'H2O_Liq',
    'H2O_Liq_Method': 'H2O_Liq_Method',
}

_ALL_OX = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO', 'MgO',
           'CaO', 'Na2O', 'K2O', 'P2O5']


_CITATION_RE = re.compile(r'([A-Za-z][A-Za-z ,.\-]+?)[\s,]*((?:19|20)\d{2})')

_THERMOBAR_OPX_MAP = {
    'SiO2': 'SiO2_Opx', 'TiO2': 'TiO2_Opx', 'Al2O3': 'Al2O3_Opx',
    'FeO_total': 'FeOt_Opx', 'MnO': 'MnO_Opx', 'MgO': 'MgO_Opx',
    'CaO': 'CaO_Opx', 'Na2O': 'Na2O_Opx', 'K2O': 'K2O_Opx',
    'Cr2O3': 'Cr2O3_Opx', 'P2O5': 'P2O5_Opx',
}
_THERMOBAR_LIQ_MAP = {
    'liq_SiO2': 'SiO2_Liq', 'liq_TiO2': 'TiO2_Liq', 'liq_Al2O3': 'Al2O3_Liq',
    'liq_FeO': 'FeOt_Liq', 'liq_MnO': 'MnO_Liq', 'liq_MgO': 'MgO_Liq',
    'liq_CaO': 'CaO_Liq', 'liq_Na2O': 'Na2O_Liq', 'liq_K2O': 'K2O_Liq',
    'liq_Cr2O3': 'Cr2O3_Liq', 'H2O_Liq': 'H2O_Liq',
    'P2O5_Liq': 'P2O5_Liq',
}


def to_thermobar_schema(df: pd.DataFrame, *, fe3_fet: float = 0.15) -> pd.DataFrame:
    """Return a copy with `_Opx` / `_Liq` suffixed columns expected by
    Thermobar's opx-liq / opx-only wrappers. Missing liq oxides are
    zero-filled to match nb04's convention. `Fe3Fet_Liq` is added with
    the project-wide fixed ratio."""
    out = pd.DataFrame(index=df.index)
    for src, dst in _THERMOBAR_OPX_MAP.items():
        out[dst] = pd.to_numeric(df.get(src, 0.0), errors='coerce')
    for src, dst in _THERMOBAR_LIQ_MAP.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors='coerce')
        else:
            out[dst] = 0.0
    out['Fe3Fet_Liq'] = fe3_fet
    if 'H2O_Liq' not in out.columns:
        out['H2O_Liq'] = 0.0
    return out


def _citation_key(s) -> str:
    s = str(s)
    m = _CITATION_RE.search(s)
    return (m.group(1).strip().lower() + '_' + m.group(2)) if m else s.strip().lower()


def load_arcpl_opx_liq(
    *,
    remove_expetdb_overlap: bool = True,
    lepr_xlsx: Path | None = None,
    expetdb_opx_parquet: Path | None = None,
) -> pd.DataFrame:
    """Return the cleaned opx-liq ArcPL external corpus.

    The pipeline reproduces nb04 Part 3 (`nb04_putirka_benchmark.ipynb`)
    byte-for-byte under the same constants from `config.py`. Every filter
    step logs its row count to stderr via print so callers can audit the
    drop waterfall against the notebook.
    """
    lepr_path = Path(lepr_xlsx or LEPR_XLSX)
    expetdb_path = Path(expetdb_opx_parquet
                        or (DATA_PROC / 'opx_clean_opx_liq.parquet'))

    lepr = pd.read_excel(lepr_path, sheet_name='Opx-Liq')
    df = lepr[lepr['Citation_x'].astype(str).str.contains(
        '_notinLEPR', na=False)].copy()
    print(f'[arcpl_opx] ArcPL subset (from LEPR Opx-Liq): n={len(df)}')

    df = df.rename(columns=_RENAME)
    if 'T_K_x' in df.columns:
        df['T_C'] = df['T_K_x'] - 273.15
    df = df.drop(columns=[c for c in df.columns if c.endswith('_y')],
                 errors='ignore')

    h2o = df.get('H2O_Liq', pd.Series(np.zeros(len(df))))
    df = df[h2o >= 0].copy()

    df['is_vbd'] = df.get(
        'H2O_Liq_Method', pd.Series([''] * len(df))
    ).astype(str).str.contains('VBD|vbd|mass_balance|diff', regex=True, na=False)

    for ox in _ALL_OX:
        if ox in df.columns:
            df[ox] = pd.to_numeric(df[ox], errors='coerce')
    present = [o for o in _ALL_OX if o in df.columns]
    df['oxide_total'] = df[present].sum(axis=1, min_count=5)
    df = df[df['oxide_total'].between(OXIDE_TOTAL_MIN, OXIDE_TOTAL_MAX)].copy()

    df = cation_recalc_6oxy(df, oxides=_ALL_OX)
    df = df[df['cation_sum'].between(CATION_SUM_MIN, CATION_SUM_MAX)].copy()
    df = df.dropna(subset=['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO']).copy()

    df = add_engineered_features(df)
    df['Wo'] = df['Wo_frac'] * 100.0
    df = df[df['Wo'] <= WO_MAX_MOL_PCT].copy()
    df = df[df['P_kbar'] <= P_CEILING_KBAR].copy()

    fe_opx = df['FeO_total'] / 71.844
    mg_opx = df['MgO'] / 40.304
    fe_liq = (df['liq_FeO'] * (1.0 - FE3_FET_RATIO)) / 71.844
    mg_liq = df['liq_MgO'] / 40.304
    kd = (fe_opx / mg_opx) / (fe_liq / mg_liq)
    df = df[(kd >= KD_FEMG_MIN) & (kd <= KD_FEMG_MAX)].copy()
    print(f'[arcpl_opx] after QC + Wo + Kd filters: n={len(df)}')

    if remove_expetdb_overlap:
        expet = pd.read_parquet(expetdb_path)
        expet_keys = set(expet['Citation'].astype(str).map(_citation_key))
        df['_cite_key'] = df['Citation'].astype(str).map(_citation_key)
        n_before = len(df)
        df = df[~df['_cite_key'].isin(expet_keys)].reset_index(drop=True)
        df = df.drop(columns=['_cite_key'])
        print(f'[arcpl_opx] overlap removal vs ExPetDB citations: '
              f'{n_before} -> {len(df)}')

    return df.reset_index(drop=True)
