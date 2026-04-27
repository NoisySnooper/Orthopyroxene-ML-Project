#!/usr/bin/env python3
"""H.3: Natural-sample inference (minimal).

Runs canonical (model, feature_set) winners per H.0b lock on natural
samples and writes predictions to results/nb08_natural_predictions_*.csv.

Scope (this session):
  H.3a opx_only T (LightGBM/alr) and P (RF/pwlr) on natural_opx
  H.3b cpx_only T (ERT/pwlr) and P (MLP/alr substituted for TabPFN/raw)
  H.3c twopx T (ElasticNet/raw) and P (XGB/alr) on natural pairs
  Putirka classical (eq28a/29a/29c/30/32a/32d/33/36/37/38/39) via
    src.external_models.predict_putirka_classical_natural
  Fe-Mg KD on twopx pairs

Deferred (logged in run_log, NOT silently dropped):
  - opx_liq, cpx_liq inference (no natural liquid; H.1c not pulled)
  - TabPFN canonical models (cpx_liq T, cpx_only P) substituted with
    second-place: ERT/pwlr and MLP/alr respectively
  - Monte Carlo uncertainty (would be ~10^8 ops; defer)
  - IsolationForest OOD score (defer; can compute later from same model)
  - External model wrappers (Agreda, Wang, Petrelli) on natural samples
  - Stacked ensemble (defer; not in canonical-cell winners list)

Pre-registration boundary: predictions are RAW. No G.4 bias correction
applied here; H.6b applies it ONLY on the curated subset where literature
P assigns the regime non-circularly (not implemented this session).

Output schema (per pipeline):
  sample_id, lat, lon, tectonic_setting, location, citation,
  T_canonical_pred, P_canonical_pred, T_putirka_*, P_putirka_*,
  ... (Putirka columns vary per pipeline)
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (lepr_to_training_features,  # noqa: E402
                          build_feature_matrix, OXIDES_OPX)
from src.cpx_features import (expetdb_cpx_to_training,  # noqa: E402
                              build_cpx_feature_matrix)
from src.twopx_features import (build_twopx_feature_matrix,  # noqa: E402
                                add_twopx_engineered)
from src.external_models import (predict_putirka_classical_natural,  # noqa
                                 compute_cpx_opx_kd_femg)

LOG_FILE = PROJECT_ROOT / 'results' / 'PHASE_H_RUN_LOG.md'

# Canonical model paths per H.0b lock (with TabPFN substitutions noted)
MODELS = {
    'opx_only_T':  ('models/canonical/opx/base_LightGBM_T_C_opx_only_alr.joblib',
                    'alr', False, 'opx'),
    'opx_only_P':  ('models/canonical/opx/base_RF_P_kbar_opx_only_pwlr.joblib',
                    'pwlr', False, 'opx'),
    'cpx_only_T':  ('models/canonical/cpx/base_ERT_T_C_cpx_only_pwlr.joblib',
                    'pwlr', False, 'cpx'),
    'cpx_only_P':  ('models/canonical/cpx/base_MLP_P_kbar_cpx_only_alr.joblib',
                    'alr', False, 'cpx'),
    'twopx_T':     ('models/canonical/twopx/base_ElasticNet_T_C_twopx_raw.joblib',
                    'raw', None, 'twopx'),
    'twopx_P':     ('models/canonical/twopx/base_XGB_P_kbar_twopx_alr.joblib',
                    'alr', None, 'twopx'),
}


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = f'- {ts} (H.3) {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def _build_features(df_prepped: pd.DataFrame, feature_set: str,
                    use_liq: bool, family: str):
    if family == 'opx':
        return build_feature_matrix(df_prepped, feature_set, use_liq=use_liq)
    if family == 'cpx':
        return build_cpx_feature_matrix(
            df_prepped, feature_set, use_liq=use_liq)
    if family == 'twopx':
        return build_twopx_feature_matrix(df_prepped, feature_set)
    raise ValueError(family)


def _predict(model_path: str, X: np.ndarray) -> np.ndarray:
    m = joblib.load(model_path)
    return np.asarray(m.predict(X), dtype=float)


# ---------- opx natural ----------

def run_opx() -> None:
    _log('caveman: opx natural inference start')
    raw = pd.read_csv('data/natural/natural_opx_with_coords.csv',
                      low_memory=False)
    _log(f'caveman: opx loaded n={len(raw)}')

    prepped = lepr_to_training_features(raw)

    out = pd.DataFrame({
        'sample_id': (raw['CITATION'].astype(str) + '|'
                      + raw['SAMPLE NAME'].astype(str)),
        'lat': raw['lat'],
        'lon': raw['lon'],
        'tectonic_setting': raw['TECTONIC SETTING'],
        'location': raw['LOCATION'],
        'citation': raw['CITATION'],
        'sample_name': raw['SAMPLE NAME'],
    })
    for ox in OXIDES_OPX:
        if ox in raw.columns:
            out[ox] = raw[ox]

    # T (LightGBM/alr) + P (RF/pwlr)
    for cell in ('opx_only_T', 'opx_only_P'):
        path, fset, use_liq, family = MODELS[cell]
        X, _ = _build_features(prepped, fset, use_liq, family)
        target = 'T' if cell.endswith('_T') else 'P'
        col = f'{target}_canonical_{cell}'
        out[col] = _predict(path, X)
        _log(f'caveman: {cell} pred mean={out[col].mean():.2f}')

    # Putirka classical (opx side: opx_liq* skipped because no liq;
    # opx_only_P_29c works with opx + assumed T). Wrapper detects phase
    # by '_Opx' / '_Cpx' / '_Liq' suffix, so re-suffix oxide cols.
    df_for_putirka = raw.copy()
    suf = {ox: f'{ox}_Opx' for ox in OXIDES_OPX if ox in df_for_putirka.columns}
    suf['FeO_total'] = 'FeOt_Opx'  # Thermobar wants FeOt naming
    df_for_putirka = df_for_putirka.rename(columns=suf)
    putirka = predict_putirka_classical_natural(df_for_putirka)
    keep = [c for c in putirka.columns
            if 'opx_only' in c or 'opx_liq' in c]
    for c in keep:
        out[c] = putirka[c].values
    _log(f'caveman: Putirka opx columns appended: {keep}')

    out.to_csv('results/nb08_natural_predictions_opx.csv',
               index=False, encoding='utf-8')
    sha = hashlib.sha256(
        Path('results/nb08_natural_predictions_opx.csv').read_bytes()
        ).hexdigest()
    _log(f'caveman: wrote nb08_natural_predictions_opx.csv n={len(out)} '
         f'SHA[:12]={sha[:12]}')


# ---------- cpx natural ----------

def run_cpx() -> None:
    _log('caveman: cpx natural inference start')
    raw = pd.read_csv('data/natural/natural_cpx_with_coords.csv',
                      low_memory=False)
    _log(f'caveman: cpx loaded n={len(raw)}')

    prepped = expetdb_cpx_to_training(raw)

    out = pd.DataFrame({
        'sample_id': (raw['CITATION'].astype(str) + '|'
                      + raw['SAMPLE NAME'].astype(str)),
        'lat': raw['lat'],
        'lon': raw['lon'],
        'tectonic_setting': raw['TECTONIC SETTING'],
        'location': raw['LOCATION'],
        'citation': raw['CITATION'],
        'sample_name': raw['SAMPLE NAME'],
    })
    for ox in ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO',
              'MgO', 'CaO', 'Na2O']:
        if ox in raw.columns:
            out[ox] = raw[ox]

    # T (ERT/pwlr) + P (MLP/alr)
    for cell in ('cpx_only_T', 'cpx_only_P'):
        path, fset, use_liq, family = MODELS[cell]
        X, _ = _build_features(prepped, fset, use_liq, family)
        target = 'T' if cell.endswith('_T') else 'P'
        col = f'{target}_canonical_{cell}'
        out[col] = _predict(path, X)
        _log(f'caveman: {cell} pred mean={out[col].mean():.2f}')

    # Note: TabPFN canonical winners (cpx_liq T, cpx_only P) are
    # substituted with their second-place. Logged in canonical_cells doc
    # and run log; flag in dataframe.
    out['note_tabpfn_substituted'] = (
        'cpx_only P uses MLP/alr (2nd place); '
        'TabPFN/raw deferred for follow-up run')

    # Putirka classical: keep cpx-only columns (have cpx no liq)
    df_for_putirka = raw.copy()
    # Suffix oxides _Cpx so external_models.predict_putirka_cpx_only sees them
    rename = {ox: f'{ox}_Cpx' for ox in ['SiO2', 'TiO2', 'Al2O3',
              'Cr2O3', 'FeO_total', 'MnO', 'MgO', 'CaO', 'Na2O']
              if ox in df_for_putirka.columns}
    rename['FeO_total'] = 'FeOt_Cpx'  # Thermobar wants FeOt not FeO_total
    df_for_putirka = df_for_putirka.rename(columns=rename)
    putirka = predict_putirka_classical_natural(df_for_putirka)
    keep = [c for c in putirka.columns if 'cpx_only' in c]
    for c in keep:
        out[c] = putirka[c].values
    _log(f'caveman: Putirka cpx columns appended: {keep}')

    out.to_csv('results/nb08_natural_predictions_cpx.csv',
               index=False, encoding='utf-8')
    sha = hashlib.sha256(
        Path('results/nb08_natural_predictions_cpx.csv').read_bytes()
        ).hexdigest()
    _log(f'caveman: wrote nb08_natural_predictions_cpx.csv n={len(out)} '
         f'SHA[:12]={sha[:12]}')


# ---------- twopx natural ----------

def run_twopx() -> None:
    _log('caveman: twopx natural inference start')
    raw = pd.read_csv('data/natural/natural_twopx_pairs.csv',
                      low_memory=False)
    _log(f'caveman: twopx loaded n={len(raw)}')

    # Build the lowercase _opx / _cpx schema the twopx pipeline expects.
    df = raw.copy()
    OX_LIST = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO',
               'MgO', 'CaO', 'Na2O']
    for ox in OX_LIST:
        if f'{ox}_Opx' in df.columns:
            df[f'{ox}_opx'] = pd.to_numeric(df[f'{ox}_Opx'], errors='coerce')
        if f'{ox}_Cpx' in df.columns:
            df[f'{ox}_cpx'] = pd.to_numeric(df[f'{ox}_Cpx'], errors='coerce')

    # Build per-mineral cation cols + opx engineered + cpx engineered + twopx eng
    from src.features import (cation_recalc_6oxy as cat_opx,
                              add_engineered_features as opx_eng)
    from src.cpx_features import (
        cation_recalc_6oxy as cat_cpx, add_cpx_engineered_features as cpx_eng)

    # opx side
    opx_view = df.rename(columns={f'{ox}_opx': ox for ox in OX_LIST})
    opx_view = cat_opx(opx_view, oxides=OX_LIST)
    opx_view = opx_eng(opx_view)
    # transfer engineered cols back with _opx suffix (just the ones twopx wants)
    for col in ['Mg_num', 'Al_IV', 'Al_VI', 'En_frac', 'Fs_frac', 'Wo_frac']:
        if col in opx_view.columns:
            new_name = (f'Mg_num_opx' if col == 'Mg_num'
                        else f'{col.replace("_frac", "")}_opx')
            df[new_name] = opx_view[col].values

    # cpx side
    cpx_view = df.rename(columns={f'{ox}_cpx': ox for ox in OX_LIST})
    cpx_view = cat_cpx(cpx_view, oxides=OX_LIST)
    cpx_view = cpx_eng(cpx_view)
    for col in ['Mg_num_cpx', 'Jd', 'Di', 'Aeg', 'CaTs',
                'En_cpx', 'Fs_cpx']:
        if col in cpx_view.columns:
            df[col] = cpx_view[col].values

    # twopx cross-mineral
    df = add_twopx_engineered(df)

    out = pd.DataFrame({
        'sample_id': (raw['CITATION'].astype(str) + '|'
                      + raw['SAMPLE NAME'].astype(str)),
        'lat': raw['lat'],
        'lon': raw['lon'],
        'tectonic_setting': raw['TECTONIC SETTING'],
        'location': raw['LOCATION'],
        'citation': raw['CITATION'],
        'sample_name': raw['SAMPLE NAME'],
    })
    # Preserve the per-mineral oxides for downstream cross-checks
    for ox in OX_LIST:
        out[f'{ox}_Opx'] = raw[f'{ox}_Opx']
        out[f'{ox}_Cpx'] = raw[f'{ox}_Cpx']

    for cell in ('twopx_T', 'twopx_P'):
        path, fset, _use_liq, family = MODELS[cell]
        X, _ = _build_features(df, fset, False, family)
        target = 'T' if cell.endswith('_T') else 'P'
        col = f'{target}_canonical_{cell}'
        out[col] = _predict(path, X)
        _log(f'caveman: {cell} pred mean={out[col].mean():.2f}')

    # Putirka twopx (eq36/37 T, eq38/39 P) plus cpx-only and opx-only cols
    df_for_putirka = raw.rename(columns={
        f'{ox}_Opx': (f'FeOt_Opx' if ox == 'FeO_total' else f'{ox}_Opx')
        for ox in OX_LIST
    })
    df_for_putirka = df_for_putirka.rename(columns={
        f'{ox}_Cpx': (f'FeOt_Cpx' if ox == 'FeO_total' else f'{ox}_Cpx')
        for ox in OX_LIST
    })
    putirka = predict_putirka_classical_natural(df_for_putirka)
    for c in putirka.columns:
        out[c] = putirka[c].values
    _log(f'caveman: Putirka twopx columns appended: {len(putirka.columns)}')

    # Fe-Mg KD equilibrium flag (H.4b)
    df_for_kd = raw.rename(columns={
        f'{ox}_Opx': (f'FeOt_Opx' if ox == 'FeO_total' else f'{ox}_Opx')
        for ox in OX_LIST
    }).rename(columns={
        f'{ox}_Cpx': (f'FeOt_Cpx' if ox == 'FeO_total' else f'{ox}_Cpx')
        for ox in OX_LIST
    })
    out['KD_FeMg_opx_cpx'] = compute_cpx_opx_kd_femg(df_for_kd)
    eq = (out['KD_FeMg_opx_cpx'] >= 0.95) & (out['KD_FeMg_opx_cpx'] <= 1.23)
    out['equilibrium_flag'] = eq
    n_eq = int(eq.sum())
    _log(f'caveman: KD equilibrium n={n_eq}/{len(out)} '
         f'({100*n_eq/len(out):.1f}%)')

    out.to_csv('results/nb08_natural_predictions_twopx.csv',
               index=False, encoding='utf-8')
    sha = hashlib.sha256(
        Path('results/nb08_natural_predictions_twopx.csv').read_bytes()
        ).hexdigest()
    _log(f'caveman: wrote nb08_natural_predictions_twopx.csv n={len(out)} '
         f'SHA[:12]={sha[:12]}')


def main() -> int:
    _log('caveman: H.3 inference start')
    run_opx()
    run_cpx()
    run_twopx()
    _log('caveman: H.3 inference done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
