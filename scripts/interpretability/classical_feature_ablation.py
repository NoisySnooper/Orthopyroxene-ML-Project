#!/usr/bin/env python3
"""Phase 2 P2.3: classical-feature ablation.

For the 4 shipped opx cells, retrain the winning family on ONLY the
features in the corresponding Putirka (2008) classical expression. RMSE
vs full-feature winner answers: does the ML rely on classical petrologic
information, or does it extract signal beyond it?

Winning families per cell (from Phase 1 T2 roster, seed 42):
  opx_liq  / T_C    -> ElasticNet / raw
  opx_liq  / P_kbar -> MLP / raw
  opx_only / T_C    -> LightGBM / alr
  opx_only / P_kbar -> RF / pwlr

Classical feature sets (mapped onto v10 raw feature names):
  opx_liq  T_C   (Putirka 28a)  : raw_liq_MgO, raw_liq_FeO, raw_liq_SiO2,
                                   raw_liq_Al2O3, raw_liq_CaO, raw_liq_Na2O,
                                   raw_liq_K2O
  opx_liq  P_kbar (Putirka 29a) : raw_liq_SiO2, raw_liq_Al2O3, Al_VI, Mg_num,
                                   raw_SiO2, raw_Al2O3
  opx_only T_C   (Brey-Kohler)  : raw_SiO2, raw_Al2O3, raw_MgO, raw_CaO,
                                   raw_FeO_total, En_frac, Fs_frac, Wo_frac,
                                   Mg_num
  opx_only P_kbar (Putirka 29c) : raw_Al2O3, raw_CaO, raw_Cr2O3, Al_VI, Al_IV,
                                   MgTs

Output: results/classical_feature_ablation.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import bootstrap_rmse_ci  # noqa: E402
from src.prepare_train_test import prepare_train_test  # noqa: E402

MODELS_DIR = PROJECT_ROOT / 'models' / 'canonical'
BOOT_CSV = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
OPTUNA_JSON = PROJECT_ROOT / 'results' / 'optuna_best_params_opx.json'
OUT_CSV = PROJECT_ROOT / 'results' / 'classical_feature_ablation.csv'


CELLS = [
    dict(pipeline='opx', track='opx_liq', target='T_C',
         family='ElasticNet', feature_set='raw',
         classical=['raw_liq_MgO', 'raw_liq_FeO', 'raw_liq_SiO2',
                    'raw_liq_Al2O3', 'raw_liq_CaO', 'raw_liq_Na2O',
                    'raw_liq_K2O']),
    dict(pipeline='opx', track='opx_liq', target='P_kbar',
         family='MLP', feature_set='raw',
         classical=['raw_liq_SiO2', 'raw_liq_Al2O3', 'Al_VI', 'Mg_num',
                    'raw_SiO2', 'raw_Al2O3']),
    dict(pipeline='opx', track='opx_only', target='T_C',
         family='LightGBM', feature_set='alr',
         classical=None),  # alr features; handle separately below
    dict(pipeline='opx', track='opx_only', target='P_kbar',
         family='RF', feature_set='pwlr',
         classical=None),  # pwlr features; handle separately below
]


def raw_classical_for(track, target):
    """Classical raw feature subset for alr/pwlr cells — reuse raw
    features and train on raw. Classical ablation must stay in 'raw'
    space because classical formulas use raw oxides."""
    if track == 'opx_only' and target == 'T_C':
        return ['raw_SiO2', 'raw_Al2O3', 'raw_MgO', 'raw_CaO',
                'raw_FeO_total', 'En_frac', 'Fs_frac', 'Wo_frac',
                'Mg_num']
    if track == 'opx_only' and target == 'P_kbar':
        return ['raw_Al2O3', 'raw_CaO', 'raw_Cr2O3', 'Al_VI', 'Al_IV',
                'MgTs']
    return None


def build_family_estimator(family, params):
    if family == 'ElasticNet':
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', ElasticNet(random_state=42, max_iter=20000, **params)),
        ])
    if family == 'MLP':
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        p = dict(params)
        n_layers = p.pop('n_layers', 1)
        layer_size = p.pop('layer_size', 64)
        p['hidden_layer_sizes'] = tuple([layer_size] * n_layers)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', MLPRegressor(random_state=42, max_iter=5000, **p)),
        ])
    if family == 'LightGBM':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=42, verbose=-1, **params)
    if family == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    raise ValueError(f'unknown family {family}')


def load_optuna_params(family, track, target, feature_set):
    d = json.load(open(OPTUNA_JSON))
    for r in d['results']:
        if (r['model'] == family and r['track'] == track
                and r['target'] == target and r['feature_set'] == feature_set):
            return r['best_params']
    raise KeyError(f'{family}/{track}/{target}/{feature_set} not in optuna')


def interpret(pct):
    if pct < 5:
        return 'ML uses classical information; black-box concern unfounded for this cell'
    if pct < 15:
        return 'ML uses classical information plus smooth residual signal; interpretable'
    return 'ML relies on non-classical features beyond classical petrologic intuition; warrants feature-level investigation'


def main():
    boot = pd.read_csv(BOOT_CSV)
    rows = []
    for cell in CELLS:
        pipe = cell['pipeline']; track = cell['track']; tgt = cell['target']
        family = cell['family']; fs = cell['feature_set']
        print(f'\n=== {track} {tgt} {family}/{fs} ===')

        # Full-feature reference from bootstrap CSV (canonical seed 42)
        ref = boot[(boot.track == track) & (boot.target == tgt)
                   & (boot.model == family) & (boot.feature_set == fs)].iloc[0]
        full_rmse = float(ref['rmse_seed42'])
        full_lo = float(ref['rmse_ci_lo']); full_hi = float(ref['rmse_ci_hi'])

        # Always retrain ablation in the raw feature space (classical
        # formulas are stated in raw oxides). For raw-winner cells this
        # is bit-consistent with the winner's feature space; for alr/pwlr
        # winner cells the ablation uses a different feature space and
        # we flag this in the interpretation by footnote in the CSV.
        ablation_fs = 'raw'
        d = prepare_train_test(pipe, track, tgt, ablation_fs)
        feat_names = d['feat_names']

        classical = cell['classical'] or raw_classical_for(track, tgt)
        missing = [c for c in classical if c not in feat_names]
        if missing:
            print(f'  MISSING classical features in raw set: {missing}')
        keep_idx = [feat_names.index(c) for c in classical if c in feat_names]
        X_tr = d['X_tr'][:, keep_idx]
        X_te = d['X_te'][:, keep_idx]

        params = load_optuna_params(family, track, tgt,
                                    fs if fs in ('raw', 'alr', 'pwlr') else 'raw')
        est = build_family_estimator(family, params)
        t0 = time.time()
        est.fit(X_tr, d['y_tr'])
        dt = time.time() - t0
        y_pred = est.predict(X_te)
        c_rmse, c_lo, c_hi = bootstrap_rmse_ci(
            d['y_te'], y_pred, n_boot=500, random_state=42)
        delta = c_rmse - full_rmse
        pct = 100.0 * delta / full_rmse
        print(f'  full={full_rmse:.2f} [{full_lo:.2f}, {full_hi:.2f}] '
              f'classical={c_rmse:.2f} [{c_lo:.2f}, {c_hi:.2f}] '
              f'delta={delta:+.2f} ({pct:+.1f}%) t={dt:.1f}s')

        rows.append({
            'track':                  track,
            'target':                 tgt,
            'family':                 family,
            'winner_feature_set':     fs,
            'ablation_feature_set':   ablation_fs,
            'n_classical_features':   len(keep_idx),
            'classical_features':     ';'.join(classical),
            'full_feature_rmse':      full_rmse,
            'full_feature_ci':        f'[{full_lo:.2f}, {full_hi:.2f}]',
            'classical_feature_rmse': c_rmse,
            'classical_feature_ci':   f'[{c_lo:.2f}, {c_hi:.2f}]',
            'rmse_delta':             delta,
            'rmse_delta_pct':         pct,
            'interpretation':         interpret(abs(pct)),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f'\nwrote {OUT_CSV}')
    print(out[['track', 'target', 'full_feature_rmse',
               'classical_feature_rmse', 'rmse_delta_pct',
               'interpretation']].to_string(index=False))


if __name__ == '__main__':
    main()
