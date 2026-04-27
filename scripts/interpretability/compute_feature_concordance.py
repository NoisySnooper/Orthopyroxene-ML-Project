#!/usr/bin/env python3
"""Phase 2 P2.2: permutation importance across families per cell.

For each of 8 (track, target) cells and 8 tuned families, compute
permutation importance at seed=42 on the held-out test set using that
family's best feature set (from bootstrap_rmse_cis_all_cells.csv). Rank
features within each (cell, family); compute pairwise Spearman rank
correlation across family pairs.

TabPFN is excluded from this pass — we only hold sample-level predictions
for it, not a fitted estimator compatible with sklearn's permutation
importance API. Documented in the output CSV (n_families_compared=8).

Output:
  results/feature_concordance_permutation_importance.csv
  results/feature_concordance_spearman_matrix.csv
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_train_test import prepare_train_test  # noqa: E402

MODELS_DIR = PROJECT_ROOT / 'models' / 'canonical'
BOOT_CSV = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
OUT_IMP = PROJECT_ROOT / 'results' / 'feature_concordance_permutation_importance.csv'
OUT_SPR = PROJECT_ROOT / 'results' / 'feature_concordance_spearman_matrix.csv'

CELLS = [
    ('opx', 'opx_liq', 'T_C'),
    ('opx', 'opx_liq', 'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
    ('cpx', 'cpx_liq', 'T_C'),
    ('cpx', 'cpx_liq', 'P_kbar'),
    ('cpx', 'cpx_only', 'T_C'),
    ('cpx', 'cpx_only', 'P_kbar'),
]
FAMILIES = ['RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM',
            'ElasticNet', 'MLP']


def best_fs_for(boot, pipeline, track, target, family):
    sub = boot[(boot.pipeline == pipeline) & (boot.track == track)
               & (boot.target == target) & (boot.model == family)]
    if len(sub) == 0:
        return None
    return sub.sort_values('rmse_point').iloc[0]['feature_set']


def canonical_path(pipeline, model, target, track, feature_set):
    return (MODELS_DIR / pipeline
            / f'base_{model}_{target}_{track}_{feature_set}.joblib')


def main():
    boot = pd.read_csv(BOOT_CSV)
    imp_rows = []
    spearman_blocks = []

    prepared = {}

    for pipeline, track, target in CELLS:
        print(f'\n=== {pipeline}/{track}/{target} ===')
        t0 = time.time()
        fam_rankings = {}
        for family in FAMILIES:
            fs = best_fs_for(boot, pipeline, track, target, family)
            if fs is None:
                print(f'  skip {family}: no bootstrap row')
                continue
            key = (pipeline, track, target, fs)
            if key not in prepared:
                prepared[key] = prepare_train_test(pipeline, track, target, fs)
            d = prepared[key]
            mp = canonical_path(pipeline, family, target, track, fs)
            if not mp.exists():
                print(f'  MISSING: {mp.name}')
                continue
            try:
                est = joblib.load(mp)
                tstart = time.time()
                r = permutation_importance(
                    est, d['X_te'], d['y_te'],
                    n_repeats=20, random_state=42, n_jobs=1,
                    scoring='neg_root_mean_squared_error',
                )
                dt = time.time() - tstart
                print(f'  {family}/{fs}: {dt:.1f}s')
            except Exception as e:
                print(f'  FAILED {family}/{fs}: {e}')
                continue
            feats = list(d['feat_names'])
            # Rank: higher importance -> rank 1 (most important)
            order = np.argsort(-r.importances_mean)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            for i, feat in enumerate(feats):
                imp_rows.append({
                    'pipeline':        pipeline,
                    'track':           track,
                    'target':          target,
                    'feature_set':     fs,
                    'family':          family,
                    'feature_name':    feat,
                    'importance_mean': float(r.importances_mean[i]),
                    'importance_std':  float(r.importances_std[i]),
                    'rank':            int(ranks[i]),
                })
            fam_rankings[family] = dict(zip(feats, ranks.astype(int).tolist()))

        # Build per-cell Spearman matrix over the intersection of features
        families_present = list(fam_rankings.keys())
        for i, fa in enumerate(families_present):
            for j, fb in enumerate(families_present):
                if j <= i:
                    continue
                common = set(fam_rankings[fa].keys()) & set(fam_rankings[fb].keys())
                if len(common) < 3:
                    rho = np.nan
                else:
                    cf = sorted(common)
                    ra = np.array([fam_rankings[fa][f] for f in cf])
                    rb = np.array([fam_rankings[fb][f] for f in cf])
                    rho, _ = spearmanr(ra, rb)
                spearman_blocks.append({
                    'pipeline':  pipeline,
                    'track':     track,
                    'target':    target,
                    'family_a':  fa,
                    'family_b':  fb,
                    'n_common_features': len(common),
                    'spearman_rho': float(rho) if np.isfinite(rho) else np.nan,
                })
        print(f'  cell elapsed: {time.time()-t0:.1f}s')

    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv(OUT_IMP, index=False)
    print(f'\nwrote {OUT_IMP}: {len(imp_df)} rows')

    spr_df = pd.DataFrame(spearman_blocks)
    spr_df.to_csv(OUT_SPR, index=False)
    print(f'wrote {OUT_SPR}: {len(spr_df)} pair rows')

    # Quick median summary
    print('\nMedian Spearman rho per cell:')
    med = spr_df.groupby(['pipeline', 'track', 'target'])['spearman_rho'].median()
    print(med.to_string())


if __name__ == '__main__':
    main()
