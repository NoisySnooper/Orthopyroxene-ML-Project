#!/usr/bin/env python3
"""Phase 2 P2.5: partial dependence across families per cell.

For each of 4 shipped opx cells, select the top-3 features by the
winner's SHAP importance, compute sklearn partial_dependence for each of
the tuned tree/ensemble/linear families on each selected feature. Exports
long-format parquet with (family, cell, feature, grid_value, pd_value)
for plotting.

TabPFN excluded (no fitted estimator). All computations on canonical seed
42 test set.

Output:
  results/partial_dependence_across_families.parquet
  figures/core/Core_17_fig_partial_dependence.pdf (+ .png)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_train_test import prepare_train_test  # noqa: E402

MODELS_DIR = PROJECT_ROOT / 'models' / 'canonical'
BOOT_CSV = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
SHAP_CSV = PROJECT_ROOT / 'results' / 'shap_importance_winners.csv'
OUT_PARQUET = PROJECT_ROOT / 'results' / 'partial_dependence_across_families.parquet'
FIG_STEM = PROJECT_ROOT / 'figures' / 'core' / 'Core_17_fig_partial_dependence'

CELLS = [
    ('opx', 'opx_liq', 'T_C'),
    ('opx', 'opx_liq', 'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
]
FAMILIES = ['RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM',
            'ElasticNet', 'MLP']


def canonical_path(pipeline, model, target, track, feature_set):
    return (MODELS_DIR / pipeline
            / f'base_{model}_{target}_{track}_{feature_set}.joblib')


def best_fs_for(boot, pipeline, track, target, family):
    sub = boot[(boot.pipeline == pipeline) & (boot.track == track)
               & (boot.target == target) & (boot.model == family)]
    if len(sub) == 0:
        return None
    return sub.sort_values('rmse_point').iloc[0]['feature_set']


def top_features_for(shap, track, target, n=3):
    sub = shap[(shap.track == track) & (shap.target == target)].copy()
    sub = sub.sort_values('rank').head(n)
    return sub['feature'].tolist()


def main():
    boot = pd.read_csv(BOOT_CSV)
    shap = pd.read_csv(SHAP_CSV)
    rows = []

    for pipe, track, tgt in CELLS:
        winner_feats = top_features_for(shap, track, tgt, n=3)
        print(f'\n=== {track}/{tgt} top features: {winner_feats} ===')

        for family in FAMILIES:
            fs = best_fs_for(boot, pipe, track, tgt, family)
            if fs is None:
                continue
            mp = canonical_path(pipe, family, tgt, track, fs)
            if not mp.exists():
                continue
            try:
                est = joblib.load(mp)
                d = prepare_train_test(pipe, track, tgt, fs)
                feat_names = list(d['feat_names'])
                X = d['X_te']
            except Exception as e:
                print(f'  {family}/{fs}: skip {e}')
                continue

            t0 = time.time()
            for feat in winner_feats:
                if feat not in feat_names:
                    # PD only on features in this family's feature set
                    continue
                idx = feat_names.index(feat)
                try:
                    pd_out = partial_dependence(
                        est, X, [idx], grid_resolution=30, kind='average')
                    grid = pd_out['grid_values'][0]
                    avg = pd_out['average'][0]
                    for g, v in zip(grid, avg):
                        rows.append({
                            'pipeline':    pipe,
                            'track':       track,
                            'target':      tgt,
                            'family':      family,
                            'feature_set': fs,
                            'feature':     feat,
                            'grid_value':  float(g),
                            'pd_value':    float(v),
                        })
                except Exception as e:
                    print(f'    FAIL {family} {feat}: {e}')
            print(f'  {family}/{fs}: {time.time()-t0:.1f}s')

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f'\nwrote {OUT_PARQUET}: {len(df)} rows, '
          f'{df.family.nunique()} families, {df.feature.nunique()} features')

    # Figure Core_17: 4x3 grid (4 cells x 3 features), one line per family
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    cmap = plt.get_cmap('tab10')
    fam_color = {f: cmap(i) for i, f in enumerate(FAMILIES)}
    for ri, (pipe, track, tgt) in enumerate(CELLS):
        feats = top_features_for(shap, track, tgt, n=3)
        for ci, feat in enumerate(feats):
            ax = axes[ri, ci]
            sub = df[(df.track == track) & (df.target == tgt)
                     & (df.feature == feat)]
            for family in FAMILIES:
                fs_rows = sub[sub.family == family]
                if len(fs_rows) == 0:
                    continue
                ax.plot(fs_rows['grid_value'], fs_rows['pd_value'],
                        color=fam_color[family], label=family, lw=1.3,
                        alpha=0.85)
            unit = '°C' if tgt == 'T_C' else 'kbar'
            ax.set_title(f'{track}/{tgt}: {feat}', fontsize=9, loc='left')
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel(f'predicted {tgt} ({unit})', fontsize=8)
            ax.tick_params(labelsize=7)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc='best', ncol=2)
    fig.suptitle('Partial dependence across families (top-3 winner SHAP '
                 'features per cell)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f'{FIG_STEM}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{FIG_STEM}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {FIG_STEM}.pdf')


if __name__ == '__main__':
    main()
