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
from scripts.figures._style import apply_pub_style, resolve_out_dir, jgr_figsize, jgr_top, jgr_bottom # noqa: E402
from scripts.figures._model_palette import MODEL_COLORS  # noqa: E402
from scripts.figures._labels import TARGET_UNIT, panel_header  # noqa: E402

apply_pub_style()

MODELS_DIR = PROJECT_ROOT / 'models' / 'canonical'
BOOT_CSV = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
SHAP_CSV = PROJECT_ROOT / 'results' / 'shap_importance_winners.csv'
OUT_PARQUET = PROJECT_ROOT / 'results' / 'partial_dependence_across_families.parquet'
OUT_DIR = resolve_out_dir(PROJECT_ROOT)
FIG_STEM = OUT_DIR / 'supp_fig_7'

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

    # Width-only JGR shrink: 12-panel grid needs absolute height for the
    # axes to remain readable. Proportional shrink would crush each
    # subplot to ~1.5×1.5 in and overlap the row labels.
    from scripts.figures._style import JGR_2COL_IN, is_jgr_mode  # noqa
    target_w = JGR_2COL_IN if is_jgr_mode() else 11.0
    fig, axes = plt.subplots(4, 3, figsize=(target_w, 10))
    panel_letters = ['a', 'b', 'c', 'd']
    for ri, (pipe, track, tgt) in enumerate(CELLS):
        feats = top_features_for(shap, track, tgt, n=3)
        unit = TARGET_UNIT[tgt]
        for ci, feat in enumerate(feats):
            ax = axes[ri, ci]
            sub = df[(df.track == track) & (df.target == tgt)
                     & (df.feature == feat)]
            for family in FAMILIES:
                fs_rows = sub[sub.family == family]
                if len(fs_rows) == 0:
                    continue
                ax.plot(fs_rows['grid_value'], fs_rows['pd_value'],
                        color=MODEL_COLORS.get(family, '#999999'),
                        label=family, lw=1.3, alpha=0.85)
            ax.set_title(f'({panel_letters[ri]}{ci+1}) {feat}', fontsize=10)
            ax.set_xlabel(feat, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f'{panel_header(track, tgt)}\npredicted '
                              f'({unit})', fontsize=9)
            ax.grid(True)
    # Build a combined legend across all panels — single panel may not
    # cover every family if its feature set differs.
    seen = set()
    handles, labels = [], []
    for ax in axes.ravel():
        h_, l_ = ax.get_legend_handles_labels()
        for h, lbl in zip(h_, l_):
            if lbl not in seen:
                seen.add(lbl); handles.append(h); labels.append(lbl)
    # Reorder to canonical MODEL_ORDER
    order = sorted(range(len(labels)),
                   key=lambda i: FAMILIES.index(labels[i])
                   if labels[i] in FAMILIES else 999)
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=8, fontsize=9)
    fig.suptitle(
        'Partial dependence across families (top-3 SHAP features per cell)\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190), seed 42'
    )
    plt.subplots_adjust(top=jgr_top(0.90), bottom=jgr_bottom(0.10), left=0.10, right=0.97,
                        hspace=0.65, wspace=0.30)
    fig.savefig(f'{FIG_STEM}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{FIG_STEM}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)

    caption = (
        'Supp. Figure 7. Partial-dependence functions across the eight '
        'tuned families for the top-three winner-SHAP features in each of '
        'the four opx (track, target) cells (rows). Family curves are '
        'colored using the paper-wide locked palette. TabPFN is omitted '
        '(no fitted estimator exposed to sklearn.partial_dependence). '
        'Convergence on the same monotone shape across families supports '
        'the corresponding feature direction; divergence flags a feature '
        'whose effect is family-specific. Source: '
        'results/partial_dependence_across_families.parquet.'
    )
    (OUT_DIR / 'supp_fig_7.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {FIG_STEM}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
