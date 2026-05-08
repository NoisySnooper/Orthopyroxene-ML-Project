#!/usr/bin/env python3
"""main_fig_8: SHAP beeswarm for the per-cell winners (opx only).

Seaborn-based per-sample SHAP visualization. One panel per opx (track,
target) cell, top-10 features ranked by mean |SHAP|. Each dot is one
held-out sample; x-axis is the SHAP value in target native units; color
is the underlying feature value normalized per feature (low = blue,
high = red), rendered via seaborn.stripplot.

Tree models (RF, LightGBM, ERT) use TreeExplainer (exact); ElasticNet
uses LinearExplainer (exact). MLP cells store permutation-importance
values broadcast across samples — for those panels every dot in a row
sits at the same x and the panel is annotated as a permutation surrogate.

Sources:
  - results/shap_values_winners.npz (per-sample SHAP, dumped by
    scripts/shap/run_shap_winners.py)
  - results/shap_importance_winners.csv (cell -> model/feature_set map
    and mean_abs_shap rankings)
  - prepare_train_test() for the matching held-out feature matrices
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._style import apply_pub_style, make_fig, resolve_out_dir  # noqa: E402
from scripts.figures._labels import TARGET_UNIT, panel_header  # noqa: E402
from src.prepare_train_test import prepare_train_test  # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

PANELS = [
    ('opx', 'opx_liq',  'T_C',    'a'),
    ('opx', 'opx_liq',  'P_kbar', 'b'),
    ('opx', 'opx_only', 'T_C',    'c'),
    ('opx', 'opx_only', 'P_kbar', 'd'),
]

TABPFN_SCORECARD_WINS = {('opx', 'opx_only', 'T_C'),
                         ('opx', 'opx_only', 'P_kbar')}

TOP_N = 10

EXPLAINER_NOTE = {
    'ElasticNet': 'LinearExplainer, exact',
    'MLP':        'perm. importance, SHAP surrogate',
    'RF':         'TreeExplainer, exact',
    'ERT':        'TreeExplainer, exact',
    'LightGBM':   'TreeExplainer, exact',
    'XGB':        'TreeExplainer, exact',
    'CatBoost':   'TreeExplainer, exact',
    'GB':         'TreeExplainer, exact',
}

PALETTE = 'coolwarm'


def _load_panel(npz, importance_df, pipeline, track, target):
    """Return long-format dataframe for the top-N features of one cell."""
    cell = importance_df[(importance_df.pipeline == pipeline)
                         & (importance_df.track == track)
                         & (importance_df.target == target)]
    if cell.empty:
        return None
    model = cell['model'].iloc[0]
    fs = cell['feature_set'].iloc[0]
    key = f'{pipeline}__{track}__{target}__{model}__{fs}'
    if key not in npz.files:
        return None
    sv = np.asarray(npz[key])
    data = prepare_train_test(pipeline=pipeline, track=track,
                              target=target, feature_set=fs)
    X_te = np.asarray(data['X_te'])
    feat_names = list(data['feat_names'])
    if sv.shape != X_te.shape:
        raise RuntimeError(
            f'shape mismatch for {key}: shap {sv.shape} vs X_te {X_te.shape}')

    mean_abs = np.mean(np.abs(sv), axis=0)
    top_idx = np.argsort(-mean_abs)[:TOP_N]
    top_idx = top_idx[np.argsort(-mean_abs[top_idx])]

    rows = []
    for j in top_idx:
        feat = feat_names[j]
        col_vals = X_te[:, j]
        v_lo, v_hi = np.nanpercentile(col_vals, [5, 95])
        denom = max(v_hi - v_lo, 1e-12)
        norm = np.clip((col_vals - v_lo) / denom, 0.0, 1.0)
        # Detect MLP broadcast case: identical SHAP value across all rows.
        is_constant = float(np.std(sv[:, j])) < 1e-12
        for i in range(sv.shape[0]):
            rows.append({
                'feature': feat,
                'shap': float(sv[i, j]),
                'feat_value_norm': float(norm[i]),
                'mean_abs': float(mean_abs[j]),
                'is_constant': is_constant,
            })
    long = pd.DataFrame(rows)
    feat_order = [feat_names[j] for j in top_idx]
    long['feature'] = pd.Categorical(long['feature'],
                                     categories=feat_order, ordered=True)
    return {
        'long': long,
        'model': model,
        'fs': fs,
        'all_constant': bool(np.all([
            np.std(sv[:, j]) < 1e-12 for j in top_idx])),
        'feat_order': feat_order,
    }


def _draw_panel(ax, panel, track, target, idx, *, tabpfn_winner):
    long = panel['long']
    model = panel['model']
    fs = panel['fs']

    sns.stripplot(
        data=long, x='shap', y='feature',
        hue='feat_value_norm', palette=PALETTE,
        hue_norm=(0.0, 1.0),
        size=2.2, jitter=0.30, alpha=0.75,
        edgecolor='none', linewidth=0,
        ax=ax, legend=False,
    )

    ax.axvline(0.0, color='#666666', lw=0.6, ls='--', zorder=1)
    ax.set_xlabel(f'SHAP value  ({TARGET_UNIT[target]})')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, axis='x')

    note = EXPLAINER_NOTE.get(model, '')
    title = panel_header(track, target, idx)
    if tabpfn_winner:
        subtitle = f'shown: {model}/{fs} (TabPFN winner has no SHAP)'
    else:
        subtitle = f'{model}/{fs} ({note})'
    ax.set_title(f'{title}\n{subtitle}', fontsize=10)

    if panel['all_constant']:
        ax.text(
            0.98, 0.04,
            'permutation surrogate:\nno per-sample variation',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, color='#444444',
            bbox=dict(facecolor='white', edgecolor='#bbbbbb',
                      alpha=0.9, pad=2),
            zorder=5,
        )


def main():
    npz = np.load('results/shap_values_winners.npz')
    importance_df = pd.read_csv('results/shap_importance_winners.csv')

    fig, axes = make_fig('two_col', nrows=2, ncols=2)
    plt.subplots_adjust(wspace=0.55, right=0.90)

    for ax, (pipe, track, target, idx) in zip(axes.ravel(), PANELS):
        panel = _load_panel(npz, importance_df, pipe, track, target)
        if panel is None:
            ax.text(0.5, 0.5, 'no SHAP data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(panel_header(track, target, idx))
            continue
        is_tab_cell = (pipe, track, target) in TABPFN_SCORECARD_WINS
        _draw_panel(ax, panel, track, target, idx, tabpfn_winner=is_tab_cell)

    cax = fig.add_axes([0.925, 0.20, 0.012, 0.55])
    cbar = mpl.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(PALETTE),
        norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
        orientation='vertical',
    )
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(['low', 'high'])
    cbar.set_label('feature value\n(per-feature 5–95% scale)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        'SHAP per-sample beeswarm — per-cell winning families\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190), seed 42'
    )

    stem = OUT_DIR / 'main_fig_8'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Figure 8. Per-sample SHAP beeswarm for the best-explainable tuned '
        'family in each of the four opx (track, target) cells, rendered via '
        'seaborn.stripplot. Each panel shows the top 10 features ranked by '
        'mean |SHAP| on the held-out test partition at canonical seed 42. '
        'Each dot is one held-out experiment; the x-axis is the SHAP value '
        'in the target native unit (°C for T, kbar for P). Dot color is the '
        'underlying feature value rescaled per feature to its 5th–95th '
        'percentile (blue = low, red = high). A dashed vertical line marks '
        'SHAP = 0. Tree families use TreeExplainer (exact); ElasticNet uses '
        'LinearExplainer (exact, scaler-aware); MLP uses '
        'sklearn.inspection.permutation_importance (20 repeats, seed 42) as '
        'a SHAP surrogate, which produces identical per-feature values '
        'across samples and is annotated in-panel. TabPFN exposes no '
        'SHAP-compatible pathway; for opx-only T and opx-only P, where '
        'TabPFN wins the scorecard, the tuned-family runner-up is shown for '
        'explainability (LightGBM/alr and RF/pwlr respectively). Absolute '
        '|SHAP| values are not directly comparable across cells with '
        'different feature_sets. Source: results/shap_values_winners.npz, '
        'results/shap_importance_winners.csv.'
    )
    (OUT_DIR / 'main_fig_8.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
