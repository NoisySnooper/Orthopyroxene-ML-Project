#!/usr/bin/env python3
"""Core_12: SHAP feature importance for the 8 shipped-winner models.

One panel per (pipeline, track, target) cell, showing top-10 mean |SHAP|
features for the winning ML family. Tree models use TreeExplainer, linear
models use LinearExplainer, MLP uses permutation-importance as a fast
SHAP surrogate (documented in the caption). Per user direction 2026-04-20,
SHAP is restricted to winners-only to keep runtime within the Phase 6
one-shot budget.

Source: results/shap_importance_winners.csv
        (produced by scripts/shap/run_shap_winners.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._model_palette import OKABE_ITO  # noqa: E402
from scripts.figures._style import apply_pub_style  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [
    ('opx', 'opx_liq',  'T_C',    '(a) Opx + Liquid  T (C)'),
    ('opx', 'opx_liq',  'P_kbar', '(b) Opx + Liquid  P (kbar)'),
    ('opx', 'opx_only', 'T_C',    '(c) Opx only  T (C)'),
    ('opx', 'opx_only', 'P_kbar', '(d) Opx only  P (kbar)'),
    ('cpx', 'cpx_liq',  'T_C',    '(e) Cpx + Liquid  T (C)'),
    ('cpx', 'cpx_liq',  'P_kbar', '(f) Cpx + Liquid  P (kbar)'),
    ('cpx', 'cpx_only', 'T_C',    '(g) Cpx only  T (C)'),
    ('cpx', 'cpx_only', 'P_kbar', '(h) Cpx only  P (kbar)'),
]

# Cells where the 5-way scorecard (results/preregistered_scorecard_
# postcorrection.csv, ALL regime) promotes TabPFN over the tuned-family
# winner. TabPFN is an in-context foundation model that exposes no SHAP
# pathway, so these panels show the tuned-family runner-up's SHAP for
# explainability. Flagged in the subtitle and caption so the reader can
# decode the mismatch between the SHAP source and the shipped winner.
TABPFN_SCORECARD_WINS = {('opx', 'opx_only', 'T_C'),
                         ('opx', 'opx_only', 'P_kbar')}

TOP_N = 10
MODEL_COLORS = {
    'ElasticNet': OKABE_ITO['yellow'],
    'MLP':        OKABE_ITO['pink'],
    'RF':         OKABE_ITO['orange'],
    'ERT':        OKABE_ITO['green'],
    'XGB':        OKABE_ITO['sky_blue'],
    'LightGBM':   OKABE_ITO['blue'],
    'CatBoost':   OKABE_ITO['vermillion'],
    'GB':         '#777777',
}

TARGET_UNIT = {'T_C': 'C', 'P_kbar': 'kbar'}


def draw_panel(ax, sub: pd.DataFrame, title: str, model: str,
               fs: str, target: str, tabpfn_winner: bool = False):
    top = sub.sort_values('mean_abs_shap', ascending=False).head(TOP_N)
    top = top.sort_values('mean_abs_shap', ascending=True)
    y = np.arange(len(top))
    color = MODEL_COLORS.get(model, '0.4')
    ax.barh(y, top['mean_abs_shap'], color=color, edgecolor='black',
            linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(top['feature'], fontsize=8)
    unit = TARGET_UNIT.get(target, '')
    explainer_note = {
        'ElasticNet': '(LinearExplainer, exact)',
        'MLP':        '(perm. importance, SHAP surrogate)',
        'RF':         '(TreeExplainer, exact)',
        'ERT':        '(TreeExplainer, exact)',
        'LightGBM':   '(TreeExplainer, exact)',
        'XGB':        '(TreeExplainer, exact)',
        'CatBoost':   '(TreeExplainer, exact)',
    }.get(model, '')
    ax.set_xlabel(f'mean |SHAP|  ({unit})', fontsize=9)
    if tabpfn_winner:
        subtitle = (f'scorecard winner: TabPFN post (no SHAP available)\n'
                    f'shown: tuned runner-up {model} + {fs}  {explainer_note}')
        title_fontsize = 9
    else:
        subtitle = f'{model} + {fs}  {explainer_note}'
        title_fontsize = 10
    ax.set_title(f'{title}\n{subtitle}', loc='left', pad=6,
                 fontsize=title_fontsize)
    ax.grid(True, axis='x')
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv('results/shap_importance_winners.csv')

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (pipe, track, target, title) in zip(axes.ravel(), PANELS):
        sub = df[(df.pipeline == pipe) & (df.track == track)
                 & (df.target == target)]
        if sub.empty:
            ax.text(0.5, 0.5, 'no SHAP data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title, loc='left', pad=6, fontsize=10)
            continue
        model = sub['model'].iloc[0]
        fs = sub['feature_set'].iloc[0]
        is_tab_cell = (pipe, track, target) in TABPFN_SCORECARD_WINS
        draw_panel(ax, sub, title, model, fs, target,
                   tabpfn_winner=is_tab_cell)

    fig.suptitle(
        'SHAP feature importance -- best-explainable tuned model per '
        'cell (top 10, pre-correction predictions)\n'
        'Panels (c) and (d): scorecard winner is TabPFN (no SHAP '
        'pathway); tuned-family runner-up shown for explainability. '
        'See caption for explainer + unit notes.',
        y=1.04, fontsize=11,
    )
    plt.tight_layout(rect=(0, 0.00, 1, 0.93))
    plt.subplots_adjust(hspace=0.65, wspace=0.55, top=0.88)

    stem = OUT_DIR / 'Core_12_fig_shap_winners'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 12. SHAP feature importance for the best-explainable '
        'tuned-family model in each of the eight pipeline x track x '
        'target cells. Each panel shows the top 10 features ranked by '
        'mean |SHAP| on the held-out test partition at canonical seed '
        '42. Bar length is mean |SHAP| in the target\'s native units (C '
        'for temperature, kbar for pressure). Tuned-family identities: '
        'opx_liq T = ElasticNet/raw, opx_liq P = MLP/raw, opx_only T = '
        'LightGBM/alr, opx_only P = RF/pwlr, cpx_liq T = ERT/pwlr, '
        'cpx_liq P = LightGBM/pwlr, cpx_only T = ERT/pwlr, cpx_only P = '
        'MLP/alr. IMPORTANT: for opx_only T (panel c) and opx_only P '
        '(panel d) the 5-way scorecard (results/preregistered_scorecard_'
        'postcorrection.csv, ALL regime) promotes TabPFN post-correction '
        'over the tuned-family winner (TabPFN post-RMSE 125.6 C vs '
        'LightGBM/alr 148.0 C, and 5.94 kbar vs RF/pwlr 6.03 kbar '
        'respectively). TabPFN is an in-context foundation model and '
        'exposes no SHAP pathway (no TreeExplainer, no LinearExplainer, '
        'no gradient surface compatible with KernelExplainer at our '
        'runtime budget), so those two panels show the tuned-family '
        'runner-up\'s SHAP for explainability. Tree-family (RF, ERT, '
        'LightGBM) uses shap.TreeExplainer (exact); ElasticNet uses '
        'shap.LinearExplainer (exact) on scaler-transformed features; '
        'MLP uses sklearn.inspection.permutation_importance (20 repeats, '
        'seed=42) as a fast SHAP surrogate. KernelExplainer for MLP '
        'would add ~30 min per cell and was deferred in this pass '
        '(Phase 6 one-shot scope, 2026-04-20). Absolute |SHAP| values '
        'are NOT comparable across cells with different feature_sets '
        '(raw, alr, pwlr) because those transforms rescale the input. '
        'Source: results/shap_importance_winners.csv; scorecard winners '
        'from results/preregistered_scorecard_postcorrection.csv.'
    )
    (OUT_DIR / 'Core_12_fig_shap_winners.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
