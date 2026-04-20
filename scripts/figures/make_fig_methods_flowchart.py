#!/usr/bin/env python3
"""Core_03: methods flowchart. Pure matplotlib, no external deps.

Left-to-right pipeline showing: ExPetDB raw -> filters ->
citation-grouped split -> 9 model families -> Optuna/TabPFN ->
freeze params -> 20/5-seed refit -> OOF bias fit -> conservative
acceptance rule -> test eval per regime.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._model_palette import OKABE_ITO  # noqa: E402

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_C  = OKABE_ITO['sky_blue']
PROC_C  = OKABE_ITO['orange']
MODEL_C = OKABE_ITO['green']
EVAL_C  = OKABE_ITO['vermillion']


def box(ax, x, y, w, h, text, color, num=None, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02',
        linewidth=1.2, facecolor=color, edgecolor='black', alpha=0.85)
    ax.add_patch(rect)
    if num is not None:
        ax.text(x + 0.04, y + h - 0.05, f'{num}',
                fontsize=11, fontweight='bold', color='white',
                va='top', ha='left',
                bbox=dict(facecolor='black', edgecolor='none',
                          boxstyle='round,pad=0.15'))
    ax.text(x + w / 2, y + h / 2, text,
            fontsize=fontsize, ha='center', va='center',
            color='white', fontweight='bold', wrap=True)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color='0.15',
                                lw=1.8, mutation_scale=14))


def main():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Stage 1: Data
    _, _ = box(ax, 0.2, 7.0, 2.2, 1.2,
               'ExPetDB raw\n(n=5917 pyroxenes)',
               DATA_C, num=1)
    _, _ = box(ax, 0.2, 5.0, 2.2, 1.2,
               'Filters\nKD window,\nP ceiling, Wo frac',
               DATA_C, num=2)
    _, _ = box(ax, 0.2, 3.0, 2.2, 1.2,
               'Citation-grouped\n10-fold CV\n(StratifiedGroupKFold)',
               DATA_C, num=3)

    # Stage 2: Training universe
    box(ax, 3.2, 4.5, 2.6, 3.5,
        '9 model families:\n\nElasticNet, RF, ERT,\n'
        'GB, XGB, LightGBM,\nCatBoost, MLP,\nTabPFN',
        MODEL_C, num=4, fontsize=8.5)

    # Stage 3: Tuning split
    box(ax, 6.6, 6.5, 2.6, 1.5,
        'Optuna tuning\n200 trials, seed 42\n(8 tuned families)',
        PROC_C, num=5, fontsize=8.5)
    box(ax, 6.6, 4.5, 2.6, 1.5,
        'TabPFN default\nn_estimators=8\n(no tuning)',
        PROC_C, num=5, fontsize=8.5)

    # Stage 4: Freeze
    box(ax, 9.8, 5.5, 2.3, 1.5,
        'Freeze params',
        PROC_C, num=6, fontsize=9)

    # Stage 5: Refit
    box(ax, 9.8, 3.0, 2.3, 1.8,
        '20-seed refit\n(5-seed for TabPFN)\nOOF predictions',
        PROC_C, num=7, fontsize=8.5)

    # Stage 6: Bias fit
    box(ax, 12.7, 5.0, 2.8, 2.2,
        'OOF bias fit:\n\nForm A (per-regime OLS)\nForm B (piecewise\nAgreda sigmoid)',
        MODEL_C, num=8, fontsize=8.5)

    # Stage 7: Ship decision
    box(ax, 12.7, 2.5, 2.8, 1.8,
        'Conservative\nacceptance rule\n(ship-if-better,\nno-regime-worsens)',
        EVAL_C, num=9, fontsize=8.5)

    # Stage 8: Evaluation
    box(ax, 6.6, 1.0, 5.5, 1.5,
        'Test evaluation per pre-registered P regime\n'
        '(shallow <5, MASH 5-15, litho 15-30, deep >=30 kbar)',
        EVAL_C, num=10, fontsize=9)

    # Arrows
    arrow(ax, 1.3, 7.0, 1.3, 6.2)   # 1->2
    arrow(ax, 1.3, 5.0, 1.3, 4.2)   # 2->3
    arrow(ax, 2.4, 3.6, 3.2, 5.5)   # 3->4 (into models)
    arrow(ax, 5.8, 7.2, 6.6, 7.2)   # 4->5a (top path, Optuna)
    arrow(ax, 5.8, 5.3, 6.6, 5.3)   # 4->5b (bottom path, TabPFN)
    arrow(ax, 9.2, 7.2, 9.8, 6.5)   # 5a->6
    arrow(ax, 9.2, 5.3, 9.8, 6.2)   # 5b->6
    arrow(ax, 10.95, 5.5, 10.95, 4.8)  # 6->7
    arrow(ax, 12.1, 3.9, 12.7, 5.5)   # 7->8
    arrow(ax, 14.1, 5.0, 14.1, 4.3)  # 8->9
    arrow(ax, 12.7, 3.4, 12.1, 2.0)  # 9->10
    arrow(ax, 9.2, 3.9, 9.2, 2.5)    # 7-> eval (direct)

    # Legend
    handles = [
        mpatches.Patch(color=DATA_C,  label='Data stage'),
        mpatches.Patch(color=MODEL_C, label='Model stage'),
        mpatches.Patch(color=PROC_C,  label='Training procedure'),
        mpatches.Patch(color=EVAL_C,  label='Evaluation / decision'),
    ]
    ax.legend(handles=handles, loc='lower right', ncol=4, frameon=False,
              bbox_to_anchor=(1.0, -0.02), fontsize=9)

    ax.set_title(
        'Methods pipeline: raw data to per-regime evaluation '
        '(9 model families x 4 tracks x 2 targets)',
        fontsize=12, pad=14,
    )

    out_stem = OUT_DIR / 'Core_03_fig_methods_flowchart'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 3. Methods flowchart. The opx ML thermobarometer pipeline, '
        'stage-by-stage. Raw ExPetDB experiments (stage 1) are filtered '
        '(KD equilibrium window, P ceiling, Wo fraction; stage 2) and '
        'citation-grouped into a 10-fold StratifiedGroupKFold split '
        '(stage 3). Nine model families (stage 4) comprise the baseline '
        'linear model (ElasticNet), six tree/boosted families (RF, ERT, '
        'GB, XGB, LightGBM, CatBoost), an MLP, and the TabPFN foundation '
        'baseline. Eight families are tuned with Optuna (200 trials, seed '
        '42; stage 5); TabPFN runs at default settings without tuning. '
        'Best parameters are frozen (stage 6) and refit with 20 seeds '
        '(5 seeds for TabPFN to bound CPU cost; stage 7) producing '
        'out-of-fold predictions. Form A (per-regime OLS) and Form B '
        '(piecewise Agreda-Lopez sigmoid) are fit on the OOF residuals '
        '(stage 8) and evaluated against the conservative acceptance rule '
        '(overall_delta > 1e-6 AND max_regime_degradation <= 1e-6; stage 9). '
        'Final test-set evaluation is reported per pre-registered pressure '
        'regime (stage 10).'
    )
    (OUT_DIR / 'Core_03_fig_methods_flowchart.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
