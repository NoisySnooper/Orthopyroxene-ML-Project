#!/usr/bin/env python3
"""Core_03: methods flowchart.

Three phase panels (DATA -> TRAINING -> EVALUATION) rendered as stacked
rounded panels. Phase I and Phase III flow left-to-right; Phase II flows
right-to-left so the inter-phase connectors enter/exit on opposite sides
(box 3 -> box 4 on the right, box 7 -> box 8 on the left). The two
between-phase arrows are drawn box-to-box straight through the phase-
panel boundaries. No crossing arrows.
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
from scripts.figures._style import apply_pub_style  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE_BG = {
    'data':  '#E8F1F8',
    'train': '#EAF3EC',
    'eval':  '#FCEBE3',
}
PHASE_BORDER = {
    'data':  OKABE_ITO['sky_blue'],
    'train': OKABE_ITO['green'],
    'eval':  OKABE_ITO['vermillion'],
}

STAGE_COLOR = {
    'data':  OKABE_ITO['sky_blue'],
    'train': OKABE_ITO['green'],
    'eval':  OKABE_ITO['vermillion'],
}


def stage_box(ax, x, y, w, h, num, title, body, fill, fontsize=11):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02',
        linewidth=1.1, facecolor=fill, edgecolor='black', alpha=0.92)
    ax.add_patch(rect)
    ax.text(x + 0.22, y + h - 0.08, f'{num}',
            fontsize=12, fontweight='bold', color='white',
            va='top', ha='center',
            bbox=dict(facecolor='black', edgecolor='none',
                      boxstyle='circle,pad=0.25'))
    # Title sits clear of the number badge; body follows immediately.
    title_y = y + h - 0.38
    ax.text(x + w / 2, title_y, title,
            fontsize=fontsize + 3.0, fontweight='bold',
            ha='center', va='top', color='white')
    ax.text(x + w / 2, title_y - 0.45, body,
            fontsize=fontsize, ha='center', va='top',
            color='white')


def phase_panel(ax, x, y, w, h, name, subtitle):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.03',
        linewidth=1.8, facecolor=PHASE_BG[name],
        edgecolor=PHASE_BORDER[name], alpha=1.0)
    ax.add_patch(rect)
    # Title centered at the top of the phase panel so it does not
    # collide with side-entry / side-exit inter-phase connectors.
    ax.text(x + w / 2, y + h - 0.12,
            f'{subtitle}',
            fontsize=14, fontweight='bold', color=PHASE_BORDER[name],
            va='top', ha='center')


def arrow(ax, x0, y0, x1, y1, color='0.2', lw=1.8,
          connectionstyle='arc3,rad=0'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color,
                                lw=lw, mutation_scale=16,
                                connectionstyle=connectionstyle))


def main():
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # ---- Phase I: DATA (top row) -------------------------------------
    phase_panel(ax, 0.3, 10.7, 13.4, 3.1, 'data',
                'Phase I. Data preparation')
    y_row = 10.95
    h_row = 2.1
    w_box = 3.9
    stage_box(ax, 0.8, y_row, w_box, h_row, 1,
              'ExPetDB raw',
              'opx n=1635\n'
              'cpx n=5282\n'
              'pyroxene core\ncompositions',
              STAGE_COLOR['data'], fontsize=11)
    stage_box(ax, 5.05, y_row, w_box, h_row, 2,
              'Equilibrium + cation filters',
              'KD(Fe-Mg) 0.23-0.35\n'
              'Wo \u2264 5 mol% (pigeonite)\n'
              'P \u2264 100 kbar\n'
              'cation sum 3.95-4.05',
              STAGE_COLOR['data'], fontsize=10)
    stage_box(ax, 9.3, y_row, w_box, h_row, 3,
              '80/20 citation-grouped split',
              'held-out test:\n'
              'opx_liq n=174\n'
              'opx_only n=190\n'
              'groups=Citation',
              STAGE_COLOR['data'], fontsize=10)
    arrow(ax, 4.7, y_row + h_row / 2, 5.05, y_row + h_row / 2)
    arrow(ax, 8.95, y_row + h_row / 2, 9.3, y_row + h_row / 2)

    # Inter-phase connector: box 3 bottom -> box 4 top, straight vertical
    # through the phase-panel boundary.
    arrow(ax, 12.175, 10.95, 12.175, 8.4, color='0.35', lw=2.2)

    # ---- Phase II: TRAINING (middle, right-to-left flow) -------------
    phase_panel(ax, 0.3, 4.9, 13.4, 5.2, 'train',
                'Phase II. Model training')
    y_top = 7.6
    y_bot = 5.4
    h_box = 1.9
    w_box = 2.55
    y_mid = 6.5
    w_box6 = 2.95

    # Box 4 (entry) -- right edge of Phase II
    stage_box(ax, 10.9, y_mid, w_box, h_box, 4,
              '9 models',
              'ElasticNet, RF, ERT,\n'
              'GB, XGB, LightGBM,\n'
              'CatBoost, MLP,\n'
              'TabPFN v2',
              STAGE_COLOR['train'], fontsize=10)
    # Box 5 (two branches) -- middle-right
    stage_box(ax, 7.7, y_top, w_box, h_box, 5,
              'Optuna TPE tuning',
              '200 trials, seed 42\n'
              'objective: 3-fold\n'
              'GroupKFold CV RMSE\n(8 tuned families)',
              STAGE_COLOR['train'], fontsize=10)
    stage_box(ax, 7.7, y_bot, w_box, h_box, 5,
              'TabPFN default',
              'n_estimators=8\n'
              'no tuning\n(foundation model,\nfrozen weights)',
              STAGE_COLOR['train'], fontsize=10)
    # Box 6 (freeze) -- middle-left
    stage_box(ax, 4.2, y_mid, w_box6, h_box, 6,
              'Freeze model configs',
              'tuned: best trial\npersisted to JSON\n'
              'TabPFN: default\nconfig (no tuning)',
              STAGE_COLOR['train'], fontsize=10)
    # Box 7 (exit) -- left edge of Phase II
    stage_box(ax, 0.55, y_mid, w_box6, h_box, 7,
              'Multi-seed refit',
              'tuned: seeds 42-61\n'
              'TabPFN: 20 seeds\n'
              'yields 95% CI\n'
              'whiskers (bootstrap)',
              STAGE_COLOR['train'], fontsize=10)

    y_mid_c = y_mid + h_box / 2
    # Box 4 -> 5 Optuna (up-left) and 4 -> 5 TabPFN (down-left)
    arrow(ax, 10.9, y_mid_c, 10.25, y_top + h_box / 2)
    arrow(ax, 10.9, y_mid_c, 10.25, y_bot + h_box / 2)
    # Box 5 Optuna -> 6 (down-left) and 5 TabPFN -> 6 (up-left): V fan-in
    # mirroring the V fan-out from box 4 to the two box 5 branches.
    arrow(ax, 7.7, y_top + h_box / 2, 7.15, y_mid_c)
    arrow(ax, 7.7, y_bot + h_box / 2, 7.15, y_mid_c)
    # Box 6 -> 7 (horizontal left)
    arrow(ax, 4.2, y_mid_c, 3.5, y_mid_c)

    # Inter-phase connector: box 7 bottom -> box 8 top, straight vertical.
    arrow(ax, 2.025, y_mid, 2.025, 3.55, color='0.35', lw=2.2)

    # ---- Phase III: EVALUATION (bottom row) --------------------------
    phase_panel(ax, 0.3, 0.3, 13.4, 4.0, 'eval',
                'Phase III. Bias correction and evaluation')
    # y_row lifted + h_row increased so the top gap between the
    # phase-title bar and the box tops matches Phase I/II (~0.75 u).
    y_row = 0.55
    h_row = 3.0
    w_box = 3.9
    stage_box(ax, 0.8, y_row, w_box, h_row, 8,
              'OOF bias fit (per regime)',
              '10-fold StratifiedGroup-\n'
              'KFold OOF on training\n'
              'partition (group=Citation);\n'
              'tuned: 20 seeds, TabPFN:\n'
              '5 seeds; test set untouched.\n'
              'Form A: y_corr = a\u00b7y_pred\n'
              '+ b per regime (OLS).\n'
              'Form B: piecewise\n'
              'Agreda-Lopez sigmoid,\n'
              '4 breakpoints.',
              STAGE_COLOR['eval'], fontsize=10)
    stage_box(ax, 5.05, y_row, w_box, h_row, 9,
              'Ship-if-better rule',
              'accept correction only if\n'
              'overall RMSE drops AND\n'
              'no regime degrades beyond\n'
              'tolerance envelope\n'
              'max(T_ABS, T_REL \u00b7 pre_rmse)\n'
              'T_ABS = 10 \u00b0C / 1 kbar\n'
              'T_REL = 0.10',
              STAGE_COLOR['eval'], fontsize=10)
    stage_box(ax, 9.3, y_row, w_box, h_row, 10,
              'Regime test + ArcPL eval',
              'held-out test:\n'
              'opx_liq n=174, opx_only n=190\n'
              'shallow < 5 kbar\n'
              'MASH 5-15 kbar\n'
              'litho 15-30 kbar\n'
              'deep \u2265 30 kbar\n'
              '+ ArcPL n=197 (OOD)',
              STAGE_COLOR['eval'], fontsize=10)
    arrow(ax, 4.7, y_row + h_row / 2, 5.05, y_row + h_row / 2)
    arrow(ax, 8.95, y_row + h_row / 2, 9.3, y_row + h_row / 2)

    ax.set_title(
        'Methods pipeline: ExPetDB raw \u2192 train 9 model families '
        '\u2192 bias-correct and evaluate per regime',
        fontsize=13, fontweight='bold', pad=14,
    )

    out_stem = OUT_DIR / 'Core_03_fig_methods_flowchart'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 3. Methods flowchart. Three phase panels, each with '
        'numbered stages. Phase I (Data) flows left-to-right: raw ExPetDB '
        '(stage 1), equilibrium + cation filters (stage 2), and the 80/20 '
        'citation-grouped train/test split that produces the held-out test '
        'partitions opx_liq n=174 and opx_only n=190 (stage 3). Phase II '
        '(Model training) flows right-to-left so the connector from Phase '
        'I enters at box 4 on the right and the connector to Phase III '
        'exits from box 7 on the left. The 9 models (stage 4) fan out into '
        'two tuning routes at stage 5 -- Optuna TPE tuning over 200 trials '
        'at seed 42 with a 3-fold GroupKFold inner-CV objective for the '
        'eight tuned families, vs. TabPFN default at n_estimators=8 with '
        'no tuning -- and both branches fan back in at stage 6, where the '
        'final model configuration is frozen (Optuna winning trial '
        'persisted to JSON for the tuned families; TabPFN default config '
        'carried through unchanged). Stage 7 then runs the multi-seed '
        'refit: seeds 42-61 for tuned families and 20 seeds for TabPFN, '
        'producing the 95% CI whiskers (bootstrap) used throughout the '
        'figure set. Phase III (Evaluation) flows left-to-right: stage 8 '
        'generates 10-fold StratifiedGroupKFold OOF predictions inside the '
        'training partition (group=Citation; 20 seeds for tuned families, '
        '5 seeds for TabPFN) -- the test set remains untouched -- and '
        'fits Form A (per-regime OLS, y_corr = a.y_pred + b) and Form B '
        '(piecewise Agreda-Lopez sigmoid, 4 breakpoints) on those OOF '
        'predictions; stage 9 applies the pre-registered tolerance '
        'envelope max(T_ABS, T_REL x pre_rmse) with T_ABS = 10 degC for T, '
        '1 kbar for P, and T_REL = 0.10 (Amendment 2); stage 10 evaluates '
        'on the held-out ExPetDB test partitions split into four pre-'
        'registered pressure regimes plus the ArcPL n=197 external-'
        'validation dataset as an out-of-distribution check.'
    )
    (OUT_DIR / 'Core_03_fig_methods_flowchart.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
