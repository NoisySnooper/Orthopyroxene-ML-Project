#!/usr/bin/env python3
"""Core_02: citation-grouped split schematic (non-data-driven).

Compares naive row-random split to citation-grouped split and shows why
the latter is needed to avoid leakage. Sized to show ~30 citations as
rows with per-experiment cells.

No external data consumed; pure schematic via matplotlib patches.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._model_palette import OKABE_ITO  # noqa: E402
from scripts.figures._style import apply_pub_style  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_C = OKABE_ITO['blue']
TEST_C = OKABE_ITO['vermillion']

N_CITATIONS = 30
N_EXPS_PER_CIT = 6

rng = np.random.default_rng(42)
EXPS_PER_CIT = rng.integers(2, N_EXPS_PER_CIT + 1, size=N_CITATIONS)


PANEL_BG = {'naive': '#FFF4E5', 'grouped': '#EAF3EC'}
# Use orange for naive (not vermillion) so the panel border doesn't
# visually merge with the vermillion "test" cells.
PANEL_BORDER = {'naive': OKABE_ITO['orange'], 'grouped': OKABE_ITO['green']}


def draw_panel(ax, label_letter, title, subtitle, assignment_matrix, kind):
    """assignment_matrix: (n_cit, n_exp) with 1=train 2=test 0=absent."""
    n_cit, n_exp = assignment_matrix.shape

    # Soft branded background panel
    bg = mpatches.FancyBboxPatch(
        (-0.4, -0.8), n_exp + 1.5, n_cit + 1.2,
        boxstyle='round,pad=0.08',
        facecolor=PANEL_BG[kind], edgecolor=PANEL_BORDER[kind],
        linewidth=1.6, zorder=0,
    )
    ax.add_patch(bg)

    for ci in range(n_cit):
        for ei in range(n_exp):
            val = assignment_matrix[ci, ei]
            if val == 0:
                continue
            color = TRAIN_C if val == 1 else TEST_C
            rect = mpatches.Rectangle(
                (ei + 0.5, n_cit - ci - 1), 0.9, 0.9,
                facecolor=color, edgecolor='white', linewidth=0.7,
                zorder=2,
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.5, n_exp + 1.2)
    ax.set_ylim(-1.0, n_cit + 0.5)
    # Do NOT set aspect='equal' - it created dead space around skinny
    # 30x6 grids. Let the panel fill its subplot; cells become
    # rectangular, which is fine for a schematic.

    ax.set_xticks([1, n_exp])
    ax.set_xticklabels(['first\nexperiment', 'last\nexperiment'],
                       fontsize=9.5)
    ax.set_yticks([0, n_cit - 1])
    ax.set_yticklabels([f'citation {n_cit}', 'citation 1'], fontsize=9.5)
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ('top', 'right', 'left', 'bottom'):
        ax.spines[spine].set_visible(False)

    ax.set_xlabel('Experiment within a publication\n(one cell = one experiment)',
                  fontsize=10.5, labelpad=6)
    ax.set_ylabel('Publications (one row = one citation)', fontsize=10.5)

    # Title + subtitle only (no (a)/(b) letter badge)
    ax.text(0.0, 1.09, title, transform=ax.transAxes,
            va='bottom', ha='left', fontsize=13, fontweight='bold',
            color=PANEL_BORDER[kind])
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes,
            va='bottom', ha='left', fontsize=9.5, style='italic',
            color='0.30')


def main():
    naive = np.zeros((N_CITATIONS, N_EXPS_PER_CIT), dtype=int)
    grouped = np.zeros_like(naive)
    for ci in range(N_CITATIONS):
        n = EXPS_PER_CIT[ci]
        for ei in range(n):
            naive[ci, ei] = 2 if rng.random() < 0.2 else 1
        if ci >= int(N_CITATIONS * 0.8):
            grouped[ci, :n] = 2
        else:
            grouped[ci, :n] = 1

    # Square-ish figure: both panels fill their cells, roughly
    # balanced aspect so the overall layout reads as square.
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(
        1, 2, wspace=0.28,
        top=0.82, bottom=0.24, left=0.08, right=0.96,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_panel(
        ax_a, '',
        'Naive row-random 80/20 split',
        'Rows are shuffled; split ignores citation.',
        naive, 'naive',
    )
    draw_panel(
        ax_b, '',
        'Citation-grouped 80/20 split',
        'Whole publications held on one side; used in this study.',
        grouped, 'grouped',
    )

    # Consequence note boxes anchored in figure coords below each panel
    fig.text(
        0.285, 0.145,
        'Leakage risk: experiments from the same publication\n'
        'end up on both sides of the split.',
        ha='center', va='top', fontsize=10.5,
        bbox=dict(facecolor=PANEL_BG['naive'],
                  edgecolor=PANEL_BORDER['naive'],
                  linewidth=1.1, alpha=0.95, pad=8,
                  boxstyle='round,pad=0.5'),
    )
    fig.text(
        0.755, 0.145,
        'No within-citation leakage: generalisation is measured\n'
        'against unseen literature, not unseen rows.',
        ha='center', va='top', fontsize=10.5,
        bbox=dict(facecolor=PANEL_BG['grouped'],
                  edgecolor=PANEL_BORDER['grouped'],
                  linewidth=1.1, alpha=0.95, pad=8,
                  boxstyle='round,pad=0.5'),
    )

    handles = [
        mpatches.Patch(facecolor=TRAIN_C, edgecolor='white',
                       label='Train (80%)'),
        mpatches.Patch(facecolor=TEST_C, edgecolor='white',
                       label='Test (20%)'),
    ]
    fig.legend(
        handles=handles, loc='lower center',
        bbox_to_anchor=(0.5, 0.025), ncol=2,
        frameon=True, framealpha=0.95, edgecolor='0.6',
        fontsize=11, title='Fold assignment', title_fontsize=11,
    )

    fig.suptitle(
        'Citation-grouped cross-validation on the ExPetDB training corpus\n'
        '(StratifiedGroupKFold, 10 folds, min_train_fold=50; schematic '
        'uses synthetic publication-level counts)',
        fontsize=14, fontweight='bold', y=0.965,
    )

    out_stem = OUT_DIR / 'Core_02_fig_citation_split'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 2. Citation-grouped cross-validation schematic on the '
        'ExPetDB training corpus (opx n=1635, cpx n=5282 after equilibrium '
        'and cation filters). '
        '(a) A naive row-random 80/20 split lets experiments from the '
        'same publication appear on both sides of the split; correlated '
        'features (same laboratory, same protocol, same starting '
        'materials) leak from train to test and inflate generalisation '
        'estimates. (b) The citation-grouped split used throughout this '
        'study: every publication is held wholly on one side. We use '
        'scikit-learn StratifiedGroupKFold with 10 folds and '
        'min_train_fold=50 to stratify on pressure regime while '
        'respecting citation grouping. Each cell represents one '
        'experiment; each row is one publication (~93 publications in '
        'the opx corpus, ~260 in the cpx corpus).'
    )
    (OUT_DIR / 'Core_02_fig_citation_split.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
