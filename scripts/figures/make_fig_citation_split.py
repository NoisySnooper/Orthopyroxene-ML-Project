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

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_C = OKABE_ITO['blue']
TEST_C = OKABE_ITO['vermillion']

N_CITATIONS = 30
N_EXPS_PER_CIT = 6

rng = np.random.default_rng(42)
EXPS_PER_CIT = rng.integers(2, N_EXPS_PER_CIT + 1, size=N_CITATIONS)


def draw_panel(ax, title, assignment_matrix, annotate):
    """assignment_matrix: (n_cit, n_exp) with 1=train 2=test 0=absent."""
    n_cit, n_exp = assignment_matrix.shape
    for ci in range(n_cit):
        for ei in range(n_exp):
            val = assignment_matrix[ci, ei]
            if val == 0:
                continue
            color = TRAIN_C if val == 1 else TEST_C
            rect = mpatches.Rectangle(
                (ei, n_cit - ci - 1), 0.9, 0.9,
                facecolor=color, edgecolor='white', linewidth=0.6)
            ax.add_patch(rect)
    ax.set_xlim(-0.5, n_exp + 0.5)
    ax.set_ylim(-0.5, n_cit + 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel('experiments within a publication ->', fontsize=9)
    ax.set_ylabel('publications (citations)', fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)
    ax.text(0.01, -0.08, annotate, transform=ax.transAxes,
            va='top', ha='left', fontsize=9, wrap=True)


def main():
    naive = np.zeros((N_CITATIONS, N_EXPS_PER_CIT), dtype=int)
    grouped = np.zeros_like(naive)
    for ci in range(N_CITATIONS):
        n = EXPS_PER_CIT[ci]
        for ei in range(n):
            naive[ci, ei] = 2 if rng.random() < 0.2 else 1  # random 20% test
        if ci >= int(N_CITATIONS * 0.8):
            grouped[ci, :n] = 2
        else:
            grouped[ci, :n] = 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 7),
                              gridspec_kw=dict(wspace=0.25))
    draw_panel(
        axes[0],
        '(a) Naive row-random split (80/20)',
        naive,
        ('Experiments from a single publication land on BOTH sides\n'
         'of the split. Correlated features (same lab, same protocol,\n'
         'same starting material) leak from train into test.'),
    )
    draw_panel(
        axes[1],
        '(b) Citation-grouped split (80/20)',
        grouped,
        ('Each publication is wholly in train OR wholly in test.\n'
         'No within-citation leakage; generalisation error is\n'
         'measured against unseen literature, not unseen rows.'),
    )

    handles = [
        mpatches.Patch(color=TRAIN_C, label='train'),
        mpatches.Patch(color=TEST_C,  label='test'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=10)

    fig.suptitle(
        'Citation-grouped split used throughout this study '
        '(StratifiedGroupKFold, 10 folds, min_train_fold=50)',
        fontsize=12, y=1.02,
    )

    out_stem = OUT_DIR / 'Core_02_fig_citation_split'
    fig.savefig(f'{out_stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out_stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 2. Citation-grouped cross-validation schematic. '
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
