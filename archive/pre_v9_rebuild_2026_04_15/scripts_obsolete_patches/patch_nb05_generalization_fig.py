"""Add a generalization figure cell to nb05 between the LOSO run (cell 5)
and the Phase 5.2 verification cell (cell 6).

Produces figures/fig_nb05_generalization.{png,pdf}: a 2x3 grid of grouped
bar charts showing T/P RMSE across LOSO / Cluster-KFold / TargetBinKFold
with one bar per model (RF, ERT, XGB, GB), family-colored, error bars
from per-fold std, and family reference lines drawn from the random-split
test RMSE in nb03_per_family_winners.json.

Idempotent: if a cell whose first line starts with
`# Phase 5.3: generalization figure` already exists, no change is made.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb05_loso_validation.ipynb"

NEW_CELL_SRC = '''# Phase 5.3: generalization figure (T and P RMSE across three CV strategies)
import matplotlib.pyplot as plt
import json
from config import FAMILY_COLORS
from src.plot_style import apply_style

apply_style()

pooled_df = pd.read_csv(RESULTS / 'nb05_loso_pooled.csv')
per_fold_df = pd.read_csv(RESULTS / 'nb05_per_fold_rmse.csv')

with open(RESULTS / 'nb03_per_family_winners.json') as _f:
    _winners = json.load(_f)
REF_RMSE = {
    ('T_C',    'forest'):  _winners['forest_family']['opx_liq_T_C']['rmse_test_mean'],
    ('T_C',    'boosted'): _winners['boosted_family']['opx_liq_T_C']['rmse_test_mean'],
    ('P_kbar', 'forest'):  _winners['forest_family']['opx_liq_P_kbar']['rmse_test_mean'],
    ('P_kbar', 'boosted'): _winners['boosted_family']['opx_liq_P_kbar']['rmse_test_mean'],
}

STRATS = ['LOSO', 'Cluster-KFold', 'TargetBinKFold']
MODELS = ['RF', 'ERT', 'XGB', 'GB']
FAMILY = {'RF': 'forest', 'ERT': 'forest', 'XGB': 'boosted', 'GB': 'boosted'}
TARGETS = [
    ('T_C',    'T RMSE (\u00B0C)'),
    ('P_kbar', 'P RMSE (kbar)'),
]

fold_std = (per_fold_df.groupby(['strategy', 'model_name', 'target'])['rmse']
            .std(ddof=1).reset_index().rename(columns={'rmse': 'rmse_std'}))

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
x = np.arange(len(MODELS))
bar_w = 0.62

for row_idx, (target, ylabel) in enumerate(TARGETS):
    for col_idx, strat in enumerate(STRATS):
        ax = axes[row_idx, col_idx]
        sub = pooled_df[(pooled_df['strategy'] == strat) &
                        (pooled_df['target'] == target)].set_index('model_name')
        std_sub = fold_std[(fold_std['strategy'] == strat) &
                           (fold_std['target'] == target)].set_index('model_name')
        heights = np.array([sub.loc[m, 'rmse'] if m in sub.index else np.nan
                            for m in MODELS], dtype=float)
        errs    = np.array([std_sub.loc[m, 'rmse_std'] if m in std_sub.index else np.nan
                            for m in MODELS], dtype=float)
        colors  = [FAMILY_COLORS[FAMILY[m]] for m in MODELS]

        bars = ax.bar(x, heights, bar_w, yerr=errs, color=colors,
                      edgecolor='black', linewidth=0.6,
                      error_kw={'elinewidth': 1.1, 'capsize': 3, 'ecolor': '#333'})
        for xi, h in zip(x, heights):
            if np.isfinite(h):
                ax.text(xi, h + (0.02 * (np.nanmax(heights) or 1)),
                        f'{h:.1f}' if target == 'T_C' else f'{h:.2f}',
                        ha='center', va='bottom', fontsize=8)

        ax.axhline(REF_RMSE[(target, 'forest')], color=FAMILY_COLORS['forest'],
                   linestyle='--', lw=1.2, alpha=0.85,
                   label=f"NB03 forest test ({REF_RMSE[(target, 'forest')]:.1f})"
                   if target == 'T_C'
                   else f"NB03 forest test ({REF_RMSE[(target, 'forest')]:.2f})")
        ax.axhline(REF_RMSE[(target, 'boosted')], color=FAMILY_COLORS['boosted'],
                   linestyle=':', lw=1.2, alpha=0.85,
                   label=f"NB03 boosted test ({REF_RMSE[(target, 'boosted')]:.1f})"
                   if target == 'T_C'
                   else f"NB03 boosted test ({REF_RMSE[(target, 'boosted')]:.2f})")

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=9)
        ax.set_ylabel(ylabel if col_idx == 0 else '')
        if row_idx == 0:
            ax.set_title(strat, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=7, frameon=True,
                  handlelength=1.6, borderpad=0.4)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, fc=FAMILY_COLORS['forest'], ec='black',
                  label='Forest (RF, ERT)'),
    plt.Rectangle((0, 0), 1, 1, fc=FAMILY_COLORS['boosted'], ec='black',
                  label='Boosted (XGB, GB)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.01), fontsize=10, frameon=False)

fig.suptitle('Generalization under three out-of-distribution CV strategies '
             '(error bars = per-fold std; dashed lines = NB03 random-split '
             'test RMSE)', fontsize=11, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

for _ext in ('png', 'pdf'):
    fig.savefig(FIGURES / f'fig_nb05_generalization.{_ext}',
                dpi=300 if _ext == 'png' else None, bbox_inches='tight')
plt.show()
plt.close(fig)

print('Saved figures/fig_nb05_generalization.{png,pdf} '
      f'({len(STRATS)} strategies x {len(MODELS)} models x {len(TARGETS)} targets).')
'''


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)

    # Idempotency: if an existing figure cell is already present anywhere,
    # update it in place rather than inserting a duplicate.
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and cell.source.lstrip().startswith('# Phase 5.3: generalization figure'):
            if cell.source == NEW_CELL_SRC:
                print(f'nb05 cell {i}: already on target source.')
                return 0
            cell.source = NEW_CELL_SRC
            cell.outputs = []
            cell.execution_count = None
            nbformat.validate(nb)
            nbformat.write(nb, str(NB))
            print(f'nb05 cell {i}: refreshed generalization figure cell.')
            return 0

    # Insert before the Phase 5.2 verification cell (currently cell 6).
    verification_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and '# Phase 5.2: verification' in cell.source:
            verification_idx = i
            break
    if verification_idx is None:
        print('nb05: Phase 5.2 verification cell not found.')
        return 1

    new_cell = nbformat.v4.new_code_cell(source=NEW_CELL_SRC)
    new_cell.pop('id', None)
    nb.cells.insert(verification_idx, new_cell)
    nbformat.validate(nb)
    nbformat.write(nb, str(NB))
    print(f'nb05: inserted generalization figure cell at index {verification_idx} '
          f'(pushed verification to {verification_idx + 1}).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
