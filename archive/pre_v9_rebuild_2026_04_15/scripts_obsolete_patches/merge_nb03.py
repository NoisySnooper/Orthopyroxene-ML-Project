"""Merge nb03b (n_aug sensitivity) into nb03c, rewrite call sites, rename to nb03.

Steps:
1. Load nb03c_baseline_models.ipynb.
2. In every cell, replace `get_features_for_method(` -> `build_feature_matrix(`.
3. Insert the n_aug sensitivity section from nb03b (intro markdown, sensitivity
   code, plot code, SET N_AUG markdown collapsed into an appendix) as an
   appendix AFTER the main benchmark (before the Phase 3R.7 verification).
   The sensitivity test now serves as the documented justification for
   dropping the `_aug` feature variants.
4. Update the cell-0 title markdown to `NB03` with a summary of what was
   merged.
5. Write to nb03_baseline_models.ipynb (overwrite the stub nb03).
6. Delete the stub nb03b_baseline_models.ipynb and nb03c_baseline_models.ipynb.

Safe to re-run as long as source nb03b/nb03c .ipynb files still exist.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / 'notebooks'

NB03 = NB / 'nb03_baseline_models.ipynb'
NB03B = NB / 'nb03b_baseline_models.ipynb'
NB03C = NB / 'nb03c_baseline_models.ipynb'

INTRO_MD = """# NB03: Multi-seed evaluation with dynamic feature-set winner selection

Tests three feature engineering methods (raw, alr, pwlr) across four tree
ensembles (RF, ERT, XGB, GB) for both `opx_liq` and `opx_only` tracks. Tuned
once on seed 42 with `HalvingRandomSearchCV` + `StratifiedGroupKFold`, then
frozen hyperparameters are evaluated on 9 additional split seeds (43-51) so
downstream selection can account for across-split variance.

The old factorial design included `raw_aug`, `alr_aug`, `pwlr_aug` variants.
The n_aug sensitivity appendix at the end of this notebook documents the
decision to drop them: see section "Appendix: N_AUG sensitivity".

Outputs:
- `results/nb03_multi_seed_results.csv`
- `results/nb03_winning_configurations.json`
- `results/nb03_canonical_test_predictions.npz`
- `figures/fig05_model_comparison.png` (downstream, from NB06)
"""

APPENDIX_HEADER_MD = """## Appendix: N_AUG sensitivity

Justifies the decision to set `N_AUG = 1` and drop the `_aug` feature variants.
Tests `n_aug in {1, 3, 5, 10, 15}` across all three representations, both
targets, and both tracks with default RF and XGB hyperparameters, then runs a
Wilcoxon signed-rank test (1 vs N) per representation. Empirically,
augmentation either hurt or failed to significantly help in every cell, so the
3-method design above is what the manuscript reports.

Writes:
- `results/nb03_n_aug_sensitivity.csv`
- `figures/fig_nb03_n_aug_sensitivity.png`
- `figures/fig_nb03_n_aug_overfit.png`
"""


def rewrite_src(src: str) -> str:
    """Replace destroyed `get_features_for_method` name with canonical."""
    return src.replace('get_features_for_method(', 'build_feature_matrix(')


def adapt_nb03b_sensitivity(src: str) -> str:
    """Rename nb03b-specific output paths to nb03 equivalents."""
    src = rewrite_src(src)
    src = src.replace('nb03b_n_aug_sensitivity.csv', 'nb03_n_aug_sensitivity.csv')
    src = src.replace('fig_nb03b_n_aug_sensitivity.png',
                      'fig_nb03_n_aug_sensitivity.png')
    src = src.replace('fig_nb03b_n_aug_overfit.png',
                      'fig_nb03_n_aug_overfit.png')
    return src


def main():
    nb = nbformat.read(str(NB03C), as_version=4)
    nbb = nbformat.read(str(NB03B), as_version=4)

    # 1. rewrite call sites in every cell of nb03c
    for c in nb.cells:
        if c.cell_type == 'code':
            c.source = rewrite_src(c.source)

    # 2. grab nb03b's sensitivity cells: 6 (md), 7 (code), 8 (code) and
    #    discard the "SET N_AUG" cells 9 & 10 — N_AUG is now fixed upstream
    sensitivity_md = nbformat.v4.new_markdown_cell(source=APPENDIX_HEADER_MD)
    sens_code = nbformat.v4.new_code_cell(source=adapt_nb03b_sensitivity(
        nbb.cells[7].source))
    sens_plot = nbformat.v4.new_code_cell(source=adapt_nb03b_sensitivity(
        nbb.cells[8].source))

    # 3. insert appendix BEFORE the Phase 3R.7 verification markdown (penultimate md).
    #    The Phase 3R.7 block in nb03c is the final (markdown, code) pair.
    #    We insert the appendix just before that pair.
    insert_at = len(nb.cells) - 2
    assert nb.cells[insert_at].cell_type == 'markdown'
    assert 'Phase 3R.7' in nb.cells[insert_at].source

    new_cells = (list(nb.cells[:insert_at])
                 + [sensitivity_md, sens_code, sens_plot]
                 + list(nb.cells[insert_at:]))

    # 4. update the title markdown (cell 0)
    new_cells[0] = nbformat.v4.new_markdown_cell(source=INTRO_MD)

    nb.cells = new_cells

    # 5. write to the canonical nb03 path (overwrites the stub)
    nbformat.write(nb, str(NB03))
    print(f'Wrote {NB03.name} ({len(nb.cells)} cells)')

    # 6. delete stub nb03b and nb03c
    for p in [NB03B, NB03C]:
        if p.exists():
            p.unlink()
            print(f'Deleted {p.name}')


if __name__ == '__main__':
    main()
