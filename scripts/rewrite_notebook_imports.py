"""Rewrite notebook cells to import canonical functions from `src/`.

For each target notebook:
  1. Replace any cell that defines make_raw_features / make_alr_features /
     make_pwlr_features / build_feature_matrix / predict_median /
     predict_iqr with a single import cell that pulls those names from
     `src.features` and `src.models`.
  2. Rewrite `from figure_style import ...` lines to `from src.plot_style
     import ...`.
  3. Insert `sys.path.insert(0, str(Path.cwd().parent))` if not already
     present so `src.` is importable from notebooks/ subfolder.
  4. Drop the obsolete `sys.path.insert(0, str(Path.cwd()))` line that
     was used to reach `notebooks/figure_style.py`.

Safe to re-run: operates on JSON in place but only touches cells that
match known patterns.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / 'notebooks'

TARGET_NBS = [
    'nb01_data_cleaning',
    'nb02_eda_pca',
    'nb03_baseline_models',
    'nb03b_baseline_models',
    'nb03c_baseline_models',
    'nb04_putirka_benchmark',
    'nb04b_lepr_arcpl_validation',
    'nb05_loso_validation',
    'nb06_shap_analysis',
    'nb07_bias_correction',
    'nb08_natural_samples',
    'nb09_manuscript_compilation',
    'nb10_extended_analyses',
    'nbF_figures',
]

DUP_DEFS = (
    'def make_raw_features',
    'def make_alr_features',
    'def make_pwlr_features',
    'def build_feature_matrix',
    'def predict_median',
    'def predict_iqr',
)

IMPORT_REPLACEMENT = (
    "# Canonical features and prediction helpers from src/ (one source of truth).\n"
    "from src.features import (\n"
    "    build_feature_matrix,\n"
    "    make_raw_features,\n"
    "    make_alr_features,\n"
    "    make_pwlr_features,\n"
    "    augment_dataframe,\n"
    ")\n"
    "from src.models import predict_median, predict_iqr\n"
)


def cell_defines_dups(src: str) -> bool:
    return any(marker in src for marker in DUP_DEFS)


def rewrite_figure_style_import(src: str) -> str:
    # Collapse `from figure_style import (...)` or single-line variants
    # into `from src.plot_style import (...)`.
    pattern = re.compile(r'from\s+figure_style\s+import\s+', re.M)
    return pattern.sub('from src.plot_style import ', src)


def ensure_parent_syspath(src: str) -> str:
    # Already present in every NB header via Path.cwd().parent.
    # Remove the stray `sys.path.insert(0, str(Path.cwd()))` line that
    # previously targeted notebooks/figure_style.py.
    return re.sub(
        r'\nsys\.path\.insert\(0,\s*str\(Path\.cwd\(\)\)\)\s*',
        '\n',
        src,
    )


def rewrite_notebook(path: Path) -> dict:
    nb = nbformat.read(str(path), as_version=4)
    stats = {'replaced_dup_cells': 0, 'figure_style_imports_rewritten': 0,
             'stray_syspath_removed': 0}

    new_cells = []
    injected_import = False
    for cell in nb.cells:
        if cell.cell_type != 'code':
            new_cells.append(cell)
            continue

        src = cell.source
        if cell_defines_dups(src):
            # Replace this cell with the import cell exactly once. If a
            # notebook has multiple dup cells, subsequent ones are dropped.
            if not injected_import:
                new_cell = nbformat.v4.new_code_cell(source=IMPORT_REPLACEMENT)
                new_cells.append(new_cell)
                injected_import = True
            stats['replaced_dup_cells'] += 1
            continue

        new_src = src
        if 'from figure_style' in new_src:
            new_src = rewrite_figure_style_import(new_src)
            stats['figure_style_imports_rewritten'] += 1
        if 'sys.path.insert(0, str(Path.cwd()))' in new_src:
            new_src = ensure_parent_syspath(new_src)
            stats['stray_syspath_removed'] += 1
        if new_src != src:
            cell.source = new_src
        new_cells.append(cell)

    nb.cells = new_cells
    nbformat.write(nb, str(path))
    return stats


def main():
    for name in TARGET_NBS:
        path = NOTEBOOKS / f'{name}.ipynb'
        if not path.exists():
            print(f'SKIP {name} (not found)')
            continue
        stats = rewrite_notebook(path)
        print(f'{name}: {stats}')


if __name__ == '__main__':
    main()
