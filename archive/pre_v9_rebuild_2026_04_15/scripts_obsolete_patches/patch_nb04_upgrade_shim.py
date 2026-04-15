"""Upgrade nb04 cells 18/24/26 from lepr_to_training_schema to
lepr_to_training_features (adds cation recalc + engineered features).

The previous shim only renamed columns, but models were trained with
engineered columns (Mg_num, Al_IV/VI, En/Fs/Wo, MgTs, liq_Mg_num) which
the renamed LEPR rows did not provide. `lepr_to_training_features` is the
full pipeline and produces DataFrames with matching feature counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"
EXEC_NB = ROOT / "notebooks" / "executed" / "nb04_putirka_benchmark_executed.ipynb"

REPLACEMENTS = [
    ('lepr_to_training_schema', 'lepr_to_training_features'),
]


def _apply(nb_path):
    if not nb_path.exists():
        return None
    nb = nbformat.read(str(nb_path), as_version=4)
    changes = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        new_src = cell.source
        for old, new in REPLACEMENTS:
            if old in new_src:
                new_src = new_src.replace(old, new)
        if new_src != cell.source:
            cell.source = new_src
            cell.outputs = []
            cell.execution_count = None
            changes += 1
    if changes:
        nbformat.validate(nb)
        nbformat.write(nb, str(nb_path))
    print(f'{nb_path.name}: updated {changes} cell(s).')
    return changes


def main() -> int:
    _apply(NB)
    _apply(EXEC_NB)
    return 0


if __name__ == '__main__':
    sys.exit(main())
