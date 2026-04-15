"""v7 hotfix: rewire nb03 Phase 3.8 ceiling cell to per-family canonical.

Cell 27 still dereferenced the legacy `config['global_feature_set']` dict to
choose its ceiling-analysis feature set. Under v7 the single-winner dict is
gone; the ceiling check picks the forest/T_C/opx_liq feature set since the
RF params used below are the forest-track T_C tuned params at seed 42.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb03_baseline_models.ipynb"

OLD = "WIN_FEAT_CEIL = config['global_feature_set']\nprint(f'Ceiling analysis feature set: {WIN_FEAT_CEIL}')"
NEW = (
    "# v7 hotfix: per-family canonical resolves the ceiling feature set.\n"
    "from src.data import canonical_model_spec\n"
    "_spec_ceil = canonical_model_spec('T_C', 'opx_liq', 'forest', RESULTS)\n"
    "WIN_FEAT_CEIL = _spec_ceil['feature_set']\n"
    "print(f'Ceiling analysis feature set (forest/T_C/opx_liq): {WIN_FEAT_CEIL}')"
)


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    cell = nb.cells[27]
    if "canonical_model_spec" in cell.source and "WIN_FEAT_CEIL" in cell.source:
        print("nb03 cell 27: already patched.")
        return 0
    if OLD not in cell.source:
        print("nb03 cell 27: anchor not found")
        return 1
    cell.source = cell.source.replace(OLD, NEW, 1)
    nbformat.write(nb, str(NB))
    print("nb03 cell 27: rewired to forest/T_C/opx_liq canonical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
