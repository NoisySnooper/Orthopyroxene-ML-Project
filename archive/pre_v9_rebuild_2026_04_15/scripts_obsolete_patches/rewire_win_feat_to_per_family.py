"""v7 Part H: replace the legacy `WIN_FEAT = config_3r['global_feature_set']`
pattern with a per-family canonical spec lookup.

The pre-v7 design assumed a single "winning" feature set for the whole
project. v7 Part B replaces that with per-(target, track, family) winners.
This script rewires every live notebook call site that still builds
`WIN_FEAT` from the legacy JSON so they read from
`nb03_per_family_winners.json` instead.

Policy: use the forest-family / T_C / opx_liq spec as the "primary" that
keeps old single-feature-set behavior working. Notebooks that produce
per-family outputs (nb04, nb04b, nb08) already resolve per-target spec
directly; this script only touches notebooks that haven't been migrated
to the two-family structure yet (nb06, nb07, nb09, nbF), where a single
primary is still acceptable for now.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / "notebooks"

TARGETS = [
    ("nb06_shap_analysis", 4),
    ("nb07_bias_correction", 4),
    ("nb09_manuscript_compilation", 1),
    ("nbF_figures", 2),
]

NEW_BLOCK = (
    "# v7 Part H: per-family canonical spec replaces the legacy\n"
    "# legacy pre-v7 single-winner assumption. Forest / T_C / opx_liq is the\n"
    "# primary single-feature-set stand-in for notebooks that have not\n"
    "# been restructured into per-family output blocks.\n"
    "from src.data import canonical_model_spec\n"
    "_spec_primary = canonical_model_spec('T_C', 'opx_liq', 'forest', RESULTS)\n"
    "WIN_FEAT = _spec_primary['feature_set']\n"
    "print(f'v7 primary (forest / T_C / opx_liq) feature set: {WIN_FEAT}')\n"
)

LEGACY_PAT = re.compile(
    r"config_3r\s*=\s*load_winning_config\(RESULTS\)\s*\n"
    r"\s*WIN_FEAT\s*=\s*config_3r\[['\"]global_feature_set['\"]\]\s*\n"
    r"(?:\s*print\([^)]*\)\s*\n)?",
    re.MULTILINE,
)


def patch_notebook(stem: str, cell_idx: int) -> dict:
    path = NBDIR / f"{stem}.ipynb"
    if not path.exists():
        return {"skipped": True}
    nb = nbformat.read(str(path), as_version=4)
    cell = nb.cells[cell_idx]
    new_src, n = LEGACY_PAT.subn(NEW_BLOCK, cell.source, count=1)
    if n:
        cell.source = new_src
        nbformat.write(nb, str(path))
    return {"replacements": n}


def main() -> int:
    for stem, cidx in TARGETS:
        result = patch_notebook(stem, cidx)
        print(f"  {stem} cell {cidx}: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
