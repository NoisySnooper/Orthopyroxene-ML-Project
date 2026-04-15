"""v7 hotfix: cap the N_AUG sensitivity test at 5 seeds.

The Phase 3.3b sensitivity loop timed out at 3h when N_SPLIT_REPS rose from
10 to 20. The main Phase 3.4 training still uses all 20 seeds; only the
appendix sensitivity sweep is subset here. 5 seeds keeps Wilcoxon valid
and all 5 N_AUG points intact, dropping the inner loop from ~2400 to
~600 iterations (~45 min).
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb03_baseline_models.ipynb"

MARKER = "# v7 hotfix: sensitivity test subsets SPLIT_SEEDS"

OLD_HEADER = (
    "n_aug_results = []\n"
    "total_iters = (len(AUG_TEST_MODELS) * len(N_AUG_TEST) * len(AUG_TEST_REPRS)\n"
    "               * len(SPLIT_SEEDS) * len(AUG_TEST_TARGETS) * 2)\n"
    "pbar = tqdm(total=total_iters, desc='N_AUG sensitivity')\n"
)
NEW_HEADER = (
    "# v7 hotfix: sensitivity test subsets SPLIT_SEEDS to first 5 seeds.\n"
    "# Main Phase 3.4 training below still uses all 20 seeds.\n"
    "_SENS_SEEDS = SPLIT_SEEDS[:5]\n"
    "n_aug_results = []\n"
    "total_iters = (len(AUG_TEST_MODELS) * len(N_AUG_TEST) * len(AUG_TEST_REPRS)\n"
    "               * len(_SENS_SEEDS) * len(AUG_TEST_TARGETS) * 2)\n"
    "pbar = tqdm(total=total_iters, desc='N_AUG sensitivity')\n"
)

OLD_LOOP = "    for seed in SPLIT_SEEDS:\n"
NEW_LOOP = "    for seed in _SENS_SEEDS:\n"


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    cell = nb.cells[8]
    if MARKER in cell.source:
        print("nb03 cell 8: already patched.")
        return 0
    if OLD_HEADER not in cell.source:
        print("nb03 cell 8: header anchor not found")
        return 1
    if OLD_LOOP not in cell.source:
        print("nb03 cell 8: loop anchor not found")
        return 1
    cell.source = cell.source.replace(OLD_HEADER, NEW_HEADER, 1)
    cell.source = cell.source.replace(OLD_LOOP, NEW_LOOP, 1)
    nbformat.write(nb, str(NB))
    print("nb03 cell 8: subset to 5 seeds (sensitivity only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
