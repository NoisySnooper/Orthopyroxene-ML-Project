"""v8-fix: add canonical-stem savefig lines to nb06 so SHAP beeswarms
match the v7/v8 figure roster (fig_nb06_shap_{P,T}_beeswarm.png).
Keeps the legacy fig07/fig08 savefigs for backwards compatibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb06_shap_analysis.ipynb"

OLD_P = "plt.savefig(FIGURES / 'fig07_shap_P_beeswarm.png', dpi=300, bbox_inches='tight')"
NEW_P = (
    "plt.savefig(FIGURES / 'fig07_shap_P_beeswarm.png', dpi=300, bbox_inches='tight')\n"
    "plt.savefig(FIGURES / 'fig_nb06_shap_P_beeswarm.png', dpi=300, bbox_inches='tight')  # v8-fix: canonical stem"
)

OLD_T = "plt.savefig(FIGURES / 'fig08_shap_T_beeswarm.png', dpi=300, bbox_inches='tight')"
NEW_T = (
    "plt.savefig(FIGURES / 'fig08_shap_T_beeswarm.png', dpi=300, bbox_inches='tight')\n"
    "plt.savefig(FIGURES / 'fig_nb06_shap_T_beeswarm.png', dpi=300, bbox_inches='tight')  # v8-fix: canonical stem"
)


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    n = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if OLD_P in src and NEW_P not in src:
            cell.source = src.replace(OLD_P, NEW_P, 1)
            cell.outputs = []
            cell.execution_count = None
            n += 1
            src = cell.source
        if OLD_T in src and NEW_T not in src:
            cell.source = src.replace(OLD_T, NEW_T, 1)
            cell.outputs = []
            cell.execution_count = None
            n += 1
    print(f"patched {n} savefig line(s)")
    nbformat.validate(nb)
    nbformat.write(nb, str(NB))
    return 0


if __name__ == "__main__":
    sys.exit(main())
