"""v7 hotfix: fix nb04 cell 18 plot_style import path.

Cell 18 did `sys.path.insert(0, str(Path.cwd().parent / 'src'))` which
assumes the notebook is run from inside `notebooks/`. Under papermill
the CWD is the project root, so `Path.cwd().parent / 'src'` resolves
to `<parent-of-project>/src`, which doesn't exist, and the try/except
silently falls back to unstyled matplotlib.

Cell 26 already uses `from src.plot_style import ...` which works
correctly. Apply the same pattern to cell 18.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"

OLD = (
    "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n"
    "try:\n"
    "    from plot_style import apply_style, one_to_one, stats_box, panel_label, \\\n"
    "        save_figure, PUTIRKA_COLOR, ML_COLOR\n"
    "    apply_style()\n"
    "    HAS_STYLE = True\n"
    "except Exception as e:\n"
    "    print(f'(plot_style unavailable: {e}; falling back to default matplotlib)')\n"
    "    PUTIRKA_COLOR = '#E69F00'\n"
    "    ML_COLOR      = '#0072B2'\n"
    "    HAS_STYLE = False"
)

NEW = (
    "from src.plot_style import (apply_style, one_to_one, stats_box,\n"
    "                            panel_label, save_figure, PUTIRKA_COLOR,\n"
    "                            ML_COLOR)\n"
    "apply_style()\n"
    "HAS_STYLE = True"
)


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    cell = nb.cells[18]
    if "from src.plot_style import" in cell.source and "sys.path.insert" not in cell.source:
        print("nb04 cell 18: already patched.")
        return 0
    if OLD not in cell.source:
        print("nb04 cell 18: anchor not found")
        return 1
    cell.source = cell.source.replace(OLD, NEW, 1)
    nbformat.write(nb, str(NB))
    print("nb04 cell 18: plot_style import rewired to src.plot_style.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
